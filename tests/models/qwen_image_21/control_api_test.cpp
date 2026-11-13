// HTTP contract of the Qwen-Image Fun ControlNet fields (gufo.control_*) on
// /v1/images/generations and /v1/images/edits, with an injected runner (no
// model, no GPU). 2.0: the host passes the extension through
// (changes/0.2/image-request-extension); the runner applies the provider's
// BuildRequest as the qwen_image_21 runner does.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/models/qwen_image_21/model.hpp"
#include "src/models/qwen_image_21/request.hpp"

namespace {

using namespace gufo::models::qwen_image_21;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

Image Solid(int w, int h, std::uint8_t value) {
  Image image{w, h, {}};
  for (int i = 0; i < w * h; ++i)
    image.rgba.insert(image.rgba.end(), {value, value, value, 255});
  return image;
}

std::string Base64(const std::vector<std::uint8_t>& bytes) {
  constexpr char kDigits[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::uint32_t v =
        bytes[i] << 16 | (i + 1 < bytes.size() ? bytes[i + 1] << 8 : 0) |
        (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    const std::size_t digits = std::min<std::size_t>(bytes.size() - i, 3) + 1;
    for (std::size_t k = 0; k < 4; ++k)
      out += k < digits ? kDigits[(v >> (18 - 6 * k)) & 63] : '=';
  }
  return out;
}

}  // namespace

int main() {
  const auto pose = Solid(96, 64, 0);  // 3:2 control image
  const auto pose_png = EncodePng(pose);
  const auto pose_b64 = Base64(pose_png);
  const auto start_png = EncodePng(Solid(64, 64, 200));
  Request seen;
  int calls = 0;
  const auto runner_for = [&](bool controlnet) {
    ProviderState state;
    state.controlnet = controlnet;
    return [&, state](const gufo::ext::ImageRequest& request,
                      const gufo::ext::CancellationCheck&) {
      seen = BuildRequest(request, state);
      ++calls;
      const auto pixels = Solid(2, 2, 7);
      return gufo::ext::ImageResult{
          .image = {.width = pixels.width,
                    .height = pixels.height,
                    .rgba = pixels.rgba}};
    };
  };
  // 2.0: the service takes an ext::ImageRunner; the callback is wrapped with
  // the provider's capabilities (edits for /v1/images/edits).
  const auto runner = [&](bool controlnet) {
    return std::make_shared<gufo::server::CallbackImageRunner>(
        runner_for(controlnet), kCapabilities);
  };
  gufo::server::ImageService service(runner(true), "image-test");
  gufo::server::ImageService plain(runner(false), "image-test");

  gufo::server::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  const auto post = [&](gufo::server::ImageService& target,
                        const std::string& body) {
    http.body = body;
    return gufo::server::HandleImageApiRequest(http, target).status;
  };
  // JSON: defaults, size auto from the control image (3:2 at ~1 MP).
  Check(post(service, R"({"prompt":"a dancer","seed":3,"gufo":{"control_image":")" +
                          pose_b64 + R"("}})") == 200,
        "json control 200");
  Check(seen.control && seen.control->image.rgba == pose.rgba &&
            seen.control->strength == 1 && seen.control->start == 0 &&
            seen.control->end == 1 && !seen.control->inpaint &&
            seen.images.empty() && !seen.init_image,
        "json control defaults");
  Check(seen.width == 1248 && seen.height == 832, "size auto from control");
  // Data URL and all options.
  Check(post(service, R"({"prompt":"a dancer","size":"512x768","gufo":{"control_image":"data:image/png;base64,)" +
                          pose_b64 +
                          R"(","control_strength":0.6,"control_start":0.1,"control_end":0.8}})") ==
            200,
        "json control options 200");
  Check(seen.control && seen.control->strength == 0.6F &&
            seen.control->start == 0.1 && seen.control->end == 0.8 &&
            seen.width == 512 && seen.height == 768,
        "json control options");
  // Without control nothing changes.
  Check(post(service, R"({"prompt":"a","size":"256x256"})") == 200 &&
            !seen.control,
        "no control");
  const int before = calls;
  for (const std::string body :
       {R"({"prompt":"a","gufo":{"control_image":")" + pose_b64 +
            R"(","control_strength":2.5}})",
        R"({"prompt":"a","gufo":{"control_image":")" + pose_b64 +
            R"(","control_strength":"1"}})",
        R"({"prompt":"a","gufo":{"control_image":")" + pose_b64 +
            R"(","control_start":0.7,"control_end":0.3}})",
        R"({"prompt":"a","gufo":{"control_image":")" + pose_b64 +
            R"(","control_inpaint":true}})",
        R"({"prompt":"a","gufo":{"control_image":")" + pose_b64 +
            R"(","control_inpaint":1}})",
        std::string(R"({"prompt":"a","gufo":{"control_strength":0.5}})"),
        // Inpaint-only control without start image and mask.
        std::string(R"({"prompt":"a","gufo":{"control_inpaint":true}})"),
        std::string(
            R"({"prompt":"a","gufo":{"control_image":"https://example.com/p.png"}})"),
        std::string(R"({"prompt":"a","gufo":{"control_image":7}})"),
        std::string(R"({"prompt":"a","gufo":{"control_image":"!!!"}})")})
    Check(post(service, body) == 400, "rejected: " + body.substr(0, 80));
  Check(post(plain, R"({"prompt":"a","gufo":{"control_image":")" + pose_b64 +
                        R"("}})") == 400,
        "server without --controlnet rejects");
  Check(calls == before, "rejected requests never reach the model");

  // Multipart edit: start image + mask + control_image file part, inpaint.
  // ("init_image": an OpenAI-style "image" would also stay a reference.)
  http.path = "/v1/images/edits";
  http.headers = {{"Content-Type", "multipart/form-data; boundary=b"}};
  const auto multipart =
      [&](const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>&
              files,
          const std::vector<std::pair<std::string, std::string>>& fields) {
        std::string body;
        for (const auto& [name, bytes] : files) {
          body += "--b\r\nContent-Disposition: form-data; name=\"" + name +
                  "\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n";
          body.append(reinterpret_cast<const char*>(bytes.data()),
                      bytes.size());
          body += "\r\n";
        }
        for (const auto& [name, value] : fields)
          body += "--b\r\nContent-Disposition: form-data; name=\"" + name +
                  "\"\r\n\r\n" + value + "\r\n";
        return body + "--b--\r\n";
      };
  const auto mask_png = EncodePng(Solid(64, 64, 255));
  Check(post(service, multipart({{"init_image", start_png},
                                 {"mask", mask_png},
                                 {"control_image", pose_png}},
                                {{"prompt", "a dancer"},
                                 {"gufo.control_strength", "0.75"},
                                 {"gufo.control_inpaint", "true"}})) == 200,
        "multipart control 200");
  Check(seen.control && seen.control->inpaint &&
            seen.control->strength == 0.75F &&
            seen.control->image.rgba == pose.rgba && seen.mask &&
            seen.init_image && seen.width == 64 && seen.height == 64,
        "multipart control fields");
  // The control image is not a reference.
  Check(seen.images.empty(), "control image is no reference");
  // Inpaint-only (ComfyUI ZImageFunControlnet without "image"): start image +
  // mask + gufo.control_inpaint, no control_image part. The control image
  // stays empty, so the control channels are zero.
  Check(post(service, multipart({{"init_image", start_png}, {"mask", mask_png}},
                                {{"prompt", "a dancer"},
                                 {"gufo.control_strength", "0.9"},
                                 {"gufo.control_inpaint", "true"}})) == 200,
        "inpaint-only control 200");
  Check(seen.control && seen.control->inpaint &&
            seen.control->strength == 0.9F && seen.control->image.width == 0 &&
            seen.control->image.rgba.empty() && seen.mask && seen.init_image,
        "inpaint-only control fields");
  Check(post(plain, multipart({{"init_image", start_png}, {"mask", mask_png}},
                              {{"prompt", "a"},
                               {"gufo.control_inpaint", "true"}})) == 400,
        "inpaint-only control needs --controlnet");
  // Reference images combine with the control since 1.4.7 (zero rows in
  // the control stream, as ComfyUI does).
  Check(post(service, multipart({{"init_image", start_png},
                                 {"mask", mask_png},
                                 {"image", start_png},
                                 {"control_image", pose_png}},
                                {{"prompt", "a"},
                                 {"gufo.control_inpaint", "true"}})) == 200 &&
            seen.control && seen.images.size() == 1,
        "control with a reference image");
  Check(post(service, multipart({{"image", start_png},
                                 {"control_image", pose_png},
                                 {"control_image", pose_png}},
                                {{"prompt", "a"}})) == 400,
        "duplicate control_image part");
  // gufo.control_image as a form field (base64) works too.
  Check(post(service, multipart({{"image", start_png}},
                                {{"prompt", "a"},
                                 {"gufo.strength", "0.5"},
                                 {"gufo.control_image", pose_b64}})) == 200 &&
            seen.control && seen.strength == 0.5 && seen.images.empty(),
        "form field control_image with image to image");
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "qwen_image_21_control_api_test: all passed\n";
  return 0;
}
