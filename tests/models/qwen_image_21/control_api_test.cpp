// HTTP contract of the Qwen-Image Fun ControlNet fields (gufo.control_*) on
// /v1/images/generations and /v1/images/edits, with an injected runner (no
// model, no GPU).

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/models/qwen_image_21/model.hpp"

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
  const auto runner = [&](const Request& request, const CancellationCheck&) {
    seen = request;
    ++calls;
    return Result{.image = Solid(2, 2, 7)};
  };
  gufo::server::ImageLoraOptions with_controlnet;
  with_controlnet.controlnet = "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors";
  gufo::server::ImageService service({}, "image-test", runner,
                                     with_controlnet);
  gufo::server::ImageService plain({}, "image-test", runner,
                                   gufo::server::ImageLoraOptions{});
  Check(service.has_controlnet() && !plain.has_controlnet(), "flag");

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
  // With the start image as a reference (OpenAI mask edit) it is refused.
  Check(post(service, multipart({{"image", start_png},
                                 {"mask", mask_png},
                                 {"control_image", pose_png}},
                                {{"prompt", "a"}})) == 400,
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
