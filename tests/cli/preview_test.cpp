#include "src/cli/serve/preview.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"

namespace {

using gufo::models::LatentPreview;
using gufo::server::LatentFormat;
using gufo::server::PreviewHub;
using gufo::server::PreviewOptions;
using gufo::server::PreviewRgb;
namespace json = gufo::json;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL preview_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition)
    Fail(message);
}

// latent2rgb is linear: rgb(a + b) - bias = (rgb(a) - bias) + (rgb(b) - bias)
// on unclamped values, and both layouts must agree pixel for pixel.
void TestLatentToRgb() {
  const int frames = 3, height = 2, width = 4, channels = 48;
  std::vector<float> channel_major(frames * height * width * channels);
  std::vector<float> token_major(channel_major.size());
  for (int c = 0; c < channels; ++c)
    for (int f = 0; f < frames; ++f)
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
          const float v = 0.05F * static_cast<float>(((c * 7 + f * 3 + y * 5 +
                                                       x * 11) % 13) - 6);
          channel_major[((c * frames + f) * height + y) * width + x] = v;
          token_major[((f * height + y) * width + x) * channels + c] = v;
        }
  LatentPreview video{.step = 1, .steps = 2, .frames = frames,
                      .height = height, .width = width, .channels = channels,
                      .channel_major = true, .data = channel_major};
  LatentPreview tokens = video;
  tokens.channel_major = false;
  tokens.data = token_major;
  const auto a = gufo::server::LatentToRgb(LatentFormat::kWan22, video);
  const auto b = gufo::server::LatentToRgb(LatentFormat::kWan22, tokens);
  Check(a.size() == 3 && b.size() == 3, "one image per latent frame");
  for (int f = 0; f < frames; ++f) {
    Check(a[f].width == width && a[f].height == height, "latent resolution");
    for (std::size_t i = 0; i < a[f].rgb.size(); ++i)
      Check(std::abs(a[f].rgb[i] - b[f].rgb[i]) <= 1,
            "channel-major and token-major layouts agree");
  }
  // A zero latent gives the bias colour everywhere.
  std::vector<float> zero(64 * 4, 0.0F);
  const auto flat = gufo::server::LatentToRgb(
      LatentFormat::kQwenImage21,
      {.frames = 1, .height = 2, .width = 2, .channels = 64, .data = zero});
  for (int p = 1; p < 4; ++p)
    for (int k = 0; k < 3; ++k)
      Check(flat[0].rgb[p * 3 + k] == flat[0].rgb[k], "constant latent");
  // Wrong channel counts are refused.
  bool thrown = false;
  try {
    (void)gufo::server::LatentToRgb(
        LatentFormat::kMinimaxH3,
        {.frames = 1, .height = 2, .width = 2, .channels = 64, .data = zero});
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  Check(thrown, "shape mismatch throws");
}

void TestEncoders() {
  PreviewRgb image{
      .width = 8, .height = 4, .rgb = std::vector<std::uint8_t>(8 * 4 * 3)};
  for (std::size_t i = 0; i < image.rgb.size(); ++i)
    image.rgb[i] = static_cast<std::uint8_t>(i * 7);
  const auto fitted = gufo::server::FitPreview(image, 64);
  Check(fitted.width == 64 && fitted.height == 32, "fit keeps the aspect");
  const auto jpeg = gufo::server::EncodePreviewJpeg(fitted);
  Check(jpeg.size() > 100 && static_cast<unsigned char>(jpeg[0]) == 0xFF &&
            static_cast<unsigned char>(jpeg[1]) == 0xD8,
        "JPEG signature");
  const auto decoded = gufo::core::DecodeImage(std::span(
      reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()));
  Check(decoded.width == 64 && decoded.height == 32, "JPEG decodes");
  const std::vector<PreviewRgb> frames{fitted, fitted, fitted};
  const auto webp = gufo::server::EncodePreviewWebp(frames, 100);
  Check(webp.starts_with("RIFF") && webp.substr(8, 4) == "WEBP" &&
            webp.find("ANIM") != std::string::npos &&
            webp.find("ANMF") != std::string::npos,
        "animated WebP container");
}

void TestParse() {
  PreviewOptions options;
  const auto id = gufo::server::ParsePreviewRequest(
      json::parse(R"({"id":"abc-1","always":true,"every":3,"max_side":256})"),
      &options);
  Check(id == "abc-1" && options.always && options.every == 3 &&
            options.max_side == 256,
        "preview options parse");
  for (const char* bad :
       {R"({"id":"a b"})", R"({"every":0})", R"({"max_side":5000})",
        R"({"method":"taesd"})", R"({"unknown":1})", R"([1])"}) {
    bool thrown = false;
    try {
      (void)gufo::server::ParsePreviewRequest(json::parse(bad), &options);
    } catch (const std::invalid_argument&) {
      thrown = true;
    }
    Check(thrown, std::string("rejects ") + bad);
  }
}

gufo::server::HttpResponse Get(PreviewHub& hub, const std::string& path,
                               const std::string& query = "") {
  gufo::server::HttpRequest request;
  request.method = "GET";
  request.path = path;
  request.query = query;
  return gufo::server::HandlePreviewRequest(request, hub);
}

std::string Header(const gufo::server::HttpResponse& response,
                   const std::string& name) {
  for (const auto& [key, value] : response.headers)
    if (key == name)
      return value;
  return {};
}

// No watcher, no work: the hook declines every step until someone polls.
void TestWatchers() {
  PreviewHub hub;
  auto hook = hub.Begin("job1", LatentFormat::kQwenImage21, {.every = 1});
  Check(!hook.Wants(1, 10), "unwatched job renders nothing");
  Check(Get(hub, "/v1/previews/job1").status == 204, "204 before a preview");
  Check(hook.Wants(1, 10), "a poll switches previews on");
  Check(!hook.Wants(10, 10), "the last step is never previewed");
  std::vector<float> latent(64 * 4 * 4, 0.1F);
  hook.publish({.step = 1, .steps = 10, .height = 4, .width = 4,
                .channels = 64, .data = latent});
  for (int i = 0; i < 200 && hub.stats().encoded == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  Check(hub.stats().encoded == 1, "encoder thread ran");
  const auto image = Get(hub, "/v1/previews/job1");
  Check(image.status == 200 && Header(image, "Content-Type") == "image/jpeg",
        "JPEG preview");
  Check(Header(image, "X-Gufo-Preview-Step") == "1" &&
            Header(image, "X-Gufo-Preview-Steps") == "10" &&
            Header(image, "X-Gufo-Preview-Seq") == "1" &&
            Header(image, "X-Gufo-Preview-State") == "running",
        "preview headers");
  const auto meta =
      json::parse(Get(hub, "/v1/previews/job1", "format=json").body);
  Check(meta.member_str("mime") == "image/jpeg" &&
            !meta.member_str("data").empty(),
        "JSON variant");
  Check(Get(hub, "/v1/previews/bad id").status == 404, "invalid id");
  hub.End("job1", true);
  Check(!hook.Wants(2, 10), "ended job renders nothing");
  Check(Header(Get(hub, "/v1/previews/job1"), "X-Gufo-Preview-State") == "done",
        "done state");

  // Explicit previews need no watcher; every-N applies.
  auto always = hub.Begin("job2", LatentFormat::kMinimaxH3,
                          {.always = true, .every = 2, .max_side = 64,
                           .pixel_frames = 22});
  Check(!always.Wants(1, 10) && always.Wants(2, 10), "every second step");
  std::vector<float> video(24 * 2 * 3 * 5, 0.2F);
  always.publish({.step = 2, .steps = 10, .frames = 2, .height = 3,
                  .width = 5, .channels = 24, .channel_major = true,
                  .data = video});
  for (int i = 0; i < 200 && hub.stats().encoded < 2; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const auto loop = Get(hub, "/v1/previews/job2");
  Check(Header(loop, "Content-Type") == "image/webp" &&
            Header(loop, "X-Gufo-Preview-Frames") == "0-21",
        "video loop");
  Check(Header(Get(hub, "/v1/previews/job2", "format=frame"), "Content-Type") ==
            "image/jpeg",
        "single frame of a video");
  Check(Get(hub, "/v1/previews/job2", "format=gif").status == 400,
        "unknown format");
}

}  // namespace

int main() {
  TestLatentToRgb();
  TestEncoders();
  TestParse();
  TestWatchers();
  std::cout << "preview_test: ok\n";
  return 0;
}
