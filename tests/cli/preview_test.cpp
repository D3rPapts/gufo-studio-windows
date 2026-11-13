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

// Wan 2.1 (Uni3C camera moves): 16 channels. Normalized latents of solid
// colours from the Wan 2.1 VAE encoder ((z - mean) / std, centre of a 64x64
// image) must preview as those colours; a 48-channel Wan 2.2 latent is not
// a Wan 2.1 latent.
void TestWan21Colours() {
  struct Colour {
    const char* name;
    float latent[16];
    int rgb[3];
  };
  const Colour colours[] = {
      {"red",
       {-0.3960F, 1.3660F, 1.3362F, -0.0737F, 1.0735F, 0.9857F, 0.0802F,
        -2.0122F, -0.4179F, -2.0639F, -1.0409F, -0.0401F, -0.7723F, 1.2452F,
        0.5258F, 0.1150F},
       {255, 0, 0}},
      {"green",
       {-1.3833F, -1.7724F, 0.8222F, 1.2714F, 0.5253F, -0.9837F, -0.2091F,
        -1.3363F, 0.4061F, 2.0203F, 0.6505F, 0.7573F, 0.9228F, -2.0237F,
        -1.8339F, 0.8341F},
       {0, 255, 0}},
      {"blue",
       {2.6825F, 1.9572F, -1.8700F, -1.4108F, -0.7144F, -1.0961F, -0.0949F,
        1.0855F, -0.3328F, 0.4840F, -0.1448F, 0.2171F, 0.4337F, 0.9248F,
        1.3279F, -0.5637F},
       {0, 0, 255}},
      {"white",
       {0.8798F, 0.4816F, 1.1298F, 1.2294F, -0.8779F, -0.1479F, -0.8306F,
        -0.6688F, -1.3269F, -0.3029F, 0.7377F, 0.8602F, -0.1037F, -0.9381F,
        1.3315F, 1.2491F},
       {255, 255, 255}},
      {"black",
       {-0.0585F, 0.5264F, -0.9525F, -1.4949F, 0.7660F, -0.4241F, 0.5503F,
        -0.3844F, 0.7959F, 0.2112F, -0.8589F, -0.3128F, 0.1313F, 0.9528F,
        -0.9413F, -1.0189F},
       {0, 0, 0}},
  };
  // 2 latent frames of 2x3 pixels, channel-major like the sampler's x0.
  const int frames = 2, height = 2, width = 3, plane = height * width;
  for (const Colour& colour : colours) {
    std::vector<float> latent(16 * frames * plane);
    for (int c = 0; c < 16; ++c)
      for (int i = 0; i < frames * plane; ++i)
        latent[c * frames * plane + i] = colour.latent[c];
    const auto images = gufo::server::LatentToRgb(
        LatentFormat::kWan21,
        {.frames = frames, .height = height, .width = width, .channels = 16,
         .channel_major = true, .data = latent});
    Check(images.size() == 2 && images[1].width == width &&
              images[1].height == height,
          "Wan 2.1: one latent-resolution image per latent frame");
    for (const auto& image : images)
      for (int p = 0; p < plane; ++p)
        for (int k = 0; k < 3; ++k)
          Check(std::abs(image.rgb[p * 3 + k] - colour.rgb[k]) <= 48,
                std::string("Wan 2.1 previews ") + colour.name);
  }
  std::vector<float> wan22(48 * 4, 0.0F);
  bool thrown = false;
  try {
    (void)gufo::server::LatentToRgb(
        LatentFormat::kWan21,
        {.frames = 1, .height = 2, .width = 2, .channels = 48, .data = wan22});
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  Check(thrown, "Wan 2.1 refuses 48 channels");
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

  // Short schedules (H3 Turbo, 8 steps) preview every step but the last;
  // long ones keep every N-th.
  auto dense = hub.Begin("job3", LatentFormat::kMinimaxH3,
                         {.always = true, .every = 2, .dense_up_to = 16});
  bool all_short = true;
  for (int step = 1; step < 8; ++step)
    all_short = all_short && dense.Wants(step, 8);
  Check(all_short && !dense.Wants(8, 8), "dense cadence: steps 1..7 of 8");
  Check(!dense.Wants(1, 49) && dense.Wants(2, 49) && !dense.Wants(49, 49),
        "long schedule keeps every second step");
  // The cadence counts from the last preview, so a model may skip steps
  // (H3 reused steps) and still preview every `every` steps after that.
  const auto encoded = hub.stats().encoded;
  std::vector<float> h3(24 * 2 * 3 * 5, 0.3F);
  dense.publish({.step = 3, .steps = 49, .frames = 2, .height = 3, .width = 5,
                 .channels = 24, .channel_major = true, .data = h3});
  for (int i = 0; i < 200 && hub.stats().encoded == encoded; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  Check(!dense.Wants(4, 49) && dense.Wants(5, 49) && dense.Wants(6, 49),
        "next preview due two steps after the last one");
}

void TestDenseParse() {
  // Video defaults keep the dense cadence; an explicit "every" wins.
  PreviewOptions options{.every = 2, .dense_up_to = 16};
  (void)gufo::server::ParsePreviewRequest(json::parse(R"({"always":true})"),
                                          &options);
  Check(options.dense_up_to == 16, "dense cadence kept without every");
  (void)gufo::server::ParsePreviewRequest(json::parse(R"({"every":3})"),
                                          &options);
  Check(options.every == 3 && options.dense_up_to == 0,
        "explicit every turns the dense cadence off");
}

}  // namespace

int main() {
  TestLatentToRgb();
  TestWan21Colours();
  TestEncoders();
  TestParse();
  TestWatchers();
  TestDenseParse();
  std::cout << "preview_test: ok\n";
  return 0;
}
