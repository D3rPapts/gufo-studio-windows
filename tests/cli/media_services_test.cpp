// gufo-studio-windows media services without a model (src/cli/serve/media):
// /v1/images/upscale request handling and PNG output through a stand-in
// engine, music / foley start-up errors, the `gufo serve` media commands and
// the video post-step options, and the post-step request parser.
// Written with AI assistance (Claude Code).
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/media/foley_api.hpp"
#include "src/cli/serve/media/media_serve.hpp"
#include "src/cli/serve/media/music_api.hpp"
#include "src/cli/serve/media/upscale_api.hpp"
#include "src/cli/serve/media/video_postprocess.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"

namespace server = gufo::server;
namespace json = gufo::json;

namespace {

// A 1x1 RGBA PNG.
constexpr std::string_view kPng =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg==";

server::HttpResponse Post(server::MediaService& service, std::string path,
                          std::string body) {
  server::HttpRequest request{
      .method = "POST",
      .path = std::move(path),
      .query = {},
      .body = std::move(body),
      .headers = {},
      .is_cancelled = {},
  };
  return service.Handle(request);
}

// Nearest-neighbour stand-in for Real-ESRGAN: x4 or into out_w x out_h.
std::unique_ptr<server::UpscaleEngine> FakeEngine(std::vector<std::string>* calls) {
  auto engine = std::make_unique<server::UpscaleEngine>();
  engine->device = "cpu";
  engine->model_names = {"anime", "general"};
  engine->runtime = "stand-in";
  engine->run = [calls](const std::string& model, const std::uint8_t* rgb24,
                        int w, int h, int out_w, int out_h,
                        const std::string& fit, server::UpscaleOutcome* out,
                        std::string*) {
    calls->push_back(model + ":" + fit);
    out->width = out_w == 0 ? w * 4 : out_w;
    out->height = out_h == 0 ? h * 4 : out_h;
    out->rgb24.resize(static_cast<std::size_t>(out->width) * out->height * 3);
    for (int y = 0; y < out->height; ++y)
      for (int x = 0; x < out->width; ++x)
        for (int c = 0; c < 3; ++c)
          out->rgb24[(static_cast<std::size_t>(y) * out->width + x) * 3 + c] =
              rgb24[(static_cast<std::size_t>(y * h / out->height) * w +
                     x * w / out->width) *
                        3 +
                    c];
    out->model_file = model + ".onnx";
    return true;
  };
  return engine;
}

void TestUpscaleService() {
  std::vector<std::string> calls;
  server::UpscaleService service(FakeEngine(&calls));
  assert(service.ready());
  assert(service.capability() == "upscale");
  assert(service.model_id() == "anime");
  assert((service.model_ids() == std::vector<std::string>{"anime", "general"}));
  assert(service.HandlesPath("/v1/images/upscale"));
  assert(!service.HandlesPath("/v1/images/generations"));
  json::Value entry = json::Value::object();
  service.Describe(&entry, false);
  assert(entry.member_str("device") == "cpu" &&
         entry.member_str("runtime") == "stand-in");

  const std::string image =
      "data:image/png;base64," + std::string(kPng);
  // Default: x4, crop, JSON with a base64 PNG.
  auto response = Post(service, "/v1/images/upscale",
                       R"({"image":")" + image + R"("})");
  assert(response.status == 200);
  const json::Value body = json::parse(response.body);
  assert(body.member_str("model") == "anime" &&
         body.member_str("fit") == "crop");
  assert(body.member_double("width") == 4 && body.member_double("height") == 4);
  std::string png;
  assert(server::DecodeAudioBase64(
      body.find("data")->items().front().member_str("b64_json"), 1 << 20,
      &png));
  const gufo::core::Image decoded = gufo::core::DecodeImage(
      std::span(reinterpret_cast<const std::uint8_t*>(png.data()), png.size()));
  assert(decoded.width == 4 && decoded.height == 4 &&
         decoded.pixels.size() == 4 * 4 * 3);

  // Named model, explicit size and fit, raw PNG.
  response = Post(
      service, "/v1/images/upscale",
      R"({"model":"general","image":")" + image +
          R"(","size":"6x2","fit":"pad","response_format":"png"})");
  assert(response.status == 200 && response.body.starts_with("\x89PNG"));
  const gufo::core::Image raw = gufo::core::DecodeImage(std::span(
      reinterpret_cast<const std::uint8_t*>(response.body.data()),
      response.body.size()));
  assert(raw.width == 6 && raw.height == 2);
  assert((calls == std::vector<std::string>{"anime:crop", "general:pad"}));

  assert(Post(service, "/v1/images/upscale",
              R"({"model":"x","image":")" + image + R"("})")
             .status == 404);
  assert(Post(service, "/v1/images/upscale",
              R"({"image":")" + image + R"(","size":"big"})")
             .status == 400);
  assert(Post(service, "/v1/images/upscale",
              R"({"image":")" + image + R"(","width":8})")
             .status == 400);
  assert(Post(service, "/v1/images/upscale", R"({"size":"8x8"})").status ==
         400);
  assert(Post(service, "/v1/images/upscale", "not json").status == 400);
  int w = 0, h = 0;
  assert(server::ParseUpscaleSize("1920x1080", &w, &h) && w == 1920 &&
         h == 1080);
  assert(!server::ParseUpscaleSize("x1080", &w, &h));
}

void TestMusicAndFoleyStartup(const std::filesystem::path& root) {
  // A directory without the model files: not ready, with a reason, and the
  // routes they own.
  std::filesystem::create_directories(root / "empty");
  server::MusicService music(root / "empty", "ace-step-1.5-xl-turbo");
  assert(!music.ready());
  assert(music.initialization_error().find("dit.safetensors") !=
         std::string::npos);
  assert(music.capability() == "music" &&
         music.model_id() == "ace-step-1.5-xl-turbo");
  assert(music.HandlesPath("/v1/audio/music") &&
         music.HandlesPath("/v1/audio/music/music_1/content") &&
         !music.HandlesPath("/v1/audio/musical"));
  assert(music.unavailable_code() == "music_service_unavailable");

  server::FoleyService foley(root / "empty", "mmaudio-44k");
  assert(!foley.ready() && !foley.initialization_error().empty());
  assert(foley.capability() == "foley");
  assert(foley.HandlesPath("/v1/audio/foley") &&
         foley.HandlesPath("/v1/audio/foley/foley_1") &&
         !foley.HandlesPath("/v1/audio/foleys"));
}

void TestServeOptions(const std::filesystem::path& root) {
  assert(server::IsMediaServeCommand("music") &&
         server::IsMediaServeCommand("foley") &&
         server::IsMediaServeCommand("upscale") &&
         !server::IsMediaServeCommand("video"));
  assert(server::MediaServeCommandsHelp().find("/v1/images/upscale") !=
         std::string::npos);

  // gufo serve upscale: NAME=FILE, default name from --served-model-name.
  std::ofstream(root / "body.onnx") << "x";
  server::MediaServeArgs args;
  gufo::cli::ArgParser parser("gufo serve upscale");
  server::AddMediaServeOptions("upscale", parser, &args);
  const std::string body = (root / "body.onnx").string();
  const std::string named = "anime=" + body;
  const char* argv[] = {"--device", "cpu",  "-m",
                        named.c_str(), "-m", body.c_str(),
                        "--served-model-name", "section"};
  std::string error;
  assert(parser.Parse(argv, &error));
  assert(server::ValidateMediaServeArgs("upscale", &args, &error));
  assert(args.upscale.models.size() == 2 &&
         args.upscale.models[0].name == "anime" &&
         args.upscale.models[1].name == "section");
  args.upscale.device = "tpu";
  assert(!server::ValidateMediaServeArgs("upscale", &args, &error) &&
         error.find("--device") != std::string::npos);

  server::MediaServeArgs music_args;
  gufo::cli::ArgParser music_parser("gufo serve music");
  server::AddMediaServeOptions("music", music_parser, &music_args);
  assert(music_args.name == "ace-step-1.5-xl-turbo");
  assert(!server::ValidateMediaServeArgs("music", &music_args, &error) &&
         error.find("--model") != std::string::npos);

  // gufo serve video --rife-model / --upscale-model.
  std::ofstream(root / "rife47.safetensors") << "x";
  std::ofstream(root / "realesr-general-x4v3.safetensors") << "x";
  server::VideoPostProcessArgs post;
  gufo::cli::ArgParser video("gufo serve video");
  server::AddVideoPostProcessOptions(video, &post);
  const std::string rife = (root / "rife47.safetensors").string();
  const std::string anime =
      "anime=" + (root / "rife47.safetensors").string();
  const std::string general =
      (root / "realesr-general-x4v3.safetensors").string();
  const char* video_argv[] = {"--rife-model", rife.c_str(), "--upscale-model",
                              anime.c_str(), "--upscale-model",
                              general.c_str()};
  assert(video.Parse(video_argv, &error));
  server::VideoPostProcessors tools;
  assert(server::BuildVideoPostProcessors(post, &tools, &error));
  assert(tools.interpolator && tools.upscaler);
  assert((tools.upscale_models ==
          std::vector<std::string>{"anime", "realesr-general-x4v3"}));
  const char* repeated[] = {"--upscale-model", anime.c_str()};
  assert(!video.Parse(repeated, &error) &&
         error.find("repeated") != std::string::npos);
  server::VideoPostProcessArgs missing;
  missing.rife_model = root / "missing.safetensors";
  assert(!server::BuildVideoPostProcessors(missing, &tools, &error) &&
         error.find("--rife-model") != std::string::npos);
#if !defined(ENGINE_ENABLE_HIP)
  // Without HIP the post-steps exist but refuse to run, with a reason.
  (void)server::BuildVideoPostProcessors(post, &tools, &error);
  std::string run_error;
  assert(!tools.interpolator(root / "in.mp4", root / "out.mp4",
                             server::VideoInterpolation{.multiplier = 2}, {},
                             nullptr, nullptr, &run_error) &&
         run_error.find("HIP") != std::string::npos);
#endif
}

void TestPostProcessParser() {
  json::Value body = json::parse(
      R"({"model":"m","gufo":{"seed":1,"interpolate":{"fps":"30000/1001"},"upscale":{"width":1280,"height":720,"fit":"crop"}}})");
  server::VideoPostProcess post;
  server::VideoPostProcessError error;
  assert(server::ExtractVideoPostProcess(&body, &post, &error));
  assert(post.interpolation.fps_num == 30000 &&
         post.interpolation.fps_den == 1001 && post.upscale.width == 1280 &&
         post.upscale.fit == "crop");
  assert(body.find("gufo")->contains("seed") &&
         !body.find("gufo")->contains("interpolate") &&
         !body.find("gufo")->contains("upscale"));

  body = json::parse(R"({"model":"m","gufo":{"interpolate":3}})");
  assert(server::ExtractVideoPostProcess(&body, &post, &error));
  assert(post.interpolation.multiplier == 3 && !post.upscale.enabled());
  assert(!body.contains("gufo") && body.contains("model"));

  for (const char* bad : {R"({"gufo":{"interpolate":0}})",
                          R"({"gufo":{"interpolate":{"fps":"x"}}})",
                          R"({"gufo":{"upscale":"1x"}})",
                          R"({"gufo":{"upscale":{"width":2,"height":2,"x":1}}})"}) {
    body = json::parse(bad);
    assert(!server::ExtractVideoPostProcess(&body, &post, &error));
    assert(error.code == "invalid_interpolate" ||
           error.code == "invalid_upscale");
  }

  server::VideoPostProcessors tools;
  std::string message;
  post = {};
  post.upscale = {.width = 1280, .height = 720};
  assert(!server::ValidateVideoPostProcess(post, tools, "mp4", &message) &&
         message.find("--upscale-model") != std::string::npos);
  tools.upscaler = [](const std::filesystem::path&,
                      const std::filesystem::path&, const server::VideoUpscale&,
                      const gufo::ext::CancellationCheck&,
                      gufo::ext::VideoProgress, void*,
                      std::string*) { return true; };
  tools.upscale_models = {"anime"};
  assert(server::ValidateVideoPostProcess(post, tools, "mp4", &message));
  assert(!server::ValidateVideoPostProcess(post, tools, "ppm", &message));
  assert(server::VideoPostProcessPercent("upscale", 1, 2) == 98);
  assert(server::VideoPostProcessPercent("upscale", 2, 2) == 99);
  assert(!server::VideoPostProcessPercent("denoiser", 1, 2).has_value());
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "gufo-media-services-test";
  std::error_code ignored;
  std::filesystem::remove_all(root, ignored);
  std::filesystem::create_directories(root);
  TestUpscaleService();
  TestMusicAndFoleyStartup(root);
  TestServeOptions(root);
  TestPostProcessParser();
  std::filesystem::remove_all(root, ignored);
  std::cout << "media services tests passed\n";
  return 0;
}
