// `gufo serve music | foley | upscale` and the video post-step options (see
// media_serve.hpp). Options and messages as in gufo-studio-windows 1.4
// (serve.cpp), moved out of serve.cpp.
// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/media_serve.hpp"

#include <system_error>

#include "src/cli/serve/media/foley_api.hpp"
#include "src/cli/serve/media/music_api.hpp"
#include "src/cli/serve/media/video_rife.hpp"
#include "src/cli/serve/media/video_upscale.hpp"

namespace gufo::server {

bool IsMediaServeCommand(std::string_view subcommand) {
  return subcommand == "music" || subcommand == "foley" ||
         subcommand == "upscale";
}

std::string MediaServeCommandsHelp() {
  return "  music     Serve ACE-Step 1.5 XL turbo text-to-music "
         "(/v1/audio/music)\n"
         "  foley     Serve MMAudio large_44k_v2 video/text-to-audio "
         "(/v1/audio/foley)\n"
         "  upscale   Serve Real-ESRGAN image upscaling on the NPU, GPU or "
         "CPU (/v1/images/upscale)\n";
}

std::string MediaServeDescription(std::string_view subcommand) {
  if (subcommand == "music")
    return "Serve ACE-Step 1.5 XL turbo text-to-music.";
  if (subcommand == "foley")
    return "Serve MMAudio large_44k_v2 video/text-to-audio.";
  return "Serve Real-ESRGAN image upscaling (POST /v1/images/upscale) on the "
         "AMD NPU (ONNX Runtime + VitisAI from Windows ML), the GPU (HIP) or "
         "the CPU (ONNX Runtime).";
}

void AddMediaServeOptions(std::string_view subcommand, cli::ArgParser& parser,
                          MediaServeArgs* args) {
  if (subcommand == "music") {
    args->name = "ace-step-1.5-xl-turbo";
    parser.AddOption("-m", "--model", "DIR",
                     "ACE-Step directory (dit.safetensors, "
                     "text_encoder.safetensors, vae.safetensors, "
                     "tokenizer.json, silence_latent.safetensors)",
                     "Model", &args->model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &args->name);
    return;
  }
  if (subcommand == "foley") {
    args->name = "mmaudio-44k";
    parser.AddOption("-m", "--model", "DIR",
                     "MMAudio directory (mmaudio_large_44k_v2_*, "
                     "mmaudio_vae_44k_*, mmaudio_synchformer_*, "
                     "apple_DFN5B-CLIP-ViT-H-14-384_* .safetensors, "
                     "bigvgan_v2_44khz_128band_512x.safetensors, tokenizer/)",
                     "Model", &args->model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &args->name);
    return;
  }
  UpscaleServeOptions& options = args->upscale;
  parser.AddOption("", "--device", "npu|gpu|cpu",
                   "Where the network runs (default: npu)", "Model",
                   &options.device);
  parser.AddCustomOption(
      "-m", "--model", "[NAME=]FILE",
      "npu/cpu: Real-ESRGAN body ONNX with a fixed input size (e.g. "
      "realesr-animevideov3=body896q_h288_h3.onnx); gpu: converted "
      "safetensors. Repeatable; one name may have one file per frame "
      "width; the first name is the default. Without NAME: "
      "--served-model-name, else the file stem",
      "Model",
      [&options](std::string_view, std::string_view value,
                 std::string* error) {
        const std::size_t equals = value.find('=');
        std::filesystem::path path(std::string(
            equals == std::string_view::npos ? value
                                             : value.substr(equals + 1)));
        std::string name = equals == std::string_view::npos
                               ? std::string()
                               : std::string(value.substr(0, equals));
        if ((equals != std::string_view::npos && name.empty()) ||
            path.empty()) {
          *error = "--model expects [NAME=]FILE";
          return false;
        }
        options.models.push_back({std::move(name), std::move(path)});
        return true;
      });
  parser.AddOption("", "--served-model-name", "NAME",
                   "Public model ID of --model files given without NAME= "
                   "(the router passes the section id)",
                   "Model", &args->name);
  parser.AddOption("", "--onnxruntime", "DLL",
                   "onnxruntime.dll 1.23+ (default: Windows ML)", "Model",
                   &options.onnxruntime);
  parser.AddOption("", "--npu-ep", "DLL",
                   "onnxruntime_vitisai_ep.dll (default: from the Windows "
                   "ML execution provider catalog)",
                   "Model", &options.npu_ep);
  parser.AddOption("", "--npu-cache", "DIR",
                   "VitisAI compile cache (default: <model dir>/npu-cache)",
                   "Model", &options.npu_cache);
  parser.AddOption("", "--npu-config", "JSON",
                   "VitisAI config_file (e.g. vaiml_config.json)", "Model",
                   &options.npu_config);
  parser.AddOption("", "--threads", "N",
                   "CPU provider threads (default: ONNX Runtime's)", "Model",
                   &options.threads);
  parser.AddFlag("", "--fp32", "gpu: FP32 validation path", "Model",
                 &options.fp32);
}

bool ValidateMediaServeArgs(std::string_view subcommand, MediaServeArgs* args,
                            std::string* error) {
  if (subcommand == "music" || subcommand == "foley") {
    if (args->model.empty()) {
      *error = "--model <DIR> is required for " + std::string(subcommand) +
               " server";
      return false;
    }
    return true;
  }
  UpscaleServeOptions& options = args->upscale;
  if (options.device != "npu" && options.device != "gpu" &&
      options.device != "cpu") {
    *error = "--device must be npu, gpu or cpu";
    return false;
  }
  if (options.models.empty()) {
    *error = "--model [NAME=]FILE is required for upscale server";
    return false;
  }
  for (UpscaleModelArg& m : options.models)
    if (m.name.empty())
      m.name = args->name.empty() ? m.path.stem().string() : args->name;
  for (const UpscaleModelArg& m : options.models) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(m.path, ec)) {
      *error = "--model file not found: " + m.path.string();
      return false;
    }
  }
  return true;
}

std::filesystem::path MediaServeArtifact(std::string_view subcommand,
                                         const MediaServeArgs& args) {
  if (subcommand == "upscale")
    return args.upscale.models.empty() ? std::filesystem::path()
                                       : args.upscale.models.front().path;
  return args.model;
}

std::shared_ptr<MediaService> LoadMediaService(std::string_view subcommand,
                                               const MediaServeArgs& args,
                                               std::string* details,
                                               std::string* error) {
  if (subcommand == "music") {
    auto music = std::make_shared<MusicService>(args.model, args.name);
    if (!music->ready()) {
      *error = "Error loading ACE-Step: " + music->initialization_error();
      return nullptr;
    }
    *details = "model=" + args.name;
    return music;
  }
  if (subcommand == "foley") {
    auto foley = std::make_shared<FoleyService>(args.model, args.name);
    if (!foley->ready()) {
      *error = "Error loading MMAudio: " + foley->initialization_error();
      return nullptr;
    }
    *details = "model=" + args.name;
    return foley;
  }
  std::string engine_error;
  std::unique_ptr<UpscaleEngine> engine =
      CreateUpscaleEngine(args.upscale, &engine_error);
  if (engine == nullptr) {
    *error = "Error loading the upscaler (" + args.upscale.device +
             "): " + engine_error;
    return nullptr;
  }
  const std::string runtime = engine->runtime;
  auto upscale = std::make_shared<UpscaleService>(std::move(engine));
  *details = "model=" + upscale->model_id() + " device=" +
             args.upscale.device + " runtime=\"" + runtime + "\"";
  return upscale;
}

void AddVideoPostProcessOptions(cli::ArgParser& parser,
                                VideoPostProcessArgs* args) {
  parser.AddOption("", "--rife-model", "SAFETENSORS",
                   "RIFE 4.7 weights; enables \"gufo\":{\"interpolate\":N or "
                   "{\"fps\":N}} and source_video jobs",
                   "Model", &args->rife_model);
  parser.AddCustomOption(
      "", "--upscale-model", "[NAME=]SAFETENSORS",
      "Real-ESRGAN SRVGGNetCompact weights (converted); enables "
      "\"gufo\":{\"upscale\":{\"width\",\"height\",\"fit\",\"model\"}}; "
      "NAME defaults to the file stem; repeatable, the first is the "
      "default",
      "Model",
      [args](std::string_view, std::string_view value, std::string* error) {
        const std::size_t equals = value.find('=');
        std::filesystem::path path(std::string(
            equals == std::string_view::npos ? value
                                             : value.substr(equals + 1)));
        std::string name = equals == std::string_view::npos
                               ? path.stem().string()
                               : std::string(value.substr(0, equals));
        if (name.empty() || path.empty()) {
          *error = "--upscale-model expects [NAME=]SAFETENSORS";
          return false;
        }
        for (const auto& [known, unused] : args->upscale_models) {
          if (known == name) {
            *error = "--upscale-model name '" + name + "' is repeated";
            return false;
          }
        }
        args->upscale_models.emplace_back(std::move(name), std::move(path));
        return true;
      });
}

bool BuildVideoPostProcessors(const VideoPostProcessArgs& args,
                              VideoPostProcessors* tools,
                              std::string* error) {
  std::error_code ec;
  if (!args.rife_model.empty() &&
      !std::filesystem::is_regular_file(args.rife_model, ec)) {
    *error = "--rife-model file not found: " + args.rife_model.string();
    return false;
  }
  for (const auto& [name, path] : args.upscale_models) {
    if (!std::filesystem::is_regular_file(path, ec)) {
      *error = "--upscale-model file not found: " + path.string();
      return false;
    }
  }
  *tools = {};
  if (!args.rife_model.empty()) {
    tools->interpolator =
        [weights = args.rife_model](
            const std::filesystem::path& input,
            const std::filesystem::path& output,
            const VideoInterpolation& spec,
            const ext::CancellationCheck& cancelled,
            ext::VideoProgress progress, void* opaque, std::string* message) {
          return RunRifeInterpolation(weights, input, output, spec, cancelled,
                                      progress, opaque, message);
        };
  }
  if (!args.upscale_models.empty()) {
    for (const auto& [name, path] : args.upscale_models)
      tools->upscale_models.push_back(name);
    tools->upscaler =
        [models = args.upscale_models](
            const std::filesystem::path& input,
            const std::filesystem::path& output, const VideoUpscale& spec,
            const ext::CancellationCheck& cancelled,
            ext::VideoProgress progress, void* opaque, std::string* message) {
          std::filesystem::path weights = models.front().second;
          for (const auto& [name, path] : models)
            if (name == spec.model)
              weights = path;
          return RunRealEsrganUpscale(weights, input, output, spec, cancelled,
                                      progress, opaque, message);
        };
  }
  return true;
}

}  // namespace gufo::server
