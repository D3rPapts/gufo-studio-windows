#include "src/cli/serve/video_upscale.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/realesrgan/upscale_video.hpp"
#endif

namespace gufo::server {

bool RunRealEsrganUpscale(const std::filesystem::path& weights,
                          const std::filesystem::path& input,
                          const std::filesystem::path& output,
                          const VideoUpscale& spec,
                          const minimax_h3::CancellationToken* cancellation,
                          minimax_h3::GenerationProgress progress,
                          void* progress_opaque, std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  namespace esr = models::realesrgan;
  if (progress != nullptr)
    progress("upscale", 0, 1, progress_opaque);
  esr::UpscaleVideoOptions options;
  options.width = spec.width;
  options.height = spec.height;
  if (!esr::ParseFit(spec.fit, &options.fit)) {
    if (error != nullptr)
      *error = "unknown upscale fit '" + spec.fit + "'";
    return false;
  }
  auto upscaler = esr::Upscaler::Load(weights, esr::Precision::kFp16, error);
  if (!upscaler)
    return false;
  options.progress = [&](int done, int total) {
    if (progress != nullptr)
      progress("upscale", done, total, progress_opaque);
  };
  options.cancelled = [cancellation] {
    return cancellation != nullptr && cancellation->IsCancelled();
  };
  return esr::UpscaleVideoFile(*upscaler, input, output, options, nullptr,
                               error);
#else
  (void)weights;
  (void)input;
  (void)output;
  (void)spec;
  (void)cancellation;
  (void)progress;
  (void)progress_opaque;
  if (error != nullptr)
    *error = "Real-ESRGAN upscaling requires a ROCm HIP-enabled build";
  return false;
#endif
}

}  // namespace gufo::server
