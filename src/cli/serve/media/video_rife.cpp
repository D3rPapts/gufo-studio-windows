#include "src/cli/serve/media/video_rife.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/rife/interpolate.hpp"
#endif

namespace gufo::server {

bool RunRifeInterpolation(const std::filesystem::path& weights,
                          const std::filesystem::path& input,
                          const std::filesystem::path& output,
                          const VideoInterpolation& spec,
                          const ext::CancellationCheck& cancelled,
                          ext::VideoProgress progress,
                          void* progress_opaque, std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  namespace rife = models::rife;
  if (progress != nullptr)
    progress("interpolate", 0, 1, progress_opaque);
  auto interpolator =
      rife::Interpolator::Load(weights, rife::Precision::kFp16, error);
  if (!interpolator)
    return false;
  rife::InterpolateVideoOptions options;
  options.multiplier = spec.multiplier;
  options.target_fps_num = spec.fps_num;
  options.target_fps_den = spec.fps_den;
  options.progress = [&](int done, int total) {
    if (progress != nullptr)
      progress("interpolate", done, total, progress_opaque);
  };
  options.cancelled = [&cancelled] { return cancelled && cancelled(); };
  return rife::InterpolateVideoFile(*interpolator, input, output, options,
                                    nullptr, error);
#else
  (void)weights;
  (void)input;
  (void)output;
  (void)spec;
  (void)cancelled;
  (void)progress;
  (void)progress_opaque;
  if (error != nullptr)
    *error = "RIFE frame interpolation requires a ROCm HIP-enabled build";
  return false;
#endif
}

}  // namespace gufo::server
