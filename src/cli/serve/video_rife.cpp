#include "src/cli/serve/video_rife.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/rife/interpolate.hpp"
#endif

namespace gufo::server {

bool RunRifeInterpolation(const std::filesystem::path& weights,
                          const std::filesystem::path& input,
                          const std::filesystem::path& output, int multiplier,
                          const minimax_h3::CancellationToken* cancellation,
                          minimax_h3::GenerationProgress progress,
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
  options.multiplier = multiplier;
  options.progress = [&](int done, int total) {
    if (progress != nullptr)
      progress("interpolate", done, total, progress_opaque);
  };
  options.cancelled = [cancellation] {
    return cancellation != nullptr && cancellation->IsCancelled();
  };
  return rife::InterpolateVideoFile(*interpolator, input, output, options,
                                    nullptr, error);
#else
  (void)weights;
  (void)input;
  (void)output;
  (void)multiplier;
  (void)cancellation;
  (void)progress;
  (void)progress_opaque;
  if (error != nullptr)
    *error = "RIFE frame interpolation requires a ROCm HIP-enabled build";
  return false;
#endif
}

}  // namespace gufo::server
