#ifndef GUFO_MODELS_RIFE_INTERPOLATE_HPP_
#define GUFO_MODELS_RIFE_INTERPOLATE_HPP_

#include <filesystem>
#include <functional>
#include <string>

#include "src/models/rife/ifnet.hpp"

namespace gufo::models::rife {

struct InterpolateVideoOptions {
  int multiplier{2};         // 2..8: output frames = (n - 1) * m + 1
  float scale_factor{1.0F};  // ComfyUI RIFE VFI "scale_factor"
  // Output frame rate; 0 keeps the motion speed (input fps * multiplier).
  // Any other rate (e.g. the input rate for slow motion) drops the audio.
  int fps_num{0};
  int fps_den{1};
  bool keep_audio{true};
  // progress(done, total) after each written frame (total may be estimated);
  // cancelled() is polled once per frame.
  std::function<void(int, int)> progress;
  std::function<bool()> cancelled;
};

struct InterpolateVideoStats {
  int width{0};
  int height{0};
  int input_frames{0};
  int output_frames{0};
  int fps_num{0};
  int fps_den{1};
  bool audio{false};
  double encode_ms{0.0};       // encoder (per input frame), summed
  double interpolate_ms{0.0};  // IFNet (per new frame), summed
  double total_seconds{0.0};
};

/// Decodes `input` with FFmpeg, inserts multiplier - 1 RIFE frames between
/// every pair (timesteps k / multiplier) and encodes `output` (H.264, the
/// audio stream copied when kept). The interpolator is configured for the
/// video's size.
[[nodiscard]] bool InterpolateVideoFile(Interpolator& interpolator,
                                        const std::filesystem::path& input,
                                        const std::filesystem::path& output,
                                        const InterpolateVideoOptions& options,
                                        InterpolateVideoStats* stats,
                                        std::string* error);

}  // namespace gufo::models::rife

#endif  // GUFO_MODELS_RIFE_INTERPOLATE_HPP_
