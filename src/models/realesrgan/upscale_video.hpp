#ifndef GUFO_MODELS_REALESRGAN_UPSCALE_VIDEO_HPP_
#define GUFO_MODELS_REALESRGAN_UPSCALE_VIDEO_HPP_

#include <filesystem>
#include <functional>
#include <string>

#include "src/models/realesrgan/upscaler.hpp"

namespace gufo::models::realesrgan {

struct UpscaleVideoOptions {
  int width{1920};
  int height{1200};
  Fit fit{Fit::kPad};
  int crf{18};  // H.264 constant rate factor of the output
  bool keep_audio{true};
  // progress(done, total) after each written frame (total may be estimated);
  // cancelled() is polled once per frame.
  std::function<void(int, int)> progress;
  std::function<bool()> cancelled;
};

struct UpscaleVideoStats {
  int in_width{0};
  int in_height{0};
  int frames{0};
  int fps_num{0};
  int fps_den{1};
  bool audio{false};
  Placement placement;
  double upscale_ms{0.0};  // network + resize + transfers, summed
  double total_seconds{0.0};
};

/// Decodes `input` with FFmpeg, upscales every frame to options.width x
/// options.height and encodes `output` (H.264 at the input frame rate, the
/// audio copied when kept; a ".png" output takes the first frame only).
[[nodiscard]] bool UpscaleVideoFile(Upscaler& upscaler,
                                    const std::filesystem::path& input,
                                    const std::filesystem::path& output,
                                    const UpscaleVideoOptions& options,
                                    UpscaleVideoStats* stats,
                                    std::string* error);

}  // namespace gufo::models::realesrgan

#endif  // GUFO_MODELS_REALESRGAN_UPSCALE_VIDEO_HPP_
