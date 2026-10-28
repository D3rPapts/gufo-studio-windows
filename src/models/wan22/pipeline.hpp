#ifndef GUFO_MODELS_WAN22_PIPELINE_HPP_
#define GUFO_MODELS_WAN22_PIPELINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// Text-to-video with Wan 2.2 TI2V-5B, as Diffusers' WanPipeline does it:
/// umT5 prompt and negative prompt (default "", the pipeline default), UniPC
/// flow sampling with classifier-free guidance, VAE decode.
struct VideoRequest {
  std::filesystem::path model_root;  // Diffusers layout
  std::string prompt;
  std::string negative_prompt;
  int width{1280};
  int height{704};
  int frames{121};
  int steps{50};
  float guidance{5.0F};
  std::uint64_t seed{42};
  /// Optional initial latents [48, (frames - 1) / 4 + 1, height / 16,
  /// width / 16]; otherwise standard normal noise from `seed`.
  std::span<const float> noise;
};

struct VideoTelemetry {
  double text_ms{0};
  double transformer_load_ms{0};
  double denoise_ms{0};
  double vae_load_ms{0};
  double vae_ms{0};
  double total_ms{0};
  int prompt_tokens{0};
  int negative_tokens{0};
  std::size_t peak_device_bytes{0};
  std::vector<double> step_ms;
};

struct VideoCallbacks {
  /// Phases: "prompt", "denoiser", "video_vae".
  std::function<void(std::string_view phase, int done, int total)> progress;
  /// Returning false cancels between steps / frames.
  std::function<bool()> keep_going;
  /// Latents after each step (step index from 0).
  std::function<void(int step, std::span<const float> latents)> on_step;
};

/// Checks width/height (multiples of 32) and frames (4k + 1).
[[nodiscard]] bool ValidateVideoShape(int width, int height, int frames,
                                      std::string* error = nullptr);

/// Deterministic standard normal noise (splitmix64 + Box-Muller).
[[nodiscard]] std::vector<float> GaussianNoise(std::size_t count,
                                               std::uint64_t seed);

/// Runs the pipeline. Fills `final_latents` (model space) and/or `video`
/// (8-bit RGB; float RGB too when `float_video` is set) when non-null.
/// Each model is loaded for its stage and released afterwards.
bool GenerateVideo(const VideoRequest& request, const VideoCallbacks& callbacks,
                   std::vector<float>* final_latents, DecodedVideo* video,
                   bool float_video, VideoTelemetry* telemetry,
                   std::string* error);

/// Encodes 8-bit RGB frames to H.264 MP4 through FFmpeg (GUFO_FFMPEG or
/// `ffmpeg` beside the executable / on PATH); writes `path` atomically.
bool WriteMp4(const std::filesystem::path& path, const DecodedVideo& video,
              int fps, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_PIPELINE_HPP_
