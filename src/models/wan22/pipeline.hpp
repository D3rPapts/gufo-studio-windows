#ifndef GUFO_MODELS_WAN22_PIPELINE_HPP_
#define GUFO_MODELS_WAN22_PIPELINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/preview_hook.hpp"
#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// Text-to-video with Wan 2.2 TI2V-5B, as Diffusers' WanPipeline does it:
/// umT5 prompt and negative prompt (default "", the pipeline default), UniPC
/// flow sampling with classifier-free guidance, VAE decode. With a first
/// frame it is image-to-video as Diffusers' WanImageToVideoPipeline with
/// expand_timesteps: the VAE-encoded image replaces the first latent frame
/// of every transformer input, whose tokens get timestep 0, and of the final
/// latents.
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
  /// Image-to-video: the first frame as RGB8 [height, width, 3] at the video
  /// size (see FitImage). Empty: text-to-video.
  std::span<const std::uint8_t> first_frame;
};

struct VideoTelemetry {
  double text_ms{0};
  double image_ms{0};              // VAE encoder load + first-frame encode
  std::vector<double> forward_ms;  // every transformer call
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
  /// Oracle replay: may replace the latents fed to a step (step from 0).
  std::function<void(int step, std::vector<float>& latents)> before_step;
  /// The guided noise prediction of each step, before the scheduler step.
  std::function<void(int step, std::span<const float> noise)> on_noise;
  /// Live preview: the denoised estimate x0 = x - sigma * v of a step,
  /// channel-major [48][latent frames][height / 16][width / 16].
  PreviewHook preview;
};

/// Checks width/height (multiples of 32) and frames (4k + 1).
[[nodiscard]] bool ValidateVideoShape(int width, int height, int frames,
                                      std::string* error = nullptr);

/// Places an RGB8 image on a width x height canvas: scale to cover, crop the
/// centre, Lanczos-3 (antialiased when shrinking). Returns RGB8
/// [height, width, 3]; empty on invalid input.
[[nodiscard]] std::vector<std::uint8_t> FitImage(
    std::span<const std::uint8_t> rgb, int source_width, int source_height,
    int width, int height);

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
