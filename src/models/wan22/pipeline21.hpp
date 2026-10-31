#ifndef GUFO_MODELS_WAN22_PIPELINE21_HPP_
#define GUFO_MODELS_WAN22_PIPELINE21_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/models/preview_hook.hpp"
#include "src/models/wan22/camera_path.hpp"
#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// Wan 2.1 image-to-video 14B with the lightx2v step/CFG-distill LoRA, as
/// ComfyUI samples it: umT5 prompt (no CFG), CLIP-H image states, VAE
/// condition (first frame then zero frames, normalized) with the 4-channel
/// first-frame mask, LCM sampling over shifted flow sigmas
/// (sigma = shift * t / (1 + (shift - 1) * t), t = 1, 1 - 1/n, ...; timestep
/// = 1000 * sigma; x <- (1 - sigma') * x0 + sigma' * fresh noise), VAE
/// decode. Optionally the Uni3C ControlNet steers the camera with a guide
/// video (VAE-encoded, residuals added after the first 20 blocks).
struct I2V21Request {
  std::filesystem::path wan21_root;  // Diffusers Wan2.1-I2V-14B-480P: vae,
                                     // image_encoder
  std::filesystem::path t5_root;     // Diffusers root with tokenizer/ and
                                     // text_encoder/ (umT5-XXL)
  std::filesystem::path dit_file;    // Kijai FP8 checkpoint
  std::filesystem::path lora_file;   // optional
  float lora_strength{1.0F};
  std::string prompt;
  int width{832};
  int height{480};
  int frames{81};
  int steps{4};
  float shift{5.0F};
  std::uint64_t seed{42};
  /// RGB8 [height, width, 3] at the video size.
  std::span<const std::uint8_t> first_frame;
  /// Uni3C: ControlNet file and guide video RGB8 [guide_frames, height,
  /// width, 3] (repeated last frame / cut to `frames`).
  std::filesystem::path uni3c_file;
  std::span<const std::uint8_t> guide;
  int guide_frames{0};
  float uni3c_strength{1.0F};
  /// Instead of `guide`: render it from the first frame with MoGe-2
  /// (`moge_file`, tools-local/moge_to_safetensors.py) along `camera`.
  std::filesystem::path moge_file;
  CameraSpec camera;
};

struct I2V21Telemetry {
  double text_ms{0}, clip_ms{0}, condition_ms{0}, guide_ms{0};
  double dit_load_ms{0}, context_ms{0}, decode_ms{0}, total_ms{0};
  double depth_ms{0}, render_ms{0};
  double scene_depth{0}, fx{0}, metric_scale{0};
  std::vector<double> step_ms;
  /// Live previews handed out and the time spent forming them.
  double preview_ms{0};
  int previews{0};
  std::size_t peak_device_bytes{0};
};

struct I2V21Callbacks {
  /// Intermediates for oracle checks: name, FP32 values, shape.
  std::function<void(const std::string& name, const std::vector<float>& values,
                     const std::vector<std::size_t>& shape)>
      dump;
  std::function<void(const std::string& phase, int done, int total)> progress;
  /// Returning false cancels between stages and steps.
  std::function<bool()> keep_going;
  /// The rendered guide video, RGB8 [frames, height, width, 3].
  std::function<void(const std::vector<std::uint8_t>& frames, int count)>
      rendered_guide;
  /// Live preview: the LCM estimate x0 = x - sigma * v of a step, normalized
  /// Wan 2.1 latents [16][frames][h][w] (channel-major). Read-only: the
  /// sampled video is the same with or without it.
  PreviewHook preview;
};

/// MoGe-2 + camera path + point render only: the Uni3C guide for
/// `request.first_frame` (RGB8 [frames, height, width, 3]).
bool RenderCameraGuide(const I2V21Request& request,
                       std::vector<std::uint8_t>* frames,
                       I2V21Telemetry* telemetry, std::string* error);

bool GenerateI2V21(const I2V21Request& request, const I2V21Callbacks& callbacks,
                   std::vector<float>* final_latents, DecodedVideo* video,
                   I2V21Telemetry* telemetry, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_PIPELINE21_HPP_
