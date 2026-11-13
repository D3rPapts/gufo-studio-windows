#ifndef GUFO_MODELS_WAN22_PIPELINE_VACE_HPP_
#define GUFO_MODELS_WAN22_PIPELINE_VACE_HPP_

// Written with AI assistance (Claude Code).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/pipeline21.hpp"
#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// Wan 2.1 VACE 14B (video editing / control video / masks / reference
/// images) as ComfyUI runs WanVaceToVideo -> KSampler -> TrimVideoLatent ->
/// VAEDecode on the T2V 14B with the VACE module:
///  - control video (RGB, cut to `frames` or padded with grey 0.5) and mask
///    (1 = generate, padded with 1) split into inactive = v * (1 - m) and
///    reactive = v * m around grey, both VAE-encoded and normalized;
///  - reference images VAE-encoded as extra latent frames in front ([latent
///    | 0], mask 0), trimmed after sampling;
///  - context [96 = 16 inactive + 16 reactive + 64 unshuffled mask];
///  - flow sampling over shifted sigmas (sigma = shift * t / (1 + (shift -
///    1) * t)), "lcm" (x0 then fresh noise, for the lightx2v step-distill
///    LoRA) or "euler"; guidance > 1 runs classifier-free guidance with the
///    negative prompt (two transformer passes a step);
///  - Wan 2.1 VAE decode.
struct Vace21Request {
  std::filesystem::path vae_root;  // Diffusers Wan 2.1 root with vae/
  std::filesystem::path t5_root;   // Diffusers root with tokenizer/ and
                                   // text_encoder/ (umT5-XXL)
  std::filesystem::path dit_file;  // Kijai T2V 14B FP8 (scaled) checkpoint
  std::filesystem::path vace_file;  // Kijai VACE module (FP8 scaled / BF16)
  std::filesystem::path lora_file;  // optional (lightx2v T2V)
  float lora_strength{1.0F};
  std::string prompt;
  std::string negative_prompt;
  float guidance{1.0F};
  std::string sampler{"lcm"};  // "lcm" or "euler"
  int width{832};
  int height{480};
  int frames{81};
  int steps{4};
  float shift{5.0F};
  std::uint64_t seed{42};
  float strength{1.0F};  // WanVaceToVideo strength
  /// Control video RGB8 [control_frames, height, width, 3] at the video
  /// size; empty: grey (pure text-to-video / reference-to-video).
  std::span<const std::uint8_t> control;
  int control_frames{0};
  /// Mask [mask_frames, height, width] in 0..1 (1 = generate); empty: all
  /// ones.
  std::span<const float> mask;
  int mask_frames{0};
  /// Reference images RGB8 [height, width, 3] at the video size.
  std::vector<std::span<const std::uint8_t>> references;
};

struct Vace21Telemetry {
  double text_ms{0}, condition_ms{0}, dit_load_ms{0}, context_ms{0};
  double decode_ms{0}, total_ms{0};
  std::vector<double> step_ms;
  double preview_ms{0};
  int previews{0};
  std::size_t peak_device_bytes{0};
};

/// `callbacks.rendered_guide` is unused. Intermediates for `dump`: text,
/// negative_text, vace_context, noise0, step<i>_velocity, final.
bool GenerateVace21(const Vace21Request& request,
                    const I2V21Callbacks& callbacks,
                    std::vector<float>* final_latents, DecodedVideo* video,
                    Vace21Telemetry* telemetry, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_PIPELINE_VACE_HPP_
