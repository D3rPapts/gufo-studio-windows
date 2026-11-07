#ifndef GUFO_MODELS_WAN22_VACE_CONTEXT_HPP_
#define GUFO_MODELS_WAN22_VACE_CONTEXT_HPP_

// Written with AI assistance (Claude Code).
// Host-side VACE conditioning (no device code; unit-tested in
// tests/models/wan22_vace_context_test.cpp).

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::wan22 {

/// Inputs of BuildVaceContext, all at the video size (width, height
/// multiples of 16, frames = 4k + 1) and already normalized latents from
/// VaeEncoder::EncodeVideo(..., normalize = true).
struct VaceContextInputs {
  int frames{0};  // pixel frames of the video
  int height{0};  // pixels
  int width{0};
  /// [16, latent_frames, height / 8, width / 8] each.
  std::span<const float> inactive;
  std::span<const float> reactive;
  /// Pixel mask [frames, height, width] in 0..1 (1 = generate); empty = all
  /// ones.
  std::span<const float> mask;
  /// Normalized latents of the reference images, [16, 1, h / 8, w / 8] each,
  /// placed in front of the video in this order.
  std::vector<std::span<const float>> references;
};

/// ComfyUI's WanVaceToVideo + WAN21_Vace.extra_conds: concatenates
/// [inactive | reactive] (16 + 16 channels; reference frames: [latent | 0]),
/// then the mask pixel-unshuffled by 8 (channel kh * 8 + kw) and resampled
/// to the latent frames with nearest-exact (zeros on the reference frames).
/// Returns [96, references + latent_frames, height / 8, width / 8], empty on
/// a size mismatch.
[[nodiscard]] std::vector<float> BuildVaceContext(
    const VaceContextInputs& inputs);

/// ComfyUI's control-video split before the VAE: with the control video v in
/// [0, 1] and the mask m, inactive = (v - 0.5) * (1 - m) + 0.5 and reactive
/// = (v - 0.5) * m + 0.5; written in [-1, 1] for VaeEncoder ([frames, height,
/// width, 3] each). `video` RGB8 [video_frames, height, width, 3] (its first
/// video_frames frames are read; later frames and an empty video are grey
/// 0.5 as ComfyUI pads), `mask` as in VaceContextInputs (empty = all ones).
void SplitVaceControl(std::span<const std::uint8_t> video, int video_frames,
                      std::span<const float> mask, int frames, int height,
                      int width, std::vector<float>* inactive,
                      std::vector<float>* reactive);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_VACE_CONTEXT_HPP_
