#ifndef GUFO_MODELS_QWEN_IMAGE_21_CONTROL_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_CONTROL_HPP_

// Qwen-Image-2.1 Fun ControlNet-Union (alibaba-pai, VideoX-Fun
// `qwenimage21_transformer2d_control.py`): host-side contract. A VACE-style
// branch of 16 copies of the base block (no own modulation) starts from
// `before_proj(control_joint) + x` and adds `after_proj(c) * scale` after
// every second base block. `control_joint` is zero on condition rows and
// `control_img_in(context)` on target rows; the context has 129 channels per
// target latent cell: control latent (64) | keep mask (1) | latent of the
// masked start image (64).

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen_image_21/model.hpp"

namespace gufo::models::qwen_image_21 {

class Weights;

/// Tensor name prefix of the control branch inside Weights.
inline constexpr std::string_view kControlnetPrefix = "controlnet.";
inline constexpr int kControlInChannels = 129;
inline constexpr int kControlLatentChannels = 64;

struct ControlnetInfo {
  int blocks{0};       // control blocks (16)
  int in_channels{0};  // control_img_in input width (129)
  int inner{0};        // hidden width (4096)
  int stride{0};       // base blocks per control block (2): block i -> 2i
  std::size_t bytes{0};
};

struct ControlTensor {
  std::vector<int> shape;
  bool bf16{true};
  std::size_t bytes{0};
};

/// Validates control-branch tensor metadata (names without prefix) against
/// the base transformer shapes (32 blocks of 4096 x 128-dim heads, MLP
/// 12288): control_img_in [4096, 129] + bias, N blocks dividing 32 with the
/// base block's projections (no biases), after_proj in every block,
/// before_proj only in block 0, nothing else. Throws std::runtime_error.
ControlnetInfo InspectControlnet(
    const std::map<std::string, ControlTensor, std::less<>>& tensors);
/// The same for the tensors mapped under `prefix` (metadata only).
ControlnetInfo InspectControlnet(const Weights& weights,
                                 std::string_view prefix = kControlnetPrefix);

/// The control image at the output canvas: alpha dropped (opaque), then
/// resized like the other inputs.
Image ControlCanvas(const Image& image, int width, int height);

/// One value per 16 x 16 latent cell, row-major: 1 keep, 0 regenerate.
/// The mask's edit value (MaskChannel rules) at the pixel (16 r, 16 c) of the
/// canvas (nearest), regenerated at >= 0.5 - as the pipeline's
/// `F.interpolate(1 - mask, mode="nearest")` of the binarized mask.
std::vector<float> ControlKeepMask(const Image& mask, MaskChannel channel,
                                   int width, int height);

/// The inpaint source for the control branch: `image` (already at the
/// canvas size) with regenerated pixels (edit value >= 0.5) set to mid-gray
/// 128, the VAE's zero; alpha opaque.
Image ControlInpaintSource(const Image& image, const Image& mask,
                           MaskChannel channel);

/// rows x 129: control latent | keep | inpaint latent. `inpaint` may be empty
/// (zeros, pure control); `keep` may be empty (zeros, pure control).
std::vector<float> PackControlContext(std::span<const float> control,
                                      std::span<const float> keep,
                                      std::span<const float> inpaint, int rows);

/// True when the control branch runs at `step` of `steps`: the step's
/// fraction step / steps lies in [start, end). end = 1 covers the last step.
bool ControlActiveAt(const ControlInput& control, int step, int steps);

/// Throws std::invalid_argument for out-of-range control settings.
void ValidateControl(const ControlInput& control, const Request& request);

}  // namespace gufo::models::qwen_image_21
#endif
