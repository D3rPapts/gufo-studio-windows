#ifndef GUFO_MODELS_PREVIEW_HOOK_HPP_
#define GUFO_MODELS_PREVIEW_HOOK_HPP_

#include <functional>
#include <span>

namespace gufo::models {

/// The sampler's current estimate of the clean latent (x0), handed to a live
/// preview. FP32 in the sampler's normalized latent space.
struct LatentPreview {
  int step{0};   // completed denoising steps (1-based)
  int steps{0};  // total denoising steps
  int frames{1};
  int height{0};
  int width{0};
  int channels{0};
  /// false: [frames][height][width][channels] (image tokens);
  /// true:  [channels][frames][height][width] (video tensors).
  bool channel_major{false};
  std::span<const float> data;
};

/// Optional live preview of a running generation. `wanted` is asked once
/// per step and must be cheap; only when it returns true does the model read
/// the latent back and call `publish`, which copies what it needs. Neither
/// changes the result: the preview reads the sampler state, never writes it.
struct PreviewHook {
  std::function<bool(int step, int steps)> wanted;
  std::function<void(const LatentPreview&)> publish;

  bool Wants(int step, int steps) const {
    return wanted && publish && wanted(step, steps);
  }
};

}  // namespace gufo::models
#endif
