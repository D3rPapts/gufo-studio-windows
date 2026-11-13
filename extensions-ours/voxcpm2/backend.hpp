// Execution backend for the VoxCPM2 AR loop (both LMs with KV caches, the
// local encoder, the DiT flow sampler, FSQ, fusion and stop heads). The CPU
// reference runtime and the gfx1151 HIP runtime implement it; text, the
// AudioVAE encoder and the streaming AudioVAE decoder stay on the CPU.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_BACKEND_HPP_
#define GUFO_MODELS_VOXCPM2_BACKEND_HPP_

#include <cstdint>

namespace gufo::models::voxcpm2 {

// UnifiedCFM.forward options (VoxCPM2 defaults).
struct FlowOptions {
  float cfg{2.0F};
  std::int32_t timesteps{10};
  float sway_sampling_coef{1.0F};
  bool cfg_zero_star{true};
};

class Backend {
public:
  virtual ~Backend() = default;
  [[nodiscard]] virtual const char* name() const noexcept = 0;

  // Prompt prefill (VoxCPM2Model._inference before the loop). `feats` are the
  // prompt latent patches [rows, patch, latent]; `audio_mask[i]` != 0 marks
  // audio rows, all other rows are text rows taking token `ids[i]`.
  // Clears and fills both KV caches; writes the last row's lm_hidden and
  // residual_hidden [hidden].
  virtual void Prefill(const std::int64_t* ids, const float* feats,
                       const std::uint8_t* audio_mask, std::int64_t rows,
                       float* lm_hidden, float* residual_hidden) = 0;

  // DiT flow sampler for one patch: noise [patch, latent] -> patch
  // [patch, latent], conditioned on lm/residual hidden and the previous patch.
  virtual void SamplePatch(const float* lm_hidden, const float* residual_hidden,
                           const float* prev_patch, const float* noise,
                           const FlowOptions& options, float* patch) = 0;

  // stop_head(silu(stop_proj(lm_hidden))) -> logits [2].
  virtual void StopLogits(const float* lm_hidden, float* logits) = 0;

  // Feeds a generated patch back: local encoder + enc_to_lm_proj, one step of
  // the base LM, FSQ, fusion and one step of the residual LM.
  virtual void Advance(const float* patch, float* lm_hidden, float* residual_hidden) = 0;
};

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_BACKEND_HPP_
