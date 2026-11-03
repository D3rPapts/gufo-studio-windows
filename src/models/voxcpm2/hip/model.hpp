// gfx1151 HIP runtime for the VoxCPM2 AR loop: base LM and residual LM with
// device KV caches (grown on demand, up to the 8192-position context), local
// encoder, local DiT with the complete Euler/CFG-zero-star sampler on the
// device (HIP graph per timestep setting), FSQ, fusion and stop heads.
// Weights are uploaded from the CPU model's mapped BF16 tensors; activations
// stay FP32 (<= 8 rows: Fish Drama-3 BF16 GEMV, more: the Ming FP32 x BF16
// tiled GEMM). Text embedding and the AudioVAE stay on the CPU.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_HIP_MODEL_HPP_
#define GUFO_MODELS_VOXCPM2_HIP_MODEL_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/models/voxcpm2/backend.hpp"
#include "src/models/voxcpm2/cpu/model.hpp"

namespace gufo::models::voxcpm2::hip {

class Model final : public Backend {
public:
  // Uploads the weights of `cpu_model` (which must outlive this object).
  // Returns nullptr (with `error`) when no HIP device is usable.
  [[nodiscard]] static std::unique_ptr<Model> Create(const cpu::Model& cpu_model,
                                                     std::string* error);
  ~Model() override;

  [[nodiscard]] const char* name() const noexcept override { return "hip"; }
  void Prefill(const std::int64_t* ids, const float* feats, const std::uint8_t* audio_mask,
               std::int64_t rows, float* lm_hidden, float* residual_hidden) override;
  void SamplePatch(const float* lm_hidden, const float* residual_hidden,
                   const float* prev_patch, const float* noise, const FlowOptions& options,
                   float* patch) override;
  void StopLogits(const float* lm_hidden, float* logits) override;
  void Advance(const float* patch, float* lm_hidden, float* residual_hidden) override;

  // Single CFG estimator call (tests): velocities [2, patch, latent].
  void Velocity(const float* x, const float* mu, const float* cond, float t, float* out);
  // Local encoder CLS outputs [count, enc_hidden] (tests).
  void LocalEncode(const float* patches, std::int64_t count, float* out);

  // HIP graphs for the flow sampler (default on; GUFO_VOXCPM_GRAPHS=0 at creation).
  void SetGraphs(bool enabled);
  [[nodiscard]] bool graphs_enabled() const noexcept;

  struct Impl;

private:
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// True when a HIP device is present (cheap probe, no allocation).
[[nodiscard]] bool DeviceAvailable() noexcept;

}  // namespace gufo::models::voxcpm2::hip

#endif  // GUFO_MODELS_VOXCPM2_HIP_MODEL_HPP_
