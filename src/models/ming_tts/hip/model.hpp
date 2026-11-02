// gfx1151 HIP runtime for Ming-3.0-tts-0.5B: backbone with KV cache, DiT
// flow head (10 Euler steps with CFG on the device), aggregator and the VAE
// decoder with its ISTFT head. Weights are uploaded from the CPU model's
// mapped BF16 tensors; activations stay FP32 (small batches use the Fish
// Drama-3 BF16 GEMV, larger ones rocBLAS with BF16 inputs and FP32 output).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_HIP_MODEL_HPP_
#define GUFO_MODELS_MING_TTS_HIP_MODEL_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/ming_tts/backend.hpp"
#include "src/models/ming_tts/cpu/model.hpp"

namespace gufo::models::ming_tts::hip {

class Model final : public Backend {
public:
  // Uploads the weights of `cpu_model` (which must outlive this object).
  // Returns nullptr (with `error`) when no HIP device is usable.
  [[nodiscard]] static std::unique_ptr<Model> Create(const cpu::Model& cpu_model,
                                                     std::string* error);
  ~Model() override;

  [[nodiscard]] const char* name() const noexcept override { return "hip"; }
  void ResetBackbone() override;
  void RunBackbone(const float* embeddings, std::int64_t rows,
                   float* normed) override;
  [[nodiscard]] float StopAt(const float* z) override;
  void SampleFlow(const float* cond, const float* history, const float* noise,
                  const float* sde_noise, const FlowOptions& options,
                  float* latent) override;
  void AggregateLatent(const float* latent, float* out) override;
  [[nodiscard]] std::vector<float> DecodeAudio(const float* latents,
                                               std::int64_t frames) override;

  // Single DiT velocity evaluation (cond, uncond) [2, latent], for tests.
  void FlowVelocity(const float* x, float t, const float* cond,
                    const float* history, float* cond_uncond);

  struct Impl;

private:
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// True when a HIP device is present (cheap probe, no allocation).
[[nodiscard]] bool DeviceAvailable() noexcept;

}  // namespace gufo::models::ming_tts::hip

#endif  // GUFO_MODELS_MING_TTS_HIP_MODEL_HPP_
