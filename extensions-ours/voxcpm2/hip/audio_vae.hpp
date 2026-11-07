// gfx1151 HIP port of the VoxCPM2 AudioVAE V2 (FP32, like the checkpoint):
// causal encoder (reference audio -> latents) and the stateful streaming
// decoder (latents -> 48 kHz). Dense/pointwise convolutions run as FP32 GEMMs
// (im2col for kernels > 1), the transposed convolutions as one GEMM over the
// previous + current frames plus a combine kernel, depthwise convs, Snake and
// the sample-rate conditioning as elementwise kernels. Same math and causal
// state handling as cpu::AudioVae. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_HIP_AUDIO_VAE_HPP_
#define GUFO_MODELS_VOXCPM2_HIP_AUDIO_VAE_HPP_

#include <memory>
#include <string>

#include "voxcpm2/cpu/audio_vae.hpp"
#include "voxcpm2/vae_runner.hpp"

namespace gufo::models::voxcpm2::hip {

class AudioVae final : public VaeRunner {
public:
  // Uploads the FP32 layers of `vae` (~380 MB). nullptr (with `error`) when
  // no HIP device is usable.
  [[nodiscard]] static std::unique_ptr<AudioVae> Create(const cpu::AudioVae& vae,
                                                        std::string* error);
  ~AudioVae() override;

  [[nodiscard]] const char* name() const noexcept override { return "hip"; }
  [[nodiscard]] std::vector<float> Encode(const float* wave, std::int64_t samples,
                                          std::int64_t* frames) override;
  void ResetDecoder() override;
  [[nodiscard]] std::vector<float> DecodeChunk(const float* z, std::int64_t frames) override;

  struct Impl;

private:
  explicit AudioVae(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::voxcpm2::hip

#endif  // GUFO_MODELS_VOXCPM2_HIP_AUDIO_VAE_HPP_
