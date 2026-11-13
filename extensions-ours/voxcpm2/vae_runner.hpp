// Where the VoxCPM2 AudioVAE runs: the CPU reference (cpu::AudioVae) or the
// gfx1151 HIP port (hip::AudioVae). One streaming decoder state per runner.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_VAE_RUNNER_HPP_
#define GUFO_MODELS_VOXCPM2_VAE_RUNNER_HPP_

#include <cstdint>
#include <vector>

namespace gufo::models::voxcpm2 {

class VaeRunner {
public:
  virtual ~VaeRunner() = default;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
  // 16 kHz mono wave (multiple of the hop) -> mu [latent, frames].
  [[nodiscard]] virtual std::vector<float> Encode(const float* wave, std::int64_t samples,
                                                  std::int64_t* frames) = 0;
  // Starts a new utterance (zero left context in every causal conv).
  virtual void ResetDecoder() = 0;
  // z [latent, frames] (channel-major) -> frames * 1920 samples at 48 kHz,
  // continuing the causal state of earlier chunks.
  [[nodiscard]] virtual std::vector<float> DecodeChunk(const float* z, std::int64_t frames) = 0;
};

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_VAE_RUNNER_HPP_
