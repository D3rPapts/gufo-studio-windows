// Execution backend for the Ming-3.0-tts AR loop and VAE decoder. The CPU
// reference runtime and the gfx1151 HIP runtime implement it; the text front
// end, CAM++ and the VAE encoder always run on the CPU.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_BACKEND_HPP_
#define GUFO_MODELS_MING_TTS_BACKEND_HPP_

#include <cstdint>
#include <vector>

namespace gufo::models::ming_tts {

// Per-step sampler options (FlowLoss.sample / CFM.sample defaults).
struct FlowOptions {
  float cfg{2.0F};
  float sigma{0.25F};
  float temperature{0.0F};
  std::int32_t steps{10};
};

class Backend {
public:
  virtual ~Backend() = default;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
  // Clears the backbone KV cache (start of a fragment).
  virtual void ResetBackbone() = 0;
  // Appends `rows` input embeddings [rows, hidden] to the backbone and writes
  // the final RMS-normed hidden states [rows, hidden] to `normed`.
  virtual void RunBackbone(const float* embeddings, std::int64_t rows,
                           float* normed) = 0;
  // softmax(stop_head(z))[1]
  [[nodiscard]] virtual float StopAt(const float* z) = 0;
  // DiT flow-matching sampler: noise [latent] -> latent [latent].
  virtual void SampleFlow(const float* cond, const float* history,
                          const float* noise, const float* sde_noise,
                          const FlowOptions& options, float* latent) = 0;
  // Aggregator: latent [latent] -> next backbone input [hidden].
  virtual void AggregateLatent(const float* latent, float* out) = 0;
  // VAE decoder: [frames, latent] -> 44.1 kHz waveform.
  [[nodiscard]] virtual std::vector<float> DecodeAudio(const float* latents,
                                                       std::int64_t frames) = 0;
};

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_BACKEND_HPP_
