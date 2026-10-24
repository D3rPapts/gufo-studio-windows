#ifndef GUFO_MODELS_QWEN3_OMNI_REFERENCE_HPP_
#define GUFO_MODELS_QWEN3_OMNI_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/models/qwen3_omni/model.hpp"

/// Straightforward F32 CPU implementation of the captioner. It reads the
/// mapped GGUF weights directly (K-quants are dequantized row by row) and
/// follows the Hugging Face modeling code operation for operation. It is the
/// numerical yardstick for the HIP runtime, not a serving path.
namespace gufo::models::qwen3_omni {

class ThreadPool;

/// The token embedding row for `token` [hidden], dequantized on the CPU.
void EmbedToken(const Model& model, std::int32_t token, float* out);

class CpuReference {
public:
  CpuReference(const Model& model, unsigned threads);
  ~CpuReference();

  CpuReference(const CpuReference&) = delete;
  CpuReference& operator=(const CpuReference&) = delete;

  /// AuT tower over mel-major log-mel features [mel_bins, frames]. Returns
  /// [tokens, output] F32 embeddings; tokens == AudioTokenCount(frames).
  [[nodiscard]] std::vector<float> EncodeAudio(std::span<const float> log_mel,
                                               std::size_t frames);

  /// Clears the KV cache.
  void Reset();

  /// Runs `count` input embeddings [count, hidden] at the next positions and
  /// writes the last row's logits [vocab].
  void Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* logits);

  /// The token embedding row for `token` [hidden].
  void Embed(std::int32_t token, float* out) const;

  [[nodiscard]] std::size_t position() const noexcept { return position_; }

private:
  void Attention(std::size_t layer, const float* x, std::size_t count,
                 float* out);
  void Moe(std::size_t layer, const float* x, std::size_t count, float* out);

  const Model& model_;
  std::unique_ptr<ThreadPool> pool_;
  std::size_t position_{0};
  // Per layer: [position][kv_heads * head_dim].
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
};

}  // namespace gufo::models::qwen3_omni

#endif  // GUFO_MODELS_QWEN3_OMNI_REFERENCE_HPP_
