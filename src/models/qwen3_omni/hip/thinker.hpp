#ifndef GUFO_MODELS_QWEN3_OMNI_HIP_THINKER_HPP_
#define GUFO_MODELS_QWEN3_OMNI_HIP_THINKER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen3_omni/model.hpp"

namespace gufo::models::qwen3_omni::hip {

/// The Qwen3-Omni thinker (48-layer Qwen3-MoE) on the GPU. Weights stay in
/// their GGUF K-quant encoding and run through the vendored llama.cpp
/// MMQ/MMVQ kernels; the KV cache is F16.
class Thinker {
public:
  static std::unique_ptr<Thinker> Create(const Model& model,
                                         std::size_t max_context,
                                         hipStream_t stream,
                                         std::string* error);
  ~Thinker();

  Thinker(const Thinker&) = delete;
  Thinker& operator=(const Thinker&) = delete;

  void Reset();

  /// Runs `count` host embeddings [count, hidden] at the next positions and
  /// copies the last row's logits [vocab] to `logits`.
  bool Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* logits, std::string* error);

  [[nodiscard]] std::size_t position() const;
  [[nodiscard]] std::size_t max_context() const;
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit Thinker(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_omni::hip

#endif  // GUFO_MODELS_QWEN3_OMNI_HIP_THINKER_HPP_
