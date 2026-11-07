// Library GEMMs for the IndexTTS-2.5 DiT/CFM on gfx1151 (model-private, after
// the EmbeddingGemma 2 / Qwen3-ASR wrappers): hipBLASLt BF16 x BF16 -> FP32
// projections with the algorithm chosen by measurement, and FP32 strided
// batched SGEMMs for the attention scores and values (head dimension 64, no
// library attention kernel in the fork covers it). Written with AI assistance
// (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_HIP_BLAS_HPP_
#define GUFO_MODELS_INDEXTTS_HIP_BLAS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::indextts::hip {

class Blas {
public:
  explicit Blas(hipStream_t stream);
  ~Blas();
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;

  /// Rows the projection GEMM computes for `rows` input rows (rounded up to
  /// 128 so that nearby lengths share one tuned plan). Buffers behind the BF16
  /// input and the FP32 output must hold PaddedRows(rows) rows; the extra rows
  /// are computed from whatever they contain and ignored.
  [[nodiscard]] static std::size_t PaddedRows(std::size_t rows);

  /// y[rows, m] = x[rows, k] * w[m, k]^T (+ y when `accumulate`), BF16
  /// operands, FP32 accumulation and output. hipBLASLt with a measured
  /// algorithm; hipBLAS GemmEx when hipBLASLt has none for the shape.
  void Gemm(const std::uint16_t* w, const std::uint16_t* x, float* y, std::size_t rows,
            std::size_t m, std::size_t k, bool accumulate);

  /// Non-causal attention of one sequence, FP32: q/k/v/out [rows, heads * 64]
  /// row-major, scores scratch [heads, rows, rows]. Scale 1/8.
  void Attention64(const float* q, const float* k, const float* v, float* out, float* scores,
                   std::size_t rows, std::size_t heads);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::indextts::hip
#endif

#endif  // GUFO_MODELS_INDEXTTS_HIP_BLAS_HPP_
