// FP32 CPU kernels for the Ming-3.0-tts reference runtime (BF16 weights are
// widened on the fly). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_CPU_OPS_HPP_
#define GUFO_MODELS_MING_TTS_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::ming_tts::cpu {

// Row-major [out_features, in_features] BF16 weight borrowed from the mapped
// checkpoint, with an optional FP32 bias.
struct Linear {
  const std::uint16_t* weight{nullptr};
  std::int64_t out_features{0};
  std::int64_t in_features{0};
  std::vector<float> bias;  // empty: no bias
};

// Diagnostics: round the activations of products with more than 8 rows to
// BF16 first, emulating the rocBLAS BF16 path of the GPU runtime.
void SetBf16ActivationEmulation(bool enabled);

// y[m, out] = x[m, in] * W^T + b
void LinearForward(const Linear& layer, const float* x, std::int64_t rows,
                   float* y);

// In-place RMSNorm over the last dimension: x * rsqrt(mean(x^2) + eps) * w.
void RmsNorm(float* x, std::int64_t rows, std::int64_t dim,
             std::span<const float> weight, float eps);
// out = RmsNorm(x) without touching x.
void RmsNormTo(const float* x, float* out, std::int64_t rows, std::int64_t dim,
               std::span<const float> weight, float eps);

void SiluMulInPlace(float* gate, const float* up, std::size_t count);
void GeluTanhInPlace(float* x, std::size_t count);

// Scaled-dot-product attention for one sequence.
// q: [q_len, heads, head_dim], k/v: [kv_len, kv_heads, head_dim],
// out: [q_len, heads, head_dim]. Query i sits at absolute position
// q_offset + i; keys at 0..kv_len-1. causal masks j > pos;
// window > 0 additionally masks j <= pos - window.
void Attention(const float* q, const float* k, const float* v, float* out,
               std::int64_t q_len, std::int64_t kv_len, std::int64_t heads,
               std::int64_t kv_heads, std::int64_t head_dim, bool causal,
               std::int64_t q_offset, std::int64_t window);

// Qwen2 (rotate_half) RoPE on [len, heads, head_dim] for positions
// first_position.. .
void RopeHalf(float* x, std::int64_t len, std::int64_t heads,
              std::int64_t head_dim, std::int64_t first_position, float theta);

// GPT-J style interleaved RoPE (x_transformers) with explicit inverse
// frequencies (head_dim / 2 values), positions 0..len-1.
void RopeInterleaved(float* x, std::int64_t len, std::int64_t heads,
                     std::int64_t head_dim, std::span<const float> inv_freq);

}  // namespace gufo::models::ming_tts::cpu

#endif  // GUFO_MODELS_MING_TTS_CPU_OPS_HPP_
