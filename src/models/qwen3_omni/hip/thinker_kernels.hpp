#ifndef GUFO_MODELS_QWEN3_OMNI_HIP_THINKER_KERNELS_HPP_
#define GUFO_MODELS_QWEN3_OMNI_HIP_THINKER_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen3_omni::hip {

/// out[r] = RMSNorm(x[r]) * gamma over `cols`; one block per row.
void LaunchRmsNorm(const float* x, const float* gamma, float* out,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream);

/// Per-head RMSNorm (q_norm / k_norm) and rotate-half RoPE at positions
/// position0 + token, in place. q is [tokens][heads][head_dim], k is
/// [tokens][kv_heads][head_dim]; head_dim <= 256 and even.
void LaunchQkNormRope(float* q, float* k, const float* q_gamma,
                      const float* k_gamma, std::size_t tokens,
                      std::size_t heads, std::size_t kv_heads,
                      std::size_t head_dim, std::size_t position0, float theta,
                      float eps, hipStream_t stream);

/// Appends k/v rows [tokens][kv_dim] to F16 caches [capacity][kv_dim] at
/// rows position0.. .
void LaunchStoreKv(const float* k, const float* v, __half* k_cache,
                   __half* v_cache, std::size_t tokens, std::size_t kv_dim,
                   std::size_t position0, hipStream_t stream);

/// Causal GQA attention of `tokens` queries at positions position0 + t over
/// the cache rows [0, position0 + t]. out is [tokens][heads][head_dim].
/// Requires head_dim == 128 and position0 + tokens <= MaxAttentionContext().
void LaunchAttention(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out, std::size_t tokens,
                     std::size_t heads, std::size_t kv_heads,
                     std::size_t head_dim, std::size_t position0,
                     hipStream_t stream);
[[nodiscard]] std::size_t MaxAttentionContext();

/// Router: logits = W[experts][hidden] . x[t] (into `logits`, [tokens]
/// [experts] scratch), softmax, top-`used` (ties to the lower expert id),
/// weights renormalized over the selection. ids/weights are [tokens][used].
/// experts <= 256, used <= 16.
void LaunchRoute(const float* x, const float* router, float* logits,
                 std::int32_t* ids, float* weights, std::size_t tokens,
                 std::size_t hidden, std::size_t experts, std::size_t used,
                 hipStream_t stream);

/// out[i] = silu(gate[i]) * up[i].
void LaunchSwiglu(const float* gate, const float* up, float* out,
                  std::size_t count, hipStream_t stream);

/// residual[t] += sum_s weights[t][s] * rows[t * used + s]  (rows are
/// [tokens * used][hidden]).
void LaunchMoeCombine(float* residual, const float* rows, const float* weights,
                      std::size_t tokens, std::size_t used, std::size_t hidden,
                      hipStream_t stream);

/// dst[r][c] = src[r][begin + c] for c < width; src rows are `src_width`
/// floats.
void LaunchCopyColumns(const float* src, std::size_t src_width,
                       std::size_t begin, std::size_t width, float* dst,
                       std::size_t rows, hipStream_t stream);

/// residual[i] += update[i].
void LaunchAdd(float* residual, const float* update, std::size_t count,
               hipStream_t stream);

}  // namespace gufo::models::qwen3_omni::hip

#endif  // GUFO_MODELS_QWEN3_OMNI_HIP_THINKER_KERNELS_HPP_
