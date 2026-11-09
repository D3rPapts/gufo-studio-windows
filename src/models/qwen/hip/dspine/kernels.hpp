#ifndef GUFO_MODELS_QWEN_HIP_DSPINE_KERNELS_HPP_
#define GUFO_MODELS_QWEN_HIP_DSPINE_KERNELS_HPP_

// Device operators of the DSpine draft (see src/models/qwen/dspine_reference.*
// for the CPU oracle they mirror). Activations live in float buffers but hold
// BF16-exact values wherever dspine_mlx stores bf16; "Bf" in a name means the
// result is rounded to the nearest bf16 (ties to even). Tables and the draft
// K/V cache are stored as raw bf16 bits (std::uint16_t).

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::dspine {

/// x[i] = bf(x[i]).
void LaunchRoundBf16(float* x, std::size_t n, hipStream_t stream);

/// y[r] = bf(x[r] * rsqrt(mean(x[r]^2) + eps) * w), one bf16 rounding
/// (mx.fast.rms_norm). x and y may alias.
void LaunchRmsNormBf(const float* x, const float* w, float* y,
                     std::uint32_t rows, std::uint32_t dim, float eps,
                     hipStream_t stream);

/// Per (row, head): RMSNorm over head_dim with `norm_w` (bf), then
/// non-traditional RoPE (halves) at position `first_position + row` (bf).
void LaunchHeadNormRopeBf(float* x, const float* norm_w, std::uint32_t rows,
                          std::uint32_t heads, std::uint32_t head_dim,
                          std::uint32_t first_position, float theta, float eps,
                          hipStream_t stream);

/// cache[(first + r) * width + i] = bf16 bits of x[r * width + i].
void LaunchStoreBf16Rows(const float* x, std::uint16_t* cache,
                         std::uint32_t rows, std::uint32_t width,
                         std::uint32_t first, hipStream_t stream);

/// DSpine draft attention: `block` query rows at positions context..context+T-1
/// see all cached context rows inside the sliding window plus the block rows
/// up to their own (causal inside the block). out = bf(softmax(qk/sqrt(d)) v).
/// `scores` holds block * heads * (context + block) floats.
void LaunchBlockAttentionBf(const float* q, const std::uint16_t* cache_k,
                            const std::uint16_t* cache_v, const float* block_k,
                            const float* block_v, float* out, float* scores,
                            std::uint32_t context, std::uint32_t block,
                            std::uint32_t sliding_window, std::uint32_t heads,
                            std::uint32_t kv_heads, std::uint32_t head_dim,
                            hipStream_t stream);

/// out = bf(a + b).
void LaunchAddBf(const float* a, const float* b, float* out, std::size_t n,
                 hipStream_t stream);

/// gate = bf(silu(gate) * up).
void LaunchSwiGluBf(float* gate, const float* up, std::size_t n,
                    hipStream_t stream);

/// Bridge read side: rms[t] = sqrt(max(mean(f^2), FLT_MIN)),
/// normalized[t] = bf(f / (rms + eps)).
void LaunchBridgeNormalize(const float* f, float* rms, float* normalized,
                           std::uint32_t rows, std::uint32_t dim, float eps,
                           hipStream_t stream);

/// codes[t] = bf(sqrt(C) * p / (||p|| + eps)) for every row given.
void LaunchBridgeCodes(const float* projected, float* codes, std::uint32_t rows,
                       std::uint32_t code_size, float eps, hipStream_t stream);

/// packed row 0 = the precomputed anchor row (bf16 table row of 3R).
void LaunchCopyBf16Row(const std::uint16_t* table, std::uint32_t row,
                       std::uint32_t width, float* out, hipStream_t stream);

/// message[0] = 0; message[t] = bf(sigmoid(q[t] + s[t-1] + bias) * v[t-1])
/// with packed rows [q | s | v] of 3R.
void LaunchBridgeMessage(const float* packed, const float* bias,
                         float* message, std::uint32_t rows, std::uint32_t rank,
                         hipStream_t stream);

/// out_w[t] = t == 0 ? f : bf(f + rms[t] * bf(delta));
/// x_in = bf(bf(out_w * rsqrt(mean(out_w^2) + eps)) * next_w).
void LaunchBridgeWriteNorm(const float* f, const float* rms, const float* delta,
                           const float* next_w, float* out_w, float* x_in,
                           std::uint32_t rows, std::uint32_t dim, float eps,
                           hipStream_t stream);

/// y = x * w (no rounding): the base term of the refinement score.
void LaunchMulRows(const float* x, const float* w, float* y, std::uint32_t rows,
                   std::uint32_t dim, hipStream_t stream);

/// Table helpers: rows of bf16 bits.
/// out = bf16(bf(in) * w) per column (head rows times the final norm).
void LaunchScaleBf16Rows(const std::uint16_t* in, const float* w,
                         std::uint16_t* out, std::size_t rows,
                         std::uint32_t dim, hipStream_t stream);
/// out = bf16(sqrt(C) * in) (codebook rows to codes).
void LaunchScaleBf16(const std::uint16_t* in, float scale, std::uint16_t* out,
                     std::size_t n, hipStream_t stream);
/// out = bf16 bits of x.
void LaunchFloatToBf16Bits(const float* x, std::uint16_t* out, std::size_t n,
                           hipStream_t stream);

/// Top-k per logit row; ids ascending (ties: lower id first when ranking).
void LaunchTopKAscending(const float* logits, std::uint32_t rows,
                         std::uint32_t vocab, std::uint32_t k,
                         std::int32_t* ids, float* values, hipStream_t stream);

/// Greedy refinement: for row i (block row i + 1) and sender s, the receiver
/// j maximizing base[i][cand_j] + rms[i+1] * (m . tg[cand_j]) where
/// m = sigmoid(q[i+1] + pred[sid] + bias) * prev[sid] and
/// sid = i == 0 ? anchor : cand[i-1][s]. Writes idx[i * k + s].
void LaunchRefineScores(const float* packed, const float* rms,
                        const float* base_logits, const std::int32_t* cand,
                        const std::uint16_t* tg, const std::uint16_t* tsp,
                        const float* bias, std::uint32_t anchor,
                        std::uint32_t rows, std::uint32_t k, std::uint32_t rank,
                        std::uint32_t vocab, std::int32_t* idx,
                        hipStream_t stream);

}  // namespace gufo::hip::dspine

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DSPINE_KERNELS_HPP_
