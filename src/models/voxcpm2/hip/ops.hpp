// HIP kernels for the VoxCPM2 gfx1151 runtime. GEMM/GEMV, RMSNorm, bias,
// residual add and SiLU come from the Ming-3.0-tts / Fish Drama-3 kernels;
// this file adds what VoxCPM2 needs on top: table RoPE (LongRoPE), attention
// without a key-length limit (online softmax in 1024-key chunks), FSQ, the
// CFG-zero-star Euler step, local-encoder / DiT input assembly and row
// selection by the audio mask. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_HIP_OPS_HPP_
#define GUFO_MODELS_VOXCPM2_HIP_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::voxcpm2::hip {

// rotate_half RoPE with precomputed cos/sin tables [positions, head_dim / 2]
// on x [rows, heads, head_dim]; row r sits at position first + (r % period).
void LaunchRopeTable(float* x, std::size_t rows, std::size_t period, std::size_t heads,
                     std::size_t head_dim, std::size_t first, const float* cos_table,
                     const float* sin_table, hipStream_t stream);

// Softmax attention, any key length. q [batch, q_len, heads, d], k/v
// [batch, kv_len, kv_heads, d] (batch strides in elements), out like q.
// Query i sits at position q_offset + i; causal masks keys j > position.
// head_dim must be <= 256.
void LaunchAttention(const float* q, const float* k, const float* v, float* out,
                     std::size_t batch, std::size_t q_len, std::size_t kv_len,
                     std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
                     std::size_t q_batch_stride, std::size_t kv_batch_stride, bool causal,
                     std::size_t q_offset, hipStream_t stream);

// x = round(tanh(x) * scale) / scale (round half to even)
void LaunchFsqQuantize(float* x, std::size_t elements, float scale, hipStream_t stream);

// Local encoder input [count, patch + 1, dim]: special token then the
// projected patch rows proj [count * patch, dim].
void LaunchAssembleLocEnc(const float* special, const float* proj, float* out,
                          std::size_t count, std::size_t patch, std::size_t dim,
                          hipStream_t stream);

// dst[i] = src[i * stride_rows] rows of `dim` floats (i < count).
void LaunchGatherRows(const float* src, float* dst, std::size_t count, std::size_t stride_rows,
                      std::size_t dim, hipStream_t stream);

// dst[r] = mask[r] ? alt[r] : dst[r] for rows of `dim` floats.
void LaunchSelectRows(float* dst, const float* alt, const std::uint8_t* mask, std::size_t rows,
                      std::size_t dim, hipStream_t stream);

// out[r] = [a[r], mask[r] ? b[r] : 0] (rows of 2 * dim); mask == nullptr: all ones.
void LaunchConcatRows(const float* a, const float* b, const std::uint8_t* mask, float* out,
                      std::size_t rows, std::size_t dim, hipStream_t stream);

// DiT input [2, mu_tokens + 1 + 2 * patch, dim]: batch 0 = [mu, t, cond, x],
// batch 1 = [0, t, cond, x] (CFG unconditional branch).
void LaunchAssembleDit(const float* mu, const float* t_row, const float* cond_rows,
                       const float* x_rows, float* out, std::size_t mu_tokens,
                       std::size_t patch, std::size_t dim, hipStream_t stream);

// One UnifiedCFM Euler step with CFG zero-star on n values (single block):
// vel = [pos (n), neg (n)]; st = <pos, neg> / (|neg|^2 + 1e-8) (or 1);
// x -= dt * (neg * st + cfg * (pos - neg * st)).
void LaunchCfgEuler(float* x, const float* vel, float cfg, float dt, std::size_t n,
                    bool zero_star, hipStream_t stream);

}  // namespace gufo::models::voxcpm2::hip
#endif

#endif  // GUFO_MODELS_VOXCPM2_HIP_OPS_HPP_
