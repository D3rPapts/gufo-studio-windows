// HIP kernels for the Ming-3.0-tts gfx1151 runtime. Matrix-vector products,
// RMSNorm and residual adds reuse the Fish Drama-3 backbone kernels; this
// file adds what Ming needs on top (QKV bias, half and interleaved RoPE,
// windowed GQA attention, GELU-tanh, the DiT input/sampler glue, linear
// upsampling and the ISTFT head). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_HIP_OPS_HPP_
#define GUFO_MODELS_MING_TTS_HIP_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::ming_tts::hip {

// y[r, c] += bias[c]
void LaunchAddBias(float* y, const float* bias, std::size_t rows,
                   std::size_t cols, hipStream_t stream);
// x[r, c] += update[r, c]
void LaunchAdd(float* x, const float* update, std::size_t elements,
               hipStream_t stream);
// y = x * rsqrt(mean(x^2) + eps) * w, one row per block.
void LaunchRmsNorm(const float* x, const float* weight, float* y,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream);
// Qwen2 rotate_half RoPE on [rows, heads, head_dim], row r at position
// first_position + r.
void LaunchRopeHalf(float* x, std::size_t rows, std::size_t heads,
                    std::size_t head_dim, std::size_t first_position,
                    float theta, hipStream_t stream);
// x_transformers interleaved RoPE on [batch * seq, heads, head_dim], row r at
// position r % seq, with explicit inverse frequencies (head_dim / 2).
void LaunchRopeInterleaved(float* x, std::size_t rows, std::size_t seq,
                           std::size_t heads, std::size_t head_dim,
                           const float* inv_freq, hipStream_t stream);
// Softmax attention. q [batch, q_len, heads, d], k/v [batch, kv_len, kv_heads,
// d] (batch strides in elements), out like q. Query i sits at position
// q_offset + i; causal masks keys j > pos, window > 0 masks j <= pos - window.
// Attended key range must not exceed 1024 positions.
void LaunchAttention(const float* q, const float* k, const float* v, float* out,
                     std::size_t batch, std::size_t q_len, std::size_t kv_len,
                     std::size_t heads, std::size_t kv_heads,
                     std::size_t head_dim, std::size_t q_batch_stride,
                     std::size_t kv_batch_stride, bool causal,
                     std::size_t q_offset, std::size_t window,
                     hipStream_t stream);
// gate = silu(gate) * up
void LaunchSiluMul(float* gate, const float* up, std::size_t elements,
                   hipStream_t stream);
void LaunchSilu(float* x, std::size_t elements, hipStream_t stream);
void LaunchGeluTanh(float* x, std::size_t elements, hipStream_t stream);
// nn.Upsample(scale_factor, mode='linear', align_corners=False) over rows.
void LaunchLinearUpsample(const float* in, float* out, std::size_t frames,
                          std::size_t cols, std::size_t factor,
                          hipStream_t stream);
// SinusPositionEmbedding(256), scale 1000, of a scalar t.
void LaunchTimestepSinus(float t, float* out, hipStream_t stream);
// DiT input [2, 3, d]: batch 0 = [t + c, x_hist, x_now], batch 1 uses the
// c_embedder bias instead of c (unconditional branch).
void LaunchAssembleDitInput(const float* t_vec, const float* c_cond,
                            const float* c_bias, const float* x_hist,
                            const float* x_now, float* out, std::size_t d,
                            hipStream_t stream);
// y += (v_c + (v_c - v_u) * cfg) * dt; v = [2, n]
void LaunchEulerStep(float* y, const float* v, float cfg, float dt,
                     std::size_t n, hipStream_t stream);
// y += scale * noise
void LaunchAxpy(float* y, const float* noise, float scale, std::size_t n,
                hipStream_t stream);
// ISTFT head: head rows [frames, n_fft + 2] (log-magnitude | phase) ->
// windowed irfft frames [frames, n_fft]. cos_sin = table of cos/sin(2 pi k / n).
void LaunchIrfftWindow(const float* head, const float* window,
                       const float* cos_sin, float* frames_out,
                       std::size_t frames, std::size_t n_fft,
                       hipStream_t stream);
// Overlap-add with "same" trimming and window-envelope normalisation:
// out [frames * hop] from frames [frames, n_fft].
void LaunchOverlapAdd(const float* frames_in, const float* window, float* out,
                      std::size_t frames, std::size_t n_fft, std::size_t hop,
                      hipStream_t stream);

}  // namespace gufo::models::ming_tts::hip
#endif

#endif  // GUFO_MODELS_MING_TTS_HIP_OPS_HPP_
