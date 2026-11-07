// HIP kernels for the IndexTTS-2.5 gfx1151 runtime (time-major [T, C]
// activations, FP32). GEMMs, LayerNorm/RMSNorm, attention, GELU/SiLU come from
// the Ming-3.0-tts / VoxCPM2 / Fish Drama-3 kernels; this file adds the glue
// IndexTTS needs: QKV split into a KV cache, adaptive RMSNorm, modulated
// LayerNorm, WaveNet gate, time-major im2col with zero/reflect padding, the
// transposed-conv gather, the fused anti-aliased SnakeBeta of BigVGAN and the
// CFG Euler update. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_HIP_OPS_HPP_
#define GUFO_MODELS_INDEXTTS_HIP_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::indextts::hip {

// qkv [rows, 3 * dim] -> q [rows, dim], k/v appended at k_out/v_out [rows, dim].
void LaunchSplitQkv(const float* qkv, float* q, float* k_out, float* v_out, std::size_t rows,
                    std::size_t dim, hipStream_t stream);

// y = (x * rsqrt(mean(x^2) + eps) * norm_w) * ss[c] + ss[dim + c] (gpt-fast
// AdaptiveLayerNorm(RMSNorm) with weight/bias = ss from project_layer(t)).
void LaunchAdaRmsNorm(const float* x, const float* norm_w, const float* ss, float* y,
                      std::size_t rows, std::size_t dim, float eps, hipStream_t stream);

// y = LayerNorm_noaffine(x, eps) * (1 + mod[dim + c]) + mod[c] (DiT FinalLayer).
void LaunchModulatedLayerNorm(const float* x, const float* mod, float* y, std::size_t rows,
                              std::size_t dim, float eps, hipStream_t stream);

// acts[r, c] = tanh(a[r, c] + g[c]) * sigmoid(a[r, dim + c] + g[dim + c]); a [rows, 2 dim].
void LaunchWaveNetGate(const float* a, const float* g, float* acts, std::size_t rows,
                       std::size_t dim, hipStream_t stream);

enum class PadMode : int { kZero = 0, kReflect = 1, kReplicate = 2 };
// col[t, j * cin + c] = x[pad(t + j * dilation - pad_left), c], t < t_out.
void LaunchIm2colTM(const float* x, float* col, std::size_t t_in, std::size_t t_out,
                    std::size_t cin, std::size_t kernel, std::size_t dilation,
                    std::ptrdiff_t pad_left, PadMode mode, hipStream_t stream);

// Transposed conv from z = x * W_all^T with W_all [kernel * cout, cin]:
// y[n, co] = bias[co] + sum_{i, j : i * stride + j - pad == n} z[i, j * cout + co].
void LaunchConvTransposeGather(const float* z, const float* bias, float* y, std::size_t t_in,
                               std::size_t t_out, std::size_t cout, std::size_t kernel,
                               std::size_t stride, std::size_t pad, hipStream_t stream);

// BigVGAN Activation1d(SnakeBeta) on x [t, c] -> y [t, c]: 2x Kaiser-sinc
// upsample (12 taps, replicate pad), x + inv_beta * sin(alpha x)^2, 12-tap
// lowpass + 2x downsample (replicate pad 5/6).
void LaunchAliasFreeSnake(const float* x, float* y, const float* alpha, const float* inv_beta,
                          const float* up_filter, const float* down_filter, std::size_t t,
                          std::size_t channels, hipStream_t stream);

// x = x + dt * ((1 + cfg) * vc - cfg * vu); rows < prompt_rows are zeroed.
void LaunchCfgEulerTM(float* x, const float* vc, const float* vu, float cfg, float dt,
                      std::size_t rows, std::size_t cols, std::size_t prompt_rows,
                      hipStream_t stream);

// y[i] = a[i] * scale (e.g. sum of three resblocks / 3).
void LaunchScale(float* y, const float* a, float scale, std::size_t n, hipStream_t stream);
// y[i] = clamp(y[i], lo, hi)
void LaunchClamp(float* y, float lo, float hi, std::size_t n, hipStream_t stream);
// dst[r, 0:a_cols] = a[r], dst[r, a_cols:] = b[r]
void LaunchConcatCols(const float* a, const float* b, float* dst, std::size_t rows,
                      std::size_t a_cols, std::size_t b_cols, hipStream_t stream);
// y = bf16(x), round to nearest even (GEMM operands of the hipBLASLt DiT path).
void LaunchF32ToBf16(const float* x, std::uint16_t* y, std::size_t n, hipStream_t stream);
// In-place FP32 softmax of each row of s [rows, cols] (attention scores).
void LaunchSoftmaxRows(float* s, std::size_t rows, std::size_t cols, hipStream_t stream);

}  // namespace gufo::models::indextts::hip
#endif

#endif  // GUFO_MODELS_INDEXTTS_HIP_OPS_HPP_
