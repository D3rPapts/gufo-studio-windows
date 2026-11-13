// MMAudio for gufo: element-wise, normalization, convolution and attention
// kernels. Activations are FP32 and channel-last: row-major [rows, width]
// (one row per token / time step).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_KERNELS_HPP_
#define GUFO_MODELS_MMAUDIO_KERNELS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::mmaudio {

// ---- conversions -----------------------------------------------------------

/// out[i] = half(clamp(in[i], +-65504)).
void LaunchToHalf(const float* in, std::size_t count, std::uint16_t* out,
                  hipStream_t stream);
/// out[r, c] (+)= float(in[r, c]) + (bias ? bias[c] : 0).
void LaunchWiden(const std::uint16_t* in, std::size_t rows, int width,
                 const float* bias, bool accumulate, float* out,
                 hipStream_t stream);
/// out[r, c] (+)= in[r, c] + (bias ? bias[c] : 0) (FP32 product of a GEMM
/// that wrote to scratch).
void LaunchAddBias(const float* in, std::size_t rows, int width,
                   const float* bias, bool accumulate, float* out,
                   hipStream_t stream);

// ---- softmax / normalization -------------------------------------------------

/// In-place softmax(scale * row) of `rows` rows of `columns`. With `causal`
/// (> 0), row r masks the columns after (r % causal).
void LaunchSoftmax(float* scores, std::size_t rows, int columns, float scale,
                   int causal, hipStream_t stream);

/// LayerNorm over `width` (two-pass FP32 statistics). `weight` / `bias` may
/// be null (no affine). When `scale` / `shift` are given the result becomes
/// y * (1 + scale[r * mod_stride + c]) + shift[r * mod_stride + c]
/// (mod_stride 0 = one vector for all rows).
void LaunchLayerNorm(const float* x, std::size_t rows, int width,
                     const float* weight, const float* bias, float eps,
                     const float* shift, const float* scale, int mod_stride,
                     float* out, hipStream_t stream);

/// x[r, :] /= max(||x[r, :]||, eps) (torch F.normalize, eps 1e-12).
void LaunchL2Normalize(float* x, std::size_t rows, int width,
                       hipStream_t stream);

/// EDM2 pixel norm over channels: out = x / (1e-4 + ||x|| / sqrt(width)).
void LaunchPixelNorm(const float* x, std::size_t rows, int width, float* out,
                     hipStream_t stream);

// ---- element-wise ------------------------------------------------------------

/// x = silu(x) * scale (scale 1 / 0.596 is EDM2's magnitude-preserving SiLU).
void LaunchSilu(float* x, std::size_t count, float scale, hipStream_t stream);
void LaunchSiluTo(const float* x, std::size_t count, float* out,
                  hipStream_t stream);
/// Exact (erf) GELU in place.
void LaunchGelu(float* x, std::size_t count, hipStream_t stream);
/// x * sigmoid(1.702 x) in place (OpenCLIP QuickGELU).
void LaunchQuickGelu(float* x, std::size_t count, hipStream_t stream);
/// out[r, f] = silu(x[r, f]) * x[r, hidden + f] for x [rows, 2 * hidden].
void LaunchSwiGlu(const float* x, std::size_t rows, int hidden, float* out,
                  hipStream_t stream);
/// x[r, c] += y[r, c] * (gate ? gate[r * gate_stride + c] : 1).
void LaunchAddGated(float* x, const float* y, std::size_t rows, int width,
                    const float* gate, int gate_stride, hipStream_t stream);
/// out = (a * (1 - t) + b * t) / sqrt((1 - t)^2 + t^2) (EDM2 mp_sum).
void LaunchMpSum(const float* a, const float* b, std::size_t count, float t,
                 float* out, hipStream_t stream);
void LaunchClamp(float* x, std::size_t count, float low, float high,
                 hipStream_t stream);
/// x[i] = x[i] * scale + add.
void LaunchAffine(float* x, std::size_t count, float scale, float add,
                  hipStream_t stream);
/// x[r, c] = x[r, c] * mul[c] + add[c] (either may be null).
void LaunchColumnAffine(float* x, std::size_t rows, int width,
                        const float* mul, const float* add,
                        hipStream_t stream);
/// x[i] += y[i].
void LaunchAdd(float* x, const float* y, std::size_t count,
               hipStream_t stream);
/// x[r, c] += v[c].
void LaunchAddRow(float* x, const float* v, std::size_t rows, int width,
                  hipStream_t stream);
/// out[r, c] = in[r % period, c] (repeats `period` rows).
void LaunchRepeatRows(const float* in, int period, std::size_t rows, int width,
                      float* out, hipStream_t stream);
/// out[c] = mean over rows of x[r, c].
void LaunchColumnMean(const float* x, std::size_t rows, int width, float* out,
                      hipStream_t stream);
/// out[r, :] = table[ids[r], :] (FP32 table).
void LaunchGatherRows(const float* table, const int* ids, std::size_t rows,
                      int width, float* out, hipStream_t stream);

// ---- 1-D convolution helpers (channel-last) ---------------------------------

/// out[t, c * kernel + k] = x[t + k * dilation - padding, c] (0 outside).
/// Combined with a GEMM against W[out, in * kernel] this is torch's conv1d
/// (stride 1) with `rows_out` = rows_in + 2 * padding - dilation * (k - 1).
void LaunchIm2Col(const float* x, int rows_in, int channels, int kernel,
                  int dilation, int padding, int rows_out, float* out,
                  hipStream_t stream);
/// Overlap-add of a transposed convolution: y [rows_in, channels * kernel]
/// holds x W for W [in, out, k] -> out[s, co] = bias[co] + sum over (t, k)
/// with t * stride - padding + k == s of y[t, co * kernel + k].
void LaunchCol2ImTransposed(const float* y, int rows_in, int channels,
                            int kernel, int stride, int padding,
                            const float* bias, int rows_out, float* out,
                            hipStream_t stream);
/// out[i, c] = x[i / factor, c] (nearest, also torch's nearest-exact for an
/// integer factor).
void LaunchUpsampleNearest(const float* x, int rows, int width, int factor,
                           float* out, hipStream_t stream);
/// torch F.interpolate(mode="nearest-exact") along rows:
/// out[i] = x[min(floor((i + 0.5) * rows_in / rows_out), rows_in - 1)].
void LaunchInterpolateNearestExact(const float* x, int rows_in, int width,
                                   int rows_out, float* out,
                                   hipStream_t stream);

/// BigVGAN v2 anti-aliased SnakeBeta (Activation1d with 2x up / down,
/// 12-tap Kaiser-sinc `filter`, replicate padding):
/// u = upsample(x); a = u + inv_beta[c] * sin(alpha[c] * u)^2;
/// out = lowpass-downsample(a). `alpha` already exp()'d, `inv_beta` =
/// 1 / (exp(beta) + 1e-9).
void LaunchSnakeAntiAlias(const float* x, int rows, int channels,
                          const float* alpha, const float* inv_beta,
                          const float* filter, float* out, hipStream_t stream);

// ---- patch embedding inputs ----------------------------------------------------

/// Image patches: pixels [images][3][size][size] -> out
/// [images][grid * grid][3 * patch * patch] (column (c * patch + py) * patch
/// + px, torch Conv2d weight order); pixels past grid * patch are ignored.
void LaunchPatchify2d(const float* pixels, int images, int size, int patch,
                      int grid, float* out, hipStream_t stream);
/// Video tubelets for the Synchformer: frames [frames][3][224][224]; segment
/// s covers frames s * step .. s * step + 15; out
/// [segments][8 * 14 * 14][3 * 2 * 16 * 16] in Conv3d weight order
/// ((c * 2 + dt) * 16 + py) * 16 + px, token (t * 14 + gy) * 14 + gx.
void LaunchPatchify3d(const float* frames, int segments, int step, float* out,
                      hipStream_t stream);

// ---- attention helpers ----------------------------------------------------------

/// MMAudio SelfAttention.pre_attention: splits qkv rows [rows, 3 * heads *
/// head_dim] laid out "(h d j)" (j fastest) into q / k / v rows of
/// heads * head_dim (written at row stride `out_stride`), RMS-normalizes q
/// and k per head with `q_norm` / `k_norm` (eps 1.1920929e-07, torch's FP32
/// default), and rotates q and k with `rope` [rows][head_dim / 2][2] (cos,
/// sin) when not null (interleaved pairs 2i, 2i + 1).
void LaunchSplitQkv(const float* qkv, int rows, int heads, int head_dim,
                    const float* q_norm, const float* k_norm,
                    const float* rope, float* q, float* k, float* v,
                    int out_stride, hipStream_t stream);

/// VAE AttnBlock1D: qkv [rows, 3 * channels] with channel index c * 3 + j
/// -> q / k / v [rows, channels], each normalized over channels per row
/// (EDM2 normalize, eps 1e-4).
void LaunchVaeQkv(const float* qkv, int rows, int channels, float* q,
                  float* k, float* v, hipStream_t stream);

enum class KeySet { kFull, kTime, kSpace };

/// Head-dim-64 attention over key subsets (Synchformer divided space-time
/// attention). Rows come in groups of `group` (= 1 + frames * patches for
/// kTime / kSpace); within a group, row 0 is the CLS token and row
/// 1 + f * patches + n is patch n of frame f. kFull: every row of the group;
/// kTime: CLS plus the same patch in every frame; kSpace: CLS plus every
/// patch of the same frame (CLS always sees the whole group). q/k/v rows of
/// `stride` floats with heads of 64 columns; out rows of heads * 64.
void LaunchSubsetAttention(const float* q, const float* k, const float* v,
                           int stride, float* out, int rows, int heads,
                           int group, int frames, int patches, KeySet keys,
                           float scale, hipStream_t stream);

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_KERNELS_HPP_
