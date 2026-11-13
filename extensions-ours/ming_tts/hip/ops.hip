// Written with AI assistance (Claude Code).
#include "ming_tts/hip/ops.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

namespace gufo::models::ming_tts::hip {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kMaxKeys = 1024;

unsigned Blocks(std::size_t elements) {
  return static_cast<unsigned>((elements + kThreads - 1) / kThreads);
}

__device__ float BlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

__device__ float BlockMax(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
    }
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

constexpr int kTile = 64;
constexpr int kTileK = 16;

__device__ inline float Bf16ToFloat(std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned>(bits) << 16U);
}

// 256 threads, each computes a 4x4 block of the 64x64 output tile.
__global__ void GemmF32Bf16Kernel(const float* a, const std::uint16_t* w, float* c,
                                  std::size_t m, std::size_t n, std::size_t k) {
  __shared__ float as[kTileK][kTile + 4];
  __shared__ float ws[kTileK][kTile + 4];
  const unsigned tx = threadIdx.x % 16;
  const unsigned ty = threadIdx.x / 16;
  const std::size_t m0 = static_cast<std::size_t>(blockIdx.y) * kTile;
  const std::size_t n0 = static_cast<std::size_t>(blockIdx.x) * kTile;
  float acc[4][4] = {};
  for (std::size_t k0 = 0; k0 < k; k0 += kTileK) {
    for (unsigned i = threadIdx.x; i < kTile * kTileK; i += blockDim.x) {
      const unsigned row = i / kTileK;
      const unsigned col = i % kTileK;
      const std::size_t gk = k0 + col;
      const std::size_t gm = m0 + row;
      const std::size_t gn = n0 + row;
      as[col][row] = (gm < m && gk < k) ? a[gm * k + gk] : 0.0F;
      ws[col][row] = (gn < n && gk < k) ? Bf16ToFloat(w[gn * k + gk]) : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      float av[4];
      float wv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        av[i] = as[kk][ty * 4 + i];
        wv[i] = ws[kk][tx * 4 + i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = fmaf(av[i], wv[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    const std::size_t gm = m0 + ty * 4 + i;
    if (gm >= m) {
      continue;
    }
    for (int j = 0; j < 4; ++j) {
      const std::size_t gn = n0 + tx * 4 + j;
      if (gn < n) {
        c[gm * n + gn] = acc[i][j];
      }
    }
  }
}

__global__ void AddBiasKernel(float* y, const float* bias, std::size_t rows,
                              std::size_t cols) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < rows * cols) {
    y[i] += bias[i % cols];
  }
}

__global__ void AddKernel(float* x, const float* update, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    x[i] += update[i];
  }
}

__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              std::size_t cols, float eps) {
  __shared__ float shared[kThreads];
  const float* row = x + blockIdx.x * cols;
  float* out = y + blockIdx.x * cols;
  float sum = 0.0F;
  for (std::size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    sum = fmaf(row[c], row[c], sum);
  }
  const float total = BlockSum(sum, shared);
  const float inv = rsqrtf(total / static_cast<float>(cols) + eps);
  for (std::size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    out[c] = weight[c] * (row[c] * inv);
  }
}

__global__ void RopeHalfKernel(float* x, std::size_t rows, std::size_t heads,
                               std::size_t head_dim, std::size_t first_position,
                               float theta) {
  const std::size_t half = head_dim / 2;
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * heads * half) {
    return;
  }
  const std::size_t pair = i % half;
  const std::size_t head = (i / half) % heads;
  const std::size_t row = i / (half * heads);
  const float inv_freq =
      1.0F / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(first_position + row) * inv_freq;
  float s, c;
  sincosf(angle, &s, &c);
  float* base = x + (row * heads + head) * head_dim;
  const float a = base[pair];
  const float b = base[pair + half];
  base[pair] = a * c - b * s;
  base[pair + half] = b * c + a * s;
}

__global__ void RopeInterleavedKernel(float* x, std::size_t rows, std::size_t seq,
                                      std::size_t heads, std::size_t head_dim,
                                      const float* inv_freq) {
  const std::size_t pairs = head_dim / 2;
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * heads * pairs) {
    return;
  }
  const std::size_t pair = i % pairs;
  const std::size_t head = (i / pairs) % heads;
  const std::size_t row = i / (pairs * heads);
  const float angle = static_cast<float>(row % seq) * inv_freq[pair];
  float s, c;
  sincosf(angle, &s, &c);
  float* base = x + (row * heads + head) * head_dim;
  const float a = base[2 * pair];
  const float b = base[2 * pair + 1];
  base[2 * pair] = a * c - b * s;
  base[2 * pair + 1] = b * c + a * s;
}

// One block per (batch, query, head); threads cover keys, then dimensions.
__global__ void AttentionKernel(const float* q, const float* k, const float* v,
                                float* out, std::size_t q_len, std::size_t kv_len,
                                std::size_t heads, std::size_t kv_heads,
                                std::size_t head_dim, std::size_t q_batch_stride,
                                std::size_t kv_batch_stride, int causal,
                                std::size_t q_offset, std::size_t window) {
  __shared__ float scores[kMaxKeys];
  __shared__ float qs[256];
  __shared__ float shared[kThreads];
  const std::size_t head = blockIdx.x % heads;
  const std::size_t query = blockIdx.x / heads;
  const std::size_t batch = blockIdx.y;
  const std::size_t kv_head = head / (heads / kv_heads);
  const std::size_t position = q_offset + query;
  long long last = static_cast<long long>(kv_len) - 1;
  long long first = 0;
  if (causal != 0) {
    last = last < static_cast<long long>(position) ? last : static_cast<long long>(position);
  }
  if (window > 0) {
    const long long lo = static_cast<long long>(position) - static_cast<long long>(window) + 1;
    first = lo > 0 ? lo : 0;
  }
  const long long count = last - first + 1;
  const float* qrow = q + batch * q_batch_stride + (query * heads + head) * head_dim;
  for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    qs[d] = qrow[d];
  }
  __syncthreads();
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float* kbase = k + batch * kv_batch_stride;
  const float* vbase = v + batch * kv_batch_stride;
  float local_max = -INFINITY;
  for (long long j = threadIdx.x; j < count; j += blockDim.x) {
    const float* krow = kbase + ((first + j) * kv_heads + kv_head) * head_dim;
    float dot = 0.0F;
    for (std::size_t d = 0; d < head_dim; ++d) {
      dot = fmaf(qs[d], krow[d], dot);
    }
    dot *= scale;
    scores[j] = dot;
    local_max = fmaxf(local_max, dot);
  }
  const float maximum = BlockMax(local_max, shared);
  float local_sum = 0.0F;
  for (long long j = threadIdx.x; j < count; j += blockDim.x) {
    const float e = expf(scores[j] - maximum);
    scores[j] = e;
    local_sum += e;
  }
  const float total = BlockSum(local_sum, shared);
  const float inverse = 1.0F / total;
  float* orow = out + batch * q_batch_stride + (query * heads + head) * head_dim;
  for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (long long j = 0; j < count; ++j) {
      acc = fmaf(scores[j], vbase[((first + j) * kv_heads + kv_head) * head_dim + d], acc);
    }
    orow[d] = acc * inverse;
  }
}

__global__ void SiluMulKernel(float* gate, const float* up, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    gate[i] = g / (1.0F + expf(-g)) * up[i];
  }
}

__global__ void SiluKernel(float* x, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    x[i] = x[i] / (1.0F + expf(-x[i]));
  }
}

__global__ void GeluTanhKernel(float* x, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    x[i] = 0.5F * v * (1.0F + tanhf(0.7978845608028654F * (v + 0.044715F * v * v * v)));
  }
}

__global__ void UpsampleKernel(const float* in, float* out, std::size_t frames,
                               std::size_t cols, std::size_t factor) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= frames * factor * cols) {
    return;
  }
  const std::size_t c = i % cols;
  const std::size_t o = i / cols;
  float source = (1.0F / static_cast<float>(factor)) * (static_cast<float>(o) + 0.5F) - 0.5F;
  source = fmaxf(source, 0.0F);
  const auto i0 = static_cast<std::size_t>(source);
  const std::size_t i1 = i0 + (i0 + 1 < frames ? 1 : 0);
  const float l1 = source - static_cast<float>(i0);
  out[i] = (1.0F - l1) * in[i0 * cols + c] + l1 * in[i1 * cols + c];
}

__global__ void TimestepSinusKernel(float t, float* out) {
  const unsigned i = threadIdx.x;  // 128 threads
  const float step = static_cast<float>(9.210340371976184 / 127.0);  // ln(1e4)/(half-1)
  const float freq = expf(static_cast<float>(i) * -step);
  const float arg = (1000.0F * t) * freq;
  out[i] = sinf(arg);
  out[i + 128] = cosf(arg);
}

__global__ void AssembleDitKernel(const float* t_vec, const float* c_cond,
                                  const float* c_bias, const float* x_hist,
                                  const float* x_now, float* out, std::size_t d) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= d) {
    return;
  }
  out[i] = t_vec[i] + c_cond[i];
  out[d + i] = x_hist[i];
  out[2 * d + i] = x_now[i];
  out[3 * d + i] = t_vec[i] + c_bias[i];
  out[4 * d + i] = x_hist[i];
  out[5 * d + i] = x_now[i];
}

__global__ void EulerKernel(float* y, const float* v, float cfg, float dt,
                            std::size_t n) {
  const std::size_t i = threadIdx.x;
  if (i < n) {
    const float pred = v[i];
    const float null_pred = v[n + i];
    y[i] += (pred + (pred - null_pred) * cfg) * dt;
  }
}

__global__ void AxpyKernel(float* y, const float* x, float a, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    y[i] += a * x[i];
  }
}

// grid (frames, ceil(n_fft / 256)); spectrum of the frame staged in shared.
__global__ void IrfftWindowKernel(const float* head, const float* window,
                                  const float* cos_sin, float* frames_out,
                                  std::size_t n_fft) {
  extern __shared__ float spectrum[];  // [2 * bins]: re | im
  const std::size_t bins = n_fft / 2 + 1;
  const float* row = head + blockIdx.x * (n_fft + 2);
  for (std::size_t k = threadIdx.x; k < bins; k += blockDim.x) {
    const float magnitude = fminf(expf(row[k]), 100.0F);
    const float phase = row[bins + k];
    spectrum[k] = magnitude * cosf(phase);
    spectrum[bins + k] = magnitude * sinf(phase);
  }
  __syncthreads();
  const std::size_t n = blockIdx.y * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (n >= n_fft) {
    return;
  }
  // x[n] = (Re0 + (-1)^n Re_{N/2} + 2 sum_k (Re_k cos - Im_k sin)) / N
  float acc = spectrum[0] + ((n & 1U) != 0 ? -spectrum[bins - 1] : spectrum[bins - 1]);
  float partial = 0.0F;
  std::size_t index = n % n_fft;  // (k * n) mod N, k = 1
  for (std::size_t k = 1; k + 1 < bins; ++k) {
    partial += spectrum[k] * cos_sin[2 * index] - spectrum[bins + k] * cos_sin[2 * index + 1];
    index += n;
    if (index >= n_fft) {
      index -= n_fft;
    }
  }
  acc += 2.0F * partial;
  frames_out[blockIdx.x * n_fft + n] = acc / static_cast<float>(n_fft) * window[n];
}

__global__ void OverlapAddKernel(const float* frames_in, const float* window,
                                 float* out, std::size_t frames, std::size_t n_fft,
                                 std::size_t hop) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= frames * hop) {
    return;
  }
  const std::size_t pad = (n_fft - hop) / 2;
  const std::size_t s = i + pad;  // position in the untrimmed signal
  float value = 0.0F;
  float envelope = 0.0F;
  const std::size_t last = s / hop < frames - 1 ? s / hop : frames - 1;
  for (long long t = static_cast<long long>(last); t >= 0; --t) {
    const std::size_t offset = s - static_cast<std::size_t>(t) * hop;
    if (offset >= n_fft) {
      break;
    }
    value += frames_in[static_cast<std::size_t>(t) * n_fft + offset];
    envelope += window[offset] * window[offset];
  }
  out[i] = value / envelope;
}

}  // namespace

void LaunchGemmF32Bf16(const float* a, const std::uint16_t* w, float* c,
                       std::size_t m, std::size_t n, std::size_t k,
                       hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((n + kTile - 1) / kTile),
                  static_cast<unsigned>((m + kTile - 1) / kTile));
  hipLaunchKernelGGL(GemmF32Bf16Kernel, grid, dim3(kThreads), 0, stream, a, w, c, m,
                     n, k);
}

void LaunchAddBias(float* y, const float* bias, std::size_t rows,
                   std::size_t cols, hipStream_t stream) {
  hipLaunchKernelGGL(AddBiasKernel, dim3(Blocks(rows * cols)), dim3(kThreads), 0,
                     stream, y, bias, rows, cols);
}

void LaunchAdd(float* x, const float* update, std::size_t elements,
               hipStream_t stream) {
  hipLaunchKernelGGL(AddKernel, dim3(Blocks(elements)), dim3(kThreads), 0, stream,
                     x, update, elements);
}

void LaunchRmsNorm(const float* x, const float* weight, float* y,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads),
                     0, stream, x, weight, y, cols, eps);
}

void LaunchRopeHalf(float* x, std::size_t rows, std::size_t heads,
                    std::size_t head_dim, std::size_t first_position,
                    float theta, hipStream_t stream) {
  hipLaunchKernelGGL(RopeHalfKernel, dim3(Blocks(rows * heads * head_dim / 2)),
                     dim3(kThreads), 0, stream, x, rows, heads, head_dim,
                     first_position, theta);
}

void LaunchRopeInterleaved(float* x, std::size_t rows, std::size_t seq,
                           std::size_t heads, std::size_t head_dim,
                           const float* inv_freq, hipStream_t stream) {
  hipLaunchKernelGGL(RopeInterleavedKernel,
                     dim3(Blocks(rows * heads * head_dim / 2)), dim3(kThreads), 0,
                     stream, x, rows, seq, heads, head_dim, inv_freq);
}

void LaunchAttention(const float* q, const float* k, const float* v, float* out,
                     std::size_t batch, std::size_t q_len, std::size_t kv_len,
                     std::size_t heads, std::size_t kv_heads,
                     std::size_t head_dim, std::size_t q_batch_stride,
                     std::size_t kv_batch_stride, bool causal,
                     std::size_t q_offset, std::size_t window,
                     hipStream_t stream) {
  hipLaunchKernelGGL(AttentionKernel,
                     dim3(static_cast<unsigned>(q_len * heads), static_cast<unsigned>(batch)),
                     dim3(kThreads), 0, stream, q, k, v, out, q_len, kv_len, heads,
                     kv_heads, head_dim, q_batch_stride, kv_batch_stride,
                     causal ? 1 : 0, q_offset, window);
}

void LaunchSiluMul(float* gate, const float* up, std::size_t elements,
                   hipStream_t stream) {
  hipLaunchKernelGGL(SiluMulKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, gate, up, elements);
}

void LaunchSilu(float* x, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(SiluKernel, dim3(Blocks(elements)), dim3(kThreads), 0, stream,
                     x, elements);
}

void LaunchGeluTanh(float* x, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(GeluTanhKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, x, elements);
}

void LaunchLinearUpsample(const float* in, float* out, std::size_t frames,
                          std::size_t cols, std::size_t factor,
                          hipStream_t stream) {
  hipLaunchKernelGGL(UpsampleKernel, dim3(Blocks(frames * factor * cols)),
                     dim3(kThreads), 0, stream, in, out, frames, cols, factor);
}

void LaunchTimestepSinus(float t, float* out, hipStream_t stream) {
  hipLaunchKernelGGL(TimestepSinusKernel, dim3(1), dim3(128), 0, stream, t, out);
}

void LaunchAssembleDitInput(const float* t_vec, const float* c_cond,
                            const float* c_bias, const float* x_hist,
                            const float* x_now, float* out, std::size_t d,
                            hipStream_t stream) {
  hipLaunchKernelGGL(AssembleDitKernel, dim3(Blocks(d)), dim3(kThreads), 0, stream,
                     t_vec, c_cond, c_bias, x_hist, x_now, out, d);
}

void LaunchEulerStep(float* y, const float* v, float cfg, float dt,
                     std::size_t n, hipStream_t stream) {
  hipLaunchKernelGGL(EulerKernel, dim3(1), dim3(static_cast<unsigned>((n + 31) / 32 * 32)),
                     0, stream, y, v, cfg, dt, n);
}

void LaunchAxpy(float* y, const float* noise, float scale, std::size_t n,
                hipStream_t stream) {
  hipLaunchKernelGGL(AxpyKernel, dim3(Blocks(n)), dim3(kThreads), 0, stream, y,
                     noise, scale, n);
}

void LaunchIrfftWindow(const float* head, const float* window,
                       const float* cos_sin, float* frames_out,
                       std::size_t frames, std::size_t n_fft,
                       hipStream_t stream) {
  const std::size_t bins = n_fft / 2 + 1;
  hipLaunchKernelGGL(IrfftWindowKernel,
                     dim3(static_cast<unsigned>(frames), Blocks(n_fft)), dim3(kThreads),
                     static_cast<unsigned>(2 * bins * sizeof(float)), stream, head,
                     window, cos_sin, frames_out, n_fft);
}

void LaunchOverlapAdd(const float* frames_in, const float* window, float* out,
                      std::size_t frames, std::size_t n_fft, std::size_t hop,
                      hipStream_t stream) {
  hipLaunchKernelGGL(OverlapAddKernel, dim3(Blocks(frames * hop)), dim3(kThreads), 0,
                     stream, frames_in, window, out, frames, n_fft, hop);
}


// BF16 GEMV and FP32 -> BF16 conversion, copied from the Fish Drama-3
// backbone kernels (rikaaa0928, MIT; Bf16GemvKernel / FloatToBf16Kernel in
// fish_drama3/hip/backbone_ops.hip) so this extension does not depend on the
// external fish_drama3 extension.
namespace bf16_detail {

constexpr std::size_t kGemvThreads = 256;
constexpr std::size_t kWaveSize = 32;
constexpr std::size_t kWavesPerBlock = kGemvThreads / kWaveSize;

__device__ __forceinline__ float Bf16Bits(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

__device__ __forceinline__ std::uint16_t ToBf16Bits(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16U);
}

__device__ __forceinline__ float WaveSum(float val) {
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, kWaveSize);
  }
  return val;
}

__global__ void FloatToBf16Kernel(const float* input, std::uint16_t* output,
                                  std::size_t elements) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = ToBf16Bits(input[idx]);
  }
}

__device__ __forceinline__ void AccumulateBf16Vec8(const uint4& w_pack,
                                                   const float4& x0,
                                                   const float4& x1,
                                                   float& acc) {
  const std::uint16_t* w16 = reinterpret_cast<const std::uint16_t*>(&w_pack);
  acc = fmaf(Bf16Bits(w16[0]), x0.x, acc);
  acc = fmaf(Bf16Bits(w16[1]), x0.y, acc);
  acc = fmaf(Bf16Bits(w16[2]), x0.z, acc);
  acc = fmaf(Bf16Bits(w16[3]), x0.w, acc);
  acc = fmaf(Bf16Bits(w16[4]), x1.x, acc);
  acc = fmaf(Bf16Bits(w16[5]), x1.y, acc);
  acc = fmaf(Bf16Bits(w16[6]), x1.z, acc);
  acc = fmaf(Bf16Bits(w16[7]), x1.w, acc);
}

// One output row per 256-thread workgroup (8 wavefronts walk the row together
// in a single coalesced stream).
__launch_bounds__(kGemvThreads) __global__
    void Bf16GemvKernel(const std::uint16_t* __restrict__ weight_bf16,
                        const float* __restrict__ input_f32,
                        float* __restrict__ output_f32, std::size_t batch,
                        std::size_t rows, std::size_t cols) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float wave_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const std::size_t vec_cols = cols / 8U;
  const uint4* w_vec = reinterpret_cast<const uint4*>(weight_bf16 + row * cols);

  for (std::size_t b = 0; b < batch; ++b) {
    const float4* x_vec = reinterpret_cast<const float4*>(input_f32 + b * cols);
    float acc = 0.0F;
    for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kGemvThreads) {
      const uint4 w_pack = w_vec[idx];
      const float4 x0 = x_vec[2U * idx];
      const float4 x1 = x_vec[2U * idx + 1U];
      AccumulateBf16Vec8(w_pack, x0, x1, acc);
    }
    const float w_sum = WaveSum(acc);
    if (lane == 0) {
      wave_sums[wave_id] = w_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0F;
#pragma unroll
      for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
        total += wave_sums[w];
      }
      output_f32[b * rows + row] = total;
    }
    if (batch > 1U) {
      __syncthreads();
    }
  }
}

}  // namespace bf16_detail

void LaunchBf16Gemv(const std::uint16_t* weight_bf16, const float* input_f32,
                    float* output_f32, std::size_t batch, std::size_t rows,
                    std::size_t cols, hipStream_t stream) {
  hipLaunchKernelGGL(bf16_detail::Bf16GemvKernel, dim3(rows),
                     dim3(bf16_detail::kGemvThreads), 0, stream, weight_bf16,
                     input_f32, output_f32, batch, rows, cols);
}

void LaunchFloatToBf16(const float* input, std::uint16_t* output,
                       std::size_t elements, hipStream_t stream) {
  const std::size_t blocks =
      (elements + bf16_detail::kGemvThreads - 1U) / bf16_detail::kGemvThreads;
  hipLaunchKernelGGL(bf16_detail::FloatToBf16Kernel, dim3(blocks),
                     dim3(bf16_detail::kGemvThreads), 0, stream, input, output,
                     elements);
}

}  // namespace gufo::models::ming_tts::hip
