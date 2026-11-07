// Written with AI assistance (Claude Code).
#include "src/models/indextts/hip/ops.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>

namespace gufo::models::indextts::hip {
namespace {

constexpr unsigned kThreads = 256;

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>(std::max<std::size_t>(1, (n + kThreads - 1) / kThreads));
}

__device__ float BlockSum(float v, float* shared) {
  for (int o = warpSize / 2; o > 0; o >>= 1) {
    v += __shfl_down(v, o);
  }
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  const int warps = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
  float total = 0.0F;
  if (threadIdx.x == 0) {
    for (int w = 0; w < warps; ++w) {
      total += shared[w];
    }
    shared[0] = total;
  }
  __syncthreads();
  total = shared[0];
  __syncthreads();
  return total;
}

__global__ void SplitQkvKernel(const float* qkv, float* q, float* k, float* v, std::size_t rows,
                               std::size_t dim) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * dim) {
    return;
  }
  const std::size_t r = i / dim;
  const std::size_t c = i % dim;
  const float* src = qkv + r * 3 * dim;
  q[i] = src[c];
  k[i] = src[dim + c];
  v[i] = src[2 * dim + c];
}

__global__ void AdaRmsNormKernel(const float* x, const float* w, const float* ss, float* y,
                                 std::size_t dim, float eps) {
  __shared__ float shared[32];
  const std::size_t r = blockIdx.x;
  const float* xr = x + r * dim;
  float acc = 0.0F;
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    acc += xr[c] * xr[c];
  }
  const float total = BlockSum(acc, shared);
  const float inv = rsqrtf(total / static_cast<float>(dim) + eps);
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    y[r * dim + c] = ss[c] * (xr[c] * inv * w[c]) + ss[dim + c];
  }
}

__global__ void ModulatedLayerNormKernel(const float* x, const float* mod, float* y,
                                         std::size_t dim, float eps) {
  __shared__ float shared[32];
  const std::size_t r = blockIdx.x;
  const float* xr = x + r * dim;
  float s = 0.0F;
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    s += xr[c];
  }
  const float mean = BlockSum(s, shared) / static_cast<float>(dim);
  float q = 0.0F;
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    const float d = xr[c] - mean;
    q += d * d;
  }
  const float inv = rsqrtf(BlockSum(q, shared) / static_cast<float>(dim) + eps);
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    y[r * dim + c] = (xr[c] - mean) * inv * (1.0F + mod[dim + c]) + mod[c];
  }
}

__global__ void WaveNetGateKernel(const float* a, const float* g, float* acts, std::size_t rows,
                                  std::size_t dim) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * dim) {
    return;
  }
  const std::size_t r = i / dim;
  const std::size_t c = i % dim;
  const float t = tanhf(a[r * 2 * dim + c] + g[c]);
  const float s = 1.0F / (1.0F + expf(-(a[r * 2 * dim + dim + c] + g[dim + c])));
  acts[i] = t * s;
}

__global__ void Im2colKernel(const float* x, float* col, std::size_t t_in, std::size_t t_out,
                             std::size_t cin, std::size_t kernel, std::size_t dilation,
                             std::ptrdiff_t pad_left, int mode) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t width = kernel * cin;
  if (i >= t_out * width) {
    return;
  }
  const std::size_t t = i / width;
  const std::size_t rest = i % width;
  const std::size_t j = rest / cin;
  const std::size_t c = rest % cin;
  std::ptrdiff_t src = static_cast<std::ptrdiff_t>(t + j * dilation) - pad_left;
  const auto n = static_cast<std::ptrdiff_t>(t_in);
  if (src < 0 || src >= n) {
    if (mode == 0) {
      col[i] = 0.0F;
      return;
    }
    if (mode == 1) {
      src = src < 0 ? -src : 2 * (n - 1) - src;
    } else {
      src = src < 0 ? 0 : n - 1;
    }
  }
  col[i] = x[static_cast<std::size_t>(src) * cin + c];
}

__global__ void ConvTransposeGatherKernel(const float* z, const float* bias, float* y,
                                          std::size_t t_in, std::size_t t_out, std::size_t cout,
                                          std::size_t kernel, std::size_t stride,
                                          std::size_t pad) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= t_out * cout) {
    return;
  }
  const std::size_t n = i / cout;
  const std::size_t co = i % cout;
  float acc = bias != nullptr ? bias[co] : 0.0F;
  for (std::size_t j = 0; j < kernel; ++j) {
    const std::ptrdiff_t num = static_cast<std::ptrdiff_t>(n + pad) - static_cast<std::ptrdiff_t>(j);
    if (num < 0 || num % static_cast<std::ptrdiff_t>(stride) != 0) {
      continue;
    }
    const std::size_t src = static_cast<std::size_t>(num) / stride;
    if (src < t_in) {
      acc += z[src * kernel * cout + j * cout + co];
    }
  }
  y[i] = acc;
}

__device__ float UpSample(const float* x, std::size_t channels, std::size_t c, std::ptrdiff_t t,
                          std::ptrdiff_t m, const float* fu) {
  // m in [0, 2t): y_full[m + 15] of conv_transpose(replicate_pad(x, 5), fu, stride 2), times 2
  const std::ptrdiff_t n = m + 15;
  float s = 0.0F;
  for (int k = (n & 1); k < 12; k += 2) {
    const std::ptrdiff_t i = (n - k) / 2;  // index into the padded input
    if (i < 0 || i > t + 9) {
      continue;
    }
    std::ptrdiff_t src = i - 5;
    src = src < 0 ? 0 : (src >= t ? t - 1 : src);
    s += x[static_cast<std::size_t>(src) * channels + c] * fu[k];
  }
  return 2.0F * s;
}

__global__ void AliasFreeSnakeKernel(const float* x, float* y, const float* alpha,
                                     const float* inv_beta, const float* fu, const float* fd,
                                     std::size_t t, std::size_t channels) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= t * channels) {
    return;
  }
  const std::size_t j = i / channels;
  const std::size_t c = i % channels;
  const float al = alpha[c];
  const float ib = inv_beta[c];
  const auto tt = static_cast<std::ptrdiff_t>(t);
  float acc = 0.0F;
  for (int k = 0; k < 12; ++k) {
    // padded upsampled index: 5 left / 6 right replicate around [0, 2t)
    std::ptrdiff_t m = static_cast<std::ptrdiff_t>(2 * j) + k - 5;
    m = m < 0 ? 0 : (m >= 2 * tt ? 2 * tt - 1 : m);
    float v = UpSample(x, channels, c, tt, m, fu);
    const float sn = sinf(v * al);
    v = v + ib * (sn * sn);
    acc += v * fd[k];
  }
  y[i] = acc;
}

__global__ void CfgEulerKernel(float* x, const float* vc, const float* vu, float cfg, float dt,
                               std::size_t rows, std::size_t cols, std::size_t prompt_rows) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * cols) {
    return;
  }
  const float a = 1.0F + cfg;
  const float v = a * vc[i] - cfg * vu[i];
  x[i] = (i / cols) < prompt_rows ? 0.0F : x[i] + dt * v;
}

__global__ void ScaleKernel(float* y, const float* a, float scale, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * scale;
  }
}

__global__ void ClampKernel(float* y, float lo, float hi, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    y[i] = fminf(fmaxf(y[i], lo), hi);
  }
}

__global__ void ConcatColsKernel(const float* a, const float* b, float* dst, std::size_t rows,
                                 std::size_t ac, std::size_t bc) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t w = ac + bc;
  if (i >= rows * w) {
    return;
  }
  const std::size_t r = i / w;
  const std::size_t c = i % w;
  dst[i] = c < ac ? a[r * ac + c] : b[r * bc + (c - ac)];
}

}  // namespace

void LaunchSplitQkv(const float* qkv, float* q, float* k_out, float* v_out, std::size_t rows,
                    std::size_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(SplitQkvKernel, dim3(Blocks(rows * dim)), dim3(kThreads), 0, stream, qkv, q,
                     k_out, v_out, rows, dim);
}

void LaunchAdaRmsNorm(const float* x, const float* norm_w, const float* ss, float* y,
                      std::size_t rows, std::size_t dim, float eps, hipStream_t stream) {
  hipLaunchKernelGGL(AdaRmsNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0, stream,
                     x, norm_w, ss, y, dim, eps);
}

void LaunchModulatedLayerNorm(const float* x, const float* mod, float* y, std::size_t rows,
                              std::size_t dim, float eps, hipStream_t stream) {
  hipLaunchKernelGGL(ModulatedLayerNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     stream, x, mod, y, dim, eps);
}

void LaunchWaveNetGate(const float* a, const float* g, float* acts, std::size_t rows,
                       std::size_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(WaveNetGateKernel, dim3(Blocks(rows * dim)), dim3(kThreads), 0, stream, a, g,
                     acts, rows, dim);
}

void LaunchIm2colTM(const float* x, float* col, std::size_t t_in, std::size_t t_out,
                    std::size_t cin, std::size_t kernel, std::size_t dilation,
                    std::ptrdiff_t pad_left, PadMode mode, hipStream_t stream) {
  hipLaunchKernelGGL(Im2colKernel, dim3(Blocks(t_out * kernel * cin)), dim3(kThreads), 0, stream, x,
                     col, t_in, t_out, cin, kernel, dilation, pad_left, static_cast<int>(mode));
}

void LaunchConvTransposeGather(const float* z, const float* bias, float* y, std::size_t t_in,
                               std::size_t t_out, std::size_t cout, std::size_t kernel,
                               std::size_t stride, std::size_t pad, hipStream_t stream) {
  hipLaunchKernelGGL(ConvTransposeGatherKernel, dim3(Blocks(t_out * cout)), dim3(kThreads), 0,
                     stream, z, bias, y, t_in, t_out, cout, kernel, stride, pad);
}

void LaunchAliasFreeSnake(const float* x, float* y, const float* alpha, const float* inv_beta,
                          const float* up_filter, const float* down_filter, std::size_t t,
                          std::size_t channels, hipStream_t stream) {
  hipLaunchKernelGGL(AliasFreeSnakeKernel, dim3(Blocks(t * channels)), dim3(kThreads), 0, stream, x,
                     y, alpha, inv_beta, up_filter, down_filter, t, channels);
}

void LaunchCfgEulerTM(float* x, const float* vc, const float* vu, float cfg, float dt,
                      std::size_t rows, std::size_t cols, std::size_t prompt_rows,
                      hipStream_t stream) {
  hipLaunchKernelGGL(CfgEulerKernel, dim3(Blocks(rows * cols)), dim3(kThreads), 0, stream, x, vc, vu,
                     cfg, dt, rows, cols, prompt_rows);
}

void LaunchScale(float* y, const float* a, float scale, std::size_t n, hipStream_t stream) {
  hipLaunchKernelGGL(ScaleKernel, dim3(Blocks(n)), dim3(kThreads), 0, stream, y, a, scale, n);
}

void LaunchClamp(float* y, float lo, float hi, std::size_t n, hipStream_t stream) {
  hipLaunchKernelGGL(ClampKernel, dim3(Blocks(n)), dim3(kThreads), 0, stream, y, lo, hi, n);
}

void LaunchConcatCols(const float* a, const float* b, float* dst, std::size_t rows,
                      std::size_t a_cols, std::size_t b_cols, hipStream_t stream) {
  hipLaunchKernelGGL(ConcatColsKernel, dim3(Blocks(rows * (a_cols + b_cols))), dim3(kThreads), 0,
                     stream, a, b, dst, rows, a_cols, b_cols);
}

}  // namespace gufo::models::indextts::hip
