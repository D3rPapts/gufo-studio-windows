// Written with AI assistance (Claude Code).
#include "src/models/voxcpm2/hip/ops.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

#include <rocwmma/rocwmma.hpp>

namespace gufo::models::voxcpm2::hip {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kChunk = 1024;  // keys per online-softmax chunk

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

__global__ void RopeTableKernel(float* x, std::size_t rows, std::size_t period,
                                std::size_t heads, std::size_t head_dim, std::size_t first,
                                const float* cos_table, const float* sin_table) {
  const std::size_t half = head_dim / 2;
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= rows * heads * half) {
    return;
  }
  const std::size_t i = index % half;
  const std::size_t h = (index / half) % heads;
  const std::size_t r = index / (half * heads);
  const std::size_t position = first + r % period;
  const float c = cos_table[position * half + i];
  const float s = sin_table[position * half + i];
  float* row = x + (r * heads + h) * head_dim;
  const float a = row[i];
  const float b = row[i + half];
  row[i] = a * c - b * s;
  row[i + half] = b * c + a * s;
}

__global__ void AttentionKernel(const float* q, const float* k, const float* v, float* out,
                                std::size_t q_len, std::size_t kv_len, std::size_t heads,
                                std::size_t kv_heads, std::size_t head_dim,
                                std::size_t q_batch_stride, std::size_t kv_batch_stride,
                                bool causal, std::size_t q_offset, float scale) {
  __shared__ float qs[256];
  __shared__ float scores[kChunk];
  __shared__ float reduce[kThreads];
  const std::size_t h = blockIdx.x;
  const std::size_t i = blockIdx.y;
  const std::size_t b = blockIdx.z;
  const std::size_t group = heads / kv_heads;
  const std::size_t kvh = h / group;
  const float* qi = q + b * q_batch_stride + (i * heads + h) * head_dim;
  const float* kb = k + b * kv_batch_stride;
  const float* vb = v + b * kv_batch_stride;
  for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    qs[d] = qi[d];
  }
  __syncthreads();
  std::size_t keys = kv_len;
  if (causal) {
    keys = q_offset + i + 1 < kv_len ? q_offset + i + 1 : kv_len;
  }
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  float acc = 0.0F;
  for (std::size_t c0 = 0; c0 < keys; c0 += kChunk) {
    const std::size_t count = keys - c0 < kChunk ? keys - c0 : static_cast<std::size_t>(kChunk);
    float local_max = -INFINITY;
    for (std::size_t j = threadIdx.x; j < count; j += blockDim.x) {
      const float* kr = kb + ((c0 + j) * kv_heads + kvh) * head_dim;
      float dot = 0.0F;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot = fmaf(qs[d], kr[d], dot);
      }
      const float s = dot * scale;
      scores[j] = s;
      local_max = fmaxf(local_max, s);
    }
    const float chunk_max = BlockMax(local_max, reduce);
    const float new_max = fmaxf(running_max, chunk_max);
    const float correction = expf(running_max - new_max);
    float local_sum = 0.0F;
    for (std::size_t j = threadIdx.x; j < count; j += blockDim.x) {
      const float e = expf(scores[j] - new_max);
      scores[j] = e;
      local_sum += e;
    }
    const float chunk_sum = BlockSum(local_sum, reduce);
    running_sum = running_sum * correction + chunk_sum;
    if (threadIdx.x < head_dim) {
      float a = acc * correction;
      for (std::size_t j = 0; j < count; ++j) {
        a = fmaf(scores[j], vb[((c0 + j) * kv_heads + kvh) * head_dim + threadIdx.x], a);
      }
      acc = a;
    }
    running_max = new_max;
    __syncthreads();
  }
  if (threadIdx.x < head_dim) {
    out[b * q_batch_stride + (i * heads + h) * head_dim + threadIdx.x] = acc / running_sum;
  }
}

__global__ void FsqKernel(float* x, std::size_t n, float scale) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    x[i] = rintf(tanhf(x[i]) * scale) / scale;
  }
}

__global__ void AssembleLocEncKernel(const float* special, const float* proj, float* out,
                                     std::size_t count, std::size_t patch, std::size_t dim) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t seq = patch + 1;
  if (index >= count * seq * dim) {
    return;
  }
  const std::size_t c = index % dim;
  const std::size_t t = (index / dim) % seq;
  const std::size_t n = index / (dim * seq);
  out[index] = t == 0 ? special[c] : proj[(n * patch + t - 1) * dim + c];
}

__global__ void GatherRowsKernel(const float* src, float* dst, std::size_t count,
                                 std::size_t stride_rows, std::size_t dim) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < count * dim) {
    const std::size_t c = index % dim;
    const std::size_t r = index / dim;
    dst[index] = src[r * stride_rows * dim + c];
  }
}

__global__ void SelectRowsKernel(float* dst, const float* alt, const std::uint8_t* mask,
                                 std::size_t rows, std::size_t dim) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index < rows * dim && mask[index / dim] != 0) {
    dst[index] = alt[index];
  }
}

__global__ void ConcatRowsKernel(const float* a, const float* b, const std::uint8_t* mask,
                                 float* out, std::size_t rows, std::size_t dim) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= rows * 2 * dim) {
    return;
  }
  const std::size_t c = index % (2 * dim);
  const std::size_t r = index / (2 * dim);
  if (c < dim) {
    out[index] = a[r * dim + c];
  } else {
    const bool keep = mask == nullptr || mask[r] != 0;
    out[index] = keep ? b[r * dim + c - dim] : 0.0F;
  }
}

__global__ void AssembleDitKernel(const float* mu, const float* t_row, const float* cond_rows,
                                  const float* x_rows, float* out, std::size_t mu_tokens,
                                  std::size_t patch, std::size_t dim) {
  const std::size_t seq = mu_tokens + 1 + 2 * patch;
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= 2 * seq * dim) {
    return;
  }
  const std::size_t c = index % dim;
  const std::size_t t = (index / dim) % seq;
  const std::size_t b = index / (dim * seq);
  float value = 0.0F;
  if (t < mu_tokens) {
    value = b == 0 ? mu[t * dim + c] : 0.0F;
  } else if (t == mu_tokens) {
    value = t_row[c];
  } else if (t < mu_tokens + 1 + patch) {
    value = cond_rows[(t - mu_tokens - 1) * dim + c];
  } else {
    value = x_rows[(t - mu_tokens - 1 - patch) * dim + c];
  }
  out[index] = value;
}

__global__ void CfgEulerKernel(float* x, const float* vel, float cfg, float dt, std::size_t n,
                               bool zero_star) {
  __shared__ float reduce[kThreads];
  const float* pos = vel;
  const float* neg = vel + n;
  float st = 1.0F;
  if (zero_star) {
    float dot = 0.0F;
    float norm = 0.0F;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
      dot = fmaf(pos[i], neg[i], dot);
      norm = fmaf(neg[i], neg[i], norm);
    }
    const float total_dot = BlockSum(dot, reduce);
    const float total_norm = BlockSum(norm, reduce);
    st = total_dot / (total_norm + 1e-8F);
  }
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float dphi = neg[i] * st + cfg * (pos[i] - neg[i] * st);
    x[i] = x[i] - dt * dphi;
  }
}

// ---- skinny split-K GEMM ---------------------------------------------------
// Block tile: 32 rows x 64 columns x one 128-wide K slice (split-K), staged in
// 32-deep chunks. v2: each thread owns 2 rows x 4 columns and reads them as
// float2/float4 from LDS (v1 read 5 LDS values per 4 FMAs and was LDS-bound).
constexpr int kSkCols = 64;
constexpr int kSkSlice = static_cast<int>(kSkinnySlice);
constexpr int kSkRows = static_cast<int>(kSkinnyRows);
constexpr int kSkChunk = 32;

__device__ inline float Bf16Bits(std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned>(bits) << 16U);
}

__global__ __launch_bounds__(256) void SkinnyGemmKernel(const float* x, const std::uint16_t* w,
                                                        float* partial, int m, int n, int k) {
  __shared__ __attribute__((aligned(16))) float xs[kSkChunk][kSkRows + 4];
  __shared__ __attribute__((aligned(16))) float ws[kSkChunk][kSkCols + 4];
  const int n0 = blockIdx.x * kSkCols;
  const int split = blockIdx.y;
  const int k_begin = split * kSkSlice;
  const int k_end = min(k, k_begin + kSkSlice);
  const int t = threadIdx.x;
  const int cg = t % 16;  // columns cg*4 .. cg*4+3
  const int rg = t / 16;  // rows rg*2, rg*2+1
  float acc[2][4] = {};
  for (int kc = k_begin; kc < k_end; kc += kSkChunk) {
    const int kn = min(kSkChunk, k_end - kc);
    {  // X chunk: 32 rows x 32 k, one float4 per thread
      const int row = t / 8;
      const int k4 = (t % 8) * 4;
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (row < m && k4 < kn) {
        v = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(row) * k + kc + k4);
      }
      xs[k4 + 0][row] = v.x;
      xs[k4 + 1][row] = v.y;
      xs[k4 + 2][row] = v.z;
      xs[k4 + 3][row] = v.w;
    }
    {  // W chunk: 64 columns x 32 k BF16, 8 values (16 bytes) per thread
      const int col = t / 4;
      const int k8 = (t % 4) * 8;
      const int j = n0 + col;
      unsigned words[4] = {0U, 0U, 0U, 0U};
      if (j < n && k8 < kn) {
        const uint4 raw =
            *reinterpret_cast<const uint4*>(w + static_cast<std::size_t>(j) * k + kc + k8);
        words[0] = raw.x;
        words[1] = raw.y;
        words[2] = raw.z;
        words[3] = raw.w;
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        ws[k8 + 2 * e][col] = Bf16Bits(static_cast<std::uint16_t>(words[e] & 0xFFFFU));
        ws[k8 + 2 * e + 1][col] = Bf16Bits(static_cast<std::uint16_t>(words[e] >> 16U));
      }
    }
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < kn; ++kk) {
      const float4 wv = *reinterpret_cast<const float4*>(&ws[kk][cg * 4]);
      const float2 xv = *reinterpret_cast<const float2*>(&xs[kk][rg * 2]);
      acc[0][0] = fmaf(xv.x, wv.x, acc[0][0]);
      acc[0][1] = fmaf(xv.x, wv.y, acc[0][1]);
      acc[0][2] = fmaf(xv.x, wv.z, acc[0][2]);
      acc[0][3] = fmaf(xv.x, wv.w, acc[0][3]);
      acc[1][0] = fmaf(xv.y, wv.x, acc[1][0]);
      acc[1][1] = fmaf(xv.y, wv.y, acc[1][1]);
      acc[1][2] = fmaf(xv.y, wv.z, acc[1][2]);
      acc[1][3] = fmaf(xv.y, wv.w, acc[1][3]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = rg * 2 + r;
    if (row >= m) {
      continue;
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = n0 + cg * 4 + c;
      if (j < n) {
        partial[(static_cast<std::size_t>(split) * kSkRows + row) * n + j] = acc[r][c];
      }
    }
  }
}

// BF16 WMMA variant (activations rounded to BF16 RNE, FP32 accumulation):
// 4 waves (wave32) per block, each a 16-column strip x both 16-row tiles.
using WmmaBf16 = rocwmma::bfloat16_t;
constexpr int kWmPitch = kSkSlice + 8;  // bf16 elements per LDS row

__device__ inline std::uint16_t F32ToBf16Bits(float value) {
  unsigned bits = __float_as_uint(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__global__ __launch_bounds__(128) void SkinnyGemmWmmaKernel(const float* x,
                                                            const std::uint16_t* w,
                                                            float* partial, int m, int n,
                                                            int k) {
  __shared__ __attribute__((aligned(16))) std::uint16_t as[kSkRows * kWmPitch];
  __shared__ __attribute__((aligned(16))) std::uint16_t bs[kSkCols * kWmPitch];
  const int n0 = blockIdx.x * kSkCols;
  const int split = blockIdx.y;
  const int k0 = split * kSkSlice;
  const int kn = min(kSkSlice, k - k0);
  for (int i = threadIdx.x; i < kSkRows * kSkSlice; i += blockDim.x) {
    const int r = i / kSkSlice;
    const int c = i % kSkSlice;
    const float v = (r < m && c < kn) ? x[static_cast<std::size_t>(r) * k + k0 + c] : 0.0F;
    as[r * kWmPitch + c] = F32ToBf16Bits(v);
  }
  for (int i = threadIdx.x; i < kSkCols * (kSkSlice / 8); i += blockDim.x) {
    const int col = i / (kSkSlice / 8);
    const int c8 = (i % (kSkSlice / 8)) * 8;
    const int j = n0 + col;
    uint4 raw = make_uint4(0U, 0U, 0U, 0U);
    if (j < n && c8 < kn) {
      raw = *reinterpret_cast<const uint4*>(w + static_cast<std::size_t>(j) * k + k0 + c8);
    }
    *reinterpret_cast<uint4*>(&bs[col * kWmPitch + c8]) = raw;
  }
  __syncthreads();
  using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, WmmaBf16, rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, WmmaBf16, rocwmma::col_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
  const int wave = threadIdx.x / 32;  // 16-column strip
  FragC acc[2];
  rocwmma::fill_fragment(acc[0], 0.0F);
  rocwmma::fill_fragment(acc[1], 0.0F);
  const auto* a_ptr = reinterpret_cast<const WmmaBf16*>(as);
  const auto* b_ptr = reinterpret_cast<const WmmaBf16*>(bs);
  for (int kk = 0; kk < kSkSlice; kk += 16) {
    FragB fb;
    rocwmma::load_matrix_sync(fb, b_ptr + wave * 16 * kWmPitch + kk, kWmPitch);
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      FragA fa;
      rocwmma::load_matrix_sync(fa, a_ptr + tile * 16 * kWmPitch + kk, kWmPitch);
      rocwmma::mma_sync(acc[tile], fa, fb, acc[tile]);
    }
  }
  if (n0 + wave * 16 + 16 <= n) {
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      rocwmma::store_matrix_sync(
          partial + (static_cast<std::size_t>(split) * kSkRows + tile * 16) * n + n0 + wave * 16,
          acc[tile], static_cast<unsigned>(n), rocwmma::mem_row_major);
    }
  }
}

// mode 0: y = bias + sum; 1: y += bias + sum (residual); 2: SiLU-gated pair,
// partial has 2 * n_out columns [gate | up]: y = silu(g) * u.
__global__ void SkinnyReduceKernel(const float* partial, const float* bias, float* y, int m,
                                   int n, int splits, int mode) {
  const int n_out = mode == 2 ? n / 2 : n;
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= static_cast<std::size_t>(m) * n_out) {
    return;
  }
  const int row = static_cast<int>(index / n_out);
  const int j = static_cast<int>(index % n_out);
  float sum = bias != nullptr ? bias[j] : 0.0F;
  for (int s = 0; s < splits; ++s) {
    sum += partial[(static_cast<std::size_t>(s) * kSkRows + row) * n + j];
  }
  if (mode == 2) {
    float up = bias != nullptr ? bias[j + n_out] : 0.0F;
    for (int s = 0; s < splits; ++s) {
      up += partial[(static_cast<std::size_t>(s) * kSkRows + row) * n + j + n_out];
    }
    y[index] = sum / (1.0F + expf(-sum)) * up;
  } else if (mode == 1) {
    y[index] += sum;
  } else {
    y[index] = sum;
  }
}

// ---- AudioVAE kernels --------------------------------------------------------
constexpr int kTile = 64;
constexpr int kTileK = 16;

// C[M, N] = A[M, K] B[K, N] (+ bias[m]); 256 threads, 4x4 outputs each.
__global__ void GemmNNKernel(const float* a, const float* b, const float* bias, float* c,
                             std::size_t m, std::size_t n, std::size_t k) {
  __shared__ float as[kTileK][kTile + 4];
  __shared__ float bs[kTileK][kTile + 4];
  const unsigned tx = threadIdx.x % 16;
  const unsigned ty = threadIdx.x / 16;
  const std::size_t m0 = static_cast<std::size_t>(blockIdx.y) * kTile;
  const std::size_t n0 = static_cast<std::size_t>(blockIdx.x) * kTile;
  float acc[4][4] = {};
  for (std::size_t k0 = 0; k0 < k; k0 += kTileK) {
    for (unsigned i = threadIdx.x; i < kTile * kTileK; i += blockDim.x) {
      const unsigned row = i / kTileK;  // A: 64 rows x 16 k
      const unsigned col = i % kTileK;
      const std::size_t gm = m0 + row;
      const std::size_t gk = k0 + col;
      as[col][row] = (gm < m && gk < k) ? a[gm * k + gk] : 0.0F;
      const unsigned kr = i / kTile;  // B: 16 k x 64 n
      const unsigned nc = i % kTile;
      const std::size_t bk = k0 + kr;
      const std::size_t bn = n0 + nc;
      bs[kr][nc] = (bk < k && bn < n) ? b[bk * n + bn] : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      float av[4];
      float bv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        av[i] = as[kk][ty * 4 + i];
        bv[i] = bs[kk][tx * 4 + i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
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
    const float add = bias != nullptr ? bias[gm] : 0.0F;
    for (int j = 0; j < 4; ++j) {
      const std::size_t gn = n0 + tx * 4 + j;
      if (gn < n) {
        c[gm * n + gn] = acc[i][j] + add;
      }
    }
  }
}

__global__ void SnakeKernel(float* x, const float* alpha, const float* inv, std::size_t channels,
                            std::size_t length) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < channels * length) {
    const std::size_t c = i / length;
    const float s = sinf(alpha[c] * x[i]);
    x[i] = x[i] + inv[c] * (s * s);
  }
}

__global__ void ChannelScaleBiasKernel(float* x, const float* scale, const float* bias,
                                       std::size_t channels, std::size_t length) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < channels * length) {
    const std::size_t c = i / length;
    x[i] = x[i] * scale[c] + bias[c];
  }
}

__global__ void DepthwiseConvKernel(const float* x, std::size_t len_in, const float* w,
                                    const float* b, float* y, std::size_t channels,
                                    std::size_t kernel, std::size_t dilation,
                                    std::size_t len_out) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= channels * len_out) {
    return;
  }
  const std::size_t c = i / len_out;
  const std::size_t t = i % len_out;
  const float* xr = x + c * len_in + t;
  const float* wr = w + c * kernel;
  float sum = 0.0F;
  for (std::size_t j = 0; j < kernel; ++j) {
    sum = fmaf(wr[j], xr[j * dilation], sum);
  }
  y[i] = b[c] + sum;
}

__global__ void Im2colKernel(const float* x, std::size_t len_in, float* col,
                             std::size_t channels, std::size_t kernel, std::size_t stride,
                             std::size_t dilation, std::size_t len_out) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= channels * kernel * len_out) {
    return;
  }
  const std::size_t t = i % len_out;
  const std::size_t row = i / len_out;  // ci * kernel + j
  const std::size_t ci = row / kernel;
  const std::size_t j = row % kernel;
  col[i] = x[ci * len_in + t * stride + j * dilation];
}

__global__ void PadLeftKernel(const float* history, const float* x, float* padded,
                              std::size_t channels, std::size_t pad, std::size_t length) {
  const std::size_t total = pad + length;
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= channels * total) {
    return;
  }
  const std::size_t c = i / total;
  const std::size_t t = i % total;
  padded[i] = t < pad ? (history != nullptr ? history[c * pad + t] : 0.0F)
                      : x[c * length + t - pad];
}

__global__ void KeepTailKernel(const float* padded, float* history, std::size_t channels,
                               std::size_t pad, std::size_t total) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < channels * pad) {
    const std::size_t c = i / pad;
    const std::size_t t = i % pad;
    history[i] = padded[c * total + total - pad + t];
  }
}

__global__ void ConvTransposeCombineKernel(const float* z, const float* bias, float* y,
                                           std::size_t cout, std::size_t stride,
                                           std::size_t length) {
  const std::size_t out_len = length * stride;
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= cout * out_len) {
    return;
  }
  const std::size_t co = i / out_len;
  const std::size_t pos = i % out_len;
  const std::size_t e = pos / stride;
  const std::size_t r = pos % stride;
  const std::size_t ext = length + 1;
  y[i] = bias[co] + z[(r * cout + co) * ext + e + 1] + z[((r + stride) * cout + co) * ext + e];
}

__global__ void TanhKernel(float* x, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    x[i] = tanhf(x[i]);
  }
}

}  // namespace

std::size_t SkinnyPartialFloats(std::size_t n, std::size_t k) {
  return (k + kSkinnySlice - 1) / kSkinnySlice * kSkinnyRows * n;
}

void LaunchSkinnyGemm(const float* x, const std::uint16_t* w, const float* bias, float* y,
                      std::size_t m, std::size_t n, std::size_t k, float* partial,
                      hipStream_t stream, SkinnyEpilogue epilogue, bool wmma) {
  const auto splits = static_cast<unsigned>((k + kSkinnySlice - 1) / kSkinnySlice);
  const auto col_blocks = static_cast<unsigned>((n + kSkCols - 1) / kSkCols);
  const bool use_wmma = wmma && n % kSkCols == 0 && k % kSkinnySlice == 0;
  const std::size_t n_out = epilogue == SkinnyEpilogue::kSiluMulPairs ? n / 2 : n;
  for (std::size_t r0 = 0; r0 < m; r0 += kSkinnyRows) {
    const std::size_t rows = m - r0 < kSkinnyRows ? m - r0 : kSkinnyRows;
    if (use_wmma) {
      hipLaunchKernelGGL(SkinnyGemmWmmaKernel, dim3(col_blocks, splits), dim3(128), 0, stream,
                         x + r0 * k, w, partial, static_cast<int>(rows), static_cast<int>(n),
                         static_cast<int>(k));
    } else {
      hipLaunchKernelGGL(SkinnyGemmKernel, dim3(col_blocks, splits), dim3(kThreads), 0, stream,
                         x + r0 * k, w, partial, static_cast<int>(rows), static_cast<int>(n),
                         static_cast<int>(k));
    }
    hipLaunchKernelGGL(SkinnyReduceKernel, dim3(Blocks(rows * n_out)), dim3(kThreads), 0, stream,
                       partial, bias, y + r0 * n_out, static_cast<int>(rows), static_cast<int>(n),
                       static_cast<int>(splits), static_cast<int>(epilogue));
  }
}

void LaunchGemmF32NN(const float* a, const float* b, const float* bias, float* c,
                     std::size_t m, std::size_t n, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((n + kTile - 1) / kTile),
                  static_cast<unsigned>((m + kTile - 1) / kTile));
  hipLaunchKernelGGL(GemmNNKernel, grid, dim3(kThreads), 0, stream, a, b, bias, c, m, n, k);
}

void LaunchSnake(float* x, const float* alpha, const float* inv, std::size_t channels,
                 std::size_t length, hipStream_t stream) {
  hipLaunchKernelGGL(SnakeKernel, dim3(Blocks(channels * length)), dim3(kThreads), 0, stream, x,
                     alpha, inv, channels, length);
}

void LaunchChannelScaleBias(float* x, const float* scale, const float* bias,
                            std::size_t channels, std::size_t length, hipStream_t stream) {
  hipLaunchKernelGGL(ChannelScaleBiasKernel, dim3(Blocks(channels * length)), dim3(kThreads), 0,
                     stream, x, scale, bias, channels, length);
}

void LaunchDepthwiseConv(const float* x, std::size_t len_in, const float* w, const float* b,
                         float* y, std::size_t channels, std::size_t kernel,
                         std::size_t dilation, std::size_t len_out, hipStream_t stream) {
  hipLaunchKernelGGL(DepthwiseConvKernel, dim3(Blocks(channels * len_out)), dim3(kThreads), 0,
                     stream, x, len_in, w, b, y, channels, kernel, dilation, len_out);
}

void LaunchIm2col(const float* x, std::size_t len_in, float* col, std::size_t channels,
                  std::size_t kernel, std::size_t stride, std::size_t dilation,
                  std::size_t len_out, hipStream_t stream) {
  hipLaunchKernelGGL(Im2colKernel, dim3(Blocks(channels * kernel * len_out)), dim3(kThreads), 0,
                     stream, x, len_in, col, channels, kernel, stride, dilation, len_out);
}

void LaunchPadLeft(const float* history, const float* x, float* padded, std::size_t channels,
                   std::size_t pad, std::size_t length, hipStream_t stream) {
  hipLaunchKernelGGL(PadLeftKernel, dim3(Blocks(channels * (pad + length))), dim3(kThreads), 0,
                     stream, history, x, padded, channels, pad, length);
}

void LaunchKeepTail(const float* padded, float* history, std::size_t channels, std::size_t pad,
                    std::size_t total, hipStream_t stream) {
  hipLaunchKernelGGL(KeepTailKernel, dim3(Blocks(channels * pad)), dim3(kThreads), 0, stream,
                     padded, history, channels, pad, total);
}

void LaunchConvTransposeCombine(const float* z, const float* bias, float* y, std::size_t cout,
                                std::size_t stride, std::size_t length, hipStream_t stream) {
  hipLaunchKernelGGL(ConvTransposeCombineKernel, dim3(Blocks(cout * length * stride)),
                     dim3(kThreads), 0, stream, z, bias, y, cout, stride, length);
}

void LaunchTanh(float* x, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(TanhKernel, dim3(Blocks(elements)), dim3(kThreads), 0, stream, x, elements);
}

void LaunchRopeTable(float* x, std::size_t rows, std::size_t period, std::size_t heads,
                     std::size_t head_dim, std::size_t first, const float* cos_table,
                     const float* sin_table, hipStream_t stream) {
  const std::size_t n = rows * heads * (head_dim / 2);
  if (n == 0) {
    return;
  }
  hipLaunchKernelGGL(RopeTableKernel, dim3(Blocks(n)), dim3(kThreads), 0, stream, x, rows,
                     period, heads, head_dim, first, cos_table, sin_table);
}

void LaunchAttention(const float* q, const float* k, const float* v, float* out,
                     std::size_t batch, std::size_t q_len, std::size_t kv_len,
                     std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
                     std::size_t q_batch_stride, std::size_t kv_batch_stride, bool causal,
                     std::size_t q_offset, hipStream_t stream) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  hipLaunchKernelGGL(AttentionKernel,
                     dim3(static_cast<unsigned>(heads), static_cast<unsigned>(q_len),
                          static_cast<unsigned>(batch)),
                     dim3(kThreads), 0, stream, q, k, v, out, q_len, kv_len, heads, kv_heads,
                     head_dim, q_batch_stride, kv_batch_stride, causal, q_offset, scale);
}

void LaunchFsqQuantize(float* x, std::size_t elements, float scale, hipStream_t stream) {
  hipLaunchKernelGGL(FsqKernel, dim3(Blocks(elements)), dim3(kThreads), 0, stream, x, elements,
                     scale);
}

void LaunchAssembleLocEnc(const float* special, const float* proj, float* out,
                          std::size_t count, std::size_t patch, std::size_t dim,
                          hipStream_t stream) {
  const std::size_t n = count * (patch + 1) * dim;
  hipLaunchKernelGGL(AssembleLocEncKernel, dim3(Blocks(n)), dim3(kThreads), 0, stream, special,
                     proj, out, count, patch, dim);
}

void LaunchGatherRows(const float* src, float* dst, std::size_t count, std::size_t stride_rows,
                      std::size_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(GatherRowsKernel, dim3(Blocks(count * dim)), dim3(kThreads), 0, stream, src,
                     dst, count, stride_rows, dim);
}

void LaunchSelectRows(float* dst, const float* alt, const std::uint8_t* mask, std::size_t rows,
                      std::size_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(SelectRowsKernel, dim3(Blocks(rows * dim)), dim3(kThreads), 0, stream, dst,
                     alt, mask, rows, dim);
}

void LaunchConcatRows(const float* a, const float* b, const std::uint8_t* mask, float* out,
                      std::size_t rows, std::size_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(ConcatRowsKernel, dim3(Blocks(rows * 2 * dim)), dim3(kThreads), 0, stream,
                     a, b, mask, out, rows, dim);
}

void LaunchAssembleDit(const float* mu, const float* t_row, const float* cond_rows,
                       const float* x_rows, float* out, std::size_t mu_tokens,
                       std::size_t patch, std::size_t dim, hipStream_t stream) {
  const std::size_t n = 2 * (mu_tokens + 1 + 2 * patch) * dim;
  hipLaunchKernelGGL(AssembleDitKernel, dim3(Blocks(n)), dim3(kThreads), 0, stream, mu, t_row,
                     cond_rows, x_rows, out, mu_tokens, patch, dim);
}

void LaunchCfgEuler(float* x, const float* vel, float cfg, float dt, std::size_t n,
                    bool zero_star, hipStream_t stream) {
  hipLaunchKernelGGL(CfgEulerKernel, dim3(1), dim3(kThreads), 0, stream, x, vel, cfg, dt, n,
                     zero_star);
}

}  // namespace gufo::models::voxcpm2::hip
