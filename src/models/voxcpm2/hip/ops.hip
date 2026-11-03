// Written with AI assistance (Claude Code).
#include "src/models/voxcpm2/hip/ops.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

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

}  // namespace

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
