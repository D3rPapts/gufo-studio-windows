#include "src/models/qwen3_omni/hip/thinker_kernels.hpp"

#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>

namespace gufo::models::qwen3_omni::hip {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kAttentionThreads = 128;
constexpr std::size_t kSharedLimitBytes = 64 * 1024;

unsigned Blocks(std::size_t count, unsigned threads = kThreads) {
  return static_cast<unsigned>((count + threads - 1) / threads);
}

/// Sum over the block; `scratch` holds blockDim.x floats. All threads get
/// the result.
__device__ float BlockSum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__device__ float BlockMax(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              float* __restrict__ out, std::size_t cols,
                              float eps) {
  __shared__ float scratch[kThreads];
  const float* in = x + blockIdx.x * cols;
  float* row = out + blockIdx.x * cols;
  float squares = 0.0F;
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    squares += in[col] * in[col];
  }
  const float total = BlockSum(squares, scratch);
  const float scale = rsqrtf(total / static_cast<float>(cols) + eps);
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    row[col] = in[col] * scale * gamma[col];
  }
}

/// One block per (token, head); thread i owns the RoPE pair (i, i + half).
__global__ void QkNormRopeKernel(float* __restrict__ q, float* __restrict__ k,
                                 const float* __restrict__ q_gamma,
                                 const float* __restrict__ k_gamma,
                                 std::size_t heads, std::size_t kv_heads,
                                 std::size_t head_dim, std::size_t position0,
                                 float theta, float eps) {
  __shared__ float scratch[128];
  const std::size_t token = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t half = head_dim / 2;
  const bool is_query = head < heads;
  float* row = is_query ? q + (token * heads + head) * head_dim
                        : k + (token * kv_heads + (head - heads)) * head_dim;
  const float* gamma = is_query ? q_gamma : k_gamma;
  const std::size_t index = threadIdx.x;
  float first = 0.0F;
  float second = 0.0F;
  if (index < half) {
    first = row[index];
    second = row[index + half];
  }
  const float total = BlockSum(first * first + second * second, scratch);
  if (index >= half) {
    return;
  }
  const float scale = rsqrtf(total / static_cast<float>(head_dim) + eps);
  first = first * scale * gamma[index];
  second = second * scale * gamma[index + half];
  const double frequency =
      pow(static_cast<double>(theta),
          -2.0 * static_cast<double>(index) / static_cast<double>(head_dim));
  const double angle = static_cast<double>(position0 + token) * frequency;
  const float cosine = static_cast<float>(cos(angle));
  const float sine = static_cast<float>(sin(angle));
  row[index] = first * cosine - second * sine;
  row[index + half] = second * cosine + first * sine;
}

__global__ void StoreKvKernel(const float* __restrict__ k,
                              const float* __restrict__ v,
                              __half* __restrict__ k_cache,
                              __half* __restrict__ v_cache, std::size_t count,
                              std::size_t offset) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  k_cache[offset + index] = __float2half(k[index]);
  v_cache[offset + index] = __float2half(v[index]);
}

/// One block per (query token, head) with one thread per head dimension.
__global__ void AttentionKernel(const float* __restrict__ q,
                                const __half* __restrict__ k_cache,
                                const __half* __restrict__ v_cache,
                                float* __restrict__ out, std::size_t heads,
                                std::size_t kv_heads, std::size_t position0) {
  constexpr std::size_t kHeadDim = kAttentionThreads;
  extern __shared__ float shared[];
  float* query = shared;
  float* scratch = shared + kHeadDim;
  float* scores = shared + 2 * kHeadDim;
  const std::size_t token = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t kv_head = head / (heads / kv_heads);
  const std::size_t kv_dim = kv_heads * kHeadDim;
  const std::size_t visible = position0 + token + 1;
  const unsigned lane = threadIdx.x;
  query[lane] = q[(token * heads + head) * kHeadDim + lane] *
                rsqrtf(static_cast<float>(kHeadDim));
  __syncthreads();

  float maximum = -FLT_MAX;
  for (std::size_t key = lane; key < visible; key += blockDim.x) {
    const __half* row = k_cache + key * kv_dim + kv_head * kHeadDim;
    float dot = 0.0F;
    for (std::size_t dim = 0; dim < kHeadDim; dim += 2) {
      const float2 pair = __half22float2(
          *reinterpret_cast<const __half2*>(row + dim));
      dot += query[dim] * pair.x + query[dim + 1] * pair.y;
    }
    scores[key] = dot;
    maximum = fmaxf(maximum, dot);
  }
  maximum = BlockMax(maximum, scratch);
  float partial = 0.0F;
  for (std::size_t key = lane; key < visible; key += blockDim.x) {
    const float weight = expf(scores[key] - maximum);
    scores[key] = weight;
    partial += weight;
  }
  const float total = BlockSum(partial, scratch);
  float accumulator = 0.0F;
  for (std::size_t key = 0; key < visible; ++key) {
    accumulator += scores[key] *
                   __half2float(v_cache[key * kv_dim + kv_head * kHeadDim + lane]);
  }
  out[(token * heads + head) * kHeadDim + lane] = accumulator / total;
}

/// One block per token. Each wavefront scores experts warp, warp + waves, ...
/// with coalesced loads and a shuffle reduction; thread 0 then runs the
/// softmax and the top-k selection.
/// Router logits spread over the GPU: one wavefront per (token, expert),
/// coalesced row loads and a shuffle reduction.
__global__ void RouterLogitsKernel(const float* __restrict__ x,
                                   const float* __restrict__ router,
                                   float* __restrict__ logits,
                                   std::size_t hidden, std::size_t experts) {
  const unsigned lane = threadIdx.x % warpSize;
  const std::size_t expert =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const std::size_t token = blockIdx.y;
  if (expert >= experts) {
    return;
  }
  const float* row = router + expert * hidden;
  const float* input = x + token * hidden;
  float dot = 0.0F;
  for (std::size_t col = lane; col < hidden; col += warpSize) {
    dot += row[col] * input[col];
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane == 0) {
    logits[token * experts + expert] = dot;
  }
}

/// One block per token over its router logits: softmax and top-k.
__global__ void RouteKernel(const float* __restrict__ logits,
                            std::int32_t* __restrict__ ids,
                            float* __restrict__ weights, std::size_t experts,
                            std::size_t used) {
  extern __shared__ float shared[];
  float* probabilities = shared;
  const std::size_t token = blockIdx.x;
  for (std::size_t index = threadIdx.x; index < experts; index += blockDim.x) {
    probabilities[index] = logits[token * experts + index];
  }
  __syncthreads();

  // Softmax and top-k across the block: thread e owns expert e.
  float* scratch = probabilities + experts;
  int* winner = reinterpret_cast<int*>(scratch + blockDim.x);
  float* picked = reinterpret_cast<float*>(winner + blockDim.x);
  const std::size_t expert = threadIdx.x;
  const bool owns = expert < experts;
  const float logit = owns ? probabilities[expert] : -FLT_MAX;
  const float maximum = BlockMax(logit, scratch);
  const float probability = owns ? expf(logit - maximum) : 0.0F;
  const float total = BlockSum(probability, scratch);
  bool taken = !owns;
  float selected = 0.0F;
  for (std::size_t slot = 0; slot < used; ++slot) {
    // Arg-max with ties to the lower expert id.
    scratch[threadIdx.x] = taken ? -1.0F : probability;
    winner[threadIdx.x] = static_cast<int>(threadIdx.x);
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other = scratch[threadIdx.x + stride];
        const int other_id = winner[threadIdx.x + stride];
        if (other > scratch[threadIdx.x] ||
            (other == scratch[threadIdx.x] && other_id < winner[threadIdx.x])) {
          scratch[threadIdx.x] = other;
          winner[threadIdx.x] = other_id;
        }
      }
      __syncthreads();
    }
    const int best = winner[0];
    const float best_probability = scratch[0];
    __syncthreads();
    if (static_cast<int>(threadIdx.x) == best) {
      taken = true;
    }
    if (threadIdx.x == 0) {
      ids[token * used + slot] = best;
      picked[slot] = best_probability;
    }
    selected += best_probability / total;
  }
  __syncthreads();
  if (threadIdx.x < used) {
    weights[token * used + threadIdx.x] = picked[threadIdx.x] / total / selected;
  }
}

__global__ void SwigluKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ out, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = gate[index];
    out[index] = value / (1.0F + expf(-value)) * up[index];
  }
}

__global__ void MoeCombineKernel(float* __restrict__ residual,
                                 const float* __restrict__ rows,
                                 const float* __restrict__ weights,
                                 std::size_t used, std::size_t hidden) {
  const std::size_t token = blockIdx.y;
  const std::size_t col =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= hidden) {
    return;
  }
  float sum = 0.0F;
  for (std::size_t slot = 0; slot < used; ++slot) {
    sum += weights[token * used + slot] *
           rows[(token * used + slot) * hidden + col];
  }
  residual[token * hidden + col] += sum;
}

__global__ void CopyColumnsKernel(const float* __restrict__ src,
                                  std::size_t src_width, std::size_t begin,
                                  std::size_t width, float* __restrict__ dst,
                                  std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const std::size_t row = index / width;
    dst[index] = src[row * src_width + begin + index % width];
  }
}

__global__ void AddKernel(float* __restrict__ residual,
                          const float* __restrict__ update, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    residual[index] += update[index];
  }
}

}  // namespace

void LaunchRmsNorm(const float* x, const float* gamma, float* out,
                   std::size_t rows, std::size_t cols, float eps,
                   hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(kThreads), 0, stream, x,
                     gamma, out, cols, eps);
}

void LaunchQkNormRope(float* q, float* k, const float* q_gamma,
                      const float* k_gamma, std::size_t tokens,
                      std::size_t heads, std::size_t kv_heads,
                      std::size_t head_dim, std::size_t position0, float theta,
                      float eps, hipStream_t stream) {
  hipLaunchKernelGGL(QkNormRopeKernel,
                     dim3(static_cast<unsigned>(tokens),
                          static_cast<unsigned>(heads + kv_heads)),
                     dim3(128), 0, stream, q, k, q_gamma, k_gamma, heads,
                     kv_heads, head_dim, position0, theta, eps);
}

void LaunchStoreKv(const float* k, const float* v, __half* k_cache,
                   __half* v_cache, std::size_t tokens, std::size_t kv_dim,
                   std::size_t position0, hipStream_t stream) {
  const std::size_t count = tokens * kv_dim;
  hipLaunchKernelGGL(StoreKvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, k, v, k_cache, v_cache, count, position0 * kv_dim);
}

std::size_t MaxAttentionContext() {
  return kSharedLimitBytes / sizeof(float) - 2 * kAttentionThreads;
}

void LaunchAttention(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out, std::size_t tokens,
                     std::size_t heads, std::size_t kv_heads, std::size_t,
                     std::size_t position0, hipStream_t stream) {
  const std::size_t visible = position0 + tokens;
  const std::size_t shared = (2 * kAttentionThreads + visible) * sizeof(float);
  hipLaunchKernelGGL(AttentionKernel,
                     dim3(static_cast<unsigned>(tokens),
                          static_cast<unsigned>(heads)),
                     dim3(kAttentionThreads), shared, stream, q, k_cache,
                     v_cache, out, heads, kv_heads, position0);
}

void LaunchRoute(const float* x, const float* router, float* logits,
                 std::int32_t* ids, float* weights, std::size_t tokens,
                 std::size_t hidden, std::size_t experts, std::size_t used,
                 hipStream_t stream) {
  static const int warp = [] {
    int size = 32;
    (void)hipDeviceGetAttribute(&size, hipDeviceAttributeWarpSize, 0);
    return size;
  }();
  const unsigned per_block = kThreads / static_cast<unsigned>(warp);
  hipLaunchKernelGGL(RouterLogitsKernel,
                     dim3(static_cast<unsigned>((experts + per_block - 1) /
                                                per_block),
                          static_cast<unsigned>(tokens)),
                     dim3(kThreads), 0, stream, x, router, logits, hidden,
                     experts);
  const std::size_t shared = (experts + 2 * kThreads + 16) * sizeof(float);
  hipLaunchKernelGGL(RouteKernel, dim3(static_cast<unsigned>(tokens)),
                     dim3(kThreads), shared, stream, logits, ids, weights,
                     experts, used);
}

void LaunchSwiglu(const float* gate, const float* up, float* out,
                  std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, out, count);
}

void LaunchMoeCombine(float* residual, const float* rows, const float* weights,
                      std::size_t tokens, std::size_t used, std::size_t hidden,
                      hipStream_t stream) {
  hipLaunchKernelGGL(MoeCombineKernel,
                     dim3(Blocks(hidden), static_cast<unsigned>(tokens)),
                     dim3(kThreads), 0, stream, residual, rows, weights, used,
                     hidden);
}

void LaunchCopyColumns(const float* src, std::size_t src_width,
                       std::size_t begin, std::size_t width, float* dst,
                       std::size_t rows, hipStream_t stream) {
  const std::size_t count = rows * width;
  hipLaunchKernelGGL(CopyColumnsKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, src, src_width, begin, width, dst, count);
}

void LaunchAdd(float* residual, const float* update, std::size_t count,
               hipStream_t stream) {
  hipLaunchKernelGGL(AddKernel, dim3(Blocks(count)), dim3(kThreads), 0, stream,
                     residual, update, count);
}

}  // namespace gufo::models::qwen3_omni::hip
