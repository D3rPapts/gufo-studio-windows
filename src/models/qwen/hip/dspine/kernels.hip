#include "src/models/qwen/hip/dspine/kernels.hpp"

#include <cfloat>
#include <cmath>

namespace gufo::hip::dspine {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxTopK = 32;

__device__ __forceinline__ float Bf(float x) {
  unsigned int u = __float_as_uint(x);
  if ((u & 0x7F800000u) == 0x7F800000u) return x;  // inf / nan unchanged
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  return __uint_as_float(u);
}

__device__ __forceinline__ std::uint16_t BfBits(float x) {
  return static_cast<std::uint16_t>(__float_as_uint(Bf(x)) >> 16);
}

__device__ __forceinline__ float FromBits(std::uint16_t b) {
  return __uint_as_float(static_cast<unsigned int>(b) << 16);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

/// Sum over the block (blockDim.x == kThreads); every thread gets the result.
__device__ float BlockSum(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
  for (int step = kThreads / 2; step > 0; step >>= 1) {
    if (static_cast<int>(threadIdx.x) < step)
      shared[threadIdx.x] += shared[threadIdx.x + step];
    __syncthreads();
  }
  const float out = shared[0];
  __syncthreads();
  return out;
}

__device__ float BlockMax(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
  for (int step = kThreads / 2; step > 0; step >>= 1) {
    if (static_cast<int>(threadIdx.x) < step)
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + step]);
    __syncthreads();
  }
  const float out = shared[0];
  __syncthreads();
  return out;
}

unsigned int Blocks(std::size_t n) {
  return static_cast<unsigned int>((n + kThreads - 1) / kThreads);
}

__global__ void RoundBf16Kernel(float* x, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) x[i] = Bf(x[i]);
}

__global__ void RmsNormBfKernel(const float* x, const float* w, float* y,
                                std::uint32_t dim, float eps) {
  __shared__ float shared[kThreads];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * dim;
  float* out = y + static_cast<std::size_t>(blockIdx.x) * dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) ss += row[i] * row[i];
  const float inv = rsqrtf(BlockSum(ss, shared) / static_cast<float>(dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads)
    out[i] = Bf(row[i] * inv * w[i]);
}

// One block per (row, head), head_dim threads (<= 256, even).
__global__ void HeadNormRopeKernel(float* x, const float* norm_w,
                                   std::uint32_t heads, std::uint32_t head_dim,
                                   std::uint32_t first_position, float theta,
                                   float eps) {
  __shared__ float shared[kThreads];
  __shared__ float normed[kThreads];
  const std::uint32_t row = blockIdx.x / heads;
  float* v = x + static_cast<std::size_t>(blockIdx.x) * head_dim;
  const std::uint32_t i = threadIdx.x;
  const float value = i < head_dim ? v[i] : 0.0f;
  const float inv = rsqrtf(BlockSum(value * value, shared) /
                               static_cast<float>(head_dim) + eps);
  if (i < head_dim) normed[i] = Bf(value * inv * norm_w[i]);
  __syncthreads();
  const std::uint32_t half = head_dim / 2;
  if (i < half) {
    const double freq =
        pow(static_cast<double>(theta), -2.0 * static_cast<double>(i) / head_dim);
    const double angle = static_cast<double>(first_position + row) * freq;
    const float c = static_cast<float>(cos(angle));
    const float s = static_cast<float>(sin(angle));
    const float x1 = normed[i], x2 = normed[i + half];
    v[i] = Bf(x1 * c - x2 * s);
    v[i + half] = Bf(x1 * s + x2 * c);
  }
}

__global__ void StoreBf16RowsKernel(const float* x, std::uint16_t* cache,
                                    std::size_t n, std::size_t offset) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) cache[offset + i] = BfBits(x[i]);
}

// One block per (query row, head). Three passes over the visible keys:
// scores (kept in global scratch), softmax statistics, weighted values.
__global__ void BlockAttentionKernel(const float* q, const std::uint16_t* cache_k,
                                     const std::uint16_t* cache_v,
                                     const float* block_k, const float* block_v,
                                     float* out, float* scores,
                                     std::uint32_t context, std::uint32_t block,
                                     std::uint32_t window, std::uint32_t heads,
                                     std::uint32_t kv_heads,
                                     std::uint32_t head_dim) {
  __shared__ float shared[kThreads];
  __shared__ float qs[256];
  __shared__ float partial[2][256];
  const std::uint32_t t = blockIdx.x / heads;
  const std::uint32_t h = blockIdx.x % heads;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t kv_width = kv_heads * head_dim;
  const float* qv = q + (static_cast<std::size_t>(t) * heads + h) * head_dim;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += kThreads) qs[i] = qv[i];
  __syncthreads();
  const std::uint32_t qpos = context + t;
  const std::uint32_t keys = context + t + 1;
  const std::uint32_t first =
      window != 0 && qpos + 1 > window ? qpos + 1 - window : 0;
  float* sc = scores + static_cast<std::size_t>(blockIdx.x) * (context + block);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float local_max = -INFINITY;
  for (std::uint32_t j = first + threadIdx.x; j < keys; j += kThreads) {
    float dot = 0.0f;
    if (j < context) {
      const std::uint16_t* kr = cache_k + static_cast<std::size_t>(j) * kv_width + kvh * head_dim;
      for (std::uint32_t i = 0; i < head_dim; ++i) dot += qs[i] * FromBits(kr[i]);
    } else {
      const float* kr = block_k + static_cast<std::size_t>(j - context) * kv_width + kvh * head_dim;
      for (std::uint32_t i = 0; i < head_dim; ++i) dot += qs[i] * kr[i];
    }
    dot *= scale;
    sc[j] = dot;
    local_max = fmaxf(local_max, dot);
  }
  const float mx = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t j = first + threadIdx.x; j < keys; j += kThreads) {
    const float e = expf(sc[j] - mx);
    sc[j] = e;
    local_sum += e;
  }
  const float denominator = BlockSum(local_sum, shared);
  __syncthreads();  // sc written by all threads before the value pass

  // Value pass: thread (half, i) sums half of the keys for dimension i.
  const std::uint32_t i = threadIdx.x % 128;
  const std::uint32_t part = threadIdx.x / 128;
  for (std::uint32_t base = 0; base < head_dim; base += 128) {
    const std::uint32_t d = base + i;
    float acc = 0.0f;
    if (d < head_dim) {
      for (std::uint32_t j = first + part; j < keys; j += 2) {
        const float p = sc[j];
        const float v = j < context
            ? FromBits(cache_v[static_cast<std::size_t>(j) * kv_width + kvh * head_dim + d])
            : block_v[static_cast<std::size_t>(j - context) * kv_width + kvh * head_dim + d];
        acc += p * v;
      }
    }
    partial[part][i] = acc;
    __syncthreads();
    if (part == 0 && d < head_dim)
      out[(static_cast<std::size_t>(t) * heads + h) * head_dim + d] =
          Bf((partial[0][i] + partial[1][i]) / denominator);
    __syncthreads();
  }
}

__global__ void AddBfKernel(const float* a, const float* b, float* out, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) out[i] = Bf(a[i] + b[i]);
}

__global__ void SwiGluBfKernel(float* gate, const float* up, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    gate[i] = Bf((g / (1.0f + expf(-g))) * up[i]);
  }
}

__global__ void BridgeNormalizeKernel(const float* f, float* rms, float* normalized,
                                      std::uint32_t dim, float eps) {
  __shared__ float shared[kThreads];
  const float* row = f + static_cast<std::size_t>(blockIdx.x) * dim;
  float* out = normalized + static_cast<std::size_t>(blockIdx.x) * dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) ss += row[i] * row[i];
  const float r = sqrtf(fmaxf(BlockSum(ss, shared) / static_cast<float>(dim), FLT_MIN));
  if (threadIdx.x == 0) rms[blockIdx.x] = r;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) out[i] = Bf(row[i] / (r + eps));
}

__global__ void BridgeCodesKernel(const float* projected, float* codes,
                                  std::uint32_t code_size, float eps) {
  __shared__ float shared[kThreads];
  const float* p = projected + static_cast<std::size_t>(blockIdx.x) * code_size;
  float* c = codes + static_cast<std::size_t>(blockIdx.x) * code_size;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < code_size; i += kThreads) ss += p[i] * p[i];
  const float norm = sqrtf(BlockSum(ss, shared));
  const float scale = sqrtf(static_cast<float>(code_size));
  for (std::uint32_t i = threadIdx.x; i < code_size; i += kThreads)
    c[i] = Bf(scale * p[i] / (norm + eps));
}

__global__ void CopyBf16RowKernel(const std::uint16_t* table, std::size_t offset,
                                  std::uint32_t width, float* out) {
  const std::uint32_t i = blockIdx.x * kThreads + threadIdx.x;
  if (i < width) out[i] = FromBits(table[offset + i]);
}

__global__ void BridgeMessageKernel(const float* packed, const float* bias,
                                    float* message, std::uint32_t rows,
                                    std::uint32_t rank) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * rank) return;
  const std::size_t t = i / rank, o = i % rank;
  if (t == 0) {
    message[i] = 0.0f;
    return;
  }
  const float* cur = packed + t * 3 * rank;
  const float* prev = packed + (t - 1) * 3 * rank;
  message[i] = Bf(Sigmoid(cur[o] + prev[rank + o] + bias[o]) * prev[2 * rank + o]);
}

__global__ void BridgeWriteNormKernel(const float* f, const float* rms,
                                      const float* delta, const float* next_w,
                                      float* out_w, float* x_in,
                                      std::uint32_t dim, float eps) {
  __shared__ float shared[kThreads];
  const std::uint32_t t = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(t) * dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    const float v = t == 0 ? Bf(f[base + i]) : Bf(f[base + i] + rms[t] * Bf(delta[base + i]));
    out_w[base + i] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, shared) / static_cast<float>(dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads)
    x_in[base + i] = Bf(Bf(out_w[base + i] * inv) * next_w[i]);
}

__global__ void MulRowsKernel(const float* x, const float* w, float* y,
                              std::size_t n, std::uint32_t dim) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) y[i] = x[i] * w[i % dim];
}

__global__ void ScaleBf16RowsKernel(const std::uint16_t* in, const float* w,
                                    std::uint16_t* out, std::size_t n,
                                    std::uint32_t dim) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) out[i] = BfBits(FromBits(in[i]) * w[i % dim]);
}

__global__ void ScaleBf16Kernel(const std::uint16_t* in, float scale,
                                std::uint16_t* out, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) out[i] = BfBits(scale * FromBits(in[i]));
}

__global__ void FloatToBf16BitsKernel(const float* x, std::uint16_t* out, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kThreads) + threadIdx.x;
  if (i < n) out[i] = BfBits(x[i]);
}

__device__ __forceinline__ bool Better(float a, std::int32_t ia, float b, std::int32_t ib) {
  return a > b || (a == b && ia < ib);
}

// One block per row. Each thread keeps a sorted local top-k, then lists are
// merged pairwise in shared memory; thread 0 sorts the winners by id.
__global__ void TopKKernel(const float* logits, std::uint32_t vocab,
                           std::uint32_t k, std::int32_t* ids, float* values) {
  __shared__ float sv[kThreads * kMaxTopK / 2];
  __shared__ std::int32_t si[kThreads * kMaxTopK / 2];
  const float* row = logits + static_cast<std::size_t>(blockIdx.x) * vocab;
  float lv[kMaxTopK];
  std::int32_t li[kMaxTopK];
  for (std::uint32_t r = 0; r < k; ++r) {
    lv[r] = -INFINITY;
    li[r] = 0x7FFFFFFF;
  }
  for (std::uint32_t v = threadIdx.x; v < vocab; v += kThreads) {
    const float x = row[v];
    const auto id = static_cast<std::int32_t>(v);
    if (!Better(x, id, lv[k - 1], li[k - 1])) continue;
    std::uint32_t pos = k - 1;
    while (pos > 0 && Better(x, id, lv[pos - 1], li[pos - 1])) {
      lv[pos] = lv[pos - 1];
      li[pos] = li[pos - 1];
      --pos;
    }
    lv[pos] = x;
    li[pos] = id;
  }
  // Pairwise merge: `active` lists of length k; the upper half hands its list
  // to shared memory, the lower half merges it into registers.
  for (std::uint32_t active = kThreads; active > 1; active /= 2) {
    const std::uint32_t half = active / 2;
    if (threadIdx.x >= half && threadIdx.x < active) {
      const std::uint32_t slot = (threadIdx.x - half) * k;
      for (std::uint32_t r = 0; r < k; ++r) {
        sv[slot + r] = lv[r];
        si[slot + r] = li[r];
      }
    }
    __syncthreads();
    if (threadIdx.x < half) {
      const std::uint32_t slot = threadIdx.x * k;
      float mv[kMaxTopK];
      std::int32_t mi[kMaxTopK];
      std::uint32_t a = 0, b = 0;
      for (std::uint32_t r = 0; r < k; ++r) {
        if (Better(lv[a], li[a], sv[slot + b], si[slot + b])) {
          mv[r] = lv[a];
          mi[r] = li[a];
          ++a;
        } else {
          mv[r] = sv[slot + b];
          mi[r] = si[slot + b];
          ++b;
        }
      }
      for (std::uint32_t r = 0; r < k; ++r) {
        lv[r] = mv[r];
        li[r] = mi[r];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (std::uint32_t r = 1; r < k; ++r)  // insertion sort by id
      for (std::uint32_t p = r; p > 0 && li[p] < li[p - 1]; --p) {
        const float tv = lv[p]; lv[p] = lv[p - 1]; lv[p - 1] = tv;
        const std::int32_t ti = li[p]; li[p] = li[p - 1]; li[p - 1] = ti;
      }
    for (std::uint32_t r = 0; r < k; ++r) {
      ids[blockIdx.x * k + r] = li[r];
      values[blockIdx.x * k + r] = lv[r];
    }
  }
}

// grid (rows, k senders), kThreads threads; rank <= 1024.
__global__ void RefineScoresKernel(const float* packed, const float* rms,
                                   const float* base_logits,
                                   const std::int32_t* cand,
                                   const std::uint16_t* tg,
                                   const std::uint16_t* tsp, const float* bias,
                                   std::uint32_t anchor, std::uint32_t k,
                                   std::uint32_t rank, std::uint32_t vocab,
                                   std::int32_t* idx) {
  __shared__ float shared[kThreads];
  __shared__ float m[1024];
  const std::uint32_t i = blockIdx.x;  // refinement row (block row i + 1)
  const std::uint32_t s = blockIdx.y;  // sender
  const std::size_t sid = i == 0 ? anchor : static_cast<std::size_t>(cand[(i - 1) * k + s]);
  const float* q = packed + static_cast<std::size_t>(i + 1) * 3 * rank;
  const std::uint16_t* sender = tsp + sid * 3 * rank;
  for (std::uint32_t o = threadIdx.x; o < rank; o += kThreads)
    m[o] = Sigmoid(q[o] + FromBits(sender[rank + o]) + bias[o]) *
           FromBits(sender[2 * rank + o]);
  __syncthreads();
  const float r = rms[i + 1];
  float best = -INFINITY;
  std::int32_t arg = 0;
  for (std::uint32_t j = 0; j < k; ++j) {
    const auto receiver = static_cast<std::size_t>(cand[i * k + j]);
    const std::uint16_t* g = tg + receiver * rank;
    float dot = 0.0f;
    for (std::uint32_t o = threadIdx.x; o < rank; o += kThreads) dot += m[o] * FromBits(g[o]);
    dot = BlockSum(dot, shared);
    const float score = base_logits[static_cast<std::size_t>(i) * vocab + receiver] + r * dot;
    if (score > best) {
      best = score;
      arg = static_cast<std::int32_t>(j);
    }
  }
  if (threadIdx.x == 0) idx[i * k + s] = arg;
}

}  // namespace

void LaunchRoundBf16(float* x, std::size_t n, hipStream_t stream) {
  if (n) RoundBf16Kernel<<<Blocks(n), kThreads, 0, stream>>>(x, n);
}

void LaunchRmsNormBf(const float* x, const float* w, float* y, std::uint32_t rows,
                     std::uint32_t dim, float eps, hipStream_t stream) {
  if (rows) RmsNormBfKernel<<<rows, kThreads, 0, stream>>>(x, w, y, dim, eps);
}

void LaunchHeadNormRopeBf(float* x, const float* norm_w, std::uint32_t rows,
                          std::uint32_t heads, std::uint32_t head_dim,
                          std::uint32_t first_position, float theta, float eps,
                          hipStream_t stream) {
  if (rows)
    HeadNormRopeKernel<<<rows * heads, kThreads, 0, stream>>>(
        x, norm_w, heads, head_dim, first_position, theta, eps);
}

void LaunchStoreBf16Rows(const float* x, std::uint16_t* cache, std::uint32_t rows,
                         std::uint32_t width, std::uint32_t first,
                         hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * width;
  if (n)
    StoreBf16RowsKernel<<<Blocks(n), kThreads, 0, stream>>>(
        x, cache, n, static_cast<std::size_t>(first) * width);
}

void LaunchBlockAttentionBf(const float* q, const std::uint16_t* cache_k,
                            const std::uint16_t* cache_v, const float* block_k,
                            const float* block_v, float* out, float* scores,
                            std::uint32_t context, std::uint32_t block,
                            std::uint32_t sliding_window, std::uint32_t heads,
                            std::uint32_t kv_heads, std::uint32_t head_dim,
                            hipStream_t stream) {
  if (block)
    BlockAttentionKernel<<<block * heads, kThreads, 0, stream>>>(
        q, cache_k, cache_v, block_k, block_v, out, scores, context, block,
        sliding_window, heads, kv_heads, head_dim);
}

void LaunchAddBf(const float* a, const float* b, float* out, std::size_t n,
                 hipStream_t stream) {
  if (n) AddBfKernel<<<Blocks(n), kThreads, 0, stream>>>(a, b, out, n);
}

void LaunchSwiGluBf(float* gate, const float* up, std::size_t n, hipStream_t stream) {
  if (n) SwiGluBfKernel<<<Blocks(n), kThreads, 0, stream>>>(gate, up, n);
}

void LaunchBridgeNormalize(const float* f, float* rms, float* normalized,
                           std::uint32_t rows, std::uint32_t dim, float eps,
                           hipStream_t stream) {
  if (rows)
    BridgeNormalizeKernel<<<rows, kThreads, 0, stream>>>(f, rms, normalized, dim, eps);
}

void LaunchBridgeCodes(const float* projected, float* codes, std::uint32_t rows,
                       std::uint32_t code_size, float eps, hipStream_t stream) {
  if (rows)
    BridgeCodesKernel<<<rows, kThreads, 0, stream>>>(projected, codes, code_size, eps);
}

void LaunchCopyBf16Row(const std::uint16_t* table, std::uint32_t row,
                       std::uint32_t width, float* out, hipStream_t stream) {
  CopyBf16RowKernel<<<Blocks(width), kThreads, 0, stream>>>(
      table, static_cast<std::size_t>(row) * width, width, out);
}

void LaunchBridgeMessage(const float* packed, const float* bias, float* message,
                         std::uint32_t rows, std::uint32_t rank,
                         hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * rank;
  if (n)
    BridgeMessageKernel<<<Blocks(n), kThreads, 0, stream>>>(packed, bias, message,
                                                            rows, rank);
}

void LaunchBridgeWriteNorm(const float* f, const float* rms, const float* delta,
                           const float* next_w, float* out_w, float* x_in,
                           std::uint32_t rows, std::uint32_t dim, float eps,
                           hipStream_t stream) {
  if (rows)
    BridgeWriteNormKernel<<<rows, kThreads, 0, stream>>>(f, rms, delta, next_w,
                                                         out_w, x_in, dim, eps);
}

void LaunchMulRows(const float* x, const float* w, float* y, std::uint32_t rows,
                   std::uint32_t dim, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * dim;
  if (n) MulRowsKernel<<<Blocks(n), kThreads, 0, stream>>>(x, w, y, n, dim);
}

void LaunchScaleBf16Rows(const std::uint16_t* in, const float* w,
                         std::uint16_t* out, std::size_t rows,
                         std::uint32_t dim, hipStream_t stream) {
  const std::size_t n = rows * dim;
  if (n) ScaleBf16RowsKernel<<<Blocks(n), kThreads, 0, stream>>>(in, w, out, n, dim);
}

void LaunchScaleBf16(const std::uint16_t* in, float scale, std::uint16_t* out,
                     std::size_t n, hipStream_t stream) {
  if (n) ScaleBf16Kernel<<<Blocks(n), kThreads, 0, stream>>>(in, scale, out, n);
}

void LaunchFloatToBf16Bits(const float* x, std::uint16_t* out, std::size_t n,
                           hipStream_t stream) {
  if (n) FloatToBf16BitsKernel<<<Blocks(n), kThreads, 0, stream>>>(x, out, n);
}

void LaunchTopKAscending(const float* logits, std::uint32_t rows,
                         std::uint32_t vocab, std::uint32_t k, std::int32_t* ids,
                         float* values, hipStream_t stream) {
  if (rows && k && k <= kMaxTopK)
    TopKKernel<<<rows, kThreads, 0, stream>>>(logits, vocab, k, ids, values);
}

void LaunchRefineScores(const float* packed, const float* rms,
                        const float* base_logits, const std::int32_t* cand,
                        const std::uint16_t* tg, const std::uint16_t* tsp,
                        const float* bias, std::uint32_t anchor,
                        std::uint32_t rows, std::uint32_t k, std::uint32_t rank,
                        std::uint32_t vocab, std::int32_t* idx,
                        hipStream_t stream) {
  if (rows && k && rank <= 1024)
    RefineScoresKernel<<<dim3(rows, k), kThreads, 0, stream>>>(
        packed, rms, base_logits, cand, tg, tsp, bias, anchor, k, rank, vocab,
        idx);
}

}  // namespace gufo::hip::dspine
