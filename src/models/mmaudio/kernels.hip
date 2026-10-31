// MMAudio for gufo: element-wise, normalization, convolution and attention
// kernels (see kernels.hpp).
// Written with AI assistance (Claude Code).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

#include "src/models/mmaudio/kernels.hpp"

namespace gufo::models::mmaudio {
namespace {

constexpr int kThreads = 256;

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float BlockMax(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float Silu(float v) {
  return v / (1.0F + expf(-v));
}

// ---- conversions -------------------------------------------------------------

__global__ void ToHalfKernel(const float* in, std::size_t count,
                             std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = fminf(fmaxf(in[i], -65504.0F), 65504.0F);
  out[i] = __half_as_ushort(__float2half(v));
}

__global__ void WidenKernel(const std::uint16_t* in, std::size_t count,
                            int width, const float* bias, bool accumulate,
                            float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float v = __half2float(__ushort_as_half(in[i]));
  if (bias != nullptr)
    v += bias[i % width];
  out[i] = accumulate ? out[i] + v : v;
}

__global__ void AddBiasKernel(const float* in, std::size_t count, int width,
                              const float* bias, bool accumulate, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float v = in[i];
  if (bias != nullptr)
    v += bias[i % width];
  out[i] = accumulate ? out[i] + v : v;
}

// ---- softmax / normalization -------------------------------------------------

__global__ void SoftmaxKernel(float* scores, int columns, float scale,
                              int causal) {
  __shared__ float shared[kThreads];
  const std::size_t r = blockIdx.x;
  float* row = scores + r * columns;
  const int limit =
      causal > 0 ? min(columns, static_cast<int>(r % causal) + 1) : columns;
  float maximum = -INFINITY;
  for (int c = threadIdx.x; c < limit; c += blockDim.x)
    maximum = fmaxf(maximum, row[c] * scale);
  maximum = BlockMax(maximum, shared);
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += blockDim.x) {
    const float e = c < limit ? expf(row[c] * scale - maximum) : 0.0F;
    row[c] = e;
    sum += e;
  }
  sum = BlockSum(sum, shared);
  const float inverse = 1.0F / sum;
  for (int c = threadIdx.x; c < limit; c += blockDim.x)
    row[c] *= inverse;
}

__global__ void LayerNormKernel(const float* x, int width, const float* weight,
                                const float* bias, float eps,
                                const float* shift, const float* scale,
                                int mod_stride, float* out) {
  __shared__ float shared[kThreads];
  const std::size_t r = blockIdx.x;
  const float* row = x + r * width;
  float sum = 0;
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    sum += row[c];
  const float mean = BlockSum(sum, shared) / width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const float d = row[c] - mean;
    squares += d * d;
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / width + eps);
  const std::size_t mod = r * static_cast<std::size_t>(mod_stride);
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    float y = (row[c] - mean) * rstd;
    if (weight != nullptr)
      y *= weight[c];
    if (bias != nullptr)
      y += bias[c];
    if (scale != nullptr)
      y = y * (1.0F + scale[mod + c]) + (shift != nullptr ? shift[mod + c] : 0);
    out[r * width + c] = y;
  }
}

__global__ void L2NormalizeKernel(float* x, int width) {
  __shared__ float shared[kThreads];
  float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    squares += row[c] * row[c];
  const float norm = sqrtf(BlockSum(squares, shared));
  const float inverse = 1.0F / fmaxf(norm, 1e-12F);
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    row[c] *= inverse;
}

__global__ void PixelNormKernel(const float* x, int width, float* out) {
  __shared__ float shared[kThreads];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    squares += x[base + c] * x[base + c];
  const float norm = sqrtf(BlockSum(squares, shared));
  const float inverse =
      1.0F / (1e-4F + norm * rsqrtf(static_cast<float>(width)));
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    out[base + c] = x[base + c] * inverse;
}

// ---- element-wise ------------------------------------------------------------

__global__ void SiluKernel(float* x, std::size_t count, float scale) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = Silu(x[i]) * scale;
}

__global__ void SiluToKernel(const float* x, std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = Silu(x[i]);
}

__global__ void GeluKernel(float* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    x[i] = 0.5F * v * (1.0F + erff(v * 0.7071067811865476F));
  }
}

__global__ void QuickGeluKernel(float* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = x[i];
    x[i] = v / (1.0F + expf(-1.702F * v));
  }
}

__global__ void SwiGluKernel(const float* x, std::size_t rows, int hidden,
                             float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * hidden)
    return;
  const std::size_t r = i / hidden;
  const int f = static_cast<int>(i % hidden);
  const float* row = x + r * 2 * hidden;
  out[i] = Silu(row[f]) * row[hidden + f];
}

__global__ void AddGatedKernel(float* x, const float* y, std::size_t rows,
                               int width, const float* gate, int gate_stride) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * width)
    return;
  float v = y[i];
  if (gate != nullptr)
    v *= gate[(i / width) * static_cast<std::size_t>(gate_stride) + i % width];
  x[i] += v;
}

__global__ void MpSumKernel(const float* a, const float* b, std::size_t count,
                            float wa, float wb, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = a[i] * wa + b[i] * wb;
}

__global__ void ClampKernel(float* x, std::size_t count, float low,
                            float high) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = fminf(fmaxf(x[i], low), high);
}

__global__ void AffineKernel(float* x, std::size_t count, float scale,
                             float add) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = x[i] * scale + add;
}

__global__ void ColumnAffineKernel(float* x, std::size_t count, int width,
                                   const float* mul, const float* add) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % width);
  float v = x[i];
  if (mul != nullptr)
    v *= mul[c];
  if (add != nullptr)
    v += add[c];
  x[i] = v;
}

__global__ void AddKernel(float* x, const float* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}

__global__ void AddRowKernel(float* x, const float* v, std::size_t count,
                             int width) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += v[i % width];
}

__global__ void RepeatRowsKernel(const float* in, int period,
                                 std::size_t count, int width, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t r = i / width;
  out[i] = in[(r % period) * width + i % width];
}

__global__ void ColumnMeanKernel(const float* x, std::size_t rows, int width,
                                 float* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= width)
    return;
  double sum = 0;
  for (std::size_t r = 0; r < rows; ++r)
    sum += x[r * width + c];
  out[c] = static_cast<float>(sum / static_cast<double>(rows));
}

__global__ void GatherRowsKernel(const float* table, const int* ids,
                                 std::size_t count, int width, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t r = i / width;
  out[i] = table[static_cast<std::size_t>(ids[r]) * width + i % width];
}

// ---- convolution helpers -----------------------------------------------------

__global__ void Im2ColKernel(const float* x, int rows_in, int channels,
                             int kernel, int dilation, int padding,
                             std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t columns = static_cast<std::size_t>(channels) * kernel;
  const std::size_t t = i / columns;
  const int column = static_cast<int>(i % columns);
  const int c = column / kernel;
  const int k = column % kernel;
  const long long source =
      static_cast<long long>(t) + static_cast<long long>(k) * dilation -
      padding;
  out[i] = source >= 0 && source < rows_in
               ? x[static_cast<std::size_t>(source) * channels + c]
               : 0.0F;
}

__global__ void Col2ImTransposedKernel(const float* y, int rows_in,
                                       int channels, int kernel, int stride,
                                       int padding, const float* bias,
                                       std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const long long s = static_cast<long long>(i / channels);
  const int co = static_cast<int>(i % channels);
  float sum = bias != nullptr ? bias[co] : 0.0F;
  // t * stride + k == s + padding with 0 <= k < kernel.
  const long long target = s + padding;
  long long t_high = target / stride;
  for (long long t = t_high; t >= 0; --t) {
    const long long k = target - t * stride;
    if (k >= kernel)
      break;
    if (t < rows_in)
      sum += y[static_cast<std::size_t>(t) * channels * kernel +
               static_cast<std::size_t>(co) * kernel + k];
  }
  out[i] = sum;
}

__global__ void UpsampleNearestKernel(const float* x, int width, int factor,
                                      std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t r = i / width;
  out[i] = x[(r / factor) * width + i % width];
}

__global__ void NearestExactKernel(const float* x, int rows_in, int width,
                                   int rows_out, std::size_t count,
                                   float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t r = i / width;
  const float scale =
      static_cast<float>(rows_in) / static_cast<float>(rows_out);
  int source = static_cast<int>(floorf((static_cast<float>(r) + 0.5F) * scale));
  source = min(source, rows_in - 1);
  out[i] = x[static_cast<std::size_t>(source) * width + i % width];
}

// Upsampled value u[m] (2x transposed conv of the replicate-padded input,
// times 2, cropped by 15) of channel c.
__device__ __forceinline__ float Upsampled(const float* x, int rows,
                                           int channels, int c, int m,
                                           const float* filter) {
  float sum = 0;
  // k with (m + 15 - k) even.
  for (int k = (m + 15) & 1; k < 12; k += 2) {
    const int i = (m + 15 - k) / 2;  // index into the 5-padded input
    const int source = min(max(i - 5, 0), rows - 1);
    sum += filter[k] * x[static_cast<std::size_t>(source) * channels + c];
  }
  return 2.0F * sum;
}

__global__ void SnakeAntiAliasKernel(const float* x, int rows, int channels,
                                     const float* alpha, const float* inv_beta,
                                     const float* filter, std::size_t count,
                                     float* out) {
  __shared__ float taps[12];
  if (threadIdx.x < 12)
    taps[threadIdx.x] = filter[threadIdx.x];
  __syncthreads();
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int t = static_cast<int>(i / channels);
  const int c = static_cast<int>(i % channels);
  const float a = alpha[c];
  const float b = inv_beta[c];
  const int up_rows = 2 * rows;
  float sum = 0;
  for (int k = 0; k < 12; ++k) {
    const int m = min(max(2 * t + k - 5, 0), up_rows - 1);
    const float u = Upsampled(x, rows, channels, c, m, taps);
    const float s = sinf(u * a);
    sum += taps[k] * (u + b * s * s);
  }
  out[i] = sum;
}

// ---- patch embedding inputs ----------------------------------------------------

__global__ void Patchify2dKernel(const float* pixels, int size, int patch,
                                 int grid, std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int values = 3 * patch * patch;
  const std::size_t token = i / values;
  const int column = static_cast<int>(i % values);
  const std::size_t image = token / (grid * grid);
  const int cell = static_cast<int>(token % (grid * grid));
  const int gy = cell / grid, gx = cell % grid;
  const int c = column / (patch * patch);
  const int py = (column / patch) % patch;
  const int px = column % patch;
  out[i] = pixels[((image * 3 + c) * size + gy * patch + py) * size +
                  gx * patch + px];
}

__global__ void Patchify3dKernel(const float* frames, int step,
                                 std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  constexpr int kValues = 3 * 2 * 16 * 16;
  constexpr int kTokens = 8 * 14 * 14;
  const std::size_t token = i / kValues;
  const int column = static_cast<int>(i % kValues);
  const std::size_t segment = token / kTokens;
  const int local = static_cast<int>(token % kTokens);
  const int t = local / 196;
  const int gy = (local / 14) % 14;
  const int gx = local % 14;
  const int c = column / 512;
  const int dt = (column / 256) % 2;
  const int py = (column / 16) % 16;
  const int px = column % 16;
  const std::size_t frame = segment * step + 2 * t + dt;
  out[i] = frames[((frame * 3 + c) * 224 + gy * 16 + py) * 224 + gx * 16 + px];
}

// ---- attention helpers ----------------------------------------------------------

// One block per row; threads loop over the heads * head_dim columns.
__global__ void SplitQkvKernel(const float* qkv, int heads, int head_dim,
                               const float* q_norm, const float* k_norm,
                               const float* rope, float* q, float* k, float* v,
                               int out_stride) {
  extern __shared__ float row_q[];
  float* row_k = row_q + heads * head_dim;
  __shared__ float inverse[2 * 64];  // per head: q, k (heads <= 64)
  const int r = blockIdx.x;
  const int width = heads * head_dim;
  const float* source = qkv + static_cast<std::size_t>(r) * 3 * width;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    row_q[c] = source[c * 3 + 0];
    row_k[c] = source[c * 3 + 1];
    v[static_cast<std::size_t>(r) * out_stride + c] = source[c * 3 + 2];
  }
  __syncthreads();
  // Per-head mean square (one thread per head and tensor).
  if (threadIdx.x < 2 * heads) {
    const int h = threadIdx.x % heads;
    const float* data = threadIdx.x < heads ? row_q : row_k;
    float sum = 0;
    for (int d = 0; d < head_dim; ++d)
      sum += data[h * head_dim + d] * data[h * head_dim + d];
    inverse[threadIdx.x] =
        rsqrtf(sum / static_cast<float>(head_dim) + 1.1920929e-07F);
  }
  __syncthreads();
  const int half = head_dim / 2;
  for (int pair = threadIdx.x; pair < heads * half; pair += blockDim.x) {
    const int h = pair / half;
    const int i = pair % half;
    const int c0 = h * head_dim + 2 * i;
    float q0 = row_q[c0] * inverse[h] * q_norm[2 * i];
    float q1 = row_q[c0 + 1] * inverse[h] * q_norm[2 * i + 1];
    float k0 = row_k[c0] * inverse[heads + h] * k_norm[2 * i];
    float k1 = row_k[c0 + 1] * inverse[heads + h] * k_norm[2 * i + 1];
    if (rope != nullptr) {
      const float* rot =
          rope + (static_cast<std::size_t>(r) * half + i) * 2;
      const float cosine = rot[0], sine = rot[1];
      const float rq0 = cosine * q0 - sine * q1;
      const float rq1 = sine * q0 + cosine * q1;
      const float rk0 = cosine * k0 - sine * k1;
      const float rk1 = sine * k0 + cosine * k1;
      q0 = rq0, q1 = rq1, k0 = rk0, k1 = rk1;
    }
    const std::size_t base = static_cast<std::size_t>(r) * out_stride + c0;
    q[base] = q0;
    q[base + 1] = q1;
    k[base] = k0;
    k[base + 1] = k1;
  }
}

__global__ void VaeQkvKernel(const float* qkv, int channels, float* q,
                             float* k, float* v) {
  __shared__ float shared[kThreads];
  const std::size_t r = blockIdx.x;
  const float* row = qkv + r * 3 * channels;
  float* targets[3] = {q, k, v};
  for (int j = 0; j < 3; ++j) {
    float squares = 0;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      const float value = row[c * 3 + j];
      squares += value * value;
    }
    const float norm = sqrtf(BlockSum(squares, shared));
    const float inverse =
        1.0F / (1e-4F + norm * rsqrtf(static_cast<float>(channels)));
    for (int c = threadIdx.x; c < channels; c += blockDim.x)
      targets[j][r * channels + c] = row[c * 3 + j] * inverse;
  }
}

// One block of 64 threads per (row, head): online softmax over the row's
// keys in chunks of 64.
__global__ void SubsetAttentionKernel(const float* q, const float* k,
                                      const float* v, int stride, float* out,
                                      int heads, int group, int frames,
                                      int patches, int keys_kind,
                                      float scale) {
  constexpr int kDim = 64;
  __shared__ float query[kDim];
  __shared__ float probs[kDim];
  __shared__ int index[kDim];
  __shared__ float reduce[kDim];
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int base = (row / group) * group;
  const int local = row % group;
  query[tid] = q[static_cast<std::size_t>(row) * stride + h * kDim + tid] *
               scale;
  // Key list of this row.
  int count;
  int f = 0, n = 0;
  if (keys_kind == 0 || local == 0) {
    count = group;
  } else {
    f = (local - 1) / patches;
    n = (local - 1) % patches;
    count = 1 + (keys_kind == 1 ? frames : patches);
  }
  __syncthreads();
  float running_max = -INFINITY;
  float running_sum = 0;
  float acc = 0;
  for (int start = 0; start < count; start += kDim) {
    const int j = start + tid;
    int key = -1;
    if (j < count) {
      if (keys_kind == 0 || local == 0)
        key = base + j;
      else if (j == 0)
        key = base;
      else if (keys_kind == 1)
        key = base + 1 + (j - 1) * patches + n;
      else
        key = base + 1 + f * patches + (j - 1);
    }
    float score = -INFINITY;
    if (key >= 0) {
      const float* kr = k + static_cast<std::size_t>(key) * stride + h * kDim;
      float dot = 0;
      for (int d = 0; d < kDim; ++d)
        dot += query[d] * kr[d];
      score = dot;
    }
    index[tid] = key;
    reduce[tid] = score;
    __syncthreads();
    for (int s = kDim / 2; s > 0; s /= 2) {
      if (tid < s)
        reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
      __syncthreads();
    }
    const float chunk_max = reduce[0];
    __syncthreads();
    const float new_max = fmaxf(running_max, chunk_max);
    const float p = key >= 0 ? expf(score - new_max) : 0.0F;
    probs[tid] = p;
    reduce[tid] = p;
    __syncthreads();
    for (int s = kDim / 2; s > 0; s /= 2) {
      if (tid < s)
        reduce[tid] += reduce[tid + s];
      __syncthreads();
    }
    const float correction =
        running_max == -INFINITY ? 0.0F : expf(running_max - new_max);
    running_sum = running_sum * correction + reduce[0];
    acc *= correction;
    const int valid = min(kDim, count - start);
    for (int jj = 0; jj < valid; ++jj)
      acc += probs[jj] *
             v[static_cast<std::size_t>(index[jj]) * stride + h * kDim + tid];
    running_max = new_max;
    __syncthreads();
  }
  out[static_cast<std::size_t>(row) * heads * kDim + h * kDim + tid] =
      acc / running_sum;
}

}  // namespace

void LaunchToHalf(const float* in, std::size_t count, std::uint16_t* out,
                  hipStream_t stream) {
  if (count > 0)
    ToHalfKernel<<<Blocks(count), kThreads, 0, stream>>>(in, count, out);
}

void LaunchWiden(const std::uint16_t* in, std::size_t rows, int width,
                 const float* bias, bool accumulate, float* out,
                 hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    WidenKernel<<<Blocks(count), kThreads, 0, stream>>>(in, count, width, bias,
                                                        accumulate, out);
}

void LaunchAddBias(const float* in, std::size_t rows, int width,
                   const float* bias, bool accumulate, float* out,
                   hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    AddBiasKernel<<<Blocks(count), kThreads, 0, stream>>>(
        in, count, width, bias, accumulate, out);
}

void LaunchSoftmax(float* scores, std::size_t rows, int columns, float scale,
                   int causal, hipStream_t stream) {
  if (rows > 0)
    SoftmaxKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        scores, columns, scale, causal);
}

void LaunchLayerNorm(const float* x, std::size_t rows, int width,
                     const float* weight, const float* bias, float eps,
                     const float* shift, const float* scale, int mod_stride,
                     float* out, hipStream_t stream) {
  if (rows > 0)
    LayerNormKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        x, width, weight, bias, eps, shift, scale, mod_stride, out);
}

void LaunchL2Normalize(float* x, std::size_t rows, int width,
                       hipStream_t stream) {
  if (rows > 0)
    L2NormalizeKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        x, width);
}

void LaunchPixelNorm(const float* x, std::size_t rows, int width, float* out,
                     hipStream_t stream) {
  if (rows > 0)
    PixelNormKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        x, width, out);
}

void LaunchSilu(float* x, std::size_t count, float scale, hipStream_t stream) {
  if (count > 0)
    SiluKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, scale);
}

void LaunchSiluTo(const float* x, std::size_t count, float* out,
                  hipStream_t stream) {
  if (count > 0)
    SiluToKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, out);
}

void LaunchGelu(float* x, std::size_t count, hipStream_t stream) {
  if (count > 0)
    GeluKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count);
}

void LaunchQuickGelu(float* x, std::size_t count, hipStream_t stream) {
  if (count > 0)
    QuickGeluKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count);
}

void LaunchSwiGlu(const float* x, std::size_t rows, int hidden, float* out,
                  hipStream_t stream) {
  const std::size_t count = rows * hidden;
  if (count > 0)
    SwiGluKernel<<<Blocks(count), kThreads, 0, stream>>>(x, rows, hidden, out);
}

void LaunchAddGated(float* x, const float* y, std::size_t rows, int width,
                    const float* gate, int gate_stride, hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    AddGatedKernel<<<Blocks(count), kThreads, 0, stream>>>(x, y, rows, width,
                                                           gate, gate_stride);
}

void LaunchMpSum(const float* a, const float* b, std::size_t count, float t,
                 float* out, hipStream_t stream) {
  const float norm = std::sqrt((1.0F - t) * (1.0F - t) + t * t);
  if (count > 0)
    MpSumKernel<<<Blocks(count), kThreads, 0, stream>>>(
        a, b, count, (1.0F - t) / norm, t / norm, out);
}

void LaunchClamp(float* x, std::size_t count, float low, float high,
                 hipStream_t stream) {
  if (count > 0)
    ClampKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, low, high);
}

void LaunchAffine(float* x, std::size_t count, float scale, float add,
                  hipStream_t stream) {
  if (count > 0)
    AffineKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, scale, add);
}

void LaunchColumnAffine(float* x, std::size_t rows, int width,
                        const float* mul, const float* add,
                        hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    ColumnAffineKernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, width,
                                                               mul, add);
}

void LaunchAdd(float* x, const float* y, std::size_t count,
               hipStream_t stream) {
  if (count > 0)
    AddKernel<<<Blocks(count), kThreads, 0, stream>>>(x, y, count);
}

void LaunchAddRow(float* x, const float* v, std::size_t rows, int width,
                  hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    AddRowKernel<<<Blocks(count), kThreads, 0, stream>>>(x, v, count, width);
}

void LaunchRepeatRows(const float* in, int period, std::size_t rows, int width,
                      float* out, hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    RepeatRowsKernel<<<Blocks(count), kThreads, 0, stream>>>(in, period, count,
                                                             width, out);
}

void LaunchColumnMean(const float* x, std::size_t rows, int width, float* out,
                      hipStream_t stream) {
  if (width > 0)
    ColumnMeanKernel<<<Blocks(width), kThreads, 0, stream>>>(x, rows, width,
                                                             out);
}

void LaunchGatherRows(const float* table, const int* ids, std::size_t rows,
                      int width, float* out, hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count > 0)
    GatherRowsKernel<<<Blocks(count), kThreads, 0, stream>>>(table, ids, count,
                                                             width, out);
}

void LaunchIm2Col(const float* x, int rows_in, int channels, int kernel,
                  int dilation, int padding, int rows_out, float* out,
                  hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(rows_out) * channels * kernel;
  if (count > 0)
    Im2ColKernel<<<Blocks(count), kThreads, 0, stream>>>(
        x, rows_in, channels, kernel, dilation, padding, count, out);
}

void LaunchCol2ImTransposed(const float* y, int rows_in, int channels,
                            int kernel, int stride, int padding,
                            const float* bias, int rows_out, float* out,
                            hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(rows_out) * channels;
  if (count > 0)
    Col2ImTransposedKernel<<<Blocks(count), kThreads, 0, stream>>>(
        y, rows_in, channels, kernel, stride, padding, bias, count, out);
}

void LaunchUpsampleNearest(const float* x, int rows, int width, int factor,
                           float* out, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(rows) * factor * width;
  if (count > 0)
    UpsampleNearestKernel<<<Blocks(count), kThreads, 0, stream>>>(
        x, width, factor, count, out);
}

void LaunchInterpolateNearestExact(const float* x, int rows_in, int width,
                                   int rows_out, float* out,
                                   hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(rows_out) * width;
  if (count > 0)
    NearestExactKernel<<<Blocks(count), kThreads, 0, stream>>>(
        x, rows_in, width, rows_out, count, out);
}

void LaunchSnakeAntiAlias(const float* x, int rows, int channels,
                          const float* alpha, const float* inv_beta,
                          const float* filter, float* out,
                          hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(rows) * channels;
  if (count > 0)
    SnakeAntiAliasKernel<<<Blocks(count), kThreads, 0, stream>>>(
        x, rows, channels, alpha, inv_beta, filter, count, out);
}

void LaunchPatchify2d(const float* pixels, int images, int size, int patch,
                      int grid, float* out, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(images) * grid * grid *
                            3 * patch * patch;
  if (count > 0)
    Patchify2dKernel<<<Blocks(count), kThreads, 0, stream>>>(
        pixels, size, patch, grid, count, out);
}

void LaunchPatchify3d(const float* frames, int segments, int step, float* out,
                      hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(segments) * (8 * 14 * 14) * (3 * 2 * 16 * 16);
  if (count > 0)
    Patchify3dKernel<<<Blocks(count), kThreads, 0, stream>>>(frames, step,
                                                             count, out);
}

void LaunchSplitQkv(const float* qkv, int rows, int heads, int head_dim,
                    const float* q_norm, const float* k_norm,
                    const float* rope, float* q, float* k, float* v,
                    int out_stride, hipStream_t stream) {
  if (rows <= 0)
    return;
  const std::size_t shared =
      static_cast<std::size_t>(2) * heads * head_dim * sizeof(float);
  SplitQkvKernel<<<static_cast<unsigned>(rows), kThreads, shared, stream>>>(
      qkv, heads, head_dim, q_norm, k_norm, rope, q, k, v, out_stride);
}

void LaunchVaeQkv(const float* qkv, int rows, int channels, float* q,
                  float* k, float* v, hipStream_t stream) {
  if (rows > 0)
    VaeQkvKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        qkv, channels, q, k, v);
}

void LaunchSubsetAttention(const float* q, const float* k, const float* v,
                           int stride, float* out, int rows, int heads,
                           int group, int frames, int patches, KeySet keys,
                           float scale, hipStream_t stream) {
  if (rows <= 0)
    return;
  const int kind = keys == KeySet::kFull ? 0 : keys == KeySet::kTime ? 1 : 2;
  SubsetAttentionKernel<<<dim3(static_cast<unsigned>(rows),
                               static_cast<unsigned>(heads)),
                          64, 0, stream>>>(q, k, v, stride, out, heads, group,
                                           frames, patches, kind, scale);
}

}  // namespace gufo::models::mmaudio
