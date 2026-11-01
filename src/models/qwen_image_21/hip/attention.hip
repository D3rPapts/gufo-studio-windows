#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen_image_21/hip/bf16.hpp"
#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {

using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ Fragment Load(const std::uint16_t* x) {
  union {
    Fragment b;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(x);
  value.u[1] = *reinterpret_cast<const uint4*>(x + 8);
  return value.b;
}

// One head's key/value segments, copied out of the kernel arguments once.
// Row r of segment i is at keys[i] + (shift[i] + r) * 128. Selecting among
// registers avoids per-row loads of the argument struct, which a per-lane
// segment index otherwise turns into dependent global loads.
struct HeadSegments {
  const std::uint16_t* keys[3];
  const std::uint16_t* values[3];
  std::ptrdiff_t shift[3];
  int end[2];
};

__device__ HeadSegments ForHead(const KeySegments& s, int head) {
  HeadSegments h;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    h.keys[i] = s.keys[i];
    h.values[i] = s.values[i];
    h.shift[i] = static_cast<std::ptrdiff_t>(head) * s.stride[i] -
                 (i == 0 ? 0 : s.end[i - 1]);
  }
  h.end[0] = s.end[0];
  h.end[1] = s.end[1];
  return h;
}

// Selects among values, not among array elements: a select of element
// addresses becomes a runtime index into the (then spilled) arrays.
template<typename T>
__device__ T Pick(int row, int end0, int end1, T a, T b, T c) {
  return row < end0 ? a : (row < end1 ? b : c);
}

template<bool Value>
__device__ const std::uint16_t* SegmentRow(const HeadSegments& h, int row) {
  const std::uint16_t* base =
      Value ? Pick(row, h.end[0], h.end[1], h.values[0], h.values[1],
                   h.values[2])
            : Pick(row, h.end[0], h.end[1], h.keys[0], h.keys[1], h.keys[2]);
  const std::ptrdiff_t shift =
      Pick(row, h.end[0], h.end[1], h.shift[0], h.shift[1], h.shift[2]);
  return base + (shift + row) * 128;
}

// Row `row` when rows [row, row + 32) share one segment (the last segment
// also covers rows past the key count, which callers never read), else
// nullptr. `row` is uniform, so the common case is one scalar pointer.
template<bool Value>
__device__ const std::uint16_t* BlockRows(const HeadSegments& h, int row) {
  if ((row < h.end[0] && row + 32 > h.end[0]) ||
      (row >= h.end[0] && row < h.end[1] && row + 32 > h.end[1]))
    return nullptr;
  return SegmentRow<Value>(h, row);
}

// Online softmax with BF16 QK matrix instructions and an FP32 value product.
// No probability narrowing and no global score/probability allocation.
// Each block owns 128 queries by all 128 output channels, consuming 32 keys at
// a time. Each wave owns 16 queries; their Q fragments stay in registers for
// the whole key sequence. A tile's 32 value rows are staged at once, so a tile
// needs three barriers. Arithmetic and its order equal the former version
// (tests/models/qwen_image_21/attention_reference.hip): bit-identical output.
// Segmented reads keys and values from `segments` (cached condition rows
// and the new rows) instead of one contiguous head-major copy.
template<bool Masked, bool Segmented>
__device__ __forceinline__ void
FusedAttentionBody(const std::uint16_t* queries, const std::uint16_t* keys,
                   const std::uint16_t* values, hip_bfloat16* output,
                   int heads, int rows, int count, int query_start,
                   int query_count, const int* limits,
                   const KeySegments& segments) {
  __shared__ float p[32][129], v[32][129], stats[128][3];
  __shared__ int last_key;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int col = blockIdx.x * 128, head = blockIdx.y;
  const int query_end = query_start + query_count;
  queries += static_cast<std::size_t>(head) * rows * 128;
  [[maybe_unused]] HeadSegments segs{};
  if constexpr (Segmented) {
    segs = ForHead(segments, head);
  } else {
    keys += static_cast<std::size_t>(head) * count * 128;
    values += static_cast<std::size_t>(head) * count * 128;
  }
  // Segmented: stage value rows [first, first + 32) (first is uniform).
  const auto stage = [&](int first, int z) -> unsigned short {
    const int i = tid + z * 256;
    return first + i / 128 < count
               ? SegmentRow<true>(segs, first + i / 128)[i % 128]
               : 0;
  };
  // Value rows [first, first + 32) into registers, element tid + z * 256.
  const auto fetch = [&](int first, unsigned short (&out)[16]) {
    if constexpr (Segmented) {
      if (const auto* rows = BlockRows<true>(segs, first)) {
#pragma unroll
        for (int z = 0; z < 16; ++z) {
          const int i = tid + z * 256;
          out[z] = first + i / 128 < count ? rows[i] : 0;
        }
      } else {
#pragma unroll
        for (int z = 0; z < 16; ++z)
          out[z] = stage(first, z);
      }
    } else {
#pragma unroll
      for (int z = 0; z < 16; ++z) {
        const int i = tid + z * 256;
        out[z] = first + i / 128 < count
                     ? values[std::size_t(first + i / 128) * 128 + i % 128]
                     : 0;
      }
    }
  };
  if (tid < 128) {
    stats[tid][0] = -INFINITY;
    stats[tid][1] = 0;
  }
  if constexpr (Masked) {
    if (tid < 32) {
      int visible = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int q = query_start + col + tid + i * 32;
        if (q < query_end)
          visible = max(visible, limits[q]);
      }
#pragma unroll
      for (int offset = 16; offset; offset /= 2)
        visible = max(visible, __shfl_xor(visible, offset, 32));
      if (tid == 0)
        last_key = visible;
    }
  }
  // This wave's 16 query rows, loaded once.
  const int qr = query_start + col + wave * 16 + lane % 16;
  Fragment query[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    query[d] = Fragment{};
    if (qr < query_end)
      query[d] = Load(queries + static_cast<std::size_t>(qr) * 128 + d * 16);
  }
  __syncthreads();
  float accum[8][8]{};
  const int visible = Masked ? last_key : count;
  // Keep the next tile in registers while QK, softmax and FP32 PV execute.
  unsigned short prefetch_v[16];
  fetch(0, prefetch_v);
  for (int base = 0; base < visible; base += 32) {
    Accumulator dot[2]{};
    [[maybe_unused]] const std::uint16_t* key_row[2]{};
    const int kr[2] = {base + lane % 16, base + 16 + lane % 16};
    if constexpr (Segmented) {
      const auto* block = BlockRows<false>(segs, base);
#pragma unroll
      for (int t = 0; t < 2; ++t)
        key_row[t] = block ? block + static_cast<std::size_t>(kr[t] - base) * 128
                           : SegmentRow<false>(segs, kr[t]);
    }
#pragma unroll
    for (int d = 0; d < 8; ++d)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        Fragment k{};
        if (kr[t] < count) {
          if constexpr (Segmented)
            k = Load(key_row[t] + d * 16);
          else
            k = Load(keys + static_cast<std::size_t>(kr[t]) * 128 + d * 16);
        }
        dot[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(query[d], k,
                                                             dot[t]);
      }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int q = wave * 16 + 2 * i + lane / 16;
        const int k = t * 16 + lane % 16;
        int limit = count;
        if constexpr (Masked) {
          const int position = query_start + col + q;
          if (position < query_end)
            limit = limits[position];
        }
        p[k][q] =
            base + k < limit ? dot[t][i] * 0.08838834764831845F : -INFINITY;
      }
    __syncthreads();
    for (int qoff = 0; qoff < 128; qoff += 64) {
      const int q = qoff + tid / 4, l = tid % 4;
      float maximum = -INFINITY;
#pragma unroll
      for (int k = l; k < 32; k += 4)
        maximum = fmaxf(maximum, p[k][q]);
      maximum = fmaxf(maximum, __shfl_xor(maximum, 2, 4));
      maximum = fmaxf(maximum, __shfl_xor(maximum, 1, 4));
      const float previous = stats[q][0];
      maximum = fmaxf(maximum, previous);
      const float correction = expf(previous - maximum);
      float sum = 0;
#pragma unroll
      for (int k = l; k < 32; k += 4) {
        const float value = expf(p[k][q] - maximum);
        p[k][q] = value;
        sum += value;
      }
      sum += __shfl_xor(sum, 2, 4);
      sum += __shfl_xor(sum, 1, 4);
      if (l == 0) {
        stats[q][0] = maximum;
        stats[q][1] = stats[q][1] * correction + sum;
        stats[q][2] = correction;
      }
    }
    // The previous tile's value reads ended at its last barrier.
#pragma unroll
    for (int z = 0; z < 16; z++) {
      const int i = tid + z * 256;
      v[i / 128][i % 128] = __uint_as_float(unsigned(prefetch_v[z]) << 16);
    }
    fetch(base + 32, prefetch_v);
    __syncthreads();
#pragma unroll
    for (int y = 0; y < 8; y++)
#pragma unroll
      for (int x = 0; x < 8; x++)
        accum[y][x] *= stats[tid / 16 + y * 16][2];
#pragma unroll 2
    for (int k = 0; k < 32; k++)
#pragma unroll
      for (int y = 0; y < 8; y++) {
        float probability = p[k][tid / 16 + y * 16];
#pragma unroll
        for (int x = 0; x < 8; x++)
          accum[y][x] = fmaf(v[k][tid % 16 + x * 16], probability, accum[y][x]);
      }
    __syncthreads();
  }
#pragma unroll
  for (int y = 0; y < 8; ++y)
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      const int query = query_start + col + tid / 16 + y * 16;
      if (query < query_end)
        output[(static_cast<std::size_t>(query) * heads + head) * 128 +
               tid % 16 + x * 16] =
            Bf16(accum[y][x] / stats[tid / 16 + y * 16][1]);
    }
}

template<bool Masked>
__global__ void FusedAttention(const std::uint16_t* queries,
                               const std::uint16_t* keys,
                               const std::uint16_t* values,
                               hip_bfloat16* output, int heads, int rows,
                               int count, int query_start, int query_count,
                               const int* limits) {
  FusedAttentionBody<Masked, false>(queries, keys, values, output, heads, rows,
                                    count, query_start, query_count, limits,
                                    {});
}

// The segment selection needs a few more registers; without the bound the
// masked variant spills under the default 1024-thread budget.
template<bool Masked>
__global__ __launch_bounds__(256) void FusedAttentionSegmented(
    const std::uint16_t* queries, KeySegments segments, hip_bfloat16* output,
    int heads, int rows, int query_start, int query_count, const int* limits) {
  FusedAttentionBody<Masked, true>(queries, nullptr, nullptr, output, heads,
                                   rows, segments.end[2], query_start,
                                   query_count, limits, segments);
}
}  // namespace

void FusedAttention128(const std::uint16_t* queries, const std::uint16_t* keys,
                       const std::uint16_t* values, std::uint16_t* output,
                       int heads, int rows, int count, int query_start,
                       hipStream_t stream, int query_count, const int* limits) {
  const dim3 grid((query_count + 127) / 128, heads);
  if (limits)
    FusedAttention<true><<<grid, 256, 0, stream>>>(
        queries, keys, values, reinterpret_cast<hip_bfloat16*>(output), heads,
        rows, count, query_start, query_count, limits);
  else
    FusedAttention<false><<<grid, 256, 0, stream>>>(
        queries, keys, values, reinterpret_cast<hip_bfloat16*>(output), heads,
        rows, count, query_start, query_count, nullptr);
}

void FusedAttention128Segmented(const std::uint16_t* queries,
                                const KeySegments& segments,
                                std::uint16_t* output, int heads, int rows,
                                int query_start, int query_count,
                                hipStream_t stream, const int* limits) {
  const dim3 grid((query_count + 127) / 128, heads);
  if (limits)
    FusedAttentionSegmented<true><<<grid, 256, 0, stream>>>(
        queries, segments, reinterpret_cast<hip_bfloat16*>(output), heads,
        rows, query_start, query_count, limits);
  else
    FusedAttentionSegmented<false><<<grid, 256, 0, stream>>>(
        queries, segments, reinterpret_cast<hip_bfloat16*>(output), heads,
        rows, query_start, query_count, nullptr);
}

}  // namespace gufo::models::qwen_image_21::hip
