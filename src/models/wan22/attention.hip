// WMMA attention for the Wan 2.2 transformer, adapted from the long-sequence
// kernel in src/models/minimax_h3/attention.hip: separate query and key
// lengths (self- and cross-attention) and launches split by query blocks.
#include <algorithm>
#include <cstddef>

#include "src/models/wan22/attention.hip.hpp"

namespace gufo::models::wan22 {
namespace {

using BufferResource = __attribute__((ext_vector_type(4))) unsigned;
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

constexpr int kHeadDim = 128;
constexpr int kQueriesPerBlock = 128;
constexpr std::uint64_t kMaxPairsPerLaunch = std::uint64_t{1} << 29U;

}  // namespace

__device__ BufferResource
WanLoadBuffer(BufferResource, unsigned, unsigned,
              unsigned) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

namespace {

__device__ BufferResource MakeBuffer(const std::uint16_t* pointer,
                                     unsigned bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return {unsigned(address), unsigned(address >> 32), bytes, 0x31004000U};
}

__device__ uint4 LoadBufferVector(BufferResource resource, unsigned offset) {
  union {
    BufferResource vector;
    uint4 packed;
  } result;
  result.vector = WanLoadBuffer(resource, offset, 0, 0);
  return result.packed;
}

__device__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__device__ std::uint16_t Bf16(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned nan = 0U - unsigned((bits & 0x7fffffffU) > 0x7f800000U);
  const unsigned rounded = bits + 0x7fffU + ((bits >> 16U) & 1U);
  return ((rounded & ~nan) | ((bits | 0x00400000U) & nan)) >> 16U;
}

__global__ void PackValues(const std::uint16_t* input, std::uint16_t* output,
                           unsigned rows) {
  __shared__ __align__(16) std::uint16_t tile[32][136];
  const unsigned tid = threadIdx.x;
  const unsigned padded_rows = PaddedRows(rows);
  input += std::size_t(blockIdx.y) * rows * kHeadDim;
  output += std::size_t(blockIdx.y) * padded_rows * kHeadDim;
#pragma unroll
  for (unsigned z = 0; z < 2; ++z) {
    const unsigned r = tid / 16 + z * 16, d = (tid % 16) * 8;
    uint4 x{};
    if (blockIdx.x * 32 + r < rows)
      x = *reinterpret_cast<const uint4*>(
          input + (std::size_t(blockIdx.x) * 32 + r) * kHeadDim + d);
    *reinterpret_cast<uint4*>(tile[r] + d) = x;
  }
  __syncthreads();
#pragma unroll
  for (unsigned z = 0; z < 2; ++z) {
    const unsigned d = tid / 4 + z * 64, r = (tid % 4) * 8;
    union {
      uint4 packed;
      std::uint16_t half[8];
    } value;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
      value.half[i] = tile[r + i][d];
    *reinterpret_cast<uint4*>(output + std::size_t(d) * padded_rows +
                              blockIdx.x * 32 + r) = value.packed;
  }
}

// Eight waves own sixteen queries each; every 32-key tile of K and then V is
// staged once in LDS for all of them. Online base-two softmax in FP32.
__global__ __launch_bounds__(256, 1) void Attention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t query_rows,
    std::uint32_t key_rows, std::uint32_t first_block) {
  union Staging {
    std::uint16_t key[32][136];
    std::uint16_t value[128][40];
  };
  __shared__ Staging staging;
  auto& staged_key = staging.key;
  auto& staged_value = staging.value;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t heads = gridDim.y;
  const int first_query =
      (first_block + blockIdx.x) * kQueriesPerBlock + wave * 16;
  query += std::size_t(blockIdx.y) * query_rows * kHeadDim;
  key += std::size_t(blockIdx.y) * key_rows * kHeadDim;
  const unsigned padded_rows = PaddedRows(key_rows);
  value += std::size_t(blockIdx.y) * padded_rows * kHeadDim;
  // Bounded descriptors read zeros past the last key of this head.
  const auto key_buffer = MakeBuffer(key, key_rows * 256);
  const auto value_buffer = MakeBuffer(value, padded_rows * 256);
  Fragment q[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    q[d] = {};
    if (static_cast<std::uint32_t>(first_query + lane % 16) < query_rows)
      q[d] = LoadFragment(query + std::size_t(first_query + lane % 16) * 128 +
                          d * 16);
  }
  Accumulator result[8]{};
  float maximum = -INFINITY, sum = 0;

  for (std::uint32_t base = 0; base < key_rows; base += 32) {
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int kr = tid / 16 + z * 16, d = (tid % 16) * 8;
      const unsigned offset = ((base + kr) * 128 + d) * 2;
      *reinterpret_cast<uint4*>(staged_key[kr] + d) =
          LoadBufferVector(key_buffer, offset);
    }
    __syncthreads();
    Accumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const Fragment k =
            LoadFragment(staged_key[half * 16 + lane % 16] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[d], scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int vd = tid / 4 + z * 64, vk = (tid % 4) * 8;
      *reinterpret_cast<uint4*>(staged_value[vd] + vk) =
          LoadBufferVector(value_buffer, (vd * padded_rows + base + vk) * 2);
    }
    if (base + 32 > key_rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (base + half * 16 + 2 * i + lane / 16 >= key_rows)
            scores[half][i] = -INFINITY;
    }
    float next = maximum;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // log2(e) / sqrt(128), for the base-two online softmax.
        scores[half][i] *= 0.12751743082459868F;
        next = fmaxf(next, scores[half][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(__builtin_amdgcn_permlanex16(
                           __float_as_uint(next), __float_as_uint(next),
                           0x76543210U, 0xfedcba98U, false, false)));
    __syncthreads();
    const float correction = __builtin_amdgcn_exp2f(maximum - next);
    float subtotal = 0;
    Fragment probability_fragments[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(scores[half][i] - next);
        subtotal += p;
        const unsigned own = Bf16(p);
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probability_fragments[half][2 * i] = lane < 16 ? own : peer;
        probability_fragments[half][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += __uint_as_float(__builtin_amdgcn_permlanex16(
        __float_as_uint(subtotal), __float_as_uint(subtotal), 0x76543210U,
        0xfedcba98U, false, false));
    sum = sum * correction + subtotal;
    maximum = next;
#pragma unroll
    for (int d = 0; d < 8; ++d)
      result[d] *= correction;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const Fragment p = probability_fragments[half];
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        const Fragment v =
            LoadFragment(staged_value[d * 16 + lane % 16] + half * 16);
        result[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, result[d]);
        if (d % 2 == 1)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
    __syncthreads();
  }
  const std::uint32_t row = first_query + lane % 16;
  if (row < query_rows) {
    const float reciprocal = __builtin_amdgcn_rcpf(sum);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[(std::size_t(row) * heads + blockIdx.y) * 128 + d * 16 + 2 * i +
               lane / 16] = Bf16(result[d][i] * reciprocal);
    }
  }
}

}  // namespace

void LaunchPackValues(const std::uint16_t* values, std::uint16_t* packed,
                      std::uint32_t rows, std::uint32_t heads,
                      hipStream_t stream) {
  PackValues<<<dim3((rows + 31) / 32, heads), 256, 0, stream>>>(values, packed,
                                                                rows);
}

void LaunchAttention(const std::uint16_t* query, const std::uint16_t* key,
                     const std::uint16_t* packed_values, std::uint16_t* output,
                     std::uint32_t query_rows, std::uint32_t key_rows,
                     std::uint32_t heads, hipStream_t stream) {
  const std::uint32_t blocks =
      (query_rows + kQueriesPerBlock - 1) / kQueriesPerBlock;
  const std::uint64_t pairs_per_block =
      std::uint64_t{kQueriesPerBlock} * key_rows * heads;
  const auto per_launch = static_cast<std::uint32_t>(std::max<std::uint64_t>(
      1, kMaxPairsPerLaunch / std::max<std::uint64_t>(1, pairs_per_block)));
  for (std::uint32_t first = 0; first < blocks; first += per_launch) {
    const std::uint32_t count = std::min(per_launch, blocks - first);
    Attention<<<dim3(count, heads), 256, 0, stream>>>(
        query, key, packed_values, output, query_rows, key_rows, first);
  }
}

}  // namespace gufo::models::wan22
