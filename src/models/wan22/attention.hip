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
using HalfFragment = __attribute__((ext_vector_type(16))) _Float16;
using Words = __attribute__((ext_vector_type(8))) unsigned;
using Accumulator = __attribute__((ext_vector_type(8))) float;

constexpr int kHeadDim = 128;
constexpr int kQueriesPerBlock = 128;
// About 30 ms per launch on gfx1151, far below Windows' 2 s GPU watchdog.
constexpr std::uint64_t kMaxPairsPerLaunch = std::uint64_t{1} << 31U;
constexpr float kScale = 0.12751743082459868F;  // log2(e) / sqrt(128)
// The running output is rescaled only when a row maximum grew by more than
// this many powers of two, so probabilities stay below 2^kSlack (FP16-safe).
constexpr float kSlack = 8.0F;

}  // namespace

__device__ BufferResource
WanLoadBuffer(BufferResource, unsigned, unsigned,
              unsigned) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

namespace {

__device__ __forceinline__ BufferResource
MakeBuffer(const std::uint16_t* pointer, unsigned bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return {unsigned(address), unsigned(address >> 32), bytes, 0x31004000U};
}

__device__ __forceinline__ uint4 LoadBufferVector(BufferResource resource,
                                                  unsigned offset) {
  union {
    BufferResource vector;
    uint4 packed;
  } result;
  result.vector = WanLoadBuffer(resource, offset, 0, 0);
  return result.packed;
}

__device__ __forceinline__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__device__ __forceinline__ std::uint16_t Bf16(float value) {
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
    for (unsigned i = 0; i < 8; ++i) {
      const float x = __uint_as_float(unsigned(tile[r + i][d]) << 16U);
      value.half[i] = __builtin_bit_cast(
          std::uint16_t,
          static_cast<_Float16>(fminf(fmaxf(x, -65504.0F), 65504.0F)));
    }
    *reinterpret_cast<uint4*>(output + std::size_t(d) * padded_rows +
                              blockIdx.x * 32 + r) = value.packed;
  }
}

// Packs two probabilities as FP16 (round to nearest even).
__device__ __forceinline__ unsigned PackHalves(float low, float high) {
  const unsigned lo =
      __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(low));
  const unsigned hi =
      __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(high));
  return lo | (hi << 16U);
}

// Exchanges a register between lanes l and l ^ 16.
__device__ __forceinline__ unsigned SwapHalves(unsigned x) {
  return __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false,
                                      false);
}

// Eight waves own sixteen queries each; every 32-key tile of K and V is
// staged once in LDS for all of them, double-buffered: the next tile is read
// from memory while the current one is used, one barrier per tile. Online
// base-two softmax in FP32 that rescales the output only when a row maximum
// grew by more than 2^kSlack; P*V runs in FP16 with an FP32 accumulator.
__global__ __launch_bounds__(256, 1) void Attention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t query_rows,
    std::uint32_t key_rows, std::uint32_t first_block) {
  __shared__ __align__(16) std::uint16_t staged_key[2][32][136];
  __shared__ __align__(16) std::uint16_t staged_value[2][128][40];
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
  // A score accumulator holds keys 2i + lane / 16 of query lane % 16; the
  // P fragment needs all sixteen keys in order, half of them from lane ^ 16.
  const unsigned select_even = lane < 16 ? 0x05040100U : 0x01000504U;
  const unsigned select_odd = lane < 16 ? 0x07060302U : 0x03020706U;
  const int key_row = tid / 16, key_dim = (tid % 16) * 8;
  const int value_dim = tid / 4, value_key = (tid % 4) * 8;
  uint4 key_stage[2], value_stage[2];
  auto fetch = [&](std::uint32_t base) {
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      key_stage[z] = LoadBufferVector(
          key_buffer, ((base + key_row + z * 16) * 128 + key_dim) * 2);
      value_stage[z] = LoadBufferVector(
          value_buffer,
          ((value_dim + z * 64) * padded_rows + base + value_key) * 2);
    }
  };
  auto stash = [&](int buffer) {
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      *reinterpret_cast<uint4*>(staged_key[buffer][key_row + z * 16] +
                                key_dim) = key_stage[z];
      *reinterpret_cast<uint4*>(staged_value[buffer][value_dim + z * 64] +
                                value_key) = value_stage[z];
    }
  };
  fetch(0);
  stash(0);
  __syncthreads();

  Accumulator result[8]{};
  float maximum = -INFINITY, sum = 0;
  const std::uint32_t tiles = (key_rows + 31) / 32;
  for (std::uint32_t tile = 0; tile < tiles; ++tile) {
    const int buffer = tile & 1;
    const std::uint32_t base = tile * 32;
    if (tile + 1 < tiles)
      fetch(base + 32);
    Accumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const Fragment k =
            LoadFragment(staged_key[buffer][half * 16 + lane % 16] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[d], scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (base + 32 > key_rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (base + half * 16 + 2 * i + lane / 16 >= key_rows)
            scores[half][i] = -INFINITY;
    }
    float top = scores[0][0];
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        top = fmaxf(top, scores[half][i]);
    top =
        fmaxf(top, __uint_as_float(SwapHalves(__float_as_uint(top)))) * kScale;
    // Wave-uniform: rescale only when some row maximum outgrew the slack.
    if (!__all(top <= maximum + kSlack)) {
      const float next = fmaxf(maximum, top);
      const float correction = __builtin_amdgcn_exp2f(maximum - next);
      maximum = next;
      sum *= correction;
#pragma unroll
      for (int d = 0; d < 8; ++d)
        result[d] *= correction;
    }
    Words probabilities[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        const float p0 = __builtin_amdgcn_exp2f(
            __builtin_fmaf(scores[half][2 * a], kScale, -maximum));
        const float p1 = __builtin_amdgcn_exp2f(
            __builtin_fmaf(scores[half][2 * a + 1], kScale, -maximum));
        sum += p0 + p1;  // this lane's keys; both halves summed at the end
        const unsigned own = PackHalves(p0, p1);
        const unsigned peer = SwapHalves(own);
        probabilities[half][2 * a] =
            __builtin_amdgcn_perm(peer, own, select_even);
        probabilities[half][2 * a + 1] =
            __builtin_amdgcn_perm(peer, own, select_odd);
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const auto p = __builtin_bit_cast(HalfFragment, probabilities[half]);
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        const auto v = __builtin_bit_cast(
            HalfFragment,
            LoadFragment(staged_value[buffer][d * 16 + lane % 16] + half * 16));
        result[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v, p, result[d]);
        if (d % 2 == 1)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
    if (tile + 1 < tiles)
      stash(buffer ^ 1);
    __syncthreads();
  }
  sum += __uint_as_float(SwapHalves(__float_as_uint(sum)));
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
  std::uint32_t per_launch = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      blocks, std::max<std::uint64_t>(
                  1, kMaxPairsPerLaunch /
                         std::max<std::uint64_t>(1, pairs_per_block))));
  // Equal launches avoid a short last one that leaves most of the GPU idle.
  const std::uint32_t launches = (blocks + per_launch - 1) / per_launch;
  per_launch = (blocks + launches - 1) / launches;
  for (std::uint32_t first = 0; first < blocks; first += per_launch) {
    const std::uint32_t count = std::min(per_launch, blocks - first);
    Attention<<<dim3(count, heads), 256, 0, stream>>>(
        query, key, packed_values, output, query_rows, key_rows, first);
  }
}

}  // namespace gufo::models::wan22
