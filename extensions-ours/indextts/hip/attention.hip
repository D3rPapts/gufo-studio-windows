// WMMA flash attention with head dimension 64 for the IndexTTS-2.5 DiT on
// gfx1151, adapted from the 128-dim kernel in src/models/wan22/attention.hip
// (itself from minimax_h3): four waves own sixteen queries each, every 32-key
// tile of K and V is staged once in LDS (double-buffered), online base-two
// softmax in FP32 that rescales only when a row maximum grew by more than
// 2^kSlack. Differences: head dim 64, Q and K enter QK^T as FP16 (not BF16:
// three more mantissa bits for the scores), FP32 input/output in the DiT's
// row-major [rows, heads * 64] layout, a batch of independent sequences
// (CFG branches) in grid.z. Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "indextts/hip/ops.hpp"

namespace gufo::models::indextts::hip {
namespace {

using BufferResource = __attribute__((ext_vector_type(4))) unsigned;
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using HalfFragment = __attribute__((ext_vector_type(16))) _Float16;
using Words = __attribute__((ext_vector_type(8))) unsigned;
using Accumulator = __attribute__((ext_vector_type(8))) float;

constexpr int kHeadDim = 64;
constexpr int kWaves = 4;
constexpr int kThreads = kWaves * 32;
constexpr int kQueriesPerBlock = kWaves * 16;
constexpr float kScale = 0.18033688011112042F;  // log2(e) / sqrt(64)
constexpr float kSlack = 8.0F;

}  // namespace

__device__ BufferResource
IttsLoadBuffer(BufferResource, unsigned, unsigned, unsigned) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

namespace {

__device__ __forceinline__ BufferResource MakeBuffer(const std::uint16_t* pointer, unsigned bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return {unsigned(address), unsigned(address >> 32), bytes, 0x31004000U};
}

__device__ __forceinline__ uint4 LoadBufferVector(BufferResource resource, unsigned offset) {
  union {
    BufferResource vector;
    uint4 packed;
  } result;
  result.vector = IttsLoadBuffer(resource, offset, 0, 0);
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

__device__ __forceinline__ std::uint16_t Half(float x) {
  return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(fminf(fmaxf(x, -65504.0F), 65504.0F)));
}

__device__ __forceinline__ unsigned PackHalves(float low, float high) {
  const unsigned lo = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(low));
  const unsigned hi = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(high));
  return lo | (hi << 16U);
}

__device__ __forceinline__ unsigned SwapHalves(unsigned x) {
  return __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false, false);
}

__host__ __device__ constexpr unsigned Padded(unsigned rows) { return (rows + 31U) & ~31U; }

// FP32 [nb][rows][heads * 64] -> FP16 head-major Q/K [nb][heads][rows][64] and
// packed V [nb][heads][64][Padded(rows)] (pad already zeroed by the caller).
__global__ void Prep(const float* q, const float* k, const float* v, std::uint16_t* q16, std::uint16_t* k16,
                     std::uint16_t* vp, unsigned rows, unsigned heads) {
  const std::size_t per = std::size_t(rows) * heads * kHeadDim;
  const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (i >= per) {
    return;
  }
  const std::size_t b = blockIdx.y;
  const unsigned r = unsigned(i / (heads * kHeadDim));
  const unsigned rem = unsigned(i % (heads * kHeadDim));
  const unsigned h = rem / kHeadDim, d = rem % kHeadDim;
  const std::size_t src = b * per + i;
  const std::size_t hm = b * per + (std::size_t(h) * rows + r) * kHeadDim + d;
  q16[hm] = Half(q[src]);
  k16[hm] = Half(k[src]);
  const unsigned padded = Padded(rows);
  vp[b * std::size_t(heads) * kHeadDim * padded + (std::size_t(h) * kHeadDim + d) * padded + r] = Half(v[src]);
}

__global__ __launch_bounds__(kThreads, 2) void Attention(const std::uint16_t* query, const std::uint16_t* key,
                                                         const std::uint16_t* value, float* output,
                                                         std::uint32_t rows) {
  __shared__ __align__(16) std::uint16_t staged_key[2][32][kHeadDim + 8];
  __shared__ __align__(16) std::uint16_t staged_value[2][kHeadDim][40];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const std::uint32_t heads = gridDim.y;
  const std::size_t b = blockIdx.z;
  const unsigned padded = Padded(rows);
  const int first_query = blockIdx.x * kQueriesPerBlock + wave * 16;
  const std::size_t per = std::size_t(rows) * heads * kHeadDim;
  query += b * per + std::size_t(blockIdx.y) * rows * kHeadDim;
  key += b * per + std::size_t(blockIdx.y) * rows * kHeadDim;
  value += b * std::size_t(heads) * kHeadDim * padded + std::size_t(blockIdx.y) * padded * kHeadDim;
  output += b * per;
  const auto key_buffer = MakeBuffer(key, rows * kHeadDim * 2);
  const auto value_buffer = MakeBuffer(value, padded * kHeadDim * 2);
  Fragment q[kHeadDim / 16];
#pragma unroll
  for (int d = 0; d < kHeadDim / 16; ++d) {
    q[d] = {};
    if (static_cast<std::uint32_t>(first_query + lane % 16) < rows) {
      q[d] = LoadFragment(query + std::size_t(first_query + lane % 16) * kHeadDim + d * 16);
    }
  }
  const unsigned select_even = lane < 16 ? 0x05040100U : 0x01000504U;
  const unsigned select_odd = lane < 16 ? 0x07060302U : 0x03020706U;
  // 32 keys x 64 dims and 64 dims x 32 keys = 256 16-byte vectors each, two per thread.
  uint4 key_stage[2], value_stage[2];
  auto fetch = [&](std::uint32_t base) {
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int idx = tid + z * kThreads;
      const int key_row = idx / 8, key_dim = (idx % 8) * 8;
      const int value_dim = idx / 4, value_key = (idx % 4) * 8;
      key_stage[z] = LoadBufferVector(key_buffer, ((base + key_row) * kHeadDim + key_dim) * 2);
      value_stage[z] = LoadBufferVector(value_buffer, (value_dim * padded + base + value_key) * 2);
    }
  };
  auto stash = [&](int buffer) {
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int idx = tid + z * kThreads;
      const int key_row = idx / 8, key_dim = (idx % 8) * 8;
      const int value_dim = idx / 4, value_key = (idx % 4) * 8;
      *reinterpret_cast<uint4*>(staged_key[buffer][key_row] + key_dim) = key_stage[z];
      *reinterpret_cast<uint4*>(staged_value[buffer][value_dim] + value_key) = value_stage[z];
    }
  };
  fetch(0);
  stash(0);
  __syncthreads();

  Accumulator result[kHeadDim / 16]{};
  float maximum = -INFINITY, sum = 0;
  const std::uint32_t tiles = (rows + 31) / 32;
  for (std::uint32_t tile = 0; tile < tiles; ++tile) {
    const int buffer = tile & 1;
    const std::uint32_t base = tile * 32;
    if (tile + 1 < tiles) {
      fetch(base + 32);
    }
    Accumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < kHeadDim / 16; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const auto k = __builtin_bit_cast(HalfFragment, LoadFragment(staged_key[buffer][half * 16 + lane % 16] + d * 16));
        scores[half] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, __builtin_bit_cast(HalfFragment, q[d]),
                                                                  scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (base + 32 > rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (base + half * 16 + 2 * i + lane / 16 >= rows) {
            scores[half][i] = -INFINITY;
          }
        }
      }
    }
    float top = scores[0][0];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        top = fmaxf(top, scores[half][i]);
      }
    }
    top = fmaxf(top, __uint_as_float(SwapHalves(__float_as_uint(top)))) * kScale;
    if (!__all(top <= maximum + kSlack)) {
      const float next = fmaxf(maximum, top);
      const float correction = __builtin_amdgcn_exp2f(maximum - next);
      maximum = next;
      sum *= correction;
#pragma unroll
      for (int d = 0; d < kHeadDim / 16; ++d) {
        result[d] *= correction;
      }
    }
    Words probabilities[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        const float p0 = __builtin_amdgcn_exp2f(__builtin_fmaf(scores[half][2 * a], kScale, -maximum));
        const float p1 = __builtin_amdgcn_exp2f(__builtin_fmaf(scores[half][2 * a + 1], kScale, -maximum));
        sum += p0 + p1;
        const unsigned own = PackHalves(p0, p1);
        const unsigned peer = SwapHalves(own);
        probabilities[half][2 * a] = __builtin_amdgcn_perm(peer, own, select_even);
        probabilities[half][2 * a + 1] = __builtin_amdgcn_perm(peer, own, select_odd);
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const auto p = __builtin_bit_cast(HalfFragment, probabilities[half]);
#pragma unroll
      for (int d = 0; d < kHeadDim / 16; ++d) {
        const auto v =
            __builtin_bit_cast(HalfFragment, LoadFragment(staged_value[buffer][d * 16 + lane % 16] + half * 16));
        result[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v, p, result[d]);
        if (d % 2 == 1) {
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
    if (tile + 1 < tiles) {
      stash(buffer ^ 1);
    }
    __syncthreads();
  }
  sum += __uint_as_float(SwapHalves(__float_as_uint(sum)));
  const std::uint32_t row = first_query + lane % 16;
  if (row < rows) {
    const float reciprocal = 1.0F / sum;
#pragma unroll
    for (int d = 0; d < kHeadDim / 16; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        output[(std::size_t(row) * heads + blockIdx.y) * kHeadDim + d * 16 + 2 * i + lane / 16] =
            result[d][i] * reciprocal;
      }
    }
  }
}

}  // namespace

std::size_t FlashAttention64ScratchHalves(std::size_t rows, std::size_t heads, std::size_t sequences) {
  return sequences * heads * kHeadDim * (2 * rows + Padded(static_cast<unsigned>(rows)));
}

void LaunchFlashAttention64(const float* q, const float* k, const float* v, float* out, std::uint16_t* scratch,
                            std::size_t rows, std::size_t heads, std::size_t sequences, hipStream_t stream) {
  const auto r = static_cast<unsigned>(rows);
  const auto h = static_cast<unsigned>(heads);
  const auto s = static_cast<unsigned>(sequences);
  const std::size_t per = rows * heads * kHeadDim;
  std::uint16_t* q16 = scratch;
  std::uint16_t* k16 = q16 + sequences * per;
  std::uint16_t* vp = k16 + sequences * per;
  const std::size_t packed = sequences * heads * kHeadDim * Padded(r);
  (void)hipMemsetAsync(vp, 0, packed * 2, stream);
  Prep<<<dim3(static_cast<unsigned>((per + 255) / 256), s), 256, 0, stream>>>(q, k, v, q16, k16, vp, r, h);
  // 1094 rows: 18 x 8 x 2 blocks; one launch stays far below Windows' 2 s watchdog for any DiT length.
  Attention<<<dim3((r + kQueriesPerBlock - 1) / kQueriesPerBlock, h, s), kThreads, 0, stream>>>(q16, k16, vp, out, r);
}

}  // namespace gufo::models::indextts::hip
