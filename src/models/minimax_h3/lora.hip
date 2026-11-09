#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <string_view>

#include "src/models/minimax_h3/lora.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr int kTile = 64;
constexpr int kDepth = 16;
constexpr int kThreads = 16;
constexpr int kPerThread = kTile / kThreads;

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t FloatToBf16(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

// weight[o][i] += scale * sum_r up[o][r] * down[r][i], 64x64 tile per block.
__global__ void MergeLoraKernel(std::uint16_t* weight, const float* up,
                                const float* down, int output, int input,
                                int rank, float scale) {
  __shared__ float up_tile[kTile][kDepth + 1];
  __shared__ float down_tile[kDepth][kTile + 1];
  const int row_base = blockIdx.y * kTile;
  const int column_base = blockIdx.x * kTile;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  float accumulator[kPerThread][kPerThread] = {};
  for (int depth = 0; depth < rank; depth += kDepth) {
    for (int index = ty * kThreads + tx; index < kTile * kDepth;
         index += kThreads * kThreads) {
      const int row = index / kDepth;
      const int column = index % kDepth;
      const int global_row = row_base + row;
      const int global_depth = depth + column;
      up_tile[row][column] =
          global_row < output && global_depth < rank
              ? up[static_cast<std::size_t>(global_row) * rank + global_depth]
              : 0.0F;
      const int down_row = index / kTile;
      const int down_column = index % kTile;
      const int global_down = depth + down_row;
      const int global_column = column_base + down_column;
      down_tile[down_row][down_column] =
          global_down < rank && global_column < input
              ? down[static_cast<std::size_t>(global_down) * input +
                     global_column]
              : 0.0F;
    }
    __syncthreads();
    for (int k = 0; k < kDepth; ++k) {
      float left[kPerThread];
      float right[kPerThread];
      for (int item = 0; item < kPerThread; ++item) {
        left[item] = up_tile[ty + item * kThreads][k];
        right[item] = down_tile[k][tx + item * kThreads];
      }
      for (int r = 0; r < kPerThread; ++r) {
        for (int c = 0; c < kPerThread; ++c) {
          accumulator[r][c] += left[r] * right[c];
        }
      }
    }
    __syncthreads();
  }
  for (int r = 0; r < kPerThread; ++r) {
    const int row = row_base + ty + r * kThreads;
    if (row >= output) {
      continue;
    }
    for (int c = 0; c < kPerThread; ++c) {
      const int column = column_base + tx + c * kThreads;
      if (column >= input) {
        continue;
      }
      std::uint16_t& target =
          weight[static_cast<std::size_t>(row) * input + column];
      target = FloatToBf16(Bf16ToFloat(target) + scale * accumulator[r][c]);
    }
  }
}

struct DeviceScratch {
  float* data{nullptr};
  std::size_t bytes{0};
  DeviceScratch() = default;
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;
  ~DeviceScratch() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
  hipError_t Reserve(std::size_t needed) {
    if (needed <= bytes) {
      return hipSuccess;
    }
    if (data != nullptr) {
      (void)hipFree(data);
      data = nullptr;
      bytes = 0;
    }
    const hipError_t status = hipMalloc(&data, needed);
    if (status == hipSuccess) {
      bytes = needed;
    } else {
      data = nullptr;
    }
    return status;
  }
};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

bool HipMergeLoraBf16(void* weight, const LoraFactor& factor,
                      const std::vector<float>& down,
                      const std::vector<float>& up, void* stream,
                      std::string* error) {
  if (weight == nullptr || down.size() != factor.rank * factor.input ||
      up.size() != factor.output * factor.rank || factor.rank > 65536 ||
      factor.output > (1U << 30U) || factor.input > (1U << 30U)) {
    SetError(error, "invalid MiniMax H3 LoRA merge arguments");
    return false;
  }
  auto* hip_stream = static_cast<hipStream_t>(stream);
  const std::size_t down_bytes = down.size() * sizeof(float);
  const std::size_t up_bytes = up.size() * sizeof(float);
  // One reusable scratch per loader thread instead of ~200 short-lived
  // allocations while the DiT fills device memory.
  thread_local DeviceScratch scratch;
  hipError_t status = scratch.Reserve(down_bytes + up_bytes);
  float* device = scratch.data;
  if (status == hipSuccess) {
    status = hipMemcpyAsync(device, down.data(), down_bytes,
                            hipMemcpyHostToDevice, hip_stream);
  }
  if (status == hipSuccess) {
    status = hipMemcpyAsync(device + down.size(), up.data(), up_bytes,
                            hipMemcpyHostToDevice, hip_stream);
  }
  if (status == hipSuccess) {
    const dim3 grid(static_cast<unsigned>((factor.input + kTile - 1) / kTile),
                    static_cast<unsigned>((factor.output + kTile - 1) / kTile));
    const dim3 block(kThreads, kThreads);
    MergeLoraKernel<<<grid, block, 0, hip_stream>>>(
        static_cast<std::uint16_t*>(weight), device + down.size(), device,
        static_cast<int>(factor.output), static_cast<int>(factor.input),
        static_cast<int>(factor.rank), factor.scale);
    status = hipGetLastError();
  }
  if (status == hipSuccess) {
    status = hipStreamSynchronize(hip_stream);
  }
  if (status != hipSuccess) {
    SetError(error, std::string("merge MiniMax H3 LoRA: ") +
                        hipGetErrorString(status));
    return false;
  }
  return true;
}

}  // namespace gufo::minimax_h3

#endif  // defined(ENGINE_ENABLE_HIP)
