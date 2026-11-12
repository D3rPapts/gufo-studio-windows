#include "src/models/moss_music/hip/stem_ops.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

// The WMMA convolutions are the Qwen3-ASR kernels (src/models/qwen3_asr/hip/
// convolution.hip) instantiated at MOSS-Music's chunk geometry; they live in
// that file's anonymous namespace, hence the copy. Rounding conventions are
// the same so that both towers stay comparable to their bf16 teachers.
namespace gufo::models::moss_music::hip {
namespace {

constexpr std::size_t kThreads = 256;
constexpr std::size_t kConvChannels = 480;
constexpr std::size_t kConvFrequency = 16;
constexpr std::size_t kConvFlattened = kConvChannels * kConvFrequency;

using B = hip_bfloat16;
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

std::size_t Blocks(std::size_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

__device__ Fragment LoadFragment(const unsigned short* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__device__ float Round(float value) { return static_cast<float>(B(value)); }

__global__ void ChunkLogMelKernel(const float* input, B* output,
                                  std::size_t frames, std::size_t mel_bins,
                                  std::size_t chunk_frames,
                                  std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t time = index % chunk_frames;
  const std::size_t remaining = index / chunk_frames;
  const std::size_t mel = remaining % mel_bins;
  const std::size_t chunk = remaining / mel_bins;
  const std::size_t source_time = chunk * chunk_frames + time;
  output[index] =
      B(source_time < frames ? input[mel * frames + source_time] : 0.0F);
}

// NCHW, 3x3, stride two, one zero-padding pixel, one input channel: the nine
// taps padded to one WMMA instruction. Input [128][Width], output [480][64][Width/2].
template<int Width>
__global__ __launch_bounds__(128) void FirstConvolution(
    const unsigned short* input, const unsigned short* weight, B* output) {
  constexpr int OW = (Width + 1) / 2, Pixels = 64 * OW, M = 96, N = 64;
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const int channel0 = blockIdx.x * M, pixel0 = blockIdx.y * N;
  input += std::size_t(blockIdx.z) * 128 * Width;
  output += std::size_t(blockIdx.z) * 480 * Pixels;
  Fragment a[3];
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    a[m] = {};
    const int channel = channel0 + (wave / 2 * 3 + m) * 16 + lane % 16;
#pragma unroll
    for (int k = 0; k < 9; ++k)
      a[m][k] = weight[channel * 9 + k];
  }
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    Fragment b{};
    const int pixel = pixel0 + (wave % 2 * 2 + n) * 16 + lane % 16;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const int y = pixel / OW * 2 - 1 + k / 3;
      const int x = pixel % OW * 2 - 1 + k % 3;
      if (pixel < Pixels && y >= 0 && y < 128 && x >= 0 && x < Width)
        b[k] = input[y * Width + x];
    }
#pragma unroll
    for (int m = 0; m < 3; ++m) {
      const auto sum =
          __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[m], b, Accumulator{});
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int channel =
            channel0 + (wave / 2 * 3 + m) * 16 + 2 * i + lane / 16;
        if (pixel < Pixels)
          output[channel * Pixels + pixel] = B(sum[i]);
      }
    }
  }
}

// 128-pixel tiles rotate the channel/tap reduction by 128 products, wrapping
// at K=4320 (the teacher's BF16 GEMM accumulation order); weights packed once
// by Qwen3-ASR's PackWeights; im2col stays in LDS.
template<int Height, int Width, int M, int N, int WM, int WN>
__global__ __launch_bounds__(WM * WN * 32) void Convolution(
    const unsigned short* input, const unsigned short* weight, B* output) {
  constexpr int C = 16, K = C * 9, T = WM * WN * 32;
  constexpr int OH = (Height + 1) / 2, OW = (Width + 1) / 2, Pixels = OH * OW;
  constexpr int MR = M / (16 * WM), NR = N / (16 * WN);
  __shared__ __align__(16) unsigned short tile[N][K + 8];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int channel0 = blockIdx.x * M, pixel0 = blockIdx.y * N;
  input += std::size_t(blockIdx.z) * 480 * Height * Width;
  output += std::size_t(blockIdx.z) * 480 * Pixels;
  Accumulator acc[MR][NR]{};
  for (int cbase = 0; cbase < 480; cbase += C) {
    for (int i = tid; i < N * K; i += T) {
      const int p = i % N, k = i / N, pixel = pixel0 + p;
      const int reduction = (pixel0 / 128 * 128 + cbase * 9 + k) % (480 * 9);
      const int y = pixel / OW * 2 - 1 + (reduction % 9) / 3;
      const int x = pixel % OW * 2 - 1 + reduction % 3;
      tile[p][k] = pixel < Pixels && y >= 0 && y < Height && x >= 0 && x < Width
                       ? input[(reduction / 9 * Height + y) * Width + x]
                       : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; k += 16) {
      Fragment a[MR];
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const int r = channel0 + (wave / WN * MR + m) * 16 + lane % 16;
        a[m] = LoadFragment(
            weight +
            std::size_t(((pixel0 / 128 * 128 + cbase * 9 + k) % (480 * 9)) /
                        16) *
                480 * 16 +
            r * 16);
      }
#pragma unroll
      for (int n = 0; n < NR; ++n) {
        const auto b =
            LoadFragment(tile[(wave % WN * NR + n) * 16 + lane % 16] + k);
#pragma unroll
        for (int m = 0; m < MR; ++m)
          acc[m][n] =
              __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, a[m], acc[m][n]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int n = 0; n < NR; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int p = pixel0 + (wave % WN * NR + n) * 16 + 2 * i + lane / 16;
        const int r = channel0 + (wave / WN * MR + m) * 16 + lane % 16;
        if (p < Pixels)
          output[r * Pixels + p] = B(acc[m][n][i]);
      }
}

__global__ void ConvOutputLayoutKernel(const B* input, B* output,
                                       std::size_t chunk_time,
                                       std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t flattened = index % kConvFlattened;
  const std::size_t row = index / kConvFlattened;
  const std::size_t time = row % chunk_time;
  const std::size_t chunk = row / chunk_time;
  const std::size_t frequency = flattened % kConvFrequency;
  const std::size_t channel = flattened / kConvFrequency;
  const std::size_t source =
      ((chunk * kConvChannels + channel) * kConvFrequency + frequency) *
          chunk_time +
      time;
  output[index] = input[source];
}

__device__ std::size_t StemOutLength(std::size_t length) {
  for (int step = 0; step < 3; ++step) {
    length = length == 0 ? 0 : (length - 1) / 2 + 1;
  }
  return length;
}

/// SinusoidsPositionEmbedding as the bf16 model evaluates it: the timescale
/// buffer, the time index and the phase are BF16, sin/cos round to BF16.
__device__ float Position(std::size_t time, std::size_t column,
                          std::size_t hidden) {
  const float log_increment =
      9.210340371976184F / static_cast<float>(hidden / 2U - 1U);
  const std::size_t frequency = column % (hidden / 2U);
  const float inverse_timescale =
      Round(expf(-log_increment * static_cast<float>(frequency)));
  const float phase = Round(Round(static_cast<float>(time)) * inverse_timescale);
  return Round(column < hidden / 2U ? sinf(phase) : cosf(phase));
}

__global__ void AddBiasPositionAndCompactKernel(
    const float* projected, const B* bias, float* compact_output,
    std::size_t feature_frames, std::size_t chunks, std::size_t chunk_frames,
    std::size_t chunk_time, std::size_t tokens, std::size_t hidden) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) {
    return;
  }
  const std::size_t column = index % hidden;
  std::size_t remaining = index / hidden;
  std::size_t selected_chunk = 0;
  std::size_t selected_time = 0;
  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    const std::size_t length =
        min(chunk_frames, feature_frames - chunk * chunk_frames);
    const std::size_t valid = StemOutLength(length);
    if (remaining < valid) {
      selected_chunk = chunk;
      selected_time = remaining;
      break;
    }
    remaining -= valid;
  }
  const std::size_t source =
      (selected_chunk * chunk_time + selected_time) * hidden + column;
  // stem_proj: one rounding after the bias (bf16 addmm), then the position.
  const float linear = Round(projected[source] + static_cast<float>(bias[column]));
  compact_output[index] =
      Round(linear + Position(selected_time, column, hidden));
}

__global__ void SiluMulToBfloat16Kernel(const float* gate, const float* up,
                                        B* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float g = Round(gate[index]);
  const float activated = Round(g / (1.0F + expf(-g)));
  output[index] = B(activated * Round(up[index]));
}

__global__ void RoundBfloat16Kernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = Round(values[index]);
  }
}

__global__ void ZeroColumnsKernel(B* values, std::size_t width,
                                  std::size_t valid, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements && index % width >= valid) {
    values[index] = B(0.0F);
  }
}

__global__ void FloatToBfloat16Kernel(const float* input, B* output,
                                      std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = B(input[index]);
  }
}

__global__ void Bfloat16ToFloatKernel(const B* input, float* output,
                                      std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = static_cast<float>(input[index]);
  }
}

__global__ void ScatterRowsRoundedKernel(float* hidden, const float* source,
                                         const std::uint32_t* rows,
                                         std::size_t count,
                                         std::size_t hidden_size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count * hidden_size) {
    return;
  }
  const std::size_t row = index / hidden_size;
  const std::size_t column = index % hidden_size;
  hidden[static_cast<std::size_t>(rows[row]) * hidden_size + column] =
      Round(source[index]);
}

__global__ void AddRowsRoundedKernel(float* hidden, const float* source,
                                     const std::uint32_t* rows,
                                     std::size_t count,
                                     std::size_t hidden_size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count * hidden_size) {
    return;
  }
  const std::size_t row = index / hidden_size;
  const std::size_t column = index % hidden_size;
  float* target =
      hidden + static_cast<std::size_t>(rows[row]) * hidden_size + column;
  *target = Round(*target + source[index]);
}

}  // namespace

void LaunchChunkLogMel(const float* input, void* output_bfloat16,
                       std::size_t frames, std::size_t chunks,
                       std::size_t mel_bins, std::size_t chunk_frames,
                       hipStream_t stream) {
  const std::size_t elements = chunks * mel_bins * chunk_frames;
  hipLaunchKernelGGL(ChunkLogMelKernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, input, static_cast<B*>(output_bfloat16), frames,
                     mel_bins, chunk_frames, elements);
}

void LaunchFirstConvolution(const void* input_bfloat16,
                            const void* weight_bfloat16, void* output_bfloat16,
                            std::size_t chunks, hipStream_t stream) {
  constexpr int kWidth = 400;
  constexpr int kPixels = 64 * (kWidth / 2);
  FirstConvolution<kWidth>
      <<<dim3(5, (kPixels + 63) / 64, chunks), 128, 0, stream>>>(
          static_cast<const unsigned short*>(input_bfloat16),
          static_cast<const unsigned short*>(weight_bfloat16),
          static_cast<B*>(output_bfloat16));
}

void LaunchConvolution(const void* input_bfloat16, const void* packed_weight,
                       void* output_bfloat16, std::size_t chunks,
                       std::size_t height, std::size_t width,
                       hipStream_t stream) {
  constexpr int M = 480, N = 32, WM = 6, WN = 2;
  const auto launch = [&]<int H, int W>() {
    constexpr int pixels = ((H + 1) / 2) * ((W + 1) / 2);
    Convolution<H, W, M, N, WM, WN>
        <<<dim3(480 / M, (pixels + N - 1) / N, chunks), WM * WN * 32, 0,
           stream>>>(static_cast<const unsigned short*>(input_bfloat16),
                     static_cast<const unsigned short*>(packed_weight),
                     static_cast<B*>(output_bfloat16));
  };
  if (height == 64 && width == 200) {
    launch.template operator()<64, 200>();
  } else if (height == 32 && width == 100) {
    launch.template operator()<32, 100>();
  } else {
    throw std::invalid_argument("unsupported MOSS-Music convolution shape");
  }
}

void LaunchConvOutputLayout(const void* input_bfloat16, void* output_bfloat16,
                            std::size_t chunks, std::size_t chunk_time,
                            hipStream_t stream) {
  const std::size_t elements = chunks * chunk_time * kConvFlattened;
  hipLaunchKernelGGL(ConvOutputLayoutKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream,
                     static_cast<const B*>(input_bfloat16),
                     static_cast<B*>(output_bfloat16), chunk_time, elements);
}

void LaunchAddBiasPositionAndCompact(const float* projected,
                                     const void* bias_bfloat16,
                                     float* compact_output,
                                     std::size_t feature_frames,
                                     std::size_t chunks,
                                     std::size_t chunk_frames,
                                     std::size_t chunk_time, std::size_t tokens,
                                     std::size_t hidden, hipStream_t stream) {
  const std::size_t elements = tokens * hidden;
  hipLaunchKernelGGL(AddBiasPositionAndCompactKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, projected,
                     static_cast<const B*>(bias_bfloat16), compact_output,
                     feature_frames, chunks, chunk_frames, chunk_time, tokens,
                     hidden);
}

void LaunchSiluMulToBfloat16(const float* gate, const float* up,
                             void* output_bfloat16, std::size_t elements,
                             hipStream_t stream) {
  hipLaunchKernelGGL(SiluMulToBfloat16Kernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, gate, up,
                     static_cast<B*>(output_bfloat16), elements);
}

void LaunchRoundBfloat16(float* values, std::size_t elements,
                         hipStream_t stream) {
  hipLaunchKernelGGL(RoundBfloat16Kernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, values, elements);
}

void LaunchZeroColumns(void* values_bfloat16, std::size_t rows,
                       std::size_t width, std::size_t valid,
                       hipStream_t stream) {
  const std::size_t elements = rows * width;
  hipLaunchKernelGGL(ZeroColumnsKernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, static_cast<B*>(values_bfloat16), width, valid,
                     elements);
}

void LaunchFloatToBfloat16(const float* input, void* output_bfloat16,
                           std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(FloatToBfloat16Kernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, input,
                     static_cast<B*>(output_bfloat16), elements);
}

void LaunchBfloat16ToFloat(const void* input_bfloat16, float* output,
                           std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(Bfloat16ToFloatKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream,
                     static_cast<const B*>(input_bfloat16), output, elements);
}

void LaunchScatterRowsRounded(float* hidden, const float* source,
                              const std::uint32_t* rows, std::size_t count,
                              std::size_t hidden_size, hipStream_t stream) {
  const std::size_t elements = count * hidden_size;
  hipLaunchKernelGGL(ScatterRowsRoundedKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, hidden, source, rows, count,
                     hidden_size);
}

void LaunchAddRowsRounded(float* hidden, const float* source,
                          const std::uint32_t* rows, std::size_t count,
                          std::size_t hidden_size, hipStream_t stream) {
  const std::size_t elements = count * hidden_size;
  hipLaunchKernelGGL(AddRowsRoundedKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, hidden, source, rows, count,
                     hidden_size);
}

}  // namespace gufo::models::moss_music::hip
