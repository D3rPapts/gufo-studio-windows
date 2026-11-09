#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kLevels = 6;
constexpr std::array<int, kLevels> kLevelChannels = {128, 256, 256,
                                                     512, 512, 1024};
constexpr std::array<bool, kLevels> kLevelDownsample = {true, true,  true,
                                                        true, false, false};
// Temporal stride of each level's downsample (the reference's time_down).
constexpr std::array<int, kLevels> kLevelTimeStride = {1, 2, 2, 1, 1, 1};
// The reference encodes clips in chunks of 17 frames and drops the last 3
// latent frames of the concatenation (clip_length / token_drop).
constexpr int kClipChunkFrames = 17;
constexpr int kClipTokenDrop = 3;
constexpr int kGroups = 32;
constexpr float kGroupNormEpsilon = 1.0e-6F;
constexpr int kSpatialRatio = 16;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }
  bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      data_ = nullptr;
      SetError(error, HipError("hipMalloc H3 VisualVAE encoder", status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }
  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
    data_ = nullptr;
    bytes_ = 0;
  }
  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

__global__ void NormalizeTileKernel(const std::uint8_t* rgb, int image_width,
                                    int tile_x, int tile_y, int tile_width,
                                    int tile_height, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = tile_width * tile_height;
  if (index >= plane) {
    return;
  }
  const int y = index / tile_width;
  const int x = index - y * tile_width;
  const std::size_t source =
      (static_cast<std::size_t>(tile_y + y) * image_width + tile_x + x) * 3U;
  const float mean[3] = {0.485F, 0.456F, 0.406F};
  const float deviation[3] = {0.229F, 0.224F, 0.225F};
  for (int channel = 0; channel < 3; ++channel) {
    const float value = static_cast<float>(rgb[source + channel]) / 255.0F;
    output[static_cast<std::size_t>(channel) * plane + index] =
        (value - mean[channel]) / deviation[channel];
  }
}

__device__ __forceinline__ int Reflect(int index, int length) {
  if (index < 0) {
    index = -index;
  }
  if (index >= length) {
    index = 2 * length - 2 - index;
  }
  return index;
}

// col[(c * 9 + ky * 3 + kx) * outputs + oy * out_w + ox]
__global__ void Im2Col3x3Kernel(const float* input, int channels, int height,
                                int width, int out_height, int out_width,
                                int stride, int pad, float* col) {
  const std::size_t outputs = static_cast<std::size_t>(out_height) * out_width;
  const std::size_t total = outputs * channels * 9U;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const std::size_t position = index % outputs;
  const std::size_t row = index / outputs;
  const int tap = static_cast<int>(row % 9U);
  const int channel = static_cast<int>(row / 9U);
  const int oy = static_cast<int>(position / out_width);
  const int ox = static_cast<int>(position % out_width);
  const int y = Reflect(oy * stride + tap / 3 - pad, height);
  const int x = Reflect(ox * stride + tap % 3 - pad, width);
  col[index] =
      input[(static_cast<std::size_t>(channel) * height + y) * width + x];
}

__global__ void AddBiasKernel(float* output, const float* bias, int channels,
                              int plane) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(channels) * plane) {
    return;
  }
  output[index] += bias[index / plane];
}

__global__ void AddKernel(float* output, const float* addend,
                          std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] += addend[index];
  }
}

constexpr int kNormThreads = 1024;

// One block per group: mean, then biased variance around it, in float.
__global__ void GroupStatsKernel(const float* input, int channels_per_group,
                                 int plane, float epsilon, float* stats) {
  __shared__ float partial[kNormThreads];
  const std::size_t count =
      static_cast<std::size_t>(channels_per_group) * plane;
  const float* group = input + static_cast<std::size_t>(blockIdx.x) * count;
  float sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += group[index];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width /= 2) {
    if (threadIdx.x < width) {
      partial[threadIdx.x] += partial[threadIdx.x + width];
    }
    __syncthreads();
  }
  const float mean = partial[0] / static_cast<float>(count);
  __syncthreads();
  float squares = 0.0F;
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    const float delta = group[index] - mean;
    squares += delta * delta;
  }
  partial[threadIdx.x] = squares;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width /= 2) {
    if (threadIdx.x < width) {
      partial[threadIdx.x] += partial[threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    stats[2 * blockIdx.x] = mean;
    stats[2 * blockIdx.x + 1] =
        rsqrtf(partial[0] / static_cast<float>(count) + epsilon);
  }
}

__global__ void GroupApplySiluKernel(const float* input, const float* stats,
                                     const float* gamma, const float* beta,
                                     int channels, int channels_per_group,
                                     int plane, float* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(channels) * plane) {
    return;
  }
  const int channel = static_cast<int>(index / plane);
  const int group = channel / channels_per_group;
  const float normalized = (input[index] - stats[2 * group]) *
                               stats[2 * group + 1] * gamma[channel] +
                           beta[channel];
  output[index] = normalized / (1.0F + expf(-normalized));
}

unsigned Blocks(std::size_t elements, unsigned threads = 256) {
  return static_cast<unsigned>((elements + threads - 1) / threads);
}

bool ReadHost(const ModelInventory& inventory,
              const TensorDescriptor& descriptor, std::vector<float>* output,
              std::string* error) {
  if (descriptor.dtype != DType::kFloat32 ||
      descriptor.payload_bytes % sizeof(float) != 0) {
    SetError(error, "MiniMax H3 VisualVAE encoder tensor is not F32: " +
                        descriptor.name);
    return false;
  }
  output->resize(descriptor.payload_bytes / sizeof(float));
  const std::filesystem::path path = inventory.model_root() / descriptor.shard;
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open MiniMax H3 VisualVAE shard " + path.string() +
                        ": " + open_error);
    return false;
  }
  std::size_t done = 0;
  const std::size_t bytes = descriptor.payload_bytes;
  bool success = true;
  while (done < bytes) {
    const std::size_t result = platform::ReadFileAt(
        file, reinterpret_cast<std::byte*>(output->data()) + done, bytes - done,
        descriptor.file_offset + done);
    if (result == 0) {
      success = false;
      break;
    }
    done += result;
  }
  platform::CloseFile(file);
  if (!success) {
    SetError(error, "short read loading MiniMax H3 VisualVAE encoder tensor " +
                        descriptor.name);
  }
  return success;
}

struct Conv {
  int input_channels{0};
  int output_channels{0};
  int kernel{0};
  DeviceBuffer weight;  // [output][input * kernel * kernel], last time tap
  DeviceBuffer bias;
  // Temporal taps 0 and 1 of a 3x3x3 kernel in the same layout; loaded only
  // for multi-frame (guide clip) encoding.
  std::array<DeviceBuffer, 2> earlier_taps;
  bool temporal{false};
};

struct Norm {
  int channels{0};
  DeviceBuffer gamma;
  DeviceBuffer beta;
};

struct Resnet {
  Norm norm1;
  Conv conv1;
  Norm norm2;
  Conv conv2;
  bool has_shortcut{false};
  Conv shortcut;
};

struct Level {
  std::array<Resnet, 2> resnets;
  bool has_downsample{false};
  Conv downsample;
};

class Encoder {
public:
  // `temporal` also loads the earlier temporal taps of every 3x3x3 kernel
  // (needed by EncodeClipTile; three times the 3x3 weight memory).
  explicit Encoder(const ModelInventory& inventory, bool temporal = false)
      : inventory_(inventory), temporal_(temporal) {}
  ~Encoder() {
    if (stream_ != nullptr) {
      (void)hipStreamSynchronize(stream_);
    }
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
  }
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  bool Load(const CancellationToken* cancellation, std::string* error) {
    hipError_t status =
        hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    if (status != hipSuccess) {
      SetError(error, HipError("create H3 VisualVAE encoder stream", status));
      return false;
    }
    if (rocblas_create_handle(&handle_) != rocblas_status_success ||
        rocblas_set_stream(handle_, stream_) != rocblas_status_success) {
      SetError(error, "cannot create H3 VisualVAE encoder rocBLAS handle");
      return false;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
    if (!LoadConv("encoder.conv_in", 3, kLevelChannels[0], 3, &conv_in_,
                  error)) {
      return false;
    }
    int channels = kLevelChannels[0];
    for (int level = 0; level < kLevels; ++level) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 VisualVAE encoder load cancelled");
        return false;
      }
      const int output = kLevelChannels[static_cast<std::size_t>(level)];
      Level& target = levels_[static_cast<std::size_t>(level)];
      for (int index = 0; index < 2; ++index) {
        const std::string prefix = "encoder.down." + std::to_string(level) +
                                   ".block." + std::to_string(index) + ".";
        Resnet& resnet = target.resnets[static_cast<std::size_t>(index)];
        const int input = index == 0 ? channels : output;
        if (!LoadNorm(prefix + "norm1", input, &resnet.norm1, error) ||
            !LoadConv(prefix + "conv1", input, output, 3, &resnet.conv1,
                      error) ||
            !LoadNorm(prefix + "norm2", output, &resnet.norm2, error) ||
            !LoadConv(prefix + "conv2", output, output, 3, &resnet.conv2,
                      error)) {
          return false;
        }
        resnet.has_shortcut = input != output;
        if (resnet.has_shortcut &&
            !LoadConv(prefix + "nin_shortcut", input, output, 1,
                      &resnet.shortcut, error)) {
          return false;
        }
      }
      target.has_downsample = kLevelDownsample[static_cast<std::size_t>(level)];
      if (target.has_downsample &&
          !LoadConv(
              "encoder.down." + std::to_string(level) + ".downsample.conv",
              output, output, 3, &target.downsample, error)) {
        return false;
      }
      channels = output;
    }
    return LoadNorm("encoder.norm_out", channels, &norm_out_, error) &&
           LoadConv("encoder.conv_out", channels, kH3VideoVaeMomentChannels, 3,
                    &conv_out_, error) &&
           LoadConv("quant_conv", kH3VideoVaeMomentChannels,
                    kH3VideoVaeMomentChannels, 1, &quant_conv_, error) &&
           stats_.Allocate(kGroups * 2 * sizeof(float), error);
  }

  bool AllocateScratch(int tile_height, int tile_width, std::string* error,
                       int frames = 1) {
    const std::size_t plane =
        static_cast<std::size_t>(tile_height) * tile_width;
    // Frame-major activations; level 0 at full resolution is the largest.
    const std::size_t activation =
        plane * kLevelChannels[0] * static_cast<std::size_t>(frames);
    // conv_in reads 3 channels at full resolution; level 0 reads 128.
    const std::size_t columns = plane * kLevelChannels[0] * 9U;
    if (activation <= activation_elements_ && columns <= column_elements_) {
      return true;
    }
    for (DeviceBuffer& buffer : activations_) {
      if (!buffer.Allocate(activation * sizeof(float), error)) {
        return false;
      }
    }
    if (!columns_.Allocate(columns * sizeof(float), error)) {
      return false;
    }
    activation_elements_ = activation;
    column_elements_ = columns;
    return true;
  }

  [[nodiscard]] std::uint64_t weight_bytes() const noexcept {
    return weight_bytes_;
  }
  [[nodiscard]] std::uint64_t scratch_bytes() const noexcept {
    std::uint64_t bytes = columns_.bytes() + stats_.bytes();
    for (const DeviceBuffer& buffer : activations_) {
      bytes += buffer.bytes();
    }
    return bytes;
  }

  // Encodes one tile of `rgb_device` (packed RGB8 of the full canvas) into
  // 48 x (tile_height / 16) x (tile_width / 16) host moments.
  bool EncodeTile(const std::uint8_t* rgb_device, int image_width, int tile_x,
                  int tile_y, int tile_width, int tile_height,
                  std::vector<float>* moments, std::string* error) {
    float* x = activations_[0].As<float>();
    float* a = activations_[1].As<float>();
    float* b = activations_[2].As<float>();
    float* c = activations_[3].As<float>();
    const std::size_t plane =
        static_cast<std::size_t>(tile_width) * tile_height;
    NormalizeTileKernel<<<Blocks(plane), 256, 0, stream_>>>(
        rgb_device, image_width, tile_x, tile_y, tile_width, tile_height, a);
    int height = tile_height;
    int width = tile_width;
    if (!Convolve(conv_in_, a, height, width, 1, 1, x, error)) {
      return false;
    }
    for (Level& level : levels_) {
      for (Resnet& resnet : level.resnets) {
        // a = silu(norm1(x)); b = conv1(a); a = silu(norm2(b)); b = conv2(a)
        if (!NormSilu(resnet.norm1, x, height, width, a, error) ||
            !Convolve(resnet.conv1, a, height, width, 1, 1, b, error) ||
            !NormSilu(resnet.norm2, b, height, width, a, error) ||
            !Convolve(resnet.conv2, a, height, width, 1, 1, b, error)) {
          return false;
        }
        const std::size_t elements =
            static_cast<std::size_t>(resnet.conv2.output_channels) * height *
            width;
        if (resnet.has_shortcut) {
          if (!Convolve(resnet.shortcut, x, height, width, 1, 0, c, error)) {
            return false;
          }
          AddKernel<<<Blocks(elements), 256, 0, stream_>>>(c, b, elements);
          std::swap(x, c);
        } else {
          AddKernel<<<Blocks(elements), 256, 0, stream_>>>(x, b, elements);
        }
      }
      if (level.has_downsample) {
        // Right/bottom reflect pad of one, then a 3x3 stride-2 convolution.
        if (!Convolve(level.downsample, x, height, width, 2, 0, b, error)) {
          return false;
        }
        height /= 2;
        width /= 2;
        std::swap(x, b);
      }
    }
    if (!NormSilu(norm_out_, x, height, width, a, error) ||
        !Convolve(conv_out_, a, height, width, 1, 1, b, error) ||
        !Convolve(quant_conv_, b, height, width, 1, 0, c, error)) {
      return false;
    }
    moments->resize(static_cast<std::size_t>(kH3VideoVaeMomentChannels) *
                    height * width);
    hipError_t status = hipPeekAtLastError();
    if (status == hipSuccess) {
      status =
          hipMemcpyAsync(moments->data(), c, moments->size() * sizeof(float),
                         hipMemcpyDeviceToHost, stream_);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream_);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("H3 VisualVAE encoder tile", status));
      return false;
    }
    return true;
  }

  // Encodes one tile of a clip: `frames` frame-major RGB8 canvases at
  // `rgb_device` through the causal 3D encoder into host moments
  // [latent frame][48][tile_height / 16][tile_width / 16]. Activations are
  // frame-major ([frame][channels][height][width]) so every per-frame step
  // reuses the 2D kernels; GroupNorm statistics are per frame as in the
  // reference's TemporalIsolatedGroupNorm.
  bool EncodeClipTile(const std::uint8_t* rgb_device, int image_width,
                      int image_height, int frames, int tile_x, int tile_y,
                      int tile_width, int tile_height,
                      std::vector<float>* moments, int* latent_frames,
                      std::string* error) {
    float* x = activations_[0].As<float>();
    float* a = activations_[1].As<float>();
    float* b = activations_[2].As<float>();
    float* c = activations_[3].As<float>();
    const std::size_t plane =
        static_cast<std::size_t>(tile_width) * tile_height;
    const std::size_t image_bytes =
        static_cast<std::size_t>(image_width) * image_height * 3U;
    for (int frame = 0; frame < frames; ++frame) {
      NormalizeTileKernel<<<Blocks(plane), 256, 0, stream_>>>(
          rgb_device + static_cast<std::size_t>(frame) * image_bytes,
          image_width, tile_x, tile_y, tile_width, tile_height,
          a + static_cast<std::size_t>(frame) * 3U * plane);
    }
    int height = tile_height;
    int width = tile_width;
    int time = frames;
    if (!ConvolveFrames(conv_in_, a, time, height, width, 1, 1, 1, x, &time,
                        error)) {
      return false;
    }
    for (std::size_t level_index = 0; level_index < levels_.size();
         ++level_index) {
      Level& level = levels_[level_index];
      for (Resnet& resnet : level.resnets) {
        // a = silu(norm1(x)); b = conv1(a); a = silu(norm2(b)); b = conv2(a)
        if (!NormSiluFrames(resnet.norm1, x, time, height, width, a, error) ||
            !ConvolveFrames(resnet.conv1, a, time, height, width, 1, 1, 1, b,
                            nullptr, error) ||
            !NormSiluFrames(resnet.norm2, b, time, height, width, a, error) ||
            !ConvolveFrames(resnet.conv2, a, time, height, width, 1, 1, 1, b,
                            nullptr, error)) {
          return false;
        }
        const std::size_t elements =
            static_cast<std::size_t>(resnet.conv2.output_channels) * height *
            width * static_cast<std::size_t>(time);
        if (resnet.has_shortcut) {
          if (!ConvolveFrames(resnet.shortcut, x, time, height, width, 1, 0, 1,
                              c, nullptr, error)) {
            return false;
          }
          AddKernel<<<Blocks(elements), 256, 0, stream_>>>(c, b, elements);
          std::swap(x, c);
        } else {
          AddKernel<<<Blocks(elements), 256, 0, stream_>>>(x, b, elements);
        }
      }
      if (level.has_downsample) {
        // Right/bottom reflect pad of one, then a 3x3x3 convolution with
        // spatial stride 2 and the level's temporal stride.
        if (!ConvolveFrames(level.downsample, x, time, height, width, 2, 0,
                            kLevelTimeStride[level_index], b, &time, error)) {
          return false;
        }
        height /= 2;
        width /= 2;
        std::swap(x, b);
      }
    }
    if (!NormSiluFrames(norm_out_, x, time, height, width, a, error) ||
        !ConvolveFrames(conv_out_, a, time, height, width, 1, 1, 1, b, nullptr,
                        error) ||
        !ConvolveFrames(quant_conv_, b, time, height, width, 1, 0, 1, c,
                        nullptr, error)) {
      return false;
    }
    moments->resize(static_cast<std::size_t>(kH3VideoVaeMomentChannels) *
                    height * width * static_cast<std::size_t>(time));
    hipError_t status = hipPeekAtLastError();
    if (status == hipSuccess) {
      status =
          hipMemcpyAsync(moments->data(), c, moments->size() * sizeof(float),
                         hipMemcpyDeviceToHost, stream_);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream_);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("H3 VisualVAE clip encoder tile", status));
      return false;
    }
    *latent_frames = time;
    return true;
  }

  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }

private:
  bool LoadConv(const std::string& prefix, int input, int output, int kernel,
                Conv* conv, std::string* error) {
    const TensorDescriptor* weight = inventory_.FindTensor(prefix + ".weight");
    const TensorDescriptor* bias = inventory_.FindTensor(prefix + ".bias");
    const std::vector<std::uint64_t> expected = {
        static_cast<std::uint64_t>(output), static_cast<std::uint64_t>(input),
        static_cast<std::uint64_t>(kernel), static_cast<std::uint64_t>(kernel),
        static_cast<std::uint64_t>(kernel)};
    if (weight == nullptr || bias == nullptr || weight->shape != expected ||
        bias->shape !=
            std::vector<std::uint64_t>{static_cast<std::uint64_t>(output)}) {
      SetError(error, "MiniMax H3 VisualVAE encoder tensor contract differs: " +
                          prefix);
      return false;
    }
    std::vector<float> host;
    std::vector<float> bias_host;
    if (!ReadHost(inventory_, *weight, &host, error) ||
        !ReadHost(inventory_, *bias, &bias_host, error)) {
      return false;
    }
    // A single causal frame sees only the last temporal tap; clips also need
    // the earlier ones (time tap `tap` of weight[o][i][kt][ky][kx]).
    const std::size_t taps = static_cast<std::size_t>(kernel) * kernel;
    const auto time_tap = [&](int tap) {
      std::vector<float> spatial(static_cast<std::size_t>(output) * input *
                                 taps);
      for (std::size_t o = 0; o < static_cast<std::size_t>(output); ++o) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(input); ++i) {
          const std::size_t source =
              ((o * input + i) * kernel + static_cast<std::size_t>(tap)) *
              taps;
          std::copy_n(host.begin() + static_cast<std::ptrdiff_t>(source), taps,
                      spatial.begin() +
                          static_cast<std::ptrdiff_t>((o * input + i) * taps));
        }
      }
      return spatial;
    };
    conv->input_channels = input;
    conv->output_channels = output;
    conv->kernel = kernel;
    const std::vector<float> spatial = time_tap(kernel - 1);
    weight_bytes_ += (spatial.size() + bias_host.size()) * sizeof(float);
    if (!Upload(spatial, &conv->weight, error) ||
        !Upload(bias_host, &conv->bias, error)) {
      return false;
    }
    conv->temporal = temporal_ && kernel == 3;
    if (conv->temporal) {
      for (int tap = 0; tap < 2; ++tap) {
        const std::vector<float> earlier = time_tap(tap);
        weight_bytes_ += earlier.size() * sizeof(float);
        if (!Upload(earlier,
                    &conv->earlier_taps[static_cast<std::size_t>(tap)],
                    error)) {
          return false;
        }
      }
    }
    return true;
  }

  bool LoadNorm(const std::string& prefix, int channels, Norm* norm,
                std::string* error) {
    const TensorDescriptor* gamma = inventory_.FindTensor(prefix + ".weight");
    const TensorDescriptor* beta = inventory_.FindTensor(prefix + ".bias");
    const std::vector<std::uint64_t> expected = {
        static_cast<std::uint64_t>(channels)};
    if (gamma == nullptr || beta == nullptr || gamma->shape != expected ||
        beta->shape != expected) {
      SetError(error, "MiniMax H3 VisualVAE encoder tensor contract differs: " +
                          prefix);
      return false;
    }
    std::vector<float> gamma_host;
    std::vector<float> beta_host;
    norm->channels = channels;
    weight_bytes_ += 2U * channels * sizeof(float);
    return ReadHost(inventory_, *gamma, &gamma_host, error) &&
           ReadHost(inventory_, *beta, &beta_host, error) &&
           Upload(gamma_host, &norm->gamma, error) &&
           Upload(beta_host, &norm->beta, error);
  }

  bool Upload(const std::vector<float>& host, DeviceBuffer* device,
              std::string* error) {
    if (!device->Allocate(host.size() * sizeof(float), error)) {
      return false;
    }
    const hipError_t status =
        hipMemcpy(device->As<float>(), host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice);
    if (status != hipSuccess) {
      SetError(error, HipError("upload H3 VisualVAE encoder weight", status));
      return false;
    }
    return true;
  }

  bool NormSilu(const Norm& norm, const float* input, int height, int width,
                float* output, std::string* error) {
    const int plane = height * width;
    const int per_group = norm.channels / kGroups;
    GroupStatsKernel<<<kGroups, kNormThreads, 0, stream_>>>(
        input, per_group, plane, kGroupNormEpsilon, stats_.As<float>());
    GroupApplySiluKernel<<<Blocks(static_cast<std::size_t>(norm.channels) *
                                  plane),
                           256, 0, stream_>>>(
        input, stats_.As<float>(), norm.gamma.As<float>(),
        norm.beta.As<float>(), norm.channels, per_group, plane, output);
    const hipError_t status = hipPeekAtLastError();
    if (status != hipSuccess) {
      SetError(error, HipError("H3 VisualVAE encoder group norm", status));
      return false;
    }
    return true;
  }

  // stride 1 / pad 1: same-size 3x3. stride 2 / pad 0: downsample with a
  // right/bottom reflect pad (output = input / 2). kernel 1: pointwise.
  bool Convolve(const Conv& conv, const float* input, int height, int width,
                int stride, int pad, float* output, std::string* error) {
    return ConvolveTap(conv, conv.weight.As<float>(), input, height, width,
                       stride, pad, 0.0F, output, error) &&
           AddBias(conv, output, height, width, stride, error);
  }

  bool NormSiluFrames(const Norm& norm, const float* input, int frames,
                      int height, int width, float* output,
                      std::string* error) {
    const std::size_t elements =
        static_cast<std::size_t>(norm.channels) * height * width;
    for (int frame = 0; frame < frames; ++frame) {
      const std::size_t offset = static_cast<std::size_t>(frame) * elements;
      if (!NormSilu(norm, input + offset, height, width, output + offset,
                    error)) {
        return false;
      }
    }
    return true;
  }

  // Causal 3D convolution of `frames` frame-major inputs: two zero frames of
  // front padding (none for a single frame, where the reference truncates
  // the temporal taps, which gives the same result) and temporal stride
  // `time_stride`, so output frame o reads input frames o * stride - 2 + kt.
  // Pointwise kernels have no temporal extent. `out_frames` may be null.
  bool ConvolveFrames(const Conv& conv, const float* input, int frames,
                      int height, int width, int stride, int pad,
                      int time_stride, float* output, int* out_frames,
                      std::string* error) {
    const int out_height = conv.kernel == 1 ? height : height / stride;
    const int out_width = conv.kernel == 1 ? width : width / stride;
    const std::size_t in_elements =
        static_cast<std::size_t>(conv.input_channels) * height * width;
    const std::size_t out_elements =
        static_cast<std::size_t>(conv.output_channels) * out_height *
        out_width;
    if (frames < 1 || time_stride < 1 ||
        (conv.kernel == 1 && time_stride != 1)) {
      SetError(error, "invalid H3 VisualVAE clip convolution");
      return false;
    }
    const int produced = (frames - 1) / time_stride + 1;
    const int taps = conv.kernel == 3 ? 3 : 1;
    for (int frame = 0; frame < produced; ++frame) {
      float* target = output + static_cast<std::size_t>(frame) * out_elements;
      bool first = true;
      for (int tap = 0; tap < taps; ++tap) {
        const int source =
            conv.kernel == 3 ? frame * time_stride + tap - 2 : frame;
        if (source < 0) {
          continue;  // zero front padding
        }
        const float* weight = nullptr;
        if (conv.kernel != 3 || tap == 2) {
          weight = conv.weight.As<float>();
        } else if (conv.temporal) {
          weight = conv.earlier_taps[static_cast<std::size_t>(tap)].As<float>();
        } else {
          SetError(error,
                   "H3 VisualVAE encoder was loaded without temporal taps");
          return false;
        }
        if (!ConvolveTap(conv, weight,
                         input + static_cast<std::size_t>(source) * in_elements,
                         height, width, stride, pad, first ? 0.0F : 1.0F,
                         target, error)) {
          return false;
        }
        first = false;
      }
      if (!AddBias(conv, target, height, width, stride, error)) {
        return false;
      }
    }
    if (out_frames != nullptr) {
      *out_frames = produced;
    }
    return true;
  }

  bool AddBias(const Conv& conv, float* output, int height, int width,
               int stride, std::string* error) {
    const int out_height = conv.kernel == 1 ? height : height / stride;
    const int out_width = conv.kernel == 1 ? width : width / stride;
    const std::size_t outputs =
        static_cast<std::size_t>(out_height) * out_width;
    AddBiasKernel<<<Blocks(outputs * conv.output_channels), 256, 0, stream_>>>(
        output, conv.bias.As<float>(), conv.output_channels,
        static_cast<int>(outputs));
    const hipError_t launch = hipPeekAtLastError();
    if (launch != hipSuccess) {
      SetError(error, HipError("H3 VisualVAE encoder convolution", launch));
      return false;
    }
    return true;
  }

  // One temporal tap: output = im2col(input) x weight + beta * output,
  // without the bias.
  bool ConvolveTap(const Conv& conv, const float* weight, const float* input,
                   int height, int width, int stride, int pad, float beta,
                   float* output, std::string* error) {
    const int out_height = conv.kernel == 1 ? height : height / stride;
    const int out_width = conv.kernel == 1 ? width : width / stride;
    const std::size_t outputs =
        static_cast<std::size_t>(out_height) * out_width;
    const float* columns = input;
    int depth = conv.input_channels;
    if (conv.kernel == 3) {
      const std::size_t elements = outputs * conv.input_channels * 9U;
      if (elements > column_elements_) {
        SetError(error, "H3 VisualVAE encoder column scratch is too small");
        return false;
      }
      Im2Col3x3Kernel<<<Blocks(elements), 256, 0, stream_>>>(
          input, conv.input_channels, height, width, out_height, out_width,
          stride, pad, columns_.As<float>());
      columns = columns_.As<float>();
      depth *= 9;
    }
    const float alpha = 1.0F;
    const rocblas_status status =
        rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_none,
                      static_cast<rocblas_int>(outputs),
                      static_cast<rocblas_int>(conv.output_channels),
                      static_cast<rocblas_int>(depth), &alpha, columns,
                      static_cast<rocblas_int>(outputs), weight,
                      static_cast<rocblas_int>(depth), &beta, output,
                      static_cast<rocblas_int>(outputs));
    if (status != rocblas_status_success) {
      SetError(error, "H3 VisualVAE encoder convolution GEMM failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    const hipError_t launch = hipPeekAtLastError();
    if (launch != hipSuccess) {
      SetError(error, HipError("H3 VisualVAE encoder convolution", launch));
      return false;
    }
    return true;
  }

  const ModelInventory& inventory_;
  bool temporal_{false};
  hipStream_t stream_{nullptr};
  rocblas_handle handle_{nullptr};
  Conv conv_in_;
  std::array<Level, kLevels> levels_;
  Norm norm_out_;
  Conv conv_out_;
  Conv quant_conv_;
  std::array<DeviceBuffer, 4> activations_;
  DeviceBuffer columns_;
  DeviceBuffer stats_;
  std::size_t activation_elements_{0};
  std::size_t column_elements_{0};
  std::uint64_t weight_bytes_{0};
};

struct TileAxis {
  std::vector<int> starts;
  std::vector<int> lengths;
  std::vector<int> overlaps;
};

// diffusers AutoencoderKLMiniMaxH3._split_tiles.
TileAxis SplitTiles(int length) {
  TileAxis axis;
  const int tile = kH3VideoVaeEncoderTile;
  const int minimum = kH3VideoVaeEncoderTileOverlap;
  if (tile >= length) {
    axis.starts = {0};
    axis.lengths = {length};
    return axis;
  }
  int count = (length + tile - 1) / tile;
  while (tile * count - minimum * (count - 1) - length < 0) {
    ++count;
  }
  axis.overlaps.assign(static_cast<std::size_t>(count - 1), minimum);
  const int remaining = tile * count - minimum * (count - 1) - length;
  for (int index = 0; index < remaining / kSpatialRatio; ++index) {
    axis.overlaps[static_cast<std::size_t>(index % (count - 1))] +=
        kSpatialRatio;
  }
  axis.starts = {0};
  for (int index = 0; index + 1 < count; ++index) {
    axis.starts.push_back(axis.starts.back() + tile -
                          axis.overlaps[static_cast<std::size_t>(index)]);
  }
  axis.lengths.assign(static_cast<std::size_t>(count), tile);
  return axis;
}

struct LatentTile {
  int height{0};
  int width{0};
  std::vector<float> data;  // [48][height][width]
  [[nodiscard]] float& At(int channel, int y, int x) {
    return data[(static_cast<std::size_t>(channel) * height + y) * width + x];
  }
  [[nodiscard]] float At(int channel, int y, int x) const {
    return data[(static_cast<std::size_t>(channel) * height + y) * width + x];
  }
};

// diffusers _blend: the first `extent` rows/columns of `b` fade in over the
// last `extent` of `a`.
LatentTile Blend(const LatentTile& a, const LatentTile& b, int extent,
                 bool vertical) {
  LatentTile result = b;
  extent = std::min(
      {extent, vertical ? a.height : a.width, vertical ? b.height : b.width});
  for (int channel = 0; channel < kH3VideoVaeMomentChannels; ++channel) {
    for (int position = 0; position < extent; ++position) {
      const float weight_b =
          static_cast<float>(position) / static_cast<float>(extent);
      const float weight_a = 1.0F - weight_b;
      if (vertical) {
        for (int x = 0; x < b.width; ++x) {
          result.At(channel, position, x) =
              a.At(channel, a.height - extent + position, x) * weight_a +
              b.At(channel, position, x) * weight_b;
        }
      } else {
        for (int y = 0; y < b.height; ++y) {
          result.At(channel, y, position) =
              a.At(channel, y, a.width - extent + position) * weight_a +
              b.At(channel, y, position) * weight_b;
        }
      }
    }
  }
  return result;
}

KeyframeMoments StitchTiles(const std::vector<std::vector<LatentTile>>& tiles,
                            const TileAxis& y_axis, const TileAxis& x_axis,
                            int latent_height, int latent_width) {
  KeyframeMoments output;
  output.latent_height = latent_height;
  output.latent_width = latent_width;
  output.moments.assign(static_cast<std::size_t>(kH3VideoVaeMomentChannels) *
                            latent_height * latent_width,
                        0.0F);
  int row_offset = 0;
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    int column_offset = 0;
    int row_height = 0;
    for (std::size_t j = 0; j < tiles[i].size(); ++j) {
      LatentTile tile = tiles[i][j];
      if (i > 0) {
        tile = Blend(tiles[i - 1][j], tile,
                     y_axis.overlaps[i - 1] / kSpatialRatio, true);
      }
      if (j > 0) {
        tile = Blend(tiles[i][j - 1], tile,
                     x_axis.overlaps[j - 1] / kSpatialRatio, false);
      }
      const int keep_height =
          i + 1 < tiles.size()
              ? tile.height - y_axis.overlaps[i] / kSpatialRatio
              : tile.height;
      const int keep_width =
          j + 1 < tiles[i].size()
              ? tile.width - x_axis.overlaps[j] / kSpatialRatio
              : tile.width;
      for (int channel = 0; channel < kH3VideoVaeMomentChannels; ++channel) {
        for (int y = 0; y < keep_height; ++y) {
          for (int x = 0; x < keep_width; ++x) {
            output.moments[(static_cast<std::size_t>(channel) * latent_height +
                            row_offset + y) *
                               latent_width +
                           column_offset + x] = tile.At(channel, y, x);
          }
        }
      }
      column_offset += keep_width;
      row_height = keep_height;
    }
    row_offset += row_height;
  }
  return output;
}

}  // namespace

bool EncodeKeyframeMoments(const ModelInventory& inventory,
                           std::span<const KeyframeImage> images,
                           const CancellationToken* cancellation,
                           std::vector<KeyframeMoments>* output,
                           VideoVaeEncoderTelemetry* telemetry,
                           std::string* error) {
  if (output == nullptr || images.empty()) {
    SetError(error, "invalid MiniMax H3 keyframe encoder arguments");
    return false;
  }
  for (const KeyframeImage& image : images) {
    if (image.width < 32 || image.height < 32 || image.width % 32 != 0 ||
        image.height % 32 != 0 ||
        image.rgb.size() !=
            static_cast<std::size_t>(image.width) * image.height * 3U) {
      SetError(error,
               "MiniMax H3 keyframes must be RGB8 canvases with sides that "
               "are multiples of 32");
      return false;
    }
  }
  VideoVaeEncoderTelemetry local;
  const auto load_begin = Clock::now();
  Encoder encoder(inventory);
  if (!encoder.Load(cancellation, error)) {
    return false;
  }
  local.load_ms = Milliseconds(load_begin, Clock::now());
  local.weight_bytes = encoder.weight_bytes();
  const auto encode_begin = Clock::now();
  output->clear();
  for (const KeyframeImage& image : images) {
    const TileAxis y_axis = SplitTiles(image.height);
    const TileAxis x_axis = SplitTiles(image.width);
    if (!encoder.AllocateScratch(y_axis.lengths.front(), x_axis.lengths.front(),
                                 error)) {
      return false;
    }
    DeviceBuffer rgb;
    if (!rgb.Allocate(image.rgb.size(), error)) {
      return false;
    }
    hipError_t status = hipMemcpy(rgb.As<std::uint8_t>(), image.rgb.data(),
                                  image.rgb.size(), hipMemcpyHostToDevice);
    if (status != hipSuccess) {
      SetError(error, HipError("upload H3 keyframe", status));
      return false;
    }
    std::vector<std::vector<LatentTile>> tiles(y_axis.starts.size());
    for (std::size_t i = 0; i < y_axis.starts.size(); ++i) {
      for (std::size_t j = 0; j < x_axis.starts.size(); ++j) {
        if (IsCancelled(cancellation)) {
          SetError(error, "MiniMax H3 keyframe encoding cancelled");
          return false;
        }
        LatentTile tile;
        tile.height = y_axis.lengths[i] / kSpatialRatio;
        tile.width = x_axis.lengths[j] / kSpatialRatio;
        if (!encoder.EncodeTile(rgb.As<std::uint8_t>(), image.width,
                                x_axis.starts[j], y_axis.starts[i],
                                x_axis.lengths[j], y_axis.lengths[i],
                                &tile.data, error)) {
          return false;
        }
        tiles[i].push_back(std::move(tile));
        ++local.tiles;
      }
    }
    output->push_back(StitchTiles(tiles, y_axis, x_axis,
                                  image.height / kSpatialRatio,
                                  image.width / kSpatialRatio));
    for (const float value : output->back().moments) {
      if (!std::isfinite(value)) {
        SetError(error,
                 "MiniMax H3 keyframe encoder produced non-finite "
                 "moments");
        return false;
      }
    }
  }
  local.encode_ms = Milliseconds(encode_begin, Clock::now());
  local.scratch_bytes = encoder.scratch_bytes();
  if (telemetry != nullptr) {
    *telemetry = local;
  }
  return true;
}

bool EncodeGuideClipMoments(const ModelInventory& inventory, int width,
                            int height,
                            std::span<const std::uint8_t> rgb_frames,
                            int frames, const CancellationToken* cancellation,
                            std::vector<KeyframeMoments>* output,
                            VideoVaeEncoderTelemetry* telemetry,
                            std::string* error) {
  const std::size_t frame_bytes =
      static_cast<std::size_t>(std::max(width, 0)) *
      static_cast<std::size_t>(std::max(height, 0)) * 3U;
  if (output == nullptr || width < 32 || height < 32 || width % 32 != 0 ||
      height % 32 != 0 || frames < 5 || (frames - 5) % 17 != 0 ||
      rgb_frames.size() != frame_bytes * static_cast<std::size_t>(frames)) {
    SetError(error,
             "MiniMax H3 guide clips must be 17k + 5 RGB8 canvases with "
             "sides that are multiples of 32");
    return false;
  }
  VideoVaeEncoderTelemetry local;
  const auto load_begin = Clock::now();
  Encoder encoder(inventory, /*temporal=*/true);
  if (!encoder.Load(cancellation, error)) {
    return false;
  }
  local.load_ms = Milliseconds(load_begin, Clock::now());
  local.weight_bytes = encoder.weight_bytes();
  const auto encode_begin = Clock::now();
  const TileAxis y_axis = SplitTiles(height);
  const TileAxis x_axis = SplitTiles(width);
  if (!encoder.AllocateScratch(y_axis.lengths.front(), x_axis.lengths.front(),
                               error, kClipChunkFrames)) {
    return false;
  }
  DeviceBuffer rgb;
  if (!rgb.Allocate(frame_bytes * kClipChunkFrames, error)) {
    return false;
  }
  output->clear();
  // The reference's encode_temporal: 17-frame chunks, the last one padded by
  // repeating its final frame, each encoded on its own (no causal cache
  // across chunks), concatenated, then the last 3 latent frames dropped.
  const int chunks = (frames + kClipChunkFrames - 1) / kClipChunkFrames;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int first = chunk * kClipChunkFrames;
    std::vector<std::uint8_t> host(frame_bytes * kClipChunkFrames);
    for (int index = 0; index < kClipChunkFrames; ++index) {
      const int source = std::min(first + index, frames - 1);
      std::copy_n(rgb_frames.begin() +
                      static_cast<std::ptrdiff_t>(frame_bytes * source),
                  frame_bytes,
                  host.begin() +
                      static_cast<std::ptrdiff_t>(frame_bytes * index));
    }
    hipError_t status = hipMemcpy(rgb.As<std::uint8_t>(), host.data(),
                                  host.size(), hipMemcpyHostToDevice);
    if (status != hipSuccess) {
      SetError(error, HipError("upload H3 guide clip", status));
      return false;
    }
    // tiles[latent frame][row][column]
    std::vector<std::vector<std::vector<LatentTile>>> tiles;
    int chunk_latents = 0;
    for (std::size_t i = 0; i < y_axis.starts.size(); ++i) {
      for (std::size_t j = 0; j < x_axis.starts.size(); ++j) {
        if (IsCancelled(cancellation)) {
          SetError(error, "MiniMax H3 guide clip encoding cancelled");
          return false;
        }
        const int tile_height = y_axis.lengths[i] / kSpatialRatio;
        const int tile_width = x_axis.lengths[j] / kSpatialRatio;
        std::vector<float> moments;
        int latents = 0;
        if (!encoder.EncodeClipTile(rgb.As<std::uint8_t>(), width, height,
                                    kClipChunkFrames, x_axis.starts[j],
                                    y_axis.starts[i], x_axis.lengths[j],
                                    y_axis.lengths[i], &moments, &latents,
                                    error)) {
          return false;
        }
        if (tiles.empty()) {
          chunk_latents = latents;
          tiles.assign(static_cast<std::size_t>(latents),
                       std::vector<std::vector<LatentTile>>(
                           y_axis.starts.size()));
        }
        const std::size_t per_frame =
            static_cast<std::size_t>(kH3VideoVaeMomentChannels) * tile_height *
            tile_width;
        if (latents != chunk_latents ||
            moments.size() != per_frame * static_cast<std::size_t>(latents)) {
          SetError(error, "MiniMax H3 guide clip tiles differ in shape");
          return false;
        }
        for (int t = 0; t < latents; ++t) {
          LatentTile tile;
          tile.height = tile_height;
          tile.width = tile_width;
          tile.data.assign(
              moments.begin() + static_cast<std::ptrdiff_t>(per_frame * t),
              moments.begin() +
                  static_cast<std::ptrdiff_t>(per_frame * (t + 1)));
          tiles[static_cast<std::size_t>(t)][i].push_back(std::move(tile));
        }
        ++local.tiles;
      }
    }
    for (const auto& frame_tiles : tiles) {
      output->push_back(StitchTiles(frame_tiles, y_axis, x_axis,
                                    height / kSpatialRatio,
                                    width / kSpatialRatio));
    }
  }
  if (output->size() <= static_cast<std::size_t>(kClipTokenDrop)) {
    SetError(error, "MiniMax H3 guide clip produced too few latent frames");
    return false;
  }
  output->resize(output->size() - kClipTokenDrop);
  if (static_cast<int>(output->size()) != GuideClipLatentFrames(frames)) {
    SetError(error, "MiniMax H3 guide clip latent frame count differs: " +
                        std::to_string(output->size()));
    return false;
  }
  for (const KeyframeMoments& moments : *output) {
    for (const float value : moments.moments) {
      if (!std::isfinite(value)) {
        SetError(error,
                 "MiniMax H3 guide clip encoder produced non-finite moments");
        return false;
      }
    }
  }
  local.encode_ms = Milliseconds(encode_begin, Clock::now());
  local.scratch_bytes = encoder.scratch_bytes();
  if (telemetry != nullptr) {
    *telemetry = local;
  }
  return true;
}

}  // namespace gufo::minimax_h3

#endif  // defined(ENGINE_ENABLE_HIP)
