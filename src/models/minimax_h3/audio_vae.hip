#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/audio_vae_ops.hip.hpp"
#include "src/models/minimax_h3/json.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWeightStagingBytes = 16ULL << 20U;
constexpr std::uint32_t kLatentDimension = 2048;
constexpr std::uint32_t kDecoderDimension = 1024;
constexpr std::uint32_t kStages = 7;
constexpr std::uint32_t kResidualBlocks = 3;
constexpr std::uint32_t kResidualPairs = 3;
constexpr std::uint32_t kFilterSize = 12;
constexpr std::array<std::uint32_t, kStages> kUpsampleRates = {5, 5, 2, 2,
                                                               2, 2, 2};
constexpr std::array<std::uint32_t, kStages> kUpsampleKernels = {9, 9, 4, 4,
                                                                 4, 4, 4};
constexpr std::array<std::uint32_t, kResidualBlocks> kResidualKernels = {3, 7,
                                                                         11};
constexpr std::array<std::uint32_t, kResidualPairs> kResidualDilations = {1, 3,
                                                                          5};

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

bool CheckLaunch(std::string_view operation, std::string* error) {
  const hipError_t status = hipPeekAtLastError();
  if (status == hipSuccess) {
    return true;
  }
  SetError(error, HipError(operation, status));
  return false;
}

bool ValidateTarget(int* device, std::string* error) {
  hipError_t status = hipGetDevice(device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDevice", status));
    return false;
  }
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, *device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDeviceProperties", status));
    return false;
  }
  if (!std::string_view(properties.gcnArchName).starts_with("gfx1151") ||
      properties.integrated == 0) {
    SetError(error,
             "MiniMax H3 AudioVAE supports only integrated gfx1151 Gufo "
             "Halo; detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
}

using Faults = platform::PageFaults;
Faults ReadFaults() {
  return platform::ReadPageFaults();
}

std::uint64_t ReadSwapBytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  std::uint64_t kib = 0;
  std::string unit;
  while (status >> key) {
    if (key == "VmSwap:") {
      status >> kib >> unit;
      return kib * 1024ULL;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0;
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
      SetError(error, HipError("hipMalloc MiniMax H3 AudioVAE", status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
    }
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
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

bool HasShape(const TensorDescriptor& descriptor, DType dtype,
              std::span<const std::uint64_t> shape) {
  return descriptor.dtype == dtype &&
         descriptor.shape ==
             std::vector<std::uint64_t>(shape.begin(), shape.end());
}

const TensorDescriptor* RequireTensor(const ModelInventory& inventory,
                                      std::string_view name,
                                      std::span<const std::uint64_t> shape,
                                      std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error,
             "MiniMax H3 AudioVAE tensor is missing: " + std::string(name));
    return nullptr;
  }
  if (!HasShape(*descriptor, DType::kFloat32, shape)) {
    SetError(error, "MiniMax H3 AudioVAE tensor contract differs: " +
                        std::string(name));
    return nullptr;
  }
  return descriptor;
}

bool LoadTensor(const ModelInventory& inventory,
                const TensorDescriptor& descriptor, int device,
                hipStream_t stream, void* staging, std::size_t staging_bytes,
                const CancellationToken* cancellation, DeviceBuffer* output,
                std::string* error) {
  if (descriptor.payload_bytes > std::numeric_limits<std::size_t>::max() ||
      !output->Allocate(static_cast<std::size_t>(descriptor.payload_bytes),
                        error)) {
    return false;
  }
  const std::filesystem::path path = inventory.model_root() / descriptor.shard;
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open MiniMax H3 AudioVAE shard " + path.string() +
                        ": " + open_error);
    output->Reset();
    return false;
  }
  std::size_t copied = 0;
  bool success = true;
  while (copied < output->bytes()) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 AudioVAE weight load cancelled");
      success = false;
      break;
    }
    const std::size_t chunk = std::min(staging_bytes, output->bytes() - copied);
    std::size_t read_bytes = 0;
    while (read_bytes < chunk) {
      const std::size_t result = platform::ReadFileAt(
          file, static_cast<std::byte*>(staging) + read_bytes,
          chunk - read_bytes, descriptor.file_offset + copied + read_bytes);
      if (result == 0) {
        SetError(error, "short read loading MiniMax H3 AudioVAE tensor " +
                            descriptor.name);
        success = false;
        break;
      }
      read_bytes += static_cast<std::size_t>(result);
    }
    if (!success) {
      break;
    }
    hipError_t status = hipSetDevice(device);
    if (status == hipSuccess) {
      status = hipMemcpyAsync(static_cast<std::byte*>(output->data()) + copied,
                              staging, chunk, hipMemcpyHostToDevice, stream);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("copy MiniMax H3 AudioVAE tensor", status));
      success = false;
      break;
    }
    copied += chunk;
  }
  platform::CloseFile(file);
  if (!success) {
    output->Reset();
  }
  return success;
}

class F32ConvGemm {
public:
  F32ConvGemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
  }
  ~F32ConvGemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  F32ConvGemm(const F32ConvGemm&) = delete;
  F32ConvGemm& operator=(const F32ConvGemm&) = delete;

  bool Run(const float* weight, const float* columns, float* output,
           std::size_t batch_rows, std::size_t output_channels,
           std::size_t reduction, hipStream_t stream, std::string* error) {
    constexpr std::size_t kMaxInt =
        static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max());
    if (handle_ == nullptr || batch_rows == 0 || output_channels == 0 ||
        reduction == 0 || batch_rows > kMaxInt || output_channels > kMaxInt ||
        reduction > kMaxInt ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "invalid MiniMax H3 AudioVAE F32 convolution GEMM");
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const rocblas_status status = rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(output_channels),
        static_cast<rocblas_int>(batch_rows),
        static_cast<rocblas_int>(reduction), &alpha, weight,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(reduction), columns,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(reduction), &beta,
        output, rocblas_datatype_f32_r,
        static_cast<rocblas_int>(output_channels), output,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(output_channels),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
      SetError(error, "MiniMax H3 AudioVAE F32 convolution GEMM failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  rocblas_handle handle_{nullptr};
};

struct ConvWeights {
  DeviceBuffer weight;
  DeviceBuffer bias;
  std::uint32_t input_channels{0};
  std::uint32_t output_channels{0};
  std::uint32_t kernel{0};
  std::uint32_t padding{0};
  std::uint32_t dilation{1};
  std::uint32_t stride{1};
  bool transpose{false};
};

struct ActivationWeights {
  DeviceBuffer alpha;
  DeviceBuffer beta;
};

struct ResidualWeights {
  std::array<ActivationWeights, kResidualPairs * 2U> activations;
  std::array<ConvWeights, kResidualPairs> conv1;
  std::array<ConvWeights, kResidualPairs> conv2;
};

struct StageWeights {
  ConvWeights upsample;
  std::array<ResidualWeights, kResidualBlocks> residuals;
};

bool IsFinite(std::span<const float> values, std::string* first_non_finite) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      *first_non_finite = "audio_pcm[" + std::to_string(index) + "]";
      return false;
    }
  }
  return true;
}

}  // namespace

struct AudioVaeDecoder::Impl {
  const ModelInventory* inventory{nullptr};
  int latent_length{0};
  int sample_count{0};
  int device{0};
  hipStream_t stream{nullptr};
  void* staging{nullptr};

  DeviceBuffer latent_mean;
  DeviceBuffer latent_std;
  DeviceBuffer upsample_filter;
  DeviceBuffer downsample_filter;
  ConvWeights input_projection;
  ConvWeights pre;
  std::array<StageWeights, kStages> stages;
  ActivationWeights post_activation;
  ConvWeights post;

  DeviceBuffer normalized_latent;
  DeviceBuffer latent_rows;
  std::array<DeviceBuffer, 5> scratch;
  DeviceBuffer convolution_columns;
  DeviceBuffer activation_upsampled;
  DeviceBuffer channel_major_output;
  std::size_t scratch_elements{0};
  std::size_t convolution_column_elements{0};
  F32ConvGemm convolution_gemm;

  AudioVaeTelemetry base_telemetry;

  ~Impl() {
    if (staging != nullptr) {
      (void)hipHostFree(staging);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  bool CopyAsync(void* target, const void* source, std::size_t bytes,
                 hipMemcpyKind kind, std::string_view label,
                 std::string* error) {
    const hipError_t status =
        hipMemcpyAsync(target, source, bytes, kind, stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  }

  bool Load(const TensorDescriptor& descriptor, DeviceBuffer* output,
            const CancellationToken* cancellation, std::string* error,
            bool resident = true) {
    if (!LoadTensor(*inventory, descriptor, device, stream, staging,
                    kWeightStagingBytes, cancellation, output, error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes += descriptor.payload_bytes;
    if (resident) {
      base_telemetry.weight_bytes += descriptor.payload_bytes;
    }
    return true;
  }

  bool LoadNamed(std::string_view name,
                 std::initializer_list<std::uint64_t> shape,
                 DeviceBuffer* output, const CancellationToken* cancellation,
                 std::string* error, bool resident = true) {
    const std::vector<std::uint64_t> expected(shape);
    const TensorDescriptor* descriptor =
        RequireTensor(*inventory, name, expected, error);
    return descriptor != nullptr &&
           Load(*descriptor, output, cancellation, error, resident);
  }

  bool LoadActivation(std::string_view prefix, std::uint32_t channels,
                      ActivationWeights* activation,
                      const CancellationToken* cancellation,
                      std::string* error) {
    return LoadNamed(std::string(prefix) + ".act.alpha", {channels},
                     &activation->alpha, cancellation, error) &&
           LoadNamed(std::string(prefix) + ".act.beta", {channels},
                     &activation->beta, cancellation, error);
  }

  bool LoadPlainConv(std::string_view prefix, std::uint32_t input_channels,
                     std::uint32_t output_channels, std::uint32_t kernel,
                     std::uint32_t padding, std::uint32_t dilation,
                     bool has_bias, ConvWeights* conv,
                     const CancellationToken* cancellation,
                     std::string* error) {
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->padding = padding;
    conv->dilation = dilation;
    if (!LoadNamed(std::string(prefix) + ".weight",
                   {output_channels, input_channels, kernel}, &conv->weight,
                   cancellation, error)) {
      return false;
    }
    return !has_bias ||
           LoadNamed(std::string(prefix) + ".bias", {output_channels},
                     &conv->bias, cancellation, error);
  }

  bool LoadNormalizedConv(std::string_view prefix, std::uint32_t input_channels,
                          std::uint32_t output_channels, std::uint32_t kernel,
                          std::uint32_t padding, std::uint32_t dilation,
                          std::uint32_t stride, bool transpose, bool has_bias,
                          ConvWeights* conv,
                          const CancellationToken* cancellation,
                          std::string* error) {
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->stride = stride;
    conv->transpose = transpose;
    const std::uint32_t outer = transpose ? input_channels : output_channels;
    const std::uint32_t inner_channels =
        transpose ? output_channels : input_channels;
    DeviceBuffer vector;
    DeviceBuffer magnitude;
    if (!LoadNamed(std::string(prefix) + ".weight_v",
                   {outer, inner_channels, kernel}, &vector, cancellation,
                   error, false) ||
        !LoadNamed(std::string(prefix) + ".weight_g", {outer, 1, 1}, &magnitude,
                   cancellation, error, false) ||
        !conv->weight.Allocate(vector.bytes(), error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes += vector.bytes();
    if (transpose) {
      audio_vae_ops::LaunchWeightNormTranspose(
          vector.As<float>(), magnitude.As<float>(), conv->weight.As<float>(),
          input_channels, output_channels, kernel, stream);
    } else {
      audio_vae_ops::LaunchWeightNorm(vector.As<float>(), magnitude.As<float>(),
                                      conv->weight.As<float>(), outer,
                                      inner_channels * kernel, stream);
    }
    if (!CheckLaunch("normalize MiniMax H3 AudioVAE convolution", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error,
               HipError("synchronize MiniMax H3 AudioVAE weight norm", status));
      return false;
    }
    base_telemetry.weight_bytes += conv->weight.bytes();
    return !has_bias ||
           LoadNamed(std::string(prefix) + ".bias", {output_channels},
                     &conv->bias, cancellation, error);
  }

  bool LoadWeights(const CancellationToken* cancellation, std::string* error) {
    const auto begin = Clock::now();
    if (!LoadNamed("decoder.activation_post.upsample.filter",
                   {1, 1, kFilterSize}, &upsample_filter, cancellation,
                   error) ||
        !LoadNamed("decoder.activation_post.downsample.lowpass.filter",
                   {1, 1, kFilterSize}, &downsample_filter, cancellation,
                   error) ||
        !LoadPlainConv("dec_in_proj", kH3AudioVaeLatentChannels,
                       kLatentDimension, 1, 0, 1, true, &input_projection,
                       cancellation, error) ||
        !LoadNormalizedConv("decoder.conv_pre", kLatentDimension,
                            kDecoderDimension, 7, 3, 1, 1, false, true, &pre,
                            cancellation, error)) {
      return false;
    }
    for (std::uint32_t stage = 0; stage < kStages; ++stage) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 AudioVAE load cancelled before stage " +
                            std::to_string(stage));
        return false;
      }
      const std::uint32_t input_channels = kDecoderDimension >> stage;
      const std::uint32_t output_channels = kDecoderDimension >> (stage + 1U);
      const std::uint32_t kernel = kUpsampleKernels[stage];
      const std::uint32_t stride = kUpsampleRates[stage];
      const std::uint32_t padding = (kernel - stride) / 2U;
      StageWeights& weights = stages[stage];
      const std::string upsample =
          "decoder.ups." + std::to_string(stage) + ".0";
      if (!LoadNormalizedConv(upsample, input_channels, output_channels, kernel,
                              padding, 1, stride, true, true, &weights.upsample,
                              cancellation, error)) {
        return false;
      }
      for (std::uint32_t block = 0; block < kResidualBlocks; ++block) {
        const std::uint32_t global = stage * kResidualBlocks + block;
        ResidualWeights& residual = weights.residuals[block];
        const std::uint32_t residual_kernel = kResidualKernels[block];
        for (std::uint32_t pair = 0; pair < kResidualPairs; ++pair) {
          for (std::uint32_t which = 0; which < 2; ++which) {
            const std::uint32_t activation = pair * 2U + which;
            const std::string prefix =
                "decoder.resblocks." + std::to_string(global) +
                ".activations." + std::to_string(activation);
            if (!LoadActivation(prefix, output_channels,
                                &residual.activations[activation], cancellation,
                                error)) {
              return false;
            }
          }
          const std::uint32_t dilation = kResidualDilations[pair];
          const std::uint32_t dilated_padding =
              dilation * (residual_kernel - 1U) / 2U;
          const std::string conv1 = "decoder.resblocks." +
                                    std::to_string(global) + ".convs1." +
                                    std::to_string(pair);
          const std::string conv2 = "decoder.resblocks." +
                                    std::to_string(global) + ".convs2." +
                                    std::to_string(pair);
          if (!LoadNormalizedConv(conv1, output_channels, output_channels,
                                  residual_kernel, dilated_padding, dilation, 1,
                                  false, true, &residual.conv1[pair],
                                  cancellation, error) ||
              !LoadNormalizedConv(conv2, output_channels, output_channels,
                                  residual_kernel, (residual_kernel - 1U) / 2U,
                                  1, 1, false, true, &residual.conv2[pair],
                                  cancellation, error)) {
            return false;
          }
        }
      }
    }
    if (!LoadActivation("decoder.activation_post", 8, &post_activation,
                        cancellation, error) ||
        !LoadNormalizedConv("decoder.conv_post", 8, 1, 7, 3, 1, 1, false, false,
                            &post, cancellation, error)) {
      return false;
    }
    base_telemetry.load_ms = Milliseconds(begin, Clock::now());
    return true;
  }

  bool ReadLatentNormalization(
      std::array<float, kH3AudioVaeLatentChannels>* mean,
      std::array<float, kH3AudioVaeLatentChannels>* deviation,
      std::string* error) {
    try {
      const json::Value config = json::ParseFile(
          inventory->model_root() / "FL2VA/audio_vae/config.json", 1U << 20U);
      const json::Value* mean_value = config.Find("latents_mean");
      const json::Value* std_value = config.Find("latents_std");
      if (mean_value == nullptr || std_value == nullptr ||
          !mean_value->IsArray() || !std_value->IsArray() ||
          mean_value->AsArray().size() != mean->size() ||
          std_value->AsArray().size() != deviation->size()) {
        throw json::Error("latent normalization shape differs");
      }
      for (std::size_t index = 0; index < mean->size(); ++index) {
        (*mean)[index] =
            static_cast<float>(mean_value->AsArray()[index].AsDouble());
        (*deviation)[index] =
            static_cast<float>(std_value->AsArray()[index].AsDouble());
        if (!std::isfinite((*mean)[index]) ||
            !std::isfinite((*deviation)[index]) ||
            (*deviation)[index] <= 0.0F) {
          throw json::Error("latent normalization contains invalid values");
        }
      }
      return true;
    } catch (const std::exception& exception) {
      SetError(error, "MiniMax H3 AudioVAE config failed: " +
                          std::string(exception.what()));
      return false;
    }
  }

  bool AllocateRuntime(
      const std::array<float, kH3AudioVaeLatentChannels>& mean,
      const std::array<float, kH3AudioVaeLatentChannels>& deviation,
      std::string* error) {
    std::uint64_t length = static_cast<std::uint64_t>(latent_length);
    const auto add_column_shape = [this](std::uint64_t output_length,
                                         std::uint64_t input_channels,
                                         std::uint64_t kernel) {
      const std::uint64_t elements =
          kH3AudioVaeStereoChannels * output_length * input_channels * kernel;
      if (elements <= std::numeric_limits<std::size_t>::max()) {
        convolution_column_elements = std::max(
            convolution_column_elements, static_cast<std::size_t>(elements));
      }
    };
    add_column_shape(length, kH3AudioVaeLatentChannels, 1);
    add_column_shape(length, kLatentDimension, 7);
    scratch_elements = static_cast<std::size_t>(kH3AudioVaeStereoChannels) *
                       latent_length * kLatentDimension;
    std::uint32_t channels = kDecoderDimension;
    for (std::uint32_t stage = 0; stage < kStages; ++stage) {
      const std::uint32_t input_channels = channels;
      length *= kUpsampleRates[stage];
      channels >>= 1U;
      add_column_shape(length, input_channels, kUpsampleKernels[stage]);
      for (std::uint32_t kernel : kResidualKernels) {
        add_column_shape(length, channels, kernel);
      }
      const std::uint64_t elements =
          static_cast<std::uint64_t>(kH3AudioVaeStereoChannels) * length *
          channels;
      if (elements > std::numeric_limits<std::size_t>::max()) {
        SetError(error, "MiniMax H3 AudioVAE scratch size overflows");
        return false;
      }
      scratch_elements =
          std::max(scratch_elements, static_cast<std::size_t>(elements));
    }
    add_column_shape(length, 8, 7);
    if (convolution_column_elements == 0) {
      SetError(error, "MiniMax H3 AudioVAE convolution workspace overflows");
      return false;
    }
    const auto f32 = [](std::size_t elements) {
      return elements * sizeof(float);
    };
    const std::size_t latent_elements =
        static_cast<std::size_t>(kH3AudioVaeLatentChannels) *
        kH3AudioVaeStereoChannels * latent_length;
    const std::size_t row_elements =
        static_cast<std::size_t>(kH3AudioVaeStereoChannels) * latent_length *
        kH3AudioVaeLatentChannels;
    if (!latent_mean.Allocate(f32(mean.size()), error) ||
        !latent_std.Allocate(f32(deviation.size()), error) ||
        !normalized_latent.Allocate(f32(latent_elements), error) ||
        !latent_rows.Allocate(f32(row_elements), error) ||
        !convolution_columns.Allocate(f32(convolution_column_elements),
                                      error) ||
        !activation_upsampled.Allocate(f32(scratch_elements * 2U), error) ||
        !channel_major_output.Allocate(
            f32(static_cast<std::size_t>(kH3AudioVaeStereoChannels) *
                sample_count),
            error)) {
      return false;
    }
    for (DeviceBuffer& buffer : scratch) {
      if (!buffer.Allocate(f32(scratch_elements), error)) {
        return false;
      }
    }
    if (!CopyAsync(latent_mean.data(), mean.data(), f32(mean.size()),
                   hipMemcpyHostToDevice, "copy H3 AudioVAE latent mean",
                   error) ||
        !CopyAsync(latent_std.data(), deviation.data(), f32(deviation.size()),
                   hipMemcpyHostToDevice, "copy H3 AudioVAE latent deviation",
                   error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 AudioVAE setup", status));
      return false;
    }
    const std::uint64_t runtime_bytes =
        latent_mean.bytes() + latent_std.bytes() + normalized_latent.bytes() +
        latent_rows.bytes() + convolution_columns.bytes() +
        activation_upsampled.bytes() + channel_major_output.bytes() +
        static_cast<std::uint64_t>(scratch.size()) * scratch.front().bytes();
    base_telemetry.persistent_bytes = base_telemetry.weight_bytes +
                                      latent_mean.bytes() + latent_std.bytes() +
                                      normalized_latent.bytes();
    base_telemetry.scratch_bytes = runtime_bytes - latent_mean.bytes() -
                                   latent_std.bytes() -
                                   normalized_latent.bytes();
    base_telemetry.peak_live_bytes =
        base_telemetry.persistent_bytes + base_telemetry.scratch_bytes;
    base_telemetry.cumulative_allocation_bytes += runtime_bytes;
    base_telemetry.scratch_address =
        reinterpret_cast<std::uintptr_t>(scratch.front().data());
    base_telemetry.swap_bytes = ReadSwapBytes();
    return true;
  }

  bool RunConv(const ConvWeights& conv, const float* input, float* output,
               std::uint32_t input_length, std::uint32_t output_length,
               std::string* error) {
    const std::size_t rows =
        static_cast<std::size_t>(kH3AudioVaeStereoChannels) * output_length;
    const std::size_t reduction =
        static_cast<std::size_t>(conv.input_channels) * conv.kernel;
    if (rows > std::numeric_limits<std::size_t>::max() / reduction ||
        rows * reduction > convolution_column_elements) {
      SetError(error, "MiniMax H3 AudioVAE convolution workspace is too small");
      return false;
    }
    if (conv.transpose) {
      audio_vae_ops::LaunchConvTranspose1dIm2Col(
          input, convolution_columns.As<float>(), kH3AudioVaeStereoChannels,
          input_length, conv.input_channels, output_length, conv.kernel,
          conv.stride, conv.padding, stream);
    } else {
      audio_vae_ops::LaunchConv1dIm2Col(
          input, convolution_columns.As<float>(), kH3AudioVaeStereoChannels,
          input_length, conv.input_channels, output_length, conv.kernel,
          conv.padding, conv.dilation, stream);
    }
    ++base_telemetry.dispatches;
    if (!CheckLaunch(conv.transpose ? "H3 AudioVAE transposed im2col"
                                    : "H3 AudioVAE im2col",
                     error) ||
        !convolution_gemm.Run(conv.weight.As<float>(),
                              convolution_columns.As<float>(), output, rows,
                              conv.output_channels, reduction, stream, error)) {
      return false;
    }
    ++base_telemetry.dispatches;
    if (conv.bias.bytes() != 0) {
      audio_vae_ops::LaunchAddConvBias(output, conv.bias.As<float>(), rows,
                                       conv.output_channels, stream);
      ++base_telemetry.dispatches;
      if (!CheckLaunch("H3 AudioVAE convolution bias", error)) {
        return false;
      }
    }
    return true;
  }

  bool RunActivation(const ActivationWeights& activation, const float* input,
                     float* output, std::uint32_t length,
                     std::uint32_t channels, std::string* error) {
    const std::size_t elements =
        static_cast<std::size_t>(kH3AudioVaeStereoChannels) * length * channels;
    if (elements > activation_upsampled.bytes() / (2U * sizeof(float))) {
      SetError(error, "MiniMax H3 AudioVAE activation workspace is too small");
      return false;
    }
    audio_vae_ops::LaunchAliasFreeSnakeUpsample(
        input, activation.alpha.As<float>(), activation.beta.As<float>(),
        upsample_filter.As<float>(), activation_upsampled.As<float>(),
        kH3AudioVaeStereoChannels, length, channels, stream);
    ++base_telemetry.dispatches;
    if (!CheckLaunch("H3 AudioVAE alias-free SnakeBeta upsample", error)) {
      return false;
    }
    audio_vae_ops::LaunchAliasFreeSnakeDownsample(
        activation_upsampled.As<float>(), downsample_filter.As<float>(), output,
        kH3AudioVaeStereoChannels, length, channels, stream);
    ++base_telemetry.dispatches;
    return CheckLaunch("H3 AudioVAE alias-free SnakeBeta downsample", error);
  }

  bool RunStage(std::uint32_t stage_index, std::uint32_t* length,
                std::size_t* hidden_index,
                const CancellationToken* cancellation, std::string* error) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 AudioVAE cancelled before stage " +
                          std::to_string(stage_index));
      return false;
    }
    const auto begin = Clock::now();
    const StageWeights& weights = stages[stage_index];
    const std::uint32_t output_length = *length * kUpsampleRates[stage_index];
    const std::uint32_t channels = kDecoderDimension >> (stage_index + 1U);
    const std::size_t elements =
        static_cast<std::size_t>(kH3AudioVaeStereoChannels) * output_length *
        channels;

    std::array<std::size_t, 5> order{};
    std::size_t next = 0;
    for (std::size_t index = 0; index < scratch.size(); ++index) {
      if (index != *hidden_index) {
        order[next++] = index;
      }
    }
    order[next] = *hidden_index;
    const std::size_t upsample_index = order[0];
    const std::size_t sum_index = order[1];
    const std::size_t work_index = order[2];
    const std::size_t activated_index = order[3];
    const std::size_t branch_index = order[4];
    float* upsampled = scratch[upsample_index].As<float>();
    float* sum = scratch[sum_index].As<float>();
    float* work = scratch[work_index].As<float>();
    float* activated = scratch[activated_index].As<float>();
    float* branch = scratch[branch_index].As<float>();

    if (!RunConv(weights.upsample, scratch[*hidden_index].As<float>(),
                 upsampled, *length, output_length, error)) {
      return false;
    }
    for (std::uint32_t block = 0; block < kResidualBlocks; ++block) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 AudioVAE cancelled in stage " +
                            std::to_string(stage_index) + " block " +
                            std::to_string(block));
        return false;
      }
      float* target = block == 0 ? sum : work;
      audio_vae_ops::LaunchCopy(upsampled, target, elements, stream);
      ++base_telemetry.dispatches;
      const ResidualWeights& residual = weights.residuals[block];
      for (std::uint32_t pair = 0; pair < kResidualPairs; ++pair) {
        if (!RunActivation(residual.activations[pair * 2U], target, activated,
                           output_length, channels, error) ||
            !RunConv(residual.conv1[pair], activated, branch, output_length,
                     output_length, error) ||
            !RunActivation(residual.activations[pair * 2U + 1U], branch,
                           activated, output_length, channels, error) ||
            !RunConv(residual.conv2[pair], activated, branch, output_length,
                     output_length, error)) {
          return false;
        }
        audio_vae_ops::LaunchAddScaled(target, branch, target, elements, 1.0F,
                                       1.0F, stream);
        ++base_telemetry.dispatches;
      }
      if (block == 1) {
        audio_vae_ops::LaunchAddScaled(sum, work, sum, elements, 1.0F, 1.0F,
                                       stream);
        ++base_telemetry.dispatches;
      } else if (block == 2) {
        audio_vae_ops::LaunchAddScaled(sum, work, sum, elements, 1.0F / 3.0F,
                                       1.0F / 3.0F, stream);
        ++base_telemetry.dispatches;
      }
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error,
               HipError("synchronize MiniMax H3 AudioVAE stage", status));
      return false;
    }
    *hidden_index = sum_index;
    *length = output_length;
    base_telemetry.stage_ms.push_back(Milliseconds(begin, Clock::now()));
    return true;
  }

  bool Execute(std::span<const float> normalized,
               const CancellationToken* cancellation, AudioVaeProgress progress,
               void* progress_opaque, AudioWaveform* output,
               AudioVaeTelemetry* telemetry, std::string* error) {
    const std::size_t expected =
        static_cast<std::size_t>(kH3AudioVaeLatentChannels) *
        kH3AudioVaeStereoChannels * latent_length;
    if (normalized.size() != expected || output == nullptr) {
      SetError(error, "invalid MiniMax H3 AudioVAE decode arguments");
      return false;
    }
    *output = {};
    base_telemetry.stage_ms.clear();
    base_telemetry.first_non_finite.clear();
    base_telemetry.dispatches = 0;
    const Faults before_faults = ReadFaults();
    const auto begin = Clock::now();
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 AudioVAE cancelled before decode");
      return false;
    }
    if (!CopyAsync(normalized_latent.data(), normalized.data(),
                   normalized_latent.bytes(), hipMemcpyHostToDevice,
                   "copy H3 AudioVAE normalized latent", error)) {
      return false;
    }
    audio_vae_ops::LaunchPrepareLatent(
        normalized_latent.As<float>(), latent_mean.As<float>(),
        latent_std.As<float>(), latent_rows.As<float>(), latent_length, stream);
    ++base_telemetry.dispatches;
    if (!CheckLaunch("prepare H3 AudioVAE latent", error) ||
        !RunConv(input_projection, latent_rows.As<float>(),
                 scratch[0].As<float>(), latent_length, latent_length, error) ||
        !RunConv(pre, scratch[0].As<float>(), scratch[1].As<float>(),
                 latent_length, latent_length, error)) {
      return false;
    }
    std::size_t hidden_index = 1;
    std::uint32_t length = static_cast<std::uint32_t>(latent_length);
    for (std::uint32_t stage = 0; stage < kStages; ++stage) {
      if (!RunStage(stage, &length, &hidden_index, cancellation, error)) {
        return false;
      }
      if (progress != nullptr) {
        progress(static_cast<int>(stage + 1U), static_cast<int>(kStages),
                 progress_opaque);
      }
    }
    std::size_t activated_index = hidden_index == 0 ? 1 : 0;
    std::size_t waveform_index =
        hidden_index != 2 && activated_index != 2 ? 2 : 3;
    if (!RunActivation(post_activation, scratch[hidden_index].As<float>(),
                       scratch[activated_index].As<float>(), length, 8,
                       error) ||
        !RunConv(post, scratch[activated_index].As<float>(),
                 scratch[waveform_index].As<float>(), length, length, error)) {
      return false;
    }
    audio_vae_ops::LaunchClipAndReorder(scratch[waveform_index].As<float>(),
                                        channel_major_output.As<float>(),
                                        length, stream);
    ++base_telemetry.dispatches;
    if (!CheckLaunch("clip H3 AudioVAE waveform", error)) {
      return false;
    }
    output->pcm.resize(static_cast<std::size_t>(kH3AudioVaeStereoChannels) *
                       length);
    if (!CopyAsync(output->pcm.data(), channel_major_output.data(),
                   output->pcm.size() * sizeof(float), hipMemcpyDeviceToHost,
                   "read H3 AudioVAE waveform", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error,
               HipError("synchronize MiniMax H3 AudioVAE decode", status));
      return false;
    }
    output->channels = kH3AudioVaeStereoChannels;
    output->samples = static_cast<int>(length);
    output->sample_rate = kH3AudioVaeSampleRate;
    base_telemetry.decode_ms = Milliseconds(begin, Clock::now());
    const Faults after_faults = ReadFaults();
    base_telemetry.minor_page_faults =
        after_faults.minor >= before_faults.minor
            ? after_faults.minor - before_faults.minor
            : 0;
    base_telemetry.major_page_faults =
        after_faults.major >= before_faults.major
            ? after_faults.major - before_faults.major
            : 0;
    base_telemetry.swap_bytes =
        std::max(base_telemetry.swap_bytes, ReadSwapBytes());
    if (!IsFinite(output->pcm, &base_telemetry.first_non_finite)) {
      SetError(error, "MiniMax H3 AudioVAE produced non-finite PCM at " +
                          base_telemetry.first_non_finite);
      *output = {};
      return false;
    }
    if (telemetry != nullptr) {
      *telemetry = base_telemetry;
    }
    return true;
  }
};

AudioVaeDecoder::AudioVaeDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AudioVaeDecoder::~AudioVaeDecoder() = default;
AudioVaeDecoder::AudioVaeDecoder(AudioVaeDecoder&&) noexcept = default;
AudioVaeDecoder& AudioVaeDecoder::operator=(AudioVaeDecoder&&) noexcept =
    default;

std::unique_ptr<AudioVaeDecoder> AudioVaeDecoder::Create(
    const ModelInventory& inventory, int latent_frames,
    const CancellationToken* cancellation, AudioVaeTelemetry* telemetry,
    std::string* error) {
  const Faults faults_before = ReadFaults();
  if (latent_frames < 1 ||
      latent_frames > std::numeric_limits<int>::max() / kH3AudioVaeHopLength) {
    SetError(error, "invalid MiniMax H3 AudioVAE latent frame count");
    return nullptr;
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 AudioVAE creation cancelled");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->inventory = &inventory;
  impl->latent_length = latent_frames;
  impl->sample_count = latent_frames * kH3AudioVaeHopLength;
  if (!ValidateTarget(&impl->device, error)) {
    return nullptr;
  }
  hipError_t status = hipSetDevice(impl->device);
  if (status == hipSuccess) {
    status = hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
  }
  if (status == hipSuccess) {
    status = hipHostMalloc(&impl->staging, kWeightStagingBytes,
                           hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    SetError(error, HipError("initialize MiniMax H3 AudioVAE", status));
    return nullptr;
  }
  std::array<float, kH3AudioVaeLatentChannels> mean{};
  std::array<float, kH3AudioVaeLatentChannels> deviation{};
  if (!impl->ReadLatentNormalization(&mean, &deviation, error) ||
      !impl->LoadWeights(cancellation, error) ||
      !impl->AllocateRuntime(mean, deviation, error)) {
    return nullptr;
  }
  status = hipHostFree(impl->staging);
  if (status != hipSuccess) {
    SetError(error,
             HipError("release MiniMax H3 AudioVAE weight staging", status));
    return nullptr;
  }
  impl->staging = nullptr;
  const Faults faults_after = ReadFaults();
  impl->base_telemetry.minor_page_faults =
      faults_after.minor >= faults_before.minor
          ? faults_after.minor - faults_before.minor
          : 0;
  impl->base_telemetry.major_page_faults =
      faults_after.major >= faults_before.major
          ? faults_after.major - faults_before.major
          : 0;
  if (telemetry != nullptr) {
    *telemetry = impl->base_telemetry;
  }
  return std::unique_ptr<AudioVaeDecoder>(new AudioVaeDecoder(std::move(impl)));
}

bool AudioVaeDecoder::Decode(std::span<const float> normalized_latent,
                             const CancellationToken* cancellation,
                             AudioVaeProgress progress, void* progress_opaque,
                             AudioWaveform* output,
                             AudioVaeTelemetry* telemetry, std::string* error) {
  if (impl_ == nullptr) {
    SetError(error, "MiniMax H3 AudioVAE decoder is empty");
    return false;
  }
  return impl_->Execute(normalized_latent, cancellation, progress,
                        progress_opaque, output, telemetry, error);
}

int AudioVaeDecoder::latent_frames() const noexcept {
  return impl_ == nullptr ? 0 : impl_->latent_length;
}

int AudioVaeDecoder::samples() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

// ---------------------------------------------------------------------------
// AudioVAE encoder (audio guides). DAC encoder on the GPU (im2col + rocBLAS
// F32 GEMM, as the decoder), then the small attention projection head on the
// host. Reference: ComfyUI comfy/ldm/minimax/audio_vae.py (encode) and the
// released FL2VA/audio_vae dac_audio_vae.py / dac_attn_proj.py.

namespace {

constexpr std::uint32_t kEncoderBaseChannels = 64;
constexpr std::array<std::uint32_t, 5> kEncoderStrides = {2, 4, 4, 5, 5};
constexpr std::array<std::uint32_t, 3> kEncoderDilations = {1, 3, 9};
constexpr std::uint32_t kEncoderAttentionHeads = 8;
constexpr std::uint32_t kEncoderMlpHidden = 64;
constexpr std::size_t kEncoderColumnElements = 32ULL << 20U;
constexpr float kLayerNormEpsilon = 1.0e-5F;

struct EncoderConv {
  DeviceBuffer weight;
  DeviceBuffer bias;
  std::uint32_t input_channels{0};
  std::uint32_t output_channels{0};
  std::uint32_t kernel{1};
  std::uint32_t stride{1};
  std::uint32_t padding{0};
  std::uint32_t dilation{1};

  [[nodiscard]] std::uint32_t OutputLength(std::uint32_t length) const {
    const std::int64_t span =
        static_cast<std::int64_t>(length) + 2 * padding -
        static_cast<std::int64_t>(dilation) * (kernel - 1) - 1;
    return span < 0 ? 0 : static_cast<std::uint32_t>(span / stride + 1);
  }
};

struct EncoderResidual {
  DeviceBuffer alpha1;
  DeviceBuffer alpha2;
  EncoderConv conv1;
  EncoderConv conv2;
};

struct EncoderBlock {
  std::array<EncoderResidual, 3> residuals;
  DeviceBuffer alpha;
  EncoderConv down;
};

// Host copies of the attention projection head (pre_block) and mean_proj.
struct EncoderHead {
  std::vector<float> norm1_weight, norm1_bias, norm3_weight, norm3_bias;
  std::vector<float> q_bias, v_bias;
  std::vector<float> attention_proj_weight, attention_proj_bias;
  std::vector<float> proj_weight, proj_bias;
  std::vector<float> norm2_weight, norm2_bias;
  std::vector<float> mlp_norm_weight, mlp_norm_bias;
  std::vector<float> w0_weight, w0_bias, w1_weight, w1_bias, w2_weight, w2_bias;
  std::vector<float> mean_weight, mean_bias;
};

void LayerNormRows(const float* input, float* output, std::size_t rows,
                   std::size_t width, const std::vector<float>& weight,
                   const std::vector<float>& bias) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* x = input + row * width;
    float* y = output + row * width;
    double sum = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
      sum += x[index];
    }
    const double mean = sum / static_cast<double>(width);
    double variance = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
      const double centered = x[index] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(width);
    const double scale = 1.0 / std::sqrt(variance + kLayerNormEpsilon);
    for (std::size_t index = 0; index < width; ++index) {
      y[index] = static_cast<float>((x[index] - mean) * scale) * weight[index] +
                 bias[index];
    }
  }
}

// y[rows][out] = x[rows][in] * weight[out][in]^T + bias.
void LinearRows(const float* input, float* output, std::size_t rows,
                std::size_t input_width, std::size_t output_width,
                const std::vector<float>& weight,
                const std::vector<float>& bias) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* x = input + row * input_width;
    for (std::size_t out = 0; out < output_width; ++out) {
      const float* w = weight.data() + out * input_width;
      double sum = bias.empty() ? 0.0 : bias[out];
      for (std::size_t index = 0; index < input_width; ++index) {
        sum += static_cast<double>(x[index]) * w[index];
      }
      output[row * output_width + out] = static_cast<float>(sum);
    }
  }
}

float GeluTanh(float x) {
  constexpr float kAlpha = 0.7978845608028654F;  // sqrt(2 / pi)
  return 0.5F * x * (1.0F + std::tanh(kAlpha * (x + 0.044715F * x * x * x)));
}

}  // namespace

struct AudioVaeEncoder::Impl {
  const ModelInventory* inventory{nullptr};
  int device{0};
  hipStream_t stream{nullptr};
  void* staging{nullptr};
  EncoderConv input;
  std::array<EncoderBlock, kEncoderStrides.size()> blocks;
  DeviceBuffer final_alpha;
  EncoderConv final_conv;
  DeviceBuffer qkv_weight;
  EncoderHead head;
  std::array<float, kH3AudioVaeLatentChannels> mean{};
  std::array<float, kH3AudioVaeLatentChannels> deviation{};
  F32ConvGemm gemm;
  DeviceBuffer columns;
  AudioVaeEncoderTelemetry base_telemetry;

  ~Impl() {
    if (staging != nullptr) {
      (void)hipHostFree(staging);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  bool LoadDevice(std::string_view name,
                  std::initializer_list<std::uint64_t> shape,
                  DeviceBuffer* output, const CancellationToken* cancellation,
                  std::string* error) {
    const std::vector<std::uint64_t> expected(shape);
    const TensorDescriptor* descriptor =
        RequireTensor(*inventory, name, expected, error);
    if (descriptor == nullptr ||
        !LoadTensor(*inventory, *descriptor, device, stream, staging,
                    kWeightStagingBytes, cancellation, output, error)) {
      return false;
    }
    base_telemetry.weight_bytes += descriptor->payload_bytes;
    return true;
  }

  bool LoadHost(std::string_view name,
                std::initializer_list<std::uint64_t> shape,
                std::vector<float>* output, std::string* error) {
    const std::vector<std::uint64_t> expected(shape);
    const TensorDescriptor* descriptor =
        RequireTensor(*inventory, name, expected, error);
    if (descriptor == nullptr) {
      return false;
    }
    output->assign(descriptor->payload_bytes / sizeof(float), 0.0F);
    const std::filesystem::path path =
        inventory->model_root() / descriptor->shard;
    std::string open_error;
    const platform::PlatformFile file =
        platform::OpenReadFile(path, &open_error);
    if (!platform::FileValid(file)) {
      SetError(error, "cannot open MiniMax H3 AudioVAE shard " + path.string() +
                          ": " + open_error);
      return false;
    }
    std::size_t done = 0;
    const std::size_t bytes = output->size() * sizeof(float);
    while (done < bytes) {
      const std::size_t read = platform::ReadFileAt(
          file, reinterpret_cast<std::byte*>(output->data()) + done,
          bytes - done, descriptor->file_offset + done);
      if (read == 0) {
        break;
      }
      done += read;
    }
    platform::CloseFile(file);
    if (done != bytes) {
      SetError(error, "short read loading MiniMax H3 AudioVAE tensor " +
                          std::string(name));
      return false;
    }
    return true;
  }

  bool LoadConv(const std::string& prefix, std::uint32_t input_channels,
                std::uint32_t output_channels, std::uint32_t kernel,
                std::uint32_t stride, std::uint32_t padding,
                std::uint32_t dilation, EncoderConv* conv,
                const CancellationToken* cancellation, std::string* error) {
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->stride = stride;
    conv->padding = padding;
    conv->dilation = dilation;
    DeviceBuffer vector;
    DeviceBuffer magnitude;
    if (!LoadDevice(prefix + ".weight_v",
                    {output_channels, input_channels, kernel}, &vector,
                    cancellation, error) ||
        !LoadDevice(prefix + ".weight_g", {output_channels, 1, 1}, &magnitude,
                    cancellation, error) ||
        !LoadDevice(prefix + ".bias", {output_channels}, &conv->bias,
                    cancellation, error) ||
        !conv->weight.Allocate(vector.bytes(), error)) {
      return false;
    }
    audio_vae_ops::LaunchWeightNorm(vector.As<float>(), magnitude.As<float>(),
                                    conv->weight.As<float>(), output_channels,
                                    input_channels * kernel, stream);
    if (!CheckLaunch("normalize MiniMax H3 AudioVAE encoder convolution",
                     error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize MiniMax H3 AudioVAE encoder "
                               "weight norm",
                               status));
      return false;
    }
    base_telemetry.weight_bytes -= vector.bytes() + magnitude.bytes();
    base_telemetry.weight_bytes += conv->weight.bytes();
    return true;
  }

  bool LoadAlpha(const std::string& name, std::uint32_t channels,
                 DeviceBuffer* output, const CancellationToken* cancellation,
                 std::string* error) {
    return LoadDevice(name, {1, channels, 1}, output, cancellation, error);
  }

  bool LoadWeights(const CancellationToken* cancellation, std::string* error) {
    const auto begin = Clock::now();
    if (!LoadConv("encoder.block.0", 1, kEncoderBaseChannels, 7, 1, 3, 1,
                  &input, cancellation, error)) {
      return false;
    }
    std::uint32_t channels = kEncoderBaseChannels;
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      const std::string prefix =
          "encoder.block." + std::to_string(index + 1U) + ".block.";
      EncoderBlock& block = blocks[index];
      for (std::size_t unit = 0; unit < block.residuals.size(); ++unit) {
        const std::string residual = prefix + std::to_string(unit) + ".block.";
        const std::uint32_t dilation = kEncoderDilations[unit];
        EncoderResidual& weights = block.residuals[unit];
        if (!LoadAlpha(residual + "0.alpha", channels, &weights.alpha1,
                       cancellation, error) ||
            !LoadConv(residual + "1", channels, channels, 7, 1, 3 * dilation,
                      dilation, &weights.conv1, cancellation, error) ||
            !LoadAlpha(residual + "2.alpha", channels, &weights.alpha2,
                       cancellation, error) ||
            !LoadConv(residual + "3", channels, channels, 1, 1, 0, 1,
                      &weights.conv2, cancellation, error)) {
          return false;
        }
      }
      const std::uint32_t stride = kEncoderStrides[index];
      if (!LoadAlpha(prefix + "3.alpha", channels, &block.alpha, cancellation,
                     error) ||
          !LoadConv(prefix + "4", channels, channels * 2U, 2U * stride, stride,
                    (stride + 1U) / 2U, 1, &block.down, cancellation, error)) {
        return false;
      }
      channels *= 2U;
    }
    if (!LoadAlpha("encoder.block.6.alpha", channels, &final_alpha,
                   cancellation, error) ||
        !LoadConv("encoder.block.7", channels, channels, 3, 1, 1, 1,
                  &final_conv, cancellation, error) ||
        !LoadDevice("pre_block.attn.qkv.weight", {3U * channels, channels},
                    &qkv_weight, cancellation, error)) {
      return false;
    }
    const std::uint64_t width = channels;
    const std::uint64_t latent = kH3AudioVaeLatentChannels;
    if (!LoadHost("pre_block.norm1.weight", {width}, &head.norm1_weight,
                  error) ||
        !LoadHost("pre_block.norm1.bias", {width}, &head.norm1_bias, error) ||
        !LoadHost("pre_block.norm3.weight", {width}, &head.norm3_weight,
                  error) ||
        !LoadHost("pre_block.norm3.bias", {width}, &head.norm3_bias, error) ||
        !LoadHost("pre_block.attn.q_bias", {width}, &head.q_bias, error) ||
        !LoadHost("pre_block.attn.v_bias", {width}, &head.v_bias, error) ||
        !LoadHost("pre_block.attn.proj.weight", {latent, latent},
                  &head.attention_proj_weight, error) ||
        !LoadHost("pre_block.attn.proj.bias", {latent},
                  &head.attention_proj_bias, error) ||
        !LoadHost("pre_block.proj.weight", {latent, width}, &head.proj_weight,
                  error) ||
        !LoadHost("pre_block.proj.bias", {latent}, &head.proj_bias, error) ||
        !LoadHost("pre_block.norm2.weight", {latent}, &head.norm2_weight,
                  error) ||
        !LoadHost("pre_block.norm2.bias", {latent}, &head.norm2_bias, error) ||
        !LoadHost("pre_block.mlp.norm.weight", {latent}, &head.mlp_norm_weight,
                  error) ||
        !LoadHost("pre_block.mlp.norm.bias", {latent}, &head.mlp_norm_bias,
                  error) ||
        !LoadHost("pre_block.mlp.w0.weight", {kEncoderMlpHidden, latent},
                  &head.w0_weight, error) ||
        !LoadHost("pre_block.mlp.w0.bias", {kEncoderMlpHidden}, &head.w0_bias,
                  error) ||
        !LoadHost("pre_block.mlp.w1.weight", {kEncoderMlpHidden, latent},
                  &head.w1_weight, error) ||
        !LoadHost("pre_block.mlp.w1.bias", {kEncoderMlpHidden}, &head.w1_bias,
                  error) ||
        !LoadHost("pre_block.mlp.w2.weight", {latent, kEncoderMlpHidden},
                  &head.w2_weight, error) ||
        !LoadHost("pre_block.mlp.w2.bias", {latent}, &head.w2_bias, error) ||
        !LoadHost("mean_proj.weight", {latent, latent, 1}, &head.mean_weight,
                  error) ||
        !LoadHost("mean_proj.bias", {latent}, &head.mean_bias, error)) {
      return false;
    }
    if (!columns.Allocate(kEncoderColumnElements * sizeof(float), error)) {
      return false;
    }
    base_telemetry.scratch_bytes += columns.bytes();
    base_telemetry.load_ms = Milliseconds(begin, Clock::now());
    return true;
  }

  bool ReadNormalization(std::string* error) {
    try {
      const json::Value config = json::ParseFile(
          inventory->model_root() / "FL2VA/audio_vae/config.json", 1U << 20U);
      const json::Value* mean_value = config.Find("latents_mean");
      const json::Value* std_value = config.Find("latents_std");
      if (mean_value == nullptr || std_value == nullptr ||
          !mean_value->IsArray() || !std_value->IsArray() ||
          mean_value->AsArray().size() != mean.size() ||
          std_value->AsArray().size() != deviation.size()) {
        throw json::Error("latent normalization shape differs");
      }
      for (std::size_t index = 0; index < mean.size(); ++index) {
        mean[index] =
            static_cast<float>(mean_value->AsArray()[index].AsDouble());
        deviation[index] =
            static_cast<float>(std_value->AsArray()[index].AsDouble());
        if (!std::isfinite(mean[index]) || !std::isfinite(deviation[index]) ||
            deviation[index] <= 0.0F) {
          throw json::Error("latent normalization contains invalid values");
        }
      }
      return true;
    } catch (const std::exception& exception) {
      SetError(error, "MiniMax H3 AudioVAE config failed: " +
                          std::string(exception.what()));
      return false;
    }
  }

  // output[output_length][out] = conv(input[input_length][in]) + bias.
  bool RunConv(const EncoderConv& conv, const float* input,
               std::uint32_t input_length, float* output,
               std::uint32_t output_length, std::string* error) {
    const std::size_t reduction =
        static_cast<std::size_t>(conv.input_channels) * conv.kernel;
    const std::uint32_t chunk = static_cast<std::uint32_t>(
        std::max<std::size_t>(1, kEncoderColumnElements / reduction));
    for (std::uint32_t begin = 0; begin < output_length; begin += chunk) {
      const std::uint32_t count = std::min(chunk, output_length - begin);
      audio_vae_ops::LaunchStridedIm2Col(
          input, columns.As<float>(), input_length, conv.input_channels, begin,
          count, conv.kernel, conv.stride, conv.padding, conv.dilation, stream);
      ++base_telemetry.dispatches;
      if (!CheckLaunch("H3 AudioVAE encoder im2col", error) ||
          !gemm.Run(
              conv.weight.As<float>(), columns.As<float>(),
              output + static_cast<std::size_t>(begin) * conv.output_channels,
              count, conv.output_channels, reduction, stream, error)) {
        return false;
      }
      ++base_telemetry.dispatches;
    }
    audio_vae_ops::LaunchAddConvBias(output, conv.bias.As<float>(),
                                     output_length, conv.output_channels,
                                     stream);
    ++base_telemetry.dispatches;
    return CheckLaunch("H3 AudioVAE encoder convolution bias", error);
  }

  bool Snake(const float* input, const DeviceBuffer& alpha, float* output,
             std::uint32_t length, std::uint32_t channels, std::string* error) {
    audio_vae_ops::LaunchSnake(input, alpha.As<float>(), output,
                               static_cast<std::size_t>(length) * channels,
                               channels, stream);
    ++base_telemetry.dispatches;
    return CheckLaunch("H3 AudioVAE encoder snake", error);
  }

  // One mono track [samples] (a multiple of the hop) -> [frames][2048].
  bool EncodeTrack(std::span<const float> pcm,
                   const CancellationToken* cancellation,
                   std::vector<float>* features, std::string* error) {
    const std::uint32_t samples = static_cast<std::uint32_t>(pcm.size());
    const std::size_t largest =
        static_cast<std::size_t>(samples) * kEncoderBaseChannels;
    std::array<DeviceBuffer, 3> buffers;
    for (DeviceBuffer& buffer : buffers) {
      if (!buffer.Allocate(largest * sizeof(float), error)) {
        return false;
      }
    }
    hipError_t status =
        hipMemcpyAsync(buffers[2].data(), pcm.data(), pcm.size_bytes(),
                       hipMemcpyHostToDevice, stream);
    if (status != hipSuccess) {
      SetError(error, HipError("copy H3 audio guide waveform", status));
      return false;
    }
    std::uint32_t length = samples;
    if (!RunConv(input, buffers[2].As<float>(), length, buffers[0].As<float>(),
                 input.OutputLength(length), error)) {
      return false;
    }
    length = input.OutputLength(length);
    std::size_t current = 0;
    std::uint32_t channels = kEncoderBaseChannels;
    for (const EncoderBlock& block : blocks) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 audio guide encoding cancelled");
        return false;
      }
      float* x = buffers[current].As<float>();
      float* t1 = buffers[(current + 1U) % 3U].As<float>();
      float* t2 = buffers[(current + 2U) % 3U].As<float>();
      const std::size_t elements = static_cast<std::size_t>(length) * channels;
      for (const EncoderResidual& unit : block.residuals) {
        if (!Snake(x, unit.alpha1, t1, length, channels, error) ||
            !RunConv(unit.conv1, t1, length, t2, length, error) ||
            !Snake(t2, unit.alpha2, t1, length, channels, error) ||
            !RunConv(unit.conv2, t1, length, t2, length, error)) {
          return false;
        }
        audio_vae_ops::LaunchAddScaled(x, t2, x, elements, 1.0F, 1.0F, stream);
        ++base_telemetry.dispatches;
        if (!CheckLaunch("H3 AudioVAE encoder residual", error)) {
          return false;
        }
      }
      const std::uint32_t next = block.down.OutputLength(length);
      if (!Snake(x, block.alpha, t1, length, channels, error) ||
          !RunConv(block.down, t1, length, t2, next, error)) {
        return false;
      }
      length = next;
      channels *= 2U;
      current = (current + 2U) % 3U;
    }
    float* x = buffers[current].As<float>();
    float* t1 = buffers[(current + 1U) % 3U].As<float>();
    float* t2 = buffers[(current + 2U) % 3U].As<float>();
    if (!Snake(x, final_alpha, t1, length, channels, error) ||
        !RunConv(final_conv, t1, length, t2, length, error)) {
      return false;
    }
    features->assign(static_cast<std::size_t>(length) * channels, 0.0F);
    status =
        hipMemcpyAsync(features->data(), t2, features->size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream);
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("read H3 audio guide features", status));
      return false;
    }
    return true;
  }

  // AttnProjection + mean_proj + normalization: [frames][2048] ->
  // normalized [frames][32].
  bool Head(const std::vector<float>& features, std::size_t frames,
            std::vector<float>* latent, std::string* error) {
    const std::size_t width = head.norm1_weight.size();
    const std::size_t heads = kEncoderAttentionHeads;
    const std::size_t head_width = width / heads;
    const std::size_t channels = kH3AudioVaeLatentChannels;
    std::vector<float> normalized(frames * width);
    LayerNormRows(features.data(), normalized.data(), frames, width,
                  head.norm1_weight, head.norm1_bias);
    // qkv on the GPU: [frames][2048] x [6144][2048]^T.
    DeviceBuffer input_device;
    DeviceBuffer qkv_device;
    std::vector<float> qkv(frames * width * 3U);
    if (!input_device.Allocate(normalized.size() * sizeof(float), error) ||
        !qkv_device.Allocate(qkv.size() * sizeof(float), error)) {
      return false;
    }
    hipError_t status = hipMemcpyAsync(input_device.data(), normalized.data(),
                                       normalized.size() * sizeof(float),
                                       hipMemcpyHostToDevice, stream);
    if (status != hipSuccess) {
      SetError(error, HipError("copy H3 audio guide attention input", status));
      return false;
    }
    if (!gemm.Run(qkv_weight.As<float>(), input_device.As<float>(),
                  qkv_device.As<float>(), frames, width * 3U, width, stream,
                  error)) {
      return false;
    }
    status = hipMemcpyAsync(qkv.data(), qkv_device.data(),
                            qkv.size() * sizeof(float), hipMemcpyDeviceToHost,
                            stream);
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("read H3 audio guide attention", status));
      return false;
    }
    for (std::size_t row = 0; row < frames; ++row) {
      float* q = qkv.data() + row * width * 3U;
      float* v = q + width * 2U;
      for (std::size_t index = 0; index < width; ++index) {
        q[index] += head.q_bias[index];
        v[index] += head.v_bias[index];
      }
    }
    // Causal attention per head, averaged over the heads.
    std::vector<float> head_mean(frames * head_width, 0.0F);
    {
      std::vector<std::vector<float>> outputs(
          heads, std::vector<float>(frames * head_width, 0.0F));
      std::vector<std::thread> workers;
      const float scale = 1.0F / std::sqrt(static_cast<float>(head_width));
      for (std::size_t h = 0; h < heads; ++h) {
        workers.emplace_back([&, h] {
          std::vector<double> scores(frames);
          for (std::size_t t = 0; t < frames; ++t) {
            const float* q = qkv.data() + t * width * 3U + h * head_width;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t s = 0; s <= t; ++s) {
              const float* k =
                  qkv.data() + s * width * 3U + width + h * head_width;
              double dot = 0.0;
              for (std::size_t i = 0; i < head_width; ++i) {
                dot += static_cast<double>(q[i]) * k[i];
              }
              scores[s] = dot * scale;
              maximum = std::max(maximum, scores[s]);
            }
            double total = 0.0;
            for (std::size_t s = 0; s <= t; ++s) {
              scores[s] = std::exp(scores[s] - maximum);
              total += scores[s];
            }
            float* out = outputs[h].data() + t * head_width;
            for (std::size_t s = 0; s <= t; ++s) {
              const float weight = static_cast<float>(scores[s] / total);
              const float* v =
                  qkv.data() + s * width * 3U + width * 2U + h * head_width;
              for (std::size_t i = 0; i < head_width; ++i) {
                out[i] += weight * v[i];
              }
            }
          }
        });
      }
      for (std::thread& worker : workers) {
        worker.join();
      }
      for (std::size_t h = 0; h < heads; ++h) {
        for (std::size_t index = 0; index < head_mean.size(); ++index) {
          head_mean[index] += outputs[h][index] / static_cast<float>(heads);
        }
      }
    }
    // adaptive_avg_pool1d(head_width -> 32): equal groups of 8.
    const std::size_t group = head_width / channels;
    std::vector<float> pooled(frames * channels, 0.0F);
    for (std::size_t t = 0; t < frames; ++t) {
      for (std::size_t c = 0; c < channels; ++c) {
        float sum = 0.0F;
        for (std::size_t i = 0; i < group; ++i) {
          sum += head_mean[t * head_width + c * group + i];
        }
        pooled[t * channels + c] = sum / static_cast<float>(group);
      }
    }
    std::vector<float> attention(frames * channels);
    LinearRows(pooled.data(), attention.data(), frames, channels, channels,
               head.attention_proj_weight, head.attention_proj_bias);
    // x = proj(norm3(features)) + attention.
    LayerNormRows(features.data(), normalized.data(), frames, width,
                  head.norm3_weight, head.norm3_bias);
    std::vector<float> x(frames * channels);
    LinearRows(normalized.data(), x.data(), frames, width, channels,
               head.proj_weight, head.proj_bias);
    for (std::size_t index = 0; index < x.size(); ++index) {
      x[index] += attention[index];
    }
    // x += mlp(norm2(x)), the GeGLU MLP normalizes its input once more.
    std::vector<float> h1(frames * channels);
    std::vector<float> h2(frames * channels);
    LayerNormRows(x.data(), h1.data(), frames, channels, head.norm2_weight,
                  head.norm2_bias);
    LayerNormRows(h1.data(), h2.data(), frames, channels, head.mlp_norm_weight,
                  head.mlp_norm_bias);
    std::vector<float> gate(frames * kEncoderMlpHidden);
    std::vector<float> up(frames * kEncoderMlpHidden);
    LinearRows(h2.data(), gate.data(), frames, channels, kEncoderMlpHidden,
               head.w0_weight, head.w0_bias);
    LinearRows(h2.data(), up.data(), frames, channels, kEncoderMlpHidden,
               head.w1_weight, head.w1_bias);
    for (std::size_t index = 0; index < gate.size(); ++index) {
      gate[index] = GeluTanh(gate[index]) * up[index];
    }
    LinearRows(gate.data(), h1.data(), frames, kEncoderMlpHidden, channels,
               head.w2_weight, head.w2_bias);
    for (std::size_t index = 0; index < x.size(); ++index) {
      x[index] += h1[index];
    }
    // mean_proj (1x1 convolution), then the latent normalization.
    LinearRows(x.data(), h1.data(), frames, channels, channels,
               head.mean_weight, head.mean_bias);
    latent->assign(frames * channels, 0.0F);
    for (std::size_t t = 0; t < frames; ++t) {
      for (std::size_t c = 0; c < channels; ++c) {
        const float value = (h1[t * channels + c] - mean[c]) / deviation[c];
        if (!std::isfinite(value)) {
          SetError(error, "MiniMax H3 audio guide latent is not finite");
          return false;
        }
        (*latent)[t * channels + c] = value;
      }
    }
    return true;
  }
};

AudioVaeEncoder::AudioVaeEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AudioVaeEncoder::~AudioVaeEncoder() = default;
AudioVaeEncoder::AudioVaeEncoder(AudioVaeEncoder&&) noexcept = default;
AudioVaeEncoder& AudioVaeEncoder::operator=(AudioVaeEncoder&&) noexcept =
    default;

std::unique_ptr<AudioVaeEncoder> AudioVaeEncoder::Create(
    const ModelInventory& inventory, const CancellationToken* cancellation,
    AudioVaeEncoderTelemetry* telemetry, std::string* error) {
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 AudioVAE encoder creation cancelled");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->inventory = &inventory;
  if (!ValidateTarget(&impl->device, error)) {
    return nullptr;
  }
  hipError_t status = hipSetDevice(impl->device);
  if (status == hipSuccess) {
    status = hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
  }
  if (status == hipSuccess) {
    status = hipHostMalloc(&impl->staging, kWeightStagingBytes,
                           hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    SetError(error, HipError("initialize MiniMax H3 AudioVAE encoder", status));
    return nullptr;
  }
  if (!impl->ReadNormalization(error) ||
      !impl->LoadWeights(cancellation, error)) {
    return nullptr;
  }
  status = hipHostFree(impl->staging);
  impl->staging = nullptr;
  if (status != hipSuccess) {
    SetError(error,
             HipError("release MiniMax H3 AudioVAE encoder staging", status));
    return nullptr;
  }
  if (telemetry != nullptr) {
    *telemetry = impl->base_telemetry;
  }
  return std::unique_ptr<AudioVaeEncoder>(new AudioVaeEncoder(std::move(impl)));
}

bool AudioVaeEncoder::Encode(const AudioWaveform& waveform,
                             const CancellationToken* cancellation,
                             std::vector<float>* latent, int* latent_frames,
                             AudioVaeEncoderTelemetry* telemetry,
                             std::string* error) {
  if (impl_ == nullptr || latent == nullptr || latent_frames == nullptr) {
    SetError(error, "MiniMax H3 AudioVAE encoder is empty");
    return false;
  }
  if (waveform.channels != kH3AudioVaeStereoChannels ||
      waveform.sample_rate != kH3AudioVaeSampleRate || waveform.samples < 1 ||
      waveform.pcm.size() != static_cast<std::size_t>(waveform.samples) *
                                 kH3AudioVaeStereoChannels) {
    SetError(error,
             "MiniMax H3 audio guides must be 32 kHz stereo before encoding");
    return false;
  }
  const auto begin = Clock::now();
  const std::size_t frames =
      (static_cast<std::size_t>(waveform.samples) + kH3AudioVaeHopLength - 1U) /
      kH3AudioVaeHopLength;
  const std::size_t padded = frames * kH3AudioVaeHopLength;
  std::vector<float> output(kH3AudioVaeLatentChannels *
                            kH3AudioVaeStereoChannels * frames);
  std::vector<float> track(padded, 0.0F);
  std::vector<float> features;
  std::vector<float> rows;
  for (int stereo = 0; stereo < kH3AudioVaeStereoChannels; ++stereo) {
    std::copy_n(waveform.pcm.begin() +
                    static_cast<std::ptrdiff_t>(stereo) * waveform.samples,
                waveform.samples, track.begin());
    std::fill(track.begin() + waveform.samples, track.end(), 0.0F);
    if (!impl_->EncodeTrack(track, cancellation, &features, error)) {
      return false;
    }
    if (features.size() != frames * impl_->head.norm1_weight.size()) {
      SetError(error, "MiniMax H3 audio guide encoder length differs");
      return false;
    }
    if (!impl_->Head(features, frames, &rows, error)) {
      return false;
    }
    for (std::size_t t = 0; t < frames; ++t) {
      for (std::size_t c = 0; c < kH3AudioVaeLatentChannels; ++c) {
        output[(c * kH3AudioVaeStereoChannels + stereo) * frames + t] =
            rows[t * kH3AudioVaeLatentChannels + c];
      }
    }
  }
  *latent = std::move(output);
  *latent_frames = static_cast<int>(frames);
  impl_->base_telemetry.encode_ms += Milliseconds(begin, Clock::now());
  if (telemetry != nullptr) {
    *telemetry = impl_->base_telemetry;
  }
  return true;
}

}  // namespace gufo::minimax_h3

#endif
