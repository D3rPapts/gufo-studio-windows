#include "src/models/qwen3_omni/hip/audio_encoder.hpp"

#include <hip/hip_bfloat16.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen3_asr/hip/audio_ops.hpp"
#include "src/models/qwen3_asr/hip/blas.hpp"
#include "src/models/qwen3_omni/hip/device_buffer.hpp"

namespace gufo::models::qwen3_omni::hip {
namespace {

namespace asr = qwen3_asr::hip;
using Bf16 = hip_bfloat16;

constexpr std::size_t kChunkFrames = 100;
constexpr std::size_t kConvChannels = 480;
constexpr std::size_t kConvTime = 13;
constexpr std::size_t kFlattened = kConvChannels * 16;

/// float -> BF16 bits, round to nearest even. On the host, TheRock's
/// hip_bfloat16(float) resolves to the integer constructor, so the bits are
/// formed here.
Bf16 FloatToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  std::uint16_t rounded = 0;
  if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0U) {
    rounded = static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);  // NaN
  } else {
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    rounded = static_cast<std::uint16_t>(bits >> 16U);
  }
  Bf16 result;
  static_assert(sizeof(result) == sizeof(rounded));
  std::memcpy(&result, &rounded, sizeof(rounded));
  return result;
}

/// Host BF16 copy of an F16 or F32 weight from its `encoded` bytes (round to
/// nearest even; exact for values that were BF16 before the GGUF conversion).
std::vector<Bf16> ToBf16(const Weight& weight,
                         std::span<const std::byte> encoded) {
  const std::size_t count =
      static_cast<std::size_t>(weight.cols * weight.rows * weight.experts);
  std::vector<Bf16> result(count);
  if (weight.type == core::GgmlType::kF32) {
    const auto* values = reinterpret_cast<const float*>(encoded.data());
    for (std::size_t index = 0; index < count; ++index) {
      result[index] = FloatToBf16(values[index]);
    }
  } else if (weight.type == core::GgmlType::kF16) {
    const auto* values =
        reinterpret_cast<const std::uint16_t*>(encoded.data());
    for (std::size_t index = 0; index < count; ++index) {
      result[index] = FloatToBf16(quant::Fp16ToFloat(values[index]));
    }
  } else {
    throw std::runtime_error("Qwen3-Omni audio weight is neither F16 nor F32");
  }
  return result;
}

struct Layer {
  DeviceBuffer<Bf16> ln1_weight, ln1_bias;
  DeviceBuffer<Bf16> q_weight, q_bias, k_weight, k_bias, v_weight, v_bias;
  DeviceBuffer<Bf16> out_weight, out_bias;
  DeviceBuffer<Bf16> ln2_weight, ln2_bias;
  DeviceBuffer<Bf16> fc1_weight, fc1_bias, fc2_weight, fc2_bias;
};

/// GUFO_OMNI_DEBUG=1: report the first stage whose output is not finite.
bool DebugEnabled() {
  static const bool enabled = std::getenv("GUFO_OMNI_DEBUG") != nullptr;
  return enabled;
}

float ToFloat(float value) {
  return value;
}

float ToFloat(Bf16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &widened, sizeof(result));
  return result;
}

template<typename Element>
void DebugCheck(const char* stage, const Element* device, std::size_t count,
                hipStream_t stream) {
  if (!DebugEnabled()) {
    return;
  }
  std::vector<Element> host(count);
  RequireHip(hipMemcpyAsync(host.data(), device, count * sizeof(Element),
                            hipMemcpyDeviceToHost, stream),
             "debug copy");
  RequireHip(hipStreamSynchronize(stream), "debug sync");
  const hipError_t launch = hipGetLastError();
  if (launch != hipSuccess) {
    std::cerr << "[omni-debug] " << stage
              << ": launch error: " << hipGetErrorString(launch) << std::endl;
  }
  std::size_t bad = 0;
  double sum = 0.0;
  float largest = 0.0F;
  for (const Element& value : host) {
    const float x = ToFloat(value);
    if (!std::isfinite(x)) {
      if (bad < 4) {
        std::cerr << "[omni-debug]   bad at " << (&value - host.data())
                  << std::endl;
      }
      ++bad;
    } else {
      sum += std::fabs(x);
      largest = std::max(largest, std::fabs(x));
    }
  }
  std::cerr << "[omni-debug] " << stage << ": " << count << " values, "
            << bad << " non-finite, mean|x| "
            << sum / static_cast<double>(std::max<std::size_t>(count, 1))
            << ", max|x| " << largest << std::endl;
}

void RequireHipblas(hipblasStatus_t status, const char* operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(status));
  }
}

}  // namespace

struct AudioEncoder::Impl {
  Impl(const Model& model, hipStream_t stream_in)
      : config(model.audio_config()),
        projector(&model.projector_reader()),
        stream(stream_in) {
    RequireHipblas(hipblasCreate(&blas), "hipblasCreate Qwen3-Omni audio");
    RequireHipblas(hipblasSetStream(blas, stream), "hipblasSetStream");
    RequireHipblas(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                   "hipblasSetAtomicsMode");
    lt = std::make_unique<asr::GemmLt>();
    Load(model.audio());
    // The read buffer only serves the load.
    std::vector<std::byte>().swap(scratch);
  }

  ~Impl() {
    if (blas != nullptr) {
      (void)hipblasDestroy(blas);
    }
  }

  DeviceBuffer<Bf16> Upload(const Weight& weight) {
    // From the file, not the mapping: mapped pages read here would stay in
    // the working set for the life of the process on Windows.
    const std::vector<Bf16> host =
        ToBf16(weight, ReadWeightBytes(*projector, weight, &scratch));
    DeviceBuffer<Bf16> device(host.size());
    RequireHip(hipMemcpy(device.get(), host.data(), host.size() * sizeof(Bf16),
                         hipMemcpyHostToDevice),
               "hipMemcpy Qwen3-Omni audio weight");
    bytes += host.size() * sizeof(Bf16);
    return device;
  }

  DeviceBuffer<Bf16> UploadConvolution(const Weight& weight) {
    DeviceBuffer<Bf16> original = Upload(weight);
    DeviceBuffer<Bf16> packed(original.size());
    asr::LaunchPackConvolutionWeights(original.get(), packed.get(), stream);
    RequireHip(hipStreamSynchronize(stream), "pack Qwen3-Omni convolution");
    return packed;
  }

  void Load(const AudioWeights& weights) {
    conv1_weight = Upload(weights.conv1_weight);
    conv1_bias = Upload(weights.conv1_bias);
    DebugCheck("conv1 weight", conv1_weight.get(), conv1_weight.size(), stream);
    DebugCheck("conv1 bias", conv1_bias.get(), conv1_bias.size(), stream);
    conv2_weight = UploadConvolution(weights.conv2_weight);
    conv2_bias = Upload(weights.conv2_bias);
    conv3_weight = UploadConvolution(weights.conv3_weight);
    conv3_bias = Upload(weights.conv3_bias);
    conv_out = Upload(weights.conv_out);
    layers.reserve(weights.layers.size());
    for (const AudioLayer& source : weights.layers) {
      Layer layer;
      layer.ln1_weight = Upload(source.ln1_weight);
      layer.ln1_bias = Upload(source.ln1_bias);
      layer.q_weight = Upload(source.q_weight);
      layer.q_bias = Upload(source.q_bias);
      layer.k_weight = Upload(source.k_weight);
      layer.k_bias = Upload(source.k_bias);
      layer.v_weight = Upload(source.v_weight);
      layer.v_bias = Upload(source.v_bias);
      layer.out_weight = Upload(source.out_weight);
      layer.out_bias = Upload(source.out_bias);
      layer.ln2_weight = Upload(source.ln2_weight);
      layer.ln2_bias = Upload(source.ln2_bias);
      layer.fc1_weight = Upload(source.fc1_weight);
      layer.fc1_bias = Upload(source.fc1_bias);
      layer.fc2_weight = Upload(source.fc2_weight);
      layer.fc2_bias = Upload(source.fc2_bias);
      layers.push_back(std::move(layer));
    }
    post_ln_weight = Upload(weights.post_ln_weight);
    post_ln_bias = Upload(weights.post_ln_bias);
    proj1_weight = Upload(weights.proj1_weight);
    proj1_bias = Upload(weights.proj1_bias);
    proj2_weight = Upload(weights.proj2_weight);
    proj2_bias = Upload(weights.proj2_bias);
  }

  void Gemm(const void* weight, const void* input, float* output,
            std::size_t rows, std::size_t columns, std::size_t reduction) {
    if (lt->Run(weight, input, output, rows, columns, reduction, stream)) {
      return;
    }
    asr::LaunchGemmBf16(blas, weight, input, output, rows, columns, reduction,
                        stream);
  }

  std::size_t Frontend(std::span<const float> log_mel, std::size_t frames) {
    const std::size_t chunks = (frames + kChunkFrames - 1) / kChunkFrames;
    const std::size_t hidden = config.hidden;
    feature_input.Ensure(log_mel.size());
    input.Ensure(chunks * config.mel_bins * kChunkFrames);
    conv1.Ensure(chunks * kConvChannels * 64U * 50U);
    conv2.Ensure(chunks * kConvChannels * 32U * 25U);
    conv3.Ensure(chunks * kConvChannels * 16U * kConvTime);
    layout.Ensure(chunks * kConvTime * kFlattened);
    projected.Ensure(chunks * kConvTime * hidden);
    compact.Ensure(chunks * kConvTime * hidden);
    RequireHip(hipMemcpyAsync(feature_input.get(), log_mel.data(),
                              log_mel.size() * sizeof(float),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync Qwen3-Omni log-mel");
    asr::LaunchChunkLogMel(feature_input.get(), input.get(), frames, chunks,
                           stream);
    DebugCheck("chunked mel", input.get(), chunks * config.mel_bins * kChunkFrames,
               stream);
    asr::LaunchConvolution3x3(input.get(), conv1_weight.get(), conv1.get(),
                              chunks, 1, 128, 100, stream);
    asr::LaunchBiasGelu(conv1.get(), conv1_bias.get(),
                        chunks * kConvChannels * 64U * 50U, kConvChannels,
                        64U * 50U, stream);
    DebugCheck("conv1", conv1.get(), chunks * kConvChannels * 64U * 50U, stream);
    asr::LaunchConvolution3x3(conv1.get(), conv2_weight.get(), conv2.get(),
                              chunks, kConvChannels, 64, 50, stream);
    asr::LaunchBiasGelu(conv2.get(), conv2_bias.get(),
                        chunks * kConvChannels * 32U * 25U, kConvChannels,
                        32U * 25U, stream);
    DebugCheck("conv2", conv2.get(), chunks * kConvChannels * 32U * 25U, stream);
    asr::LaunchConvolution3x3(conv2.get(), conv3_weight.get(), conv3.get(),
                              chunks, kConvChannels, 32, 25, stream);
    asr::LaunchBiasGelu(conv3.get(), conv3_bias.get(),
                        chunks * kConvChannels * 16U * kConvTime,
                        kConvChannels, 16U * kConvTime, stream);
    DebugCheck("conv3", conv3.get(), chunks * kConvChannels * 16U * kConvTime,
               stream);
    asr::LaunchConvOutputLayout(conv3.get(), layout.get(), chunks, stream);
    Gemm(conv_out.get(), layout.get(), projected.get(), chunks * kConvTime,
         hidden, kFlattened);
    DebugCheck("conv_out", projected.get(), chunks * kConvTime * hidden, stream);
    asr::LaunchAddPositionAndCompact(projected.get(), compact.get(), frames,
                                     chunks, stream, hidden);
    DebugCheck("positions", compact.get(),
               AudioTokenCount(frames, config.n_window) * hidden, stream);
    return AudioTokenCount(frames, config.n_window);
  }

  void Encode(std::span<const float> log_mel, std::size_t frames,
              const std::function<bool()>& is_cancelled) {
    const auto checkpoint = [&] {
      if (is_cancelled && is_cancelled()) {
        throw std::runtime_error("Qwen3-Omni audio encoder cancelled");
      }
    };
    if (frames == 0U || log_mel.size() != config.mel_bins * frames) {
      throw std::invalid_argument("Qwen3-Omni log-mel must be [128, frames]");
    }
    checkpoint();
    const std::size_t count = Frontend(log_mel, frames);
    const std::size_t hidden = config.hidden;
    const std::size_t ffn_width = config.ffn;
    const std::size_t output_width = config.output;
    const std::size_t heads = config.heads;
    const std::size_t head_dim = config.HeadDim();
    // Attention windows of n_window_infer frames: 8 chunks of 13 tokens
    // (fewer for a clip shorter than one chunk).
    const std::size_t max_chunk_tokens =
        frames >= kChunkFrames ? kConvTime
                               : AudioTokenCount(frames, config.n_window);
    const std::size_t window =
        max_chunk_tokens * (config.n_window_infer / config.ChunkFrames());

    hidden_state.Ensure(count * hidden);
    normalized.Ensure(count * std::max(hidden, ffn_width));
    q.Ensure(count * hidden);
    k.Ensure(count * hidden);
    v.Ensure(count * hidden);
    attention.Ensure(count * hidden);
    attention_bf16.Ensure(count * hidden);
    projection.Ensure(count * std::max(hidden, output_width));
    ffn.Ensure(count * ffn_width);
    ffn_bf16.Ensure(count * ffn_width);
    RequireHip(hipMemcpyAsync(hidden_state.get(), compact.get(),
                              count * hidden * sizeof(float),
                              hipMemcpyDeviceToDevice, stream),
               "hipMemcpyAsync Qwen3-Omni frontend");
    const std::size_t elements = count * hidden;
    for (const Layer& layer : layers) {
      checkpoint();
      asr::LaunchLayerNormToBfloat16(hidden_state.get(), layer.ln1_weight.get(),
                                     layer.ln1_bias.get(), normalized.get(),
                                     count, hidden, config.eps, stream);
      Gemm(layer.q_weight.get(), normalized.get(), q.get(), count, hidden,
           hidden);
      Gemm(layer.k_weight.get(), normalized.get(), k.get(), count, hidden,
           hidden);
      Gemm(layer.v_weight.get(), normalized.get(), v.get(), count, hidden,
           hidden);
      asr::LaunchBiasRoundBfloat16(q.get(), layer.q_bias.get(), count, hidden,
                                   stream);
      asr::LaunchBiasRoundBfloat16(k.get(), layer.k_bias.get(), count, hidden,
                                   stream);
      asr::LaunchBiasRoundBfloat16(v.get(), layer.v_bias.get(), count, hidden,
                                   stream);
      asr::LaunchSegmentedSelfAttention(q.get(), k.get(), v.get(),
                                        attention.get(), attention_bf16.get(),
                                        count, window, heads, head_dim, stream);
      Gemm(layer.out_weight.get(), attention_bf16.get(), projection.get(),
           count, hidden, hidden);
      asr::LaunchBiasRoundBfloat16(projection.get(), layer.out_bias.get(),
                                   count, hidden, stream);
      asr::LaunchResidualAddBfloat16(hidden_state.get(), projection.get(),
                                     elements, stream);
      if (&layer == &layers.front()) {
        DebugCheck("layer0 q", q.get(), elements, stream);
        DebugCheck("layer0 attention", attention.get(), elements, stream);
        DebugCheck("layer0 after attention", hidden_state.get(), elements,
                   stream);
      }

      asr::LaunchLayerNormToBfloat16(hidden_state.get(), layer.ln2_weight.get(),
                                     layer.ln2_bias.get(), normalized.get(),
                                     count, hidden, config.eps, stream);
      Gemm(layer.fc1_weight.get(), normalized.get(), ffn.get(), count,
           ffn_width, hidden);
      asr::LaunchBiasGeluToBfloat16(ffn.get(), layer.fc1_bias.get(),
                                    ffn_bf16.get(), count, ffn_width, stream);
      Gemm(layer.fc2_weight.get(), ffn_bf16.get(), projection.get(), count,
           hidden, ffn_width);
      asr::LaunchBiasRoundBfloat16(projection.get(), layer.fc2_bias.get(),
                                   count, hidden, stream);
      asr::LaunchResidualAddBfloat16(hidden_state.get(), projection.get(),
                                     elements, stream);
      DebugCheck("layer", hidden_state.get(), elements, stream);
    }

    asr::LaunchLayerNormToBfloat16(hidden_state.get(), post_ln_weight.get(),
                                   post_ln_bias.get(), normalized.get(), count,
                                   hidden, config.eps, stream);
    Gemm(proj1_weight.get(), normalized.get(), projection.get(), count, hidden,
         hidden);
    asr::LaunchBiasGeluToBfloat16(projection.get(), proj1_bias.get(),
                                  normalized.get(), count, hidden, stream);
    Gemm(proj2_weight.get(), normalized.get(), projection.get(), count,
         output_width, hidden);
    asr::LaunchBiasRoundBfloat16(projection.get(), proj2_bias.get(), count,
                                 output_width, stream);

    std::uint32_t finite = 1U;
    RequireHip(hipMemcpyAsync(finite_flag.get(), &finite, sizeof(finite),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync Qwen3-Omni finite flag");
    asr::LaunchAllFinite(projection.get(), count * output_width,
                         finite_flag.get(), stream);
    RequireHip(hipMemcpyAsync(&finite, finite_flag.get(), sizeof(finite),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync Qwen3-Omni finite result");
    RequireHip(hipStreamSynchronize(stream), "Qwen3-Omni audio encoder");
    if (finite == 0U) {
      throw std::runtime_error(
          "Qwen3-Omni audio encoder produced non-finite embeddings");
    }
    tokens = count;
  }

  AudioConfig config;
  const core::GgufReader* projector{nullptr};
  std::vector<std::byte> scratch;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<asr::GemmLt> lt;
  std::size_t bytes{0};
  std::size_t tokens{0};

  DeviceBuffer<Bf16> conv1_weight, conv1_bias, conv2_weight, conv2_bias;
  DeviceBuffer<Bf16> conv3_weight, conv3_bias, conv_out;
  std::vector<Layer> layers;
  DeviceBuffer<Bf16> post_ln_weight, post_ln_bias;
  DeviceBuffer<Bf16> proj1_weight, proj1_bias, proj2_weight, proj2_bias;

  DeviceBuffer<float> feature_input;
  DeviceBuffer<Bf16> input, conv1, conv2, conv3, layout;
  DeviceBuffer<float> projected, compact;
  DeviceBuffer<float> hidden_state;
  DeviceBuffer<Bf16> normalized;
  DeviceBuffer<float> q, k, v, attention;
  DeviceBuffer<Bf16> attention_bf16;
  DeviceBuffer<float> projection, ffn;
  DeviceBuffer<Bf16> ffn_bf16;
  DeviceBuffer<std::uint32_t> finite_flag{1U};
};

AudioEncoder::AudioEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AudioEncoder::~AudioEncoder() = default;

std::unique_ptr<AudioEncoder> AudioEncoder::Create(const Model& model,
                                                   hipStream_t stream,
                                                   std::string* error) {
  try {
    return std::unique_ptr<AudioEncoder>(
        new AudioEncoder(std::make_unique<Impl>(model, stream)));
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
}

bool AudioEncoder::Encode(std::span<const float> log_mel, std::size_t frames,
                          const float** device_output, std::size_t* tokens,
                          std::string* error,
                          const std::function<bool()>& is_cancelled) {
  try {
    impl_->Encode(log_mel, frames, is_cancelled);
    *device_output = impl_->projection.get();
    *tokens = impl_->tokens;
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

std::size_t AudioEncoder::weight_bytes() const {
  return impl_->bytes;
}

}  // namespace gufo::models::qwen3_omni::hip
