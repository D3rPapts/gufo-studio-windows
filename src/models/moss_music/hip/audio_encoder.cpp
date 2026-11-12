#include "src/models/moss_music/hip/audio_encoder.hpp"

#include <hip/hip_bfloat16.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/moss_music/hip/stem_ops.hpp"
#include "src/models/qwen3_asr/hip/audio_ops.hpp"
#include "src/models/qwen3_asr/hip/blas.hpp"

namespace gufo::models::moss_music::hip {
namespace {

namespace asr = qwen3_asr::hip;
using Bf16 = hip_bfloat16;

void RequireHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void RequireHipblas(hipblasStatus_t status, const char* operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(status));
  }
}

/// Owning device allocation of `count` elements.
template<typename Element>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Reset(count); }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  void Reset(std::size_t count) {
    if (data_ != nullptr) {
      RequireHip(hipFree(data_), "hipFree MOSS-Music buffer");
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0U) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("MOSS-Music HIP buffer size overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc MOSS-Music buffer (" +
            std::to_string(count * sizeof(Element) >> 20) + " MiB)");
    count_ = count;
  }
  /// Grows (never shrinks) to at least `count` elements; contents are lost.
  void Ensure(std::size_t count) {
    if (count > count_) {
      Reset(count);
    }
  }
  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

std::uint16_t FloatToBf16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);  // NaN
  }
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

/// Host BF16 bits of a weight: BF16 copied, F32 / F16 rounded to nearest even.
std::vector<std::uint16_t> ToBf16(const Weight& weight) {
  const std::size_t count = static_cast<std::size_t>(weight.cols * weight.rows);
  std::vector<std::uint16_t> result(count);
  switch (weight.type) {
    case core::GgmlType::kBF16:
      std::memcpy(result.data(), weight.data, count * sizeof(std::uint16_t));
      break;
    case core::GgmlType::kF32: {
      const auto* values = static_cast<const float*>(weight.data);
      for (std::size_t index = 0; index < count; ++index) {
        result[index] = FloatToBf16Bits(values[index]);
      }
      break;
    }
    case core::GgmlType::kF16: {
      const auto* values = static_cast<const std::uint16_t*>(weight.data);
      for (std::size_t index = 0; index < count; ++index) {
        result[index] = FloatToBf16Bits(quant::Fp16ToFloat(values[index]));
      }
      break;
    }
    default:
      throw std::runtime_error("MOSS-Music audio weight is not BF16/F16/F32");
  }
  return result;
}

struct Layer {
  DeviceBuffer<Bf16> ln1_weight, ln1_bias;
  DeviceBuffer<Bf16> q_weight, q_bias, k_weight, v_weight, v_bias;
  DeviceBuffer<Bf16> out_weight, out_bias;
  DeviceBuffer<Bf16> ln2_weight, ln2_bias;
  DeviceBuffer<Bf16> fc1_weight, fc1_bias, fc2_weight, fc2_bias;
};

struct GatedMlp {
  DeviceBuffer<Bf16> gate, up, down;
  std::size_t input{0}, hidden{0}, output{0};
};

}  // namespace

struct AudioEncoder::Impl {
  Impl(const AudioModel& model, hipStream_t stream_in)
      : config(model.config()), stream(stream_in) {
    RequireHipblas(hipblasCreate(&blas), "hipblasCreate MOSS-Music audio");
    RequireHipblas(hipblasSetStream(blas, stream), "hipblasSetStream");
    RequireHipblas(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                   "hipblasSetAtomicsMode");
    lt = std::make_unique<asr::GemmLt>();
    Load(model);
  }

  ~Impl() {
    if (blas != nullptr) {
      (void)hipblasDestroy(blas);
    }
  }

  DeviceBuffer<Bf16> Upload(const Weight& weight) {
    const std::vector<std::uint16_t> host = ToBf16(weight);
    DeviceBuffer<Bf16> device(host.size());
    RequireHip(hipMemcpy(device.get(), host.data(),
                         host.size() * sizeof(std::uint16_t),
                         hipMemcpyHostToDevice),
               "hipMemcpy MOSS-Music audio weight");
    bytes += host.size() * sizeof(std::uint16_t);
    return device;
  }

  DeviceBuffer<Bf16> UploadConvolution(const Weight& weight) {
    DeviceBuffer<Bf16> original = Upload(weight);
    DeviceBuffer<Bf16> packed(original.size());
    asr::LaunchPackConvolutionWeights(original.get(), packed.get(), stream);
    RequireHip(hipStreamSynchronize(stream), "pack MOSS-Music convolution");
    return packed;
  }

  GatedMlp UploadGatedMlp(const GatedMlpWeights& weights) {
    GatedMlp mlp;
    mlp.gate = Upload(weights.gate);
    mlp.up = Upload(weights.up);
    mlp.down = Upload(weights.down);
    mlp.input = static_cast<std::size_t>(weights.gate.cols);
    mlp.hidden = static_cast<std::size_t>(weights.gate.rows);
    mlp.output = static_cast<std::size_t>(weights.down.rows);
    return mlp;
  }

  void Load(const AudioModel& model) {
    const AudioWeights& weights = model.audio();
    conv1_weight = Upload(weights.stem.conv1_weight);
    conv1_bias = Upload(weights.stem.conv1_bias);
    conv2_weight = UploadConvolution(weights.stem.conv2_weight);
    conv2_bias = Upload(weights.stem.conv2_bias);
    conv3_weight = UploadConvolution(weights.stem.conv3_weight);
    conv3_bias = Upload(weights.stem.conv3_bias);
    proj_weight = Upload(weights.stem.proj_weight);
    proj_bias = Upload(weights.stem.proj_bias);
    layers.reserve(weights.layers.size());
    for (const AudioLayerWeights& source : weights.layers) {
      Layer layer;
      layer.ln1_weight = Upload(source.ln1_weight);
      layer.ln1_bias = Upload(source.ln1_bias);
      layer.q_weight = Upload(source.q_weight);
      layer.q_bias = Upload(source.q_bias);
      layer.k_weight = Upload(source.k_weight);
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
    adapter = UploadGatedMlp(model.adapter());
    for (std::size_t k = 0; k < mergers.size(); ++k) {
      mergers[k] = UploadGatedMlp(model.mergers()[k]);
    }
  }

  void Gemm(const void* weight, const void* input, float* output,
            std::size_t rows, std::size_t columns, std::size_t reduction) {
    if (lt->Run(weight, input, output, rows, columns, reduction, stream)) {
      return;
    }
    asr::LaunchGemmBf16(blas, weight, input, output, rows, columns, reduction,
                        stream);
  }

  /// Chunking, the Conv2d stem, stem_proj and positions -> `compact`
  /// [tokens, d_model]. Every chunk is computed at 400 frames (zero-padded),
  /// which matches the reference whenever a full chunk shares the batch. A
  /// clip shorter than one chunk is evaluated unpadded by the reference:
  /// zeroing the padded time columns after each convolution reproduces that
  /// exactly, because the next convolution then reads zeros past the valid
  /// edge just as the reference's one-pixel zero padding does.
  std::size_t Frontend(std::span<const float> log_mel, std::size_t frames) {
    const AudioEncoderConfig& audio = config.audio;
    const std::size_t chunk_frames = audio.ChunkFrames();
    const std::size_t chunk_time = audio.ChunkTokens();
    const std::size_t channels = audio.conv_channels;
    const std::size_t hidden = audio.d_model;
    const std::size_t chunks = (frames + chunk_frames - 1) / chunk_frames;
    const std::size_t tokens =
        AudioTokenCount(static_cast<std::uint32_t>(frames), audio);
    const std::size_t conv1_elements = channels * 64U * (chunk_frames / 2);
    const std::size_t conv2_elements = channels * 32U * (chunk_frames / 4);
    const std::size_t conv3_elements = channels * 16U * chunk_time;
    feature_input.Ensure(log_mel.size());
    input.Ensure(chunks * audio.mel_bins * chunk_frames);
    conv1.Ensure(chunks * conv1_elements);
    conv2.Ensure(chunks * conv2_elements);
    conv3.Ensure(chunks * conv3_elements);
    layout.Ensure(chunks * chunk_time * audio.StemFlattened());
    projected.Ensure(chunks * chunk_time * hidden);
    compact.Ensure(tokens * hidden);
    RequireHip(hipMemcpyAsync(feature_input.get(), log_mel.data(),
                              log_mel.size() * sizeof(float),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync MOSS-Music log-mel");
    LaunchChunkLogMel(feature_input.get(), input.get(), frames, chunks,
                      audio.mel_bins, chunk_frames, stream);
    const bool unpadded = chunks == 1U && frames < chunk_frames;
    std::size_t valid = frames;
    LaunchFirstConvolution(input.get(), conv1_weight.get(), conv1.get(), chunks,
                           stream);
    asr::LaunchBiasGelu(conv1.get(), conv1_bias.get(), chunks * conv1_elements,
                        channels, 64U * (chunk_frames / 2), stream);
    if (unpadded) {
      valid = ConvOutLength(static_cast<std::uint32_t>(valid));
      LaunchZeroColumns(conv1.get(), channels * 64U, chunk_frames / 2, valid,
                        stream);
    }
    LaunchConvolution(conv1.get(), conv2_weight.get(), conv2.get(), chunks, 64,
                      chunk_frames / 2, stream);
    asr::LaunchBiasGelu(conv2.get(), conv2_bias.get(), chunks * conv2_elements,
                        channels, 32U * (chunk_frames / 4), stream);
    if (unpadded) {
      valid = ConvOutLength(static_cast<std::uint32_t>(valid));
      LaunchZeroColumns(conv2.get(), channels * 32U, chunk_frames / 4, valid,
                        stream);
    }
    LaunchConvolution(conv2.get(), conv3_weight.get(), conv3.get(), chunks, 32,
                      chunk_frames / 4, stream);
    asr::LaunchBiasGelu(conv3.get(), conv3_bias.get(), chunks * conv3_elements,
                        channels, 16U * chunk_time, stream);
    LaunchConvOutputLayout(conv3.get(), layout.get(), chunks, chunk_time,
                           stream);
    Gemm(proj_weight.get(), layout.get(), projected.get(), chunks * chunk_time,
         hidden, audio.StemFlattened());
    LaunchAddBiasPositionAndCompact(projected.get(), proj_bias.get(),
                                    compact.get(), frames, chunks, chunk_frames,
                                    chunk_time, tokens, hidden, stream);
    return tokens;
  }

  void RunGatedMlp(const GatedMlp& mlp, const Bf16* input, std::size_t rows,
                   float* output) {
    gate.Ensure(rows * mlp.hidden);
    up.Ensure(rows * mlp.hidden);
    activation.Ensure(rows * mlp.hidden);
    Gemm(mlp.gate.get(), input, gate.get(), rows, mlp.hidden, mlp.input);
    Gemm(mlp.up.get(), input, up.get(), rows, mlp.hidden, mlp.input);
    LaunchSiluMulToBfloat16(gate.get(), up.get(), activation.get(),
                            rows * mlp.hidden, stream);
    Gemm(mlp.down.get(), activation.get(), output, rows, mlp.output,
         mlp.hidden);
    LaunchRoundBfloat16(output, rows * mlp.output, stream);
  }

  void Encode(std::span<const float> log_mel, std::size_t frames,
              const std::function<bool()>& is_cancelled) {
    const auto checkpoint = [&] {
      if (is_cancelled && is_cancelled()) {
        throw std::runtime_error("MOSS-Music audio encoder cancelled");
      }
    };
    const AudioEncoderConfig& audio = config.audio;
    if (frames == 0U || log_mel.size() != audio.mel_bins * frames) {
      throw std::invalid_argument("MOSS-Music log-mel must be [128, frames]");
    }
    checkpoint();
    const std::size_t count = Frontend(log_mel, frames);
    const std::size_t hidden = audio.d_model;
    const std::size_t ffn_width = audio.ffn;
    const std::size_t heads = audio.heads;
    const std::size_t head_dim = audio.HeadDim();
    const std::size_t segment = audio.ChunkTokens();
    const std::size_t elements = count * hidden;

    hidden_state.Ensure(elements);
    normalized.Ensure(count * std::max(hidden, ffn_width));
    q.Ensure(elements);
    k.Ensure(elements);
    v.Ensure(elements);
    attention.Ensure(elements);
    attention_bf16.Ensure(elements);
    projection.Ensure(elements);
    ffn.Ensure(count * ffn_width);
    ffn_bf16.Ensure(count * ffn_width);
    final.Ensure(elements);
    for (DeviceBuffer<float>& tap : taps) {
      tap.Ensure(elements);
    }
    tap_bf16.Ensure(elements);
    adapter_out.Ensure(count * adapter.output);
    for (DeviceBuffer<float>& out : merger_out) {
      out.Ensure(count * adapter.output);
    }
    RequireHip(hipMemcpyAsync(hidden_state.get(), compact.get(),
                              elements * sizeof(float), hipMemcpyDeviceToDevice,
                              stream),
               "hipMemcpyAsync MOSS-Music frontend");

    for (std::size_t index = 0; index < layers.size(); ++index) {
      checkpoint();
      const Layer& layer = layers[index];
      asr::LaunchLayerNormToBfloat16(hidden_state.get(), layer.ln1_weight.get(),
                                     layer.ln1_bias.get(), normalized.get(),
                                     count, hidden, audio.eps, stream);
      Gemm(layer.q_weight.get(), normalized.get(), q.get(), count, hidden,
           hidden);
      Gemm(layer.k_weight.get(), normalized.get(), k.get(), count, hidden,
           hidden);
      Gemm(layer.v_weight.get(), normalized.get(), v.get(), count, hidden,
           hidden);
      asr::LaunchBiasRoundBfloat16(q.get(), layer.q_bias.get(), count, hidden,
                                   stream);
      LaunchRoundBfloat16(k.get(), elements, stream);  // k_proj has no bias
      asr::LaunchBiasRoundBfloat16(v.get(), layer.v_bias.get(), count, hidden,
                                   stream);
      asr::LaunchSegmentedSelfAttention(q.get(), k.get(), v.get(),
                                        attention.get(), attention_bf16.get(),
                                        count, segment, heads, head_dim, stream);
      Gemm(layer.out_weight.get(), attention_bf16.get(), projection.get(),
           count, hidden, hidden);
      asr::LaunchBiasRoundBfloat16(projection.get(), layer.out_bias.get(),
                                   count, hidden, stream);
      asr::LaunchResidualAddBfloat16(hidden_state.get(), projection.get(),
                                     elements, stream);
      asr::LaunchLayerNormToBfloat16(hidden_state.get(), layer.ln2_weight.get(),
                                     layer.ln2_bias.get(), normalized.get(),
                                     count, hidden, audio.eps, stream);
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
      for (std::size_t tap = 0; tap < audio.deepstack_layers.size(); ++tap) {
        if (audio.deepstack_layers[tap] == index) {
          RequireHip(hipMemcpyAsync(taps[tap].get(), hidden_state.get(),
                                    elements * sizeof(float),
                                    hipMemcpyDeviceToDevice, stream),
                     "hipMemcpyAsync MOSS-Music DeepStack tap");
        }
      }
    }

    asr::LaunchLayerNormToBfloat16(hidden_state.get(), post_ln_weight.get(),
                                   post_ln_bias.get(), normalized.get(), count,
                                   hidden, audio.eps, stream);
    LaunchBfloat16ToFloat(normalized.get(), final.get(), elements, stream);
    checkpoint();
    RunGatedMlp(adapter, normalized.get(), count, adapter_out.get());
    for (std::size_t tap = 0; tap < mergers.size(); ++tap) {
      LaunchFloatToBfloat16(taps[tap].get(), tap_bf16.get(), elements, stream);
      RunGatedMlp(mergers[tap], tap_bf16.get(), count, merger_out[tap].get());
    }

    std::uint32_t finite = 1U;
    RequireHip(hipMemcpyAsync(finite_flag.get(), &finite, sizeof(finite),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync MOSS-Music finite flag");
    asr::LaunchAllFinite(adapter_out.get(), count * adapter.output,
                         finite_flag.get(), stream);
    RequireHip(hipMemcpyAsync(&finite, finite_flag.get(), sizeof(finite),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync MOSS-Music finite result");
    RequireHip(hipStreamSynchronize(stream), "MOSS-Music audio encoder");
    if (finite == 0U) {
      throw std::runtime_error(
          "MOSS-Music audio encoder produced non-finite embeddings");
    }
    tokens = count;
  }

  ModelConfig config;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<asr::GemmLt> lt;
  std::size_t bytes{0};
  std::size_t tokens{0};

  DeviceBuffer<Bf16> conv1_weight, conv1_bias, conv2_weight, conv2_bias;
  DeviceBuffer<Bf16> conv3_weight, conv3_bias, proj_weight, proj_bias;
  std::vector<Layer> layers;
  DeviceBuffer<Bf16> post_ln_weight, post_ln_bias;
  GatedMlp adapter;
  std::array<GatedMlp, 3> mergers;

  DeviceBuffer<float> feature_input;
  DeviceBuffer<Bf16> input, conv1, conv2, conv3, layout;
  DeviceBuffer<float> projected, compact;
  DeviceBuffer<float> hidden_state;
  DeviceBuffer<Bf16> normalized;
  DeviceBuffer<float> q, k, v, attention;
  DeviceBuffer<Bf16> attention_bf16;
  DeviceBuffer<float> projection, ffn;
  DeviceBuffer<Bf16> ffn_bf16;
  DeviceBuffer<float> final;
  std::array<DeviceBuffer<float>, 3> taps;
  DeviceBuffer<Bf16> tap_bf16;
  DeviceBuffer<float> gate, up;
  DeviceBuffer<Bf16> activation;
  DeviceBuffer<float> adapter_out;
  std::array<DeviceBuffer<float>, 3> merger_out;
  DeviceBuffer<std::uint32_t> finite_flag{1U};
};

AudioEncoder::AudioEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AudioEncoder::~AudioEncoder() = default;

std::unique_ptr<AudioEncoder> AudioEncoder::Create(const AudioModel& model,
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
                          EncodedAudio* output, std::string* error,
                          const std::function<bool()>& is_cancelled) {
  try {
    impl_->Encode(log_mel, frames, is_cancelled);
    output->tokens = impl_->tokens;
    output->frontend = impl_->compact.get();
    for (std::size_t tap = 0; tap < output->taps.size(); ++tap) {
      output->taps[tap] = impl_->taps[tap].get();
      output->mergers[tap] = impl_->merger_out[tap].get();
    }
    output->final = impl_->final.get();
    output->adapter = impl_->adapter_out.get();
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

bool AudioEncoder::CopyToHost(const float* device, std::size_t count,
                              std::vector<float>* host,
                              std::string* error) const {
  host->resize(count);
  const hipError_t status =
      hipMemcpy(host->data(), device, count * sizeof(float),
                hipMemcpyDeviceToHost);
  if (status != hipSuccess) {
    if (error != nullptr) {
      *error = std::string("hipMemcpy MOSS-Music result: ") +
               hipGetErrorString(status);
    }
    return false;
  }
  return true;
}

std::size_t AudioEncoder::weight_bytes() const { return impl_->bytes; }

}  // namespace gufo::models::moss_music::hip
