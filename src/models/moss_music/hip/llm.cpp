#include "src/models/moss_music/hip/llm.hpp"

#include <hip/hip_bfloat16.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/moss_music/hip/stem_ops.hpp"
#include "src/models/qwen3_asr/hip/blas.hpp"
#include "src/models/qwen3_asr/hip/text_ops.hpp"

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
      RequireHip(hipFree(data_), "hipFree MOSS-Music LLM buffer");
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0U) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("MOSS-Music LLM buffer size overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc MOSS-Music LLM buffer (" +
            std::to_string(count * sizeof(Element) >> 20) + " MiB)");
    count_ = count;
  }
  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t MetadataU32(const core::GgufReader& reader, std::string_view key) {
  const auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    throw std::runtime_error("MOSS-Music LLM GGUF metadata missing: " +
                             std::string(key));
  }
  return *value;
}

const core::GgufTensorInfo& Require(const core::GgufReader& reader,
                                    std::string_view name,
                                    std::initializer_list<std::uint64_t> shape,
                                    core::GgmlType type) {
  const core::GgufTensorInfo* info = reader.FindTensor(name);
  if (info == nullptr) {
    throw std::runtime_error("MOSS-Music LLM tensor missing: " +
                             std::string(name));
  }
  bool ok = info->type == type && info->dimensions.size() == shape.size();
  std::size_t index = 0;
  for (const std::uint64_t dimension : shape) {
    ok = ok && index < info->dimensions.size() &&
         info->dimensions[index++] == dimension;
  }
  if (!ok) {
    throw std::runtime_error("MOSS-Music LLM tensor " + std::string(name) +
                             " has an unexpected shape or type");
  }
  return *info;
}

struct LayerWeights {
  DeviceBuffer<Bf16> q, k, v, o, gate, up, down;
  DeviceBuffer<float> input_norm, q_norm, k_norm, post_norm;
};

struct GemmProjection {
  const void* weights{nullptr};
  float* output{nullptr};
  std::size_t rows{0U};
};

}  // namespace

struct Llm::Impl {
  Impl(const std::filesystem::path& gguf, std::size_t max_context_in,
       hipStream_t stream_in)
      : maximum_tokens(max_context_in), stream(stream_in) {
    if (maximum_tokens == 0U) {
      throw std::invalid_argument("MOSS-Music LLM context must be positive");
    }
    std::string open_error;
    reader = core::GgufReader::OpenFile(gguf, &open_error);
    if (reader == nullptr) {
      throw std::runtime_error("cannot open " + gguf.string() + ": " +
                               open_error);
    }
    const auto arch = reader->GetMetadataString("general.architecture");
    if (!arch.has_value() || *arch != "qwen3") {
      throw std::runtime_error(gguf.string() + " is not a qwen3 GGUF");
    }
    config.layers = MetadataU32(*reader, "qwen3.block_count");
    config.hidden = MetadataU32(*reader, "qwen3.embedding_length");
    config.heads = MetadataU32(*reader, "qwen3.attention.head_count");
    config.kv_heads = MetadataU32(*reader, "qwen3.attention.head_count_kv");
    config.intermediate = MetadataU32(*reader, "qwen3.feed_forward_length");
    config.head_dim = reader->GetMetadataUint32("qwen3.attention.key_length")
                          .value_or(config.hidden / config.heads);
    config.max_positions = MetadataU32(*reader, "qwen3.context_length");
    config.rope_theta =
        reader->GetMetadataFloat32("qwen3.rope.freq_base").value_or(1.0e6F);
    config.eps =
        reader->GetMetadataFloat32("qwen3.attention.layer_norm_rms_epsilon")
            .value_or(1.0e-6F);
    const core::GgufTensorInfo* embed = reader->FindTensor("token_embd.weight");
    if (embed == nullptr || embed->dimensions.size() != 2U) {
      throw std::runtime_error("MOSS-Music LLM GGUF has no token_embd.weight");
    }
    config.vocab = static_cast<std::uint32_t>(embed->dimensions[1]);
    if (maximum_tokens > config.max_positions) {
      throw std::invalid_argument("MOSS-Music LLM context exceeds the model's");
    }

    RequireHipblas(hipblasCreate(&blas), "hipblasCreate MOSS-Music LLM");
    RequireHipblas(hipblasSetStream(blas, stream), "hipblasSetStream");
    RequireHipblas(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                   "hipblasSetAtomicsMode");
    prefill_lt = std::make_unique<asr::GemmLt>();
    LoadWeights();
    AllocateState();
  }

  ~Impl() {
    if (blas != nullptr) {
      (void)hipblasDestroy(blas);
    }
  }

  /// A BF16 matrix straight from the file into device memory; the mapped
  /// pages are never read on the host (Windows would keep them resident).
  DeviceBuffer<Bf16> UploadMatrix(gufo::hip::WeightUpload& upload,
                                  std::string_view name, std::uint64_t cols,
                                  std::uint64_t rows) {
    const core::GgufTensorInfo& info =
        Require(*reader, name, {cols, rows}, core::GgmlType::kBF16);
    DeviceBuffer<Bf16> device(static_cast<std::size_t>(cols * rows));
    std::string error;
    if (!upload.CopyMapped(info.data, info.size_bytes, device.get(), &error)) {
      throw std::runtime_error("MOSS-Music LLM upload of " + std::string(name) +
                               " failed: " + error);
    }
    bytes += info.size_bytes;
    return device;
  }

  DeviceBuffer<float> UploadNorm(std::string_view name, std::uint64_t size) {
    const core::GgufTensorInfo& info =
        Require(*reader, name, {size}, core::GgmlType::kF32);
    DeviceBuffer<float> device(static_cast<std::size_t>(size));
    RequireHip(hipMemcpy(device.get(), info.data, info.size_bytes,
                         hipMemcpyHostToDevice),
               "hipMemcpy MOSS-Music LLM norm");
    bytes += info.size_bytes;
    return device;
  }

  void LoadWeights() {
    std::string error;
    std::unique_ptr<gufo::hip::WeightUpload> upload =
        gufo::hip::WeightUpload::Create(reader->GetMappedRegions(), &error);
    if (upload == nullptr) {
      throw std::runtime_error("MOSS-Music LLM weight upload: " + error);
    }
    const std::uint64_t h = config.hidden;
    embedding = UploadMatrix(*upload, "token_embd.weight", h, config.vocab);
    lm_head = UploadMatrix(*upload, "output.weight", h, config.vocab);
    final_norm = UploadNorm("output_norm.weight", h);
    layers.reserve(config.layers);
    for (std::uint32_t layer = 0; layer < config.layers; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      LayerWeights weights;
      weights.q = UploadMatrix(*upload, p + "attn_q.weight", h, config.QDim());
      weights.k = UploadMatrix(*upload, p + "attn_k.weight", h, config.KvDim());
      weights.v = UploadMatrix(*upload, p + "attn_v.weight", h, config.KvDim());
      weights.o =
          UploadMatrix(*upload, p + "attn_output.weight", config.QDim(), h);
      weights.gate =
          UploadMatrix(*upload, p + "ffn_gate.weight", h, config.intermediate);
      weights.up =
          UploadMatrix(*upload, p + "ffn_up.weight", h, config.intermediate);
      weights.down =
          UploadMatrix(*upload, p + "ffn_down.weight", config.intermediate, h);
      weights.input_norm = UploadNorm(p + "attn_norm.weight", h);
      weights.post_norm = UploadNorm(p + "ffn_norm.weight", h);
      weights.q_norm = UploadNorm(p + "attn_q_norm.weight", config.head_dim);
      weights.k_norm = UploadNorm(p + "attn_k_norm.weight", config.head_dim);
      layers.push_back(std::move(weights));
    }
    if (!upload->Finish(&error)) {
      throw std::runtime_error("MOSS-Music LLM weight upload: " + error);
    }
  }

  void AllocateState() {
    const std::size_t capacity = maximum_tokens;
    hidden.Reset(capacity * config.hidden);
    q.Reset(capacity * config.QDim());
    k.Reset(capacity * config.KvDim());
    v.Reset(capacity * config.KvDim());
    attention.Reset(capacity * config.hidden);
    projection.Reset(capacity * config.hidden);
    gate.Reset(capacity * config.intermediate);
    up.Reset(capacity * config.intermediate);
    feed_forward.Reset(capacity * config.hidden);
    bfloat16_scratch.Reset(capacity *
                           std::max(config.hidden, config.intermediate));
    gemv_input.Reset(std::max(config.hidden, config.intermediate));
    const std::size_t cache =
        static_cast<std::size_t>(config.layers) * config.kv_heads * capacity *
        config.head_dim;
    key_cache.Reset(cache);
    value_cache.Reset(cache);
    logits.Reset(config.vocab);
    token_ids.Reset(capacity);
    audio_rows.Reset(capacity);
  }

  void GemmFused(const GemmProjection* projections, std::uint32_t count,
                 const void* inputs_bfloat16, std::size_t batch,
                 std::size_t columns) {
    if (batch == 1U && count <= asr::kTextDecodeGemvMaxTensors) {
      std::array<asr::TextDecodeGemvTensor, asr::kTextDecodeGemvMaxTensors>
          tensors{};
      bool representable = true;
      for (std::uint32_t index = 0U; index < count; ++index) {
        if (projections[index].rows >
            std::numeric_limits<std::uint32_t>::max()) {
          representable = false;
          break;
        }
        tensors[index] = {
            .weights_bfloat16 = projections[index].weights,
            .output = projections[index].output,
            .rows = static_cast<std::uint32_t>(projections[index].rows)};
      }
      if (representable &&
          asr::LaunchTextDecodeGemv(tensors.data(), count, inputs_bfloat16,
                                    columns, stream)) {
        return;
      }
    }
    for (std::uint32_t index = 0U; index < count; ++index) {
      if (batch == 1U) {
        asr::LaunchTextBfloat16ToFloat(inputs_bfloat16, gemv_input.get(),
                                       columns, stream);
        asr::LaunchTextRowGemv(projections[index].weights, gemv_input.get(),
                               projections[index].output,
                               projections[index].rows, columns, stream);
        continue;
      }
      if (prefill_lt != nullptr &&
          prefill_lt->Run(projections[index].weights, inputs_bfloat16,
                          projections[index].output, batch,
                          projections[index].rows, columns, stream)) {
        continue;
      }
      asr::LaunchGemmBf16(blas, projections[index].weights, inputs_bfloat16,
                          projections[index].output, batch,
                          projections[index].rows, columns, stream);
    }
  }

  void Gemm(const void* weights, const void* inputs_bfloat16, float* output,
            std::size_t batch, std::size_t rows, std::size_t columns) {
    const GemmProjection projection{
        .weights = weights, .output = output, .rows = rows};
    GemmFused(&projection, 1U, inputs_bfloat16, batch, columns);
  }

  void CopyRows(const float* device, std::size_t count,
                std::vector<float>* host) {
    host->resize(count);
    RequireHip(hipMemcpyAsync(host->data(), device, count * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync MOSS-Music LLM trace");
    RequireHip(hipStreamSynchronize(stream), "MOSS-Music LLM trace sync");
  }

  /// `tokens` rows of `hidden` at positions start.., with `deepstack[j]`
  /// (device, [audio_count, hidden]) added to rows `audio_row_count` of
  /// `audio_rows` after layers 0..2 when present.
  void RunTokens(std::size_t tokens, std::uint32_t start_position,
                 const std::array<const float*, 3>& deepstack,
                 std::size_t audio_count, PrefillTrace* trace,
                 const std::function<bool()>& is_cancelled) {
    if (tokens == 0U ||
        static_cast<std::size_t>(start_position) + tokens > maximum_tokens) {
      throw std::length_error("MOSS-Music LLM step exceeds the KV capacity");
    }
    const std::size_t hidden_elements = tokens * config.hidden;
    const std::size_t ffn_elements = tokens * config.intermediate;
    if (trace != nullptr) {
      CopyRows(hidden.get(), hidden_elements, &trace->hs0);
    }
    asr::LaunchTextRMSNorm(hidden.get(), layers.front().input_norm.get(),
                           bfloat16_scratch.get(), tokens, config.hidden,
                           config.eps, stream);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
      if (is_cancelled && is_cancelled()) {
        throw std::runtime_error("MOSS-Music generation cancelled");
      }
      const LayerWeights& weights = layers[layer];
      const std::array<GemmProjection, 3> qkv{
          GemmProjection{.weights = weights.q.get(),
                         .output = q.get(),
                         .rows = config.QDim()},
          GemmProjection{.weights = weights.k.get(),
                         .output = k.get(),
                         .rows = config.KvDim()},
          GemmProjection{.weights = weights.v.get(),
                         .output = v.get(),
                         .rows = config.KvDim()}};
      GemmFused(qkv.data(), 3U, bfloat16_scratch.get(), tokens, config.hidden);
      if (!asr::LaunchTextQkNormRoPE(
              q.get(), k.get(), v.get(), weights.q_norm.get(),
              weights.k_norm.get(), tokens, config.heads, config.kv_heads,
              config.head_dim, start_position, config.rope_theta, config.eps,
              key_cache.get(), value_cache.get(),
              static_cast<std::uint32_t>(layer),
              static_cast<std::uint32_t>(maximum_tokens), stream)) {
        throw std::runtime_error(
            "MOSS-Music LLM q/k normalization shape is unsupported");
      }
      if (tokens != 1U ||
          !asr::LaunchTextDecodeAttention(
              q.get(), key_cache.get(), value_cache.get(), attention.get(),
              bfloat16_scratch.get(), static_cast<std::uint32_t>(layer),
              start_position, static_cast<std::uint32_t>(maximum_tokens),
              config.heads, config.kv_heads, config.head_dim, stream)) {
        asr::LaunchTextBatchedAttention(
            q.get(), key_cache.get(), value_cache.get(), attention.get(),
            bfloat16_scratch.get(), static_cast<std::uint32_t>(layer),
            start_position, tokens, static_cast<std::uint32_t>(maximum_tokens),
            config.heads, config.kv_heads, config.head_dim, stream);
      }
      Gemm(weights.o.get(), bfloat16_scratch.get(), projection.get(), tokens,
           config.hidden, config.QDim());
      asr::LaunchTextResidualAddRMSNorm(
          hidden.get(), projection.get(), weights.post_norm.get(),
          bfloat16_scratch.get(), tokens, config.hidden, config.eps, stream);
      const std::array<GemmProjection, 2> gate_up{
          GemmProjection{.weights = weights.gate.get(),
                         .output = gate.get(),
                         .rows = config.intermediate},
          GemmProjection{.weights = weights.up.get(),
                         .output = up.get(),
                         .rows = config.intermediate}};
      GemmFused(gate_up.data(), 2U, bfloat16_scratch.get(), tokens,
                config.hidden);
      asr::LaunchTextSwiGLU(gate.get(), up.get(), bfloat16_scratch.get(),
                            ffn_elements, stream);
      Gemm(weights.down.get(), bfloat16_scratch.get(), feed_forward.get(),
           tokens, config.hidden, config.intermediate);
      const float* next_norm = layer + 1U < layers.size()
                                   ? layers[layer + 1U].input_norm.get()
                                   : final_norm.get();
      asr::LaunchTextResidualAddRMSNorm(hidden.get(), feed_forward.get(),
                                        next_norm, bfloat16_scratch.get(),
                                        tokens, config.hidden, config.eps,
                                        stream);
      // DeepStack: the merger of encoder layer 8/16/24 joins the audio rows
      // after LLM layer 0/1/2 (a forward hook on the layer's output); the
      // normalization of the next layer's input is then redone.
      if (layer < deepstack.size() && deepstack[layer] != nullptr &&
          audio_count > 0U) {
        LaunchAddRowsRounded(hidden.get(), deepstack[layer], audio_rows.get(),
                             audio_count, config.hidden, stream);
        asr::LaunchTextRMSNorm(hidden.get(), next_norm, bfloat16_scratch.get(),
                               tokens, config.hidden, config.eps, stream);
      }
      if (trace != nullptr && layer < trace->hs_layer.size()) {
        CopyRows(hidden.get(), hidden_elements, &trace->hs_layer[layer]);
      }
    }
    if (trace != nullptr) {
      asr::LaunchTextBfloat16ToFloat(bfloat16_scratch.get(), feed_forward.get(),
                                     hidden_elements, stream);
      CopyRows(feed_forward.get(), hidden_elements, &trace->hs_final);
    }
    const Bf16* last_hidden =
        bfloat16_scratch.get() + ((tokens - 1U) * config.hidden);
    Gemm(lm_head.get(), last_hidden, logits.get(), 1U, config.vocab,
         config.hidden);
    asr::LaunchTextRoundBfloat16(logits.get(), config.vocab, stream);
  }

  void FetchLogits(std::vector<float>* host) {
    host->resize(config.vocab);
    RequireHip(hipMemcpyAsync(host->data(), logits.get(),
                              config.vocab * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync MOSS-Music logits");
    RequireHip(hipStreamSynchronize(stream), "MOSS-Music LLM step");
    RequireHip(hipGetLastError(), "MOSS-Music LLM kernels");
  }

  void Prefill(std::span<const std::int32_t> prompt,
               std::span<const std::uint32_t> rows, const float* audio_device,
               const std::array<const float*, 3>& deepstack,
               std::vector<float>* host_logits, PrefillTrace* trace,
               const std::function<bool()>& is_cancelled) {
    if (prompt.empty() || prompt.size() > maximum_tokens) {
      throw std::invalid_argument("MOSS-Music prompt length is invalid");
    }
    if (!rows.empty() && audio_device == nullptr) {
      throw std::invalid_argument("MOSS-Music audio rows without embeddings");
    }
    std::vector<std::uint32_t> ids(prompt.size());
    for (std::size_t index = 0; index < prompt.size(); ++index) {
      if (prompt[index] < 0 ||
          static_cast<std::uint32_t>(prompt[index]) >= config.vocab) {
        throw std::out_of_range("MOSS-Music prompt token is out of range");
      }
      ids[index] = static_cast<std::uint32_t>(prompt[index]);
    }
    for (const std::uint32_t row : rows) {
      if (row >= prompt.size()) {
        throw std::out_of_range("MOSS-Music audio row is outside the prompt");
      }
    }
    RequireHip(hipMemcpyAsync(token_ids.get(), ids.data(),
                              ids.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync MOSS-Music prompt ids");
    asr::LaunchTextEmbeddingLookup(embedding.get(), token_ids.get(),
                                   hidden.get(), prompt.size(), config.hidden,
                                   stream);
    if (!rows.empty()) {
      RequireHip(hipMemcpyAsync(audio_rows.get(), rows.data(),
                                rows.size() * sizeof(std::uint32_t),
                                hipMemcpyHostToDevice, stream),
                 "hipMemcpyAsync MOSS-Music audio rows");
      LaunchScatterRowsRounded(hidden.get(), audio_device, audio_rows.get(),
                               rows.size(), config.hidden, stream);
    }
    RunTokens(prompt.size(), 0U, deepstack, rows.size(), trace, is_cancelled);
    cache_tokens = prompt.size();
    FetchLogits(host_logits);
  }

  void Decode(std::int32_t token, std::vector<float>* host_logits) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= config.vocab) {
      throw std::out_of_range("MOSS-Music generated token is out of range");
    }
    if (cache_tokens >= maximum_tokens) {
      throw std::length_error("MOSS-Music generation exceeds the KV capacity");
    }
    const auto id = static_cast<std::uint32_t>(token);
    RequireHip(hipMemcpyAsync(token_ids.get(), &id, sizeof(id),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync MOSS-Music token");
    asr::LaunchTextEmbeddingLookup(embedding.get(), token_ids.get(),
                                   hidden.get(), 1U, config.hidden, stream);
    RunTokens(1U, static_cast<std::uint32_t>(cache_tokens), {}, 0U, nullptr,
              {});
    ++cache_tokens;
    FetchLogits(host_logits);
  }

  std::unique_ptr<core::GgufReader> reader;
  TextConfig config;
  std::size_t maximum_tokens{0};
  std::size_t cache_tokens{0};
  std::size_t bytes{0};
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<asr::GemmLt> prefill_lt;

  DeviceBuffer<Bf16> embedding, lm_head;
  DeviceBuffer<float> final_norm;
  std::vector<LayerWeights> layers;

  DeviceBuffer<float> hidden, q, k, v, attention, projection, gate, up,
      feed_forward;
  DeviceBuffer<Bf16> bfloat16_scratch;
  DeviceBuffer<float> gemv_input;
  DeviceBuffer<float> key_cache, value_cache;
  DeviceBuffer<float> logits;
  DeviceBuffer<std::uint32_t> token_ids;
  DeviceBuffer<std::uint32_t> audio_rows;
};

Llm::Llm(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Llm::~Llm() = default;

std::unique_ptr<Llm> Llm::Create(const std::filesystem::path& gguf,
                                 std::size_t max_context, hipStream_t stream,
                                 std::string* error) {
  try {
    return std::unique_ptr<Llm>(
        new Llm(std::make_unique<Impl>(gguf, max_context, stream)));
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
}

const TextConfig& Llm::config() const { return impl_->config; }

const core::GgufReader& Llm::reader() const { return *impl_->reader; }

std::size_t Llm::position() const { return impl_->cache_tokens; }

std::size_t Llm::max_context() const { return impl_->maximum_tokens; }

std::size_t Llm::weight_bytes() const { return impl_->bytes; }

void Llm::Reset() { impl_->cache_tokens = 0; }

bool Llm::Prefill(std::span<const std::int32_t> prompt,
                  std::span<const std::uint32_t> audio_rows,
                  const float* audio_device,
                  const std::array<const float*, 3>& deepstack_device,
                  std::vector<float>* logits, std::string* error,
                  PrefillTrace* trace,
                  const std::function<bool()>& is_cancelled) {
  try {
    impl_->Prefill(prompt, audio_rows, audio_device, deepstack_device, logits,
                   trace, is_cancelled);
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

bool Llm::Decode(std::int32_t token, std::vector<float>* logits,
                 std::string* error) {
  try {
    impl_->Decode(token, logits);
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

}  // namespace gufo::models::moss_music::hip
