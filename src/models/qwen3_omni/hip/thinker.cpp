#include "src/models/qwen3_omni/hip/thinker.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "ds4_mmq.h"
#include "src/core/platform/pages.hpp"
#include "src/models/qwen3_omni/hip/device_buffer.hpp"
#include "src/models/qwen3_omni/hip/thinker_kernels.hpp"

namespace gufo::models::qwen3_omni::hip {
namespace {

/// Tokens per prefill pass; longer inputs run in consecutive chunks.
constexpr std::size_t kChunk = 512;
/// The vector (MMVQ) kernels serve up to this many activation rows.
constexpr std::size_t kVectorRows = 8;

struct DeviceWeight {
  DeviceBuffer<std::uint8_t> bytes;
  core::GgmlType type{core::GgmlType::kF32};
  int rows{0};
  int cols{0};
  int experts{1};

  [[nodiscard]] const void* data() const { return bytes.get(); }
  [[nodiscard]] const float* f32() const {
    return reinterpret_cast<const float*>(bytes.get());
  }
};

struct Layer {
  DeviceWeight attn_norm, o, q_norm, k_norm, ffn_norm, router;
  // q and k stacked into one matrix, with v when it shares their K-quant
  // type (one activation quantize and one launch instead of three).
  DeviceWeight qkv;
  DeviceWeight v;  // only when v is not part of qkv
  bool v_in_qkv{false};
  DeviceWeight gate_exps, up_exps, down_exps;
};

void Check(int status, const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) + " failed (" +
                             std::to_string(status) + ")");
  }
}

}  // namespace

/// GUFO_OMNI_PROFILE=1: synchronized wall time per decode phase (one-token
/// forwards only), printed when the thinker is destroyed.
enum Phase : std::size_t {
  kQkv,
  kV,
  kAttention,
  kOutput,
  kRoute,
  kGateUp,
  kDown,
  kHead,
  kPhases
};
constexpr std::array<const char*, kPhases> kPhaseNames = {
    "norm+qkv", "v", "rope+kv+attention", "o+residual",
    "norm+route", "gate/up", "down+combine", "head"};

struct Thinker::Impl {
  bool profile{std::getenv("GUFO_OMNI_PROFILE") != nullptr};
  bool profiling_step{false};
  std::array<double, kPhases> phase_ms{};
  std::size_t profiled_steps{0};
  std::chrono::steady_clock::time_point mark;

  void Mark(Phase phase) {
    if (!profiling_step) {
      return;
    }
    RequireHip(hipStreamSynchronize(stream), "profile sync");
    const auto now = std::chrono::steady_clock::now();
    phase_ms[phase] +=
        std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
  }

  ~Impl() {
    if (profile && profiled_steps > 0) {
      double total = 0.0;
      for (const double value : phase_ms) {
        total += value;
      }
      std::cerr << "[omni-profile] " << profiled_steps
                << " decode steps, ms per token:\n";
      for (std::size_t phase = 0; phase < kPhases; ++phase) {
        std::cerr << "  " << kPhaseNames[phase] << ": "
                  << phase_ms[phase] / static_cast<double>(profiled_steps)
                  << '\n';
      }
      std::cerr << "  total: " << total / static_cast<double>(profiled_steps)
                << '\n';
    }
  }

  Impl(const Model& model, std::size_t context, hipStream_t stream_in)
      : config(model.text_config()), capacity(context), stream(stream_in) {
    if (capacity == 0U || capacity > MaxAttentionContext()) {
      throw std::invalid_argument(
          "Qwen3-Omni context must be 1.." +
          std::to_string(MaxAttentionContext()) + " tokens");
    }
    if (config.head_dim != 128U || config.experts > 256U ||
        config.experts_used > 16U || config.heads % config.kv_heads != 0U) {
      throw std::runtime_error("Qwen3-Omni thinker geometry unsupported");
    }
    Check(ds4_mmq_init(0), "ds4_mmq_init");
    const TextWeights& weights = model.text();
    output_norm = Upload(weights.output_norm);
    output = Upload(weights.output);
    layers.reserve(weights.layers.size());
    for (const TextLayer& source : weights.layers) {
      Layer layer;
      layer.attn_norm = Upload(source.attn_norm);
      if (source.k.type != source.q.type) {
        throw std::runtime_error("Qwen3-Omni q and k differ in type");
      }
      layer.v_in_qkv = source.v.type == source.q.type;
      layer.qkv = layer.v_in_qkv
                      ? UploadStacked({&source.q, &source.k, &source.v})
                      : UploadStacked({&source.q, &source.k});
      if (!layer.v_in_qkv) {
        layer.v = Upload(source.v);
      }
      layer.o = Upload(source.o);
      layer.q_norm = Upload(source.q_norm);
      layer.k_norm = Upload(source.k_norm);
      layer.ffn_norm = Upload(source.ffn_norm);
      layer.router = Upload(source.router);
      layer.gate_exps = Upload(source.gate_exps);
      layer.up_exps = Upload(source.up_exps);
      layer.down_exps = Upload(source.down_exps);
      layers.push_back(std::move(layer));
      // The uploads above are synchronous. Drop the layer's mapped pages:
      // Windows keeps them resident and the 21 GB GGUF fills the host.
      gufo::platform::TrimResidentPages();
    }
    const std::size_t kv = capacity * config.KvDim();
    keys.resize(config.layers);
    values.resize(config.layers);
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
      keys[layer].Reset(kv);
      values[layer].Reset(kv);
    }
    const std::size_t hidden = config.hidden;
    const std::size_t rows = kChunk * config.experts_used;
    residual.Reset(kChunk * hidden);
    normalized.Reset(kChunk * hidden);
    qkv_out.Reset(kChunk * (config.QDim() + 2 * config.KvDim()));
    q.Reset(kChunk * config.QDim());
    k.Reset(kChunk * config.KvDim());
    v.Reset(kChunk * config.KvDim());
    attended.Reset(kChunk * config.QDim());
    projected.Reset(kChunk * hidden);
    ids.Reset(rows);
    route_weights.Reset(rows);
    router_logits.Reset(kChunk * config.experts);
    gate.Reset(rows * config.expert_ff);
    up.Reset(rows * config.expert_ff);
    activated.Reset(rows * config.expert_ff);
    down.Reset(rows * hidden);
    logits.Reset(config.vocab);
    q8.Reset(std::max({ds4_mmq_q8_1_bytes(static_cast<int>(hidden), 1),
                       ds4_mmq_q8_1_bytes(static_cast<int>(config.QDim()), 1),
                       ds4_mmq_q8_1_bytes(static_cast<int>(config.expert_ff),
                                          static_cast<int>(
                                              config.experts_used))}));
  }

  /// Decode: Q8_1 rows of x, shared by every matrix that reads x next.
  void Quantize(const float* x, std::size_t cols, std::size_t rows) {
    Check(ds4_mmq_quantize_q8_1(x, q8.get(), static_cast<int>(cols),
                                static_cast<int>(rows), stream),
          "Qwen3-Omni activation quantize");
  }

  /// Decode: MMVQ on the activation last passed to Quantize. Without `route`
  /// a dense matrix over `columns` rows; with it an expert stack (and with
  /// `gate` the fused (W . x) * silu(gate . x)).
  void Vector(const DeviceWeight& weight, const DeviceWeight* gate,
              const std::int32_t* route, float* out, std::size_t columns,
              std::size_t slots) {
    Check(ds4_mmq_vec_preq(static_cast<int>(weight.type), weight.data(),
                           gate != nullptr ? gate->data() : nullptr, q8.get(),
                           route, out, weight.rows, weight.cols,
                           static_cast<int>(columns), weight.experts,
                           static_cast<int>(slots), stream),
          "Qwen3-Omni vector projection");
  }

  DeviceWeight Upload(const Weight& source) {
    DeviceWeight weight;
    weight.type = source.type;
    weight.cols = static_cast<int>(source.cols);
    weight.rows = static_cast<int>(source.rows);
    weight.experts = static_cast<int>(source.experts);
    weight.bytes.Reset(source.Bytes());
    RequireHip(hipMemcpy(weight.bytes.get(), source.data, source.Bytes(),
                         hipMemcpyHostToDevice),
               "hipMemcpy Qwen3-Omni thinker weight");
    bytes += source.Bytes();
    return weight;
  }

  /// Matrices of one type and width, stacked row-wise into one weight.
  DeviceWeight UploadStacked(std::initializer_list<const Weight*> parts) {
    DeviceWeight weight;
    const Weight& first = **parts.begin();
    weight.type = first.type;
    weight.cols = static_cast<int>(first.cols);
    std::size_t total = 0;
    for (const Weight* part : parts) {
      if (part->type != first.type || part->cols != first.cols) {
        throw std::runtime_error("Qwen3-Omni stacked weights differ");
      }
      weight.rows += static_cast<int>(part->rows);
      total += part->Bytes();
    }
    weight.bytes.Reset(total);
    std::size_t offset = 0;
    for (const Weight* part : parts) {
      RequireHip(hipMemcpy(weight.bytes.get() + offset, part->data,
                           part->Bytes(), hipMemcpyHostToDevice),
                 "hipMemcpy Qwen3-Omni stacked weight");
      offset += part->Bytes();
    }
    bytes += total;
    return weight;
  }

  /// y[n][rows] = x[n][cols] . W^T for a K-quant matrix.
  void Dense(const DeviceWeight& weight, const float* x, float* y,
             std::size_t n) {
    const int count = static_cast<int>(n);
    const bool q4 = weight.type == core::GgmlType::kQ4_K;
    if (n <= kVectorRows) {
      Check(q4 ? ds4_mmq_q4_K_dense_vec(weight.data(), x, y, weight.rows,
                                        count, weight.cols, stream)
               : ds4_mmq_q6_K_dense_vec(weight.data(), x, y, weight.rows,
                                        count, weight.cols, stream),
            "Qwen3-Omni dense vector projection");
      return;
    }
    Check(q4 ? ds4_mmq_q4_K_dense(weight.data(), x, y, weight.rows, count,
                                  weight.cols, stream)
             : ds4_mmq_q6_K_dense(weight.data(), x, y, weight.rows, count,
                                  weight.cols, stream),
          "Qwen3-Omni dense projection");
  }

  /// out[(t * used + s)][rows] = x[t] . W[ids[t][s]]^T.
  void Experts(const DeviceWeight& weight, const float* x, float* out,
               std::size_t tokens, std::size_t used) {
    const int count = static_cast<int>(tokens);
    const int slots = static_cast<int>(used);
    const bool q4 = weight.type == core::GgmlType::kQ4_K;
    if (tokens <= kVectorRows) {
      Check(q4 ? ds4_mmq_q4_K_moe_vec(weight.data(), x, ids.get(), out,
                                      weight.rows, weight.cols, count,
                                      weight.experts, slots, stream)
               : ds4_mmq_q6_K_moe_vec(weight.data(), x, ids.get(), out,
                                      weight.rows, weight.cols, count,
                                      weight.experts, slots, stream),
            "Qwen3-Omni expert vector projection");
      return;
    }
    Check(q4 ? ds4_mmq_q4_K_moe(weight.data(), x, ids.get(), out, weight.rows,
                                weight.cols, count, weight.experts, slots,
                                stream)
             : ds4_mmq_q6_K_moe(weight.data(), x, ids.get(), out, weight.rows,
                                weight.cols, count, weight.experts, slots,
                                stream),
          "Qwen3-Omni expert projection");
  }

  void RunLayer(std::size_t index, std::size_t count) {
    const Layer& layer = layers[index];
    const std::size_t hidden = config.hidden;
    const std::size_t used = config.experts_used;
    LaunchRmsNorm(residual.get(), layer.attn_norm.f32(), normalized.get(),
                  count, hidden, config.eps, stream);
    const std::size_t q_dim = config.QDim();
    const std::size_t kv_dim = config.KvDim();
    const std::size_t width = static_cast<std::size_t>(layer.qkv.rows);
    const bool decode = count == 1;
    if (decode) {
      Quantize(normalized.get(), hidden, 1);
      Vector(layer.qkv, nullptr, nullptr, qkv_out.get(), 1, 1);
    } else {
      Dense(layer.qkv, normalized.get(), qkv_out.get(), count);
    }
    Mark(kQkv);
    float* q_rows = qkv_out.get();
    float* k_rows = q_rows + q_dim;
    float* v_rows = layer.v_in_qkv ? k_rows + kv_dim : v.get();
    if (count > 1) {
      // Rows of the stacked projection interleave q, k and v per token.
      LaunchCopyColumns(qkv_out.get(), width, 0, q_dim, q.get(), count, stream);
      LaunchCopyColumns(qkv_out.get(), width, q_dim, kv_dim, k.get(), count,
                        stream);
      if (layer.v_in_qkv) {
        LaunchCopyColumns(qkv_out.get(), width, q_dim + kv_dim, kv_dim,
                          v.get(), count, stream);
      }
      q_rows = q.get();
      k_rows = k.get();
      v_rows = v.get();
    }
    if (!layer.v_in_qkv) {
      if (decode) {
        Vector(layer.v, nullptr, nullptr, v.get(), 1, 1);
      } else {
        Dense(layer.v, normalized.get(), v.get(), count);
      }
    }
    Mark(kV);
    LaunchQkNormRope(q_rows, k_rows, layer.q_norm.f32(), layer.k_norm.f32(),
                     count, config.heads, config.kv_heads, config.head_dim,
                     position, config.rope_theta, config.eps, stream);
    LaunchStoreKv(k_rows, v_rows, keys[index].get(), values[index].get(),
                  count, kv_dim, position, stream);
    LaunchAttention(q_rows, keys[index].get(), values[index].get(),
                    attended.get(), count, config.heads, config.kv_heads,
                    config.head_dim, position, stream);
    Mark(kAttention);
    if (decode) {
      Quantize(attended.get(), config.QDim(), 1);
      Vector(layer.o, nullptr, nullptr, projected.get(), 1, 1);
    } else {
      Dense(layer.o, attended.get(), projected.get(), count);
    }
    LaunchAdd(residual.get(), projected.get(), count * hidden, stream);
    Mark(kOutput);

    LaunchRmsNorm(residual.get(), layer.ffn_norm.f32(), normalized.get(),
                  count, hidden, config.eps, stream);
    LaunchRoute(normalized.get(), layer.router.f32(), router_logits.get(),
                ids.get(), route_weights.get(), count, hidden, config.experts,
                used, stream);
    Mark(kRoute);
    if (decode) {
      // One fused launch: (up . x) * silu(gate . x).
      Quantize(normalized.get(), hidden, 1);
      Vector(layer.up_exps, &layer.gate_exps, ids.get(), activated.get(), 1,
             used);
    } else {
      Experts(layer.gate_exps, normalized.get(), gate.get(), count, used);
      Experts(layer.up_exps, normalized.get(), up.get(), count, used);
      LaunchSwiglu(gate.get(), up.get(), activated.get(),
                   count * used * config.expert_ff, stream);
    }
    Mark(kGateUp);
    // Every (token, slot) row is its own activation: [count * used] rows
    // with one expert each; ids [count][used] is that list already.
    if (decode) {
      Quantize(activated.get(), config.expert_ff, used);
      Vector(layer.down_exps, nullptr, ids.get(), down.get(), used, 1);
    } else {
      Experts(layer.down_exps, activated.get(), down.get(), count * used, 1);
    }
    LaunchMoeCombine(residual.get(), down.get(), route_weights.get(), count,
                     used, hidden, stream);
    Mark(kDown);
  }

  void Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* host_logits) {
    const std::size_t hidden = config.hidden;
    if (count == 0U || embeddings.size() != count * hidden) {
      throw std::invalid_argument(
          "Qwen3-Omni embeddings must be [count, hidden]");
    }
    if (position + count > capacity) {
      throw std::length_error("Qwen3-Omni context capacity exceeded (" +
                              std::to_string(capacity) + " tokens)");
    }
    profiling_step = profile && count == 1;
    if (profiling_step) {
      RequireHip(hipStreamSynchronize(stream), "profile sync");
      mark = std::chrono::steady_clock::now();
      ++profiled_steps;
    }
    std::size_t last = 0;
    for (std::size_t offset = 0; offset < count; offset += kChunk) {
      const std::size_t chunk = std::min(kChunk, count - offset);
      RequireHip(hipMemcpyAsync(residual.get(),
                                embeddings.data() + offset * hidden,
                                chunk * hidden * sizeof(float),
                                hipMemcpyHostToDevice, stream),
                 "hipMemcpyAsync Qwen3-Omni embeddings");
      for (std::size_t layer = 0; layer < config.layers; ++layer) {
        RunLayer(layer, chunk);
      }
      position += chunk;
      last = chunk - 1;
    }
    LaunchRmsNorm(residual.get() + last * hidden, output_norm.f32(),
                  normalized.get(), 1, hidden, config.eps, stream);
    Quantize(normalized.get(), hidden, 1);
    Vector(output, nullptr, nullptr, logits.get(), 1, 1);
    host_logits->resize(config.vocab);
    RequireHip(hipMemcpyAsync(host_logits->data(), logits.get(),
                              config.vocab * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync Qwen3-Omni logits");
    RequireHip(hipStreamSynchronize(stream), "Qwen3-Omni thinker forward");
    Mark(kHead);
    profiling_step = false;
  }

  TextConfig config;
  std::size_t capacity{0};
  hipStream_t stream{nullptr};
  std::size_t position{0};
  std::size_t bytes{0};

  DeviceWeight output_norm, output;
  std::vector<Layer> layers;
  std::vector<DeviceBuffer<__half>> keys, values;

  DeviceBuffer<float> residual, normalized, qkv_out, q, k, v, attended,
      projected;
  DeviceBuffer<std::int32_t> ids;
  DeviceBuffer<float> route_weights, router_logits, gate, up, activated, down,
      logits;
  DeviceBuffer<std::uint8_t> q8;
};

Thinker::Thinker(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Thinker::~Thinker() = default;

std::unique_ptr<Thinker> Thinker::Create(const Model& model,
                                         std::size_t max_context,
                                         hipStream_t stream,
                                         std::string* error) {
  try {
    return std::unique_ptr<Thinker>(
        new Thinker(std::make_unique<Impl>(model, max_context, stream)));
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
}

void Thinker::Reset() {
  impl_->position = 0;
}

bool Thinker::Forward(std::span<const float> embeddings, std::size_t count,
                      std::vector<float>* logits, std::string* error) {
  try {
    impl_->Forward(embeddings, count, logits);
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

std::size_t Thinker::position() const {
  return impl_->position;
}

std::size_t Thinker::max_context() const {
  return impl_->capacity;
}

std::size_t Thinker::weight_bytes() const {
  return impl_->bytes;
}

}  // namespace gufo::models::qwen3_omni::hip
