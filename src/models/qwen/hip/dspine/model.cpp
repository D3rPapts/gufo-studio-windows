#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gufo/ext/experimental/ggml_dequant.hpp>
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/models/qwen/hip/dspine.hpp"
#include "src/models/qwen/hip/dspine/kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"

namespace gufo::hip {
namespace {

/// Table rows per GPU pass (head / codebook chunk).
constexpr std::size_t kTableChunkRows = 4096;

std::uint16_t ToBf16Bits(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float FromBf16Bits(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

const core::GgufTensorInfo& Tensor(const core::GgufReader& reader,
                                   std::string_view name) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr || info->data == nullptr)
    throw std::runtime_error("dspine: missing tensor " + std::string(name));
  return *info;
}

/// Host float copy of `count` values starting at element `first`.
std::vector<float> HostFloats(const core::GgufTensorInfo& t, std::size_t first,
                              std::size_t count) {
  std::vector<float> out(count);
  switch (t.type) {
    case core::GgmlType::kF32:
      std::memcpy(out.data(), static_cast<const float*>(t.data) + first,
                  count * sizeof(float));
      break;
    case core::GgmlType::kBF16: {
      const auto* s = static_cast<const std::uint16_t*>(t.data) + first;
      for (std::size_t i = 0; i < count; ++i) out[i] = FromBf16Bits(s[i]);
      break;
    }
    default:
      throw std::runtime_error("dspine: vector " + std::string(t.name) +
                               " must be F32 or BF16");
  }
  return out;
}

void ToDevice(void* destination, const void* source, std::size_t bytes) {
  HIP_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice));
}

}  // namespace

QwenDSpineGpuModel::~QwenDSpineGpuModel() {
  for (void* pointer : allocations_)
    if (pointer != nullptr) (void)hipFree(pointer);
}

void* QwenDSpineGpuModel::Allocate(std::size_t bytes) {
  void* pointer = detail::AllocateDevice(bytes);
  allocations_.push_back(pointer);
  bytes_ += bytes;
  return pointer;
}

models::QwenTensorRef QwenDSpineGpuModel::Matrix(const core::GgufReader& reader,
                                                 std::string_view name) {
  const auto& t = Tensor(reader, name);
  const std::size_t n = t.ElementCount();
  switch (t.type) {
    case core::GgmlType::kBF16:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ6_K: {
      // Packed storage as in the file: the shared GEMMs read these directly.
      const std::size_t bytes = quant::EncodedSizeBytes(t.type, n);
      void* device = Allocate(bytes);
      ToDevice(device, t.data, bytes);
      return {.data = device, .type = t.type, .num_elements = n};
    }
    case core::GgmlType::kF32: {
      std::vector<std::uint16_t> host(n);
      const auto* s = static_cast<const float*>(t.data);
      for (std::size_t i = 0; i < n; ++i) host[i] = ToBf16Bits(s[i]);
      void* device = Allocate(n * sizeof(std::uint16_t));
      ToDevice(device, host.data(), n * sizeof(std::uint16_t));
      return {.data = device, .type = core::GgmlType::kBF16, .num_elements = n};
    }
    default:
      throw std::runtime_error("dspine: unsupported matrix type for " +
                               std::string(name));
  }
}

models::QwenTensorRef QwenDSpineGpuModel::Vector(const core::GgufReader& reader,
                                                 std::string_view name) {
  const auto& t = Tensor(reader, name);
  const auto host = HostFloats(t, 0, t.ElementCount());
  void* device = Allocate(host.size() * sizeof(float));
  ToDevice(device, host.data(), host.size() * sizeof(float));
  return {.data = device, .type = core::GgmlType::kF32, .num_elements = host.size()};
}

void QwenDSpineGpuModel::Upload(const core::GgufReader& reader) {
  const auto& c = config_;
  const std::size_t H = c.hidden, C = c.code_size, R = c.rank;
  fc_ = Matrix(reader, "fc.weight");
  hidden_norm_ = Vector(reader, "enc.output_norm.weight");
  output_norm_ = Vector(reader, "output_norm.weight");

  // Bridge: BF16 by construction of the converter (exact rounding matters).
  const auto& early = Tensor(reader, "dspine.bridge.early_read");
  const auto& read = Tensor(reader, "dspine.bridge.read");
  const auto& write = Tensor(reader, "dspine.bridge.feature_write");
  for (const auto* t : {&early, &read, &write})
    if (t->type != core::GgmlType::kBF16)
      throw std::runtime_error("dspine: bridge tensor " + std::string(t->name) +
                               " must be BF16 (reconvert)");
  if (early.ElementCount() != (c.layers - 1) * C * H ||
      read.ElementCount() != C * H || write.ElementCount() != c.layers * H * R)
    throw std::runtime_error("dspine: bridge shapes do not match the config");
  const auto early_ref = Matrix(reader, "dspine.bridge.early_read");
  const auto read_ref = Matrix(reader, "dspine.bridge.read");
  const auto write_ref = Matrix(reader, "dspine.bridge.feature_write");

  layers_.resize(c.layers);
  for (std::uint32_t L = 0; L < c.layers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    auto& layer = layers_[L];
    layer.attn_norm = Vector(reader, p + "attn_norm.weight");
    layer.ffn_norm = Vector(reader, p + "ffn_norm.weight");
    layer.q_norm = Vector(reader, p + "attn_q_norm.weight");
    layer.k_norm = Vector(reader, p + "attn_k_norm.weight");
    layer.attn_q = Matrix(reader, p + "attn_q.weight");
    layer.attn_k = Matrix(reader, p + "attn_k.weight");
    layer.attn_v = Matrix(reader, p + "attn_v.weight");
    layer.attn_output = Matrix(reader, p + "attn_output.weight");
    layer.ffn_gate = Matrix(reader, p + "ffn_gate.weight");
    layer.ffn_up = Matrix(reader, p + "ffn_up.weight");
    layer.ffn_down = Matrix(reader, p + "ffn_down.weight");
    const bool last = L + 1 == c.layers;
    layer.bridge_read = {
        .data = last ? read_ref.data
                     : static_cast<const std::uint16_t*>(early_ref.data) + L * C * H,
        .type = core::GgmlType::kBF16,
        .num_elements = C * H};
    layer.feature_write = {
        .data = static_cast<const std::uint16_t*>(write_ref.data) + L * H * R,
        .type = core::GgmlType::kBF16,
        .num_elements = H * R};
  }

  // [gate; predecessor_gate; prev] -> one [3R, C] BF16 matrix.
  std::vector<std::uint16_t> packed(3 * R * C);
  const char* names[3] = {"dspine.bridge.gate.weight",
                          "dspine.bridge.predecessor_gate.weight",
                          "dspine.bridge.prev.weight"};
  for (std::size_t part = 0; part < 3; ++part) {
    const auto& t = Tensor(reader, names[part]);
    if (t.ElementCount() != R * C)
      throw std::runtime_error("dspine: bridge gate shapes do not match");
    const auto values = HostFloats(t, 0, R * C);
    for (std::size_t i = 0; i < R * C; ++i)
      packed[part * R * C + i] = ToBf16Bits(values[i]);
  }
  void* device = Allocate(packed.size() * sizeof(std::uint16_t));
  ToDevice(device, packed.data(), packed.size() * sizeof(std::uint16_t));
  bridge_packed_ = {.data = device, .type = core::GgmlType::kBF16,
                    .num_elements = packed.size()};
  bridge_bias_ = Vector(reader, "dspine.bridge.gate.bias");
}

void QwenDSpineGpuModel::BuildTables(const core::GgufReader& reader) {
  const auto start = std::chrono::steady_clock::now();
  const auto& c = config_;
  const std::size_t H = c.hidden, C = c.code_size, R = c.rank, V = c.vocab;
  tg_ = static_cast<std::uint16_t*>(Allocate(V * R * sizeof(std::uint16_t)));
  tsp_ = static_cast<std::uint16_t*>(Allocate(V * 3 * R * sizeof(std::uint16_t)));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  hipblasHandle_t handle = nullptr;
  if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
    (void)hipStreamDestroy(stream);
    throw std::runtime_error("dspine: hipblasCreate failed");
  }
  (void)hipblasSetStream(handle, stream);
  std::vector<void*> scratch;
  const auto scratch_alloc = [&](std::size_t bytes) {
    void* p = detail::AllocateDevice(bytes);
    scratch.push_back(p);
    return p;
  };
  try {
    const std::size_t width = std::max(H, C);
    auto* rows_in = static_cast<std::uint16_t*>(
        scratch_alloc(kTableChunkRows * width * sizeof(std::uint16_t)));
    auto* rows_scaled = static_cast<std::uint16_t*>(
        scratch_alloc(kTableChunkRows * width * sizeof(std::uint16_t)));
    auto* product = static_cast<float*>(
        scratch_alloc(kTableChunkRows * 3 * R * sizeof(float)));

    // TG: W_last^T [R, H] so that Y[B, R] = X[B, H] @ W_last.
    const auto& write = Tensor(reader, "dspine.bridge.feature_write");
    const auto* w_last =
        static_cast<const std::uint16_t*>(write.data) + (c.layers - 1) * H * R;
    std::vector<std::uint16_t> transposed(R * H);
    for (std::size_t i = 0; i < H; ++i)
      for (std::size_t o = 0; o < R; ++o) transposed[o * H + i] = w_last[i * R + o];
    void* d_w_last_t = scratch_alloc(transposed.size() * sizeof(std::uint16_t));
    ToDevice(d_w_last_t, transposed.data(), transposed.size() * sizeof(std::uint16_t));

    const auto& head = target_model_->GetWeights().output;
    const std::size_t head_row_bytes = quant::EncodedSizeBytes(head.type, H);
    if (head.data == nullptr || head_row_bytes == 0)
      throw std::runtime_error("dspine: target head is not row addressable");
    for (std::size_t first = 0; first < V; first += kTableChunkRows) {
      const std::size_t n = std::min(kTableChunkRows, V - first);
      const auto* source = static_cast<const std::uint8_t*>(head.data) + first * head_row_bytes;
      const std::uint16_t* bf_rows = rows_in;
      if (head.type == core::GgmlType::kBF16) {
        bf_rows = reinterpret_cast<const std::uint16_t*>(source);
      } else {
        LaunchDequantizeToBf16(head.type, source,
                               reinterpret_cast<hip_bfloat16*>(rows_in), n * H, stream);
      }
      dspine::LaunchScaleBf16Rows(bf_rows, OutputNorm(), rows_scaled, n,
                                  static_cast<std::uint32_t>(H), stream);
      LaunchHipblasGEMMBF16(handle, d_w_last_t, rows_scaled, product, n, R, H, stream);
      dspine::LaunchFloatToBf16Bits(product, tg_ + first * R, n * R, stream);
    }

    // TSP: codes of every token through [gate; pred; prev].
    const auto& codebook = Tensor(reader, "dspine.bridge.codebook");
    if (codebook.type != core::GgmlType::kBF16 || codebook.ElementCount() != V * C)
      throw std::runtime_error("dspine: codebook must be BF16 [V, C]");
    const float scale = std::sqrt(static_cast<float>(C));
    for (std::size_t first = 0; first < V; first += kTableChunkRows) {
      const std::size_t n = std::min(kTableChunkRows, V - first);
      HIP_CHECK(hipMemcpyAsync(rows_in,
                               static_cast<const std::uint16_t*>(codebook.data) + first * C,
                               n * C * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                               stream));
      dspine::LaunchScaleBf16(rows_in, scale, rows_scaled, n * C, stream);
      LaunchHipblasGEMMBF16(handle, bridge_packed_.data, rows_scaled, product, n,
                            3 * R, C, stream);
      dspine::LaunchFloatToBf16Bits(product, tsp_ + first * 3 * R, n * 3 * R, stream);
      // The host chunk is reused next iteration only after the copy finished.
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  } catch (...) {
    for (void* p : scratch) (void)hipFree(p);
    (void)hipblasDestroy(handle);
    (void)hipStreamDestroy(stream);
    throw;
  }
  for (void* p : scratch) (void)hipFree(p);
  (void)hipblasDestroy(handle);
  (void)hipStreamDestroy(stream);
  table_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::shared_ptr<const QwenDSpineGpuModel> QwenDSpineGpuModel::Create(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const QwenGpuModel> target_model, std::string* error_msg) {
  if (reader == nullptr || target_model == nullptr) {
    if (error_msg) *error_msg = "DSpine reader and target GPU model are required";
    return nullptr;
  }
  auto config = models::qwen::ReadDSpineConfig(*reader, error_msg);
  if (!config) return nullptr;
  const auto& target = target_model->GetConfig();
  if (target.hidden_size != config->hidden || target.vocab_size != config->vocab) {
    if (error_msg) *error_msg = "DSpine hidden/vocabulary size does not match the target";
    return nullptr;
  }
  if (config->target_layers.back() >= target.num_layers) {
    if (error_msg) *error_msg = "DSpine target layer taps exceed the target layer count";
    return nullptr;
  }
  std::shared_ptr<QwenDSpineGpuModel> model(new QwenDSpineGpuModel());
  model->config_ = std::move(*config);
  model->target_model_ = std::move(target_model);
  try {
    model->Upload(*reader);
    model->BuildTables(*reader);
  } catch (const std::exception& e) {
    if (error_msg) *error_msg = e.what();
    return nullptr;  // destructor frees what was allocated
  }
  model->reader_ = std::move(reader);
  return model;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
