#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/text_encoder.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr int kLayers = 24;
constexpr int kHeads = 64;
constexpr int kHeadDim = 64;
constexpr int kWidth = 4096;
constexpr int kFeedForward = 10240;
constexpr int kVocab = 256384;
constexpr int kBuckets = 32;
constexpr int kMaxDistance = 128;
constexpr int kMaxTokens = static_cast<int>(TextEncoder::kMaxTokens);
constexpr float kEpsilon = 1e-6F;
constexpr std::size_t kStagingBytes = 32U << 20U;
constexpr int kNormThreads = 256;
constexpr int kAttentionThreads = 64;

__device__ __forceinline__ float FromBf16(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t ToBf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>(
      (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

__device__ __forceinline__ float Round(float value) {
  return FromBf16(ToBf16(value));
}

// Tree reduction over a block of `kThreads` threads.
template<int kThreads, bool kMaximum>
__device__ float BlockReduce(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] = kMaximum ? fmaxf(shared[tid], shared[tid + stride])
                             : shared[tid] + shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

// out = weight * bf16(x * rsqrt(mean(x^2) + eps)), one block per row.
__global__ void RmsNormKernel(const float* x, const std::uint16_t* weight,
                              std::uint16_t* out) {
  __shared__ float shared[kNormThreads];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * kWidth;
  float sum = 0;
  for (int c = threadIdx.x; c < kWidth; c += kNormThreads)
    sum += row[c] * row[c];
  sum = BlockReduce<kNormThreads, false>(sum, shared);
  const float scale = rsqrtf(sum / kWidth + kEpsilon);
  std::uint16_t* target = out + static_cast<std::size_t>(blockIdx.x) * kWidth;
  for (int c = threadIdx.x; c < kWidth; c += kNormThreads)
    target[c] = ToBf16(FromBf16(weight[c]) * Round(row[c] * scale));
}

// One block per (query, head). Like the reference's SDPA math path, scores,
// the BF16 relative position bias and softmax combine in FP32 and only the
// output rounds to BF16. Scores are unscaled.
__global__ void AttentionKernel(const std::uint16_t* qkv,
                                const std::uint16_t* bias_table,
                                const int* buckets, int tokens,
                                std::uint16_t* out) {
  __shared__ float query[kHeadDim];
  __shared__ float probability[kMaxTokens];
  __shared__ float shared[kAttentionThreads];
  constexpr std::size_t kStride = 3 * kWidth;
  const int i = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  query[tid] = FromBf16(qkv[i * kStride + head * kHeadDim + tid]);
  __syncthreads();
  float maximum = -INFINITY;
  for (int j = tid; j < tokens; j += kAttentionThreads) {
    const std::uint16_t* key = qkv + j * kStride + kWidth + head * kHeadDim;
    float score = 0;
#pragma unroll 16
    for (int d = 0; d < kHeadDim; ++d)
      score += query[d] * FromBf16(key[d]);
    const int bucket = buckets[j - i + kMaxTokens - 1];
    score += FromBf16(bias_table[bucket * kHeads + head]);
    probability[j] = score;
    maximum = fmaxf(maximum, score);
  }
  maximum = BlockReduce<kAttentionThreads, true>(maximum, shared);
  float sum = 0;
  for (int j = tid; j < tokens; j += kAttentionThreads) {
    const float value = expf(probability[j] - maximum);
    probability[j] = value;
    sum += value;
  }
  sum = BlockReduce<kAttentionThreads, false>(sum, shared);
  const float inverse = 1.0F / sum;
  for (int j = tid; j < tokens; j += kAttentionThreads)
    probability[j] *= inverse;
  __syncthreads();
  const std::uint16_t* value = qkv + 2 * kWidth + head * kHeadDim + tid;
  float output = 0;
  for (int j = 0; j < tokens; ++j)
    output += probability[j] * FromBf16(value[j * kStride]);
  out[static_cast<std::size_t>(i) * kWidth + head * kHeadDim + tid] =
      ToBf16(output);
}

// out = gelu_new(wi_0 x) * wi_1 x; `projected` holds both halves. gelu_new
// rounds to BF16 after every operation, as the reference's eager BF16 ops do.
__global__ void GatedGeluKernel(const std::uint16_t* projected, int tokens,
                                std::uint16_t* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(tokens) * kFeedForward)
    return;
  const std::size_t row = index / kFeedForward;
  const std::size_t column = index % kFeedForward;
  const std::uint16_t* source = projected + row * 2 * kFeedForward + column;
  const float gate = FromBf16(source[0]);
  const float linear = FromBf16(source[kFeedForward]);
  const float cube = Round(0.044715F * Round(gate * gate * gate));
  const float inner = Round(0.7978845608028654F * Round(gate + cube));
  const float gelu = Round(0.5F * gate) * Round(1.0F + Round(tanhf(inner)));
  out[index] = ToBf16(Round(gelu) * linear);
}

// The residual stream stays FP32: rounding it to BF16 after every add, as
// the reference does, costs more accuracy than any other step.
__global__ void AddKernel(float* x, const std::uint16_t* y, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    x[index] += FromBf16(y[index]);
}

__global__ void WidenKernel(const std::uint16_t* x, std::size_t count,
                            float* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    out[index] = FromBf16(x[index]);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("Wan umT5 ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
}

// T5's bidirectional bucket of relative position `distance` = key - query,
// in the reference's FP32 arithmetic.
int RelativeBucket(int distance) {
  constexpr int kHalf = kBuckets / 2;
  constexpr int kExact = kHalf / 2;
  const int bucket = distance > 0 ? kHalf : 0;
  const int magnitude = std::abs(distance);
  if (magnitude < kExact)
    return bucket + magnitude;
  float ratio = std::log(static_cast<float>(magnitude) / kExact) /
                static_cast<float>(std::log(double{kMaxDistance} / kExact));
  ratio *= static_cast<float>(kHalf - kExact);
  return bucket + std::min(kExact + static_cast<int>(ratio), kHalf - 1);
}

std::optional<json::Value> ReadJson(const std::filesystem::path& path,
                                    std::string* error) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto size = in.tellg();
  if (!in || size <= 0 || size > (16 << 20)) {
    SetError(error, "cannot read " + path.string());
    return std::nullopt;
  }
  std::string data(static_cast<std::size_t>(size), '\0');
  in.seekg(0);
  if (!in.read(data.data(), size)) {
    SetError(error, "cannot read " + path.string());
    return std::nullopt;
  }
  try {
    return json::parse(data);
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return std::nullopt;
  }
}

struct TensorLocation {
  std::size_t shard{0};
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::vector<std::size_t> shape;
};

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  bool Allocate(std::size_t bytes, std::string* error) {
    return Check(hipMalloc(&data_, bytes), "device allocation", error);
  }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }

private:
  void* data_{nullptr};
};

struct Layer {
  DeviceBuffer attention_norm;
  DeviceBuffer qkv;  // q, k, v rows stacked: [3 * kWidth, kWidth]
  DeviceBuffer output;
  DeviceBuffer bias;  // [kBuckets, kHeads]
  DeviceBuffer feed_forward_norm;
  DeviceBuffer up;  // wi_0, wi_1 rows stacked: [2 * kFeedForward, kWidth]
  DeviceBuffer down;
};

}  // namespace

struct TextEncoder::Impl {
  ~Impl() {
    for (auto file : shards)
      if (platform::FileValid(file))
        platform::CloseFile(file);
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
    for (void* buffer : staging)
      if (buffer != nullptr)
        (void)hipHostFree(buffer);
    for (hipEvent_t event : staged)
      if (event != nullptr)
        (void)hipEventDestroy(event);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  bool Index(const std::filesystem::path& dir, std::string* error);
  bool Upload(const std::string& name, std::initializer_list<std::size_t> shape,
              std::uint16_t* target, std::string* error);
  bool Gemm(const std::uint16_t* weight, const std::uint16_t* input,
            std::uint16_t* output, int tokens, int rows, int columns,
            std::string* error);

  std::vector<platform::PlatformFile> shards;
  std::map<std::string, TensorLocation, std::less<>> tensors;
  TensorLocation embedding;
  std::vector<Layer> layers;
  DeviceBuffer final_norm;
  DeviceBuffer buckets;
  DeviceBuffer hidden;  // FP32 residual stream
  DeviceBuffer normed;
  DeviceBuffer qkv;
  DeviceBuffer attention;
  DeviceBuffer projected;
  DeviceBuffer up;
  DeviceBuffer gated;
  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  void* staging[2]{};
  hipEvent_t staged[2]{};
  int next_slot{0};
  std::size_t device_bytes{0};
};

bool TextEncoder::Impl::Index(const std::filesystem::path& dir,
                              std::string* error) {
  const auto index = ReadJson(dir / "model.safetensors.index.json", error);
  if (!index)
    return false;
  const json::Value* map = index->find("weight_map");
  if (map == nullptr || !map->is_object()) {
    SetError(error, "missing umT5 weight map");
    return false;
  }
  std::map<std::string, std::size_t> shard_ids;
  std::vector<std::string> shard_names;
  for (const auto& [name, file] : map->members()) {
    const std::string& shard = file.str();
    if (shard.empty() ||
        std::filesystem::path(shard).filename().string() != shard) {
      SetError(error, "invalid umT5 shard name for " + name);
      return false;
    }
    if (shard_ids.emplace(shard, shard_names.size()).second)
      shard_names.push_back(shard);
  }
  for (std::size_t s = 0; s < shard_names.size(); ++s) {
    const auto path = dir / shard_names[s];
    std::string open_error;
    shards.push_back(platform::OpenReadFile(path, &open_error));
    if (!platform::FileValid(shards.back())) {
      SetError(error, "cannot open " + path.string() + ": " + open_error);
      return false;
    }
    const auto file_size = platform::FileSizeBytes(shards.back());
    std::uint64_t header_size = 0;
    if (!file_size || platform::ReadFileAt(shards.back(), &header_size, 8, 0) !=
                          8 ||
        header_size == 0 || header_size > (32U << 20U) ||
        header_size > *file_size - 8) {
      SetError(error, "invalid safetensors header in " + path.string());
      return false;
    }
    std::string text(static_cast<std::size_t>(header_size), '\0');
    if (platform::ReadFileAt(shards.back(), text.data(), text.size(), 8) !=
        text.size()) {
      SetError(error, "short read of " + path.string());
      return false;
    }
    json::Value header;
    try {
      header = json::parse(text);
    } catch (const std::exception& e) {
      SetError(error, path.string() + ": " + e.what());
      return false;
    }
    const std::uint64_t payload = 8 + header_size;
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (entry.member_str("dtype") != "BF16" || shape == nullptr ||
          !shape->is_array() || offsets == nullptr || !offsets->is_array() ||
          offsets->size() != 2) {
        SetError(error, "unsupported umT5 tensor " + name);
        return false;
      }
      TensorLocation location;
      location.shard = s;
      std::uint64_t bytes = 2;
      for (const auto& dim : shape->items()) {
        location.shape.push_back(dim.as_size());
        bytes *= location.shape.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end < begin || end - begin != bytes ||
          payload + end > *file_size) {
        SetError(error, "invalid umT5 tensor extent " + name);
        return false;
      }
      location.offset = payload + begin;
      location.bytes = bytes;
      tensors[name] = std::move(location);
    }
  }
  return true;
}

bool TextEncoder::Impl::Upload(const std::string& name,
                               std::initializer_list<std::size_t> shape,
                               std::uint16_t* target, std::string* error) {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    SetError(error, "missing umT5 tensor " + name);
    return false;
  }
  const TensorLocation& location = it->second;
  if (!std::equal(location.shape.begin(), location.shape.end(), shape.begin(),
                  shape.end())) {
    SetError(error, "unexpected shape of umT5 tensor " + name);
    return false;
  }
  // Two pinned buffers alternate: the disk fills one while the other copies.
  for (std::uint64_t copied = 0; copied < location.bytes;) {
    const std::size_t chunk = static_cast<std::size_t>(
        std::min<std::uint64_t>(kStagingBytes, location.bytes - copied));
    const int slot = next_slot;
    next_slot ^= 1;
    if (!Check(hipEventSynchronize(staged[slot]), "weight staging", error))
      return false;
    auto* buffer = static_cast<std::byte*>(staging[slot]);
    for (std::size_t read = 0; read < chunk;) {
      const std::size_t result =
          platform::ReadFileAt(shards[location.shard], buffer + read,
                               chunk - read, location.offset + copied + read);
      if (result == 0) {
        SetError(error, "short read of umT5 tensor " + name);
        return false;
      }
      read += result;
    }
    if (!Check(hipMemcpyAsync(reinterpret_cast<std::byte*>(target) + copied,
                              buffer, chunk, hipMemcpyHostToDevice, stream),
               "weight upload", error) ||
        !Check(hipEventRecord(staged[slot], stream), "weight upload", error))
      return false;
    copied += chunk;
  }
  return true;
}

bool TextEncoder::Impl::Gemm(const std::uint16_t* weight,
                             const std::uint16_t* input, std::uint16_t* output,
                             int tokens, int rows, int columns,
                             std::string* error) {
  // Row-major output[tokens, rows] = input[tokens, columns] * weight^T, with
  // weight row-major [rows, columns]. Pinned solution indices are absent from
  // the Windows rocBLAS database, so the standard selection is used.
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const rocblas_status status = rocblas_gemm_ex(
      blas, rocblas_operation_transpose, rocblas_operation_none, rows, tokens,
      columns, &alpha, weight, rocblas_datatype_bf16_r, columns, input,
      rocblas_datatype_bf16_r, columns, &beta, output, rocblas_datatype_bf16_r,
      rows, output, rocblas_datatype_bf16_r, rows, rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0, 0);
  if (status != rocblas_status_success) {
    SetError(error, "Wan umT5 GEMM failed: " +
                        std::to_string(static_cast<int>(status)));
    return false;
  }
  return true;
}

TextEncoder::TextEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
TextEncoder::~TextEncoder() = default;

std::size_t TextEncoder::device_bytes() const { return impl_->device_bytes; }

std::unique_ptr<TextEncoder> TextEncoder::Load(
    const std::filesystem::path& dir, std::string* error) {
  const auto config = ReadJson(dir / "config.json", error);
  if (!config)
    return nullptr;
  for (auto [key, value] : {std::pair{"d_model", kWidth},
                            {"d_ff", kFeedForward},
                            {"d_kv", kHeadDim},
                            {"num_heads", kHeads},
                            {"num_layers", kLayers},
                            {"vocab_size", kVocab},
                            {"relative_attention_num_buckets", kBuckets},
                            {"relative_attention_max_distance", kMaxDistance}})
    if (config->member_double(key, -1) != value) {
      SetError(error, std::string("unsupported umT5 configuration: ") + key);
      return nullptr;
    }
  if (config->member_str("model_type") != "umt5" ||
      config->member_str("feed_forward_proj") != "gated-gelu" ||
      config->member_str("dense_act_fn") != "gelu_new" ||
      config->member_double("layer_norm_epsilon") != 1e-6) {
    SetError(error, "unsupported umT5 configuration");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  if (!impl->Index(dir, error))
    return nullptr;
  const auto embedding = impl->tensors.find("shared.weight");
  if (embedding == impl->tensors.end() ||
      embedding->second.shape !=
          std::vector<std::size_t>{static_cast<std::size_t>(kVocab), kWidth}) {
    SetError(error, "missing or malformed umT5 shared.weight");
    return nullptr;
  }
  impl->embedding = embedding->second;
  if (!Check(hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking),
             "stream", error))
    return nullptr;
  if (rocblas_create_handle(&impl->blas) != rocblas_status_success ||
      rocblas_set_stream(impl->blas, impl->stream) != rocblas_status_success) {
    impl->blas = nullptr;
    SetError(error, "cannot create rocBLAS handle");
    return nullptr;
  }
  for (int slot = 0; slot < 2; ++slot)
    if (!Check(hipHostMalloc(&impl->staging[slot], kStagingBytes,
                             hipHostMallocDefault),
               "pinned staging", error) ||
        !Check(hipEventCreateWithFlags(&impl->staged[slot],
                                       hipEventDisableTiming),
               "staging event", error) ||
        !Check(hipEventRecord(impl->staged[slot], impl->stream),
               "staging event", error))
      return nullptr;
  constexpr std::size_t W = kWidth;
  constexpr std::size_t F = kFeedForward;
  constexpr std::size_t kBf16 = sizeof(std::uint16_t);
  auto allocate = [&](DeviceBuffer& buffer, std::size_t elements) {
    impl->device_bytes += elements * kBf16;
    return buffer.Allocate(elements * kBf16, error);
  };
  impl->layers = std::vector<Layer>(kLayers);
  for (int index = 0; index < kLayers; ++index) {
    Layer& layer = impl->layers[index];
    const std::string prefix = "encoder.block." + std::to_string(index) + ".";
    const std::string attention = prefix + "layer.0.SelfAttention.";
    const std::string feed_forward = prefix + "layer.1.DenseReluDense.";
    if (!allocate(layer.attention_norm, W) || !allocate(layer.qkv, 3 * W * W) ||
        !allocate(layer.output, W * W) ||
        !allocate(layer.bias, kBuckets * kHeads) ||
        !allocate(layer.feed_forward_norm, W) ||
        !allocate(layer.up, 2 * F * W) || !allocate(layer.down, W * F) ||
        !impl->Upload(prefix + "layer.0.layer_norm.weight", {W},
                      layer.attention_norm.get(), error) ||
        !impl->Upload(attention + "q.weight", {W, W}, layer.qkv.get(), error) ||
        !impl->Upload(attention + "k.weight", {W, W}, layer.qkv.get() + W * W,
                      error) ||
        !impl->Upload(attention + "v.weight", {W, W},
                      layer.qkv.get() + 2 * W * W, error) ||
        !impl->Upload(attention + "o.weight", {W, W}, layer.output.get(),
                      error) ||
        !impl->Upload(attention + "relative_attention_bias.weight",
                      {kBuckets, kHeads}, layer.bias.get(), error) ||
        !impl->Upload(prefix + "layer.1.layer_norm.weight", {W},
                      layer.feed_forward_norm.get(), error) ||
        !impl->Upload(feed_forward + "wi_0.weight", {F, W}, layer.up.get(),
                      error) ||
        !impl->Upload(feed_forward + "wi_1.weight", {F, W},
                      layer.up.get() + F * W, error) ||
        !impl->Upload(feed_forward + "wo.weight", {W, F}, layer.down.get(),
                      error))
      return nullptr;
  }
  if (!allocate(impl->final_norm, W) ||
      !impl->Upload("encoder.final_layer_norm.weight", {W},
                    impl->final_norm.get(), error))
    return nullptr;
  const std::size_t T = kMaxTokens;
  if (!allocate(impl->hidden, 2 * T * W) || !allocate(impl->normed, T * W) ||
      !allocate(impl->qkv, T * 3 * W) || !allocate(impl->attention, T * W) ||
      !allocate(impl->projected, T * W) || !allocate(impl->up, T * 2 * F) ||
      !allocate(impl->gated, T * F) ||
      !allocate(impl->buckets, 2 * T * sizeof(int) / kBf16))
    return nullptr;
  // Bucket of key - query + kMaxTokens - 1.
  std::vector<int> buckets(2 * T - 1);
  for (int k = 0; k < static_cast<int>(buckets.size()); ++k)
    buckets[k] = RelativeBucket(k - (kMaxTokens - 1));
  if (!Check(hipMemcpyAsync(impl->buckets.get(), buckets.data(),
                            buckets.size() * sizeof(int),
                            hipMemcpyHostToDevice, impl->stream),
             "bucket upload", error) ||
      !Check(hipStreamSynchronize(impl->stream), "weight upload", error))
    return nullptr;
  return std::unique_ptr<TextEncoder>(new TextEncoder(std::move(impl)));
}

bool TextEncoder::EncodeToDevice(std::span<const std::int32_t> ids,
                                 void* output, std::string* error) {
  Impl& m = *impl_;
  const int tokens = static_cast<int>(ids.size());
  if (ids.empty() || ids.size() > kMaxTokens) {
    SetError(error, "Wan umT5 takes 1 to 512 tokens");
    return false;
  }
  constexpr std::size_t kRowBytes = kWidth * sizeof(std::uint16_t);
  // The embedding rows come straight from the shard into pinned memory.
  if (!Check(hipEventSynchronize(m.staged[0]), "embedding", error))
    return false;
  auto* rows = static_cast<std::byte*>(m.staging[0]);
  for (int t = 0; t < tokens; ++t) {
    if (ids[t] < 0 || ids[t] >= kVocab) {
      SetError(error, "Wan umT5 token id out of range");
      return false;
    }
    const std::uint64_t offset = m.embedding.offset + ids[t] * kRowBytes;
    for (std::size_t read = 0; read < kRowBytes;) {
      const std::size_t result =
          platform::ReadFileAt(m.shards[m.embedding.shard],
                               rows + t * kRowBytes + read,
                               kRowBytes - read, offset + read);
      if (result == 0) {
        SetError(error, "short read of umT5 embedding");
        return false;
      }
      read += result;
    }
  }
  if (!Check(hipMemcpyAsync(m.normed.get(), rows, tokens * kRowBytes,
                            hipMemcpyHostToDevice, m.stream),
             "embedding upload", error) ||
      !Check(hipEventRecord(m.staged[0], m.stream), "embedding", error))
    return false;
  const std::size_t count = static_cast<std::size_t>(tokens) * kWidth;
  const unsigned add_blocks = static_cast<unsigned>((count + 255) / 256);
  WidenKernel<<<add_blocks, 256, 0, m.stream>>>(m.normed.get(), count,
                                                m.hidden.f32());
  const unsigned gelu_blocks = static_cast<unsigned>(
      (static_cast<std::size_t>(tokens) * kFeedForward + 255) / 256);
  for (const Layer& layer : m.layers) {
    RmsNormKernel<<<tokens, kNormThreads, 0, m.stream>>>(
        m.hidden.f32(), layer.attention_norm.get(), m.normed.get());
    if (!m.Gemm(layer.qkv.get(), m.normed.get(), m.qkv.get(), tokens,
                3 * kWidth, kWidth, error))
      return false;
    AttentionKernel<<<dim3(tokens, kHeads), kAttentionThreads, 0, m.stream>>>(
        m.qkv.get(), layer.bias.get(),
        reinterpret_cast<const int*>(m.buckets.get()), tokens,
        m.attention.get());
    if (!m.Gemm(layer.output.get(), m.attention.get(), m.projected.get(),
                tokens, kWidth, kWidth, error))
      return false;
    AddKernel<<<add_blocks, 256, 0, m.stream>>>(m.hidden.f32(),
                                                 m.projected.get(), count);
    RmsNormKernel<<<tokens, kNormThreads, 0, m.stream>>>(
        m.hidden.f32(), layer.feed_forward_norm.get(), m.normed.get());
    if (!m.Gemm(layer.up.get(), m.normed.get(), m.up.get(), tokens,
                2 * kFeedForward, kWidth, error))
      return false;
    GatedGeluKernel<<<gelu_blocks, 256, 0, m.stream>>>(m.up.get(), tokens,
                                                       m.gated.get());
    if (!m.Gemm(layer.down.get(), m.gated.get(), m.projected.get(), tokens,
                kWidth, kFeedForward, error))
      return false;
    AddKernel<<<add_blocks, 256, 0, m.stream>>>(m.hidden.f32(),
                                                 m.projected.get(), count);
    if (!Check(hipGetLastError(), "layer launch", error))
      return false;
  }
  RmsNormKernel<<<tokens, kNormThreads, 0, m.stream>>>(
      m.hidden.f32(), m.final_norm.get(), static_cast<std::uint16_t*>(output));
  return Check(hipGetLastError(), "final norm", error) &&
         Check(hipStreamSynchronize(m.stream), "encode", error);
}

bool TextEncoder::Encode(std::span<const std::int32_t> ids,
                         std::vector<float>* states, std::string* error) {
  if (!EncodeToDevice(ids, impl_->normed.get(), error))
    return false;
  const std::size_t count = ids.size() * kWidth;
  std::vector<std::uint16_t> bf16(count);
  if (!Check(hipMemcpy(bf16.data(), impl_->normed.get(),
                       count * sizeof(std::uint16_t), hipMemcpyDeviceToHost),
             "readback", error))
    return false;
  states->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(bf16[i]) << 16U;
    std::memcpy(&(*states)[i], &bits, sizeof(float));
  }
  return true;
}

}  // namespace gufo::models::wan22
