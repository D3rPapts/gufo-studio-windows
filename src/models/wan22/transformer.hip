#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/attention.hip.hpp"
#include "src/models/wan22/gemm.hpp"
#include "src/models/wan22/transformer.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr int kLayers = 30;
constexpr int kHeads = 24;
constexpr int kHeadDim = 128;
constexpr int kWidth = Transformer::kModelWidth;
constexpr int kFeedForward = 14336;
// Tokens per feed-forward pass: bounds the [tokens, 14336] hidden buffer
// (GEMM launches are split into 4096-row chunks anyway).
constexpr int kFeedForwardChunk = 8192;
constexpr int kChannels = Transformer::kChannels;
constexpr int kPatchValues = kChannels * 4;  // patch 1 x 2 x 2
constexpr int kFrequencies = 256;
constexpr int kModulation = 6 * kWidth;
constexpr int kRopePairs = kHeadDim / 2;
constexpr int kRopeTimePairs = 22;  // 44 dims; height and width get 42 each
constexpr int kRopeSpacePairs = 21;
constexpr float kEpsilon = 1e-6F;
constexpr std::size_t kStagingBytes = 32U << 20U;
constexpr std::size_t kSector = 4096;
constexpr std::size_t kStagingSlack = 2U * kSector;
constexpr int kRowThreads = 256;
constexpr int kRowValues = kWidth / kRowThreads;  // 12

static_assert(kWidth == kHeads * kHeadDim);
static_assert(kWidth % kRowThreads == 0);

__device__ __forceinline__ float FromBf16(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t ToBf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

__device__ __forceinline__ float Round(float value) {
  return FromBf16(ToBf16(value));
}

__device__ __forceinline__ float GeluTanh(float x) {
  const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0F + expf(-x));
}

__device__ float BlockSum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kRowThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__global__ void ConvertKernel(const float* input, std::size_t count,
                              std::uint16_t* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = ToBf16(input[index]);
}

// patches[t, c * 4 + kh * 2 + kw] = latent[c, f, 2h + kh, 2w + kw], BF16.
__global__ void PatchKernel(const float* latent, int frames, int height,
                            int width, std::uint16_t* patches) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ph = height / 2, pw = width / 2;
  const std::size_t tokens = static_cast<std::size_t>(frames) * ph * pw;
  if (index >= tokens * kPatchValues)
    return;
  const std::size_t token = index / kPatchValues;
  const int column = static_cast<int>(index % kPatchValues);
  const int c = column / 4, kh = (column / 2) % 2, kw = column % 2;
  const int w = static_cast<int>(token % pw);
  const int h = static_cast<int>((token / pw) % ph);
  const int f = static_cast<int>(token / (static_cast<std::size_t>(pw) * ph));
  patches[index] =
      ToBf16(latent[((static_cast<std::size_t>(c) * frames + f) * height +
                     2 * h + kh) *
                        width +
                    2 * w + kw]);
}

// out = bf16(gelu_tanh(x)) of BF16 linear outputs.
__global__ void GeluKernel(const std::uint16_t* x, std::size_t count,
                           std::uint16_t* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    out[index] = ToBf16(GeluTanh(FromBf16(x[index])));
}

// Sinusoidal timestep embedding (flip_sin_to_cos, shift 0), FP32 like
// Diffusers' get_timestep_embedding.
__global__ void SinusoidKernel(const float* timesteps, float* out) {
  const int u = blockIdx.x;
  const int i = threadIdx.x;  // 0..127
  constexpr int kHalf = kFrequencies / 2;
  const float exponent =
      static_cast<float>(i) * -9.210340371976184F / static_cast<float>(kHalf);
  const float angle = timesteps[u] * expf(exponent);
  out[u * kFrequencies + i] = cosf(angle);
  out[u * kFrequencies + kHalf + i] = sinf(angle);
}

enum class Activation { kNone, kSilu, kSiluBf16 };

// out[u, r] = W[r, :] . act(in[u, :]) + b[r], one block per (r, u). With
// kRoundOut the result is rounded to BF16 (kept as FP32).
template<typename Weight, Activation kInput, bool kRoundOut>
__global__ void MatVecKernel(const Weight* weight, const Weight* bias,
                             const float* in, int columns, int rows,
                             float* out) {
  __shared__ float shared[kRowThreads];
  const int r = blockIdx.x;
  const int u = blockIdx.y;
  const float* x = in + static_cast<std::size_t>(u) * columns;
  const Weight* w = weight + static_cast<std::size_t>(r) * columns;
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kRowThreads) {
    float v = x[c];
    if constexpr (kInput == Activation::kSilu)
      v = Silu(v);
    else if constexpr (kInput == Activation::kSiluBf16)
      v = Round(Silu(v));
    float wv;
    if constexpr (std::is_same_v<Weight, float>)
      wv = w[c];
    else
      wv = FromBf16(w[c]);
    sum += wv * v;
  }
  sum = BlockSum(sum, shared);
  if (threadIdx.x == 0) {
    float b;
    if constexpr (std::is_same_v<Weight, float>)
      b = bias[r];
    else
      b = FromBf16(bias[r]);
    const float y = sum + b;
    out[static_cast<std::size_t>(u) * rows + r] = kRoundOut ? Round(y) : y;
  }
}

// FP32 layer norm of the residual (two-pass), then either
//   out = bf16(y * (1 + scale) + shift), scale/shift = table + proj[mod[t]]
// or, with `weight`, out = bf16(y * weight + bias).
struct Modulation {
  const float* shift_table;
  const float* scale_table;
  const float* shift_proj;  // per unique timestep, stride proj_stride
  const float* scale_proj;
  int proj_stride;
  const int* index;  // unique timestep of each token
};

__global__ void LayerNormKernel(const float* hidden, Modulation modulation,
                                const float* weight, const float* bias,
                                std::uint16_t* out) {
  __shared__ float shared[kRowThreads];
  const std::size_t row = blockIdx.x;
  const float* x = hidden + row * kWidth;
  float values[kRowValues];
  float sum = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] = x[threadIdx.x + i * kRowThreads];
    sum += values[i];
  }
  const float mean = BlockSum(sum, shared) / kWidth;
  float squares = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] -= mean;
    squares += values[i] * values[i];
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / kWidth + kEpsilon);
  std::uint16_t* target = out + row * kWidth;
  if (weight != nullptr) {
#pragma unroll
    for (int i = 0; i < kRowValues; ++i) {
      const int c = threadIdx.x + i * kRowThreads;
      target[c] = ToBf16(values[i] * rstd * weight[c] + bias[c]);
    }
    return;
  }
  const std::size_t u =
      static_cast<std::size_t>(modulation.index[row]) * modulation.proj_stride;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    const int c = threadIdx.x + i * kRowThreads;
    const float scale =
        modulation.scale_table[c] + modulation.scale_proj[u + c];
    const float shift =
        modulation.shift_table[c] + modulation.shift_proj[u + c];
    target[c] = ToBf16(values[i] * rstd * (1.0F + scale) + shift);
  }
}

// One projection (q, k or v) of one row of BF16 linear outputs, optionally RMS
// normalized across all heads (bf16(x * rstd * weight)) and rotated, written
// head-major [heads][rows][128].
__global__ void HeadsKernel(const std::uint16_t* linear, int stride, int rows,
                            const std::uint16_t* norm_weight,
                            const float2* rope, std::uint16_t* out) {
  __shared__ float shared[kRowThreads];
  const std::size_t row = blockIdx.x;
  const std::uint16_t* x = linear + row * stride;
  constexpr int kPairs = kRowValues / 2;
  float values[kRowValues];
  float squares = 0;
#pragma unroll
  for (int p = 0; p < kPairs; ++p) {
    const int c = 2 * (threadIdx.x + p * kRowThreads);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float v = FromBf16(x[c + j]);
      values[2 * p + j] = v;
      squares += v * v;
    }
  }
  if (norm_weight != nullptr) {
    const float rstd = rsqrtf(BlockSum(squares, shared) / kWidth + kEpsilon);
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
      const int c = 2 * (threadIdx.x + p * kRowThreads);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        values[2 * p + j] =
            Round(values[2 * p + j] * rstd * FromBf16(norm_weight[c + j]));
    }
  }
#pragma unroll
  for (int p = 0; p < kPairs; ++p) {
    const int c = 2 * (threadIdx.x + p * kRowThreads);
    const int head = c / kHeadDim;
    const int d = c % kHeadDim;
    float a = values[2 * p], b = values[2 * p + 1];
    if (rope != nullptr) {
      const float2 cs = rope[row * kRopePairs + d / 2];
      const float ra = a * cs.x - b * cs.y;
      const float rb = a * cs.y + b * cs.x;
      a = ra;
      b = rb;
    }
    std::uint16_t* target =
        out + (static_cast<std::size_t>(head) * rows + row) * kHeadDim + d;
    target[0] = ToBf16(a);
    target[1] = ToBf16(b);
  }
}

// hidden += y * gate, gate = table + proj[mod[t]] (or 1), y BF16.
__global__ void ResidualKernel(const std::uint16_t* linear,
                               const float* gate_table, const float* gate_proj,
                               const int* index, std::size_t count,
                               float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % kWidth);
  const float y = FromBf16(linear[i]);
  if (gate_table == nullptr) {
    hidden[i] += y;
    return;
  }
  const std::size_t row = i / kWidth;
  const float gate =
      gate_table[c] +
      gate_proj[static_cast<std::size_t>(index[row]) * kModulation + c];
  hidden[i] += y * gate;
}

// noise[c, f, 2h + kh, 2w + kw] = linear[t, (kh * 2 + kw) * 48 + c].
__global__ void UnpatchKernel(const std::uint16_t* linear, int frames,
                              int height, int width, float* noise) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ph = height / 2, pw = width / 2;
  const std::size_t tokens = static_cast<std::size_t>(frames) * ph * pw;
  if (index >= tokens * kPatchValues)
    return;
  const std::size_t token = index / kPatchValues;
  const int column = static_cast<int>(index % kPatchValues);
  const int kh = column / (2 * kChannels);
  const int kw = (column / kChannels) % 2;
  const int c = column % kChannels;
  const int w = static_cast<int>(token % pw);
  const int h = static_cast<int>((token / pw) % ph);
  const int f = static_cast<int>(token / (static_cast<std::size_t>(pw) * ph));
  noise[((static_cast<std::size_t>(c) * frames + f) * height + 2 * h + kh) *
            width +
        2 * w + kw] = FromBf16(linear[index]);
}

__global__ void WidenKernel(const std::uint16_t* x, std::size_t count,
                            float* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    out[index] = FromBf16(x[index]);
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + 255) / 256);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("Wan transformer ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
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
  bool f32{false};
  std::vector<std::size_t> shape;
};

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { Release(); }
  bool Allocate(std::size_t bytes, std::string* error) {
    Release();
    return Check(hipMalloc(&data_, std::max<std::size_t>(bytes, 256)),
                 "device allocation", error);
  }
  void Release() {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
  }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }
  int* i32() const { return static_cast<int*>(data_); }

private:
  void* data_{nullptr};
};

struct Block {
  DeviceBuffer qkv, qkv_bias;  // to_q | to_k | to_v rows: [3 * W, W]
  DeviceBuffer norm_q, norm_k;
  DeviceBuffer out, out_bias;
  DeviceBuffer cross_q, cross_q_bias;
  DeviceBuffer cross_kv, cross_kv_bias;  // to_k | to_v rows: [2 * W, W]
  DeviceBuffer cross_norm_q, cross_norm_k;
  DeviceBuffer cross_out, cross_out_bias;
  DeviceBuffer norm2_weight, norm2_bias;  // FP32
  DeviceBuffer up, up_bias;
  DeviceBuffer down, down_bias;
  DeviceBuffer modulation;  // FP32 [6, W]
};

// Reads `bytes` at `offset` into a staging buffer of kStagingBytes +
// kStagingSlack as its sector-aligned superset, so ReadFileParallel can use
// unbuffered parallel reads. Returns the first requested byte, or nullptr.
const std::byte* ReadStaged(platform::PlatformFile file, std::byte* buffer,
                            std::size_t bytes, std::uint64_t offset) {
  const std::uint64_t aligned = offset / kSector * kSector;
  const auto lead = static_cast<std::size_t>(offset - aligned);
  const std::size_t span = (lead + bytes + kSector - 1U) / kSector * kSector;
  return platform::ReadFileParallel(file, buffer, span, aligned) >= lead + bytes
             ? buffer + lead
             : nullptr;
}

}  // namespace

struct Transformer::Impl {
  ~Impl() {
    for (auto file : shards)
      if (platform::FileValid(file))
        platform::CloseFile(file);
    for (void* buffer : staging)
      if (buffer != nullptr)
        (void)hipHostFree(buffer);
    for (hipEvent_t event : staged)
      if (event != nullptr)
        (void)hipEventDestroy(event);
    for (hipEvent_t event : timing)
      if (event != nullptr)
        (void)hipEventDestroy(event);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  bool Index(const std::filesystem::path& dir, std::string* error);
  // Uploads a tensor as BF16 (bf16 = true) or FP32 to `target`.
  bool Upload(const TensorLocation& location, const std::string& name,
              void* target, bool bf16, std::string* error);
  bool LoadWeight(
      DeviceBuffer& buffer,
      std::initializer_list<std::pair<std::string, std::size_t>> parts,
      std::size_t columns, bool bf16, std::string* error);
  // Row-major out[tokens, rows] = bf16(in[tokens, columns] * weight^T + b).
  bool Gemm(const DeviceBuffer& weight, const DeviceBuffer& bias,
            const std::uint16_t* input, std::uint16_t* out, int tokens,
            int rows, int columns, std::string* error) {
    return gemm.Run(weight.get(), bias.get(), input, out, tokens, rows, columns,
                    error);
  }
  bool Reserve(int tokens, int text_rows, int unique, std::string* error);
  bool Download(const void* device, bool bf16, std::size_t count,
                std::vector<float>* out, std::string* error);
  // Uploads `text` [rows, 4096] as BF16 into text_in and runs the text
  // embedding into text_states.
  bool EmbedText(std::span<const float> text, int rows, std::string* error);
  // Cross-attention keys (normalized, head-major) and packed values of one
  // block for the rows in text_states.
  bool CrossKeysValues(const Block& b, int rows, std::uint16_t* key,
                       std::uint16_t* packed_values, std::string* error);

  struct TextCache {
    int rows{0};
    DeviceBuffer key;     // [layers][heads][rows][128]
    DeviceBuffer packed;  // [layers] x packed values (PaddedRows(rows))
  };
  TextCache text_cache[kTextSlots];

  std::vector<platform::PlatformFile> shards;
  std::map<std::string, TensorLocation, std::less<>> tensors;
  std::vector<Block> blocks;
  DeviceBuffer patch_weight, patch_bias;
  DeviceBuffer time1, time1_bias, time2, time2_bias;  // FP32
  DeviceBuffer time_proj, time_proj_bias;
  DeviceBuffer text1, text1_bias, text2, text2_bias;
  DeviceBuffer out_modulation;  // FP32 [2, W]
  DeviceBuffer proj_out, proj_out_bias;

  // Activations, sized by Reserve.
  int reserved_tokens{0};
  int reserved_text{0};
  int reserved_unique{0};
  DeviceBuffer latent, patches;
  DeviceBuffer hidden;                     // FP32 residual stream [T, W]
  DeviceBuffer normed;                     // BF16 [T, W]
  DeviceBuffer acc;                        // BF16 linear outputs [T, 3W]
  DeviceBuffer query, key, value, packed;  // head-major BF16
  DeviceBuffer attention;                  // BF16 [T, W]
  DeviceBuffer ffn;  // BF16 [min(T, kFeedForwardChunk), F] hidden of the FFN
  int rope_frames{0}, rope_height{0}, rope_width{0};  // geometry of `rope`
  DeviceBuffer text_in, text_hidden, text_states, text_acc;
  DeviceBuffer text_key, text_value, text_packed;
  DeviceBuffer rope;         // float2 [T, 64]
  DeviceBuffer mod_index;    // int [T]
  DeviceBuffer timesteps;    // FP32 [U]
  DeviceBuffer sinusoid;     // FP32 [U, 256]
  DeviceBuffer time_hidden;  // FP32 [U, W]
  DeviceBuffer temb;         // FP32 [U, W], BF16 values
  DeviceBuffer proj;         // FP32 [U, 6W], BF16 values
  DeviceBuffer noise;        // FP32 [48, F, H, W]
  std::size_t activation_bytes{0};

  hipStream_t stream{nullptr};
  LinearGemm gemm;
  void* staging[2]{};
  DeviceBuffer staging_device[2];
  hipEvent_t staged[2]{};
  int next_slot{0};
  std::size_t weight_bytes{0};
  hipEvent_t timing[2 * kLayers + 2]{};
  double attention_ms{0};
  double blocks_ms{0};
};

bool Transformer::Impl::Index(const std::filesystem::path& dir,
                              std::string* error) {
  const auto index =
      ReadJson(dir / "diffusion_pytorch_model.safetensors.index.json", error);
  if (!index)
    return false;
  const json::Value* map = index->find("weight_map");
  if (map == nullptr || !map->is_object()) {
    SetError(error, "missing transformer weight map");
    return false;
  }
  std::map<std::string, std::size_t> shard_ids;
  std::vector<std::string> shard_names;
  for (const auto& [name, file] : map->members()) {
    const std::string& shard = file.str();
    if (shard.empty() ||
        std::filesystem::path(shard).filename().string() != shard) {
      SetError(error, "invalid transformer shard name for " + name);
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
    if (!file_size ||
        platform::ReadFileAt(shards.back(), &header_size, 8, 0) != 8 ||
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
      const std::string dtype = entry.member_str("dtype");
      if ((dtype != "F32" && dtype != "BF16") || shape == nullptr ||
          !shape->is_array() || offsets == nullptr || !offsets->is_array() ||
          offsets->size() != 2) {
        SetError(error, "unsupported transformer tensor " + name);
        return false;
      }
      TensorLocation location;
      location.shard = s;
      location.f32 = dtype == "F32";
      std::uint64_t bytes = location.f32 ? 4 : 2;
      for (const auto& dim : shape->items()) {
        location.shape.push_back(dim.as_size());
        bytes *= location.shape.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end < begin || end - begin != bytes || payload + end > *file_size) {
        SetError(error, "invalid transformer tensor extent " + name);
        return false;
      }
      location.offset = payload + begin;
      location.bytes = bytes;
      tensors[name] = std::move(location);
    }
  }
  return true;
}

bool Transformer::Impl::Upload(const TensorLocation& location,
                               const std::string& name, void* target, bool bf16,
                               std::string* error) {
  if (!location.f32 && !bf16) {
    SetError(error, "transformer tensor " + name + " must be FP32");
    return false;
  }
  // Two pinned buffers alternate: the disk fills one while the other is
  // copied (and, FP32 -> BF16, converted on the device).
  for (std::uint64_t copied = 0; copied < location.bytes;) {
    const std::size_t chunk = static_cast<std::size_t>(
        std::min<std::uint64_t>(kStagingBytes, location.bytes - copied));
    const int slot = next_slot;
    next_slot ^= 1;
    if (!Check(hipEventSynchronize(staged[slot]), "weight staging", error))
      return false;
    const std::byte* buffer = ReadStaged(shards[location.shard],
                                         static_cast<std::byte*>(staging[slot]),
                                         chunk, location.offset + copied);
    if (buffer == nullptr) {
      SetError(error, "short read of transformer tensor " + name);
      return false;
    }
    if (location.f32 && bf16) {
      const std::size_t count = chunk / 4;
      if (!Check(hipMemcpyAsync(staging_device[slot].get(), buffer, chunk,
                                hipMemcpyHostToDevice, stream),
                 "weight upload", error))
        return false;
      ConvertKernel<<<Blocks(count), 256, 0, stream>>>(
          staging_device[slot].f32(), count,
          static_cast<std::uint16_t*>(target) + copied / 4);
    } else if (!Check(
                   hipMemcpyAsync(static_cast<std::byte*>(target) + copied,
                                  buffer, chunk, hipMemcpyHostToDevice, stream),
                   "weight upload", error)) {
      return false;
    }
    if (!Check(hipEventRecord(staged[slot], stream), "weight upload", error))
      return false;
    copied += chunk;
  }
  return true;
}

// Stacks the rows of `parts` (name, rows) into one [sum rows, columns]
// matrix; biases and vectors use columns = 1.
bool Transformer::Impl::LoadWeight(
    DeviceBuffer& buffer,
    std::initializer_list<std::pair<std::string, std::size_t>> parts,
    std::size_t columns, bool bf16, std::string* error) {
  std::size_t rows = 0;
  for (const auto& part : parts)
    rows += part.second;
  const std::size_t element = bf16 ? 2 : 4;
  weight_bytes += rows * columns * element;
  if (!buffer.Allocate(rows * columns * element, error))
    return false;
  std::size_t row = 0;
  for (const auto& [name, part_rows] : parts) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
      SetError(error, "missing transformer tensor " + name);
      return false;
    }
    // Shapes are checked by element count here; the caller's layout
    // (patch [3072, 48, 1, 2, 2], tables [1, k, 3072]) flattens row-major.
    std::size_t count = 1;
    for (std::size_t dim : it->second.shape)
      count *= dim;
    if (count != part_rows * columns) {
      SetError(error, "unexpected shape of transformer tensor " + name);
      return false;
    }
    void* target =
        reinterpret_cast<std::byte*>(buffer.get()) + row * columns * element;
    if (!Upload(it->second, name, target, bf16, error))
      return false;
    row += part_rows;
  }
  return true;
}

bool Transformer::Impl::Reserve(int tokens, int text_rows, int unique,
                                std::string* error) {
  if (tokens <= reserved_tokens && text_rows <= reserved_text &&
      unique <= reserved_unique)
    return true;
  tokens = std::max(tokens, reserved_tokens);
  text_rows = std::max(text_rows, reserved_text);
  unique = std::max(unique, reserved_unique);
  const std::size_t T = tokens, L = text_rows, U = unique;
  const std::size_t W = kWidth, F = kFeedForward;
  const std::size_t TP = PaddedRows(tokens), LP = PaddedRows(text_rows);
  activation_bytes = 0;
  auto allocate = [&](DeviceBuffer& buffer, std::size_t bytes) {
    activation_bytes += bytes;
    return buffer.Allocate(bytes, error);
  };
  rope_frames = rope_height = rope_width = 0;
  if (!allocate(latent, T * kPatchValues * 4) ||
      !allocate(patches, T * kPatchValues * 2) ||
      !allocate(hidden, T * W * 4) || !allocate(normed, T * W * 2) ||
      !allocate(acc, T * 3 * W * 2) || !allocate(query, T * W * 2) ||
      !allocate(key, T * W * 2) || !allocate(value, T * W * 2) ||
      !allocate(packed, TP * W * 2) || !allocate(attention, T * W * 2) ||
      !allocate(ffn, std::min<std::size_t>(T, kFeedForwardChunk) * F * 2) ||
      !allocate(text_in, L * kTextWidth * 2) ||
      !allocate(text_hidden, L * W * 2) || !allocate(text_states, L * W * 2) ||
      !allocate(text_acc, L * 2 * W * 2) || !allocate(text_key, L * W * 2) ||
      !allocate(text_value, L * W * 2) || !allocate(text_packed, LP * W * 2) ||
      !allocate(rope, T * kRopePairs * 8) || !allocate(mod_index, T * 4) ||
      !allocate(timesteps, U * 4) ||
      !allocate(sinusoid, U * kFrequencies * 4) ||
      !allocate(time_hidden, U * W * 4) || !allocate(temb, U * W * 4) ||
      !allocate(proj, U * kModulation * 4) ||
      !allocate(noise, T * kPatchValues * 4)) {
    reserved_tokens = reserved_text = reserved_unique = 0;
    return false;
  }
  reserved_tokens = tokens;
  reserved_text = text_rows;
  reserved_unique = unique;
  return true;
}

bool Transformer::Impl::Download(const void* device, bool bf16,
                                 std::size_t count, std::vector<float>* out,
                                 std::string* error) {
  out->resize(count);
  if (!bf16)
    return Check(hipMemcpyAsync(out->data(), device, count * 4,
                                hipMemcpyDeviceToHost, stream),
                 "readback", error) &&
           Check(hipStreamSynchronize(stream), "readback", error);
  std::vector<std::uint16_t> half(count);
  if (!Check(hipMemcpyAsync(half.data(), device, count * 2,
                            hipMemcpyDeviceToHost, stream),
             "readback", error) ||
      !Check(hipStreamSynchronize(stream), "readback", error))
    return false;
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(half[i]) << 16U;
    std::memcpy(&(*out)[i], &bits, sizeof(float));
  }
  return true;
}

bool Transformer::Impl::EmbedText(std::span<const float> text, int rows,
                                  std::string* error) {
  std::vector<std::uint16_t> half(text.size());
  for (std::size_t i = 0; i < half.size(); ++i) {
    // Round to nearest even like the BF16 cast of the pipeline.
    std::uint32_t bits;
    std::memcpy(&bits, &text[i], 4);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    half[i] = static_cast<std::uint16_t>(bits >> 16U);
  }
  if (!Check(hipMemcpyAsync(text_in.get(), half.data(), half.size() * 2,
                            hipMemcpyHostToDevice, stream),
             "text upload", error))
    return false;
  // Text embedding: linear, GELU (tanh), linear.
  const std::size_t count = static_cast<std::size_t>(rows) * kWidth;
  if (!Gemm(text1, text1_bias, text_in.get(), text_acc.get(), rows, kWidth,
            kTextWidth, error))
    return false;
  GeluKernel<<<Blocks(count), 256, 0, stream>>>(text_acc.get(), count,
                                                text_hidden.get());
  if (!Gemm(text2, text2_bias, text_hidden.get(), text_states.get(), rows,
            kWidth, kWidth, error))
    return false;
  // The host buffer must stay alive until the copy is done.
  return Check(hipStreamSynchronize(stream), "text embedding", error);
}

bool Transformer::Impl::CrossKeysValues(const Block& b, int rows,
                                        std::uint16_t* key_out,
                                        std::uint16_t* packed_values,
                                        std::string* error) {
  if (!Gemm(b.cross_kv, b.cross_kv_bias, text_states.get(), text_acc.get(),
            rows, 2 * kWidth, kWidth, error))
    return false;
  HeadsKernel<<<rows, kRowThreads, 0, stream>>>(
      text_acc.get(), 2 * kWidth, rows, b.cross_norm_k.get(), nullptr, key_out);
  HeadsKernel<<<rows, kRowThreads, 0, stream>>>(text_acc.get() + kWidth,
                                                2 * kWidth, rows, nullptr,
                                                nullptr, text_value.get());
  LaunchPackValues(text_value.get(), packed_values, rows, kHeads, stream);
  return true;
}

bool Transformer::PrepareText(int slot, std::span<const float> text,
                              int text_rows, std::string* error) {
  Impl& m = *impl_;
  if (slot < 0 || slot >= kTextSlots || text_rows <= 0 || text_rows > 4096 ||
      text.size() != static_cast<std::size_t>(text_rows) * kTextWidth) {
    SetError(error, "Wan transformer: invalid text slot or text states");
    return false;
  }
  if (!m.Reserve(std::max(m.reserved_tokens, 1), text_rows,
                 std::max(m.reserved_unique, 1), error) ||
      !m.EmbedText(text, text_rows, error))
    return false;
  Impl::TextCache& cache = m.text_cache[slot];
  const std::size_t key_count = static_cast<std::size_t>(text_rows) * kWidth;
  const std::size_t packed_count =
      static_cast<std::size_t>(PaddedRows(text_rows)) * kWidth;
  if (cache.rows != text_rows) {
    cache.rows = 0;
    if (!cache.key.Allocate(kLayers * key_count * 2, error) ||
        !cache.packed.Allocate(kLayers * packed_count * 2, error))
      return false;
  }
  for (int layer = 0; layer < kLayers; ++layer)
    if (!m.CrossKeysValues(m.blocks[layer], text_rows,
                           cache.key.get() + layer * key_count,
                           cache.packed.get() + layer * packed_count, error))
      return false;
  if (!Check(hipGetLastError(), "text keys", error) ||
      !Check(hipStreamSynchronize(m.stream), "text keys", error))
    return false;
  cache.rows = text_rows;
  return true;
}

Transformer::Transformer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Transformer::~Transformer() = default;

std::size_t Transformer::weight_bytes() const {
  return impl_->weight_bytes;
}
std::size_t Transformer::activation_bytes() const {
  return impl_->activation_bytes;
}
double Transformer::last_attention_ms() const {
  return impl_->attention_ms;
}
double Transformer::last_blocks_ms() const {
  return impl_->blocks_ms;
}

std::unique_ptr<Transformer> Transformer::Load(const std::filesystem::path& dir,
                                               std::string* error) {
  const auto config = ReadJson(dir / "config.json", error);
  if (!config)
    return nullptr;
  for (auto [key, value] : {std::pair{"num_attention_heads", kHeads},
                            {"attention_head_dim", kHeadDim},
                            {"in_channels", kChannels},
                            {"out_channels", kChannels},
                            {"ffn_dim", kFeedForward},
                            {"freq_dim", kFrequencies},
                            {"num_layers", kLayers},
                            {"text_dim", kTextWidth}})
    if (config->member_double(key, -1) != value) {
      SetError(error,
               std::string("unsupported transformer configuration: ") + key);
      return nullptr;
    }
  const json::Value* patch = config->find("patch_size");
  if (patch == nullptr || !patch->is_array() || patch->size() != 3 ||
      patch->items()[0].as_size() != 1 || patch->items()[1].as_size() != 2 ||
      patch->items()[2].as_size() != 2 ||
      config->member_str("qk_norm") != "rms_norm_across_heads" ||
      config->member_double("eps") != 1e-6 ||
      config->member_double("rope_max_seq_len") != 1024) {
    SetError(error, "unsupported transformer configuration");
    return nullptr;
  }
  const json::Value* cross_norm = config->find("cross_attn_norm");
  const json::Value* image_dim = config->find("image_dim");
  const json::Value* added_kv = config->find("added_kv_proj_dim");
  if (cross_norm == nullptr || !cross_norm->is_bool() ||
      !cross_norm->as_bool() ||
      (image_dim != nullptr && !image_dim->is_null()) ||
      (added_kv != nullptr && !added_kv->is_null())) {
    SetError(error,
             "unsupported transformer configuration (text-to-video "
             "TI2V-5B only)");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  if (!impl->Index(dir, error))
    return nullptr;
  if (!Check(hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking),
             "stream", error))
    return nullptr;
  if (!impl->gemm.Init(impl->stream, error))
    return nullptr;
  for (int slot = 0; slot < 2; ++slot)
    if (!Check(
            hipHostMalloc(&impl->staging[slot], kStagingBytes + kStagingSlack,
                          hipHostMallocDefault),
            "pinned staging", error) ||
        !impl->staging_device[slot].Allocate(kStagingBytes, error) ||
        !Check(
            hipEventCreateWithFlags(&impl->staged[slot], hipEventDisableTiming),
            "staging event", error) ||
        !Check(hipEventRecord(impl->staged[slot], impl->stream),
               "staging event", error))
      return nullptr;
  for (hipEvent_t& event : impl->timing)
    if (!Check(hipEventCreate(&event), "timing event", error))
      return nullptr;

  constexpr std::size_t W = kWidth;
  constexpr std::size_t F = kFeedForward;
  Impl& m = *impl;
  const auto load = [&](DeviceBuffer& buffer, const std::string& name,
                        std::size_t rows, std::size_t columns, bool bf16) {
    return m.LoadWeight(buffer, {{name, rows}}, columns, bf16, error);
  };
  if (!load(m.patch_weight, "patch_embedding.weight", W, kPatchValues, true) ||
      !load(m.patch_bias, "patch_embedding.bias", W, 1, true) ||
      !load(m.time1, "condition_embedder.time_embedder.linear_1.weight", W,
            kFrequencies, false) ||
      !load(m.time1_bias, "condition_embedder.time_embedder.linear_1.bias", W,
            1, false) ||
      !load(m.time2, "condition_embedder.time_embedder.linear_2.weight", W, W,
            false) ||
      !load(m.time2_bias, "condition_embedder.time_embedder.linear_2.bias", W,
            1, false) ||
      !load(m.time_proj, "condition_embedder.time_proj.weight", kModulation, W,
            true) ||
      !load(m.time_proj_bias, "condition_embedder.time_proj.bias", kModulation,
            1, true) ||
      !load(m.text1, "condition_embedder.text_embedder.linear_1.weight", W,
            kTextWidth, true) ||
      !load(m.text1_bias, "condition_embedder.text_embedder.linear_1.bias", W,
            1, true) ||
      !load(m.text2, "condition_embedder.text_embedder.linear_2.weight", W, W,
            true) ||
      !load(m.text2_bias, "condition_embedder.text_embedder.linear_2.bias", W,
            1, true) ||
      !load(m.out_modulation, "scale_shift_table", 2, W, false) ||
      !load(m.proj_out, "proj_out.weight", kPatchValues, W, true) ||
      !load(m.proj_out_bias, "proj_out.bias", kPatchValues, 1, true))
    return nullptr;
  m.blocks = std::vector<Block>(kLayers);
  for (int index = 0; index < kLayers; ++index) {
    Block& b = m.blocks[index];
    const std::string p = "blocks." + std::to_string(index) + ".";
    const std::string a1 = p + "attn1.";
    const std::string a2 = p + "attn2.";
    if (!m.LoadWeight(b.qkv,
                      {{a1 + "to_q.weight", W},
                       {a1 + "to_k.weight", W},
                       {a1 + "to_v.weight", W}},
                      W, true, error) ||
        !m.LoadWeight(b.qkv_bias,
                      {{a1 + "to_q.bias", W},
                       {a1 + "to_k.bias", W},
                       {a1 + "to_v.bias", W}},
                      1, true, error) ||
        !load(b.norm_q, a1 + "norm_q.weight", W, 1, true) ||
        !load(b.norm_k, a1 + "norm_k.weight", W, 1, true) ||
        !load(b.out, a1 + "to_out.0.weight", W, W, true) ||
        !load(b.out_bias, a1 + "to_out.0.bias", W, 1, true) ||
        !load(b.cross_q, a2 + "to_q.weight", W, W, true) ||
        !load(b.cross_q_bias, a2 + "to_q.bias", W, 1, true) ||
        !m.LoadWeight(b.cross_kv,
                      {{a2 + "to_k.weight", W}, {a2 + "to_v.weight", W}}, W,
                      true, error) ||
        !m.LoadWeight(b.cross_kv_bias,
                      {{a2 + "to_k.bias", W}, {a2 + "to_v.bias", W}}, 1, true,
                      error) ||
        !load(b.cross_norm_q, a2 + "norm_q.weight", W, 1, true) ||
        !load(b.cross_norm_k, a2 + "norm_k.weight", W, 1, true) ||
        !load(b.cross_out, a2 + "to_out.0.weight", W, W, true) ||
        !load(b.cross_out_bias, a2 + "to_out.0.bias", W, 1, true) ||
        !load(b.norm2_weight, p + "norm2.weight", W, 1, false) ||
        !load(b.norm2_bias, p + "norm2.bias", W, 1, false) ||
        !load(b.up, p + "ffn.net.0.proj.weight", F, W, true) ||
        !load(b.up_bias, p + "ffn.net.0.proj.bias", F, 1, true) ||
        !load(b.down, p + "ffn.net.2.weight", W, F, true) ||
        !load(b.down_bias, p + "ffn.net.2.bias", W, 1, true) ||
        !load(b.modulation, p + "scale_shift_table", 6, W, false))
      return nullptr;
  }
  if (!Check(hipStreamSynchronize(m.stream), "weight upload", error))
    return nullptr;
  for (auto& buffer : m.staging_device)
    buffer.Release();
  return std::unique_ptr<Transformer>(new Transformer(std::move(impl)));
}

bool Transformer::Forward(const TransformerInput& input,
                          std::vector<float>* noise, TransformerProbes* probes,
                          std::string* error) {
  Impl& m = *impl_;
  if (input.frames <= 0 || input.height <= 0 || input.width <= 0 ||
      input.height % 2 != 0 || input.width % 2 != 0 || input.frames > 1024 ||
      input.height / 2 > 1024 || input.width / 2 > 1024) {
    SetError(error, "Wan transformer: invalid latent shape");
    return false;
  }
  const int ph = input.height / 2, pw = input.width / 2;
  const int tokens = input.frames * ph * pw;
  const std::size_t latent_count = static_cast<std::size_t>(kChannels) *
                                   input.frames * input.height * input.width;
  if (input.latent.size() != latent_count) {
    SetError(error, "Wan transformer: latent size does not match its shape");
    return false;
  }
  if (input.timesteps.size() != 1 &&
      input.timesteps.size() != static_cast<std::size_t>(tokens)) {
    SetError(error, "Wan transformer: need one timestep or one per token");
    return false;
  }
  // Cached text states (PrepareText) unless probing, which needs the
  // embedding of this call.
  const bool cached = input.text_slot >= 0 && probes == nullptr;
  if (input.text_slot >= kTextSlots ||
      (input.text_slot >= 0 && m.text_cache[input.text_slot].rows == 0)) {
    SetError(error, "Wan transformer: text slot was not prepared");
    return false;
  }
  const int text_rows =
      cached ? m.text_cache[input.text_slot].rows : input.text_rows;
  if (!cached &&
      (text_rows <= 0 || text_rows > 4096 ||
       input.text.size() != static_cast<std::size_t>(text_rows) * kTextWidth)) {
    SetError(error, "Wan transformer: invalid text states");
    return false;
  }
  // Tokens sharing a timestep share their modulation rows.
  std::vector<float> unique;
  std::vector<int> index(tokens);
  for (int t = 0; t < tokens; ++t) {
    const float value = input.timesteps[input.timesteps.size() == 1 ? 0 : t];
    auto it = std::find(unique.begin(), unique.end(), value);
    if (it == unique.end()) {
      if (unique.size() == 64) {
        SetError(error, "Wan transformer: more than 64 distinct timesteps");
        return false;
      }
      unique.push_back(value);
      it = unique.end() - 1;
    }
    index[t] = static_cast<int>(it - unique.begin());
  }
  const int U = static_cast<int>(unique.size());
  if (!m.Reserve(tokens, text_rows, U, error))
    return false;
  hipStream_t s = m.stream;
  // 3D RoPE: 22 frequency pairs over frames, 21 over rows, 21 over columns
  // (computed once per latent geometry).
  std::vector<float> rope;
  const bool new_rope = m.rope_frames != input.frames ||
                        m.rope_height != input.height ||
                        m.rope_width != input.width;
  if (new_rope)
    rope.resize(static_cast<std::size_t>(tokens) * kRopePairs * 2);
  for (int t = 0; new_rope && t < tokens; ++t) {
    const int position[3] = {t / (ph * pw), (t / pw) % ph, t % pw};
    for (int j = 0; j < kRopePairs; ++j) {
      const int axis = j < kRopeTimePairs
                           ? 0
                           : (j < kRopeTimePairs + kRopeSpacePairs ? 1 : 2);
      const int k =
          axis == 0 ? j : j - kRopeTimePairs - (axis == 2) * kRopeSpacePairs;
      const int dims = axis == 0 ? 2 * kRopeTimePairs : 2 * kRopeSpacePairs;
      const double frequency =
          1.0 / std::pow(10000.0, static_cast<double>(2 * k) / dims);
      const double angle = position[axis] * frequency;
      rope[(static_cast<std::size_t>(t) * kRopePairs + j) * 2] =
          static_cast<float>(std::cos(angle));
      rope[(static_cast<std::size_t>(t) * kRopePairs + j) * 2 + 1] =
          static_cast<float>(std::sin(angle));
    }
  }
  if (!Check(hipMemcpyAsync(m.latent.f32(), input.latent.data(),
                            latent_count * 4, hipMemcpyHostToDevice, s),
             "input upload", error) ||
      (new_rope &&
       !Check(hipMemcpyAsync(m.rope.f32(), rope.data(), rope.size() * 4,
                             hipMemcpyHostToDevice, s),
              "input upload", error)) ||
      !Check(hipMemcpyAsync(m.mod_index.i32(), index.data(), index.size() * 4,
                            hipMemcpyHostToDevice, s),
             "input upload", error) ||
      !Check(hipMemcpyAsync(m.timesteps.f32(), unique.data(), U * 4,
                            hipMemcpyHostToDevice, s),
             "input upload", error))
    return false;
  // Host buffers must stay alive until the copies are done.
  if (!Check(hipStreamSynchronize(s), "input upload", error))
    return false;
  if (new_rope) {
    m.rope_frames = input.frames;
    m.rope_height = input.height;
    m.rope_width = input.width;
  }
  if (!cached && !m.EmbedText(input.text, text_rows, error))
    return false;

  const std::size_t T = tokens;
  const std::size_t W = kWidth;
  const std::size_t count = T * W;
  const auto* rope_table = reinterpret_cast<const float2*>(m.rope.f32());

  // Patch embedding: a [T, 192] x [192, W] GEMM.
  PatchKernel<<<Blocks(T * kPatchValues), 256, 0, s>>>(
      m.latent.f32(), input.frames, input.height, input.width, m.patches.get());
  if (!m.Gemm(m.patch_weight, m.patch_bias, m.patches.get(), m.acc.get(),
              tokens, W, kPatchValues, error))
    return false;
  WidenKernel<<<Blocks(count), 256, 0, s>>>(m.acc.get(), count, m.hidden.f32());

  // Time embedding (FP32 time embedder, BF16 temb and projection).
  SinusoidKernel<<<U, kFrequencies / 2, 0, s>>>(m.timesteps.f32(),
                                                m.sinusoid.f32());
  MatVecKernel<float, Activation::kNone, false>
      <<<dim3(W, U), kRowThreads, 0, s>>>(m.time1.f32(), m.time1_bias.f32(),
                                          m.sinusoid.f32(), kFrequencies, W,
                                          m.time_hidden.f32());
  MatVecKernel<float, Activation::kSilu, true>
      <<<dim3(W, U), kRowThreads, 0, s>>>(m.time2.f32(), m.time2_bias.f32(),
                                          m.time_hidden.f32(), W, W,
                                          m.temb.f32());
  MatVecKernel<std::uint16_t, Activation::kSiluBf16, true>
      <<<dim3(kModulation, U), kRowThreads, 0, s>>>(
          m.time_proj.get(), m.time_proj_bias.get(), m.temb.f32(), W,
          kModulation, m.proj.f32());

  const std::size_t text_count = static_cast<std::size_t>(text_rows) * W;
  if (!Check(hipGetLastError(), "embedding launch", error))
    return false;

  if (probes != nullptr &&
      (!m.Download(m.hidden.f32(), false, count, &probes->patch, error) ||
       !m.Download(m.temb.f32() + static_cast<std::size_t>(index[0]) * W, false,
                   W, &probes->temb, error) ||
       !m.Download(
           m.proj.f32() + static_cast<std::size_t>(index[0]) * kModulation,
           false, kModulation, &probes->timestep_proj, error) ||
       !m.Download(m.text_states.get(), true, text_count, &probes->text_states,
                   error)))
    return false;

  const unsigned row_blocks = static_cast<unsigned>(T);
  const int* mod_index = m.mod_index.i32();
  const auto modulation = [&](const Block& b, int shift, int scale) {
    return Modulation{b.modulation.f32() + shift * W,
                      b.modulation.f32() + scale * W,
                      m.proj.f32() + shift * W,
                      m.proj.f32() + scale * W,
                      kModulation,
                      mod_index};
  };
  std::uint16_t* linear = m.acc.get();
  const std::size_t text_key_count = static_cast<std::size_t>(text_rows) * W;
  const std::size_t text_packed_count =
      static_cast<std::size_t>(PaddedRows(text_rows)) * W;
  if (!Check(hipEventRecord(m.timing[2 * kLayers], s), "timing", error))
    return false;
  for (int layer = 0; layer < kLayers; ++layer) {
    const Block& b = m.blocks[layer];
    const bool probing = probes != nullptr && layer == 0;
    // 1. Self-attention.
    LayerNormKernel<<<row_blocks, kRowThreads, 0, s>>>(
        m.hidden.f32(), modulation(b, 0, 1), nullptr, nullptr, m.normed.get());
    if (!m.Gemm(b.qkv, b.qkv_bias, m.normed.get(), linear, tokens, 3 * W, W,
                error))
      return false;
    HeadsKernel<<<row_blocks, kRowThreads, 0, s>>>(
        linear, 3 * W, tokens, b.norm_q.get(), rope_table, m.query.get());
    HeadsKernel<<<row_blocks, kRowThreads, 0, s>>>(
        linear + W, 3 * W, tokens, b.norm_k.get(), rope_table, m.key.get());
    HeadsKernel<<<row_blocks, kRowThreads, 0, s>>>(
        linear + 2 * W, 3 * W, tokens, nullptr, nullptr, m.value.get());
    LaunchPackValues(m.value.get(), m.packed.get(), tokens, kHeads, s);
    (void)hipEventRecord(m.timing[2 * layer], s);
    LaunchAttention(m.query.get(), m.key.get(), m.packed.get(),
                    m.attention.get(), tokens, tokens, kHeads, s);
    (void)hipEventRecord(m.timing[2 * layer + 1], s);
    if (!m.Gemm(b.out, b.out_bias, m.attention.get(), linear, tokens, W, W,
                error))
      return false;
    if (probing &&
        !m.Download(linear, true, count, &probes->block0_attn1, error))
      return false;
    ResidualKernel<<<Blocks(count), 256, 0, s>>>(
        linear, b.modulation.f32() + 2 * W, m.proj.f32() + 2 * W, mod_index,
        count, m.hidden.f32());

    // 2. Cross-attention to the text states.
    LayerNormKernel<<<row_blocks, kRowThreads, 0, s>>>(
        m.hidden.f32(), Modulation{}, b.norm2_weight.f32(), b.norm2_bias.f32(),
        m.normed.get());
    if (!m.Gemm(b.cross_q, b.cross_q_bias, m.normed.get(), linear, tokens, W, W,
                error))
      return false;
    HeadsKernel<<<row_blocks, kRowThreads, 0, s>>>(
        linear, W, tokens, b.cross_norm_q.get(), nullptr, m.query.get());
    const std::uint16_t* text_key = m.text_key.get();
    const std::uint16_t* text_packed = m.text_packed.get();
    if (cached) {
      const Impl::TextCache& cache = m.text_cache[input.text_slot];
      text_key = cache.key.get() + layer * text_key_count;
      text_packed = cache.packed.get() + layer * text_packed_count;
    } else if (!m.CrossKeysValues(b, text_rows, m.text_key.get(),
                                  m.text_packed.get(), error)) {
      return false;
    }
    LaunchAttention(m.query.get(), text_key, text_packed, m.attention.get(),
                    tokens, text_rows, kHeads, s);
    if (!m.Gemm(b.cross_out, b.cross_out_bias, m.attention.get(), linear,
                tokens, W, W, error))
      return false;
    if (probing &&
        !m.Download(linear, true, count, &probes->block0_attn2, error))
      return false;
    ResidualKernel<<<Blocks(count), 256, 0, s>>>(
        linear, nullptr, nullptr, nullptr, count, m.hidden.f32());

    // 3. Feed-forward (GELU tanh).
    LayerNormKernel<<<row_blocks, kRowThreads, 0, s>>>(
        m.hidden.f32(), modulation(b, 3, 4), nullptr, nullptr, m.normed.get());
    for (int first = 0; first < tokens; first += kFeedForwardChunk) {
      const int rows = std::min(kFeedForwardChunk, tokens - first);
      const std::size_t up_count =
          static_cast<std::size_t>(rows) * kFeedForward;
      if (!m.Gemm(b.up, b.up_bias, m.normed.get() + first * W, m.ffn.get(),
                  rows, kFeedForward, W, error))
        return false;
      GeluKernel<<<Blocks(up_count), 256, 0, s>>>(m.ffn.get(), up_count,
                                                  m.ffn.get());
      if (!m.Gemm(b.down, b.down_bias, m.ffn.get(), linear + first * W, rows, W,
                  kFeedForward, error))
        return false;
    }
    if (probing && !m.Download(linear, true, count, &probes->block0_ffn, error))
      return false;
    ResidualKernel<<<Blocks(count), 256, 0, s>>>(
        linear, b.modulation.f32() + 5 * W, m.proj.f32() + 5 * W, mod_index,
        count, m.hidden.f32());
    if (!Check(hipGetLastError(), "block launch", error))
      return false;
    if (probing &&
        !m.Download(m.hidden.f32(), false, count, &probes->block0, error))
      return false;
  }
  if (!Check(hipEventRecord(m.timing[2 * kLayers + 1], s), "timing", error))
    return false;
  if (probes != nullptr &&
      !m.Download(m.hidden.f32(), false, count, &probes->block_last, error))
    return false;

  // Output norm, modulation by temb, projection and unpatchify.
  const Modulation final_modulation{m.out_modulation.f32(),
                                    m.out_modulation.f32() + W,
                                    m.temb.f32(),
                                    m.temb.f32(),
                                    kWidth,
                                    mod_index};
  LayerNormKernel<<<row_blocks, kRowThreads, 0, s>>>(
      m.hidden.f32(), final_modulation, nullptr, nullptr, m.normed.get());
  if (!m.Gemm(m.proj_out, m.proj_out_bias, m.normed.get(), linear, tokens,
              kPatchValues, W, error))
    return false;
  UnpatchKernel<<<Blocks(T * kPatchValues), 256, 0, s>>>(
      linear, input.frames, input.height, input.width, m.noise.f32());
  if (!Check(hipGetLastError(), "output launch", error) ||
      !m.Download(m.noise.f32(), false, latent_count, noise, error))
    return false;
  m.attention_ms = 0;
  for (int layer = 0; layer < kLayers; ++layer) {
    float ms = 0;
    if (hipEventElapsedTime(&ms, m.timing[2 * layer],
                            m.timing[2 * layer + 1]) == hipSuccess)
      m.attention_ms += ms;
  }
  float ms = 0;
  m.blocks_ms = hipEventElapsedTime(&ms, m.timing[2 * kLayers],
                                    m.timing[2 * kLayers + 1]) == hipSuccess
                    ? ms
                    : -1;
  return true;
}

}  // namespace gufo::models::wan22
