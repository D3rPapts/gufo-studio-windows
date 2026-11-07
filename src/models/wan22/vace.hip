// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/attention.hip.hpp"
#include "src/models/wan22/gemm.hpp"
#include "src/models/wan22/vace.hpp"

// Wan 2.1 VACE context blocks. The kernels repeat transformer21.hip's (same
// width, heads, norms and modulation; see there); the blocks run on the
// transformer's stream inside its Forward through Transformer21Hook and use
// its workspace (Transformer21BlockState).

namespace gufo::models::wan22 {
namespace {

constexpr int kDitLayers = 40;
constexpr int kHeads = 40;
constexpr int kHeadDim = 128;
constexpr int kWidth = Transformer21::kModelWidth;
constexpr int kFeedForward = 13824;
constexpr int kPatchIn = Vace::kContextChannels * 4;  // 384
constexpr int kRopePairs = kHeadDim / 2;
constexpr float kEpsilon = 1e-6F;
constexpr int kRowThreads = 256;
constexpr int kRowValues = kWidth / kRowThreads;  // 20

static_assert(kWidth == kHeads * kHeadDim);

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

__device__ __forceinline__ float GeluTanh(float x) {
  const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

// OCP FP8 e4m3fn (bias 7, no infinities).
__device__ __forceinline__ float Fp8(unsigned int byte) {
  const unsigned int e = (byte >> 3U) & 15U, m = byte & 7U;
  const float v = e == 0 ? static_cast<float>(m) * 0.001953125F
                         : ldexpf(1.0F + static_cast<float>(m) * 0.125F,
                                  static_cast<int>(e) - 7);
  return (byte & 0x80U) != 0 ? -v : v;
}

__device__ __forceinline__ float BlockSum(float value, float* shared) {
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

// out[i] = bf16(fp8(w[i]) * scale), four values a thread.
__global__ void UnpackKernel(const std::uint32_t* weight, std::size_t quads,
                             float scale, std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= quads)
    return;
  const std::uint32_t w = weight[i];
  ushort4 v;
  v.x = ToBf16(Fp8(w & 0xFFU) * scale);
  v.y = ToBf16(Fp8((w >> 8U) & 0xFFU) * scale);
  v.z = ToBf16(Fp8((w >> 16U) & 0xFFU) * scale);
  v.w = ToBf16(Fp8(w >> 24U) * scale);
  reinterpret_cast<ushort4*>(out)[i] = v;
}

// patches[t, c * 4 + kh * 2 + kw] = context[c, f, 2h + kh, 2w + kw], BF16.
__global__ void ContextPatchKernel(const float* context, int frames,
                                   int height, int width,
                                   std::uint16_t* patches) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ph = height / 2, pw = width / 2;
  const std::size_t tokens = static_cast<std::size_t>(frames) * ph * pw;
  if (index >= tokens * kPatchIn)
    return;
  const std::size_t token = index / kPatchIn;
  const int column = static_cast<int>(index % kPatchIn);
  const int c = column / 4, kh = (column / 2) % 2, kw = column % 2;
  const int w = static_cast<int>(token % pw);
  const int h = static_cast<int>((token / pw) % ph);
  const int f = static_cast<int>(token / (static_cast<std::size_t>(pw) * ph));
  patches[index] =
      ToBf16(context[((static_cast<std::size_t>(c) * frames + f) * height +
                      2 * h + kh) *
                         width +
                     2 * w + kw]);
}

__global__ void GeluKernel(std::uint16_t* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = ToBf16(GeluTanh(FromBf16(x[i])));
}

struct NormArgs {
  const float* shift_table{nullptr};
  const float* scale_table{nullptr};
  const float* shift_proj{nullptr};
  const float* scale_proj{nullptr};
  const float* weight{nullptr};
  const float* bias{nullptr};
};

// FP32 layer norm of [rows, 5120]: modulated (table + projection) or affine.
__global__ void LayerNormKernel(const float* hidden, NormArgs a,
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
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    const int c = threadIdx.x + i * kRowThreads;
    float y = values[i] * rstd;
    if (a.scale_table != nullptr)
      y = y * (1.0F + a.scale_table[c] + a.scale_proj[c]) + a.shift_table[c] +
          a.shift_proj[c];
    else if (a.weight != nullptr)
      y = y * a.weight[c] + a.bias[c];
    target[c] = ToBf16(y);
  }
}

// One projection of BF16 linear rows, optionally RMS normalized across all
// heads and rotated, written head-major [heads][rows][128].
__global__ void HeadsKernel(const std::uint16_t* linear, int stride, int rows,
                            const float* norm_weight, const float2* rope,
                            std::uint16_t* out) {
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
        values[2 * p + j] = values[2 * p + j] * rstd * norm_weight[c + j];
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

// hidden += y * (gate_table + gate_proj) (or y), y BF16.
__global__ void ResidualKernel(const std::uint16_t* linear,
                               const float* gate_table, const float* gate_proj,
                               std::size_t count, float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % kWidth);
  const float y = FromBf16(linear[i]);
  hidden[i] += gate_table == nullptr ? y : y * (gate_table[c] + gate_proj[c]);
}

// out = y + x (y BF16, x FP32): the first block's before_proj(c) + x.
__global__ void WidenAddKernel(const std::uint16_t* linear, const float* x,
                               std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = FromBf16(linear[i]) + x[i];
}

// hidden += strength * y (the hint into the DiT stream).
__global__ void ScaledAddKernel(const std::uint16_t* linear, float strength,
                                std::size_t count, float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    hidden[i] += strength * FromBf16(linear[i]);
}

__global__ void NarrowKernel(const float* x, std::size_t count,
                             std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(x[i]);
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
  SetError(error, std::string("Wan 2.1 VACE ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
}

float HalfToFloat(std::uint16_t h) {
  const int sign = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float value;
  if (e == 0)
    value = std::ldexp(static_cast<float>(m), -24);
  else if (e == 31)
    value = m != 0 ? NAN : INFINITY;
  else
    value = std::ldexp(static_cast<float>(1024 + m), e - 25);
  return sign != 0 ? -value : value;
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

std::uint16_t HostBf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)) {}
  ~DeviceBuffer() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  bool Allocate(std::size_t bytes, std::string* error) {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    return Check(hipMalloc(&data_, std::max<std::size_t>(bytes, 256)),
                 "device allocation", error);
  }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }
  void* raw() const { return data_; }

private:
  void* data_{nullptr};
};

enum class DType { kF8, kF16, kBF16, kF32 };

struct TensorLocation {
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::uint64_t count{0};
  DType dtype{DType::kF32};
  std::vector<std::size_t> shape;
};

// A linear layer: FP8 bytes with a per-tensor scale (Kijai's scaled FP8) or
// BF16 values, [rows, columns]; BF16 bias (may be empty).
struct Linear {
  DeviceBuffer weight;
  bool fp8{false};
  float scale{1};
  int rows{0}, columns{0};
  DeviceBuffer bias;
};

struct Block {
  Linear q, k, v, o, cross_q, cross_k, cross_v, cross_o, up, down;
  Linear before, after;
  bool has_before{false};
  DeviceBuffer qkv_bias;                                    // BF16 [3W]
  DeviceBuffer norm_q, norm_k, cross_norm_q, cross_norm_k;  // FP32
  DeviceBuffer norm3_weight, norm3_bias;                    // FP32
  DeviceBuffer modulation;                                  // FP32 [6, W]
};

struct TextCache {
  std::uint64_t serial{0};
  int rows{0};
  DeviceBuffer key, packed;  // [blocks] x [heads][rows][128] / packed
};

}  // namespace

struct Vace::Impl {
  ~Impl() {
    if (platform::FileValid(file))
      platform::CloseFile(file);
  }
  bool Index(const std::filesystem::path& path, std::string* error);
  const TensorLocation* Find(const std::string& name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
  }
  bool ReadFloats(const std::string& name, std::size_t count,
                  std::vector<float>* out, std::string* error);
  bool UploadBf16(DeviceBuffer& buffer, const std::vector<float>& host,
                  std::string* error);
  bool UploadF32(DeviceBuffer& buffer, const std::vector<float>& host,
                 std::string* error);
  bool LoadVector(DeviceBuffer& buffer, const std::string& name,
                  std::size_t count, bool bf16, std::string* error);
  // `<base>.weight` (FP8 with `<base>.scale_weight`, or F16/BF16/F32) and,
  // when `with_bias`, `<base>.bias`.
  bool LoadLinear(Linear& linear, const std::string& base, int rows,
                  int columns, bool with_bias, std::string* error);
  // `linear` as BF16 into `target` (FP8 unpacked, BF16 copied).
  bool Unpack(const Linear& linear, std::uint16_t* target,
              std::string* error);
  bool Gemm(const Linear& linear, const std::uint16_t* input,
            std::uint16_t* out, int tokens, std::uint16_t* scratch,
            std::string* error) {
    return Unpack(linear, scratch, error) &&
           gemm.Run(scratch, linear.bias.get(), input, out, tokens,
                    linear.rows, linear.columns, error);
  }
  bool PrepareText(const Transformer21BlockState& state, std::string* error);
  bool RunBlock(int n, std::string* error);

  platform::PlatformFile file{platform::kInvalidFile};
  std::map<std::string, TensorLocation, std::less<>> tensors;
  std::vector<Block> blocks;
  int stride{5};  // DiT blocks per VACE block
  DeviceBuffer patch, patch_bias;  // BF16 [W, 384], [W]
  std::size_t weight_bytes{0};
  hipStream_t stream{nullptr};
  LinearGemm gemm;
  float strength{1.0F};
  bool probe{false};
  std::vector<float> hint_first;

  DeviceBuffer context;  // FP32 [96, frames, height, width]
  int frames{0}, height{0}, width{0};
  DeviceBuffer hidden;  // FP32 [tokens, W]: the context stream c
  int reserved{0};
  TextCache text[Transformer21::kContextSlots];
  DeviceBuffer text_linear, text_values;  // BF16 [rows, W]
  int text_reserved{0};
  Transformer21BlockState state;
  bool active{false};
};

bool Vace::Impl::Index(const std::filesystem::path& path, std::string* error) {
  std::string open_error;
  file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return false;
  }
  const auto file_size = platform::FileSizeBytes(file);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file, text.data(), text.size(), 8) != text.size()) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      // Bare Kijai keys or a ComfyUI checkpoint prefix; VACE tensors only.
      std::string key = name;
      for (const char* prefix : {"model.diffusion_model.", "diffusion_model."})
        if (key.rfind(prefix, 0) == 0)
          key = key.substr(std::strlen(prefix));
      if (key.rfind("vace_", 0) != 0)
        continue;
      const std::string dtype = entry.member_str("dtype");
      TensorLocation location;
      std::uint64_t element;
      if (dtype == "F8_E4M3") {
        location.dtype = DType::kF8;
        element = 1;
      } else if (dtype == "F16") {
        location.dtype = DType::kF16;
        element = 2;
      } else if (dtype == "BF16") {
        location.dtype = DType::kBF16;
        element = 2;
      } else if (dtype == "F32") {
        location.dtype = DType::kF32;
        element = 4;
      } else {
        SetError(error, "unsupported VACE tensor type " + dtype + " of " +
                            name);
        return false;
      }
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (shape == nullptr || offsets == nullptr || offsets->size() != 2) {
        SetError(error, "invalid tensor entry " + name);
        return false;
      }
      location.count = 1;
      for (const auto& dim : shape->items()) {
        location.shape.push_back(dim.as_size());
        location.count *= location.shape.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end < begin || end - begin != location.count * element ||
          8 + header_size + end > *file_size) {
        SetError(error, "invalid tensor extent " + name);
        return false;
      }
      location.offset = 8 + header_size + begin;
      location.bytes = end - begin;
      tensors[key] = std::move(location);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return false;
  }
  return true;
}

bool Vace::Impl::ReadFloats(const std::string& name, std::size_t count,
                            std::vector<float>* out, std::string* error) {
  const TensorLocation* location = Find(name);
  if (location == nullptr || location->count != count ||
      location->dtype == DType::kF8) {
    SetError(error, "missing or misshaped VACE tensor " + name);
    return false;
  }
  std::vector<std::byte> bytes(location->bytes);
  if (platform::ReadFileAt(file, bytes.data(), bytes.size(),
                           location->offset) != bytes.size()) {
    SetError(error, "short read of " + name);
    return false;
  }
  out->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t half;
    switch (location->dtype) {
    case DType::kF32:
      std::memcpy(&(*out)[i], bytes.data() + 4 * i, 4);
      break;
    case DType::kF16:
      std::memcpy(&half, bytes.data() + 2 * i, 2);
      (*out)[i] = HalfToFloat(half);
      break;
    case DType::kBF16:
      std::memcpy(&half, bytes.data() + 2 * i, 2);
      (*out)[i] = Bf16ToFloat(half);
      break;
    case DType::kF8:
      break;
    }
  }
  return true;
}

bool Vace::Impl::UploadBf16(DeviceBuffer& buffer,
                            const std::vector<float>& host,
                            std::string* error) {
  std::vector<std::uint16_t> half(host.size());
  for (std::size_t i = 0; i < host.size(); ++i)
    half[i] = HostBf16(host[i]);
  weight_bytes += half.size() * 2;
  return buffer.Allocate(half.size() * 2, error) &&
         Check(hipMemcpy(buffer.raw(), half.data(), half.size() * 2,
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Vace::Impl::UploadF32(DeviceBuffer& buffer, const std::vector<float>& host,
                           std::string* error) {
  weight_bytes += host.size() * 4;
  return buffer.Allocate(host.size() * 4, error) &&
         Check(hipMemcpy(buffer.raw(), host.data(), host.size() * 4,
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Vace::Impl::LoadVector(DeviceBuffer& buffer, const std::string& name,
                            std::size_t count, bool bf16, std::string* error) {
  std::vector<float> host;
  return ReadFloats(name, count, &host, error) &&
         (bf16 ? UploadBf16(buffer, host, error)
               : UploadF32(buffer, host, error));
}

bool Vace::Impl::LoadLinear(Linear& linear, const std::string& base, int rows,
                            int columns, bool with_bias, std::string* error) {
  const std::string name = base + ".weight";
  const TensorLocation* w = Find(name);
  const std::size_t count = static_cast<std::size_t>(rows) * columns;
  if (w == nullptr || w->count != count) {
    SetError(error, "missing or misshaped VACE tensor " + name);
    return false;
  }
  linear.rows = rows;
  linear.columns = columns;
  if (with_bias &&
      !LoadVector(linear.bias, base + ".bias", rows, true, error))
    return false;
  if (w->dtype != DType::kF8) {
    std::vector<float> host;
    return ReadFloats(name, count, &host, error) &&
           UploadBf16(linear.weight, host, error);
  }
  std::vector<float> scale;
  if (!ReadFloats(base + ".scale_weight", 1, &scale, error))
    return false;
  linear.fp8 = true;
  linear.scale = scale[0];
  std::vector<std::byte> bytes(w->bytes);
  if (platform::ReadFileAt(file, bytes.data(), bytes.size(), w->offset) !=
      bytes.size()) {
    SetError(error, "short read of " + name);
    return false;
  }
  weight_bytes += bytes.size();
  return linear.weight.Allocate(bytes.size(), error) &&
         Check(hipMemcpy(linear.weight.raw(), bytes.data(), bytes.size(),
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Vace::Impl::Unpack(const Linear& linear, std::uint16_t* target,
                        std::string* error) {
  const std::size_t count =
      static_cast<std::size_t>(linear.rows) * linear.columns;
  if (!linear.fp8)
    return Check(hipMemcpyAsync(target, linear.weight.raw(), count * 2,
                                hipMemcpyDeviceToDevice, stream),
                 "weight copy", error);
  UnpackKernel<<<Blocks(count / 4), 256, 0, stream>>>(
      static_cast<const std::uint32_t*>(linear.weight.raw()), count / 4,
      linear.scale, target);
  return true;
}

Vace::Vace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Vace::~Vace() = default;

void Vace::set_strength(float strength) {
  impl_->strength = strength;
}
int Vace::blocks() const {
  return static_cast<int>(impl_->blocks.size());
}
int Vace::dit_layer(int block) const {
  return block * impl_->stride;
}
void Vace::set_probe(bool probe) {
  impl_->probe = probe;
}
const std::vector<float>& Vace::hint_first() const {
  return impl_->hint_first;
}
std::size_t Vace::weight_bytes() const {
  return impl_->weight_bytes;
}

std::unique_ptr<Vace> Vace::Load(const std::filesystem::path& path,
                                 hipStream_t stream, std::string* error) {
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.stream = stream;
  if (!m.Index(path, error))
    return nullptr;
  int count = 0;
  while (m.Find("vace_blocks." + std::to_string(count) + ".after_proj.weight") !=
         nullptr)
    ++count;
  const TensorLocation* patch = m.Find("vace_patch_embedding.weight");
  if (count == 0 || kDitLayers % count != 0 || patch == nullptr ||
      patch->count != static_cast<std::uint64_t>(kWidth) * kPatchIn ||
      m.Find("vace_blocks.0.before_proj.weight") == nullptr) {
    SetError(error, "not a Wan 2.1 VACE 14B module (vace_blocks.*, "
                    "vace_patch_embedding [5120, 96, 1, 2, 2]): " +
                        path.string());
    return nullptr;
  }
  m.stride = kDitLayers / count;
  if (!m.gemm.Init(stream, error))
    return nullptr;
  constexpr std::size_t W = kWidth;
  {
    // Conv3d [5120, 96, 1, 2, 2] flattens to c * 4 + kh * 2 + kw.
    std::vector<float> host;
    if (!m.ReadFloats("vace_patch_embedding.weight", W * kPatchIn, &host,
                      error) ||
        !m.UploadBf16(m.patch, host, error) ||
        !m.LoadVector(m.patch_bias, "vace_patch_embedding.bias", W, true,
                      error))
      return nullptr;
  }
  m.blocks = std::vector<Block>(count);
  for (int index = 0; index < count; ++index) {
    Block& b = m.blocks[index];
    const std::string p = "vace_blocks." + std::to_string(index) + ".";
    const std::string s = p + "self_attn.";
    const std::string c = p + "cross_attn.";
    const int w = kWidth;
    std::vector<float> qkv_bias, part;
    for (const char* name : {"q", "k", "v"}) {
      if (!m.ReadFloats(s + name + ".bias", W, &part, error))
        return nullptr;
      qkv_bias.insert(qkv_bias.end(), part.begin(), part.end());
    }
    b.has_before = m.Find(p + "before_proj.weight") != nullptr;
    if (!m.UploadBf16(b.qkv_bias, qkv_bias, error) ||
        !m.LoadLinear(b.q, s + "q", w, w, false, error) ||
        !m.LoadLinear(b.k, s + "k", w, w, false, error) ||
        !m.LoadLinear(b.v, s + "v", w, w, false, error) ||
        !m.LoadLinear(b.o, s + "o", w, w, true, error) ||
        !m.LoadVector(b.norm_q, s + "norm_q.weight", W, false, error) ||
        !m.LoadVector(b.norm_k, s + "norm_k.weight", W, false, error) ||
        !m.LoadLinear(b.cross_q, c + "q", w, w, true, error) ||
        !m.LoadLinear(b.cross_k, c + "k", w, w, true, error) ||
        !m.LoadLinear(b.cross_v, c + "v", w, w, true, error) ||
        !m.LoadLinear(b.cross_o, c + "o", w, w, true, error) ||
        !m.LoadVector(b.cross_norm_q, c + "norm_q.weight", W, false, error) ||
        !m.LoadVector(b.cross_norm_k, c + "norm_k.weight", W, false, error) ||
        !m.LoadVector(b.norm3_weight, p + "norm3.weight", W, false, error) ||
        !m.LoadVector(b.norm3_bias, p + "norm3.bias", W, false, error) ||
        !m.LoadLinear(b.up, p + "ffn.0", kFeedForward, w, true, error) ||
        !m.LoadLinear(b.down, p + "ffn.2", w, kFeedForward, true, error) ||
        !m.LoadVector(b.modulation, p + "modulation", 6 * W, false, error) ||
        !m.LoadLinear(b.after, p + "after_proj", w, w, true, error) ||
        (b.has_before &&
         !m.LoadLinear(b.before, p + "before_proj", w, w, true, error)))
      return nullptr;
  }
  platform::CloseFile(m.file);
  m.file = platform::kInvalidFile;
  return std::unique_ptr<Vace>(new Vace(std::move(impl)));
}

bool Vace::SetContext(std::span<const float> context, int frames, int height,
                      int width, std::string* error) {
  Impl& m = *impl_;
  if (frames <= 0 || height <= 0 || width <= 0 || height % 2 != 0 ||
      width % 2 != 0 ||
      context.size() !=
          static_cast<std::size_t>(kContextChannels) * frames * height *
              width) {
    SetError(error, "Wan 2.1 VACE: context size does not match its shape");
    return false;
  }
  if (!m.context.Allocate(context.size() * 4, error) ||
      !Check(hipMemcpy(m.context.raw(), context.data(), context.size() * 4,
                       hipMemcpyHostToDevice),
             "context upload", error))
    return false;
  m.frames = frames;
  m.height = height;
  m.width = width;
  return true;
}

bool Vace::Begin(const float* input, int frames, int height, int width,
                 const float* temb, hipStream_t stream, std::string* error) {
  (void)input;
  (void)temb;
  (void)stream;
  Impl& m = *impl_;
  m.active = false;
  if (m.frames != 0 &&
      (frames != m.frames || height != m.height || width != m.width)) {
    SetError(error, "Wan 2.1 VACE: context and video latent differ in shape");
    return false;
  }
  return true;
}

// Cross-attention keys and values of every VACE block on the slot's
// embedded text (once per PrepareContext).
bool Vace::Impl::PrepareText(const Transformer21BlockState& s,
                             std::string* error) {
  TextCache& cache = text[s.context_slot];
  if (cache.serial == s.context_serial && cache.rows == s.text_rows)
    return true;
  const std::size_t W = kWidth, L = s.text_rows, LP = PaddedRows(s.text_rows);
  const std::size_t n = blocks.size();
  if (cache.rows != s.text_rows) {
    cache.rows = 0;
    if (!cache.key.Allocate(n * L * W * 2, error) ||
        !cache.packed.Allocate(n * LP * W * 2, error))
      return false;
  }
  if (text_reserved < s.text_rows) {
    text_reserved = 0;
    if (!text_linear.Allocate(L * W * 2, error) ||
        !text_values.Allocate(L * W * 2, error))
      return false;
    text_reserved = s.text_rows;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const Block& b = blocks[i];
    if (!Gemm(b.cross_k, s.text, text_linear.get(), s.text_rows, s.weights,
              error))
      return false;
    HeadsKernel<<<s.text_rows, kRowThreads, 0, stream>>>(
        text_linear.get(), kWidth, s.text_rows, b.cross_norm_k.f32(), nullptr,
        cache.key.get() + i * L * W);
    if (!Gemm(b.cross_v, s.text, text_linear.get(), s.text_rows, s.weights,
              error))
      return false;
    HeadsKernel<<<s.text_rows, kRowThreads, 0, stream>>>(
        text_linear.get(), kWidth, s.text_rows, nullptr, nullptr,
        text_values.get());
    LaunchPackValues(text_values.get(), cache.packed.get() + i * LP * W,
                     s.text_rows, kHeads, stream);
  }
  if (!Check(hipGetLastError(), "text context", error))
    return false;
  cache.rows = s.text_rows;
  cache.serial = s.context_serial;
  return true;
}

bool Vace::BeginBlocks(const Transformer21BlockState& s, hipStream_t stream,
                       std::string* error) {
  Impl& m = *impl_;
  m.active = false;
  if (m.frames == 0)
    return true;  // no context set: plain text-to-video
  if (s.frames != m.frames || s.height != m.height || s.width != m.width ||
      s.context_slot < 0 || s.context_slot >= Transformer21::kContextSlots ||
      s.text == nullptr || s.text_rows <= 0) {
    SetError(error, "Wan 2.1 VACE: invalid transformer state");
    return false;
  }
  const std::size_t T = s.tokens, count = T * kWidth;
  if (m.reserved < s.tokens) {
    m.reserved = 0;
    if (!m.hidden.Allocate(count * 4, error))
      return false;
    m.reserved = s.tokens;
  }
  if (!m.PrepareText(s, error))
    return false;
  // c = before_proj(vace_patch_embedding(context)) + x.
  ContextPatchKernel<<<Blocks(T * kPatchIn), 256, 0, stream>>>(
      m.context.f32(), m.frames, m.height, m.width, s.linear);
  if (!m.gemm.Run(m.patch.get(), m.patch_bias.get(), s.linear, s.normed,
                  s.tokens, kWidth, kPatchIn, error))
    return false;
  const Block& first = m.blocks.front();
  if (first.has_before) {
    if (!m.Gemm(first.before, s.normed, s.linear, s.tokens, s.weights, error))
      return false;
    WidenAddKernel<<<Blocks(count), 256, 0, stream>>>(s.linear, s.hidden,
                                                      count, m.hidden.f32());
  } else {
    WidenAddKernel<<<Blocks(count), 256, 0, stream>>>(s.normed, s.hidden,
                                                      count, m.hidden.f32());
  }
  m.state = s;
  m.state.hidden = nullptr;  // only valid here
  m.active = true;
  return Check(hipGetLastError(), "begin", error);
}

bool Vace::Impl::RunBlock(int n, std::string* error) {
  const Transformer21BlockState& s = state;
  const Block& b = blocks[n];
  const TextCache& context = text[s.context_slot];
  const std::size_t W = kWidth, T = s.tokens, count = T * W;
  const std::size_t L = context.rows, LP = PaddedRows(context.rows);
  const unsigned rows = static_cast<unsigned>(T);
  const int tokens = s.tokens;
  const float* proj = s.modulation;
  const auto* rope = reinterpret_cast<const float2*>(s.rope);
  float* c = hidden.f32();
  hipStream_t st = stream;
  const auto modulated = [&](int shift, int scale) {
    NormArgs a;
    a.shift_table = b.modulation.f32() + shift * W;
    a.scale_table = b.modulation.f32() + scale * W;
    a.shift_proj = proj + shift * W;
    a.scale_proj = proj + scale * W;
    return a;
  };
  // 1. Self-attention (fused q | k | v GEMM).
  LayerNormKernel<<<rows, kRowThreads, 0, st>>>(c, modulated(0, 1), s.normed);
  if (!Unpack(b.q, s.weights, error) ||
      !Unpack(b.k, s.weights + W * W, error) ||
      !Unpack(b.v, s.weights + 2 * W * W, error) ||
      !gemm.Run(s.weights, b.qkv_bias.get(), s.normed, s.linear, tokens,
                3 * kWidth, kWidth, error))
    return false;
  HeadsKernel<<<rows, kRowThreads, 0, st>>>(s.linear, 3 * kWidth, tokens,
                                            b.norm_q.f32(), rope, s.query);
  HeadsKernel<<<rows, kRowThreads, 0, st>>>(s.linear + W, 3 * kWidth, tokens,
                                            b.norm_k.f32(), rope, s.key);
  HeadsKernel<<<rows, kRowThreads, 0, st>>>(s.linear + 2 * W, 3 * kWidth,
                                            tokens, nullptr, nullptr, s.value);
  LaunchPackValues(s.value, s.packed, tokens, kHeads, st);
  LaunchAttention(s.query, s.key, s.packed, s.attention, tokens, tokens,
                  kHeads, st);
  if (!Gemm(b.o, s.attention, s.linear, tokens, s.weights, error))
    return false;
  ResidualKernel<<<Blocks(count), 256, 0, st>>>(
      s.linear, b.modulation.f32() + 2 * W, proj + 2 * W, count, c);

  // 2. Cross-attention to the text (norm3 with affine).
  NormArgs cross_norm;
  cross_norm.weight = b.norm3_weight.f32();
  cross_norm.bias = b.norm3_bias.f32();
  LayerNormKernel<<<rows, kRowThreads, 0, st>>>(c, cross_norm, s.normed);
  if (!Gemm(b.cross_q, s.normed, s.linear, tokens, s.weights, error))
    return false;
  HeadsKernel<<<rows, kRowThreads, 0, st>>>(s.linear, kWidth, tokens,
                                            b.cross_norm_q.f32(), nullptr,
                                            s.query);
  LaunchAttention(s.query, context.key.get() + n * L * W,
                  context.packed.get() + n * LP * W, s.attention, tokens,
                  static_cast<std::uint32_t>(L), kHeads, st);
  if (!Gemm(b.cross_o, s.attention, s.linear, tokens, s.weights, error))
    return false;
  ResidualKernel<<<Blocks(count), 256, 0, st>>>(s.linear, nullptr, nullptr,
                                                count, c);

  // 3. Feed-forward (GELU tanh) in chunks of the workspace's FFN rows.
  LayerNormKernel<<<rows, kRowThreads, 0, st>>>(c, modulated(3, 4), s.normed);
  std::uint16_t* down = s.weights + static_cast<std::size_t>(kFeedForward) * W;
  if (!Unpack(b.up, s.weights, error) || !Unpack(b.down, down, error))
    return false;
  for (int first = 0; first < tokens; first += s.ffn_rows) {
    const int chunk = std::min(s.ffn_rows, tokens - first);
    const std::size_t up_count = static_cast<std::size_t>(chunk) * kFeedForward;
    if (!gemm.Run(s.weights, b.up.bias.get(), s.normed + first * W, s.ffn,
                  chunk, kFeedForward, kWidth, error))
      return false;
    GeluKernel<<<Blocks(up_count), 256, 0, st>>>(s.ffn, up_count);
    if (!gemm.Run(down, b.down.bias.get(), s.ffn, s.linear + first * W, chunk,
                  kWidth, kFeedForward, error))
      return false;
  }
  ResidualKernel<<<Blocks(count), 256, 0, st>>>(
      s.linear, b.modulation.f32() + 5 * W, proj + 5 * W, count, c);
  return Check(hipGetLastError(), "block", error);
}

bool Vace::AfterBlock(int layer, float* dit_hidden, hipStream_t stream,
                      std::string* error) {
  Impl& m = *impl_;
  if (!m.active || layer % m.stride != 0)
    return true;
  const int n = layer / m.stride;
  if (n >= static_cast<int>(m.blocks.size()))
    return true;
  const Transformer21BlockState& s = m.state;
  const std::size_t count = static_cast<std::size_t>(s.tokens) * kWidth;
  if (!m.RunBlock(n, error))
    return false;
  // Hint: after_proj(c) (c rounded to BF16 for the GEMM), x += strength * it.
  NarrowKernel<<<Blocks(count), 256, 0, stream>>>(m.hidden.f32(), count,
                                                  s.normed);
  if (!m.Gemm(m.blocks[n].after, s.normed, s.linear, s.tokens, s.weights,
              error))
    return false;
  ScaledAddKernel<<<Blocks(count), 256, 0, stream>>>(s.linear, m.strength,
                                                     count, dit_hidden);
  if (m.probe && n == 0) {
    std::vector<std::uint16_t> half(count);
    if (!Check(hipMemcpyAsync(half.data(), s.linear, count * 2,
                              hipMemcpyDeviceToHost, stream),
               "probe", error) ||
        !Check(hipStreamSynchronize(stream), "probe", error))
      return false;
    m.hint_first.resize(count);
    for (std::size_t i = 0; i < count; ++i)
      m.hint_first[i] = Bf16ToFloat(half[i]);
  }
  return Check(hipGetLastError(), "hint", error);
}

}  // namespace gufo::models::wan22
