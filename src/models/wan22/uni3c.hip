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
#include "src/models/wan22/uni3c.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr int kLayers = 20;
constexpr int kDim = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;
constexpr int kPaddedHead = 128;  // the attention kernel's head width
constexpr int kFfn = 4096;
constexpr int kOut = 5120;
constexpr int kPatchIn = 36 * 4;
constexpr int kRopePairs = kHeadDim / 2;  // 12 time + 10 height + 10 width
constexpr int kThreads = 256;
constexpr int kRowValues = kDim / kThreads;  // 4

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

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

// patches[t, c * 4 + kh * 2 + kw] of the control input: channels 0..19 from
// the DiT input [36, F, H, W], 20..35 from the guide latent [16, F, H, W].
__global__ void ControlPatchKernel(const float* input, const float* guide,
                                   int frames, int height, int width,
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
  const std::size_t pixel =
      (static_cast<std::size_t>(f) * height + 2 * h + kh) * width + 2 * w + kw;
  const std::size_t plane = static_cast<std::size_t>(frames) * height * width;
  const float v = c < 20 ? input[c * plane + pixel] : guide[(c - 20) * plane + pixel];
  patches[index] = ToBf16(v);
}

__global__ void NarrowKernel(const float* x, std::size_t count,
                             std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(x[i]);
}

__global__ void WidenKernel(const std::uint16_t* x, std::size_t count,
                            float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = FromBf16(x[i]);
}

// out[r] = W[r, :] . silu(in) + b[r], BF16 weights, FP32 everything else.
__global__ void ModulationKernel(const std::uint16_t* weight,
                                 const float* bias, const float* in,
                                 int columns, float* out) {
  __shared__ float shared[kThreads];
  const int r = blockIdx.x;
  const std::uint16_t* w = weight + static_cast<std::size_t>(r) * columns;
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kThreads) {
    const float v = in[c];
    sum += FromBf16(w[c]) * (v / (1.0F + expf(-v)));
  }
  sum = BlockSum(sum, shared);
  if (threadIdx.x == 0)
    out[r] = sum + bias[r];
}

// bf16(layer_norm(x) * w + b) * (1 + scale) + shift, eps 1e-5, rows of 1024;
// mod = [shift | scale | gate].
__global__ void NormKernel(const float* hidden, const float* weight,
                           const float* bias, const float* mod,
                           std::uint16_t* out) {
  __shared__ float shared[kThreads];
  const std::size_t row = blockIdx.x;
  const float* x = hidden + row * kDim;
  float values[kRowValues];
  float sum = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] = x[threadIdx.x + i * kThreads];
    sum += values[i];
  }
  const float mean = BlockSum(sum, shared) / kDim;
  float squares = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] -= mean;
    squares += values[i] * values[i];
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / kDim + 1e-5F);
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    const int c = threadIdx.x + i * kThreads;
    const float y = values[i] * rstd * weight[c] + bias[c];
    out[row * kDim + c] = ToBf16(y * (1.0F + mod[kDim + c]) + mod[c]);
  }
}

// One of q, k, v (row stride `stride`): optional RMS norm across 1024 (eps
// 1e-6) and RoPE, times `scale`, written head-major [16][rows][128] with the
// upper 64 dims zero.
__global__ void HeadsKernel(const std::uint16_t* linear, int stride, int rows,
                            const float* norm_weight, const float2* rope,
                            float scale, std::uint16_t* out) {
  __shared__ float shared[kThreads];
  const std::size_t row = blockIdx.x;
  const std::uint16_t* x = linear + row * stride;
  // Thread t owns the pair (2t, 2t + 1) and the pair 512 + 2t.
  float values[kRowValues];
  float squares = 0;
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float v = FromBf16(x[2 * threadIdx.x + 512 * p + j]);
      values[2 * p + j] = v;
      squares += v * v;
    }
  float rstd = 1.0F;
  if (norm_weight != nullptr)
    rstd = rsqrtf(BlockSum(squares, shared) / kDim + 1e-6F);
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int c = 2 * threadIdx.x + 512 * p;
    float a = values[2 * p], b = values[2 * p + 1];
    if (norm_weight != nullptr) {
      a = a * rstd * norm_weight[c];
      b = b * rstd * norm_weight[c + 1];
    }
    const int head = c / kHeadDim, d = c % kHeadDim;
    if (rope != nullptr) {
      const float2 cs = rope[row * kRopePairs + d / 2];
      const float ra = a * cs.x - b * cs.y;
      const float rb = a * cs.y + b * cs.x;
      a = ra;
      b = rb;
    }
    std::uint16_t* target =
        out + (static_cast<std::size_t>(head) * rows + row) * kPaddedHead + d;
    target[0] = ToBf16(a * scale);
    target[1] = ToBf16(b * scale);
    target[kHeadDim] = 0;
    target[kHeadDim + 1] = 0;
  }
}

// [rows][16 * 128] attention output -> [rows][1024] (first 64 of each head).
__global__ void CompactKernel(const std::uint16_t* in, std::size_t rows,
                              std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * kDim)
    return;
  const std::size_t row = i / kDim;
  const int c = static_cast<int>(i % kDim);
  out[i] = in[row * kHeads * kPaddedHead + (c / kHeadDim) * kPaddedHead +
              c % kHeadDim];
}

__global__ void GeluKernel(std::uint16_t* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = FromBf16(x[i]);
  const float inner = 0.7978845608028654F * (v + 0.044715F * v * v * v);
  x[i] = ToBf16(0.5F * v * (1.0F + tanhf(inner)));
}

// hidden[i] += y[i] * gate[i % width] (gate null: * scale).
__global__ void AddKernel(const std::uint16_t* y, const float* gate,
                          float scale, int width, std::size_t count,
                          float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float g = gate != nullptr ? gate[i % width] : scale;
  hidden[i] += FromBf16(y[i]) * g;
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("Uni3C ") + operation + ": " +
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

std::uint16_t HostBf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
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

struct Layer {
  DeviceBuffer mod1, mod1_bias, mod2, mod2_bias;  // BF16 [3072, 5120], FP32
  DeviceBuffer norm1_weight, norm1_bias, norm2_weight, norm2_bias;  // FP32
  DeviceBuffer qkv, qkv_bias;                      // BF16 [3072, 1024]
  DeviceBuffer norm_q, norm_k;                     // FP32
  DeviceBuffer out, out_bias, up, up_bias, down, down_bias;  // BF16
  DeviceBuffer proj, proj_bias;                    // BF16 [5120, 1024]
};

}  // namespace

struct Uni3C::Impl {
  bool Reserve(int tokens, std::string* error);

  std::vector<Layer> layers;
  DeviceBuffer patch, patch_bias, proj_in, proj_in_bias;  // BF16
  std::size_t weight_bytes{0};
  hipStream_t stream{nullptr};
  LinearGemm gemm;
  float strength{1.0F};
  bool probe{false};
  std::vector<float> residual_first, residual_last;

  DeviceBuffer guide;  // FP32 [16, F, H, W]
  int guide_frames{0}, guide_height{0}, guide_width{0};
  int tokens{0}, reserved{0};
  bool active{false};
  int rope_frames{0}, rope_height{0}, rope_width{0};
  DeviceBuffer patches, wide, hidden, normed, acc, query, key, value, packed,
      attention, compact, ffn, residual, rope, mods;
};

bool Uni3C::Impl::Reserve(int count, std::string* error) {
  if (count <= reserved)
    return true;
  const std::size_t T = count, TP = PaddedRows(count);
  const std::size_t P = kHeads * kPaddedHead;
  reserved = 0;
  rope_frames = 0;
  if (!patches.Allocate(T * kPatchIn * 2, error) ||
      !wide.Allocate(T * kOut * 2, error) ||
      !hidden.Allocate(T * kDim * 4, error) ||
      !normed.Allocate(T * kDim * 2, error) ||
      !acc.Allocate(T * 3 * kDim * 2, error) ||
      !query.Allocate(T * P * 2, error) || !key.Allocate(T * P * 2, error) ||
      !value.Allocate(T * P * 2, error) || !packed.Allocate(TP * P * 2, error) ||
      !attention.Allocate(T * P * 2, error) ||
      !compact.Allocate(T * kDim * 2, error) ||
      !ffn.Allocate(T * kFfn * 2, error) ||
      !residual.Allocate(T * kOut * 2, error) ||
      !rope.Allocate(T * kRopePairs * 8, error))
    return false;
  reserved = count;
  return true;
}

Uni3C::Uni3C(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Uni3C::~Uni3C() = default;

void Uni3C::set_strength(float strength) {
  impl_->strength = strength;
}
void Uni3C::set_probe(bool probe) {
  impl_->probe = probe;
}
const std::vector<float>& Uni3C::residual_first() const {
  return impl_->residual_first;
}
const std::vector<float>& Uni3C::residual_last() const {
  return impl_->residual_last;
}
std::size_t Uni3C::weight_bytes() const {
  return impl_->weight_bytes;
}

std::unique_ptr<Uni3C> Uni3C::Load(const std::filesystem::path& path,
                                   hipStream_t stream, std::string* error) {
  std::string open_error;
  auto file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return nullptr;
  }
  struct Closer {
    platform::PlatformFile file;
    ~Closer() { platform::CloseFile(file); }
  } closer{file};
  const auto file_size = platform::FileSizeBytes(file);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (32U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return nullptr;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file, text.data(), text.size(), 8) != text.size()) {
    SetError(error, "short read of " + path.string());
    return nullptr;
  }
  struct Location {
    std::uint64_t offset{0}, count{0};
    bool f32{false};
  };
  std::map<std::string, Location, std::less<>> tensors;
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const std::string dtype = entry.member_str("dtype");
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if ((dtype != "F16" && dtype != "F32") || shape == nullptr ||
          offsets == nullptr || offsets->size() != 2) {
        SetError(error, "unsupported Uni3C tensor " + name);
        return nullptr;
      }
      std::uint64_t count = 1;
      for (const auto& dim : shape->items())
        count *= dim.as_size();
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      const std::uint64_t element = dtype == "F32" ? 4 : 2;
      if (end - begin != count * element || 8 + header_size + end > *file_size) {
        SetError(error, "invalid Uni3C tensor extent " + name);
        return nullptr;
      }
      tensors[name] = {8 + header_size + begin, count, dtype == "F32"};
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.stream = stream;
  if (!m.gemm.Init(stream, error))
    return nullptr;
  const auto read = [&](const std::string& name, std::uint64_t count,
                        std::vector<float>* out) {
    const auto it = tensors.find(name);
    if (it == tensors.end() || it->second.count != count) {
      SetError(error, "missing or misshaped Uni3C tensor " + name);
      return false;
    }
    const std::size_t bytes = count * (it->second.f32 ? 4 : 2);
    std::vector<std::byte> raw(bytes);
    if (platform::ReadFileAt(file, raw.data(), bytes, it->second.offset) !=
        bytes) {
      SetError(error, "short read of Uni3C tensor " + name);
      return false;
    }
    out->resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (it->second.f32) {
        std::memcpy(&(*out)[i], raw.data() + 4 * i, 4);
      } else {
        std::uint16_t half;
        std::memcpy(&half, raw.data() + 2 * i, 2);
        (*out)[i] = HalfToFloat(half);
      }
    }
    return true;
  };
  const auto upload = [&](DeviceBuffer& buffer, const std::vector<float>& host,
                          bool bf16) {
    if (bf16) {
      std::vector<std::uint16_t> half(host.size());
      for (std::size_t i = 0; i < host.size(); ++i)
        half[i] = HostBf16(host[i]);
      m.weight_bytes += half.size() * 2;
      return buffer.Allocate(half.size() * 2, error) &&
             Check(hipMemcpy(buffer.raw(), half.data(), half.size() * 2,
                             hipMemcpyHostToDevice),
                   "weight upload", error);
    }
    m.weight_bytes += host.size() * 4;
    return buffer.Allocate(host.size() * 4, error) &&
           Check(hipMemcpy(buffer.raw(), host.data(), host.size() * 4,
                           hipMemcpyHostToDevice),
                 "weight upload", error);
  };
  const auto load = [&](DeviceBuffer& buffer, const std::string& name,
                        std::uint64_t count, bool bf16) {
    std::vector<float> host;
    return read(name, count, &host) && upload(buffer, host, bf16);
  };
  if (!load(m.patch, "controlnet_patch_embedding.weight",
            std::uint64_t{kOut} * kPatchIn, true) ||
      !load(m.patch_bias, "controlnet_patch_embedding.bias", kOut, true) ||
      !load(m.proj_in, "proj_in.weight", std::uint64_t{kDim} * kOut, true) ||
      !load(m.proj_in_bias, "proj_in.bias", kDim, true))
    return nullptr;
  m.layers = std::vector<Layer>(kLayers);
  for (int i = 0; i < kLayers; ++i) {
    Layer& l = m.layers[i];
    const std::string p = "controlnet_blocks." + std::to_string(i) + ".";
    const std::string a = p + "self_attn.";
    std::vector<float> qkv, qkv_bias, part;
    for (const char* name : {"to_q", "to_k", "to_v"}) {
      if (!read(a + name + ".weight", std::uint64_t{kDim} * kDim, &part))
        return nullptr;
      qkv.insert(qkv.end(), part.begin(), part.end());
      if (!read(a + name + ".bias", kDim, &part))
        return nullptr;
      qkv_bias.insert(qkv_bias.end(), part.begin(), part.end());
    }
    if (!upload(l.qkv, qkv, true) || !upload(l.qkv_bias, qkv_bias, true) ||
        !load(l.mod1, p + "norm1.linear.weight", std::uint64_t{3 * kDim} * kOut,
              true) ||
        !load(l.mod1_bias, p + "norm1.linear.bias", 3 * kDim, false) ||
        !load(l.mod2, p + "norm2.linear.weight", std::uint64_t{3 * kDim} * kOut,
              true) ||
        !load(l.mod2_bias, p + "norm2.linear.bias", 3 * kDim, false) ||
        !load(l.norm1_weight, p + "norm1.norm.weight", kDim, false) ||
        !load(l.norm1_bias, p + "norm1.norm.bias", kDim, false) ||
        !load(l.norm2_weight, p + "norm2.norm.weight", kDim, false) ||
        !load(l.norm2_bias, p + "norm2.norm.bias", kDim, false) ||
        !load(l.norm_q, a + "norm_q.weight", kDim, false) ||
        !load(l.norm_k, a + "norm_k.weight", kDim, false) ||
        !load(l.out, a + "to_out.0.weight", std::uint64_t{kDim} * kDim, true) ||
        !load(l.out_bias, a + "to_out.0.bias", kDim, true) ||
        !load(l.up, p + "ffn.0.weight", std::uint64_t{kFfn} * kDim, true) ||
        !load(l.up_bias, p + "ffn.0.bias", kFfn, true) ||
        !load(l.down, p + "ffn.2.weight", std::uint64_t{kDim} * kFfn, true) ||
        !load(l.down_bias, p + "ffn.2.bias", kDim, true) ||
        !load(l.proj, "proj_out." + std::to_string(i) + ".weight",
              std::uint64_t{kOut} * kDim, true) ||
        !load(l.proj_bias, "proj_out." + std::to_string(i) + ".bias", kOut,
              true))
      return nullptr;
  }
  if (!m.mods.Allocate(std::size_t{kLayers} * 2 * 3 * kDim * 4, error))
    return nullptr;
  return std::unique_ptr<Uni3C>(new Uni3C(std::move(impl)));
}

bool Uni3C::SetGuide(std::span<const float> latent, int frames, int height,
                     int width, std::string* error) {
  Impl& m = *impl_;
  if (latent.size() != std::size_t{16} * frames * height * width) {
    SetError(error, "Uni3C: guide latent size does not match its shape");
    return false;
  }
  if (!m.guide.Allocate(latent.size() * 4, error) ||
      !Check(hipMemcpy(m.guide.raw(), latent.data(), latent.size() * 4,
                       hipMemcpyHostToDevice),
             "guide upload", error))
    return false;
  m.guide_frames = frames;
  m.guide_height = height;
  m.guide_width = width;
  return true;
}

bool Uni3C::Begin(const float* input, int frames, int height, int width,
                  const float* temb, hipStream_t s, std::string* error) {
  Impl& m = *impl_;
  m.active = false;
  if (m.guide_frames == 0)
    return true;
  if (frames != m.guide_frames || height != m.guide_height ||
      width != m.guide_width) {
    SetError(error, "Uni3C: guide latent and video latent differ in shape");
    return false;
  }
  const int ph = height / 2, pw = width / 2;
  const int tokens = frames * ph * pw;
  if (!m.Reserve(tokens, error))
    return false;
  m.tokens = tokens;
  if (m.rope_frames != frames || m.rope_height != height ||
      m.rope_width != width) {
    // EmbedND(64, theta 1e4, axes [24, 20, 20]) at integer (t, h, w).
    std::vector<float> rope(static_cast<std::size_t>(tokens) * kRopePairs * 2);
    for (int t = 0; t < tokens; ++t) {
      const int position[3] = {t / (ph * pw), (t / pw) % ph, t % pw};
      for (int j = 0; j < kRopePairs; ++j) {
        const int axis = j < 12 ? 0 : (j < 22 ? 1 : 2);
        const int k = axis == 0 ? j : (axis == 1 ? j - 12 : j - 22);
        const int dims = axis == 0 ? 24 : 20;
        const double omega =
            1.0 / std::pow(10000.0, static_cast<double>(2 * k) / dims);
        const double angle = position[axis] * omega;
        rope[(static_cast<std::size_t>(t) * kRopePairs + j) * 2] =
            static_cast<float>(std::cos(angle));
        rope[(static_cast<std::size_t>(t) * kRopePairs + j) * 2 + 1] =
            static_cast<float>(std::sin(angle));
      }
    }
    if (!Check(hipMemcpyAsync(m.rope.raw(), rope.data(), rope.size() * 4,
                              hipMemcpyHostToDevice, s),
               "rope upload", error) ||
        !Check(hipStreamSynchronize(s), "rope upload", error))
      return false;
    m.rope_frames = frames;
    m.rope_height = height;
    m.rope_width = width;
  }
  const std::size_t T = tokens;
  ControlPatchKernel<<<Blocks(T * kPatchIn), kThreads, 0, s>>>(
      input, m.guide.f32(), frames, height, width, m.patches.get());
  if (!m.gemm.Run(m.patch.get(), m.patch_bias.get(), m.patches.get(),
                  m.wide.get(), tokens, kOut, kPatchIn, error) ||
      !m.gemm.Run(m.proj_in.get(), m.proj_in_bias.get(), m.wide.get(),
                  m.normed.get(), tokens, kDim, kOut, error))
    return false;
  WidenKernel<<<Blocks(T * kDim), kThreads, 0, s>>>(m.normed.get(), T * kDim,
                                                    m.hidden.f32());
  for (int i = 0; i < kLayers; ++i) {
    float* mods = m.mods.f32() + static_cast<std::size_t>(i) * 2 * 3 * kDim;
    ModulationKernel<<<3 * kDim, kThreads, 0, s>>>(
        m.layers[i].mod1.get(), m.layers[i].mod1_bias.f32(), temb, kOut, mods);
    ModulationKernel<<<3 * kDim, kThreads, 0, s>>>(
        m.layers[i].mod2.get(), m.layers[i].mod2_bias.f32(), temb, kOut,
        mods + 3 * kDim);
  }
  m.active = true;
  return Check(hipGetLastError(), "begin", error);
}

bool Uni3C::AfterBlock(int layer, float* dit_hidden, hipStream_t s,
                       std::string* error) {
  Impl& m = *impl_;
  if (!m.active || layer >= kLayers)
    return true;
  const Layer& l = m.layers[layer];
  const int tokens = m.tokens;
  const std::size_t T = tokens;
  const unsigned rows = static_cast<unsigned>(T);
  const float* mods = m.mods.f32() + static_cast<std::size_t>(layer) * 6 * kDim;
  const auto* rope = reinterpret_cast<const float2*>(m.rope.f32());
  // 1. Self-attention.
  NormKernel<<<rows, kThreads, 0, s>>>(m.hidden.f32(), l.norm1_weight.f32(),
                                       l.norm1_bias.f32(), mods,
                                       m.normed.get());
  if (!m.gemm.Run(l.qkv.get(), l.qkv_bias.get(), m.normed.get(), m.acc.get(),
                  tokens, 3 * kDim, kDim, error))
    return false;
  // The kernel scales by 1/sqrt(128); the heads need 1/sqrt(64).
  HeadsKernel<<<rows, kThreads, 0, s>>>(m.acc.get(), 3 * kDim, tokens,
                                        l.norm_q.f32(), rope,
                                        1.4142135623730951F, m.query.get());
  HeadsKernel<<<rows, kThreads, 0, s>>>(m.acc.get() + kDim, 3 * kDim, tokens,
                                        l.norm_k.f32(), rope, 1.0F,
                                        m.key.get());
  HeadsKernel<<<rows, kThreads, 0, s>>>(m.acc.get() + 2 * kDim, 3 * kDim,
                                        tokens, nullptr, nullptr, 1.0F,
                                        m.value.get());
  LaunchPackValues(m.value.get(), m.packed.get(), tokens, kHeads, s);
  LaunchAttention(m.query.get(), m.key.get(), m.packed.get(),
                  m.attention.get(), tokens, tokens, kHeads, s);
  CompactKernel<<<Blocks(T * kDim), kThreads, 0, s>>>(m.attention.get(), T,
                                                      m.compact.get());
  if (!m.gemm.Run(l.out.get(), l.out_bias.get(), m.compact.get(),
                  m.normed.get(), tokens, kDim, kDim, error))
    return false;
  AddKernel<<<Blocks(T * kDim), kThreads, 0, s>>>(
      m.normed.get(), mods + 2 * kDim, 0, kDim, T * kDim, m.hidden.f32());
  // 2. Feed-forward.
  NormKernel<<<rows, kThreads, 0, s>>>(m.hidden.f32(), l.norm2_weight.f32(),
                                       l.norm2_bias.f32(), mods + 3 * kDim,
                                       m.normed.get());
  if (!m.gemm.Run(l.up.get(), l.up_bias.get(), m.normed.get(), m.ffn.get(),
                  tokens, kFfn, kDim, error))
    return false;
  GeluKernel<<<Blocks(T * kFfn), kThreads, 0, s>>>(m.ffn.get(), T * kFfn);
  if (!m.gemm.Run(l.down.get(), l.down_bias.get(), m.ffn.get(),
                  m.normed.get(), tokens, kDim, kFfn, error))
    return false;
  AddKernel<<<Blocks(T * kDim), kThreads, 0, s>>>(
      m.normed.get(), mods + 5 * kDim, 0, kDim, T * kDim, m.hidden.f32());
  // 3. Residual into the DiT: hidden += strength * proj_out[i](state), the
  // control state rounded to BF16 for the GEMM.
  NarrowKernel<<<Blocks(T * kDim), kThreads, 0, s>>>(m.hidden.f32(), T * kDim,
                                                     m.compact.get());
  if (!m.gemm.Run(l.proj.get(), l.proj_bias.get(), m.compact.get(),
                  m.residual.get(), tokens, kOut, kDim, error))
    return false;
  AddKernel<<<Blocks(T * kOut), kThreads, 0, s>>>(
      m.residual.get(), nullptr, m.strength, kOut, T * kOut, dit_hidden);
  if (m.probe && (layer == 0 || layer == kLayers - 1)) {
    std::vector<std::uint16_t> half(T * kOut);
    if (!Check(hipMemcpyAsync(half.data(), m.residual.raw(), half.size() * 2,
                              hipMemcpyDeviceToHost, s),
               "probe", error) ||
        !Check(hipStreamSynchronize(s), "probe", error))
      return false;
    auto& target = layer == 0 ? m.residual_first : m.residual_last;
    target.resize(half.size());
    for (std::size_t i = 0; i < half.size(); ++i)
      target[i] = Bf16ToFloat(half[i]);
  }
  return Check(hipGetLastError(), "block", error);
}

}  // namespace gufo::models::wan22
