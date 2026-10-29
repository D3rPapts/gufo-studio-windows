#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/attention.hip.hpp"
#include "src/models/wan22/gemm.hpp"
#include "src/models/wan22/transformer21.hpp"

// Wan 2.1 I2V 14B. Kernels follow transformer.hip (TI2V-5B) at width 5120;
// see transformer21.hpp for the weight handling.

namespace gufo::models::wan22 {
namespace {

constexpr int kLayers = 40;
constexpr int kHeads = 40;
constexpr int kHeadDim = 128;
constexpr int kWidth = Transformer21::kModelWidth;
constexpr int kFeedForward = 13824;
constexpr int kFeedForwardChunk = 8192;
constexpr int kPatchIn = Transformer21::kInChannels * 4;    // 144
constexpr int kPatchOut = Transformer21::kOutChannels * 4;  // 64
constexpr int kFrequencies = 256;
constexpr int kModulation = 6 * kWidth;
constexpr int kImageRows = Transformer21::kImageTokens;
constexpr int kImageWidth = Transformer21::kImageWidth;
constexpr int kRopePairs = kHeadDim / 2;
constexpr int kRopeTimePairs = 22;
constexpr int kRopeSpacePairs = 21;
constexpr float kEpsilon = 1e-6F;
constexpr std::size_t kStagingBytes = 32U << 20U;
constexpr std::size_t kSector = 4096;
constexpr std::size_t kStagingSlack = 2U * kSector;
constexpr int kRowThreads = 256;
constexpr int kRowValues = kWidth / kRowThreads;  // 20

static_assert(kWidth == kHeads * kHeadDim);
static_assert(kWidth % (2 * kRowThreads) == 0);

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

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0F + expf(-x));
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

__global__ void ConvertKernel(const float* input, std::size_t count,
                              std::uint16_t* output) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = ToBf16(input[i]);
}

// patches[t, c * 4 + kh * 2 + kw] = x[c, f, 2h + kh, 2w + kw], BF16.
__global__ void PatchKernel(const float* latent, int frames, int height,
                            int width, std::uint16_t* patches) {
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
      ToBf16(latent[((static_cast<std::size_t>(c) * frames + f) * height +
                     2 * h + kh) *
                        width +
                    2 * w + kw]);
}

template<bool kExact>
__global__ void GeluKernel(std::uint16_t* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = FromBf16(x[i]);
  x[i] = ToBf16(kExact ? 0.5F * v * (1.0F + erff(v * 0.7071067811865476F))
                       : GeluTanh(v));
}

__global__ void AddKernel(std::uint16_t* out, const std::uint16_t* a,
                          std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(FromBf16(out[i]) + FromBf16(a[i]));
}

__global__ void SinusoidKernel(float timestep, float* out) {
  const int i = threadIdx.x;  // 0..127
  constexpr int kHalf = kFrequencies / 2;
  const float exponent =
      static_cast<float>(i) * -9.210340371976184F / static_cast<float>(kHalf);
  const float angle = timestep * expf(exponent);
  out[i] = cosf(angle);
  out[kHalf + i] = sinf(angle);
}

// out[r] = W[r, :] . act(in) + b[r] (FP32 or BF16 weights, FP32 bias).
template<typename Weight, bool kSiluInput>
__global__ void MatVecKernel(const Weight* weight, const float* bias,
                             const float* in, int columns, float* out) {
  __shared__ float shared[kRowThreads];
  const int r = blockIdx.x;
  const Weight* w = weight + static_cast<std::size_t>(r) * columns;
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kRowThreads) {
    float v = in[c];
    if constexpr (kSiluInput)
      v = Silu(v);
    if constexpr (std::is_same_v<Weight, float>)
      sum += w[c] * v;
    else
      sum += FromBf16(w[c]) * v;
  }
  sum = BlockSum(sum, shared);
  if (threadIdx.x == 0)
    out[r] = sum + bias[r];
}

// FP32 layer norm of a [rows, width] FP32 (or BF16) input, then
//   out = bf16(y * (1 + (table_scale + proj_scale)) + table_shift +
//   proj_shift) when `scale_table` is set, or bf16(y * weight + bias) when
//   `weight` is set, else bf16(y).
struct NormArgs {
  const float* shift_table{nullptr};
  const float* scale_table{nullptr};
  const float* shift_proj{nullptr};
  const float* scale_proj{nullptr};
  const float* weight{nullptr};
  const float* bias{nullptr};
  float epsilon{kEpsilon};
};

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
  const float rstd = rsqrtf(BlockSum(squares, shared) / kWidth + a.epsilon);
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

// Layer norm with affine of BF16 rows of any width (the image embedding).
__global__ void LayerNormAnyKernel(const std::uint16_t* x, int width,
                                   const float* weight, const float* bias,
                                   float epsilon, std::uint16_t* out) {
  __shared__ float shared[kRowThreads];
  const std::uint16_t* row = x + static_cast<std::size_t>(blockIdx.x) * width;
  float sum = 0;
  for (int c = threadIdx.x; c < width; c += kRowThreads)
    sum += FromBf16(row[c]);
  const float mean = BlockSum(sum, shared) / width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += kRowThreads) {
    const float d = FromBf16(row[c]) - mean;
    squares += d * d;
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / width + epsilon);
  std::uint16_t* target = out + static_cast<std::size_t>(blockIdx.x) * width;
  for (int c = threadIdx.x; c < width; c += kRowThreads)
    target[c] = ToBf16((FromBf16(row[c]) - mean) * rstd * weight[c] + bias[c]);
}

// FP32 input variant of the above (the CLIP states).
__global__ void LayerNormAnyF32Kernel(const float* x, int width,
                                      const float* weight, const float* bias,
                                      float epsilon, std::uint16_t* out) {
  __shared__ float shared[kRowThreads];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
  float sum = 0;
  for (int c = threadIdx.x; c < width; c += kRowThreads)
    sum += row[c];
  const float mean = BlockSum(sum, shared) / width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += kRowThreads) {
    const float d = row[c] - mean;
    squares += d * d;
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / width + epsilon);
  std::uint16_t* target = out + static_cast<std::size_t>(blockIdx.x) * width;
  for (int c = threadIdx.x; c < width; c += kRowThreads)
    target[c] = ToBf16((row[c] - mean) * rstd * weight[c] + bias[c]);
}

// One projection (q, k or v) of one row of BF16 linear outputs, optionally RMS
// normalized across all heads (FP32 weight) and rotated, written head-major
// [heads][rows][128].
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

// noise[c, f, 2h + kh, 2w + kw] = linear[t, (kh * 2 + kw) * 16 + c].
__global__ void UnpatchKernel(const std::uint16_t* linear, int frames,
                              int height, int width, float* noise) {
  constexpr int kChannels = Transformer21::kOutChannels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ph = height / 2, pw = width / 2;
  const std::size_t tokens = static_cast<std::size_t>(frames) * ph * pw;
  if (index >= tokens * kPatchOut)
    return;
  const std::size_t token = index / kPatchOut;
  const int column = static_cast<int>(index % kPatchOut);
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
  SetError(error, std::string("Wan 2.1 transformer ") + operation + ": " +
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
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
  }
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
  void* raw() const { return data_; }

private:
  void* data_{nullptr};
};

enum class DType { kF8, kF16, kBF16, kF32 };

struct TensorLocation {
  int file{0};
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::uint64_t count{0};
  DType dtype{DType::kF32};
  std::vector<std::size_t> shape;
};

struct Fp8Linear {
  DeviceBuffer weight;  // FP8 bytes [rows, columns]
  float scale{1};
  int rows{0}, columns{0};
  DeviceBuffer up, down;  // BF16 [rows, rank], [rank, columns]
  int rank{0};
};

struct Block {
  Fp8Linear q, k, v, o;
  Fp8Linear cross_q, cross_k, cross_v, cross_o, cross_k_img, cross_v_img;
  Fp8Linear up, down;
  DeviceBuffer qkv_bias, o_bias, cross_q_bias, cross_k_bias, cross_v_bias,
      cross_o_bias, cross_k_img_bias, cross_v_img_bias, up_bias,
      down_bias;  // BF16
  DeviceBuffer norm_q, norm_k, cross_norm_q, cross_norm_k,
      cross_norm_k_img;                   // FP32
  DeviceBuffer norm2_weight, norm2_bias;  // FP32
  DeviceBuffer modulation;                // FP32 [6, W]
};

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

struct Transformer21::Impl {
  ~Impl() {
    for (auto file : files)
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
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  bool Index(const std::filesystem::path& path, int file, std::string prefix,
             std::string* error);
  const TensorLocation* Find(const std::string& name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
  }
  // Host FP32 copy of a (non-FP8) tensor of `count` values with the LoRA's
  // `diff` (weights) or `diff_b` (biases) added.
  bool ReadHost(const std::string& name, std::size_t count,
                std::vector<float>* out, std::string* error);
  bool UploadRaw(const TensorLocation& location, void* target,
                 std::string* error);
  bool UploadF32(DeviceBuffer& buffer, const std::vector<float>& host,
                 std::string* error);
  bool UploadBf16(DeviceBuffer& buffer, const std::vector<float>& host,
                  std::string* error);
  // A vector (bias, norm weight) as FP32 or BF16.
  bool LoadVector(DeviceBuffer& buffer, const std::string& name,
                  std::size_t count, bool bf16, std::string* error);
  // A dense F16/F32 matrix with the LoRA product merged on the device.
  bool LoadDense(DeviceBuffer& buffer, const std::string& name,
                 std::size_t rows, std::size_t columns, bool bf16,
                 std::string* error);
  bool LoadLora(const std::string& base, std::size_t rows, std::size_t columns,
                DeviceBuffer& up, DeviceBuffer& down, int* rank,
                std::string* error);
  bool LoadFp8(Fp8Linear& linear, const std::string& base, int rows,
               int columns, std::string* error);
  // Unpacks `linear` (with its LoRA) into BF16 `target`.
  bool Unpack(const Fp8Linear& linear, std::uint16_t* target,
              std::string* error);
  bool Gemm(const std::uint16_t* weight, const DeviceBuffer& bias,
            const std::uint16_t* input, std::uint16_t* out, int tokens,
            int rows, int columns, std::string* error) {
    return gemm.Run(weight, bias.get(), input, out, tokens, rows, columns,
                    error);
  }
  bool Reserve(int tokens, std::string* error);
  bool Download(const void* device, bool bf16, std::size_t count,
                std::vector<float>* out, std::string* error);

  std::vector<platform::PlatformFile> files;
  std::map<std::string, TensorLocation, std::less<>> tensors;
  float lora_strength{1};
  std::vector<Block> blocks;
  DeviceBuffer patch_weight, patch_bias;               // BF16
  DeviceBuffer time1, time1_bias, time2, time2_bias;  // FP32
  DeviceBuffer time_proj;                             // BF16
  DeviceBuffer time_proj_bias;                        // FP32
  DeviceBuffer text1, text1_bias, text2, text2_bias;  // BF16
  DeviceBuffer img_norm1_weight, img_norm1_bias, img_norm2_weight,
      img_norm2_bias;                                 // FP32
  DeviceBuffer img1, img1_bias, img2, img2_bias;      // BF16
  DeviceBuffer out_modulation;                        // FP32 [2, W]
  DeviceBuffer proj_out, proj_out_bias;               // BF16

  struct ContextCache {
    int text_rows{0};
    DeviceBuffer text_key, text_packed;  // [layers] x ...
    DeviceBuffer image_key, image_packed;
  };
  ContextCache contexts[kContextSlots];

  // Activations.
  int reserved_tokens{0};
  DeviceBuffer latent, patches, hidden, normed, acc, query, key, value, packed,
      attention, attention_image, ffn, rope, noise;
  DeviceBuffer scratch;  // unpacked BF16 weights (2 x FFN matrix)
  int rope_frames{0}, rope_height{0}, rope_width{0};
  DeviceBuffer sinusoid, time_hidden, temb, proj;  // FP32
  std::size_t activation_bytes{0};
  std::size_t weight_bytes{0};

  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  LinearGemm gemm;
  void* staging[2]{};
  hipEvent_t staged[2]{};
  int next_slot{0};
  hipEvent_t timing[2 * kLayers + 2]{};
  double attention_ms{0};
  double blocks_ms{0};
};

bool Transformer21::Impl::Index(const std::filesystem::path& path, int file,
                                std::string prefix, std::string* error) {
  std::string open_error;
  files.push_back(platform::OpenReadFile(path, &open_error));
  if (!platform::FileValid(files.back())) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return false;
  }
  const auto file_size = platform::FileSizeBytes(files.back());
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(files.back(), &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(files.back(), text.data(), text.size(), 8) !=
      text.size()) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const std::string dtype = entry.member_str("dtype");
      TensorLocation location;
      location.file = file;
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
        SetError(error, "unsupported tensor type " + dtype + " of " + name);
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
      std::string key = name;
      if (!prefix.empty()) {
        if (key.rfind(prefix, 0) != 0)
          continue;
        key = "lora:" + key.substr(prefix.size());
      }
      tensors[key] = std::move(location);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return false;
  }
  return true;
}

bool Transformer21::Impl::ReadHost(const std::string& name, std::size_t count,
                                   std::vector<float>* out,
                                   std::string* error) {
  const auto read = [&](const TensorLocation& location,
                        std::vector<float>* values) {
    std::vector<std::byte> bytes(location.bytes);
    if (platform::ReadFileAt(files[location.file], bytes.data(), bytes.size(),
                             location.offset) != bytes.size()) {
      SetError(error, "short read of " + name);
      return false;
    }
    values->resize(location.count);
    for (std::size_t i = 0; i < location.count; ++i) {
      std::uint16_t half;
      switch (location.dtype) {
      case DType::kF32:
        std::memcpy(&(*values)[i], bytes.data() + 4 * i, 4);
        break;
      case DType::kF16:
        std::memcpy(&half, bytes.data() + 2 * i, 2);
        (*values)[i] = HalfToFloat(half);
        break;
      case DType::kBF16:
        std::memcpy(&half, bytes.data() + 2 * i, 2);
        (*values)[i] = Bf16ToFloat(half);
        break;
      case DType::kF8:
        SetError(error, "unexpected FP8 tensor " + name);
        return false;
      }
    }
    return true;
  };
  const TensorLocation* location = Find(name);
  if (location == nullptr || location->count != count) {
    SetError(error, "missing or misshaped tensor " + name);
    return false;
  }
  if (!read(*location, out))
    return false;
  // LoRA differences: "<base>.diff" for weights, "<base>.diff_b" for biases.
  const auto dot = name.rfind('.');
  const std::string base = name.substr(0, dot);
  const std::string kind = name.substr(dot + 1);
  const TensorLocation* diff = nullptr;
  if (kind == "weight")
    diff = Find("lora:" + base + ".diff");
  else if (kind == "bias")
    diff = Find("lora:" + base + ".diff_b");
  if (diff != nullptr) {
    std::vector<float> values;
    if (diff->count != count || !read(*diff, &values)) {
      SetError(error, "misshaped LoRA difference for " + name);
      return false;
    }
    for (std::size_t i = 0; i < count; ++i)
      (*out)[i] += lora_strength * values[i];
  }
  return true;
}

bool Transformer21::Impl::UploadRaw(const TensorLocation& location,
                                    void* target, std::string* error) {
  for (std::uint64_t copied = 0; copied < location.bytes;) {
    const std::size_t chunk = static_cast<std::size_t>(
        std::min<std::uint64_t>(kStagingBytes, location.bytes - copied));
    const int slot = next_slot;
    next_slot ^= 1;
    if (!Check(hipEventSynchronize(staged[slot]), "weight staging", error))
      return false;
    const std::byte* buffer = ReadStaged(files[location.file],
                                         static_cast<std::byte*>(staging[slot]),
                                         chunk, location.offset + copied);
    if (buffer == nullptr) {
      SetError(error, "short weight read");
      return false;
    }
    if (!Check(hipMemcpyAsync(static_cast<std::byte*>(target) + copied, buffer,
                              chunk, hipMemcpyHostToDevice, stream),
               "weight upload", error) ||
        !Check(hipEventRecord(staged[slot], stream), "weight upload", error))
      return false;
    copied += chunk;
  }
  return true;
}

bool Transformer21::Impl::UploadF32(DeviceBuffer& buffer,
                                    const std::vector<float>& host,
                                    std::string* error) {
  weight_bytes += host.size() * 4;
  return buffer.Allocate(host.size() * 4, error) &&
         Check(hipMemcpy(buffer.raw(), host.data(), host.size() * 4,
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Transformer21::Impl::UploadBf16(DeviceBuffer& buffer,
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

bool Transformer21::Impl::LoadVector(DeviceBuffer& buffer,
                                     const std::string& name,
                                     std::size_t count, bool bf16,
                                     std::string* error) {
  std::vector<float> host;
  return ReadHost(name, count, &host, error) &&
         (bf16 ? UploadBf16(buffer, host, error)
               : UploadF32(buffer, host, error));
}

bool Transformer21::Impl::LoadLora(const std::string& base, std::size_t rows,
                                   std::size_t columns, DeviceBuffer& up,
                                   DeviceBuffer& down, int* rank,
                                   std::string* error) {
  *rank = 0;
  const TensorLocation* u = Find("lora:" + base + ".lora_up.weight");
  const TensorLocation* d = Find("lora:" + base + ".lora_down.weight");
  if (u == nullptr && d == nullptr)
    return true;
  if (u == nullptr || d == nullptr || u->shape.size() != 2 ||
      d->shape.size() != 2 || u->shape[0] != rows || d->shape[1] != columns ||
      u->shape[1] != d->shape[0] || u->dtype != DType::kBF16 ||
      d->dtype != DType::kBF16) {
    SetError(error, "unsupported LoRA tensors for " + base);
    return false;
  }
  *rank = static_cast<int>(u->shape[1]);
  weight_bytes += u->bytes + d->bytes;
  return up.Allocate(u->bytes, error) && down.Allocate(d->bytes, error) &&
         UploadRaw(*u, up.raw(), error) && UploadRaw(*d, down.raw(), error);
}

// row-major C[rows, columns] += strength * up[rows, r] . down[r, columns]:
// column-major C^T (columns x rows) += down^T (columns x r) . up^T (r x rows).
bool AddLowRank(rocblas_handle blas, const void* up, const void* down,
                int rank, int rows, int columns, float strength, void* target,
                bool target_f32, std::string* error) {
  const float beta = 1.0F;
  const rocblas_datatype c_type =
      target_f32 ? rocblas_datatype_f32_r : rocblas_datatype_bf16_r;
  if (rocblas_gemm_ex(blas, rocblas_operation_none, rocblas_operation_none,
                      columns, rows, rank, &strength, down,
                      rocblas_datatype_bf16_r, columns, up,
                      rocblas_datatype_bf16_r, rank, &beta, target, c_type,
                      columns, target, c_type, columns, rocblas_datatype_f32_r,
                      rocblas_gemm_algo_standard, 0, 0) !=
      rocblas_status_success) {
    SetError(error, "Wan 2.1 transformer: LoRA GEMM failed");
    return false;
  }
  return true;
}

bool Transformer21::Impl::LoadDense(DeviceBuffer& buffer,
                                    const std::string& name, std::size_t rows,
                                    std::size_t columns, bool bf16,
                                    std::string* error) {
  std::vector<float> host;
  if (!ReadHost(name + ".weight", rows * columns, &host, error))
    return false;
  DeviceBuffer full, up, down;
  int rank = 0;
  const std::size_t before = weight_bytes;
  if (!UploadF32(full, host, error) ||
      !LoadLora(name, rows, columns, up, down, &rank, error))
    return false;
  if (rank > 0 &&
      !AddLowRank(blas, up.raw(), down.raw(), rank, static_cast<int>(rows),
                  static_cast<int>(columns), lora_strength, full.raw(), true,
                  error))
    return false;
  weight_bytes = before + host.size() * (bf16 ? 2 : 4);
  if (!bf16) {
    buffer = std::move(full);
    return Check(hipStreamSynchronize(stream), "dense weight", error);
  }
  if (!buffer.Allocate(host.size() * 2, error))
    return false;
  ConvertKernel<<<Blocks(host.size()), 256, 0, stream>>>(
      full.f32(), host.size(), buffer.get());
  return Check(hipStreamSynchronize(stream), "dense weight", error);
}

bool Transformer21::Impl::LoadFp8(Fp8Linear& linear, const std::string& base,
                                  int rows, int columns, std::string* error) {
  const TensorLocation* w = Find(base + ".weight");
  const TensorLocation* s = Find(base + ".scale_weight");
  if (w == nullptr || s == nullptr || w->dtype != DType::kF8 ||
      w->count != static_cast<std::uint64_t>(rows) * columns ||
      s->count != 1) {
    SetError(error, "missing or misshaped FP8 tensor " + base);
    return false;
  }
  std::vector<float> scale;
  if (!ReadHost(base + ".scale_weight", 1, &scale, error))
    return false;
  linear.scale = scale[0];
  linear.rows = rows;
  linear.columns = columns;
  weight_bytes += w->bytes;
  return linear.weight.Allocate(w->bytes, error) &&
         UploadRaw(*w, linear.weight.raw(), error) &&
         LoadLora(base, rows, columns, linear.up, linear.down, &linear.rank,
                  error);
}

bool Transformer21::Impl::Unpack(const Fp8Linear& linear,
                                 std::uint16_t* target, std::string* error) {
  const std::size_t quads =
      static_cast<std::size_t>(linear.rows) * linear.columns / 4;
  UnpackKernel<<<Blocks(quads), 256, 0, stream>>>(
      static_cast<const std::uint32_t*>(linear.weight.raw()), quads,
      linear.scale, target);
  return linear.rank == 0 ||
         AddLowRank(blas, linear.up.raw(), linear.down.raw(), linear.rank,
                    linear.rows, linear.columns, lora_strength, target, false,
                    error);
}

bool Transformer21::Impl::Reserve(int tokens, std::string* error) {
  if (tokens <= reserved_tokens)
    return true;
  const std::size_t T = tokens, W = kWidth, F = kFeedForward;
  const std::size_t TP = PaddedRows(tokens);
  activation_bytes = 0;
  auto allocate = [&](DeviceBuffer& buffer, std::size_t bytes) {
    activation_bytes += bytes;
    return buffer.Allocate(bytes, error);
  };
  rope_frames = rope_height = rope_width = 0;
  reserved_tokens = 0;
  if (!allocate(latent, T * kPatchIn * 4) ||
      !allocate(patches, T * kPatchIn * 2) || !allocate(hidden, T * W * 4) ||
      !allocate(normed, T * W * 2) || !allocate(acc, T * 3 * W * 2) ||
      !allocate(query, T * W * 2) || !allocate(key, T * W * 2) ||
      !allocate(value, T * W * 2) || !allocate(packed, TP * W * 2) ||
      !allocate(attention, T * W * 2) ||
      !allocate(attention_image, T * W * 2) ||
      !allocate(ffn, std::min<std::size_t>(T, kFeedForwardChunk) * F * 2) ||
      !allocate(rope, T * kRopePairs * 8) ||
      !allocate(noise, T * kPatchOut * 4))
    return false;
  reserved_tokens = tokens;
  return true;
}

bool Transformer21::Impl::Download(const void* device, bool bf16,
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
  for (std::size_t i = 0; i < count; ++i)
    (*out)[i] = Bf16ToFloat(half[i]);
  return true;
}

Transformer21::Transformer21(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
Transformer21::~Transformer21() = default;

std::size_t Transformer21::weight_bytes() const {
  return impl_->weight_bytes;
}
std::size_t Transformer21::activation_bytes() const {
  return impl_->activation_bytes;
}
double Transformer21::last_attention_ms() const {
  return impl_->attention_ms;
}
double Transformer21::last_blocks_ms() const {
  return impl_->blocks_ms;
}
hipStream_t Transformer21::stream() const {
  return impl_->stream;
}

std::unique_ptr<Transformer21> Transformer21::Load(
    const std::filesystem::path& dit_file,
    const std::filesystem::path& lora_file, float lora_strength,
    std::string* error) {
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.lora_strength = lora_strength;
  if (!m.Index(dit_file, 0, "", error) ||
      (!lora_file.empty() &&
       !m.Index(lora_file, 1, "diffusion_model.", error)))
    return nullptr;
  if (m.Find("blocks.0.cross_attn.k_img.weight") == nullptr ||
      m.Find("blocks.39.ffn.2.scale_weight") == nullptr ||
      m.Find("blocks.40.ffn.0.weight") != nullptr) {
    SetError(error, "not a Wan 2.1 I2V 14B FP8 (scaled) checkpoint");
    return nullptr;
  }
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.gemm.Init(m.stream, error))
    return nullptr;
  if (rocblas_create_handle(&m.blas) != rocblas_status_success ||
      rocblas_set_stream(m.blas, m.stream) != rocblas_status_success) {
    SetError(error, "Wan 2.1 transformer rocBLAS handle");
    return nullptr;
  }
  for (int slot = 0; slot < 2; ++slot)
    if (!Check(hipHostMalloc(&m.staging[slot], kStagingBytes + kStagingSlack,
                             hipHostMallocDefault),
               "pinned staging", error) ||
        !Check(hipEventCreateWithFlags(&m.staged[slot], hipEventDisableTiming),
               "staging event", error) ||
        !Check(hipEventRecord(m.staged[slot], m.stream), "staging event",
               error))
      return nullptr;
  for (hipEvent_t& event : m.timing)
    if (!Check(hipEventCreate(&event), "timing event", error))
      return nullptr;

  constexpr std::size_t W = kWidth;
  constexpr std::size_t F = kFeedForward;
  // Patch embedding [5120, 36, 1, 2, 2] -> [5120, 144] (c * 4 + kh * 2 + kw).
  if (!m.LoadDense(m.patch_weight, "patch_embedding", W, kPatchIn, true,
                   error) ||
      !m.LoadVector(m.patch_bias, "patch_embedding.bias", W, true, error) ||
      !m.LoadDense(m.time1, "time_embedding.0", W, kFrequencies, false,
                   error) ||
      !m.LoadVector(m.time1_bias, "time_embedding.0.bias", W, false, error) ||
      !m.LoadDense(m.time2, "time_embedding.2", W, W, false, error) ||
      !m.LoadVector(m.time2_bias, "time_embedding.2.bias", W, false, error) ||
      !m.LoadDense(m.time_proj, "time_projection.1", kModulation, W, true,
                   error) ||
      !m.LoadVector(m.time_proj_bias, "time_projection.1.bias", kModulation,
                    false, error) ||
      !m.LoadDense(m.text1, "text_embedding.0", W,
                   Transformer21::kTextWidth, true, error) ||
      !m.LoadVector(m.text1_bias, "text_embedding.0.bias", W, true, error) ||
      !m.LoadDense(m.text2, "text_embedding.2", W, W, true, error) ||
      !m.LoadVector(m.text2_bias, "text_embedding.2.bias", W, true, error) ||
      !m.LoadVector(m.img_norm1_weight, "img_emb.proj.0.weight", kImageWidth,
                    false, error) ||
      !m.LoadVector(m.img_norm1_bias, "img_emb.proj.0.bias", kImageWidth,
                    false, error) ||
      !m.LoadDense(m.img1, "img_emb.proj.1", kImageWidth, kImageWidth, true,
                   error) ||
      !m.LoadVector(m.img1_bias, "img_emb.proj.1.bias", kImageWidth, true,
                    error) ||
      !m.LoadDense(m.img2, "img_emb.proj.3", W, kImageWidth, true, error) ||
      !m.LoadVector(m.img2_bias, "img_emb.proj.3.bias", W, true, error) ||
      !m.LoadVector(m.img_norm2_weight, "img_emb.proj.4.weight", W, false,
                    error) ||
      !m.LoadVector(m.img_norm2_bias, "img_emb.proj.4.bias", W, false,
                    error) ||
      !m.LoadVector(m.out_modulation, "head.modulation", 2 * W, false,
                    error) ||
      !m.LoadDense(m.proj_out, "head.head", kPatchOut, W, true, error) ||
      !m.LoadVector(m.proj_out_bias, "head.head.bias", kPatchOut, true, error))
    return nullptr;
  m.blocks = std::vector<Block>(kLayers);
  for (int index = 0; index < kLayers; ++index) {
    Block& b = m.blocks[index];
    const std::string p = "blocks." + std::to_string(index) + ".";
    const std::string s = p + "self_attn.";
    const std::string c = p + "cross_attn.";
    const int w = kWidth;
    // q | k | v biases stacked for the fused GEMM.
    std::vector<float> qkv_bias, part;
    for (const char* name : {"q", "k", "v"}) {
      if (!m.ReadHost(s + name + ".bias", W, &part, error))
        return nullptr;
      qkv_bias.insert(qkv_bias.end(), part.begin(), part.end());
    }
    if (!m.UploadBf16(b.qkv_bias, qkv_bias, error) ||
        !m.LoadFp8(b.q, s + "q", w, w, error) ||
        !m.LoadFp8(b.k, s + "k", w, w, error) ||
        !m.LoadFp8(b.v, s + "v", w, w, error) ||
        !m.LoadFp8(b.o, s + "o", w, w, error) ||
        !m.LoadVector(b.o_bias, s + "o.bias", W, true, error) ||
        !m.LoadVector(b.norm_q, s + "norm_q.weight", W, false, error) ||
        !m.LoadVector(b.norm_k, s + "norm_k.weight", W, false, error) ||
        !m.LoadFp8(b.cross_q, c + "q", w, w, error) ||
        !m.LoadVector(b.cross_q_bias, c + "q.bias", W, true, error) ||
        !m.LoadFp8(b.cross_k, c + "k", w, w, error) ||
        !m.LoadVector(b.cross_k_bias, c + "k.bias", W, true, error) ||
        !m.LoadFp8(b.cross_v, c + "v", w, w, error) ||
        !m.LoadVector(b.cross_v_bias, c + "v.bias", W, true, error) ||
        !m.LoadFp8(b.cross_o, c + "o", w, w, error) ||
        !m.LoadVector(b.cross_o_bias, c + "o.bias", W, true, error) ||
        !m.LoadFp8(b.cross_k_img, c + "k_img", w, w, error) ||
        !m.LoadVector(b.cross_k_img_bias, c + "k_img.bias", W, true, error) ||
        !m.LoadFp8(b.cross_v_img, c + "v_img", w, w, error) ||
        !m.LoadVector(b.cross_v_img_bias, c + "v_img.bias", W, true, error) ||
        !m.LoadVector(b.cross_norm_q, c + "norm_q.weight", W, false, error) ||
        !m.LoadVector(b.cross_norm_k, c + "norm_k.weight", W, false, error) ||
        !m.LoadVector(b.cross_norm_k_img, c + "norm_k_img.weight", W, false,
                      error) ||
        !m.LoadVector(b.norm2_weight, p + "norm3.weight", W, false, error) ||
        !m.LoadVector(b.norm2_bias, p + "norm3.bias", W, false, error) ||
        !m.LoadFp8(b.up, p + "ffn.0", kFeedForward, w, error) ||
        !m.LoadVector(b.up_bias, p + "ffn.0.bias", F, true, error) ||
        !m.LoadFp8(b.down, p + "ffn.2", w, kFeedForward, error) ||
        !m.LoadVector(b.down_bias, p + "ffn.2.bias", W, true, error) ||
        !m.LoadVector(b.modulation, p + "modulation", 6 * W, false, error))
      return nullptr;
  }
  if (!m.scratch.Allocate(2 * F * W * 2, error) ||
      !m.sinusoid.Allocate(kFrequencies * 4, error) ||
      !m.time_hidden.Allocate(W * 4, error) || !m.temb.Allocate(W * 4, error) ||
      !m.proj.Allocate(kModulation * 4, error) ||
      !Check(hipStreamSynchronize(m.stream), "weight upload", error))
    return nullptr;
  m.activation_bytes += 2 * F * W * 2;
  for (int slot = 0; slot < 2; ++slot) {
    (void)hipHostFree(m.staging[slot]);
    m.staging[slot] = nullptr;
  }
  return std::unique_ptr<Transformer21>(new Transformer21(std::move(impl)));
}

bool Transformer21::PrepareContext(int slot, std::span<const float> text,
                                   int text_rows, std::span<const float> image,
                                   std::vector<float>* context_text,
                                   std::vector<float>* context_image,
                                   std::string* error) {
  Impl& m = *impl_;
  constexpr std::size_t W = kWidth;
  if (slot < 0 || slot >= kContextSlots || text_rows <= 0 ||
      text_rows > 4096 ||
      text.size() != static_cast<std::size_t>(text_rows) * kTextWidth ||
      image.size() != static_cast<std::size_t>(kImageRows) * kImageWidth) {
    SetError(error, "Wan 2.1 transformer: invalid context");
    return false;
  }
  hipStream_t s = m.stream;
  const std::size_t L = text_rows, LP = PaddedRows(text_rows);
  const std::size_t I = kImageRows, IP = PaddedRows(kImageRows);
  DeviceBuffer text_in, image_in, text_hidden, text_states, image_normed,
      image_hidden, image_states, linear, keys_tmp;
  std::vector<std::uint16_t> half(text.size());
  for (std::size_t i = 0; i < half.size(); ++i)
    half[i] = HostBf16(text[i]);
  if (!text_in.Allocate(half.size() * 2, error) ||
      !image_in.Allocate(image.size() * 4, error) ||
      !text_hidden.Allocate(L * W * 2, error) ||
      !text_states.Allocate(L * W * 2, error) ||
      !image_normed.Allocate(I * kImageWidth * 2, error) ||
      !image_hidden.Allocate(I * W * 2, error) ||
      !image_states.Allocate(I * W * 2, error) ||
      !linear.Allocate(std::max(L, I) * W * 2, error) ||
      !keys_tmp.Allocate(std::max(L, I) * W * 2, error) ||
      !Check(hipMemcpyAsync(text_in.raw(), half.data(), half.size() * 2,
                            hipMemcpyHostToDevice, s),
             "context upload", error) ||
      !Check(hipMemcpyAsync(image_in.raw(), image.data(), image.size() * 4,
                            hipMemcpyHostToDevice, s),
             "context upload", error))
    return false;
  // Text: linear, GELU (tanh), linear.
  if (!m.Gemm(m.text1.get(), m.text1_bias, text_in.get(), text_hidden.get(),
              text_rows, kWidth, kTextWidth, error))
    return false;
  GeluKernel<false>
      <<<Blocks(L * W), 256, 0, s>>>(text_hidden.get(), L * W);
  if (!m.Gemm(m.text2.get(), m.text2_bias, text_hidden.get(),
              text_states.get(), text_rows, kWidth, kWidth, error))
    return false;
  // Image: layer norm, linear, GELU (erf), linear, layer norm (eps 1e-5).
  LayerNormAnyF32Kernel<<<kImageRows, kRowThreads, 0, s>>>(
      image_in.f32(), kImageWidth, m.img_norm1_weight.f32(),
      m.img_norm1_bias.f32(), 1e-5F, image_normed.get());
  if (!m.Gemm(m.img1.get(), m.img1_bias, image_normed.get(),
              image_hidden.get(), kImageRows, kImageWidth, kImageWidth, error))
    return false;
  GeluKernel<true><<<Blocks(I * kImageWidth), 256, 0, s>>>(image_hidden.get(),
                                                           I * kImageWidth);
  if (!m.Gemm(m.img2.get(), m.img2_bias, image_hidden.get(), linear.get(),
              kImageRows, kWidth, kImageWidth, error))
    return false;
  LayerNormAnyKernel<<<kImageRows, kRowThreads, 0, s>>>(
      linear.get(), kWidth, m.img_norm2_weight.f32(), m.img_norm2_bias.f32(),
      1e-5F, image_states.get());
  if ((context_text != nullptr &&
       !m.Download(text_states.get(), true, L * W, context_text, error)) ||
      (context_image != nullptr &&
       !m.Download(image_states.get(), true, I * W, context_image, error)))
    return false;

  Impl::ContextCache& cache = m.contexts[slot];
  if (cache.text_rows != text_rows) {
    cache.text_rows = 0;
    if (!cache.text_key.Allocate(kLayers * L * W * 2, error) ||
        !cache.text_packed.Allocate(kLayers * LP * W * 2, error) ||
        !cache.image_key.Allocate(kLayers * I * W * 2, error) ||
        !cache.image_packed.Allocate(kLayers * IP * W * 2, error))
      return false;
  }
  std::uint16_t* scratch = m.scratch.get();
  const auto keys_values = [&](const Fp8Linear& kl, const DeviceBuffer& kb,
                               const DeviceBuffer& norm, const Fp8Linear& vl,
                               const DeviceBuffer& vb,
                               const std::uint16_t* states, int rows,
                               std::uint16_t* key_out,
                               std::uint16_t* packed_out) {
    if (!m.Unpack(kl, scratch, error) ||
        !m.Gemm(scratch, kb, states, linear.get(), rows, kWidth, kWidth,
                error))
      return false;
    HeadsKernel<<<rows, kRowThreads, 0, s>>>(linear.get(), kWidth, rows,
                                             norm.f32(), nullptr, key_out);
    if (!m.Unpack(vl, scratch, error) ||
        !m.Gemm(scratch, vb, states, linear.get(), rows, kWidth, kWidth,
                error))
      return false;
    HeadsKernel<<<rows, kRowThreads, 0, s>>>(linear.get(), kWidth, rows,
                                             nullptr, nullptr, keys_tmp.get());
    LaunchPackValues(keys_tmp.get(), packed_out, rows, kHeads, s);
    return true;
  };
  for (int layer = 0; layer < kLayers; ++layer) {
    const Block& b = m.blocks[layer];
    if (!keys_values(b.cross_k, b.cross_k_bias, b.cross_norm_k, b.cross_v,
                     b.cross_v_bias, text_states.get(), text_rows,
                     cache.text_key.get() + layer * L * W,
                     cache.text_packed.get() + layer * LP * W) ||
        !keys_values(b.cross_k_img, b.cross_k_img_bias, b.cross_norm_k_img,
                     b.cross_v_img, b.cross_v_img_bias, image_states.get(),
                     kImageRows, cache.image_key.get() + layer * I * W,
                     cache.image_packed.get() + layer * IP * W))
      return false;
  }
  if (!Check(hipGetLastError(), "context", error) ||
      !Check(hipStreamSynchronize(s), "context", error))
    return false;
  cache.text_rows = text_rows;
  return true;
}

bool Transformer21::Forward(const Transformer21Input& input,
                            std::vector<float>* noise,
                            Transformer21Probes* probes, std::string* error) {
  Impl& m = *impl_;
  if (input.frames <= 0 || input.height <= 0 || input.width <= 0 ||
      input.height % 2 != 0 || input.width % 2 != 0 ||
      input.height / 2 > 1024 || input.width / 2 > 1024 ||
      input.frames > 1024) {
    SetError(error, "Wan 2.1 transformer: invalid latent shape");
    return false;
  }
  if (input.context_slot < 0 || input.context_slot >= kContextSlots ||
      m.contexts[input.context_slot].text_rows == 0) {
    SetError(error, "Wan 2.1 transformer: context slot was not prepared");
    return false;
  }
  const Impl::ContextCache& context = m.contexts[input.context_slot];
  const int ph = input.height / 2, pw = input.width / 2;
  const int tokens = input.frames * ph * pw;
  const std::size_t latent_count = static_cast<std::size_t>(kInChannels) *
                                   input.frames * input.height * input.width;
  if (input.latent.size() != latent_count) {
    SetError(error, "Wan 2.1 transformer: input size does not match its shape");
    return false;
  }
  if (!m.Reserve(tokens, error))
    return false;
  hipStream_t s = m.stream;
  // 3D RoPE as the TI2V-5B port (head dim 128 in both).
  std::vector<float> rope;
  const bool new_rope = m.rope_frames != input.frames ||
                        m.rope_height != input.height ||
                        m.rope_width != input.width;
  if (new_rope) {
    rope.resize(static_cast<std::size_t>(tokens) * kRopePairs * 2);
    for (int t = 0; t < tokens; ++t) {
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
  }
  if (!Check(hipMemcpyAsync(m.latent.f32(), input.latent.data(),
                            latent_count * 4, hipMemcpyHostToDevice, s),
             "input upload", error) ||
      (new_rope &&
       !Check(hipMemcpyAsync(m.rope.f32(), rope.data(), rope.size() * 4,
                             hipMemcpyHostToDevice, s),
              "input upload", error)) ||
      !Check(hipStreamSynchronize(s), "input upload", error))
    return false;
  if (new_rope) {
    m.rope_frames = input.frames;
    m.rope_height = input.height;
    m.rope_width = input.width;
  }
  const std::size_t T = tokens;
  const std::size_t W = kWidth;
  const std::size_t count = T * W;
  const auto* rope_table = reinterpret_cast<const float2*>(m.rope.f32());
  std::uint16_t* scratch = m.scratch.get();
  std::uint16_t* scratch2 =
      scratch + static_cast<std::size_t>(kFeedForward) * W;

  // Patch embedding: a [T, 144] x [144, W] GEMM.
  PatchKernel<<<Blocks(T * kPatchIn), 256, 0, s>>>(
      m.latent.f32(), input.frames, input.height, input.width, m.patches.get());
  if (!m.Gemm(m.patch_weight.get(), m.patch_bias, m.patches.get(),
              m.acc.get(), tokens, kWidth, kPatchIn, error))
    return false;
  // FP32 residual stream from the BF16 GEMM output.
  if (!Check(hipMemsetAsync(m.hidden.f32(), 0, count * 4, s), "zero", error))
    return false;
  ResidualKernel<<<Blocks(count), 256, 0, s>>>(m.acc.get(), nullptr, nullptr,
                                               count, m.hidden.f32());

  // Time embedding in FP32.
  SinusoidKernel<<<1, kFrequencies / 2, 0, s>>>(input.timestep,
                                                m.sinusoid.f32());
  MatVecKernel<float, false><<<kWidth, kRowThreads, 0, s>>>(
      m.time1.f32(), m.time1_bias.f32(), m.sinusoid.f32(), kFrequencies,
      m.time_hidden.f32());
  MatVecKernel<float, true><<<kWidth, kRowThreads, 0, s>>>(
      m.time2.f32(), m.time2_bias.f32(), m.time_hidden.f32(), kWidth,
      m.temb.f32());
  MatVecKernel<std::uint16_t, true><<<kModulation, kRowThreads, 0, s>>>(
      m.time_proj.get(), m.time_proj_bias.f32(), m.temb.f32(), kWidth,
      m.proj.f32());
  if (!Check(hipGetLastError(), "embedding launch", error))
    return false;
  if (probes != nullptr &&
      !m.Download(m.temb.f32(), false, W, &probes->temb, error))
    return false;
  if (input.hook != nullptr &&
      !input.hook->Begin(m.latent.f32(), input.frames, input.height,
                         input.width, m.temb.f32(), s, error))
    return false;

  const float* proj = m.proj.f32();
  const auto modulated = [&](const Block& b, int shift, int scale) {
    NormArgs a;
    a.shift_table = b.modulation.f32() + shift * W;
    a.scale_table = b.modulation.f32() + scale * W;
    a.shift_proj = proj + shift * W;
    a.scale_proj = proj + scale * W;
    return a;
  };
  std::uint16_t* linear = m.acc.get();
  const std::size_t L = context.text_rows, LP = PaddedRows(context.text_rows);
  const std::size_t I = kImageRows, IP = PaddedRows(kImageRows);
  const unsigned rows = static_cast<unsigned>(T);
  if (!Check(hipEventRecord(m.timing[2 * kLayers], s), "timing", error))
    return false;
  for (int layer = 0; layer < kLayers; ++layer) {
    const Block& b = m.blocks[layer];
    // 1. Self-attention.
    LayerNormKernel<<<rows, kRowThreads, 0, s>>>(
        m.hidden.f32(), modulated(b, 0, 1), m.normed.get());
    if (!m.Unpack(b.q, scratch, error) ||
        !m.Unpack(b.k, scratch + W * W, error) ||
        !m.Unpack(b.v, scratch + 2 * W * W, error) ||
        !m.Gemm(scratch, b.qkv_bias, m.normed.get(), linear, tokens, 3 * kWidth,
                kWidth, error))
      return false;
    HeadsKernel<<<rows, kRowThreads, 0, s>>>(linear, 3 * kWidth, tokens,
                                             b.norm_q.f32(), rope_table,
                                             m.query.get());
    HeadsKernel<<<rows, kRowThreads, 0, s>>>(linear + W, 3 * kWidth, tokens,
                                             b.norm_k.f32(), rope_table,
                                             m.key.get());
    HeadsKernel<<<rows, kRowThreads, 0, s>>>(linear + 2 * W, 3 * kWidth,
                                             tokens, nullptr, nullptr,
                                             m.value.get());
    LaunchPackValues(m.value.get(), m.packed.get(), tokens, kHeads, s);
    (void)hipEventRecord(m.timing[2 * layer], s);
    LaunchAttention(m.query.get(), m.key.get(), m.packed.get(),
                    m.attention.get(), tokens, tokens, kHeads, s);
    (void)hipEventRecord(m.timing[2 * layer + 1], s);
    if (!m.Unpack(b.o, scratch, error) ||
        !m.Gemm(scratch, b.o_bias, m.attention.get(), linear, tokens, kWidth,
                kWidth, error))
      return false;
    ResidualKernel<<<Blocks(count), 256, 0, s>>>(
        linear, b.modulation.f32() + 2 * W, proj + 2 * W, count,
        m.hidden.f32());

    // 2. Cross-attention to text and image context.
    NormArgs cross_norm;
    cross_norm.weight = b.norm2_weight.f32();
    cross_norm.bias = b.norm2_bias.f32();
    LayerNormKernel<<<rows, kRowThreads, 0, s>>>(m.hidden.f32(), cross_norm,
                                                 m.normed.get());
    if (!m.Unpack(b.cross_q, scratch, error) ||
        !m.Gemm(scratch, b.cross_q_bias, m.normed.get(), linear, tokens,
                kWidth, kWidth, error))
      return false;
    HeadsKernel<<<rows, kRowThreads, 0, s>>>(linear, kWidth, tokens,
                                             b.cross_norm_q.f32(), nullptr,
                                             m.query.get());
    LaunchAttention(m.query.get(), context.text_key.get() + layer * L * W,
                    context.text_packed.get() + layer * LP * W,
                    m.attention.get(), tokens, static_cast<std::uint32_t>(L),
                    kHeads, s);
    LaunchAttention(m.query.get(), context.image_key.get() + layer * I * W,
                    context.image_packed.get() + layer * IP * W,
                    m.attention_image.get(), tokens, kImageRows, kHeads, s);
    AddKernel<<<Blocks(count), 256, 0, s>>>(m.attention.get(),
                                            m.attention_image.get(), count);
    if (!m.Unpack(b.cross_o, scratch, error) ||
        !m.Gemm(scratch, b.cross_o_bias, m.attention.get(), linear, tokens,
                kWidth, kWidth, error))
      return false;
    ResidualKernel<<<Blocks(count), 256, 0, s>>>(linear, nullptr, nullptr,
                                                 count, m.hidden.f32());

    // 3. Feed-forward (GELU tanh).
    LayerNormKernel<<<rows, kRowThreads, 0, s>>>(
        m.hidden.f32(), modulated(b, 3, 4), m.normed.get());
    if (!m.Unpack(b.up, scratch, error) || !m.Unpack(b.down, scratch2, error))
      return false;
    for (int first = 0; first < tokens; first += kFeedForwardChunk) {
      const int chunk = std::min(kFeedForwardChunk, tokens - first);
      const std::size_t up_count =
          static_cast<std::size_t>(chunk) * kFeedForward;
      if (!m.Gemm(scratch, b.up_bias, m.normed.get() + first * W, m.ffn.get(),
                  chunk, kFeedForward, kWidth, error))
        return false;
      GeluKernel<false><<<Blocks(up_count), 256, 0, s>>>(m.ffn.get(), up_count);
      if (!m.Gemm(scratch2, b.down_bias, m.ffn.get(), linear + first * W,
                  chunk, kWidth, kFeedForward, error))
        return false;
    }
    ResidualKernel<<<Blocks(count), 256, 0, s>>>(
        linear, b.modulation.f32() + 5 * W, proj + 5 * W, count,
        m.hidden.f32());
    if (!Check(hipGetLastError(), "block launch", error))
      return false;
    if (input.hook != nullptr &&
        !input.hook->AfterBlock(layer, m.hidden.f32(), s, error))
      return false;
    if (probes != nullptr && layer == 0 &&
        !m.Download(m.hidden.f32(), false, count, &probes->block0, error))
      return false;
  }
  if (!Check(hipEventRecord(m.timing[2 * kLayers + 1], s), "timing", error))
    return false;

  // Output norm modulated by temb, projection, unpatchify.
  NormArgs head;
  head.shift_table = m.out_modulation.f32();
  head.scale_table = m.out_modulation.f32() + W;
  head.shift_proj = m.temb.f32();
  head.scale_proj = m.temb.f32();
  LayerNormKernel<<<rows, kRowThreads, 0, s>>>(m.hidden.f32(), head,
                                               m.normed.get());
  if (!m.Gemm(m.proj_out.get(), m.proj_out_bias, m.normed.get(), linear,
              tokens, kPatchOut, kWidth, error))
    return false;
  UnpatchKernel<<<Blocks(T * kPatchOut), 256, 0, s>>>(
      linear, input.frames, input.height, input.width, m.noise.f32());
  const std::size_t out_count = static_cast<std::size_t>(kOutChannels) *
                                input.frames * input.height * input.width;
  if (!Check(hipGetLastError(), "output launch", error) ||
      !m.Download(m.noise.f32(), false, out_count, noise, error))
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
