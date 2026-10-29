#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/wan22/clip_vision.hpp"
#include "src/models/wan22/gemm.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr int kLayersUsed = 31;  // hidden_states[-2]: after layer 31 of 32
constexpr int kWidth = ClipVision::kWidth;
constexpr int kTokens = ClipVision::kTokens;
constexpr int kHeads = 16;
constexpr int kHeadDim = 80;
constexpr int kMlp = 5120;
constexpr int kPatch = 14;
constexpr int kGrid = ClipVision::kImageSize / kPatch;  // 16
constexpr float kEpsilon = 1e-5F;
constexpr int kThreads = 256;
constexpr int kRowValues = kWidth / kThreads;  // 5

static_assert(kWidth == kHeads * kHeadDim && kWidth % kThreads == 0);

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

// out = bf16(layer_norm(hidden) * weight + bias), FP32 two-pass statistics.
__global__ void LayerNormKernel(const float* hidden, const float* weight,
                                const float* bias, std::uint16_t* out) {
  __shared__ float shared[kThreads];
  const std::size_t row = blockIdx.x;
  const float* x = hidden + row * kWidth;
  float values[kRowValues];
  float sum = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] = x[threadIdx.x + i * kThreads];
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
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    const int c = threadIdx.x + i * kThreads;
    out[row * kWidth + c] = ToBf16(values[i] * rstd * weight[c] + bias[c]);
  }
}

// Exact (erf) GELU in place on BF16.
__global__ void GeluKernel(std::uint16_t* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = FromBf16(x[i]);
  x[i] = ToBf16(0.5F * v * (1.0F + erff(v * 0.7071067811865476F)));
}

__global__ void ResidualKernel(const std::uint16_t* y, std::size_t count,
                               float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    hidden[i] += FromBf16(y[i]);
}

__global__ void WidenKernel(const std::uint16_t* x, std::size_t count,
                            float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = FromBf16(x[i]);
}

__global__ void NarrowKernel(const float* x, std::size_t count,
                             std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(x[i]);
}

// In-place softmax(scale * row) of FP32 rows of `columns`.
__global__ void SoftmaxKernel(float* scores, int columns, float scale) {
  __shared__ float shared[kThreads];
  float* row = scores + static_cast<std::size_t>(blockIdx.x) * columns;
  float maximum = -INFINITY;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    maximum = fmaxf(maximum, row[c] * scale);
  shared[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = shared[0];
  __syncthreads();
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kThreads) {
    const float e = expf(row[c] * scale - maximum);
    row[c] = e;
    sum += e;
  }
  sum = BlockSum(sum, shared);
  const float inverse = 1.0F / sum;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    row[c] *= inverse;
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
  SetError(error, std::string("CLIP vision ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
}

float HostFromBf16(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
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
  bool Upload(const void* host, std::size_t bytes, std::string* error) {
    return Allocate(bytes, error) &&
           Check(hipMemcpy(data_, host, bytes, hipMemcpyHostToDevice),
                 "weight upload", error);
  }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }

private:
  void* data_{nullptr};
};

struct Layer {
  DeviceBuffer norm1_weight, norm1_bias, norm2_weight, norm2_bias;  // FP32
  DeviceBuffer qkv, qkv_bias;  // BF16 [3 * 1280, 1280]
  DeviceBuffer out, out_bias;
  DeviceBuffer fc1, fc1_bias, fc2, fc2_bias;
};

// PIL's bicubic (a = -0.5) resampling coefficients for one axis, in its
// 22-bit fixed point, with antialiasing when downscaling.
struct Coefficients {
  std::vector<int> first, count;
  std::vector<std::int32_t> weights;  // [out, ksize]
  int ksize{0};
};

double Bicubic(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0)
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

Coefficients MakeCoefficients(int in_size, int out_size) {
  constexpr int kPrecision = 22;
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = std::max(scale, 1.0);
  const double support = 2.0 * filterscale;
  Coefficients c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.first.resize(out_size);
  c.count.resize(out_size);
  c.weights.assign(static_cast<std::size_t>(out_size) * c.ksize, 0);
  std::vector<double> k(c.ksize);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    xmin = std::max(xmin, 0);
    int xmax = static_cast<int>(center + support + 0.5);
    xmax = std::min(xmax, in_size) - xmin;
    double ww = 0;
    for (int x = 0; x < xmax; ++x) {
      k[x] = Bicubic((x + xmin - center + 0.5) * ss);
      ww += k[x];
    }
    for (int x = 0; x < xmax; ++x) {
      const double w = ww != 0.0 ? k[x] / ww : k[x];
      c.weights[static_cast<std::size_t>(xx) * c.ksize + x] =
          static_cast<std::int32_t>(w < 0 ? w * (1 << kPrecision) - 0.5
                                          : w * (1 << kPrecision) + 0.5);
    }
    c.first[xx] = xmin;
    c.count[xx] = xmax;
  }
  return c;
}

std::uint8_t Clip8(std::int64_t value) {
  value >>= 22;
  return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value, 0, 255));
}

}  // namespace

struct ClipVision::Impl {
  ~Impl() {
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
  bool Run(std::string* error);

  // Host FP32 copies of the embeddings.
  std::vector<float> patch_weight;  // [1280, 3 * 14 * 14]
  std::vector<float> class_embedding, position, pre_weight, pre_bias;
  std::vector<Layer> layers;
  std::size_t weight_bytes{0};

  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  LinearGemm gemm;
  DeviceBuffer hidden;   // FP32 [257, 1280]
  DeviceBuffer normed;   // BF16 [257, 1280]
  DeviceBuffer qkv;      // BF16 [257, 3840]
  DeviceBuffer qkv32;    // FP32 [257, 3840]
  DeviceBuffer scores;   // FP32 [16, 257, 257]
  DeviceBuffer attn32;   // FP32 [257, 1280]
  DeviceBuffer attn;     // BF16 [257, 1280]
  DeviceBuffer linear;   // BF16 [257, 1280]
  DeviceBuffer mlp;      // BF16 [257, 5120]
};

ClipVision::ClipVision(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ClipVision::~ClipVision() = default;

std::size_t ClipVision::weight_bytes() const {
  return impl_->weight_bytes;
}

std::vector<float> ClipVision::Preprocess(std::span<const std::uint8_t> rgb,
                                          int width, int height) {
  // Horizontal pass, then vertical, each rounded to 8 bits (PIL).
  const Coefficients cx = MakeCoefficients(width, kImageSize);
  const Coefficients cy = MakeCoefficients(height, kImageSize);
  std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(height) *
                                       kImageSize * 3);
  for (int y = 0; y < height; ++y)
    for (int xx = 0; xx < kImageSize; ++xx)
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = std::int64_t{1} << 21;
        for (int x = 0; x < cx.count[xx]; ++x)
          sum += static_cast<std::int64_t>(
                     cx.weights[static_cast<std::size_t>(xx) * cx.ksize + x]) *
                 rgb[(static_cast<std::size_t>(y) * width + cx.first[xx] + x) *
                         3 +
                     c];
        horizontal[(static_cast<std::size_t>(y) * kImageSize + xx) * 3 + c] =
            Clip8(sum);
      }
  constexpr float kMean[3] = {0.48145466F, 0.4578275F, 0.40821073F};
  constexpr float kStd[3] = {0.26862954F, 0.26130258F, 0.27577711F};
  std::vector<float> pixels(static_cast<std::size_t>(3) * kImageSize *
                            kImageSize);
  for (int yy = 0; yy < kImageSize; ++yy)
    for (int x = 0; x < kImageSize; ++x)
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = std::int64_t{1} << 21;
        for (int y = 0; y < cy.count[yy]; ++y)
          sum += static_cast<std::int64_t>(
                     cy.weights[static_cast<std::size_t>(yy) * cy.ksize + y]) *
                 horizontal[(static_cast<std::size_t>(cy.first[yy] + y) *
                                 kImageSize +
                             x) *
                                3 +
                            c];
        const float unit = static_cast<float>(Clip8(sum)) / 255.0F;
        pixels[(static_cast<std::size_t>(c) * kImageSize + yy) * kImageSize +
               x] = (unit - kMean[c]) / kStd[c];
      }
  return pixels;
}

std::unique_ptr<ClipVision> ClipVision::Load(const std::filesystem::path& dir,
                                             std::string* error) {
  const auto path = dir / "model.safetensors";
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
  };
  std::map<std::string, Location, std::less<>> tensors;
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (entry.member_str("dtype") != "BF16" || shape == nullptr ||
          offsets == nullptr || offsets->size() != 2) {
        SetError(error, "unsupported CLIP tensor " + name);
        return nullptr;
      }
      std::uint64_t count = 1;
      for (const auto& dim : shape->items())
        count *= dim.as_size();
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end - begin != count * 2 || 8 + header_size + end > *file_size) {
        SetError(error, "invalid CLIP tensor extent " + name);
        return nullptr;
      }
      tensors[name] = {8 + header_size + begin, count};
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return nullptr;
  }
  const auto read = [&](const std::string& name, std::uint64_t count,
                        std::vector<std::uint16_t>* out) {
    const auto it = tensors.find(name);
    if (it == tensors.end() || it->second.count != count) {
      SetError(error, "missing or misshaped CLIP tensor " + name);
      return false;
    }
    out->resize(count);
    if (platform::ReadFileAt(file, out->data(), count * 2, it->second.offset) !=
        count * 2) {
      SetError(error, "short read of CLIP tensor " + name);
      return false;
    }
    return true;
  };
  const auto read_float = [&](const std::string& name, std::uint64_t count,
                              std::vector<float>* out) {
    std::vector<std::uint16_t> half;
    if (!read(name, count, &half))
      return false;
    out->resize(count);
    for (std::size_t i = 0; i < half.size(); ++i)
      (*out)[i] = HostFromBf16(half[i]);
    return true;
  };

  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  const std::string e = "vision_model.embeddings.";
  if (!read_float(e + "patch_embedding.weight", kWidth * 3 * kPatch * kPatch,
                  &m.patch_weight) ||
      !read_float(e + "class_embedding", kWidth, &m.class_embedding) ||
      !read_float(e + "position_embedding.weight",
                  static_cast<std::uint64_t>(kTokens) * kWidth, &m.position) ||
      !read_float("vision_model.pre_layrnorm.weight", kWidth, &m.pre_weight) ||
      !read_float("vision_model.pre_layrnorm.bias", kWidth, &m.pre_bias))
    return nullptr;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.gemm.Init(m.stream, error))
    return nullptr;
  if (rocblas_create_handle(&m.blas) != rocblas_status_success ||
      rocblas_set_stream(m.blas, m.stream) != rocblas_status_success) {
    SetError(error, "CLIP vision rocBLAS handle");
    return nullptr;
  }
  m.layers.resize(kLayersUsed);
  std::vector<std::uint16_t> a, b, c;
  std::vector<float> f;
  const auto upload_bf16 = [&](DeviceBuffer& buffer,
                               const std::vector<std::uint16_t>& host) {
    m.weight_bytes += host.size() * 2;
    return buffer.Upload(host.data(), host.size() * 2, error);
  };
  const auto upload_f32 = [&](DeviceBuffer& buffer,
                              const std::vector<float>& host) {
    m.weight_bytes += host.size() * 4;
    return buffer.Upload(host.data(), host.size() * 4, error);
  };
  for (int index = 0; index < kLayersUsed; ++index) {
    Layer& layer = m.layers[index];
    const std::string p =
        "vision_model.encoder.layers." + std::to_string(index) + ".";
    const std::uint64_t square = static_cast<std::uint64_t>(kWidth) * kWidth;
    const auto concat = [&](const char* kind, std::uint64_t count,
                            DeviceBuffer& target) {
      std::vector<std::uint16_t> all;
      for (const char* proj : {"q_proj", "k_proj", "v_proj"}) {
        if (!read(p + "self_attn." + proj + "." + kind, count, &a))
          return false;
        all.insert(all.end(), a.begin(), a.end());
      }
      return upload_bf16(target, all);
    };
    if (!read_float(p + "layer_norm1.weight", kWidth, &f) ||
        !upload_f32(layer.norm1_weight, f) ||
        !read_float(p + "layer_norm1.bias", kWidth, &f) ||
        !upload_f32(layer.norm1_bias, f) ||
        !read_float(p + "layer_norm2.weight", kWidth, &f) ||
        !upload_f32(layer.norm2_weight, f) ||
        !read_float(p + "layer_norm2.bias", kWidth, &f) ||
        !upload_f32(layer.norm2_bias, f) ||
        !concat("weight", square, layer.qkv) ||
        !concat("bias", kWidth, layer.qkv_bias) ||
        !read(p + "self_attn.out_proj.weight", square, &a) ||
        !upload_bf16(layer.out, a) ||
        !read(p + "self_attn.out_proj.bias", kWidth, &a) ||
        !upload_bf16(layer.out_bias, a) ||
        !read(p + "mlp.fc1.weight", static_cast<std::uint64_t>(kMlp) * kWidth,
              &a) ||
        !upload_bf16(layer.fc1, a) ||
        !read(p + "mlp.fc1.bias", kMlp, &a) || !upload_bf16(layer.fc1_bias, a) ||
        !read(p + "mlp.fc2.weight", static_cast<std::uint64_t>(kWidth) * kMlp,
              &a) ||
        !upload_bf16(layer.fc2, a) || !read(p + "mlp.fc2.bias", kWidth, &a) ||
        !upload_bf16(layer.fc2_bias, a))
      return nullptr;
  }
  const std::size_t T = kTokens, W = kWidth;
  if (!m.hidden.Allocate(T * W * 4, error) ||
      !m.normed.Allocate(T * W * 2, error) ||
      !m.qkv.Allocate(T * 3 * W * 2, error) ||
      !m.qkv32.Allocate(T * 3 * W * 4, error) ||
      !m.scores.Allocate(kHeads * T * T * 4, error) ||
      !m.attn32.Allocate(T * W * 4, error) ||
      !m.attn.Allocate(T * W * 2, error) ||
      !m.linear.Allocate(T * W * 2, error) ||
      !m.mlp.Allocate(T * kMlp * 2, error))
    return nullptr;
  return std::unique_ptr<ClipVision>(new ClipVision(std::move(impl)));
}

bool ClipVision::Impl::Run(std::string* error) {
  const std::size_t T = kTokens, W = kWidth;
  const std::size_t count = T * W;
  const float one = 1.0F, zero = 0.0F;
  for (const Layer& layer : layers) {
    LayerNormKernel<<<kTokens, kThreads, 0, stream>>>(
        hidden.f32(), layer.norm1_weight.f32(), layer.norm1_bias.f32(),
        normed.get());
    if (!gemm.Run(layer.qkv.get(), layer.qkv_bias.get(), normed.get(),
                  qkv.get(), kTokens, 3 * kWidth, kWidth, error))
      return false;
    WidenKernel<<<Blocks(3 * count), kThreads, 0, stream>>>(
        qkv.get(), 3 * count, qkv32.f32());
    // Per head (column-major view): scores[h] (keys x queries) = K_h^T Q_h,
    // then out_h (80 x queries) = V_h P_h.
    const float* q = qkv32.f32();
    const float* k = q + W;
    const float* v = q + 2 * W;
    if (rocblas_sgemm_strided_batched(
            blas, rocblas_operation_transpose, rocblas_operation_none, kTokens,
            kTokens, kHeadDim, &one, k, 3 * kWidth, kHeadDim, q, 3 * kWidth,
            kHeadDim, &zero, scores.f32(), kTokens,
            static_cast<rocblas_stride>(T * T), kHeads) !=
        rocblas_status_success) {
      SetError(error, "CLIP vision attention scores failed");
      return false;
    }
    SoftmaxKernel<<<kHeads * kTokens, kThreads, 0, stream>>>(
        scores.f32(), kTokens, 1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    if (rocblas_sgemm_strided_batched(
            blas, rocblas_operation_none, rocblas_operation_none, kHeadDim,
            kTokens, kTokens, &one, v, 3 * kWidth, kHeadDim, scores.f32(),
            kTokens, static_cast<rocblas_stride>(T * T), &zero, attn32.f32(),
            kWidth, kHeadDim, kHeads) != rocblas_status_success) {
      SetError(error, "CLIP vision attention values failed");
      return false;
    }
    NarrowKernel<<<Blocks(count), kThreads, 0, stream>>>(attn32.f32(), count,
                                                         attn.get());
    if (!gemm.Run(layer.out.get(), layer.out_bias.get(), attn.get(),
                  linear.get(), kTokens, kWidth, kWidth, error))
      return false;
    ResidualKernel<<<Blocks(count), kThreads, 0, stream>>>(linear.get(), count,
                                                           hidden.f32());
    LayerNormKernel<<<kTokens, kThreads, 0, stream>>>(
        hidden.f32(), layer.norm2_weight.f32(), layer.norm2_bias.f32(),
        normed.get());
    if (!gemm.Run(layer.fc1.get(), layer.fc1_bias.get(), normed.get(),
                  mlp.get(), kTokens, kMlp, kWidth, error))
      return false;
    GeluKernel<<<Blocks(T * kMlp), kThreads, 0, stream>>>(mlp.get(), T * kMlp);
    if (!gemm.Run(layer.fc2.get(), layer.fc2_bias.get(), mlp.get(),
                  linear.get(), kTokens, kWidth, kMlp, error))
      return false;
    ResidualKernel<<<Blocks(count), kThreads, 0, stream>>>(linear.get(), count,
                                                           hidden.f32());
  }
  return Check(hipGetLastError(), "layers", error);
}

bool ClipVision::Encode(std::span<const float> pixels,
                        std::vector<float>* states, std::string* error) {
  Impl& m = *impl_;
  constexpr int kSide = kImageSize;
  if (pixels.size() != static_cast<std::size_t>(3) * kSide * kSide) {
    SetError(error, "CLIP vision: pixels must be [3, 224, 224]");
    return false;
  }
  // Embeddings in FP32: class token, then the 16 x 16 patches (conv 14x14,
  // stride 14, no bias), plus positions, then the pre layer norm.
  constexpr int kPatchValues = 3 * kPatch * kPatch;
  std::vector<float> x(static_cast<std::size_t>(kTokens) * kWidth);
  std::vector<float> patch(kPatchValues);
  for (int c = 0; c < kWidth; ++c)
    x[c] = m.class_embedding[c];
  for (int gy = 0; gy < kGrid; ++gy)
    for (int gx = 0; gx < kGrid; ++gx) {
      for (int ch = 0; ch < 3; ++ch)
        for (int py = 0; py < kPatch; ++py)
          for (int px = 0; px < kPatch; ++px)
            patch[(ch * kPatch + py) * kPatch + px] =
                pixels[(static_cast<std::size_t>(ch) * kSide + gy * kPatch +
                        py) *
                           kSide +
                       gx * kPatch + px];
      float* row = x.data() + static_cast<std::size_t>(1 + gy * kGrid + gx) *
                                  kWidth;
      for (int o = 0; o < kWidth; ++o) {
        const float* w =
            m.patch_weight.data() + static_cast<std::size_t>(o) * kPatchValues;
        double sum = 0;
        for (int i = 0; i < kPatchValues; ++i)
          sum += static_cast<double>(w[i]) * patch[i];
        row[o] = static_cast<float>(sum);
      }
    }
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] += m.position[i];
  for (int t = 0; t < kTokens; ++t) {
    float* row = x.data() + static_cast<std::size_t>(t) * kWidth;
    double mean = 0, var = 0;
    for (int c = 0; c < kWidth; ++c)
      mean += row[c];
    mean /= kWidth;
    for (int c = 0; c < kWidth; ++c)
      var += (row[c] - mean) * (row[c] - mean);
    const double rstd = 1.0 / std::sqrt(var / kWidth + kEpsilon);
    for (int c = 0; c < kWidth; ++c)
      row[c] = static_cast<float>((row[c] - mean) * rstd * m.pre_weight[c] +
                                  m.pre_bias[c]);
  }
  states->resize(x.size());
  return Check(hipMemcpyAsync(m.hidden.f32(), x.data(), x.size() * 4,
                              hipMemcpyHostToDevice, m.stream),
               "input upload", error) &&
         m.Run(error) &&
         Check(hipMemcpyAsync(states->data(), m.hidden.f32(), x.size() * 4,
                              hipMemcpyDeviceToHost, m.stream),
               "readback", error) &&
         Check(hipStreamSynchronize(m.stream), "encode", error);
}

}  // namespace gufo::models::wan22
