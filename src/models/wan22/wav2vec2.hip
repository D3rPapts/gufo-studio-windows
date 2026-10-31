// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "src/models/wan22/tensor_file.hpp"
#include "src/models/wan22/wav2vec2.hpp"

// chinese-wav2vec2-base as ComfyUI's comfy/audio_encoders/wav2vec2.py runs it
// with the "base" config (conv_norm False, conv_bias False, do_normalize
// False, do_stable_layer_norm False).

namespace gufo::models::wan22 {
namespace {

constexpr int kConv = 512;
constexpr int kWidth = Wav2Vec2::kWidth;
constexpr int kHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kFfn = 3072;
constexpr int kPosKernel = 128;
constexpr int kPosGroups = 16;
constexpr int kPosGroupWidth = kWidth / kPosGroups;  // 48
constexpr float kEpsilon = 1e-5F;
constexpr int kThreads = 256;
// (kernel, stride) of the seven feature convolutions.
constexpr int kKernels[7] = {10, 3, 3, 3, 3, 2, 2};
constexpr int kStrides[7] = {5, 2, 2, 2, 2, 2, 2};

static_assert(kWidth == kHeads * kHeadDim);

__device__ __forceinline__ float Gelu(float v) {
  return 0.5F * v * (1.0F + erff(v * 0.7071067811865476F));
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

// out[t][o] = sum_j x[5t + j] w[o][j] (first convolution, no bias).
__global__ void Conv0Kernel(const float* x, const float* w, int frames,
                            float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(frames) * kConv)
    return;
  const std::size_t t = i / kConv;
  const int o = static_cast<int>(i % kConv);
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 10; ++j)
    sum += x[5 * t + j] * w[o * 10 + j];
  out[i] = sum;
}

// GroupNorm with one channel per group over time: mean and 1/std of channel
// blockIdx.x of a time-major [frames][512] input.
__global__ void ChannelStatsKernel(const float* x, int frames, float* stats) {
  __shared__ float shared[kThreads];
  const int c = blockIdx.x;
  float sum = 0;
  for (int t = threadIdx.x; t < frames; t += kThreads)
    sum += x[static_cast<std::size_t>(t) * kConv + c];
  const float mean = BlockSum(sum, shared) / frames;
  float squares = 0;
  for (int t = threadIdx.x; t < frames; t += kThreads) {
    const float d = x[static_cast<std::size_t>(t) * kConv + c] - mean;
    squares += d * d;
  }
  const float var = BlockSum(squares, shared) / frames;
  if (threadIdx.x == 0) {
    stats[2 * c] = mean;
    stats[2 * c + 1] = rsqrtf(var + kEpsilon);
  }
}

__global__ void ChannelNormGeluKernel(float* x, int frames, const float* stats,
                                      const float* weight, const float* bias) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(frames) * kConv)
    return;
  const int c = static_cast<int>(i % kConv);
  x[i] = Gelu((x[i] - stats[2 * c]) * stats[2 * c + 1] * weight[c] + bias[c]);
}

__global__ void GeluKernel(float* x, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = Gelu(x[i]);
}

// x[t][c] += bias[c] for rows of `width`.
__global__ void BiasKernel(float* x, const float* bias, int width,
                           std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += bias[i % width];
}

// out = layer_norm(x (+ residual)) * weight + bias, rows of `width`.
__global__ void LayerNormKernel(const float* x, const float* residual,
                                int width, const float* weight,
                                const float* bias, float* out) {
  __shared__ float shared[kThreads];
  const std::size_t row = blockIdx.x;
  const float* a = x + row * width;
  const float* r = residual == nullptr ? nullptr : residual + row * width;
  float sum = 0;
  for (int c = threadIdx.x; c < width; c += kThreads)
    sum += a[c] + (r != nullptr ? r[c] : 0.0F);
  const float mean = BlockSum(sum, shared) / width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += kThreads) {
    const float d = a[c] + (r != nullptr ? r[c] : 0.0F) - mean;
    squares += d * d;
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / width + kEpsilon);
  float* o = out + row * width;
  for (int c = threadIdx.x; c < width; c += kThreads)
    o[c] = (a[c] + (r != nullptr ? r[c] : 0.0F) - mean) * rstd * weight[c] +
           bias[c];
}

// Grouped positional convolution (kernel 128, padding 64, last output
// dropped), then x += gelu(conv). One thread per output value.
__global__ void PositionalKernel(const float* x, const float* w,
                                 const float* bias, int frames, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(frames) * kWidth)
    return;
  const int t = static_cast<int>(i / kWidth);
  const int o = static_cast<int>(i % kWidth);
  const int group = o / kPosGroupWidth;
  const float* wo = w + static_cast<std::size_t>(o) * kPosGroupWidth *
                            kPosKernel;
  float sum = bias[o];
  const int lead = kPosKernel / 2 - t;
  const int tail = frames + kPosKernel / 2 - t;
  const int first = lead > 0 ? lead : 0;
  const int last = tail < kPosKernel ? tail : kPosKernel;
  for (int c = 0; c < kPosGroupWidth; ++c) {
    const float* xc = x + group * kPosGroupWidth + c;
    const float* wc = wo + c * kPosKernel;
    for (int k = first; k < last; ++k)
      sum += xc[static_cast<std::size_t>(t + k - kPosKernel / 2) * kWidth] *
             wc[k];
  }
  out[i] = x[i] + Gelu(sum);
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
  SetError(error, std::string("wav2vec2 ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
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
  bool Upload(const std::vector<float>& host, std::string* error) {
    return Allocate(host.size() * 4, error) &&
           Check(hipMemcpy(data_, host.data(), host.size() * 4,
                           hipMemcpyHostToDevice),
                 "weight upload", error);
  }
  float* f32() const { return static_cast<float*>(data_); }

private:
  void* data_{nullptr};
};

struct Layer {
  DeviceBuffer qkv, qkv_bias;  // [3 * 768, 768] (q, k, v)
  DeviceBuffer out, out_bias;
  DeviceBuffer norm_weight, norm_bias;
  DeviceBuffer up, up_bias, down, down_bias;
  DeviceBuffer final_weight, final_bias;
};

}  // namespace

struct Wav2Vec2::Impl {
  ~Impl() {
    if (blas != nullptr)
      (void)rocblas_destroy_handle(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
  // out[T, N] = in[T, K] * W[N, K]^T (+ bias).
  bool Linear(const float* in, const DeviceBuffer& weight,
              const DeviceBuffer* bias, int tokens, int n, int k, float* out,
              std::string* error) {
    const float one = 1.0F, zero = 0.0F;
    if (rocblas_sgemm(blas, rocblas_operation_transpose,
                      rocblas_operation_none, n, tokens, k, &one,
                      weight.f32(), k, in, k, &zero, out,
                      n) != rocblas_status_success) {
      SetError(error, "wav2vec2 GEMM failed");
      return false;
    }
    if (bias != nullptr) {
      const std::size_t count = static_cast<std::size_t>(tokens) * n;
      BiasKernel<<<Blocks(count), kThreads, 0, stream>>>(out, bias->f32(), n,
                                                         count);
    }
    return true;
  }

  DeviceBuffer conv0, conv0_norm_weight, conv0_norm_bias;  // [512, 10]
  DeviceBuffer convs[6];  // [kernel][512 out][512 in]
  DeviceBuffer projection_norm_weight, projection_norm_bias;
  DeviceBuffer projection, projection_bias;  // [768, 512]
  DeviceBuffer positional, positional_bias;  // [768, 48, 128] (weight norm)
  DeviceBuffer encoder_norm_weight, encoder_norm_bias;
  std::vector<Layer> layers;
  std::size_t weight_bytes{0};
  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
};

Wav2Vec2::Wav2Vec2(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Wav2Vec2::~Wav2Vec2() = default;

std::size_t Wav2Vec2::weight_bytes() const {
  return impl_->weight_bytes;
}

int Wav2Vec2::FramesFor(std::size_t samples) {
  long long length = static_cast<long long>(samples);
  for (int i = 0; i < 7; ++i) {
    if (length < kKernels[i])
      return 0;
    length = (length - kKernels[i]) / kStrides[i] + 1;
  }
  return static_cast<int>(length);
}

std::unique_ptr<Wav2Vec2> Wav2Vec2::Load(const std::filesystem::path& file,
                                         std::string* error) {
  TensorFile tensors;
  if (!tensors.Open(file, error))
    return nullptr;
  tensors.set_prefix("wav2vec2.");
  if (!tensors.Has("encoder.layer_norm.bias") ||
      tensors.Shape("encoder.layer_norm.bias") !=
          std::vector<std::size_t>{static_cast<std::size_t>(kWidth)} ||
      !tensors.Has("feature_extractor.conv_layers.0.layer_norm.weight")) {
    SetError(error, file.string() +
                        " is not a wav2vec2-base checkpoint (768 wide, "
                        "GroupNorm feature extractor)");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error))
    return nullptr;
  if (rocblas_create_handle(&m.blas) != rocblas_status_success ||
      rocblas_set_stream(m.blas, m.stream) != rocblas_status_success) {
    SetError(error, "wav2vec2 rocBLAS handle");
    return nullptr;
  }
  std::vector<float> host;
  const auto load = [&](DeviceBuffer& buffer, const std::string& name,
                        std::size_t count) {
    if (!tensors.ReadF32(name, count, &host, error) ||
        !buffer.Upload(host, error))
      return false;
    m.weight_bytes += host.size() * 4;
    return true;
  };
  const std::string f = "feature_extractor.conv_layers.";
  if (!load(m.conv0, f + "0.conv.weight", kConv * 10) ||
      !load(m.conv0_norm_weight, f + "0.layer_norm.weight", kConv) ||
      !load(m.conv0_norm_bias, f + "0.layer_norm.bias", kConv))
    return nullptr;
  for (int i = 1; i < 7; ++i) {
    const int k = kKernels[i];
    std::vector<float> weight;
    if (!tensors.ReadF32(f + std::to_string(i) + ".conv.weight",
                         static_cast<std::size_t>(kConv) * kConv * k, &weight,
                         error))
      return nullptr;
    // [out][in][k] -> [k][out][in].
    host.assign(weight.size(), 0.0F);
    for (int o = 0; o < kConv; ++o)
      for (int c = 0; c < kConv; ++c)
        for (int j = 0; j < k; ++j)
          host[(static_cast<std::size_t>(j) * kConv + o) * kConv + c] =
              weight[(static_cast<std::size_t>(o) * kConv + c) * k + j];
    if (!m.convs[i - 1].Upload(host, error))
      return nullptr;
    m.weight_bytes += host.size() * 4;
  }
  if (!load(m.projection_norm_weight, "feature_projection.layer_norm.weight",
            kConv) ||
      !load(m.projection_norm_bias, "feature_projection.layer_norm.bias",
            kConv) ||
      !load(m.projection, "feature_projection.projection.weight",
            static_cast<std::size_t>(kWidth) * kConv) ||
      !load(m.projection_bias, "feature_projection.projection.bias", kWidth) ||
      !load(m.positional_bias, "encoder.pos_conv_embed.conv.bias", kWidth) ||
      !load(m.encoder_norm_weight, "encoder.layer_norm.weight", kWidth) ||
      !load(m.encoder_norm_bias, "encoder.layer_norm.bias", kWidth))
    return nullptr;
  // Weight norm over the kernel axis: w[:, :, k] = g[k] v[:, :, k] /
  // ||v[:, :, k]||. Older exports name them weight_g / weight_v, torch's
  // parametrization original0 / original1.
  {
    const std::string p = "encoder.pos_conv_embed.conv.";
    const std::string g_name = tensors.Has(p + "weight_g")
                                   ? p + "weight_g"
                                   : p + "parametrizations.weight.original0";
    const std::string v_name = tensors.Has(p + "weight_v")
                                   ? p + "weight_v"
                                   : p + "parametrizations.weight.original1";
    std::vector<float> g, v;
    const std::size_t count =
        static_cast<std::size_t>(kWidth) * kPosGroupWidth * kPosKernel;
    if (!tensors.ReadF32(g_name, kPosKernel, &g, error) ||
        !tensors.ReadF32(v_name, count, &v, error))
      return nullptr;
    std::vector<double> norm(kPosKernel, 0.0);
    for (std::size_t i = 0; i < count; ++i)
      norm[i % kPosKernel] += static_cast<double>(v[i]) * v[i];
    host.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      const int k = static_cast<int>(i % kPosKernel);
      host[i] = static_cast<float>(g[k] * v[i] / std::sqrt(norm[k]));
    }
    if (!m.positional.Upload(host, error))
      return nullptr;
    m.weight_bytes += host.size() * 4;
  }
  m.layers.resize(Wav2Vec2::kLayers);
  for (int index = 0; index < Wav2Vec2::kLayers; ++index) {
    Layer& layer = m.layers[index];
    const std::string p = "encoder.layers." + std::to_string(index) + ".";
    std::vector<float> all, bias_all;
    for (const char* proj : {"q_proj", "k_proj", "v_proj"}) {
      if (!tensors.ReadF32(p + "attention." + proj + ".weight",
                           static_cast<std::size_t>(kWidth) * kWidth, &host,
                           error))
        return nullptr;
      all.insert(all.end(), host.begin(), host.end());
      if (!tensors.ReadF32(p + "attention." + proj + ".bias", kWidth, &host,
                           error))
        return nullptr;
      bias_all.insert(bias_all.end(), host.begin(), host.end());
    }
    if (!layer.qkv.Upload(all, error) ||
        !layer.qkv_bias.Upload(bias_all, error))
      return nullptr;
    m.weight_bytes += (all.size() + bias_all.size()) * 4;
    if (!load(layer.out, p + "attention.out_proj.weight",
              static_cast<std::size_t>(kWidth) * kWidth) ||
        !load(layer.out_bias, p + "attention.out_proj.bias", kWidth) ||
        !load(layer.norm_weight, p + "layer_norm.weight", kWidth) ||
        !load(layer.norm_bias, p + "layer_norm.bias", kWidth) ||
        !load(layer.up, p + "feed_forward.intermediate_dense.weight",
              static_cast<std::size_t>(kFfn) * kWidth) ||
        !load(layer.up_bias, p + "feed_forward.intermediate_dense.bias",
              kFfn) ||
        !load(layer.down, p + "feed_forward.output_dense.weight",
              static_cast<std::size_t>(kWidth) * kFfn) ||
        !load(layer.down_bias, p + "feed_forward.output_dense.bias", kWidth) ||
        !load(layer.final_weight, p + "final_layer_norm.weight", kWidth) ||
        !load(layer.final_bias, p + "final_layer_norm.bias", kWidth))
      return nullptr;
  }
  return std::unique_ptr<Wav2Vec2>(new Wav2Vec2(std::move(impl)));
}

bool Wav2Vec2::Encode(std::span<const float> audio, std::vector<float>* layers,
                      int* frames, std::string* error) {
  Impl& m = *impl_;
  const int T = FramesFor(audio.size());
  if (T < 1 || T > kMaxFrames) {
    SetError(error, "wav2vec2: audio must be 0.03..80 seconds at 16 kHz");
    return false;
  }
  // Lengths after each convolution.
  int lengths[8];
  lengths[0] = static_cast<int>(audio.size());
  for (int i = 0; i < 7; ++i)
    lengths[i + 1] = (lengths[i] - kKernels[i]) / kStrides[i] + 1;
  const std::size_t first = static_cast<std::size_t>(lengths[1]);
  hipStream_t s = m.stream;
  DeviceBuffer wave, a, b, stats;
  if (!wave.Allocate(audio.size() * 4, error) ||
      !a.Allocate(first * kConv * 4, error) ||
      !b.Allocate(first * kConv * 4, error) ||
      !stats.Allocate(2 * kConv * 4, error) ||
      !Check(hipMemcpyAsync(wave.f32(), audio.data(), audio.size() * 4,
                            hipMemcpyHostToDevice, s),
             "input upload", error))
    return false;
  // 1. Feature extractor (time-major [frames][512]).
  Conv0Kernel<<<Blocks(first * kConv), kThreads, 0, s>>>(
      wave.f32(), m.conv0.f32(), lengths[1], a.f32());
  ChannelStatsKernel<<<kConv, kThreads, 0, s>>>(a.f32(), lengths[1],
                                                stats.f32());
  ChannelNormGeluKernel<<<Blocks(first * kConv), kThreads, 0, s>>>(
      a.f32(), lengths[1], stats.f32(), m.conv0_norm_weight.f32(),
      m.conv0_norm_bias.f32());
  float* in = a.f32();
  float* out = b.f32();
  const float one = 1.0F, zero = 0.0F;
  for (int i = 1; i < 7; ++i) {
    const int k = kKernels[i];
    const int length = lengths[i + 1];
    // out^T (512 x L) = sum_j W_j (512 x 512) * x[2t + j] (512 x L).
    for (int j = 0; j < k; ++j) {
      if (rocblas_sgemm(m.blas, rocblas_operation_transpose,
                        rocblas_operation_none, kConv, length, kConv, &one,
                        m.convs[i - 1].f32() +
                            static_cast<std::size_t>(j) * kConv * kConv,
                        kConv, in + static_cast<std::size_t>(j) * kConv,
                        kStrides[i] * kConv, j == 0 ? &zero : &one, out,
                        kConv) != rocblas_status_success) {
        SetError(error, "wav2vec2 convolution GEMM failed");
        return false;
      }
    }
    GeluKernel<<<Blocks(static_cast<std::size_t>(length) * kConv), kThreads,
                 0, s>>>(out, static_cast<std::size_t>(length) * kConv);
    std::swap(in, out);
  }
  // 2. Feature projection, positional convolution, encoder norm.
  const std::size_t count = static_cast<std::size_t>(T) * kWidth;
  DeviceBuffer hidden, scratch, normed, qkv, scores, attention, ffn;
  if (!hidden.Allocate(count * 4, error) ||
      !scratch.Allocate(count * 4, error) ||
      !normed.Allocate(static_cast<std::size_t>(T) * kConv * 4, error) ||
      !qkv.Allocate(3 * count * 4, error) ||
      !scores.Allocate(static_cast<std::size_t>(kHeads) * T * T * 4, error) ||
      !attention.Allocate(count * 4, error) ||
      !ffn.Allocate(static_cast<std::size_t>(T) * kFfn * 4, error))
    return false;
  LayerNormKernel<<<T, kThreads, 0, s>>>(in, nullptr, kConv,
                                         m.projection_norm_weight.f32(),
                                         m.projection_norm_bias.f32(),
                                         normed.f32());
  if (!m.Linear(normed.f32(), m.projection, &m.projection_bias, T, kWidth,
                kConv, scratch.f32(), error))
    return false;
  PositionalKernel<<<Blocks(count), kThreads, 0, s>>>(
      scratch.f32(), m.positional.f32(), m.positional_bias.f32(), T,
      attention.f32());
  LayerNormKernel<<<T, kThreads, 0, s>>>(attention.f32(), nullptr, kWidth,
                                         m.encoder_norm_weight.f32(),
                                         m.encoder_norm_bias.f32(),
                                         hidden.f32());
  // 3. Post-norm layers; keep every layer's output.
  layers->assign(static_cast<std::size_t>(kLayers) * count, 0.0F);
  for (int index = 0; index < kLayers; ++index) {
    Layer& layer = m.layers[index];
    if (!m.Linear(hidden.f32(), layer.qkv, &layer.qkv_bias, T, 3 * kWidth,
                  kWidth, qkv.f32(), error))
      return false;
    const float* q = qkv.f32();
    const float* k = q + kWidth;
    const float* v = q + 2 * kWidth;
    if (rocblas_sgemm_strided_batched(
            m.blas, rocblas_operation_transpose, rocblas_operation_none, T, T,
            kHeadDim, &one, k, 3 * kWidth, kHeadDim, q, 3 * kWidth, kHeadDim,
            &zero, scores.f32(), T, static_cast<rocblas_stride>(T) * T,
            kHeads) != rocblas_status_success) {
      SetError(error, "wav2vec2 attention scores failed");
      return false;
    }
    SoftmaxKernel<<<kHeads * T, kThreads, 0, s>>>(
        scores.f32(), T, 1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    if (rocblas_sgemm_strided_batched(
            m.blas, rocblas_operation_none, rocblas_operation_none, kHeadDim,
            T, T, &one, v, 3 * kWidth, kHeadDim, scores.f32(), T,
            static_cast<rocblas_stride>(T) * T, &zero, attention.f32(),
            kWidth, kHeadDim, kHeads) != rocblas_status_success) {
      SetError(error, "wav2vec2 attention values failed");
      return false;
    }
    if (!m.Linear(attention.f32(), layer.out, &layer.out_bias, T, kWidth,
                  kWidth, scratch.f32(), error))
      return false;
    // x = layer_norm(x + attention); x = final_norm(x + ffn(x)).
    LayerNormKernel<<<T, kThreads, 0, s>>>(hidden.f32(), scratch.f32(), kWidth,
                                           layer.norm_weight.f32(),
                                           layer.norm_bias.f32(),
                                           attention.f32());
    if (!m.Linear(attention.f32(), layer.up, &layer.up_bias, T, kFfn, kWidth,
                  ffn.f32(), error))
      return false;
    GeluKernel<<<Blocks(static_cast<std::size_t>(T) * kFfn), kThreads, 0, s>>>(
        ffn.f32(), static_cast<std::size_t>(T) * kFfn);
    if (!m.Linear(ffn.f32(), layer.down, &layer.down_bias, T, kWidth, kFfn,
                  scratch.f32(), error))
      return false;
    LayerNormKernel<<<T, kThreads, 0, s>>>(attention.f32(), scratch.f32(),
                                           kWidth, layer.final_weight.f32(),
                                           layer.final_bias.f32(),
                                           hidden.f32());
    if (!Check(hipGetLastError(), "layer launch", error) ||
        !Check(hipMemcpyAsync(layers->data() + index * count, hidden.f32(),
                              count * 4, hipMemcpyDeviceToHost, s),
               "readback", error))
      return false;
  }
  *frames = T;
  return Check(hipStreamSynchronize(s), "encode", error);
}

std::vector<float> TalkAudioFeatures(std::span<const float> layers,
                                     int frames50, int* frames25) {
  constexpr int kLayers = Wav2Vec2::kLayers;
  // int(T / 50.0 * 25), as WanInfiniteTalkToVideo's linear_interpolation.
  const int out = static_cast<int>(static_cast<double>(frames50) / 50.0 * 25.0);
  *frames25 = out;
  std::vector<float> features(static_cast<std::size_t>(std::max(out, 0)) *
                              kLayers * kWidth);
  for (int i = 0; i < out; ++i) {
    // align_corners: source = i * (in - 1) / (out - 1).
    const double position =
        out > 1 ? static_cast<double>(i) * (frames50 - 1) / (out - 1) : 0.0;
    const int low = std::min(static_cast<int>(position), frames50 - 1);
    const int high = std::min(low + 1, frames50 - 1);
    const float t = static_cast<float>(position - low);
    for (int l = 0; l < kLayers; ++l) {
      const float* a =
          layers.data() + (static_cast<std::size_t>(l) * frames50 + low) *
                              kWidth;
      const float* b =
          layers.data() + (static_cast<std::size_t>(l) * frames50 + high) *
                              kWidth;
      float* target =
          features.data() + (static_cast<std::size_t>(i) * kLayers + l) *
                                kWidth;
      for (int c = 0; c < kWidth; ++c)
        target[c] = a[c] + (b[c] - a[c]) * t;
    }
  }
  return features;
}

}  // namespace gufo::models::wan22
