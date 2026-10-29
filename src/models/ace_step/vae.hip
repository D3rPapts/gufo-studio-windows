// Oobleck VAE decoder of ACE-Step 1.5 (Diffusers AutoencoderOobleck):
//   conv1 (64 -> 2048, k7) -> 5 x [Snake, ConvTranspose (k 2s, stride s,
//   pad s/2), 3 residual units (Snake, conv k7 dilation 1/3/9, Snake, conv
//   k1)] -> Snake -> conv2 (128 -> 2, k7, no bias).
// Activations are channels-last FP32 [samples, channels]; convolutions are
// im2col + BF16 GEMMs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "src/models/ace_step/gemm.hpp"
#include "src/models/ace_step/model.hpp"
#include "src/models/ace_step/vae.hpp"
#include "src/models/ace_step/weights.hpp"

namespace gufo::models::ace_step {
namespace {

constexpr int kThreads = 256;
constexpr int kBlocks = 5;
constexpr int kStrides[kBlocks] = {10, 6, 4, 4, 2};
constexpr int kChunkFrames = 128;
// Receptive field of the decoder is below 10 latent frames on either side.
constexpr int kOverlapFrames = 16;
constexpr std::size_t kIm2colBytes = 256U << 20U;

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

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// out = bf16(x + inv_beta * sin(alpha * x)^2), per channel alpha / inv_beta.
__global__ void SnakeKernel(const float* x, const float* alpha,
                            const float* inv_beta, int channels,
                            std::size_t count, std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % channels);
  const float v = x[i];
  const float s = sinf(alpha[c] * v);
  out[i] = ToBf16(v + inv_beta[c] * s * s);
}

// cols[r, k * C + c] = in[row0 + r + k * dilation - pad, c] (zero outside).
__global__ void Im2colKernel(const std::uint16_t* in, int length, int channels,
                             int taps, int dilation, int pad, int row0,
                             std::size_t count, std::uint16_t* cols) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t width = static_cast<std::size_t>(taps) * channels;
  const int r = static_cast<int>(i / width);
  const int column = static_cast<int>(i % width);
  const int k = column / channels, c = column % channels;
  const int t = row0 + r + k * dilation - pad;
  cols[i] = t >= 0 && t < length
                ? in[static_cast<std::size_t>(t) * channels + c]
                : std::uint16_t{0};
}

// out[i] = y[i] + bias[c] (+ residual[i]).
__global__ void EpilogueKernel(const std::uint16_t* y, const float* bias,
                               const float* residual, int channels,
                               std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float v = FromBf16(y[i]);
  if (bias != nullptr)
    v += bias[i % channels];
  if (residual != nullptr)
    v += residual[i];
  out[i] = v;
}

// Transposed convolution (kernel 2s, stride s, pad p) from the per-input
// products z[j, k * O + o] = sum_c x[j, c] W[c, o, k]:
// out[n, o] = bias[o] + z[j1, k1 O + o] + z[j1 - 1, (k1 + s) O + o],
// j1 = (n + p) / s, k1 = (n + p) % s.
__global__ void TransposeScatterKernel(const std::uint16_t* z, int length,
                                       int stride, int channels, int pad,
                                       const float* bias, std::size_t count,
                                       float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int o = static_cast<int>(i % channels);
  const int n = static_cast<int>(i / channels);
  const int j1 = (n + pad) / stride, k1 = (n + pad) % stride;
  const std::size_t width = static_cast<std::size_t>(2 * stride) * channels;
  float v = bias[o];
  if (j1 < length)
    v += FromBf16(z[static_cast<std::size_t>(j1) * width +
                    static_cast<std::size_t>(k1) * channels + o]);
  if (j1 >= 1 && j1 - 1 < length)
    v += FromBf16(z[static_cast<std::size_t>(j1 - 1) * width +
                    static_cast<std::size_t>(k1 + stride) * channels + o]);
  out[i] = v;
}

struct Conv {
  DeviceBuffer weight;  // BF16 [out, taps * in] (tap-major columns)
  DeviceBuffer bias;    // FP32 [out] (may be empty)
  int in{0}, out{0}, taps{0}, dilation{1};
  bool has_bias{false};
};

struct TransposedConv {
  DeviceBuffer weight;  // BF16 [2 s * out, in], rows k * out + o
  DeviceBuffer bias;    // FP32 [out]
  int in{0}, out{0}, stride{0};
};

struct Snake {
  DeviceBuffer alpha, inv_beta;  // FP32 [channels]
};

struct ResidualUnit {
  Snake snake1, snake2;
  Conv conv1, conv2;
};

struct Block {
  Snake snake;
  TransposedConv up;
  ResidualUnit units[3];
};

bool UploadF32(DeviceBuffer& buffer, const std::vector<float>& host,
               std::string* error) {
  return buffer.Allocate(host.size() * 4, error) &&
         Check(hipMemcpy(buffer.get(), host.data(), host.size() * 4,
                         hipMemcpyHostToDevice),
               "VAE upload", error);
}

bool UploadBf16(DeviceBuffer& buffer, const std::vector<float>& host,
                std::string* error) {
  std::vector<std::uint16_t> bits(host.size());
  for (std::size_t i = 0; i < host.size(); ++i)
    bits[i] = HostBf16(host[i]);
  return buffer.Allocate(bits.size() * 2, error) &&
         Check(hipMemcpy(buffer.get(), bits.data(), bits.size() * 2,
                         hipMemcpyHostToDevice),
               "VAE upload", error);
}

// Folds weight norm: w = g * v / ||v||, the norm over all dims but the first.
bool ReadWeightNorm(SafetensorsFile& file, const std::string& p,
                    std::vector<float>* w, std::vector<std::size_t>* shape,
                    std::string* error) {
  const auto* v = file.Find(p + ".weight_v");
  if (v == nullptr || v->shape.size() != 3)
    return Fail(error, "missing VAE tensor " + p + ".weight_v");
  *shape = v->shape;
  std::vector<float> g;
  if (!file.ReadHost(p + ".weight_v", v->count(), w, error) ||
      !file.ReadHost(p + ".weight_g", v->shape[0], &g, error))
    return false;
  const std::size_t inner = v->shape[1] * v->shape[2];
  for (std::size_t a = 0; a < v->shape[0]; ++a) {
    double norm = 0;
    for (std::size_t i = 0; i < inner; ++i)
      norm += double((*w)[a * inner + i]) * (*w)[a * inner + i];
    const float scale = static_cast<float>(g[a] / std::sqrt(norm));
    for (std::size_t i = 0; i < inner; ++i)
      (*w)[a * inner + i] *= scale;
  }
  return true;
}

bool LoadConv(SafetensorsFile& file, const std::string& p, int dilation,
              bool bias, Conv& conv, std::string* error) {
  std::vector<float> w;
  std::vector<std::size_t> shape;
  if (!ReadWeightNorm(file, p, &w, &shape, error))
    return false;
  conv.out = int(shape[0]);
  conv.in = int(shape[1]);
  conv.taps = int(shape[2]);
  conv.dilation = dilation;
  // [o][c][k] -> [o][k * in + c].
  std::vector<float> reordered(w.size());
  for (int o = 0; o < conv.out; ++o)
    for (int c = 0; c < conv.in; ++c)
      for (int k = 0; k < conv.taps; ++k)
        reordered[(std::size_t(o) * conv.taps + k) * conv.in + c] =
            w[(std::size_t(o) * conv.in + c) * conv.taps + k];
  if (!UploadBf16(conv.weight, reordered, error))
    return false;
  conv.has_bias = bias;
  if (bias) {
    std::vector<float> b;
    if (!file.ReadHost(p + ".bias", conv.out, &b, error) ||
        !UploadF32(conv.bias, b, error))
      return false;
  }
  return true;
}

bool LoadTransposed(SafetensorsFile& file, const std::string& p, int stride,
                    TransposedConv& conv, std::string* error) {
  std::vector<float> w;
  std::vector<std::size_t> shape;
  if (!ReadWeightNorm(file, p, &w, &shape, error))
    return false;
  conv.in = int(shape[0]);
  conv.out = int(shape[1]);
  conv.stride = stride;
  if (int(shape[2]) != 2 * stride)
    return Fail(error, "unexpected VAE transposed kernel " + p);
  // [c][o][k] -> rows k * out + o, columns c.
  const int taps = 2 * stride;
  std::vector<float> reordered(w.size());
  for (int c = 0; c < conv.in; ++c)
    for (int o = 0; o < conv.out; ++o)
      for (int k = 0; k < taps; ++k)
        reordered[(std::size_t(k) * conv.out + o) * conv.in + c] =
            w[(std::size_t(c) * conv.out + o) * taps + k];
  std::vector<float> b;
  return UploadBf16(conv.weight, reordered, error) &&
         file.ReadHost(p + ".bias", conv.out, &b, error) &&
         UploadF32(conv.bias, b, error);
}

bool LoadSnake(SafetensorsFile& file, const std::string& p, Snake& snake,
               std::string* error) {
  const auto* a = file.Find(p + ".alpha");
  if (a == nullptr)
    return Fail(error, "missing VAE tensor " + p + ".alpha");
  const std::size_t channels = a->count();
  std::vector<float> alpha, beta;
  if (!file.ReadHost(p + ".alpha", channels, &alpha, error) ||
      !file.ReadHost(p + ".beta", channels, &beta, error))
    return false;
  for (std::size_t c = 0; c < channels; ++c) {
    alpha[c] = std::exp(alpha[c]);
    beta[c] = 1.0F / (std::exp(beta[c]) + 1e-9F);
  }
  return UploadF32(snake.alpha, alpha, error) &&
         UploadF32(snake.inv_beta, beta, error);
}

}  // namespace

struct VaeDecoder::Impl {
  hipStream_t stream{nullptr};
  LinearGemm gemm;
  Conv conv1, conv2;
  Block blocks[kBlocks];
  Snake snake;
  DeviceBuffer x, t, act, cols, z, gemm_out, latents;

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
  }

  void ApplySnake(const Snake& s, const float* in, int length, int channels) {
    const std::size_t count = std::size_t(length) * channels;
    SnakeKernel<<<Blocks(count), kThreads, 0, stream>>>(
        in, s.alpha.f32(), s.inv_beta.f32(), channels, count, act.bf16());
  }

  // out = conv(act [length, in]) + bias (+ residual), "same" padding.
  bool Convolve(const Conv& conv, int length, float* out, const float* residual,
                std::string* error) {
    const int pad = (conv.taps - 1) * conv.dilation / 2;
    const std::size_t width = std::size_t(conv.taps) * conv.in;
    const int chunk = conv.taps == 1 ? length
                                     : static_cast<int>(std::max<std::size_t>(
                                           64, kIm2colBytes / (width * 2)));
    for (int row0 = 0; row0 < length; row0 += chunk) {
      const int rows = std::min(chunk, length - row0);
      const std::uint16_t* input = act.bf16();
      if (conv.taps != 1) {
        const std::size_t count = std::size_t(rows) * width;
        Im2colKernel<<<Blocks(count), kThreads, 0, stream>>>(
            act.bf16(), length, conv.in, conv.taps, conv.dilation, pad, row0,
            count, cols.bf16());
        input = cols.bf16();
      } else {
        input = act.bf16() + std::size_t(row0) * conv.in;
      }
      if (!gemm.Run(conv.weight.bf16(), nullptr, input, gemm_out.bf16(), rows,
                    conv.out, static_cast<int>(width), error))
        return false;
      const std::size_t count = std::size_t(rows) * conv.out;
      const std::size_t offset = std::size_t(row0) * conv.out;
      EpilogueKernel<<<Blocks(count), kThreads, 0, stream>>>(
          gemm_out.bf16(), conv.has_bias ? conv.bias.f32() : nullptr,
          residual != nullptr ? residual + offset : nullptr, conv.out, count,
          out + offset);
    }
    return true;
  }

  // Decodes latent frames [frames, 64] (FP32, on the device in `latents`)
  // into x [frames * 1920, 2].
  bool Run(int frames, std::string* error) {
    int length = frames;
    // conv1 reads the latents, uploaded as BF16 into `act`.
    if (!Convolve(conv1, length, x.f32(), nullptr, error))
      return false;
    int channels = conv1.out;
    for (int b = 0; b < kBlocks; ++b) {
      Block& block = blocks[b];
      ApplySnake(block.snake, x.f32(), length, channels);
      const int s = block.up.stride;
      if (!gemm.Run(block.up.weight.bf16(), nullptr, act.bf16(), z.bf16(),
                    length, 2 * s * block.up.out, channels, error))
        return false;
      const std::size_t count = std::size_t(length) * s * block.up.out;
      TransposeScatterKernel<<<Blocks(count), kThreads, 0, stream>>>(
          z.bf16(), length, s, block.up.out, (s + 1) / 2, block.up.bias.f32(),
          count, x.f32());
      length *= s;
      channels = block.up.out;
      for (ResidualUnit& unit : block.units) {
        ApplySnake(unit.snake1, x.f32(), length, channels);
        if (!Convolve(unit.conv1, length, t.f32(), nullptr, error))
          return false;
        ApplySnake(unit.snake2, t.f32(), length, channels);
        if (!Convolve(unit.conv2, length, x.f32(), x.f32(), error))
          return false;
      }
    }
    ApplySnake(snake, x.f32(), length, channels);
    return Convolve(conv2, length, t.f32(), nullptr, error);
  }
};

VaeDecoder::VaeDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VaeDecoder::~VaeDecoder() = default;

std::unique_ptr<VaeDecoder> VaeDecoder::Load(
    const std::filesystem::path& safetensors, std::string* error) {
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (!Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking),
             "stream", error) ||
      !m.gemm.Init(m.stream, error))
    return nullptr;
  SafetensorsFile file;
  if (!file.Open(safetensors, m.stream, error))
    return nullptr;
  bool ok = LoadConv(file, "decoder.conv1", 1, true, m.conv1, error);
  for (int b = 0; ok && b < kBlocks; ++b) {
    const std::string p = "decoder.block." + std::to_string(b);
    Block& block = m.blocks[b];
    ok = LoadSnake(file, p + ".snake1", block.snake, error) &&
         LoadTransposed(file, p + ".conv_t1", kStrides[b], block.up, error);
    const int dilations[3] = {1, 3, 9};
    for (int u = 0; ok && u < 3; ++u) {
      const std::string q = p + ".res_unit" + std::to_string(u + 1);
      ResidualUnit& unit = block.units[u];
      ok =
          LoadSnake(file, q + ".snake1", unit.snake1, error) &&
          LoadConv(file, q + ".conv1", dilations[u], true, unit.conv1, error) &&
          LoadSnake(file, q + ".snake2", unit.snake2, error) &&
          LoadConv(file, q + ".conv2", 1, true, unit.conv2, error);
    }
  }
  ok = ok && LoadSnake(file, "decoder.snake1", m.snake, error) &&
       LoadConv(file, "decoder.conv2", 1, false, m.conv2, error);
  if (!ok)
    return nullptr;
  if (m.conv1.in != kLatentChannels || m.conv2.out != 2) {
    Fail(error, "unexpected ACE-Step VAE layout");
    return nullptr;
  }
  return std::unique_ptr<VaeDecoder>(new VaeDecoder(std::move(impl)));
}

bool VaeDecoder::Decode(std::span<const float> latents, int frames,
                        std::vector<float>* audio, std::string* error) {
  Impl& m = *impl_;
  if (frames <= 0 || latents.size() != std::size_t(frames) * kLatentChannels)
    return Fail(error, "ACE-Step VAE input has the wrong size");
  // Buffer sizes for the widest chunk.
  const int span = std::min(frames, kChunkFrames + 2 * kOverlapFrames);
  std::size_t most = std::size_t(span) * m.conv1.out, most_z = 0;
  {
    std::size_t length = span;
    for (const Block& block : m.blocks) {
      most_z = std::max(most_z, length * 2 * block.up.stride * block.up.out);
      length *= block.up.stride;
      most = std::max(most, length * block.up.out);
    }
  }
  if (!m.x.Reserve(most * 4, error) || !m.t.Reserve(most * 4, error) ||
      !m.act.Reserve(std::max<std::size_t>(most, std::size_t(span) * 64) * 2,
                     error) ||
      !m.cols.Reserve(kIm2colBytes + (1U << 20U), error) ||
      !m.z.Reserve(most_z * 2, error) || !m.gemm_out.Reserve(most * 2, error))
    return false;
  const std::size_t samples = std::size_t(frames) * kSamplesPerFrame;
  audio->assign(2 * samples, 0.0F);
  std::vector<float> chunk;
  for (int first = 0; first < frames; first += kChunkFrames) {
    const int last = std::min(frames, first + kChunkFrames);
    const int begin = std::max(0, first - kOverlapFrames);
    const int end = std::min(frames, last + kOverlapFrames);
    const int count = end - begin;
    // conv1 reads `act` as BF16 [count, 64].
    std::vector<std::uint16_t> bits(std::size_t(count) * kLatentChannels);
    for (std::size_t i = 0; i < bits.size(); ++i)
      bits[i] = HostBf16(latents[std::size_t(begin) * kLatentChannels + i]);
    if (!Check(hipMemcpyAsync(m.act.get(), bits.data(), bits.size() * 2,
                              hipMemcpyHostToDevice, m.stream),
               "VAE upload", error) ||
        !m.Run(count, error))
      return false;
    chunk.resize(std::size_t(count) * kSamplesPerFrame * 2);
    if (!Check(hipMemcpyAsync(chunk.data(), m.t.get(), chunk.size() * 4,
                              hipMemcpyDeviceToHost, m.stream),
               "VAE download", error) ||
        !Check(hipStreamSynchronize(m.stream), "VAE decode", error))
      return false;
    const std::size_t skip = std::size_t(first - begin) * kSamplesPerFrame;
    const std::size_t keep = std::size_t(last - first) * kSamplesPerFrame;
    for (std::size_t i = 0; i < keep; ++i)
      for (int ch = 0; ch < 2; ++ch)
        (*audio)[ch * samples + std::size_t(first) * kSamplesPerFrame + i] =
            chunk[(skip + i) * 2 + ch];
  }
  return true;
}

}  // namespace gufo::models::ace_step
