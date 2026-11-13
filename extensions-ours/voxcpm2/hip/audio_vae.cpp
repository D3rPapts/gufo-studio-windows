// Written with AI assistance (Claude Code).
#include "voxcpm2/hip/audio_vae.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ming_tts/hip/ops.hpp"
#include "voxcpm2/hip/ops.hpp"

namespace gufo::models::voxcpm2::hip {
namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("HIP AudioVAE ") + what + ": " +
                             hipGetErrorString(status));
  }
}

struct Buffer {
  float* data{nullptr};
  std::size_t capacity{0};
  float* Ensure(std::size_t elements) {
    if (elements > capacity) {
      if (data != nullptr) {
        (void)hipFree(data);
        data = nullptr;
      }
      capacity = std::max<std::size_t>(elements, 256);
      Check(hipMalloc(reinterpret_cast<void**>(&data), capacity * sizeof(float)), "alloc");
    }
    return data;
  }
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
};

struct DConv {
  std::size_t in{0}, out{0}, kernel{1}, stride{1}, dilation{1}, left_pad{0};
  bool depthwise{false};
  const float* w{nullptr};  // [out, in / groups * kernel]
  const float* b{nullptr};
};

struct DConvT {
  std::size_t in{0}, out{0}, stride{1};
  const float* a{nullptr};  // [(2 * stride) * out, in]
  const float* b{nullptr};
};

struct DSnake {
  const float* alpha{nullptr};
  const float* inv{nullptr};
};

struct DUnit {
  DSnake snake1, snake2;
  DConv conv1, conv2;
};

struct DEncBlock {
  DUnit units[3];
  DSnake snake;
  DConv down;
};

struct DDecBlock {
  const float* sr_scale{nullptr};
  const float* sr_bias{nullptr};
  DSnake snake;
  DConvT up;
  DUnit units[3];
};

}  // namespace

struct AudioVae::Impl {
  hipStream_t stream{nullptr};
  std::vector<void*> allocations;
  AudioVaeConfig config;
  DConv enc_in, enc_mu, dec_depthwise, dec_pointwise, dec_out;
  std::vector<DEncBlock> enc_blocks;
  std::vector<DDecBlock> dec_blocks;
  DSnake dec_snake;
  std::size_t encoder_dim{0}, decoder_dim{0};
  // decoder state: one left-context buffer per causal conv
  std::vector<float*> history;
  std::vector<std::size_t> history_size;
  bool history_clean{false};
  Buffer main0, main1, h, a, b, padded, col, xe, zt, io;

  ~Impl() {
    for (void* p : allocations) {
      (void)hipFree(p);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  const float* Upload(const std::vector<float>& host) {
    void* device = nullptr;
    Check(hipMalloc(&device, std::max<std::size_t>(host.size(), 1) * sizeof(float)), "weight alloc");
    allocations.push_back(device);
    Check(hipMemcpy(device, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice),
          "weight upload");
    return static_cast<const float*>(device);
  }
  DConv UploadConv(const cpu::Conv1d& c) {
    DConv d;
    d.in = static_cast<std::size_t>(c.in);
    d.out = static_cast<std::size_t>(c.out);
    d.kernel = static_cast<std::size_t>(c.kernel);
    d.stride = static_cast<std::size_t>(c.stride);
    d.dilation = static_cast<std::size_t>(c.dilation);
    d.left_pad = static_cast<std::size_t>(c.left_pad);
    d.depthwise = c.groups != 1;
    d.w = Upload(c.weight);
    d.b = Upload(c.bias);
    return d;
  }
  DConvT UploadConvT(const cpu::ConvTranspose1d& c) {
    DConvT d;
    d.in = static_cast<std::size_t>(c.in);
    d.out = static_cast<std::size_t>(c.out);
    d.stride = static_cast<std::size_t>(c.stride);
    const std::size_t k = 2 * d.stride;
    // cpu layout [out, in, k] -> A[(j * out + co), ci]
    std::vector<float> a(d.out * d.in * k);
    for (std::size_t co = 0; co < d.out; ++co) {
      for (std::size_t ci = 0; ci < d.in; ++ci) {
        for (std::size_t j = 0; j < k; ++j) {
          a[(j * d.out + co) * d.in + ci] = c.weight[(co * d.in + ci) * k + j];
        }
      }
    }
    d.a = Upload(a);
    d.b = Upload(c.bias);
    return d;
  }
  DSnake UploadSnake(const cpu::Snake& s) { return {Upload(s.alpha), Upload(s.inv)}; }
  DUnit UploadUnit(const cpu::ResidualUnit& u) {
    return {UploadSnake(u.snake1), UploadSnake(u.snake2), UploadConv(u.conv1), UploadConv(u.conv2)};
  }

  // Causal conv of x [in, length]; `history` (in * left_pad) carries the left
  // context between chunks (nullptr: zeros, encoder). Writes into `out`.
  float* Conv(const DConv& c, const float* x, std::size_t length, float* history,
              std::size_t* len_out, Buffer& out) {
    const float* input = x;
    std::size_t len_in = length;
    if (c.left_pad > 0) {
      len_in = c.left_pad + length;
      float* p = padded.Ensure(c.in * len_in);
      LaunchPadLeft(history, x, p, c.in, c.left_pad, length, stream);
      if (history != nullptr) {
        LaunchKeepTail(p, history, c.in, c.left_pad, len_in, stream);
      }
      input = p;
    }
    const std::size_t span = c.dilation * (c.kernel - 1) + 1;
    const std::size_t n = len_in < span ? 0 : (len_in - span) / c.stride + 1;
    *len_out = n;
    float* y = out.Ensure(c.out * std::max<std::size_t>(n, 1));
    if (n == 0) {
      return y;
    }
    if (c.depthwise) {
      LaunchDepthwiseConv(input, len_in, c.w, c.b, y, c.out, c.kernel, c.dilation, n, stream);
    } else if (c.kernel == 1 && c.stride == 1) {
      LaunchGemmF32NN(c.w, input, c.b, y, c.out, n, c.in, stream);
    } else {
      float* cols = col.Ensure(c.in * c.kernel * n);
      LaunchIm2col(input, len_in, cols, c.in, c.kernel, c.stride, c.dilation, n, stream);
      LaunchGemmF32NN(c.w, cols, c.b, y, c.out, n, c.in * c.kernel, stream);
    }
    return y;
  }

  // x [dim, length] in place: x + conv2(snake2(conv1(snake1(x)))).
  void Unit(const DUnit& u, float* x, std::size_t dim, std::size_t length, float* history) {
    float* hx = h.Ensure(dim * length);
    Check(hipMemcpyAsync(hx, x, dim * length * sizeof(float), hipMemcpyDeviceToDevice, stream),
          "copy");
    LaunchSnake(hx, u.snake1.alpha, u.snake1.inv, dim, length, stream);
    std::size_t n = 0;
    float* ax = Conv(u.conv1, hx, length, history, &n, a);
    LaunchSnake(ax, u.snake2.alpha, u.snake2.inv, dim, length, stream);
    float* bx = Conv(u.conv2, ax, length, nullptr, &n, b);
    ming_tts::hip::LaunchAdd(x, bx, dim * length, stream);
  }

  void ToHost(float* host, const float* device, std::size_t count) {
    Check(hipMemcpyAsync(host, device, count * sizeof(float), hipMemcpyDeviceToHost, stream),
          "download");
    Check(hipStreamSynchronize(stream), "synchronize");
  }
};

AudioVae::AudioVae(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AudioVae::~AudioVae() = default;

std::unique_ptr<AudioVae> AudioVae::Create(const cpu::AudioVae& vae, std::string* error) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    if (error != nullptr) {
      *error = "no HIP device";
    }
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  try {
    Impl& m = *impl;
    m.config = vae.config();
    m.encoder_dim = static_cast<std::size_t>(m.config.encoder_dim);
    m.decoder_dim = static_cast<std::size_t>(m.config.decoder_dim);
    Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "stream");
    m.enc_in = m.UploadConv(vae.enc_in());
    for (const cpu::EncoderBlock& block : vae.enc_blocks()) {
      DEncBlock d;
      for (int u = 0; u < 3; ++u) {
        d.units[u] = m.UploadUnit(block.units[u]);
      }
      d.snake = m.UploadSnake(block.snake);
      d.down = m.UploadConv(block.down);
      m.enc_blocks.push_back(d);
    }
    m.enc_mu = m.UploadConv(vae.enc_mu());
    m.dec_depthwise = m.UploadConv(vae.dec_depthwise());
    m.dec_pointwise = m.UploadConv(vae.dec_pointwise());
    // history slots: depthwise input conv, per block (transpose + 3 units), output conv
    m.history_size.push_back(m.dec_depthwise.in * m.dec_depthwise.left_pad);
    for (const cpu::DecoderBlock& block : vae.dec_blocks()) {
      DDecBlock d;
      d.sr_scale = m.Upload(block.sr_scale);
      d.sr_bias = m.Upload(block.sr_bias);
      d.snake = m.UploadSnake(block.snake);
      d.up = m.UploadConvT(block.up);
      m.history_size.push_back(d.up.in);
      for (int u = 0; u < 3; ++u) {
        d.units[u] = m.UploadUnit(block.units[u]);
        m.history_size.push_back(d.units[u].conv1.in * d.units[u].conv1.left_pad);
      }
      m.dec_blocks.push_back(d);
    }
    m.dec_snake = m.UploadSnake(vae.dec_snake());
    m.dec_out = m.UploadConv(vae.dec_out());
    m.history_size.push_back(m.dec_out.in * m.dec_out.left_pad);
    for (const std::size_t size : m.history_size) {
      void* device = nullptr;
      Check(hipMalloc(&device, std::max<std::size_t>(size, 1) * sizeof(float)), "history alloc");
      m.allocations.push_back(device);
      m.history.push_back(static_cast<float*>(device));
    }
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
  auto out = std::unique_ptr<AudioVae>(new AudioVae(std::move(impl)));
  out->ResetDecoder();
  return out;
}

void AudioVae::ResetDecoder() {
  Impl& m = *impl_;
  for (std::size_t i = 0; i < m.history.size(); ++i) {
    Check(hipMemsetAsync(m.history[i], 0, std::max<std::size_t>(m.history_size[i], 1) * sizeof(float),
                         m.stream),
          "history reset");
  }
}

std::vector<float> AudioVae::Encode(const float* wave, std::int64_t samples,
                                    std::int64_t* frames) {
  Impl& m = *impl_;
  const auto hop = static_cast<std::int64_t>(m.config.chunk_size());
  if (samples <= 0 || samples % hop != 0) {
    throw std::invalid_argument("AudioVAE encode: length must be a multiple of the hop");
  }
  std::size_t length = static_cast<std::size_t>(samples);
  float* in = m.io.Ensure(length);
  Check(hipMemcpyAsync(in, wave, length * sizeof(float), hipMemcpyHostToDevice, m.stream),
        "upload");
  std::size_t n = 0;
  float* x = m.Conv(m.enc_in, in, length, nullptr, &n, m.main0);
  length = n;
  std::size_t dim = m.encoder_dim;
  bool in_main0 = true;
  for (const DEncBlock& block : m.enc_blocks) {
    for (const DUnit& unit : block.units) {
      m.Unit(unit, x, dim, length, nullptr);
    }
    LaunchSnake(x, block.snake.alpha, block.snake.inv, dim, length, m.stream);
    x = m.Conv(block.down, x, length, nullptr, &n, in_main0 ? m.main1 : m.main0);
    in_main0 = !in_main0;
    length = n;
    dim *= 2;
  }
  float* mu = m.Conv(m.enc_mu, x, length, nullptr, &n, m.io);
  std::vector<float> out(m.enc_mu.out * n);
  m.ToHost(out.data(), mu, out.size());
  *frames = static_cast<std::int64_t>(n);
  return out;
}

std::vector<float> AudioVae::DecodeChunk(const float* z, std::int64_t frames) {
  Impl& m = *impl_;
  std::size_t length = static_cast<std::size_t>(frames);
  const std::size_t latent = m.dec_depthwise.in;
  float* in = m.io.Ensure(latent * length);
  Check(hipMemcpyAsync(in, z, latent * length * sizeof(float), hipMemcpyHostToDevice, m.stream),
        "upload");
  std::size_t slot = 0;
  std::size_t n = 0;
  float* x = m.Conv(m.dec_depthwise, in, length, m.history[slot++], &n, m.main1);
  x = m.Conv(m.dec_pointwise, x, length, nullptr, &n, m.main0);
  bool in_main0 = true;
  std::size_t channels = m.decoder_dim;
  for (const DDecBlock& block : m.dec_blocks) {
    LaunchChannelScaleBias(x, block.sr_scale, block.sr_bias, channels, length, m.stream);
    LaunchSnake(x, block.snake.alpha, block.snake.inv, channels, length, m.stream);
    // x_ext = [previous frame, x]; the last frame becomes the new history
    const std::size_t ext = length + 1;
    float* xe = m.xe.Ensure(block.up.in * ext);
    float* hist = m.history[slot++];
    LaunchPadLeft(hist, x, xe, block.up.in, 1, length, m.stream);
    LaunchKeepTail(xe, hist, block.up.in, 1, ext, m.stream);
    float* zt = m.zt.Ensure(2 * block.up.stride * block.up.out * ext);
    LaunchGemmF32NN(block.up.a, xe, nullptr, zt, 2 * block.up.stride * block.up.out, ext,
                    block.up.in, m.stream);
    Buffer& target = in_main0 ? m.main1 : m.main0;
    in_main0 = !in_main0;
    float* y = target.Ensure(block.up.out * length * block.up.stride);
    LaunchConvTransposeCombine(zt, block.up.b, y, block.up.out, block.up.stride, length, m.stream);
    x = y;
    length *= block.up.stride;
    channels = block.up.out;
    for (const DUnit& unit : block.units) {
      m.Unit(unit, x, channels, length, m.history[slot++]);
    }
  }
  LaunchSnake(x, m.dec_snake.alpha, m.dec_snake.inv, channels, length, m.stream);
  float* wave = m.Conv(m.dec_out, x, length, m.history[slot++], &n, m.io);
  LaunchTanh(wave, n, m.stream);
  std::vector<float> out(n);
  m.ToHost(out.data(), wave, n);
  return out;
}

}  // namespace gufo::models::voxcpm2::hip
