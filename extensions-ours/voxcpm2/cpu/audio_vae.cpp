#include "src/models/voxcpm2/cpu/audio_vae.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "src/models/voxcpm2/vae_weights.hpp"

namespace gufo::models::voxcpm2::cpu {
namespace {

const VaeTensor& Get(const VaeWeights& weights, const std::string& name) {
  const auto found = weights.find(name);
  if (found == weights.end()) {
    throw std::runtime_error("missing AudioVAE tensor " + name);
  }
  return found->second;
}

Conv1d LoadConv(const VaeWeights& weights, const std::string& prefix, std::int32_t kernel,
                std::int32_t stride, std::int32_t dilation, std::int32_t groups,
                std::int32_t left_pad) {
  const VaeTensor& w = Get(weights, prefix + ".weight");
  if (w.shape.size() != 3 || w.shape[2] != kernel) {
    throw std::runtime_error(prefix + ".weight has an unexpected shape");
  }
  Conv1d conv;
  conv.out = static_cast<std::int32_t>(w.shape[0]);
  conv.in = static_cast<std::int32_t>(w.shape[1]) * groups;
  conv.kernel = kernel;
  conv.stride = stride;
  conv.dilation = dilation;
  conv.groups = groups;
  conv.left_pad = left_pad;
  conv.weight = w.data;
  if (weights.contains(prefix + ".bias")) {
    conv.bias = Get(weights, prefix + ".bias").data;
  } else {
    conv.bias.assign(static_cast<std::size_t>(conv.out), 0.0F);
  }
  if (groups != 1 && !(groups == conv.in && groups == conv.out)) {
    throw std::runtime_error(prefix + ": only depthwise or dense convolutions");
  }
  return conv;
}

ConvTranspose1d LoadConvTranspose(const VaeWeights& weights, const std::string& prefix,
                                  std::int32_t stride) {
  const VaeTensor& w = Get(weights, prefix + ".weight");  // [in, out, kernel]
  if (w.shape.size() != 3 || w.shape[2] != 2 * stride) {
    throw std::runtime_error(prefix + ".weight has an unexpected shape");
  }
  ConvTranspose1d conv;
  conv.in = static_cast<std::int32_t>(w.shape[0]);
  conv.out = static_cast<std::int32_t>(w.shape[1]);
  conv.stride = stride;
  const std::int64_t k = 2 * stride;
  conv.weight.resize(w.data.size());
  for (std::int64_t i = 0; i < conv.in; ++i) {
    for (std::int64_t o = 0; o < conv.out; ++o) {
      for (std::int64_t j = 0; j < k; ++j) {
        conv.weight[static_cast<std::size_t>((o * conv.in + i) * k + j)] =
            w.data[static_cast<std::size_t>((i * conv.out + o) * k + j)];
      }
    }
  }
  conv.bias = Get(weights, prefix + ".bias").data;
  return conv;
}

Snake LoadSnake(const VaeWeights& weights, const std::string& name) {
  Snake snake;
  snake.alpha = Get(weights, name).data;
  snake.inv.resize(snake.alpha.size());
  for (std::size_t i = 0; i < snake.alpha.size(); ++i) {
    snake.inv[i] = 1.0F / (snake.alpha[i] + 1e-9F);
  }
  return snake;
}

ResidualUnit LoadUnit(const VaeWeights& weights, const std::string& prefix, std::int32_t dim,
                      std::int32_t dilation) {
  ResidualUnit unit;
  unit.snake1 = LoadSnake(weights, prefix + ".block.0.alpha");
  unit.conv1 = LoadConv(weights, prefix + ".block.1", 7, 1, dilation, dim, 6 * dilation);
  unit.snake2 = LoadSnake(weights, prefix + ".block.2.alpha");
  unit.conv2 = LoadConv(weights, prefix + ".block.3", 1, 1, 1, 1, 0);
  return unit;
}

// Snake in place on [channels, length].
void ApplySnake(const Snake& snake, float* x, std::int64_t channels, std::int64_t length) {
#pragma omp parallel for schedule(static) if (channels * length >= 65536)
  for (std::int64_t c = 0; c < channels; ++c) {
    const float a = snake.alpha[static_cast<std::size_t>(c)];
    const float inv = snake.inv[static_cast<std::size_t>(c)];
    float* row = x + c * length;
    for (std::int64_t t = 0; t < length; ++t) {
      const float s = std::sin(a * row[t]);
      row[t] = row[t] + inv * (s * s);
    }
  }
}

// Valid convolution over x [in, len_in] (left context already in front).
std::vector<float> ConvValid(const Conv1d& c, const float* x, std::int64_t len_in,
                             std::int64_t* len_out) {
  const std::int64_t span = static_cast<std::int64_t>(c.dilation) * (c.kernel - 1) + 1;
  const std::int64_t n = len_in < span ? 0 : (len_in - span) / c.stride + 1;
  *len_out = n;
  std::vector<float> y(static_cast<std::size_t>(c.out * n));
  if (n == 0) {
    return y;
  }
  const std::int64_t k = c.kernel;
  if (c.groups != 1) {  // depthwise
#pragma omp parallel for schedule(static) if (c.out * n >= 32768)
    for (std::int64_t ch = 0; ch < c.out; ++ch) {
      const float* w = c.weight.data() + ch * k;
      const float* xr = x + ch * len_in;
      float* yr = y.data() + ch * n;
      const float b = c.bias[static_cast<std::size_t>(ch)];
      for (std::int64_t t = 0; t < n; ++t) {
        float sum = 0.0F;
        const float* base = xr + t * c.stride;
        for (std::int64_t j = 0; j < k; ++j) {
          sum += w[j] * base[j * c.dilation];
        }
        yr[t] = b + sum;
      }
    }
    return y;
  }
  constexpr std::int64_t kTimeBlock = 256;
  constexpr std::int64_t kOutBlock = 16;
  const std::int64_t time_blocks = (n + kTimeBlock - 1) / kTimeBlock;
  const std::int64_t out_blocks = (c.out + kOutBlock - 1) / kOutBlock;
  const std::int64_t in = c.in;
#pragma omp parallel for schedule(dynamic) collapse(2) if (c.out * n * in >= 262144)
  for (std::int64_t ob = 0; ob < out_blocks; ++ob) {
    for (std::int64_t tb = 0; tb < time_blocks; ++tb) {
      const std::int64_t t0 = tb * kTimeBlock;
      const std::int64_t t1 = std::min(n, t0 + kTimeBlock);
      const std::int64_t o1 = std::min<std::int64_t>(c.out, (ob + 1) * kOutBlock);
      for (std::int64_t o = ob * kOutBlock; o < o1; ++o) {
        float* yr = y.data() + o * n;
        const float b = c.bias[static_cast<std::size_t>(o)];
        for (std::int64_t t = t0; t < t1; ++t) {
          yr[t] = 0.0F;
        }
        for (std::int64_t i = 0; i < in; ++i) {
          const float* xr = x + i * len_in;
          const float* w = c.weight.data() + (o * in + i) * k;
          for (std::int64_t j = 0; j < k; ++j) {
            const float wj = w[j];
            const float* xs = xr + j * c.dilation;
            if (c.stride == 1) {
              for (std::int64_t t = t0; t < t1; ++t) {
                yr[t] += wj * xs[t];
              }
            } else {
              for (std::int64_t t = t0; t < t1; ++t) {
                yr[t] += wj * xs[t * c.stride];
              }
            }
          }
        }
        for (std::int64_t t = t0; t < t1; ++t) {
          yr[t] += b;
        }
      }
    }
  }
  return y;
}

// Causal conv on x [in, length] with zero (or carried) left context.
std::vector<float> CausalConv(const Conv1d& c, const float* x, std::int64_t length,
                              std::vector<float>* history, std::int64_t* len_out) {
  const std::int64_t pad = c.left_pad;
  if (pad == 0) {
    return ConvValid(c, x, length, len_out);
  }
  const std::int64_t total = pad + length;
  std::vector<float> padded(static_cast<std::size_t>(c.in * total), 0.0F);
  for (std::int64_t ch = 0; ch < c.in; ++ch) {
    float* dst = padded.data() + ch * total;
    if (history != nullptr && !history->empty()) {
      std::copy(history->begin() + ch * pad, history->begin() + (ch + 1) * pad, dst);
    }
    std::copy(x + ch * length, x + (ch + 1) * length, dst + pad);
  }
  if (history != nullptr) {
    history->resize(static_cast<std::size_t>(c.in * pad));
    for (std::int64_t ch = 0; ch < c.in; ++ch) {
      const float* src = padded.data() + ch * total + total - pad;
      std::copy(src, src + pad, history->begin() + ch * pad);
    }
  }
  return ConvValid(c, padded.data(), total, len_out);
}

// Causal transposed conv (kernel 2s, stride s, right trim s): output
// [out, length * s]; `history` carries the previous input frame [in].
std::vector<float> CausalConvTranspose(const ConvTranspose1d& c, const float* x,
                                       std::int64_t length, std::vector<float>* history) {
  const std::int64_t s = c.stride;
  const std::int64_t k = 2 * s;
  const std::int64_t in = c.in;
  const std::int64_t n = length * s;
  // Extended input [in, length + 1]: previous frame (or zeros) then x.
  const std::int64_t ext = length + 1;
  std::vector<float> xe(static_cast<std::size_t>(in * ext), 0.0F);
  for (std::int64_t i = 0; i < in; ++i) {
    if (history != nullptr && !history->empty()) {
      xe[static_cast<std::size_t>(i * ext)] = (*history)[static_cast<std::size_t>(i)];
    }
    std::copy(x + i * length, x + (i + 1) * length, xe.begin() + i * ext + 1);
  }
  if (history != nullptr) {
    history->resize(static_cast<std::size_t>(in));
    for (std::int64_t i = 0; i < in; ++i) {
      (*history)[static_cast<std::size_t>(i)] = x[i * length + length - 1];
    }
  }
  std::vector<float> y(static_cast<std::size_t>(c.out * n));
#pragma omp parallel for schedule(static) if (c.out * n * in >= 262144)
  for (std::int64_t o = 0; o < c.out; ++o) {
    float* yr = y.data() + o * n;
    const float b = c.bias[static_cast<std::size_t>(o)];
    std::fill(yr, yr + n, 0.0F);
    // output m = e * s + j over the extended input e; keep m in [s, s + n)
    for (std::int64_t i = 0; i < in; ++i) {
      const float* w = c.weight.data() + (o * in + i) * k;
      const float* xr = xe.data() + i * ext;
      for (std::int64_t e = 0; e < ext; ++e) {
        const float xv = xr[e];
        const std::int64_t base = e * s - s;
        for (std::int64_t j = 0; j < k; ++j) {
          const std::int64_t m = base + j;
          if (m >= 0 && m < n) {
            yr[m] += w[j] * xv;
          }
        }
      }
    }
    for (std::int64_t m = 0; m < n; ++m) {
      yr[m] += b;
    }
  }
  return y;
}

// x [dim, length] in place: x + conv2(snake2(conv1(snake1(x)))).
void RunUnit(const ResidualUnit& unit, std::vector<float>* x, std::int64_t dim,
             std::int64_t length, std::vector<float>* history) {
  std::vector<float> h = *x;
  ApplySnake(unit.snake1, h.data(), dim, length);
  std::int64_t n = 0;
  std::vector<float> a = CausalConv(unit.conv1, h.data(), length, history, &n);
  ApplySnake(unit.snake2, a.data(), dim, length);
  std::vector<float> b = ConvValid(unit.conv2, a.data(), length, &n);
  for (std::size_t i = 0; i < x->size(); ++i) {
    (*x)[i] += b[i];
  }
}

}  // namespace

std::unique_ptr<AudioVae> AudioVae::Load(const std::string& model_dir,
                                         const AudioVaeConfig& config, std::string* error) {
  VaeWeights weights;
  if (!LoadVaeWeights(model_dir, &weights, error)) {
    return nullptr;
  }
  auto vae = std::unique_ptr<AudioVae>(new AudioVae());
  vae->config_ = config;
  try {
    vae->enc_in_ = LoadConv(weights, "encoder.block.0", 7, 1, 1, 1, 6);
    std::int32_t dim = config.encoder_dim;
    for (std::size_t b = 0; b < config.encoder_rates.size(); ++b) {
      const std::int32_t stride = config.encoder_rates[b];
      const std::string p = "encoder.block." + std::to_string(b + 1) + ".block.";
      EncoderBlock block;
      for (int u = 0; u < 3; ++u) {
        block.units[u] = LoadUnit(weights, p + std::to_string(u), dim, u == 0 ? 1 : (u == 1 ? 3 : 9));
      }
      block.snake = LoadSnake(weights, p + "3.alpha");
      // padding ceil(s/2), output_padding s % 2 -> left pad 2 * ceil(s/2) - s % 2
      const std::int32_t pad = 2 * ((stride + 1) / 2) - stride % 2;
      block.down = LoadConv(weights, p + "4", 2 * stride, stride, 1, 1, pad);
      vae->enc_blocks_.push_back(std::move(block));
      dim *= 2;
    }
    vae->enc_mu_ = LoadConv(weights, "encoder.fc_mu", 3, 1, 1, 1, 2);
    vae->dec_depthwise_ =
        LoadConv(weights, "decoder.model.0", 7, 1, 1, config.latent_dim, 6);
    vae->dec_pointwise_ = LoadConv(weights, "decoder.model.1", 1, 1, 1, 1, 0);
    std::int64_t bucket = 0;
    for (const std::int32_t boundary : config.sr_bin_boundaries) {
      if (boundary < config.out_sample_rate) {
        ++bucket;
      }
    }
    std::int32_t channels = config.decoder_dim;
    for (std::size_t b = 0; b < config.decoder_rates.size(); ++b) {
      const std::int32_t stride = config.decoder_rates[b];
      const std::int32_t out = channels / 2;
      const std::string m = "decoder.model." + std::to_string(b + 2);
      DecoderBlock block;
      const VaeTensor& scale = Get(weights, "decoder.sr_cond_model." + std::to_string(b + 2) +
                                                ".scale_embed.weight");
      const VaeTensor& bias = Get(weights, "decoder.sr_cond_model." + std::to_string(b + 2) +
                                               ".bias_embed.weight");
      if (scale.shape.size() != 2 || scale.shape[1] != channels || bucket >= scale.shape[0]) {
        throw std::runtime_error(m + ": sample-rate conditioning has an unexpected shape");
      }
      block.sr_scale.assign(scale.data.begin() + bucket * channels,
                            scale.data.begin() + (bucket + 1) * channels);
      block.sr_bias.assign(bias.data.begin() + bucket * channels,
                           bias.data.begin() + (bucket + 1) * channels);
      block.snake = LoadSnake(weights, m + ".block.0.alpha");
      block.up = LoadConvTranspose(weights, m + ".block.1", stride);
      for (int u = 0; u < 3; ++u) {
        block.units[u] = LoadUnit(weights, m + ".block." + std::to_string(u + 2), out,
                                  u == 0 ? 1 : (u == 1 ? 3 : 9));
      }
      vae->dec_blocks_.push_back(std::move(block));
      channels = out;
    }
    const std::size_t last = config.decoder_rates.size() + 2;
    vae->dec_snake_ = LoadSnake(weights, "decoder.model." + std::to_string(last) + ".alpha");
    vae->dec_out_ = LoadConv(weights, "decoder.model." + std::to_string(last + 1), 7, 1, 1, 1, 6);
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("VoxCPM2 AudioVAE: ") + exception.what();
    }
    return nullptr;
  }
  return vae;
}

std::vector<float> AudioVae::Encode(const float* wave, std::int64_t samples,
                                    std::int64_t* frames, EncodeTrace* trace) const {
  const std::int64_t hop = config_.chunk_size();
  if (samples <= 0 || samples % hop != 0) {
    throw std::invalid_argument("AudioVAE encode: length must be a multiple of the hop");
  }
  std::int64_t length = samples;
  std::vector<float> x = CausalConv(enc_in_, wave, length, nullptr, &length);
  if (trace != nullptr) {
    trace->block0 = x;
  }
  std::int64_t dim = config_.encoder_dim;
  for (const EncoderBlock& block : enc_blocks_) {
    for (const ResidualUnit& unit : block.units) {
      RunUnit(unit, &x, dim, length, nullptr);
    }
    ApplySnake(block.snake, x.data(), dim, length);
    std::int64_t out_length = 0;
    x = CausalConv(block.down, x.data(), length, nullptr, &out_length);
    length = out_length;
    dim *= 2;
  }
  if (trace != nullptr) {
    trace->block4 = x;
  }
  std::vector<float> mu = CausalConv(enc_mu_, x.data(), length, nullptr, &length);
  *frames = length;
  return mu;
}

std::vector<float> AudioVae::DecodeChunk(DecoderState* state, const float* z,
                                         std::int64_t frames, DecodeTrace* trace) const {
  // History slots: depthwise input conv, per block (transpose + 3 units), output conv.
  const std::size_t slots = 2 + dec_blocks_.size() * 4;
  if (state->history.size() != slots) {
    state->history.assign(slots, {});
  }
  std::size_t slot = 0;
  std::int64_t length = frames;
  std::vector<float> x = CausalConv(dec_depthwise_, z, length, &state->history[slot++], &length);
  x = ConvValid(dec_pointwise_, x.data(), length, &length);
  if (trace != nullptr) {
    trace->model1 = x;
  }
  std::int64_t channels = config_.decoder_dim;
  for (std::size_t b = 0; b < dec_blocks_.size(); ++b) {
    const DecoderBlock& block = dec_blocks_[b];
    for (std::int64_t c = 0; c < channels; ++c) {
      const float scale = block.sr_scale[static_cast<std::size_t>(c)];
      const float bias = block.sr_bias[static_cast<std::size_t>(c)];
      float* row = x.data() + c * length;
      for (std::int64_t t = 0; t < length; ++t) {
        row[t] = row[t] * scale + bias;
      }
    }
    ApplySnake(block.snake, x.data(), channels, length);
    x = CausalConvTranspose(block.up, x.data(), length, &state->history[slot++]);
    length *= block.up.stride;
    channels = block.up.out;
    for (const ResidualUnit& unit : block.units) {
      RunUnit(unit, &x, channels, length, &state->history[slot++]);
    }
    if (trace != nullptr && b == 0) {
      trace->model2 = x;
    }
    if (trace != nullptr && b + 1 == dec_blocks_.size()) {
      trace->last_block = x;
    }
  }
  ApplySnake(dec_snake_, x.data(), channels, length);
  std::vector<float> wave = CausalConv(dec_out_, x.data(), length, &state->history[slot++], &length);
  for (float& value : wave) {
    value = std::tanh(value);
  }
  state->started = true;
  return wave;
}

}  // namespace gufo::models::voxcpm2::cpu
