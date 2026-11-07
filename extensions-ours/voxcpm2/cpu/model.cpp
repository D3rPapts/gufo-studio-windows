#include "src/models/voxcpm2/cpu/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace gufo::models::voxcpm2::cpu {
namespace {

using ming_tts::DType;
using ming_tts::SafetensorsFile;
using ming_tts::TensorView;
namespace ops = ming_tts::cpu;

Linear LoadLinear(const SafetensorsFile& file, const std::string& prefix, bool bias,
                  std::int64_t out_features, std::int64_t in_features) {
  const TensorView& weight = file.Require(prefix + ".weight");
  if (weight.dtype != DType::kBF16 || weight.shape.size() != 2) {
    throw std::runtime_error(prefix + ".weight must be a 2-D BF16 tensor");
  }
  Linear layer;
  layer.weight = reinterpret_cast<const std::uint16_t*>(weight.data);
  layer.out_features = weight.shape[0];
  layer.in_features = weight.shape[1];
  if (layer.out_features != out_features || layer.in_features != in_features) {
    throw std::runtime_error(prefix + ".weight has an unexpected shape");
  }
  if (bias) {
    layer.bias = file.ToF32(prefix + ".bias");
    if (static_cast<std::int64_t>(layer.bias.size()) != out_features) {
      throw std::runtime_error(prefix + ".bias has the wrong length");
    }
  } else if (file.Find(prefix + ".bias") != nullptr) {
    throw std::runtime_error(prefix + ".bias is not expected");
  }
  return layer;
}

std::vector<float> LoadVector(const SafetensorsFile& file, const std::string& name,
                              std::int64_t length) {
  std::vector<float> values = file.ToF32(name);
  if (static_cast<std::int64_t>(values.size()) != length) {
    throw std::runtime_error(name + " has the wrong length");
  }
  return values;
}

Stack LoadStack(const SafetensorsFile& file, const std::string& prefix,
                const StackConfig& config) {
  Stack stack;
  stack.config = config;
  const std::int64_t h = config.hidden_size;
  const std::int64_t q = static_cast<std::int64_t>(config.num_heads) * config.head_dim;
  const std::int64_t kv = static_cast<std::int64_t>(config.num_kv_heads) * config.head_dim;
  const std::int64_t f = config.intermediate_size;
  for (std::int32_t index = 0; index < config.num_layers; ++index) {
    const std::string p = prefix + "layers." + std::to_string(index) + ".";
    Layer layer;
    layer.q = LoadLinear(file, p + "self_attn.q_proj", false, q, h);
    layer.k = LoadLinear(file, p + "self_attn.k_proj", false, kv, h);
    layer.v = LoadLinear(file, p + "self_attn.v_proj", false, kv, h);
    layer.o = LoadLinear(file, p + "self_attn.o_proj", false, h, q);
    layer.gate = LoadLinear(file, p + "mlp.gate_proj", false, f, h);
    layer.up = LoadLinear(file, p + "mlp.up_proj", false, f, h);
    layer.down = LoadLinear(file, p + "mlp.down_proj", false, h, f);
    layer.input_norm = LoadVector(file, p + "input_layernorm.weight", h);
    layer.post_norm = LoadVector(file, p + "post_attention_layernorm.weight", h);
    stack.layers.push_back(std::move(layer));
  }
  stack.final_norm = LoadVector(file, prefix + "norm.weight", h);
  return stack;
}

inline float SiLU(float x) { return x / (1.0F + std::exp(-x)); }

}  // namespace

std::unique_ptr<Model> Model::Load(const std::string& model_dir, std::string* error) {
  auto set_error = [error](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
  };
  auto config = LoadModelConfigFromPath(model_dir, error);
  if (!config.has_value()) {
    return nullptr;
  }
  auto model = std::unique_ptr<Model>(new Model());
  model->config_ = *config;
  model->weights_ = SafetensorsFile::Open(
      std::filesystem::path(model_dir) / "model.safetensors", error);
  if (model->weights_ == nullptr) {
    return nullptr;
  }
  const SafetensorsFile& file = *model->weights_;
  const ModelConfig& c = model->config_;
  try {
    const TensorView& embed = file.Require("base_lm.embed_tokens.weight");
    if (embed.dtype != DType::kBF16 || embed.shape.size() != 2 ||
        embed.shape[0] != c.vocab_size || embed.shape[1] != c.base_lm.hidden_size) {
      throw std::runtime_error("base_lm.embed_tokens.weight has an unexpected shape");
    }
    model->embed_ = reinterpret_cast<const std::uint16_t*>(embed.data);
    model->base_ = LoadStack(file, "base_lm.", c.base_lm);
    model->residual_ = LoadStack(file, "residual_lm.", c.residual_lm);
    const std::int64_t h = c.base_lm.hidden_size;
    const std::int64_t eh = c.encoder.hidden_size;
    const std::int64_t dh = c.dit.hidden_size;
    const std::int64_t d = c.feat_dim;
    model->locenc_.in_proj = LoadLinear(file, "feat_encoder.in_proj", true, eh, d);
    model->locenc_.special_token = LoadVector(file, "feat_encoder.special_token", eh);
    model->locenc_.stack = LoadStack(file, "feat_encoder.encoder.", c.encoder);
    const std::string e = "feat_decoder.estimator.";
    model->dit_.in_proj = LoadLinear(file, e + "in_proj", true, dh, d);
    model->dit_.cond_proj = LoadLinear(file, e + "cond_proj", true, dh, d);
    model->dit_.out_proj = LoadLinear(file, e + "out_proj", true, d, dh);
    model->dit_.time_mlp1 = LoadLinear(file, e + "time_mlp.linear_1", true, dh, dh);
    model->dit_.time_mlp2 = LoadLinear(file, e + "time_mlp.linear_2", true, dh, dh);
    model->dit_.delta_mlp1 = LoadLinear(file, e + "delta_time_mlp.linear_1", true, dh, dh);
    model->dit_.delta_mlp2 = LoadLinear(file, e + "delta_time_mlp.linear_2", true, dh, dh);
    model->dit_.stack = LoadStack(file, e + "decoder.", c.dit);
    model->fsq_.in_proj = LoadLinear(file, "fsq_layer.in_proj", true, c.fsq_latent_dim, h);
    model->fsq_.out_proj = LoadLinear(file, "fsq_layer.out_proj", true, h, c.fsq_latent_dim);
    model->enc_to_lm_ = LoadLinear(file, "enc_to_lm_proj", true, h, eh);
    model->lm_to_dit_ = LoadLinear(file, "lm_to_dit_proj", true, dh, h);
    model->res_to_dit_ = LoadLinear(file, "res_to_dit_proj", true, dh, h);
    model->fusion_ = LoadLinear(file, "fusion_concat_proj", true, h, 2 * h);
    model->stop_proj_ = LoadLinear(file, "stop_proj", true, h, h);
    model->stop_head_ = LoadLinear(file, "stop_head", false, 2, h);
  } catch (const std::exception& exception) {
    set_error(std::string("VoxCPM2 weights: ") + exception.what());
    return nullptr;
  }
  if (2 * c.dit.hidden_size != c.base_lm.hidden_size) {
    set_error("VoxCPM2: the DiT condition must split into two hidden-size tokens");
    return nullptr;
  }
  // delta_time_mlp(SinusoidalPosEmb(0)): constant while mean_mode is off.
  {
    const std::int64_t dh = c.dit.hidden_size;
    std::vector<float> emb(static_cast<std::size_t>(dh));
    std::vector<float> hidden(static_cast<std::size_t>(dh));
    model->TimeEmbedding(0.0F, emb.data());
    ops::LinearForward(model->dit_.delta_mlp1, emb.data(), 1, hidden.data());
    for (float& value : hidden) {
      value = SiLU(value);
    }
    model->delta_time_.resize(static_cast<std::size_t>(dh));
    ops::LinearForward(model->dit_.delta_mlp2, hidden.data(), 1, model->delta_time_.data());
  }
  model->EnsureRope(1024);
  return model;
}

void Model::EnsureRope(std::int64_t positions) const {
  if (positions <= rope_positions_) {
    return;
  }
  positions = std::max<std::int64_t>(positions, 2 * rope_positions_);
  const std::int64_t half = config_.base_lm.head_dim / 2;
  const std::vector<float>& factors = config_.rope_factors();
  const float scaling = config_.rope_scaling();
  std::vector<float> inv_freq(static_cast<std::size_t>(half));
  for (std::int64_t i = 0; i < half; ++i) {
    // 1.0 / (base ** (arange(0, dim, 2).float() / dim)), fp32
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(2 * half);
    inv_freq[static_cast<std::size_t>(i)] = 1.0F / std::pow(config_.rope_theta, exponent);
  }
  rope_cos_.resize(static_cast<std::size_t>(positions * half));
  rope_sin_.resize(static_cast<std::size_t>(positions * half));
  for (std::int64_t t = 0; t < positions; ++t) {
    for (std::int64_t i = 0; i < half; ++i) {
      // torch.outer(t, 1.0 / ext_factors) * inv_freq, all fp32
      const float reciprocal = 1.0F / factors[static_cast<std::size_t>(i)];
      const float freq = (static_cast<float>(t) * reciprocal) * inv_freq[static_cast<std::size_t>(i)];
      rope_cos_[static_cast<std::size_t>(t * half + i)] = std::cos(freq) * scaling;
      rope_sin_[static_cast<std::size_t>(t * half + i)] = std::sin(freq) * scaling;
    }
  }
  rope_positions_ = positions;
}

void Model::ApplyRope(float* x, std::int64_t seq, std::int64_t heads, std::int64_t head_dim,
                      std::int64_t first_position) const {
  EnsureRope(first_position + seq);
  const std::int64_t half = head_dim / 2;
  for (std::int64_t t = 0; t < seq; ++t) {
    const float* c = rope_cos_.data() + (first_position + t) * half;
    const float* s = rope_sin_.data() + (first_position + t) * half;
    for (std::int64_t h = 0; h < heads; ++h) {
      float* row = x + (t * heads + h) * head_dim;
      for (std::int64_t i = 0; i < half; ++i) {
        const float a = row[i];
        const float b = row[i + half];
        row[i] = a * c[i] - b * s[i];
        row[i + half] = b * c[i] + a * s[i];
      }
    }
  }
}

void Model::StackForward(const Stack& stack, float* x, std::int64_t batch, std::int64_t seq,
                         bool causal, KvCache* cache, std::vector<float>* layer0) const {
  const StackConfig& c = stack.config;
  const std::int64_t rows = batch * seq;
  const std::int64_t h = c.hidden_size;
  const std::int64_t hd = c.head_dim;
  const std::int64_t qd = c.num_heads * hd;
  const std::int64_t kvd = c.num_kv_heads * hd;
  const float eps = config_.rms_norm_eps;
  const float residual_scale =
      config_.use_mup ? config_.scale_depth / std::sqrt(static_cast<float>(c.num_layers)) : 1.0F;
  std::vector<float> normed(static_cast<std::size_t>(rows * h));
  std::vector<float> q(static_cast<std::size_t>(rows * qd));
  std::vector<float> k(static_cast<std::size_t>(rows * kvd));
  std::vector<float> v(static_cast<std::size_t>(rows * kvd));
  std::vector<float> attn(static_cast<std::size_t>(rows * qd));
  std::vector<float> proj(static_cast<std::size_t>(rows * h));
  std::vector<float> gate(static_cast<std::size_t>(rows * c.intermediate_size));
  std::vector<float> up(static_cast<std::size_t>(rows * c.intermediate_size));
  const std::int64_t first = cache != nullptr ? cache->length : 0;
  if (cache != nullptr) {
    if (batch != 1) {
      throw std::logic_error("KV-cached stacks run with batch 1");
    }
    if (cache->k.size() != stack.layers.size()) {
      cache->k.assign(stack.layers.size(), {});
      cache->v.assign(stack.layers.size(), {});
    }
  }
  for (std::size_t index = 0; index < stack.layers.size(); ++index) {
    const Layer& layer = stack.layers[index];
    ops::RmsNormTo(x, normed.data(), rows, h, layer.input_norm, eps);
    ops::LinearForward(layer.q, normed.data(), rows, q.data());
    ops::LinearForward(layer.k, normed.data(), rows, k.data());
    ops::LinearForward(layer.v, normed.data(), rows, v.data());
    if (c.rope) {
      for (std::int64_t b = 0; b < batch; ++b) {
        ApplyRope(q.data() + b * seq * qd, seq, c.num_heads, hd, first);
        ApplyRope(k.data() + b * seq * kvd, seq, c.num_kv_heads, hd, first);
      }
    }
    if (cache != nullptr) {
      auto& ck = cache->k[index];
      auto& cv = cache->v[index];
      ck.resize(static_cast<std::size_t>((first + seq) * kvd));
      cv.resize(static_cast<std::size_t>((first + seq) * kvd));
      std::copy(k.begin(), k.end(), ck.begin() + first * kvd);
      std::copy(v.begin(), v.end(), cv.begin() + first * kvd);
      ops::Attention(q.data(), ck.data(), cv.data(), attn.data(), seq, first + seq,
                     c.num_heads, c.num_kv_heads, hd, causal, first, 0);
    } else {
      for (std::int64_t b = 0; b < batch; ++b) {
        ops::Attention(q.data() + b * seq * qd, k.data() + b * seq * kvd,
                       v.data() + b * seq * kvd, attn.data() + b * seq * qd, seq, seq,
                       c.num_heads, c.num_kv_heads, hd, causal, 0, 0);
      }
    }
    ops::LinearForward(layer.o, attn.data(), rows, proj.data());
    for (std::int64_t i = 0; i < rows * h; ++i) {
      x[i] += residual_scale == 1.0F ? proj[static_cast<std::size_t>(i)]
                                     : proj[static_cast<std::size_t>(i)] * residual_scale;
    }
    ops::RmsNormTo(x, normed.data(), rows, h, layer.post_norm, eps);
    ops::LinearForward(layer.gate, normed.data(), rows, gate.data());
    ops::LinearForward(layer.up, normed.data(), rows, up.data());
    ops::SiluMulInPlace(gate.data(), up.data(), gate.size());
    ops::LinearForward(layer.down, gate.data(), rows, proj.data());
    for (std::int64_t i = 0; i < rows * h; ++i) {
      x[i] += residual_scale == 1.0F ? proj[static_cast<std::size_t>(i)]
                                     : proj[static_cast<std::size_t>(i)] * residual_scale;
    }
    if (index == 0 && layer0 != nullptr) {
      layer0->assign(x, x + rows * h);
    }
  }
  ops::RmsNorm(x, rows, h, stack.final_norm, eps);
  if (cache != nullptr) {
    cache->length = first + seq;
  }
}

void Model::Embed(std::span<const std::int64_t> ids, float* out) const {
  const std::int64_t h = config_.base_lm.hidden_size;
  const float scale = config_.use_mup ? config_.scale_emb : 1.0F;
  for (std::size_t r = 0; r < ids.size(); ++r) {
    const std::int64_t id = ids[r];
    if (id < 0 || id >= config_.vocab_size) {
      throw std::out_of_range("VoxCPM2 token id out of range");
    }
    const std::uint16_t* row = embed_ + id * h;
    for (std::int64_t i = 0; i < h; ++i) {
      out[static_cast<std::int64_t>(r) * h + i] = ming_tts::Bf16ToF32(row[i]) * scale;
    }
  }
}

void Model::LocalEncode(const float* patches, std::int64_t count, float* out,
                        std::vector<float>* layer0_row0) const {
  const std::int64_t p = config_.patch_size;
  const std::int64_t d = config_.feat_dim;
  const std::int64_t eh = config_.encoder.hidden_size;
  const std::int64_t seq = p + 1;
  std::vector<float> x(static_cast<std::size_t>(count * seq * eh));
  std::vector<float> projected(static_cast<std::size_t>(count * p * eh));
  ops::LinearForward(locenc_.in_proj, patches, count * p, projected.data());
  for (std::int64_t n = 0; n < count; ++n) {
    float* dst = x.data() + n * seq * eh;
    std::copy(locenc_.special_token.begin(), locenc_.special_token.end(), dst);
    std::copy(projected.begin() + n * p * eh, projected.begin() + (n + 1) * p * eh, dst + eh);
  }
  (void)d;
  std::vector<float> l0;
  StackForward(locenc_.stack, x.data(), count, seq, false, nullptr,
               layer0_row0 != nullptr ? &l0 : nullptr);
  if (layer0_row0 != nullptr) {
    layer0_row0->assign(l0.begin(), l0.begin() + seq * eh);
  }
  for (std::int64_t n = 0; n < count; ++n) {
    std::copy(x.begin() + n * seq * eh, x.begin() + n * seq * eh + eh, out + n * eh);
  }
}

void Model::FsqForward(const float* x, std::int64_t rows, float* out) const {
  const auto latent = static_cast<std::size_t>(config_.fsq_latent_dim);
  std::vector<float> hidden(static_cast<std::size_t>(rows) * latent);
  ops::LinearForward(fsq_.in_proj, x, rows, hidden.data());
  const auto scale = static_cast<float>(config_.fsq_scale);
  for (float& value : hidden) {
    value = std::nearbyint(std::tanh(value) * scale) / scale;  // round half to even
  }
  ops::LinearForward(fsq_.out_proj, hidden.data(), rows, out);
}

void Model::PrefillTraced(const std::int64_t* ids, const float* feats,
                          const std::uint8_t* audio_mask, std::int64_t rows, float* lm_hidden,
                          float* residual_hidden, PrefillTrace* trace) {
  const std::int64_t h = config_.base_lm.hidden_size;
  const std::int64_t eh = config_.encoder.hidden_size;
  const auto hs = static_cast<std::size_t>(h);
  std::vector<float> cls(static_cast<std::size_t>(rows * eh));
  LocalEncode(feats, rows, cls.data(), trace != nullptr ? &trace->locenc_l0_row0 : nullptr);
  std::vector<float> feat_embed(static_cast<std::size_t>(rows) * hs);
  ops::LinearForward(enc_to_lm_, cls.data(), rows, feat_embed.data());
  std::vector<float> combined(static_cast<std::size_t>(rows) * hs);
  Embed(std::span<const std::int64_t>(ids, static_cast<std::size_t>(rows)), combined.data());
  for (std::int64_t r = 0; r < rows; ++r) {
    if (audio_mask[r] != 0) {
      std::copy(feat_embed.begin() + r * h, feat_embed.begin() + (r + 1) * h,
                combined.begin() + r * h);
    }
  }
  if (trace != nullptr) {
    trace->locenc_out = cls;
    trace->feat_embed = feat_embed;
    trace->combined = combined;
  }
  base_cache_ = {};
  residual_cache_ = {};
  std::vector<float> enc = combined;
  StackForward(base_, enc.data(), 1, rows, true, &base_cache_,
               trace != nullptr ? &trace->base_l0 : nullptr);
  if (trace != nullptr) {
    trace->base_out = enc;
  }
  // FSQ on audio rows, raw LM output on text rows.
  std::vector<std::int64_t> audio_rows;
  for (std::int64_t r = 0; r < rows; ++r) {
    if (audio_mask[r] != 0) {
      audio_rows.push_back(r);
    }
  }
  if (!audio_rows.empty()) {
    const auto n = static_cast<std::int64_t>(audio_rows.size());
    std::vector<float> gathered(static_cast<std::size_t>(n) * hs);
    std::vector<float> quantized(static_cast<std::size_t>(n) * hs);
    for (std::int64_t i = 0; i < n; ++i) {
      std::copy(enc.begin() + audio_rows[static_cast<std::size_t>(i)] * h,
                enc.begin() + (audio_rows[static_cast<std::size_t>(i)] + 1) * h,
                gathered.begin() + i * h);
    }
    FsqForward(gathered.data(), n, quantized.data());
    for (std::int64_t i = 0; i < n; ++i) {
      std::copy(quantized.begin() + i * h, quantized.begin() + (i + 1) * h,
                enc.begin() + audio_rows[static_cast<std::size_t>(i)] * h);
    }
  }
  std::copy(enc.end() - h, enc.end(), lm_hidden);
  std::vector<float> fusion_in(static_cast<std::size_t>(rows) * 2 * hs, 0.0F);
  for (std::int64_t r = 0; r < rows; ++r) {
    std::copy(enc.begin() + r * h, enc.begin() + (r + 1) * h, fusion_in.begin() + r * 2 * h);
    if (audio_mask[r] != 0) {
      std::copy(feat_embed.begin() + r * h, feat_embed.begin() + (r + 1) * h,
                fusion_in.begin() + r * 2 * h + h);
    }
  }
  std::vector<float> residual(static_cast<std::size_t>(rows) * hs);
  ops::LinearForward(fusion_, fusion_in.data(), rows, residual.data());
  if (trace != nullptr) {
    trace->enc_outputs = enc;
    trace->residual_inputs = residual;
  }
  StackForward(residual_, residual.data(), 1, rows, true, &residual_cache_, nullptr);
  if (trace != nullptr) {
    trace->residual_out = residual;
  }
  std::copy(residual.end() - h, residual.end(), residual_hidden);
}

void Model::DitCondition(const float* lm_hidden, const float* residual_hidden,
                         float* mu) const {
  const std::int64_t dh = config_.dit.hidden_size;
  ops::LinearForward(lm_to_dit_, lm_hidden, 1, mu);
  ops::LinearForward(res_to_dit_, residual_hidden, 1, mu + dh);
}

void Model::TimeEmbedding(float t, float* out) const {
  const std::int64_t dim = config_.dit.hidden_size;
  const std::int64_t half = dim / 2;
  // emb = log(10000) / (half - 1); exp(arange(half) * -emb); 1000 * t * emb
  const auto step = static_cast<float>(-(std::log(10000.0) / static_cast<double>(half - 1)));
  const float scaled_t = 1000.0F * t;
  for (std::int64_t i = 0; i < half; ++i) {
    const float freq = std::exp(static_cast<float>(i) * step);
    const float angle = scaled_t * freq;
    out[i] = std::sin(angle);
    out[i + half] = std::cos(angle);
  }
}

void Model::Velocity(const float* x, const float* mu, const float* cond, float t,
                     float* out) const {
  const std::int64_t p = config_.patch_size;
  const std::int64_t d = config_.feat_dim;
  const std::int64_t dh = config_.dit.hidden_size;
  const std::int64_t mu_tokens = config_.base_lm.hidden_size / dh;  // 2
  const std::int64_t seq = mu_tokens + 1 + 2 * p;
  std::vector<float> temb(static_cast<std::size_t>(dh));
  std::vector<float> hidden(static_cast<std::size_t>(dh));
  std::vector<float> time(static_cast<std::size_t>(dh));
  TimeEmbedding(t, temb.data());
  ops::LinearForward(dit_.time_mlp1, temb.data(), 1, hidden.data());
  for (float& value : hidden) {
    value = SiLU(value);
  }
  ops::LinearForward(dit_.time_mlp2, hidden.data(), 1, time.data());
  for (std::int64_t i = 0; i < dh; ++i) {
    time[static_cast<std::size_t>(i)] += delta_time_[static_cast<std::size_t>(i)];
  }
  std::vector<float> cond_rows(static_cast<std::size_t>(p * dh));
  std::vector<float> x_rows(static_cast<std::size_t>(p * dh));
  ops::LinearForward(dit_.cond_proj, cond, p, cond_rows.data());
  ops::LinearForward(dit_.in_proj, x, p, x_rows.data());
  std::vector<float> buf(static_cast<std::size_t>(2 * seq * dh), 0.0F);
  for (std::int64_t b = 0; b < 2; ++b) {
    float* dst = buf.data() + b * seq * dh;
    if (b == 0) {
      std::copy(mu, mu + mu_tokens * dh, dst);
    }
    std::copy(time.begin(), time.end(), dst + mu_tokens * dh);
    std::copy(cond_rows.begin(), cond_rows.end(), dst + (mu_tokens + 1) * dh);
    std::copy(x_rows.begin(), x_rows.end(), dst + (mu_tokens + 1 + p) * dh);
  }
  StackForward(dit_.stack, buf.data(), 2, seq, false, nullptr, nullptr);
  std::vector<float> tail(static_cast<std::size_t>(2 * p * dh));
  for (std::int64_t b = 0; b < 2; ++b) {
    const float* src = buf.data() + (b * seq + mu_tokens + 1 + p) * dh;
    std::copy(src, src + p * dh, tail.begin() + b * p * dh);
  }
  ops::LinearForward(dit_.out_proj, tail.data(), 2 * p, out);
  (void)d;
}

std::vector<float> Model::TimeSpan(std::int32_t timesteps, float sway) {
  // torch.linspace(1, 0, n + 1) in fp32 (CPU kernel: halves from each end).
  const std::int32_t steps = timesteps + 1;
  std::vector<float> span(static_cast<std::size_t>(steps));
  const float start = 1.0F;
  const float end = 0.0F;
  const float step = (end - start) / static_cast<float>(steps - 1);
  const std::int32_t halfway = steps / 2;
  for (std::int32_t i = 0; i < steps; ++i) {
    span[static_cast<std::size_t>(i)] =
        i < halfway ? start + step * static_cast<float>(i)
                    : end - step * static_cast<float>(steps - i - 1);
  }
  const auto half_pi = static_cast<float>(3.14159265358979323846 / 2.0);
  for (float& t : span) {
    t = t + sway * (std::cos(half_pi * t) - 1.0F + t);
  }
  return span;
}

void Model::SampleFlow(const float* mu, const float* cond, const float* noise,
                       const FlowOptions& options, float* patch) const {
  const std::int64_t n = static_cast<std::int64_t>(config_.patch_size) * config_.feat_dim;
  const std::vector<float> span = TimeSpan(options.timesteps, options.sway_sampling_coef);
  std::vector<float> x(noise, noise + n);
  std::vector<float> velocity(static_cast<std::size_t>(2 * n));
  std::vector<float> dphi(static_cast<std::size_t>(n));
  float t = span[0];
  float dt = span[0] - span[1];
  const auto steps = static_cast<std::int32_t>(span.size());
  const std::int32_t zero_init_steps =
      std::max(1, static_cast<std::int32_t>(static_cast<double>(steps) * 0.04));
  for (std::int32_t step = 1; step < steps; ++step) {
    if (options.cfg_zero_star && step <= zero_init_steps) {
      std::fill(dphi.begin(), dphi.end(), 0.0F);
    } else {
      Velocity(x.data(), mu, cond, t, velocity.data());
      const float* pos = velocity.data();
      const float* neg = velocity.data() + n;
      float st = 1.0F;
      if (options.cfg_zero_star) {
        float dot = 0.0F;
        float norm = 0.0F;
        for (std::int64_t i = 0; i < n; ++i) {
          dot += pos[i] * neg[i];
          norm += neg[i] * neg[i];
        }
        st = dot / (norm + 1e-8F);
      }
      for (std::int64_t i = 0; i < n; ++i) {
        dphi[static_cast<std::size_t>(i)] = neg[i] * st + options.cfg * (pos[i] - neg[i] * st);
      }
    }
    for (std::int64_t i = 0; i < n; ++i) {
      x[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - dt * dphi[static_cast<std::size_t>(i)];
    }
    t = t - dt;
    if (step < steps - 1) {
      dt = t - span[static_cast<std::size_t>(step + 1)];
    }
  }
  std::copy(x.begin(), x.end(), patch);
}

void Model::SamplePatch(const float* lm_hidden, const float* residual_hidden,
                        const float* prev_patch, const float* noise,
                        const FlowOptions& options, float* patch) {
  std::vector<float> mu(static_cast<std::size_t>(config_.base_lm.hidden_size));
  DitCondition(lm_hidden, residual_hidden, mu.data());
  SampleFlow(mu.data(), prev_patch, noise, options, patch);
}

void Model::StopLogits(const float* lm_hidden, float* logits) {
  std::vector<float> hidden(static_cast<std::size_t>(config_.base_lm.hidden_size));
  ops::LinearForward(stop_proj_, lm_hidden, 1, hidden.data());
  for (float& value : hidden) {
    value = SiLU(value);
  }
  ops::LinearForward(stop_head_, hidden.data(), 1, logits);
}

void Model::Advance(const float* patch, float* lm_hidden, float* residual_hidden) {
  const std::int64_t h = config_.base_lm.hidden_size;
  std::vector<float> cls(static_cast<std::size_t>(config_.encoder.hidden_size));
  LocalEncode(patch, 1, cls.data());
  std::vector<float> curr(static_cast<std::size_t>(h));
  ops::LinearForward(enc_to_lm_, cls.data(), 1, curr.data());
  std::vector<float> hidden = curr;
  StackForward(base_, hidden.data(), 1, 1, true, &base_cache_, nullptr);
  FsqForward(hidden.data(), 1, lm_hidden);
  std::vector<float> fusion_in(static_cast<std::size_t>(2 * h));
  std::copy(lm_hidden, lm_hidden + h, fusion_in.begin());
  std::copy(curr.begin(), curr.end(), fusion_in.begin() + h);
  ops::LinearForward(fusion_, fusion_in.data(), 1, residual_hidden);
  StackForward(residual_, residual_hidden, 1, 1, true, &residual_cache_, nullptr);
}

}  // namespace gufo::models::voxcpm2::cpu
