#include "indextts/gpt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "indextts/cpu/ops.hpp"

namespace gufo::models::indextts {
namespace {

constexpr float kLnEps = 1e-5F;

// ------------------------------------------------- emotion conformer ------
constexpr std::int64_t kEmoDim = 512;
constexpr std::int64_t kEmoHeads = 4;
constexpr std::int64_t kEmoHeadDim = 128;
constexpr std::int64_t kEmoFfn = 1024;
constexpr std::int64_t kEmoBlocks = 4;
constexpr std::int64_t kEmoKernel = 15;
constexpr std::int64_t kIn = 1024;
constexpr std::int64_t kSubW = (kIn - 3) / 2 + 1;  // 511

}  // namespace

std::unique_ptr<UnifiedVoice> UnifiedVoice::Load(const std::filesystem::path& gpt_pth,
                                                 std::string* error) {
  auto sd = PthArchive::Open(gpt_pth, error);
  if (sd == nullptr) {
    return nullptr;
  }
  auto out = std::unique_ptr<UnifiedVoice>(new UnifiedVoice());
  out->sd_ = std::move(sd);
  try {
    const PthTensor& mel = out->sd_->Require("model.mel_embedding.weight");
    const PthTensor& text = out->sd_->Require("model.text_embedding.weight");
    out->config_.dim = mel.shape.at(1);
    out->config_.mel_codes = mel.shape.at(0);
    out->config_.max_mel_positions = out->sd_->Require("model.mel_pos_embedding.emb.weight").shape.at(0);
    out->config_.max_text_positions = out->sd_->Require("model.text_pos_embedding.emb.weight").shape.at(0);
    std::int64_t layers = 0;
    while (out->sd_->Has("model.gpt.h." + std::to_string(layers) + ".ln_1.weight")) {
      ++layers;
    }
    out->config_.layers = layers;
    (void)text;
    if (layers == 0 || !out->sd_->Has("model.spk_emb_proj.weight") ||
        !out->sd_->Has("model.emo_conditioning_encoder.embed.out.0.weight")) {
      throw std::runtime_error("not an IndexTTS-2.5 UnifiedVoice checkpoint");
    }
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = gpt_pth.string() + ": " + e.what();
    }
    return nullptr;
  }
  return out;
}

const float* UnifiedVoice::W(const std::string& name, std::int64_t numel) const {
  return sd_->F32("model." + name, numel);
}

std::vector<float> UnifiedVoice::EmoVec(const float* cond, std::int64_t rows, const Tap& tap) const {
  const std::string e = "emo_conditioning_encoder.";
  // Conv2dSubsampling2: conv2d(1, 512, 3, stride 2) + ReLU, then Linear(512*511 -> 512)
  const std::int64_t t = (rows - 3) / 2 + 1;
  if (t < 1) {
    throw std::invalid_argument("emotion reference too short");
  }
  std::vector<float> sub(static_cast<std::size_t>(t * kEmoDim * kSubW));
  {
    const float* w = W(e + "embed.conv.0.weight", kEmoDim * 9);
    const float* b = W(e + "embed.conv.0.bias", kEmoDim);
#pragma omp parallel for schedule(static) collapse(2)
    for (std::int64_t ti = 0; ti < t; ++ti) {
      for (std::int64_t c = 0; c < kEmoDim; ++c) {
        float* o = sub.data() + (ti * kEmoDim + c) * kSubW;
        const float* wc = w + c * 9;
        for (std::int64_t f = 0; f < kSubW; ++f) {
          float s = b[c];
          for (int ky = 0; ky < 3; ++ky) {
            const float* row = cond + (2 * ti + ky) * kIn + 2 * f;
            s += wc[ky * 3] * row[0] + wc[ky * 3 + 1] * row[1] + wc[ky * 3 + 2] * row[2];
          }
          o[f] = std::max(s, 0.0F);
        }
      }
    }
  }
  std::vector<float> x(static_cast<std::size_t>(t * kEmoDim));
  cpu::Linear(sub.data(), t, kEmoDim * kSubW, W(e + "embed.out.0.weight", kEmoDim * kEmoDim * kSubW),
              kEmoDim, W(e + "embed.out.0.bias", kEmoDim), x.data());
  sub.clear();
  sub.shrink_to_fit();
  // embed.out alone is 535 MB of F32, read once per reference (#829).
  sd_->ReleaseTensorPages("model." + e + "embed.");
  const float xscale = std::sqrt(static_cast<float>(kEmoDim));
  for (float& v : x) {
    v *= xscale;
  }
  if (tap) {
    tap("emo_embed", x);
  }
  const float* pe = W(e + "embed.pos_enc.pe");  // [1, 5000, 512]
  std::vector<float> y(x.size());
  std::vector<float> q(x.size()), k(x.size()), v(x.size()), p(x.size()), att(x.size());
  std::vector<float> wide(static_cast<std::size_t>(t * kEmoFfn));
  for (std::int64_t layer = 0; layer < kEmoBlocks; ++layer) {
    const std::string lp = e + "encoders." + std::to_string(layer) + ".";
    // multi-head attention with relative positions (no rel_shift)
    cpu::LayerNorm(x.data(), y.data(), t, kEmoDim, W(lp + "norm_mha.weight", kEmoDim),
                   W(lp + "norm_mha.bias", kEmoDim), kLnEps);
    cpu::Linear(y.data(), t, kEmoDim, W(lp + "self_attn.linear_q.weight"), kEmoDim, W(lp + "self_attn.linear_q.bias"), q.data());
    cpu::Linear(y.data(), t, kEmoDim, W(lp + "self_attn.linear_k.weight"), kEmoDim, W(lp + "self_attn.linear_k.bias"), k.data());
    cpu::Linear(y.data(), t, kEmoDim, W(lp + "self_attn.linear_v.weight"), kEmoDim, W(lp + "self_attn.linear_v.bias"), v.data());
    cpu::Linear(pe, t, kEmoDim, W(lp + "self_attn.linear_pos.weight"), kEmoDim, nullptr, p.data());
    const float* bu = W(lp + "self_attn.pos_bias_u", kEmoDim);
    const float* bv = W(lp + "self_attn.pos_bias_v", kEmoDim);
    const float inv = 1.0F / std::sqrt(static_cast<float>(kEmoHeadDim));
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (std::int64_t h = 0; h < kEmoHeads; ++h) {
      for (std::int64_t i = 0; i < t; ++i) {
        std::vector<float> sc(static_cast<std::size_t>(t));
        float qu[kEmoHeadDim];
        float qv[kEmoHeadDim];
        for (std::int64_t d = 0; d < kEmoHeadDim; ++d) {
          const float qd = q[static_cast<std::size_t>(i * kEmoDim + h * kEmoHeadDim + d)];
          qu[d] = qd + bu[h * kEmoHeadDim + d];
          qv[d] = qd + bv[h * kEmoHeadDim + d];
        }
        float mx = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = 0; j < t; ++j) {
          const float* kj = k.data() + j * kEmoDim + h * kEmoHeadDim;
          const float* pj = p.data() + j * kEmoDim + h * kEmoHeadDim;
          float ac = 0.0F;
          float bd = 0.0F;
          for (std::int64_t d = 0; d < kEmoHeadDim; ++d) {
            ac += qu[d] * kj[d];
            bd += qv[d] * pj[d];
          }
          const float s = (ac + bd) * inv;
          sc[static_cast<std::size_t>(j)] = s;
          mx = std::max(mx, s);
        }
        double sum = 0.0;
        for (float& s : sc) {
          s = std::exp(s - mx);
          sum += s;
        }
        float* o = att.data() + i * kEmoDim + h * kEmoHeadDim;
        std::fill(o, o + kEmoHeadDim, 0.0F);
        for (std::int64_t j = 0; j < t; ++j) {
          const float pr = static_cast<float>(sc[static_cast<std::size_t>(j)] / sum);
          const float* vj = v.data() + j * kEmoDim + h * kEmoHeadDim;
          for (std::int64_t d = 0; d < kEmoHeadDim; ++d) {
            o[d] += pr * vj[d];
          }
        }
      }
    }
    cpu::Linear(att.data(), t, kEmoDim, W(lp + "self_attn.linear_out.weight"), kEmoDim,
                W(lp + "self_attn.linear_out.bias"), y.data());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += y[i];
    }
    // convolution module
    cpu::LayerNorm(x.data(), y.data(), t, kEmoDim, W(lp + "norm_conv.weight"), W(lp + "norm_conv.bias"), kLnEps);
    std::vector<float> pw(static_cast<std::size_t>(t * 2 * kEmoDim));
    cpu::Linear(y.data(), t, kEmoDim, W(lp + "conv_module.pointwise_conv1.weight", 2 * kEmoDim * kEmoDim),
                2 * kEmoDim, W(lp + "conv_module.pointwise_conv1.bias"), pw.data());
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t c = 0; c < kEmoDim; ++c) {
        y[static_cast<std::size_t>(r * kEmoDim + c)] =
            pw[static_cast<std::size_t>(r * 2 * kEmoDim + c)] *
            cpu::Sigmoid(pw[static_cast<std::size_t>(r * 2 * kEmoDim + kEmoDim + c)]);
      }
    }
    const float* dw = W(lp + "conv_module.depthwise_conv.weight", kEmoDim * kEmoKernel);
    const float* db = W(lp + "conv_module.depthwise_conv.bias", kEmoDim);
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t c = 0; c < kEmoDim; ++c) {
        float s = db[c];
        for (std::int64_t j = 0; j < kEmoKernel; ++j) {
          const std::int64_t src = r - kEmoKernel / 2 + j;
          if (src >= 0 && src < t) {
            s += dw[c * kEmoKernel + j] * y[static_cast<std::size_t>(src * kEmoDim + c)];
          }
        }
        att[static_cast<std::size_t>(r * kEmoDim + c)] = s;
      }
    }
    cpu::LayerNorm(att.data(), y.data(), t, kEmoDim, W(lp + "conv_module.norm.weight"),
                   W(lp + "conv_module.norm.bias"), kLnEps);
    cpu::SiluInPlace(y.data(), y.size());
    cpu::Linear(y.data(), t, kEmoDim, W(lp + "conv_module.pointwise_conv2.weight", kEmoDim * kEmoDim), kEmoDim,
                W(lp + "conv_module.pointwise_conv2.bias"), att.data());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += att[i];
    }
    // feed forward
    cpu::LayerNorm(x.data(), y.data(), t, kEmoDim, W(lp + "norm_ff.weight"), W(lp + "norm_ff.bias"), kLnEps);
    cpu::Linear(y.data(), t, kEmoDim, W(lp + "feed_forward.w_1.weight"), kEmoFfn, W(lp + "feed_forward.w_1.bias"),
                wide.data());
    cpu::SiluInPlace(wide.data(), wide.size());
    cpu::Linear(wide.data(), t, kEmoFfn, W(lp + "feed_forward.w_2.weight"), kEmoDim,
                W(lp + "feed_forward.w_2.bias"), y.data());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += y[i];
    }
    cpu::LayerNorm(x.data(), x.data(), t, kEmoDim, W(lp + "norm_final.weight"), W(lp + "norm_final.bias"), kLnEps);
    if (tap && layer == 0) {
      tap("emo_layer0", x);
    }
  }
  cpu::LayerNorm(x.data(), x.data(), t, kEmoDim, W(e + "after_norm.weight"), W(e + "after_norm.bias"), kLnEps);
  if (tap) {
    tap("emo_encoded", x);
  }
  // PerceiverResampler(1024, context 512, 1 latent, depth 2, heads 4, ff_mult 2)
  constexpr std::int64_t kD = 1024;
  constexpr std::int64_t kInner = 256;
  constexpr std::int64_t kPh = 4;
  constexpr std::int64_t kPd = 64;
  constexpr std::int64_t kFf = 1365;
  const std::string pp = "emo_perceiver_encoder.";
  std::vector<float> ctx(static_cast<std::size_t>((t + 1) * kD));
  cpu::Linear(x.data(), t, kEmoDim, W(pp + "proj_context.weight", kD * kEmoDim), kD,
              W(pp + "proj_context.bias", kD), ctx.data() + kD);
  std::vector<float> lat(W(pp + "latents", kD), W(pp + "latents", kD) + kD);
  for (int layer = 0; layer < 2; ++layer) {
    const std::string lp = pp + "layers." + std::to_string(layer) + ".";
    std::copy(lat.begin(), lat.end(), ctx.begin());  // cross_attn_include_queries
    std::vector<float> qq(kInner);
    std::vector<float> kv(static_cast<std::size_t>((t + 1) * 2 * kInner));
    cpu::Linear(lat.data(), 1, kD, W(lp + "0.to_q.weight", kInner * kD), kInner, nullptr, qq.data());
    cpu::Linear(ctx.data(), t + 1, kD, W(lp + "0.to_kv.weight", 2 * kInner * kD), 2 * kInner, nullptr,
                kv.data());
    std::vector<float> o(kInner);
    cpu::Mha(qq.data(), kInner, kv.data(), 2 * kInner, kv.data() + kInner, 2 * kInner, o.data(), kInner, 1,
             t + 1, kPh, kPd, 1.0F / 8.0F, false);
    std::vector<float> ao(kD);
    cpu::Linear(o.data(), 1, kInner, W(lp + "0.to_out.weight", kD * kInner), kD, nullptr, ao.data());
    for (std::int64_t i = 0; i < kD; ++i) {
      lat[static_cast<std::size_t>(i)] += ao[static_cast<std::size_t>(i)];
    }
    std::vector<float> ff(2 * kFf);
    cpu::Linear(lat.data(), 1, kD, W(lp + "1.0.weight", 2 * kFf * kD), 2 * kFf, W(lp + "1.0.bias", 2 * kFf),
                ff.data());
    std::vector<float> g(kFf);
    for (std::int64_t i = 0; i < kFf; ++i) {
      g[static_cast<std::size_t>(i)] = cpu::GeluErf(ff[static_cast<std::size_t>(kFf + i)]) * ff[static_cast<std::size_t>(i)];
    }
    cpu::Linear(g.data(), 1, kFf, W(lp + "1.2.weight", kD * kFf), kD, W(lp + "1.2.bias", kD), ao.data());
    for (std::int64_t i = 0; i < kD; ++i) {
      lat[static_cast<std::size_t>(i)] += ao[static_cast<std::size_t>(i)];
    }
  }
  {
    double ss = 0.0;
    for (const float vv : lat) {
      ss += static_cast<double>(vv) * vv;
    }
    const float norm = std::max(static_cast<float>(std::sqrt(ss)), 1e-12F);
    const float* gamma = W(pp + "norm.gamma", kD);
    const float scale = std::sqrt(static_cast<float>(kD));
    for (std::int64_t i = 0; i < kD; ++i) {
      lat[static_cast<std::size_t>(i)] = lat[static_cast<std::size_t>(i)] / norm * scale * gamma[i];
    }
  }
  if (tap) {
    tap("emo_perceiver", lat);
  }
  const std::int64_t dim = config_.dim;
  std::vector<float> a(static_cast<std::size_t>(dim));
  std::vector<float> b(static_cast<std::size_t>(dim));
  cpu::Linear(lat.data(), 1, kD, W("emovec_layer.weight", dim * kD), dim, W("emovec_layer.bias", dim), a.data());
  cpu::Linear(a.data(), 1, dim, W("emo_layer.weight", dim * dim), dim, W("emo_layer.bias", dim), b.data());
  sd_->ReleaseTensorPages("model." + e);
  sd_->ReleaseTensorPages("model." + pp);
  return b;
}

std::vector<float> UnifiedVoice::SpeakerLatent(std::span<const float> style) const {
  std::vector<float> out(static_cast<std::size_t>(config_.dim));
  cpu::Linear(style.data(), 1, 192, W("spk_emb_proj.weight", config_.dim * 192), config_.dim,
              W("spk_emb_proj.bias", config_.dim), out.data());
  return out;
}

std::vector<float> UnifiedVoice::BuildPrefix(std::span<const float> spk_latent,
                                             std::span<const float> emovec,
                                             std::span<const std::int32_t> text_ids,
                                             std::int32_t lang_token, std::int64_t* rows) const {
  const std::int64_t dim = config_.dim;
  std::vector<std::int32_t> text = {config_.start_text};
  for (const std::int32_t id : text_ids) {
    if (id != config_.stop_text && id != config_.start_text) {
      text.push_back(id);
    }
  }
  text.push_back(config_.stop_text);
  const auto n = static_cast<std::int64_t>(text.size());
  if (n > config_.max_text_positions) {
    throw std::invalid_argument("text segment too long for the GPT text positions");
  }
  *rows = 3 + n;
  std::vector<float> out(static_cast<std::size_t>(*rows * dim), 0.0F);
  for (std::int64_t i = 0; i < dim; ++i) {
    out[static_cast<std::size_t>(i)] = spk_latent[static_cast<std::size_t>(i)] + emovec[static_cast<std::size_t>(i)];
  }
  const float* emb = W("text_embedding.weight");
  const float* pos = W("text_pos_embedding.emb.weight");
  const float* lang = W("lang_embedding.weight") + static_cast<std::int64_t>(lang_token) * dim;
  for (std::int64_t r = 0; r < n; ++r) {
    float* o = out.data() + (3 + r) * dim;
    const float* e = emb + static_cast<std::int64_t>(text[static_cast<std::size_t>(r)]) * dim;
    const float* p = pos + r * dim;
    for (std::int64_t i = 0; i < dim; ++i) {
      o[i] = (e[i] + p[i]) + lang[i];
    }
  }
  return out;
}

void UnifiedVoice::Forward(std::vector<float>& x, std::int64_t rows, State* st,
                           float* last_logits) const {
  const std::int64_t dim = config_.dim;
  const std::int64_t heads = config_.heads;
  const std::int64_t hd = dim / heads;
  if (st->len + rows > st->capacity) {
    throw std::runtime_error("GPT KV cache full");
  }
  std::vector<float> y(x.size());
  std::vector<float> qkv(static_cast<std::size_t>(rows * 3 * dim));
  std::vector<float> att(x.size());
  std::vector<float> mlp(static_cast<std::size_t>(rows * 4 * dim));
  for (std::int64_t l = 0; l < config_.layers; ++l) {
    const std::string p = "gpt.h." + std::to_string(l) + ".";
    cpu::LayerNorm(x.data(), y.data(), rows, dim, W(p + "ln_1.weight"), W(p + "ln_1.bias"), kLnEps);
    cpu::LinearInOut(y.data(), rows, dim, W(p + "attn.c_attn.weight", dim * 3 * dim), 3 * dim,
                     W(p + "attn.c_attn.bias"), qkv.data());
    float* kc = st->k[static_cast<std::size_t>(l)].data();
    float* vc = st->v[static_cast<std::size_t>(l)].data();
    for (std::int64_t r = 0; r < rows; ++r) {
      std::copy_n(qkv.data() + r * 3 * dim + dim, dim, kc + (st->len + r) * dim);
      std::copy_n(qkv.data() + r * 3 * dim + 2 * dim, dim, vc + (st->len + r) * dim);
    }
    cpu::Mha(qkv.data(), 3 * dim, kc, dim, vc, dim, att.data(), dim, rows, st->len + rows, heads, hd,
             1.0F / std::sqrt(static_cast<float>(hd)), true);
    cpu::LinearInOut(att.data(), rows, dim, W(p + "attn.c_proj.weight", dim * dim), dim,
                     W(p + "attn.c_proj.bias"), y.data());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += y[i];
    }
    cpu::LayerNorm(x.data(), y.data(), rows, dim, W(p + "ln_2.weight"), W(p + "ln_2.bias"), kLnEps);
    cpu::LinearInOut(y.data(), rows, dim, W(p + "mlp.c_fc.weight", dim * 4 * dim), 4 * dim,
                     W(p + "mlp.c_fc.bias"), mlp.data());
    cpu::GeluTanhInPlace(mlp.data(), mlp.size());
    cpu::LinearInOut(mlp.data(), rows, 4 * dim, W(p + "mlp.c_proj.weight", 4 * dim * dim), dim,
                     W(p + "mlp.c_proj.bias"), y.data());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += y[i];
    }
  }
  st->len += rows;
  if (last_logits != nullptr) {
    const float* last = x.data() + (rows - 1) * dim;
    std::vector<float> a(static_cast<std::size_t>(dim));
    std::vector<float> b(static_cast<std::size_t>(dim));
    cpu::LayerNorm(last, a.data(), 1, dim, W("gpt.ln_f.weight"), W("gpt.ln_f.bias"), kLnEps);
    cpu::LayerNorm(a.data(), b.data(), 1, dim, W("final_norm.weight"), W("final_norm.bias"), kLnEps);
    cpu::Linear(b.data(), 1, dim, W("mel_head.weight", config_.mel_codes * dim), config_.mel_codes,
                W("mel_head.bias"), last_logits);
  }
}

void UnifiedVoice::Prefill(const std::vector<float>& prefix, std::int64_t rows, State* st,
                           std::vector<float>* logits, std::int64_t max_new) const {
  const std::int64_t dim = config_.dim;
  st->capacity = rows + 1 + max_new + 1;
  st->k.assign(static_cast<std::size_t>(config_.layers),
               std::vector<float>(static_cast<std::size_t>(st->capacity * dim)));
  st->v.assign(static_cast<std::size_t>(config_.layers),
               std::vector<float>(static_cast<std::size_t>(st->capacity * dim)));
  st->len = 0;
  std::vector<float> x(prefix.begin(), prefix.begin() + (rows * dim));
  x.resize(static_cast<std::size_t>((rows + 1) * dim));
  const float* mel = W("mel_embedding.weight") + static_cast<std::int64_t>(config_.start_mel) * dim;
  const float* pos = W("mel_pos_embedding.emb.weight");  // position 0
  for (std::int64_t i = 0; i < dim; ++i) {
    x[static_cast<std::size_t>(rows * dim + i)] = mel[i] + pos[i];
  }
  logits->assign(static_cast<std::size_t>(config_.mel_codes), 0.0F);
  Forward(x, rows + 1, st, logits->data());
  st->mel_tokens = 1;
}

void UnifiedVoice::Step(std::int32_t token, State* st, std::vector<float>* logits) const {
  const std::int64_t dim = config_.dim;
  // GPT2InferenceModel: position = attention_mask.shape[1] - mel_len, i.e. the
  // k-th generated token (k >= 1) sits at mel position k + 1.
  const std::int64_t position = st->mel_tokens + 1;
  if (position >= config_.max_mel_positions) {
    throw std::runtime_error("GPT mel positions exhausted");
  }
  std::vector<float> x(static_cast<std::size_t>(dim));
  const float* mel = W("mel_embedding.weight") + static_cast<std::int64_t>(token) * dim;
  const float* pos = W("mel_pos_embedding.emb.weight") + position * dim;
  for (std::int64_t i = 0; i < dim; ++i) {
    x[static_cast<std::size_t>(i)] = mel[i] + pos[i];
  }
  logits->assign(static_cast<std::size_t>(config_.mel_codes), 0.0F);
  Forward(x, 1, st, logits->data());
  ++st->mel_tokens;
}

void ApplyRepetitionPenalty(std::span<float> scores, std::span<const std::int32_t> history,
                            float penalty) {
  std::vector<std::uint8_t> seen(scores.size(), 0);
  for (const std::int32_t id : history) {
    if (id >= 0 && static_cast<std::size_t>(id) < scores.size()) {
      seen[static_cast<std::size_t>(id)] = 1;
    }
  }
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (seen[i] != 0) {
      scores[i] = scores[i] < 0.0F ? scores[i] * penalty : scores[i] / penalty;
    }
  }
}

std::int32_t SampleToken(std::span<const float> penalized, const SamplingOptions& o, double u,
                         std::int32_t* kept) {
  const std::size_t n = penalized.size();
  std::vector<float> s(penalized.begin(), penalized.end());
  if (o.temperature != 1.0F) {
    for (float& v : s) {
      v = v / o.temperature;
    }
  }
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  if (o.top_k > 0 && static_cast<std::size_t>(o.top_k) < n) {
    std::vector<float> sorted(s);
    std::nth_element(sorted.begin(), sorted.begin() + (o.top_k - 1), sorted.end(), std::greater<>());
    const float kth = sorted[static_cast<std::size_t>(o.top_k - 1)];
    for (float& v : s) {
      if (v < kth) {
        v = kNegInf;
      }
    }
  }
  if (o.top_p < 1.0F) {
    // ascending sort, softmax, cumsum (float32); drop cum <= 1 - top_p, keep the last
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return s[a] < s[b]; });
    float mx = kNegInf;
    for (const float v : s) {
      mx = std::max(mx, v);
    }
    std::vector<float> probs(n);
    float total = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
      probs[i] = std::exp(s[idx[i]] - mx);
      total += probs[i];
    }
    float cum = 0.0F;
    const float threshold = 1.0F - o.top_p;
    for (std::size_t i = 0; i < n; ++i) {
      cum += probs[i] / total;
      if (cum <= threshold && i + 1 < n) {
        s[idx[i]] = kNegInf;
      }
    }
  }
  float mx = kNegInf;
  for (const float v : s) {
    mx = std::max(mx, v);
  }
  std::vector<double> probs(n);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    probs[i] = s[i] == kNegInf ? 0.0 : static_cast<double>(std::exp(s[i] - mx));
    sum += probs[i];
  }
  // float32 softmax, then float64 cumsum and searchsorted(u * total, "right")
  std::vector<double> cum(n);
  double acc = 0.0;
  std::int32_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto p = static_cast<double>(static_cast<float>(probs[i] / sum));
    if (p > 0.0) {
      ++count;
    }
    probs[i] = p;
    acc += p;
    cum[i] = acc;
  }
  if (kept != nullptr) {
    *kept = count;
  }
  const double target = u * acc;
  std::size_t tok = static_cast<std::size_t>(std::upper_bound(cum.begin(), cum.end(), target) - cum.begin());
  tok = std::min(tok, n - 1);
  while (probs[tok] == 0.0 && tok > 0) {
    --tok;
  }
  return static_cast<std::int32_t>(tok);
}

std::vector<float> NormalizeEmotionVector(std::span<const float> v) {
  static constexpr float kBias[8] = {0.9375F, 0.875F, 1.0F, 1.0F, 0.9375F, 0.9375F, 0.6875F, 0.5625F};
  std::vector<double> out(8, 0.0);
  double sum = 0.0;
  for (std::size_t i = 0; i < 8 && i < v.size(); ++i) {
    out[i] = static_cast<double>(v[i]) * kBias[i];
    sum += out[i];
  }
  if (sum > 0.8) {
    for (double& x : out) {
      x = x * 0.8 / sum;
    }
  }
  return {out.begin(), out.end()};
}

std::unique_ptr<EmotionMatrix> EmotionMatrix::Load(const std::filesystem::path& feat1,
                                                   const std::filesystem::path& feat2,
                                                   std::string* error) {
  auto a = PthArchive::Open(feat1, error);
  auto b = PthArchive::Open(feat2, error);
  if (a == nullptr || b == nullptr) {
    return nullptr;
  }
  auto out = std::unique_ptr<EmotionMatrix>(new EmotionMatrix());
  try {
    out->spk_ = a->ToF32("");
    out->emo_ = b->ToF32("");
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = e.what();
    }
    return nullptr;
  }
  if (out->spk_.size() != 73 * 192 || out->emo_.size() != 73 * 1280) {
    if (error != nullptr) {
      *error = "feat1/feat2 have unexpected shapes";
    }
    return nullptr;
  }
  return out;
}

std::vector<float> EmotionMatrix::Mix(std::span<const float> w, std::span<const float> style,
                                      std::vector<std::int32_t>* index_out) const {
  static constexpr int kNum[8] = {3, 17, 2, 8, 4, 5, 10, 24};
  std::vector<float> out(1280, 0.0F);
  int base = 0;
  double snorm = 0.0;
  for (int i = 0; i < 192; ++i) {
    snorm += static_cast<double>(style[static_cast<std::size_t>(i)]) * style[static_cast<std::size_t>(i)];
  }
  if (index_out != nullptr) {
    index_out->clear();
  }
  for (int g = 0; g < 8; ++g) {
    int best = 0;
    double best_sim = -2.0;
    for (int r = 0; r < kNum[g]; ++r) {
      const float* row = spk_.data() + static_cast<std::size_t>(base + r) * 192;
      double dot = 0.0;
      double rn = 0.0;
      for (int i = 0; i < 192; ++i) {
        dot += static_cast<double>(style[static_cast<std::size_t>(i)]) * row[i];
        rn += static_cast<double>(row[i]) * row[i];
      }
      const double sim = dot / std::sqrt(std::max(snorm * rn, 1e-16));
      if (sim > best_sim) {
        best_sim = sim;
        best = r;
      }
    }
    if (index_out != nullptr) {
      index_out->push_back(best);
    }
    const float* e = emo_.data() + static_cast<std::size_t>(base + best) * 1280;
    for (int i = 0; i < 1280; ++i) {
      out[static_cast<std::size_t>(i)] += w[static_cast<std::size_t>(g)] * e[i];
    }
    base += kNum[g];
  }
  return out;
}

}  // namespace gufo::models::indextts
