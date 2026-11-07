#include "src/models/indextts/s2mel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/indextts/cpu/ops.hpp"

namespace gufo::models::indextts {
namespace {

constexpr std::int64_t kSem = 1024;
constexpr std::int64_t kVocos = 384;
constexpr std::int64_t kVocosFfn = 2048;
constexpr int kVocosLayers = 12;

constexpr std::int64_t kD = 512;      // DiT hidden
constexpr std::int64_t kHeads = 8;
constexpr std::int64_t kHd = 64;
constexpr std::int64_t kFfn = 1536;   // find_multiple(int(2 * 2048 / 3), 256)
constexpr int kDepth = 13;
constexpr std::int64_t kMel = 80;
constexpr std::int64_t kStyle = 192;
constexpr std::int64_t kWn = 512;
constexpr int kWnLayers = 8;
constexpr std::int64_t kMaxRope = 16384;

void Mish(std::vector<float>* x) {
  for (float& v : *x) {
    v = cpu::Mish(v);
  }
}

}  // namespace

std::vector<float> TorchLinspace01(std::int32_t n) {
  const std::int32_t steps = n + 1;
  std::vector<float> out(static_cast<std::size_t>(steps));
  const float start = 0.0F;
  const float end = 1.0F;
  const float step = (end - start) / static_cast<float>(steps - 1);
  const std::int32_t halfway = steps / 2;
  for (std::int32_t i = 0; i < steps; ++i) {
    out[static_cast<std::size_t>(i)] =
        i < halfway ? start + step * static_cast<float>(i)
                    : end - step * static_cast<float>(steps - i - 1);
  }
  return out;
}

// ------------------------------------------------------------- codec ------

std::unique_ptr<SemanticCodec> SemanticCodec::Load(const std::filesystem::path& path,
                                                   std::string* error) {
  auto sd = PthArchive::Open(path, error);
  if (sd == nullptr) {
    return nullptr;
  }
  auto out = std::unique_ptr<SemanticCodec>(new SemanticCodec());
  try {
    out->out_project_ = FoldWeightNorm(*sd, "model.quantizer.quantizers.0.out_project");
    out->embed_.Set(sd->F32("model.decoder.0.embed.weight", kVocos * kSem * 7),
                    sd->F32("model.decoder.0.embed.bias", kVocos), kSem, kVocos, 7, 1, 3, 3);
    out->up_.Set(sd->F32("model.up.weight", kSem * kSem * 3), sd->F32("model.up.bias", kSem), kSem,
                 kSem, 3, 1, 1, 1);
    (void)sd->F32("model.quantizer.quantizers.0.codebook.weight", 8192 * 8);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = path.string() + ": " + e.what();
    }
    return nullptr;
  }
  out->sd_ = std::move(sd);
  return out;
}

std::vector<float> SemanticCodec::Decode(std::span<const std::int32_t> codes, std::vector<float>* vq,
                                         std::vector<float>* dec) const {
  const PthArchive& sd = *sd_;
  const auto n = static_cast<std::int64_t>(codes.size());
  const float* book = sd.F32("model.quantizer.quantizers.0.codebook.weight", 8192 * 8);
  const float* ob = sd.F32("model.quantizer.quantizers.0.out_project.bias", kSem);
  std::vector<float> q(static_cast<std::size_t>(n * kSem));
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int32_t c = codes[static_cast<std::size_t>(i)];
    if (c < 0 || c >= 8192) {
      throw std::invalid_argument("semantic code out of range");
    }
    cpu::Linear(book + static_cast<std::int64_t>(c) * 8, 1, 8, out_project_.data(), kSem, ob,
                q.data() + i * kSem);
  }
  if (vq != nullptr) {
    *vq = q;
  }
  // VocosBackbone
  std::vector<float> x;
  embed_.Run(q.data(), n, &x);
  const std::string p = "model.decoder.0.";
  cpu::LayerNorm(x.data(), x.data(), n, kVocos, sd.F32(p + "norm.weight"), sd.F32(p + "norm.bias"), 1e-6F);
  std::vector<float> y(x.size());
  std::vector<float> wide(static_cast<std::size_t>(n * kVocosFfn));
  for (int l = 0; l < kVocosLayers; ++l) {
    const std::string lp = p + "convnext." + std::to_string(l) + ".";
    cpu::DepthwiseTM(x.data(), n, kVocos, sd.F32(lp + "dwconv.weight", kVocos * 7), sd.F32(lp + "dwconv.bias"), 7,
                     3, y.data());
    cpu::LayerNorm(y.data(), y.data(), n, kVocos, sd.F32(lp + "norm.weight"), sd.F32(lp + "norm.bias"), 1e-6F);
    cpu::Linear(y.data(), n, kVocos, sd.F32(lp + "pwconv1.weight", kVocosFfn * kVocos), kVocosFfn,
                sd.F32(lp + "pwconv1.bias"), wide.data());
    cpu::GeluErfInPlace(wide.data(), wide.size());
    cpu::Linear(wide.data(), n, kVocosFfn, sd.F32(lp + "pwconv2.weight", kVocos * kVocosFfn), kVocos,
                sd.F32(lp + "pwconv2.bias"), y.data());
    const float* gamma = sd.F32(lp + "gamma", kVocos);
    for (std::int64_t r = 0; r < n; ++r) {
      for (std::int64_t c = 0; c < kVocos; ++c) {
        x[static_cast<std::size_t>(r * kVocos + c)] += gamma[c] * y[static_cast<std::size_t>(r * kVocos + c)];
      }
    }
  }
  cpu::LayerNorm(x.data(), x.data(), n, kVocos, sd.F32(p + "final_layer_norm.weight"),
                 sd.F32(p + "final_layer_norm.bias"), 1e-6F);
  std::vector<float> h(static_cast<std::size_t>(n * kSem));
  cpu::Linear(x.data(), n, kVocos, sd.F32("model.decoder.1.weight", kSem * kVocos), kSem,
              sd.F32("model.decoder.1.bias", kSem), h.data());
  if (dec != nullptr) {
    *dec = h;
  }
  // nearest x2 + up conv
  std::vector<float> h2(static_cast<std::size_t>(2 * n * kSem));
  for (std::int64_t r = 0; r < 2 * n; ++r) {
    std::copy_n(h.data() + (r / 2) * kSem, kSem, h2.data() + r * kSem);
  }
  std::vector<float> s;
  up_.Run(h2.data(), 2 * n, &s);
  return s;
}

// ------------------------------------------------------------- s2mel ------

std::unique_ptr<S2Mel> S2Mel::Load(const std::filesystem::path& path, std::string* error) {
  auto sd = PthArchive::Open(path, error);
  if (sd == nullptr) {
    return nullptr;
  }
  auto out = std::unique_ptr<S2Mel>(new S2Mel());
  try {
    const std::string lr = "net.length_regulator.model.";
    for (int i = 0; i < 4; ++i) {
      const std::string cp = lr + std::to_string(i * 3);
      out->lr_convs_[i].Set(sd->F32(cp + ".weight", kD * kD * 3), sd->F32(cp + ".bias", kD), kD, kD, 3, 1,
                            1, 1);
    }
    out->lr_out_.Set(sd->F32(lr + "12.weight", kD * kD), sd->F32(lr + "12.bias", kD), kD, kD, 1, 1, 0, 0);
    const std::string wn = "net.cfm.estimator.wavenet.";
    out->wn_cond_ = FoldWeightNorm(*sd, wn + "cond_layer.conv.conv");
    for (int i = 0; i < kWnLayers; ++i) {
      const std::string ip = wn + "in_layers." + std::to_string(i) + ".conv.conv";
      const std::vector<float> w = FoldWeightNorm(*sd, ip);
      out->wn_in_[i].Set(w.data(), sd->F32(ip + ".bias", 2 * kWn), kWn, 2 * kWn, 5, 1, 2, 2,
                         cpu::PadMode::kReflect);
      const std::string rp = wn + "res_skip_layers." + std::to_string(i) + ".conv.conv";
      out->wn_res_[i] = FoldWeightNorm(*sd, rp);
      out->wn_res_out_[i] = i < kWnLayers - 1 ? 2 * kWn : kWn;
    }
    out->final_linear_ = FoldWeightNorm(*sd, "net.cfm.estimator.final_layer.linear");
    // gpt-fast precompute_freqs_cis(16384, 64, 10000) in float32
    out->rope_cos_.resize(static_cast<std::size_t>(kMaxRope * kHd / 2));
    out->rope_sin_.resize(out->rope_cos_.size());
    for (std::int64_t i = 0; i < kHd / 2; ++i) {
      const float freq =
          1.0F / std::pow(10000.0F, static_cast<float>(2 * i) / static_cast<float>(kHd));
      for (std::int64_t pos = 0; pos < kMaxRope; ++pos) {
        const float angle = static_cast<float>(pos) * freq;
        out->rope_cos_[static_cast<std::size_t>(pos * kHd / 2 + i)] = std::cos(angle);
        out->rope_sin_[static_cast<std::size_t>(pos * kHd / 2 + i)] = std::sin(angle);
      }
    }
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = path.string() + ": " + e.what();
    }
    return nullptr;
  }
  out->sd_ = std::move(sd);
  return out;
}

const float* S2Mel::W(const std::string& name, std::int64_t numel) const {
  return sd_->F32("net.cfm.estimator." + name, numel);
}

std::vector<float> S2Mel::Regulate(const float* x, std::int64_t t, std::int64_t ylens) const {
  const PthArchive& sd = *sd_;
  std::vector<float> h(static_cast<std::size_t>(t * kD));
  cpu::Linear(x, t, kSem, sd.F32("net.length_regulator.content_in_proj.weight", kD * kSem), kD,
              sd.F32("net.length_regulator.content_in_proj.bias", kD), h.data());
  // F.interpolate(mode="nearest", size=ylens)
  std::vector<float> y(static_cast<std::size_t>(ylens * kD));
  const float scale = static_cast<float>(t) / static_cast<float>(ylens);
  for (std::int64_t o = 0; o < ylens; ++o) {
    std::int64_t src;
    if (ylens == t) {
      src = o;
    } else if (ylens == 2 * t) {
      src = o >> 1;
    } else {
      src = std::min(static_cast<std::int64_t>(std::floor(static_cast<float>(o) * scale)), t - 1);
    }
    std::copy_n(h.data() + src * kD, kD, y.data() + o * kD);
  }
  std::vector<float> z;
  for (int i = 0; i < 4; ++i) {
    lr_convs_[i].Run(y.data(), ylens, &z);
    const std::string gp = "net.length_regulator.model." + std::to_string(i * 3 + 1);
    cpu::GroupNorm1TM(z.data(), ylens, kD, sd.F32(gp + ".weight", kD), sd.F32(gp + ".bias", kD), 1e-5F);
    Mish(&z);
    y.swap(z);
  }
  lr_out_.Run(y.data(), ylens, &z);
  return z;
}

std::vector<float> S2Mel::TimeEmbed(const std::string& prefix, float time) const {
  const float* freqs = W(prefix + ".freqs", 128);
  std::vector<float> emb(256);
  for (int i = 0; i < 128; ++i) {
    const float arg = (1000.0F * time) * freqs[i];
    emb[static_cast<std::size_t>(i)] = std::cos(arg);
    emb[static_cast<std::size_t>(128 + i)] = std::sin(arg);
  }
  std::vector<float> h(kD);
  cpu::Linear(emb.data(), 1, 256, W(prefix + ".mlp.0.weight", kD * 256), kD, W(prefix + ".mlp.0.bias"), h.data());
  cpu::SiluInPlace(h.data(), h.size());
  std::vector<float> out(kD);
  cpu::Linear(h.data(), 1, kD, W(prefix + ".mlp.2.weight", kD * kD), kD, W(prefix + ".mlp.2.bias"), out.data());
  return out;
}

S2Mel::TimeConditioning S2Mel::TimeCondition(float time) const {
  TimeConditioning out;
  const std::vector<float> t1 = TimeEmbed("t_embedder", time);
  out.ada.resize(static_cast<std::size_t>((2 * kDepth + 1) * 2 * kD));
  auto proj = [&](const std::string& p, float* dst) {
    cpu::Linear(t1.data(), 1, kD, W(p + ".project_layer.weight", 2 * kD * kD), 2 * kD,
                W(p + ".project_layer.bias"), dst);
  };
  for (int l = 0; l < kDepth; ++l) {
    const std::string p = "transformer.layers." + std::to_string(l);
    proj(p + ".attention_norm", out.ada.data() + (2 * l) * 2 * kD);
    proj(p + ".ffn_norm", out.ada.data() + (2 * l + 1) * 2 * kD);
  }
  proj("transformer.norm", out.ada.data() + (2 * kDepth) * 2 * kD);
  const std::vector<float> t2 = TimeEmbed("t_embedder2", time);
  out.wavenet.resize(static_cast<std::size_t>(2 * kWn * kWnLayers));
  cpu::Linear(t2.data(), 1, kWn, wn_cond_.data(), 2 * kWn * kWnLayers,
              W("wavenet.cond_layer.conv.conv.bias", 2 * kWn * kWnLayers), out.wavenet.data());
  std::vector<float> st1(t1);
  cpu::SiluInPlace(st1.data(), st1.size());
  out.final_mod.resize(2 * kWn);
  cpu::Linear(st1.data(), 1, kWn, W("final_layer.adaLN_modulation.1.weight", 2 * kWn * kWn), 2 * kWn,
              W("final_layer.adaLN_modulation.1.bias"), out.final_mod.data());
  return out;
}

std::vector<float> S2Mel::Velocity(const float* x, const float* prompt, float time, const float* style,
                                   const float* cond, std::int64_t t) const {
  const std::vector<float> t1 = TimeEmbed("t_embedder", time);
  // x_in = [x, prompt, cond_projection(cond), style] -> cond_x_merge_linear
  constexpr std::int64_t kIn = kMel * 2 + kD + kStyle;
  std::vector<float> cp(static_cast<std::size_t>(t * kD));
  cpu::Linear(cond, t, kD, W("cond_projection.weight", kD * kD), kD, W("cond_projection.bias"), cp.data());
  std::vector<float> xin(static_cast<std::size_t>(t * kIn));
  for (std::int64_t r = 0; r < t; ++r) {
    float* o = xin.data() + r * kIn;
    std::copy_n(x + r * kMel, kMel, o);
    std::copy_n(prompt + r * kMel, kMel, o + kMel);
    std::copy_n(cp.data() + r * kD, kD, o + 2 * kMel);
    std::copy_n(style, kStyle, o + 2 * kMel + kD);
  }
  std::vector<float> h(static_cast<std::size_t>(t * kD));
  cpu::Linear(xin.data(), t, kIn, W("cond_x_merge_linear.weight", kD * kIn), kD, W("cond_x_merge_linear.bias"),
              h.data());
  // gpt-fast Transformer with adaptive RMSNorm and UViT skips
  auto ada_norm = [&](const std::string& p, const float* in, float* out) {
    std::vector<float> wb(2 * kD);
    cpu::Linear(t1.data(), 1, kD, W(p + ".project_layer.weight", 2 * kD * kD), 2 * kD, W(p + ".project_layer.bias"),
                wb.data());
    cpu::RmsNorm(in, out, t, kD, W(p + ".norm.weight", kD), 1e-5F);
    for (std::int64_t r = 0; r < t; ++r) {
      float* row = out + r * kD;
      for (std::int64_t c = 0; c < kD; ++c) {
        row[c] = wb[static_cast<std::size_t>(c)] * row[c] + wb[static_cast<std::size_t>(kD + c)];
      }
    }
  };
  std::vector<std::vector<float>> skips;
  std::vector<float> y(h.size());
  std::vector<float> qkv(static_cast<std::size_t>(t * 3 * kD));
  std::vector<float> att(h.size());
  std::vector<float> f1(static_cast<std::size_t>(t * kFfn));
  std::vector<float> f3(f1.size());
  std::vector<float> cat(static_cast<std::size_t>(t * 2 * kD));
  for (int l = 0; l < kDepth; ++l) {
    const std::string p = "transformer.layers." + std::to_string(l);
    if (l > kDepth / 2) {  // layers_receive_skip
      const std::vector<float> skip = std::move(skips.back());
      skips.pop_back();
      for (std::int64_t r = 0; r < t; ++r) {
        std::copy_n(h.data() + r * kD, kD, cat.data() + r * 2 * kD);
        std::copy_n(skip.data() + r * kD, kD, cat.data() + r * 2 * kD + kD);
      }
      cpu::Linear(cat.data(), t, 2 * kD, W(p + ".skip_in_linear.weight", kD * 2 * kD), kD,
                  W(p + ".skip_in_linear.bias"), h.data());
    }
    ada_norm(p + ".attention_norm", h.data(), y.data());
    cpu::Linear(y.data(), t, kD, W(p + ".attention.wqkv.weight", 3 * kD * kD), 3 * kD, nullptr, qkv.data());
    // interleaved RoPE on q and k
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < t; ++r) {
      for (int part = 0; part < 2; ++part) {
        float* v = qkv.data() + r * 3 * kD + part * kD;
        for (std::int64_t hh = 0; hh < kHeads; ++hh) {
          for (std::int64_t i = 0; i < kHd / 2; ++i) {
            const float c = rope_cos_[static_cast<std::size_t>(r * kHd / 2 + i)];
            const float s = rope_sin_[static_cast<std::size_t>(r * kHd / 2 + i)];
            float* pr = v + hh * kHd + 2 * i;
            const float a = pr[0];
            const float b = pr[1];
            pr[0] = a * c - b * s;
            pr[1] = b * c + a * s;
          }
        }
      }
    }
    cpu::Mha(qkv.data(), 3 * kD, qkv.data() + kD, 3 * kD, qkv.data() + 2 * kD, 3 * kD, att.data(), kD, t, t,
             kHeads, kHd, 1.0F / std::sqrt(static_cast<float>(kHd)), false);
    cpu::Linear(att.data(), t, kD, W(p + ".attention.wo.weight", kD * kD), kD, nullptr, y.data());
    for (std::size_t i = 0; i < h.size(); ++i) {
      h[i] += y[i];
    }
    ada_norm(p + ".ffn_norm", h.data(), y.data());
    cpu::Linear(y.data(), t, kD, W(p + ".feed_forward.w1.weight", kFfn * kD), kFfn, nullptr, f1.data());
    cpu::Linear(y.data(), t, kD, W(p + ".feed_forward.w3.weight", kFfn * kD), kFfn, nullptr, f3.data());
    for (std::size_t i = 0; i < f1.size(); ++i) {
      f1[i] = cpu::Silu(f1[i]) * f3[i];
    }
    cpu::Linear(f1.data(), t, kFfn, W(p + ".feed_forward.w2.weight", kD * kFfn), kD, nullptr, y.data());
    for (std::size_t i = 0; i < h.size(); ++i) {
      h[i] += y[i];
    }
    if (l < kDepth / 2) {  // layers_emit_skip
      skips.push_back(h);
    }
  }
  ada_norm("transformer.norm", h.data(), y.data());
  // long skip: skip_linear(cat[x_res, x])
  std::vector<float> c2(static_cast<std::size_t>(t * (kD + kMel)));
  for (std::int64_t r = 0; r < t; ++r) {
    std::copy_n(y.data() + r * kD, kD, c2.data() + r * (kD + kMel));
    std::copy_n(x + r * kMel, kMel, c2.data() + r * (kD + kMel) + kD);
  }
  std::vector<float> xres(static_cast<std::size_t>(t * kD));
  cpu::Linear(c2.data(), t, kD + kMel, W("skip_linear.weight", kD * (kD + kMel)), kD, W("skip_linear.bias"),
              xres.data());
  // WaveNet head
  std::vector<float> wx(static_cast<std::size_t>(t * kWn));
  cpu::Linear(xres.data(), t, kD, W("conv1.weight", kWn * kD), kWn, W("conv1.bias"), wx.data());
  const std::vector<float> t2 = TimeEmbed("t_embedder2", time);
  std::vector<float> g(static_cast<std::size_t>(2 * kWn * kWnLayers));
  cpu::Linear(t2.data(), 1, kWn, wn_cond_.data(), 2 * kWn * kWnLayers,
              W("wavenet.cond_layer.conv.conv.bias", 2 * kWn * kWnLayers), g.data());
  std::vector<float> output(static_cast<std::size_t>(t * kWn), 0.0F);
  std::vector<float> xin_l;
  std::vector<float> acts(static_cast<std::size_t>(t * kWn));
  std::vector<float> rs(static_cast<std::size_t>(t * 2 * kWn));
  for (int i = 0; i < kWnLayers; ++i) {
    wn_in_[i].Run(wx.data(), t, &xin_l);
    const float* gl = g.data() + i * 2 * kWn;
    for (std::int64_t r = 0; r < t; ++r) {
      const float* a = xin_l.data() + r * 2 * kWn;
      float* o = acts.data() + r * kWn;
      for (std::int64_t c = 0; c < kWn; ++c) {
        o[c] = std::tanh(a[c] + gl[c]) * cpu::Sigmoid(a[kWn + c] + gl[kWn + c]);
      }
    }
    const std::int64_t ro = wn_res_out_[i];
    const std::string rp = "wavenet.res_skip_layers." + std::to_string(i) + ".conv.conv.bias";
    cpu::Linear(acts.data(), t, kWn, wn_res_[i].data(), ro, W(rp, ro), rs.data());
    if (i < kWnLayers - 1) {
      for (std::int64_t r = 0; r < t; ++r) {
        for (std::int64_t c = 0; c < kWn; ++c) {
          wx[static_cast<std::size_t>(r * kWn + c)] += rs[static_cast<std::size_t>(r * ro + c)];
          output[static_cast<std::size_t>(r * kWn + c)] += rs[static_cast<std::size_t>(r * ro + kWn + c)];
        }
      }
    } else {
      for (std::int64_t r = 0; r < t; ++r) {
        for (std::int64_t c = 0; c < kWn; ++c) {
          output[static_cast<std::size_t>(r * kWn + c)] += rs[static_cast<std::size_t>(r * ro + c)];
        }
      }
    }
  }
  std::vector<float> rp(static_cast<std::size_t>(t * kWn));
  cpu::Linear(xres.data(), t, kD, W("res_projection.weight", kWn * kD), kWn, W("res_projection.bias"), rp.data());
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] += rp[i];
  }
  // FinalLayer(t1): adaLN modulation, LN (no affine, 1e-6), weight-normed linear
  std::vector<float> st1(t1);
  cpu::SiluInPlace(st1.data(), st1.size());
  std::vector<float> mod(2 * kWn);
  cpu::Linear(st1.data(), 1, kWn, W("final_layer.adaLN_modulation.1.weight", 2 * kWn * kWn), 2 * kWn,
              W("final_layer.adaLN_modulation.1.bias"), mod.data());
  cpu::LayerNorm(output.data(), output.data(), t, kWn, nullptr, nullptr, 1e-6F);
  for (std::int64_t r = 0; r < t; ++r) {
    float* row = output.data() + r * kWn;
    for (std::int64_t c = 0; c < kWn; ++c) {
      row[c] = row[c] * (1.0F + mod[static_cast<std::size_t>(kWn + c)]) + mod[static_cast<std::size_t>(c)];
    }
  }
  std::vector<float> fl(static_cast<std::size_t>(t * kWn));
  cpu::Linear(output.data(), t, kWn, final_linear_.data(), kWn, W("final_layer.linear.bias", kWn), fl.data());
  std::vector<float> v(static_cast<std::size_t>(t * kMel));
  cpu::Linear(fl.data(), t, kWn, W("conv2.weight", kMel * kWn), kMel, W("conv2.bias"), v.data());
  return v;
}

std::vector<float> S2Mel::Sample(std::vector<float> x, const float* mu, std::int64_t t,
                                 const float* prompt_mel, std::int64_t prompt_len, const float* style,
                                 const CfmOptions& options, const StepTap& velocity_tap) const {
  const std::vector<float> span = TorchLinspace01(options.steps);
  std::vector<float> prompt_x(static_cast<std::size_t>(t * kMel), 0.0F);
  std::copy_n(prompt_mel, prompt_len * kMel, prompt_x.data());
  std::fill(x.begin(), x.begin() + prompt_len * kMel, 0.0F);
  const std::vector<float> zeros_prompt(prompt_x.size(), 0.0F);
  const std::vector<float> zeros_mu(static_cast<std::size_t>(t * kD), 0.0F);
  const std::vector<float> zeros_style(kStyle, 0.0F);
  float time = span[0];
  for (int step = 1; step <= options.steps; ++step) {
    const float dt = span[static_cast<std::size_t>(step)] - span[static_cast<std::size_t>(step - 1)];
    std::vector<float> vc = Velocity(x.data(), prompt_x.data(), time, style, mu, t);
    if (options.cfg_rate > 0.0F) {
      const std::vector<float> vu =
          Velocity(x.data(), zeros_prompt.data(), time, zeros_style.data(), zeros_mu.data(), t);
      if (velocity_tap) {
        velocity_tap(step - 1, vc, vu);
      }
      const float a = 1.0F + options.cfg_rate;
      for (std::size_t i = 0; i < vc.size(); ++i) {
        vc[i] = a * vc[i] - options.cfg_rate * vu[i];
      }
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = x[i] + dt * vc[i];
    }
    time = time + dt;
    std::fill(x.begin(), x.begin() + prompt_len * kMel, 0.0F);
  }
  return x;
}

}  // namespace gufo::models::indextts
