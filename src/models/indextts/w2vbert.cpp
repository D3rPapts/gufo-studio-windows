#include "src/models/indextts/w2vbert.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/models/indextts/cpu/ops.hpp"

namespace gufo::models::indextts {
namespace {

constexpr std::int64_t kH = W2vBert::kHidden;
constexpr std::int64_t kHeads = 16;
constexpr std::int64_t kHeadDim = 64;
constexpr std::int64_t kFfn = 4096;
constexpr std::int64_t kLeft = 64;
constexpr std::int64_t kRight = 8;
constexpr std::int64_t kKernel = 31;
constexpr float kEps = 1e-5F;

}  // namespace

std::unique_ptr<W2vBert> W2vBert::Load(const std::filesystem::path& model_dir,
                                       const std::filesystem::path& stats_pt,
                                       std::string* error) {
  auto st = ming_tts::SafetensorsFile::Open(model_dir / "model.safetensors", error);
  if (st == nullptr) {
    return nullptr;
  }
  auto stats = PthArchive::Open(stats_pt, error);
  if (stats == nullptr) {
    return nullptr;
  }
  auto out = std::unique_ptr<W2vBert>(new W2vBert());
  out->st_ = std::move(st);
  try {
    out->mean_ = stats->ToF32("mean");
    const std::vector<float> var = stats->ToF32("var");
    out->std_.resize(var.size());
    for (std::size_t i = 0; i < var.size(); ++i) {
      out->std_[i] = std::sqrt(var[i]);
    }
    (void)out->F("feature_projection.projection.weight", kH * 160);
    (void)out->F("encoder.layers.16.final_layer_norm.weight", kH);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("w2v-bert-2.0: ") + e.what();
    }
    return nullptr;
  }
  return out;
}

const float* W2vBert::F(const std::string& name, std::int64_t numel) const {
  const ming_tts::TensorView& t = st_->Require(name);
  if (t.dtype != ming_tts::DType::kF32 || t.NumElements() != numel) {
    throw std::runtime_error("w2v-bert tensor " + name + " has unexpected dtype/size");
  }
  return reinterpret_cast<const float*>(t.data);
}

std::vector<float> W2vBert::Embed(const W2vFeatures& in, const Tap& tap) const {
  const std::int64_t t = in.rows;
  std::vector<float> h(static_cast<std::size_t>(t * kH));
  {
    std::vector<float> ln(static_cast<std::size_t>(t * 160));
    cpu::LayerNorm(in.features.data(), ln.data(), t, 160, F("feature_projection.layer_norm.weight", 160),
                   F("feature_projection.layer_norm.bias", 160), kEps);
    cpu::Linear(ln.data(), t, 160, F("feature_projection.projection.weight", kH * 160), kH,
                F("feature_projection.projection.bias", kH), h.data());
  }
  if (tap) {
    tap("proj", h);
  }
  std::vector<std::uint8_t> valid(in.mask.begin(), in.mask.end());
  for (std::int64_t r = 0; r < t; ++r) {
    if (valid[static_cast<std::size_t>(r)] == 0) {
      std::fill(h.begin() + r * kH, h.begin() + (r + 1) * kH, 0.0F);
    }
  }
  std::vector<float> x(static_cast<std::size_t>(t * kH));
  std::vector<float> mid(static_cast<std::size_t>(t * kFfn));
  std::vector<float> qkv(static_cast<std::size_t>(3 * t * kH));
  std::vector<float> att(static_cast<std::size_t>(t * kH));
  std::vector<float> glu(static_cast<std::size_t>(t * 2 * kH));
  const float min_value = std::numeric_limits<float>::lowest();
  for (int layer = 0; layer < kLayers; ++layer) {
    const std::string p = "encoder.layers." + std::to_string(layer) + ".";
    auto ffn = [&](const std::string& name) {
      cpu::LayerNorm(h.data(), x.data(), t, kH, F(p + name + "_layer_norm.weight", kH),
                     F(p + name + "_layer_norm.bias", kH), kEps);
      cpu::Linear(x.data(), t, kH, F(p + name + ".intermediate_dense.weight", kFfn * kH), kFfn,
                  F(p + name + ".intermediate_dense.bias", kFfn), mid.data());
      cpu::SiluInPlace(mid.data(), mid.size());
      cpu::Linear(mid.data(), t, kFfn, F(p + name + ".output_dense.weight", kH * kFfn), kH,
                  F(p + name + ".output_dense.bias", kH), x.data());
      for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] = x[i] * 0.5F + h[i];
      }
    };
    ffn("ffn1");
    // self attention with relative_key distance embeddings
    cpu::LayerNorm(h.data(), x.data(), t, kH, F(p + "self_attn_layer_norm.weight", kH),
                   F(p + "self_attn_layer_norm.bias", kH), kEps);
    float* q = qkv.data();
    float* k = q + t * kH;
    float* v = k + t * kH;
    cpu::Linear(x.data(), t, kH, F(p + "self_attn.linear_q.weight", kH * kH), kH,
                F(p + "self_attn.linear_q.bias", kH), q);
    cpu::Linear(x.data(), t, kH, F(p + "self_attn.linear_k.weight", kH * kH), kH,
                F(p + "self_attn.linear_k.bias", kH), k);
    cpu::Linear(x.data(), t, kH, F(p + "self_attn.linear_v.weight", kH * kH), kH,
                F(p + "self_attn.linear_v.bias", kH), v);
    const float* dist = F(p + "self_attn.distance_embedding.weight", (kLeft + kRight + 1) * kHeadDim);
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (std::int64_t head = 0; head < kHeads; ++head) {
      for (std::int64_t l = 0; l < t; ++l) {
        const float* ql = q + l * kH + head * kHeadDim;
        float rel[kLeft + kRight + 1];
        for (std::int64_t d = 0; d < kLeft + kRight + 1; ++d) {
          float s = 0.0F;
          for (std::int64_t e = 0; e < kHeadDim; ++e) {
            s += ql[e] * dist[d * kHeadDim + e];
          }
          rel[d] = s;
        }
        std::vector<float> sc(static_cast<std::size_t>(t));
        float mx = -std::numeric_limits<float>::infinity();
        for (std::int64_t r = 0; r < t; ++r) {
          const float* kr = k + r * kH + head * kHeadDim;
          float s = 0.0F;
          for (std::int64_t e = 0; e < kHeadDim; ++e) {
            s += ql[e] * kr[e];
          }
          const std::int64_t d = std::clamp<std::int64_t>(r - l, -kLeft, kRight) + kLeft;
          float val = s / std::sqrt(static_cast<float>(kHeadDim)) + rel[d] * scale;
          if (valid[static_cast<std::size_t>(r)] == 0) {
            val += min_value;
          }
          sc[static_cast<std::size_t>(r)] = val;
          mx = std::max(mx, val);
        }
        double sum = 0.0;
        for (std::int64_t r = 0; r < t; ++r) {
          const float e = std::exp(sc[static_cast<std::size_t>(r)] - mx);
          sc[static_cast<std::size_t>(r)] = e;
          sum += e;
        }
        const auto inv = static_cast<float>(1.0 / sum);
        float* o = att.data() + l * kH + head * kHeadDim;
        std::fill(o, o + kHeadDim, 0.0F);
        for (std::int64_t r = 0; r < t; ++r) {
          const float pr = sc[static_cast<std::size_t>(r)] * inv;
          const float* vr = v + r * kH + head * kHeadDim;
          for (std::int64_t e = 0; e < kHeadDim; ++e) {
            o[e] += pr * vr[e];
          }
        }
      }
    }
    cpu::Linear(att.data(), t, kH, F(p + "self_attn.linear_out.weight", kH * kH), kH,
                F(p + "self_attn.linear_out.bias", kH), x.data());
    for (std::size_t i = 0; i < h.size(); ++i) {
      h[i] += x[i];
    }
    // convolution module
    cpu::LayerNorm(h.data(), x.data(), t, kH, F(p + "conv_module.layer_norm.weight", kH),
                   F(p + "conv_module.layer_norm.bias", kH), kEps);
    for (std::int64_t r = 0; r < t; ++r) {
      if (valid[static_cast<std::size_t>(r)] == 0) {
        std::fill(x.begin() + r * kH, x.begin() + (r + 1) * kH, 0.0F);
      }
    }
    cpu::Linear(x.data(), t, kH, F(p + "conv_module.pointwise_conv1.weight", 2 * kH * kH), 2 * kH,
                nullptr, glu.data());
    for (std::int64_t r = 0; r < t; ++r) {  // GLU over channels
      float* g = glu.data() + r * 2 * kH;
      float* o = x.data() + r * kH;
      for (std::int64_t c = 0; c < kH; ++c) {
        o[c] = g[c] * cpu::Sigmoid(g[kH + c]);
      }
    }
    const float* dw = F(p + "conv_module.depthwise_conv.weight", kH * kKernel);
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < t; ++r) {  // causal depthwise conv (left pad 30)
      float* o = att.data() + r * kH;
      for (std::int64_t c = 0; c < kH; ++c) {
        float s = 0.0F;
        for (std::int64_t j = 0; j < kKernel; ++j) {
          const std::int64_t src = r - (kKernel - 1) + j;
          if (src >= 0) {
            s += dw[c * kKernel + j] * x[static_cast<std::size_t>(src * kH + c)];
          }
        }
        o[c] = s;
      }
    }
    cpu::LayerNorm(att.data(), x.data(), t, kH, F(p + "conv_module.depthwise_layer_norm.weight", kH),
                   F(p + "conv_module.depthwise_layer_norm.bias", kH), kEps);
    cpu::SiluInPlace(x.data(), x.size());
    cpu::Linear(x.data(), t, kH, F(p + "conv_module.pointwise_conv2.weight", kH * kH), kH, nullptr,
                att.data());
    for (std::size_t i = 0; i < h.size(); ++i) {
      h[i] += att[i];
    }
    ffn("ffn2");
    cpu::LayerNorm(h.data(), h.data(), t, kH, F(p + "final_layer_norm.weight", kH),
                   F(p + "final_layer_norm.bias", kH), kEps);
    if (tap && (layer == 0 || layer == 8)) {
      tap("layer" + std::to_string(layer), h);
    }
    // ~96 MB of F32 weights per layer, read once per reference: keep only the
    // current layer resident (#829).
    st_->ReleaseResidentPages();
  }
  if (tap) {
    tap("hidden17", h);
  }
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t c = 0; c < kH; ++c) {
      float& val = h[static_cast<std::size_t>(r * kH + c)];
      val = (val - mean_[static_cast<std::size_t>(c)]) / std_[static_cast<std::size_t>(c)];
    }
  }
  return h;
}

}  // namespace gufo::models::indextts
