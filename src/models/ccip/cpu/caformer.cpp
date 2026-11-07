// CAFormer backbone + AttentionPool2d head of CCIP, CPU fp32.
// Graph semantics (read off model_feat.onnx, see reference/convert_weights.py):
//   stem: conv 7x7/4 pad 2 (+bias) -> LayerNorm(post)            [NHWC]
//   downsample i>0: LayerNorm(pre) -> conv 3x3/2 pad 1 (+bias)
//   SepConv block:  x += pw2(dw7x7(StarReLU(pw1(LN1(x)))));  x += fc2(StarReLU(fc1(LN2(x))))
//   Attention block: x = rs1*x + proj(MHSA(LN1(x)));  x = rs2*x + fc2(StarReLU(fc1(LN2(x))))
//     MHSA: qkv = x W (no bias) as (N, 3, heads, hd); softmax((q k^T) * scale) v
//   head: tokens (h*w, 512) -> [mean; tokens] + pos -> one query (token 0)
//         CLIP attention, 8 heads, q / 8 -> c_proj (512 -> 768).
// Written with AI assistance (Claude Code).
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"

#include "src/models/ccip/ccip.hpp"
#include "src/models/ccip/cpu/kernels.hpp"
#include "src/models/ccip/tensors.hpp"

namespace gufo::models::ccip {
namespace {

struct Conv {
  const float* w{nullptr};
  const float* b{nullptr};
  int in{0}, out{0}, kernel{0}, stride{1}, pad{0};
};

struct Block {
  bool attention{false};
  const float* norm1{nullptr};
  const float* norm2{nullptr};
  const float* res_scale1{nullptr};
  const float* res_scale2{nullptr};
  // sepconv
  const float* pw1{nullptr};  // [C, 2C]
  const float* dw{nullptr};   // [2C, 1, k, k]
  const float* pw2{nullptr};  // [2C, C]
  int mid{0}, dw_kernel{0}, dw_pad{0};
  float act1_scale{1}, act1_bias{0};
  // attention
  const float* qkv{nullptr};   // [C, 3C]
  const float* proj{nullptr};  // [C, C]
  int heads{0}, head_dim{0};
  float attn_scale{1};
  // mlp
  const float* fc1{nullptr};
  const float* fc2{nullptr};
  int hidden{0};
  float mlp_scale{1}, mlp_bias{0};
};

struct Stage {
  int dim{0};
  Conv down;
  const float* pre_norm{nullptr};
  const float* post_norm{nullptr};
  std::vector<Block> blocks;
};

class CpuCcip final : public CcipModel {
public:
  bool Init(const std::filesystem::path& dir, std::string* error) {
    if (!weights_.Load(dir / "ccip_caformer.safetensors", error)) return false;
    std::ifstream in(dir / "ccip_caformer.json", std::ios::binary);
    if (!in) {
      if (error) *error = "missing " + (dir / "ccip_caformer.json").string();
      return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try {
      Build(json::parse(ss.str()));
    } catch (const std::exception& e) {
      if (error) *error = std::string("ccip: ") + e.what();
      return false;
    }
    return true;
  }

  int OutputDim() const override { return out_dim_; }

  std::vector<float> Forward(const float* chw, Taps* taps) const override {
    using namespace cpu;
    const int size = input_size_;
    int h = 0, w = 0;
    std::vector<float> x;
    std::vector<float> tmp;
    for (std::size_t s = 0; s < stages_.size(); ++s) {
      const Stage& st = stages_[s];
      const Conv& cv = st.down;
      if (s == 0) {
        x.resize(static_cast<std::size_t>((size + 2 * cv.pad - cv.kernel) / cv.stride + 1) *
                 ((size + 2 * cv.pad - cv.kernel) / cv.stride + 1) * cv.out);
        ConvToNhwc(chw, true, size, size, cv.in, cv.w, cv.b, cv.out, cv.kernel, cv.stride, cv.pad,
                   x.data(), &h, &w);
        if (st.post_norm != nullptr) LayerNorm(x.data(), st.post_norm, x.data(), h * w, cv.out, eps_);
        if (taps) (*taps)["stem"] = x;
      } else {
        tmp.resize(x.size());
        LayerNorm(x.data(), st.pre_norm, tmp.data(), h * w, cv.in, eps_);
        const int oh = (h + 2 * cv.pad - cv.kernel) / cv.stride + 1;
        const int ow = (w + 2 * cv.pad - cv.kernel) / cv.stride + 1;
        x.assign(static_cast<std::size_t>(oh) * ow * cv.out, 0.0f);
        ConvToNhwc(tmp.data(), false, h, w, cv.in, cv.w, cv.b, cv.out, cv.kernel, cv.stride,
                   cv.pad, x.data(), &h, &w);
      }
      const int n = h * w;
      for (std::size_t b = 0; b < st.blocks.size(); ++b) {
        RunBlock(st.blocks[b], x, n, h, w, st.dim);
        if (taps && s == 2 && b == 0) (*taps)["stage2_block0"] = x;
      }
      if (taps) (*taps)["stage" + std::to_string(s)] = x;
    }
    return Head(x, h * w);
  }

private:
  void Build(const json::Value& arch) {
    eps_ = static_cast<float>(arch.member_double("norm_eps", 1e-6));
    const json::Value* input = arch.find("input");
    if (input) input_size_ = static_cast<int>(input->member_size("size", 384));
    const json::Value* down = arch.find("downsample");
    const json::Value* stages = arch.find("stages");
    if (!down || !stages || down->size() != stages->size()) throw std::runtime_error("bad arch json");
    for (std::size_t s = 0; s < stages->size(); ++s) {
      Stage st;
      const json::Value& sj = stages->items()[s];
      const json::Value& dj = down->items()[s];
      const json::Value& cj = *dj.find("conv");
      st.dim = static_cast<int>(sj.member_size("dim"));
      st.down.in = static_cast<int>(cj.member_size("in"));
      st.down.out = static_cast<int>(cj.member_size("out"));
      st.down.kernel = static_cast<int>(cj.member_size("kernel"));
      st.down.stride = static_cast<int>(cj.member_size("stride"));
      st.down.pad = static_cast<int>(cj.member_size("pad"));
      const std::string dp = "downsample." + std::to_string(s);
      st.down.w = W(dp + ".conv.weight",
                    static_cast<std::int64_t>(st.down.out) * st.down.in * st.down.kernel * st.down.kernel);
      st.down.b = W(dp + ".conv.bias", st.down.out);
      if (dj.find("pre_norm")->as_bool()) st.pre_norm = W(dp + ".pre_norm.weight", st.down.in);
      if (dj.find("post_norm")->as_bool()) st.post_norm = W(dp + ".post_norm.weight", st.down.out);
      if (st.down.out != st.dim) throw std::runtime_error("stage dim mismatch");
      const json::Value& blocks = *sj.find("blocks");
      const int c = st.dim;
      for (std::size_t b = 0; b < blocks.size(); ++b) {
        const json::Value& bj = blocks.items()[b];
        const std::string p = "stages." + std::to_string(s) + "." + std::to_string(b);
        Block blk;
        blk.norm1 = W(p + ".norm1.weight", c);
        blk.norm2 = W(p + ".norm2.weight", c);
        if (bj.find("res_scale1")->as_bool()) blk.res_scale1 = W(p + ".res_scale1", c);
        if (bj.find("res_scale2")->as_bool()) blk.res_scale2 = W(p + ".res_scale2", c);
        const std::string mixer = bj.member_str("mixer");
        if (mixer == "attention") {
          blk.attention = true;
          blk.heads = static_cast<int>(bj.member_size("heads"));
          blk.head_dim = static_cast<int>(bj.member_size("head_dim"));
          blk.attn_scale = static_cast<float>(bj.member_double("attn_scale"));
          const std::int64_t inner = static_cast<std::int64_t>(blk.heads) * blk.head_dim;
          blk.qkv = W(p + ".attn.qkv.weight", c * 3 * inner);
          blk.proj = W(p + ".attn.proj.weight", inner * c);
          if (inner != c) throw std::runtime_error("attention inner dim != dim");
        } else if (mixer == "sepconv") {
          const auto* t = weights_.Find(p + ".sepconv.pwconv1.weight");
          if (!t || t->shape.size() != 2) throw std::runtime_error("missing pwconv1 " + p);
          blk.mid = static_cast<int>(t->shape[1]);
          blk.dw_kernel = static_cast<int>(bj.member_size("dw_kernel"));
          blk.dw_pad = static_cast<int>(bj.member_size("dw_pad"));
          blk.pw1 = W(p + ".sepconv.pwconv1.weight", static_cast<std::int64_t>(c) * blk.mid);
          blk.dw = W(p + ".sepconv.dwconv.weight",
                     static_cast<std::int64_t>(blk.mid) * blk.dw_kernel * blk.dw_kernel);
          blk.pw2 = W(p + ".sepconv.pwconv2.weight", static_cast<std::int64_t>(blk.mid) * c);
          const json::Value& a1 = *bj.find("act1");
          blk.act1_scale = static_cast<float>(a1.member_double("scale"));
          blk.act1_bias = static_cast<float>(a1.member_double("bias"));
        } else {
          throw std::runtime_error("unknown mixer " + mixer);
        }
        blk.hidden = static_cast<int>(bj.member_size("mlp_hidden"));
        blk.fc1 = W(p + ".mlp.fc1.weight", static_cast<std::int64_t>(c) * blk.hidden);
        blk.fc2 = W(p + ".mlp.fc2.weight", static_cast<std::int64_t>(blk.hidden) * c);
        const json::Value& ma = *bj.find("mlp_act");
        blk.mlp_scale = static_cast<float>(ma.member_double("scale"));
        blk.mlp_bias = static_cast<float>(ma.member_double("bias"));
        st.blocks.push_back(blk);
      }
      stages_.push_back(std::move(st));
    }
    const json::Value& ap = *arch.find("attnpool");
    pool_tokens_ = static_cast<int>(ap.member_size("tokens"));
    pool_dim_ = static_cast<int>(ap.member_size("dim"));
    pool_heads_ = static_cast<int>(ap.member_size("heads"));
    pool_q_div_ = static_cast<float>(ap.member_double("q_divisor"));
    out_dim_ = static_cast<int>(ap.member_size("out_dim"));
    const std::int64_t d = pool_dim_;
    pos_ = W("attnpool.positional_embedding", pool_tokens_ * d);
    for (int i = 0; i < 3; ++i) {
      const std::string nm = std::string(1, "qkv"[i]);
      pw_[i] = W("attnpool." + nm + "_proj.weight", d * d);
      pb_[i] = W("attnpool." + nm + "_proj.bias", d);
    }
    cw_ = W("attnpool.c_proj.weight", static_cast<std::int64_t>(out_dim_) * d);
    cb_ = W("attnpool.c_proj.bias", out_dim_);
    if (stages_.back().dim != pool_dim_) throw std::runtime_error("attnpool dim mismatch");
  }

  const float* W(const std::string& name, std::int64_t numel) const {
    return weights_.F32(name, numel).data();
  }

  void RunBlock(const Block& blk, std::vector<float>& x, int n, int h, int w, int c) const {
    using namespace cpu;
    const std::size_t nc = static_cast<std::size_t>(n) * c;
    std::vector<float> y(nc);
    LayerNorm(x.data(), blk.norm1, y.data(), n, c, eps_);
    std::vector<float> mixed(nc);
    if (!blk.attention) {
      std::vector<float> a(static_cast<std::size_t>(n) * blk.mid);
      Gemm(y.data(), blk.pw1, nullptr, a.data(), n, blk.mid, c);
      StarRelu(a.data(), a.size(), blk.act1_scale, blk.act1_bias);
      std::vector<float> d(a.size());
      DepthwiseConvNhwc(a.data(), blk.dw, d.data(), h, w, blk.mid, blk.dw_kernel, blk.dw_pad);
      Gemm(d.data(), blk.pw2, nullptr, mixed.data(), n, c, blk.mid);
    } else {
      Attention(blk, y.data(), n, c, mixed.data());
    }
    Residual(x, blk.res_scale1, mixed, n, c);
    LayerNorm(x.data(), blk.norm2, y.data(), n, c, eps_);
    std::vector<float> hdn(static_cast<std::size_t>(n) * blk.hidden);
    Gemm(y.data(), blk.fc1, nullptr, hdn.data(), n, blk.hidden, c);
    StarRelu(hdn.data(), hdn.size(), blk.mlp_scale, blk.mlp_bias);
    Gemm(hdn.data(), blk.fc2, nullptr, mixed.data(), n, c, blk.hidden);
    Residual(x, blk.res_scale2, mixed, n, c);
  }

  static void Residual(std::vector<float>& x, const float* scale, const std::vector<float>& add,
                       int n, int c) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      float* xr = x.data() + static_cast<std::size_t>(i) * c;
      const float* ar = add.data() + static_cast<std::size_t>(i) * c;
      if (scale != nullptr) {
        for (int j = 0; j < c; ++j) xr[j] = xr[j] * scale[j] + ar[j];
      } else {
        for (int j = 0; j < c; ++j) xr[j] = xr[j] + ar[j];
      }
    }
  }

  void Attention(const Block& blk, const float* x, int n, int c, float* out) const {
    using namespace cpu;
    const int c3 = 3 * c;
    std::vector<float> qkv(static_cast<std::size_t>(n) * c3);
    cpu::Gemm(x, blk.qkv, nullptr, qkv.data(), n, c3, c);
    std::vector<float> ctx(static_cast<std::size_t>(n) * c);
    const int hd = blk.head_dim;
    for (int hh = 0; hh < blk.heads; ++hh) {
      // Gather per-head q, k, v (n x hd) for contiguous dot products.
      std::vector<float> q(static_cast<std::size_t>(n) * hd), k(q.size()), v(q.size());
      for (int i = 0; i < n; ++i) {
        const float* row = qkv.data() + static_cast<std::size_t>(i) * c3;
        std::memcpy(&q[static_cast<std::size_t>(i) * hd], row + hh * hd, sizeof(float) * hd);
        std::memcpy(&k[static_cast<std::size_t>(i) * hd], row + c + hh * hd, sizeof(float) * hd);
        std::memcpy(&v[static_cast<std::size_t>(i) * hd], row + 2 * c + hh * hd, sizeof(float) * hd);
      }
      std::vector<float> scores(static_cast<std::size_t>(n) * n);
      GemmBT(q.data(), k.data(), nullptr, scores.data(), n, n, hd);
      const float sc = blk.attn_scale;
#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        float* r = scores.data() + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) r[j] *= sc;
        SoftmaxRows(r, 1, n);
      }
      std::vector<float> o(static_cast<std::size_t>(n) * hd);
      cpu::Gemm(scores.data(), v.data(), nullptr, o.data(), n, hd, n);
      for (int i = 0; i < n; ++i) {
        std::memcpy(ctx.data() + static_cast<std::size_t>(i) * c + hh * hd,
                    &o[static_cast<std::size_t>(i) * hd], sizeof(float) * hd);
      }
    }
    cpu::Gemm(ctx.data(), blk.proj, nullptr, out, n, c, c);
  }

  std::vector<float> Head(const std::vector<float>& x, int n) const {
    using namespace cpu;
    const int d = pool_dim_;
    const int t = n + 1;
    if (t != pool_tokens_) throw std::runtime_error("attnpool token count mismatch");
    std::vector<float> tok(static_cast<std::size_t>(t) * d);
    for (int j = 0; j < d; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += x[static_cast<std::size_t>(i) * d + j];
      tok[j] = static_cast<float>(s / n);
    }
    std::memcpy(tok.data() + d, x.data(), sizeof(float) * n * d);
    for (std::size_t i = 0; i < tok.size(); ++i) tok[i] += pos_[i];
    std::vector<float> q(d), k(static_cast<std::size_t>(t) * d), v(k.size());
    cpu::Gemm(tok.data(), pw_[0], pb_[0], q.data(), 1, d, d);
    cpu::Gemm(tok.data(), pw_[1], pb_[1], k.data(), t, d, d);
    cpu::Gemm(tok.data(), pw_[2], pb_[2], v.data(), t, d, d);
    const int hd = d / pool_heads_;
    std::vector<float> ctx(d);
    std::vector<float> sc(t);
    for (int hh = 0; hh < pool_heads_; ++hh) {
      for (int j = 0; j < t; ++j) {
        float s = 0.0f;
        for (int e = 0; e < hd; ++e) {
          s += (q[hh * hd + e] / pool_q_div_) * k[static_cast<std::size_t>(j) * d + hh * hd + e];
        }
        sc[j] = s;
      }
      SoftmaxRows(sc.data(), 1, t);
      for (int e = 0; e < hd; ++e) {
        float s = 0.0f;
        for (int j = 0; j < t; ++j) s += sc[j] * v[static_cast<std::size_t>(j) * d + hh * hd + e];
        ctx[hh * hd + e] = s;
      }
    }
    std::vector<float> out(out_dim_);
    GemmBT(ctx.data(), cw_, cb_, out.data(), 1, out_dim_, d);
    return out;
  }

  TensorFile weights_;
  std::vector<Stage> stages_;
  float eps_{1e-6f};
  int input_size_{384};
  int pool_tokens_{0}, pool_dim_{0}, pool_heads_{0}, out_dim_{0};
  float pool_q_div_{1};
  const float* pos_{nullptr};
  const float* pw_[3]{};
  const float* pb_[3]{};
  const float* cw_{nullptr};
  const float* cb_{nullptr};
};

}  // namespace

std::unique_ptr<CcipModel> CcipModel::Load(const std::filesystem::path& dir, std::string* error) {
  auto m = std::make_unique<CpuCcip>();
  if (!m->Init(dir, error)) return nullptr;
  return m;
}

std::vector<float> CcipModel::Embed(const RgbImage& image) const {
  const std::vector<float> x = CcipInput(image);
  std::vector<float> f = Forward(x.data());
  double s = 0.0;
  for (const float v : f) s += static_cast<double>(v) * v;
  const float inv = s > 0 ? static_cast<float>(1.0 / std::sqrt(s)) : 0.0f;
  for (float& v : f) v *= inv;
  return f;
}

double CcipDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  const double cos = (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : 0.0;
  const double d = (1.0 - cos) / 2.0;
  return d < 0 ? 0 : (d > 1 ? 1 : d);
}

}  // namespace gufo::models::ccip
