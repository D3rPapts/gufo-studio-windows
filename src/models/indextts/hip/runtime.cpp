// Written with AI assistance (Claude Code).
#include "src/models/indextts/hip/runtime.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "src/models/fish_drama3/hip/codec_ops.hpp"
#include "src/models/indextts/hip/blas.hpp"
#include "src/models/indextts/hip/ops.hpp"
#include "src/models/ming_tts/hip/ops.hpp"
#include "src/models/voxcpm2/hip/ops.hpp"
#endif

namespace gufo::models::indextts::hip {

#if !defined(ENGINE_ENABLE_HIP)

struct Runtime::Impl {};
Runtime::~Runtime() = default;
std::unique_ptr<Runtime> Runtime::Create(const UnifiedVoice&, const S2Mel&, const BigVgan&,
                                         std::string* error) {
  if (error != nullptr) {
    *error = "built without HIP";
  }
  return nullptr;
}
void Runtime::GptPrefill(const std::vector<float>&, std::int64_t, std::int64_t, std::vector<float>*) {}
void Runtime::GptStep(std::int32_t, std::vector<float>*) {}
std::vector<float> Runtime::CfmSample(const std::vector<float>&, const float*, std::int64_t, const float*,
                                      std::int64_t, const float*, const CfmOptions&) {
  return {};
}
std::vector<float> Runtime::Velocity(const float*, const float*, float, const float*, const float*,
                                     std::int64_t) {
  return {};
}
std::vector<float> Runtime::Vocode(const float*, std::int64_t) { return {}; }
void Runtime::SetFastDit(bool) {}
std::string Runtime::LastCfmProfile() const { return {}; }

#else

namespace {

namespace ming = ming_tts::hip;
namespace vox = voxcpm2::hip;
namespace fish = fish_drama3::hip;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(status));
  }
}

std::uint16_t ToBf16(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if ((bits & 0x7F800000U) == 0x7F800000U) {
    return static_cast<std::uint16_t>(bits >> 16U);
  }
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

// Owns every device allocation.
struct Arena {
  std::vector<void*> blocks;
  ~Arena() {
    for (void* p : blocks) {
      (void)hipFree(p);
    }
  }
  template <typename T>
  T* Alloc(std::size_t count) {
    void* p = nullptr;
    Check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "alloc");
    blocks.push_back(p);
    return static_cast<T*>(p);
  }
  float* F32(const float* host, std::size_t count) {
    float* d = Alloc<float>(count);
    Check(hipMemcpy(d, host, count * sizeof(float), hipMemcpyHostToDevice), "upload f32");
    return d;
  }
  float* F32(const std::vector<float>& host) { return F32(host.data(), host.size()); }
  // BF16 copy of a row-major [rows, cols] matrix (optionally given as [cols, rows] and transposed).
  std::uint16_t* Bf16(const float* host, std::size_t rows, std::size_t cols, bool transpose = false) {
    std::vector<std::uint16_t> tmp(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < cols; ++c) {
        tmp[r * cols + c] = ToBf16(transpose ? host[c * rows + r] : host[r * cols + c]);
      }
    }
    std::uint16_t* d = Alloc<std::uint16_t>(tmp.size());
    Check(hipMemcpy(d, tmp.data(), tmp.size() * 2, hipMemcpyHostToDevice), "upload bf16");
    return d;
  }
};

// Grow-only device scratch buffer.
struct Buf {
  float* p{nullptr};
  std::size_t cap{0};
  float* Ensure(std::size_t n) {
    if (n > cap) {
      if (p != nullptr) {
        (void)hipFree(p);
      }
      cap = std::max<std::size_t>(n, 256);
      Check(hipMalloc(reinterpret_cast<void**>(&p), cap * sizeof(float)), "scratch");
    }
    return p;
  }
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  ~Buf() {
    if (p != nullptr) {
      (void)hipFree(p);
    }
  }
};

struct Lin {  // y = x W^T (+ b), W [out, in] BF16
  const std::uint16_t* w{nullptr};
  const float* b{nullptr};
  std::int64_t out{0}, in{0};
};

struct Conv {  // im2col conv, W [cout, k * cin]
  Lin lin;
  std::int64_t cin{0}, k{1}, dil{1}, pad_left{0}, pad_right{0};
  PadMode mode{PadMode::kZero};
};

struct ActDev {
  const float* alpha{nullptr};
  const float* inv_beta{nullptr};
  const float* up{nullptr};
  const float* down{nullptr};
};

// Optional per-category GPU timing of the CFM loop (GUFO_INDEXTTS_PROFILE=1):
// an event is recorded at every category boundary on the single stream; the
// time between two marks is charged to the category of the first one.
struct Prof {
  enum Cat { kGemm = 0, kAttention, kConv, kOther, kCount };
  bool on{false};
  std::vector<hipEvent_t> pool;
  std::vector<int> cats;
  std::size_t used{0};
  double ms[kCount]{};
  ~Prof() {
    for (hipEvent_t e : pool) {
      (void)hipEventDestroy(e);
    }
  }
  void Reset() {
    for (double& m : ms) {
      m = 0.0;
    }
    used = 0;
    cats.clear();
  }
  void Mark(Cat cat, hipStream_t stream) {
    if (!on) {
      return;
    }
    if (used == pool.size()) {
      hipEvent_t e = nullptr;
      Check(hipEventCreate(&e), "profile event");
      pool.push_back(e);
    }
    Check(hipEventRecord(pool[used], stream), "profile record");
    cats.push_back(cat);
    ++used;
  }
  void Flush() {
    if (!on) {
      return;
    }
    if (used >= 2) {
      Check(hipEventSynchronize(pool[used - 1]), "profile sync");
      for (std::size_t i = 0; i + 1 < used; ++i) {
        float m = 0.0F;
        (void)hipEventElapsedTime(&m, pool[i], pool[i + 1]);
        ms[cats[i]] += m;
      }
    }
    used = 0;
    cats.clear();
  }
};

}  // namespace

struct Runtime::Impl {
  hipStream_t stream{nullptr};
  Arena arena;
  // ---- GPT
  struct GptLayer {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    Lin attn, proj, fc, mproj;
  };
  std::vector<GptLayer> gpt;
  const float *lnf_w{nullptr}, *lnf_b{nullptr}, *fn_w{nullptr}, *fn_b{nullptr};
  Lin mel_head;
  const UnifiedVoice* voice{nullptr};
  std::vector<float*> kcache, vcache;
  std::int64_t cap{0}, len{0}, mel_tokens{0};
  Buf x, y, qkv, q, att, mlp, partial, logits;
  // ---- DiT
  const S2Mel* s2mel{nullptr};
  struct DitLayer {
    const float* attn_norm;
    const float* ffn_norm;
    Lin wqkv, wo, w1, w3, w2, skip_in;
  };
  std::vector<DitLayer> dit;
  const float* final_norm_w{nullptr};
  Lin merge_x, merge_rest, cond_proj, skip_linear, conv1, res_projection, final_linear, conv2;
  Conv wn_in[8];
  Lin wn_res[8], wn_skip[8];
  float* inv_freq{nullptr};
  Buf h, h2, xin, dq, dk, dv, datt, f1, f3, col, c_cond, c_uncond, wx, out, rs, sk, acts, v_c, v_u,
      dx, dprompt, dcond, ada, wng, fmod, tmp, xres, xb, scores;
  std::vector<std::unique_ptr<Buf>> skips;
  // Fast DiT path (default): CFG branches as one batch of 2, hipBLASLt BF16
  // projections, FP32 GEMM attention. GUFO_INDEXTTS_DIT_FAST=0 selects the
  // first port's kernels (tiled FP32xBF16 GEMM, per-row attention) for A/B.
  bool fast{true};
  std::unique_ptr<Blas> blas;
  Prof prof;
  std::string last_profile;
  // time-only conditioning of the current step (device) and the cache of the
  // whole Euler time grid (the same for every request with the same step count)
  const float *cur_ada{nullptr}, *cur_wng{nullptr}, *cur_fmod{nullptr};
  std::vector<float> grid_times;
  Buf grid_ada, grid_wng, grid_fmod;
  std::size_t ada_n{0}, wng_n{0}, fmod_n{0};
  // ---- BigVGAN
  const BigVgan* voc{nullptr};
  Conv pre, post;
  struct UpDev {
    Lin all;  // [k * cout, cin]
    std::int64_t cin, cout, k, stride, pad;
  };
  std::vector<UpDev> ups;
  struct BlockDev {
    Conv c1[3], c2[3];
    ActDev acts[6];
  };
  std::vector<BlockDev> blocks;
  ActDev post_act;
  Buf va, vb, vc, vacc, vh, vt;

  // y = x W^T + b for any row count (FP32 activations, BF16 weights)
  void Linear(const float* xin_, std::int64_t rows, const Lin& l, float* yout) {
    if (rows <= 32) {
      float* part = partial.Ensure(vox::SkinnyPartialFloats(static_cast<std::size_t>(l.out),
                                                            static_cast<std::size_t>(l.in)) +
                                   64);
      vox::LaunchSkinnyGemm(xin_, l.w, l.b, yout, static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(l.out), static_cast<std::size_t>(l.in), part, stream);
      return;
    }
    ming::LaunchGemmF32Bf16(xin_, l.w, yout, static_cast<std::size_t>(rows), static_cast<std::size_t>(l.out),
                            static_cast<std::size_t>(l.in), stream);
    if (l.b != nullptr) {
      ming::LaunchAddBias(yout, l.b, static_cast<std::size_t>(rows), static_cast<std::size_t>(l.out), stream);
    }
  }
  // y += x W^T + b
  void LinearAdd(const float* xin_, std::int64_t rows, const Lin& l, float* yout) {
    if (rows <= 32) {
      float* part = partial.Ensure(vox::SkinnyPartialFloats(static_cast<std::size_t>(l.out),
                                                            static_cast<std::size_t>(l.in)) +
                                   64);
      vox::LaunchSkinnyGemm(xin_, l.w, l.b, yout, static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(l.out), static_cast<std::size_t>(l.in), part, stream,
                            vox::SkinnyEpilogue::kAccumulate);
      return;
    }
    float* t = tmp.Ensure(static_cast<std::size_t>(rows * l.out));
    Linear(xin_, rows, l, t);
    ming::LaunchAdd(yout, t, static_cast<std::size_t>(rows * l.out), stream);
  }
  void RunConv(const Conv& c, const float* xin_, std::int64_t t, float* yout) {
    const std::int64_t t_out = t + c.pad_left + c.pad_right - c.dil * (c.k - 1);
    float* colp = col.Ensure(static_cast<std::size_t>(t_out * c.k * c.cin));
    LaunchIm2colTM(xin_, colp, static_cast<std::size_t>(t), static_cast<std::size_t>(t_out),
                   static_cast<std::size_t>(c.cin), static_cast<std::size_t>(c.k),
                   static_cast<std::size_t>(c.dil), c.pad_left, c.mode, stream);
    Linear(colp, t_out, c.lin, yout);
  }
  Lin MakeLin(const float* w, const float* b, std::int64_t out_, std::int64_t in_, bool transposed = false) {
    Lin l;
    l.w = arena.Bf16(w, static_cast<std::size_t>(out_), static_cast<std::size_t>(in_), transposed);
    l.b = b != nullptr ? arena.F32(b, static_cast<std::size_t>(out_)) : nullptr;
    l.out = out_;
    l.in = in_;
    return l;
  }
  Conv MakeConv(const cpu::Conv1dTM& c, PadMode mode) {
    // taps [k][cout][cin] -> W [cout, k * cin]
    std::vector<float> w(static_cast<std::size_t>(c.cout * c.kernel * c.cin));
    for (std::int64_t j = 0; j < c.kernel; ++j) {
      for (std::int64_t o = 0; o < c.cout; ++o) {
        std::memcpy(w.data() + (o * c.kernel + j) * c.cin, c.taps.data() + (j * c.cout + o) * c.cin,
                    static_cast<std::size_t>(c.cin) * 4);
      }
    }
    Conv out;
    out.lin = MakeLin(w.data(), c.bias.empty() ? nullptr : c.bias.data(), c.cout, c.kernel * c.cin);
    out.cin = c.cin;
    out.k = c.kernel;
    out.dil = c.dilation;
    out.pad_left = c.pad_left;
    out.pad_right = c.pad_right;
    out.mode = mode;
    return out;
  }
  ActDev MakeAct(const BigVgan::Act& a) {
    ActDev d;
    d.alpha = arena.F32(a.alpha);
    d.inv_beta = arena.F32(a.inv_beta);
    d.up = arena.F32(a.up, 12);
    d.down = arena.F32(a.down, 12);
    return d;
  }

  void LoadGpt(const UnifiedVoice& g) {
    voice = &g;
    const PthArchive& sd = g.archive();
    const GptConfig& c = g.config();
    const std::int64_t d = c.dim;
    auto W = [&](const std::string& n) { return sd.F32("model." + n); };
    for (std::int64_t l = 0; l < c.layers; ++l) {
      const std::string p = "gpt.h." + std::to_string(l) + ".";
      GptLayer L{};
      L.ln1_w = arena.F32(W(p + "ln_1.weight"), static_cast<std::size_t>(d));
      L.ln1_b = arena.F32(W(p + "ln_1.bias"), static_cast<std::size_t>(d));
      L.ln2_w = arena.F32(W(p + "ln_2.weight"), static_cast<std::size_t>(d));
      L.ln2_b = arena.F32(W(p + "ln_2.bias"), static_cast<std::size_t>(d));
      // GPT-2 Conv1D weights are [in, out]: transpose to [out, in]
      L.attn = MakeLin(W(p + "attn.c_attn.weight"), W(p + "attn.c_attn.bias"), 3 * d, d, true);
      L.proj = MakeLin(W(p + "attn.c_proj.weight"), W(p + "attn.c_proj.bias"), d, d, true);
      L.fc = MakeLin(W(p + "mlp.c_fc.weight"), W(p + "mlp.c_fc.bias"), 4 * d, d, true);
      L.mproj = MakeLin(W(p + "mlp.c_proj.weight"), W(p + "mlp.c_proj.bias"), d, 4 * d, true);
      gpt.push_back(L);
    }
    lnf_w = arena.F32(W("gpt.ln_f.weight"), static_cast<std::size_t>(d));
    lnf_b = arena.F32(W("gpt.ln_f.bias"), static_cast<std::size_t>(d));
    fn_w = arena.F32(W("final_norm.weight"), static_cast<std::size_t>(d));
    fn_b = arena.F32(W("final_norm.bias"), static_cast<std::size_t>(d));
    mel_head = MakeLin(W("mel_head.weight"), W("mel_head.bias"), c.mel_codes, d);
  }

  void LoadDit(const S2Mel& s) {
    s2mel = &s;
    constexpr std::int64_t D = 512;
    for (int l = 0; l < 13; ++l) {
      const std::string p = "transformer.layers." + std::to_string(l) + ".";
      DitLayer L{};
      L.attn_norm = arena.F32(s.W(p + "attention_norm.norm.weight", D), D);
      L.ffn_norm = arena.F32(s.W(p + "ffn_norm.norm.weight", D), D);
      L.wqkv = MakeLin(s.W(p + "attention.wqkv.weight"), nullptr, 3 * D, D);
      L.wo = MakeLin(s.W(p + "attention.wo.weight"), nullptr, D, D);
      L.w1 = MakeLin(s.W(p + "feed_forward.w1.weight"), nullptr, 1536, D);
      L.w3 = MakeLin(s.W(p + "feed_forward.w3.weight"), nullptr, 1536, D);
      L.w2 = MakeLin(s.W(p + "feed_forward.w2.weight"), nullptr, D, 1536);
      L.skip_in = MakeLin(s.W(p + "skip_in_linear.weight"), s.W(p + "skip_in_linear.bias"), D, 2 * D);
      dit.push_back(L);
    }
    final_norm_w = arena.F32(s.W("transformer.norm.norm.weight", D), D);
    // cond_x_merge_linear [512, 864] = [x 80 | prompt 80 | cond 512 | style 192]
    const float* merge = s.W("cond_x_merge_linear.weight", D * 864);
    std::vector<float> wx(static_cast<std::size_t>(D * 80));
    std::vector<float> wr(static_cast<std::size_t>(D * 784));
    for (std::int64_t o = 0; o < D; ++o) {
      std::memcpy(wx.data() + o * 80, merge + o * 864, 80 * 4);
      std::memcpy(wr.data() + o * 784, merge + o * 864 + 80, 784 * 4);
    }
    merge_x = MakeLin(wx.data(), nullptr, D, 80);
    merge_rest = MakeLin(wr.data(), s.W("cond_x_merge_linear.bias", D), D, 784);
    cond_proj = MakeLin(s.W("cond_projection.weight"), s.W("cond_projection.bias", D), D, D);
    skip_linear = MakeLin(s.W("skip_linear.weight"), s.W("skip_linear.bias", D), D, D + 80);
    conv1 = MakeLin(s.W("conv1.weight"), s.W("conv1.bias", D), D, D);
    res_projection = MakeLin(s.W("res_projection.weight"), s.W("res_projection.bias", D), D, D);
    final_linear = MakeLin(s.final_linear().data(), s.W("final_layer.linear.bias", D), D, D);
    conv2 = MakeLin(s.W("conv2.weight"), s.W("conv2.bias", 80), 80, D);
    for (int i = 0; i < 8; ++i) {
      wn_in[i] = MakeConv(s.wavenet_in(i), PadMode::kReflect);
      const std::vector<float>& rw = s.wavenet_res(i);
      const float* rb = s.W("wavenet.res_skip_layers." + std::to_string(i) + ".conv.conv.bias");
      if (i < 7) {
        wn_res[i] = MakeLin(rw.data(), rb, D, D);
        wn_skip[i] = MakeLin(rw.data() + D * D, rb + D, D, D);
      } else {
        wn_skip[i] = MakeLin(rw.data(), rb, D, D);
      }
    }
    std::vector<float> f(32);
    for (int i = 0; i < 32; ++i) {
      f[static_cast<std::size_t>(i)] = 1.0F / std::pow(10000.0F, static_cast<float>(2 * i) / 64.0F);
    }
    inv_freq = arena.F32(f);
  }

  void LoadVocoder(const BigVgan& v) {
    voc = &v;
    pre = MakeConv(v.pre(), PadMode::kZero);
    post = MakeConv(v.post(), PadMode::kZero);
    for (const cpu::ConvTranspose1dTM& u : v.ups()) {
      UpDev d;
      d.all = MakeLin(u.taps.data(), nullptr, u.kernel * u.cout, u.cin);
      d.cin = u.cin;
      d.cout = u.cout;
      d.k = u.kernel;
      d.stride = u.stride;
      d.pad = u.padding;
      d.all.b = u.bias.empty() ? nullptr : arena.F32(u.bias);  // used by the gather, not the GEMM
      ups.push_back(d);
    }
    for (const BigVgan::ResBlock& rb : v.blocks()) {
      BlockDev b;
      for (int i = 0; i < 3; ++i) {
        b.c1[i] = MakeConv(rb.c1[static_cast<std::size_t>(i)], PadMode::kZero);
        b.c2[i] = MakeConv(rb.c2[static_cast<std::size_t>(i)], PadMode::kZero);
      }
      for (int i = 0; i < 6; ++i) {
        b.acts[i] = MakeAct(rb.acts[static_cast<std::size_t>(i)]);
      }
      blocks.push_back(b);
    }
    post_act = MakeAct(v.post_act());
  }

  void GptForward(std::int64_t rows, std::vector<float>* out_logits) {
    const std::int64_t d = voice->config().dim;
    const std::int64_t heads = voice->config().heads;
    float* xp = x.p;
    float* yp = y.Ensure(static_cast<std::size_t>(rows * d));
    float* qkvp = qkv.Ensure(static_cast<std::size_t>(rows * 3 * d));
    float* qp = q.Ensure(static_cast<std::size_t>(rows * d));
    float* ap = att.Ensure(static_cast<std::size_t>(rows * d));
    float* mp = mlp.Ensure(static_cast<std::size_t>(rows * 4 * d));
    for (std::size_t l = 0; l < gpt.size(); ++l) {
      const GptLayer& L = gpt[l];
      fish::LaunchCodecLayerNorm(xp, L.ln1_w, L.ln1_b, yp, static_cast<std::size_t>(rows),
                                 static_cast<std::size_t>(d), 1e-5F, stream);
      Linear(yp, rows, L.attn, qkvp);
      LaunchSplitQkv(qkvp, qp, kcache[l] + len * d, vcache[l] + len * d, static_cast<std::size_t>(rows),
                     static_cast<std::size_t>(d), stream);
      vox::LaunchAttention(qp, kcache[l], vcache[l], ap, 1, static_cast<std::size_t>(rows),
                           static_cast<std::size_t>(len + rows), static_cast<std::size_t>(heads),
                           static_cast<std::size_t>(heads), static_cast<std::size_t>(d / heads),
                           static_cast<std::size_t>(rows * d), static_cast<std::size_t>(cap * d), true,
                           static_cast<std::size_t>(len), stream);
      LinearAdd(ap, rows, L.proj, xp);
      fish::LaunchCodecLayerNorm(xp, L.ln2_w, L.ln2_b, yp, static_cast<std::size_t>(rows),
                                 static_cast<std::size_t>(d), 1e-5F, stream);
      Linear(yp, rows, L.fc, mp);
      ming::LaunchGeluTanh(mp, static_cast<std::size_t>(rows * 4 * d), stream);
      LinearAdd(mp, rows, L.mproj, xp);
    }
    len += rows;
    const float* last = xp + (rows - 1) * d;
    float* a = yp;
    float* b = qp;
    fish::LaunchCodecLayerNorm(last, lnf_w, lnf_b, a, 1, static_cast<std::size_t>(d), 1e-5F, stream);
    fish::LaunchCodecLayerNorm(a, fn_w, fn_b, b, 1, static_cast<std::size_t>(d), 1e-5F, stream);
    const std::int64_t codes = voice->config().mel_codes;
    float* lp = logits.Ensure(static_cast<std::size_t>(codes));
    Linear(b, 1, mel_head, lp);
    out_logits->resize(static_cast<std::size_t>(codes));
    Check(hipMemcpyAsync(out_logits->data(), lp, static_cast<std::size_t>(codes) * 4, hipMemcpyDeviceToHost,
                         stream),
          "logits");
    Check(hipStreamSynchronize(stream), "gpt sync");
  }

  // ---- DiT building blocks: fast path (hipBLASLt / GEMM attention) or the first port's kernels
  // y = x W^T + b (+ y when `accumulate`). Fast path: y must hold Blas::PaddedRows(rows) rows.
  void DLinear(const float* x_in, std::int64_t rows, const Lin& l, float* y_out, bool accumulate) {
    prof.Mark(Prof::kGemm, stream);
    if (!fast) {
      if (accumulate) {
        LinearAdd(x_in, rows, l, y_out);
      } else {
        Linear(x_in, rows, l, y_out);
      }
      prof.Mark(Prof::kOther, stream);
      return;
    }
    const auto r = static_cast<std::size_t>(rows);
    const std::size_t padded = Blas::PaddedRows(r);
    auto* x16 = reinterpret_cast<std::uint16_t*>(xb.Ensure((padded * static_cast<std::size_t>(l.in) + 1) / 2));
    LaunchF32ToBf16(x_in, x16, r * static_cast<std::size_t>(l.in), stream);
    blas->Gemm(l.w, x16, y_out, r, static_cast<std::size_t>(l.out), static_cast<std::size_t>(l.in), accumulate);
    if (l.b != nullptr) {
      ming::LaunchAddBias(y_out, l.b, r, static_cast<std::size_t>(l.out), stream);
    }
    prof.Mark(Prof::kOther, stream);
  }
  // Non-causal 8-head attention of `nb` independent sequences of t rows ([nb * t, 512]).
  void DAttention(const float* qp, const float* kp, const float* vp, float* ap, std::size_t t, std::size_t nb) {
    constexpr std::size_t D = 512;
    prof.Mark(Prof::kAttention, stream);
    if (!fast) {
      vox::LaunchAttention(qp, kp, vp, ap, nb, t, t, 8, 8, 64, t * D, t * D, false, 0, stream);
    } else {
      float* s = scores.Ensure(8 * t * t);
      for (std::size_t b = 0; b < nb; ++b) {
        blas->Attention64(qp + b * t * D, kp + b * t * D, vp + b * t * D, ap + b * t * D, s, t, 8);
      }
    }
    prof.Mark(Prof::kOther, stream);
  }
  // Time-major conv of `nb` sequences of t rows (im2col per sequence, one GEMM).
  void DConv(const Conv& c, const float* x_in, std::int64_t t, std::int64_t nb, float* y_out) {
    const std::int64_t t_out = t + c.pad_left + c.pad_right - c.dil * (c.k - 1);
    const auto per = static_cast<std::size_t>(t_out * c.k * c.cin);
    float* colp = col.Ensure(per * static_cast<std::size_t>(nb));
    prof.Mark(Prof::kConv, stream);
    for (std::int64_t b = 0; b < nb; ++b) {
      LaunchIm2colTM(x_in + b * t * c.cin, colp + static_cast<std::size_t>(b) * per, static_cast<std::size_t>(t),
                     static_cast<std::size_t>(t_out), static_cast<std::size_t>(c.cin),
                     static_cast<std::size_t>(c.k), static_cast<std::size_t>(c.dil), c.pad_left, c.mode, stream);
    }
    DLinear(colp, t_out * nb, c.lin, y_out, false);
    prof.Mark(Prof::kOther, stream);
  }

  // One DiT estimator evaluation of `nb` sequences (CFG: nb = 2, conditional then unconditional) on
  // device buffers: x/v [nb * t, 80], c = merge(prompt, cond, style) [nb * t, 512]. Time-only terms from
  // cur_ada / cur_wng / cur_fmod.
  void Estimator(const float* xd, const float* cd, std::int64_t t, std::int64_t nb, float* vout) {
    constexpr std::int64_t D = 512;
    const auto tt = static_cast<std::size_t>(t);
    const auto rows = static_cast<std::size_t>(t * nb);
    const std::int64_t R = t * nb;
    const std::size_t pr = Blas::PaddedRows(rows);  // GEMM outputs hold padded rows
    float* hp = h.Ensure(pr * D);
    float* yp = h2.Ensure(pr * D);
    prof.Mark(Prof::kOther, stream);
    DLinear(xd, R, merge_x, hp, false);
    ming::LaunchAdd(hp, cd, rows * D, stream);
    float* qkvp = xin.Ensure(pr * 3 * D);
    float* qp = dq.Ensure(pr * D);
    float* kp = dk.Ensure(pr * D);
    float* vp = dv.Ensure(pr * D);
    float* ap = datt.Ensure(pr * D);
    float* f1p = f1.Ensure(pr * 1536);
    float* f3p = f3.Ensure(pr * 1536);
    while (skips.size() < 6) {
      skips.push_back(std::make_unique<Buf>());
    }
    std::size_t skip_top = 0;
    for (int l = 0; l < 13; ++l) {
      const DitLayer& L = dit[static_cast<std::size_t>(l)];
      if (l > 6) {
        float* cat = col.Ensure(rows * 2 * D);
        LaunchConcatCols(hp, skips[--skip_top]->p, cat, rows, D, D, stream);
        DLinear(cat, R, L.skip_in, hp, false);
      }
      LaunchAdaRmsNorm(hp, L.attn_norm, cur_ada + (2 * l) * 2 * D, yp, rows, D, 1e-5F, stream);
      DLinear(yp, R, L.wqkv, qkvp, false);
      LaunchSplitQkv(qkvp, qp, kp, vp, rows, D, stream);
      ming::LaunchRopeInterleaved(qp, rows, tt, 8, 64, inv_freq, stream);  // position = row % t
      ming::LaunchRopeInterleaved(kp, rows, tt, 8, 64, inv_freq, stream);
      DAttention(qp, kp, vp, ap, tt, static_cast<std::size_t>(nb));
      DLinear(ap, R, L.wo, hp, true);
      LaunchAdaRmsNorm(hp, L.ffn_norm, cur_ada + (2 * l + 1) * 2 * D, yp, rows, D, 1e-5F, stream);
      DLinear(yp, R, L.w1, f1p, false);
      DLinear(yp, R, L.w3, f3p, false);
      ming::LaunchSiluMul(f1p, f3p, rows * 1536, stream);
      DLinear(f1p, R, L.w2, hp, true);
      if (l < 6) {
        float* s = skips[skip_top++]->Ensure(rows * D);
        Check(hipMemcpyAsync(s, hp, rows * D * 4, hipMemcpyDeviceToDevice, stream), "skip");
      }
    }
    LaunchAdaRmsNorm(hp, final_norm_w, cur_ada + 26 * 2 * D, yp, rows, D, 1e-5F, stream);
    float* cat = col.Ensure(rows * (D + 80));
    LaunchConcatCols(yp, xd, cat, rows, D, 80, stream);
    float* xr = xres.Ensure(pr * D);
    DLinear(cat, R, skip_linear, xr, false);
    float* w = wx.Ensure(pr * D);
    DLinear(xr, R, conv1, w, false);
    float* o = out.Ensure(pr * D);
    Check(hipMemsetAsync(o, 0, rows * D * 4, stream), "zero");
    float* a2 = rs.Ensure(pr * 2 * D);
    float* ac = acts.Ensure(pr * D);
    for (int i = 0; i < 8; ++i) {
      DConv(wn_in[i], w, t, nb, a2);
      LaunchWaveNetGate(a2, cur_wng + i * 2 * D, ac, rows, D, stream);
      if (i < 7) {
        DLinear(ac, R, wn_res[i], w, true);
      }
      DLinear(ac, R, wn_skip[i], o, true);
    }
    DLinear(xr, R, res_projection, o, true);
    LaunchModulatedLayerNorm(o, cur_fmod, yp, rows, D, 1e-6F, stream);
    DLinear(yp, R, final_linear, ap, false);
    DLinear(ap, R, conv2, vout, false);
    prof.Mark(Prof::kOther, stream);
  }

  // Time-only conditioning of one time value into the single-step buffers (tests / Velocity).
  void UploadTime(float time) {
    const S2Mel::TimeConditioning tc = s2mel->TimeCondition(time);
    Check(hipMemcpyAsync(ada.Ensure(tc.ada.size()), tc.ada.data(), tc.ada.size() * 4, hipMemcpyHostToDevice,
                         stream),
          "ada");
    Check(hipMemcpyAsync(wng.Ensure(tc.wavenet.size()), tc.wavenet.data(), tc.wavenet.size() * 4,
                         hipMemcpyHostToDevice, stream),
          "wn");
    Check(hipMemcpyAsync(fmod.Ensure(tc.final_mod.size()), tc.final_mod.data(), tc.final_mod.size() * 4,
                         hipMemcpyHostToDevice, stream),
          "fmod");
    Check(hipStreamSynchronize(stream), "time sync");
    cur_ada = ada.p;
    cur_wng = wng.p;
    cur_fmod = fmod.p;
  }

  // The time conditioning of every Euler step depends only on the step count, so it is computed once on
  // the CPU and kept on the device (the first port recomputed and uploaded it every step).
  void PrepareTimeGrid(const std::vector<float>& times) {
    if (times == grid_times) {
      return;
    }
    std::vector<float> a, w, f;
    for (const float time : times) {
      const S2Mel::TimeConditioning tc = s2mel->TimeCondition(time);
      ada_n = tc.ada.size();
      wng_n = tc.wavenet.size();
      fmod_n = tc.final_mod.size();
      a.insert(a.end(), tc.ada.begin(), tc.ada.end());
      w.insert(w.end(), tc.wavenet.begin(), tc.wavenet.end());
      f.insert(f.end(), tc.final_mod.begin(), tc.final_mod.end());
    }
    Check(hipMemcpy(grid_ada.Ensure(a.size()), a.data(), a.size() * 4, hipMemcpyHostToDevice), "grid ada");
    Check(hipMemcpy(grid_wng.Ensure(w.size()), w.data(), w.size() * 4, hipMemcpyHostToDevice), "grid wn");
    Check(hipMemcpy(grid_fmod.Ensure(f.size()), f.data(), f.size() * 4, hipMemcpyHostToDevice), "grid fmod");
    grid_times = times;
  }
  void SelectGridStep(std::size_t step) {
    cur_ada = grid_ada.p + step * ada_n;
    cur_wng = grid_wng.p + step * wng_n;
    cur_fmod = grid_fmod.p + step * fmod_n;
  }

  // c = merge_rest([prompt, cond_projection(mu), style]) + bias for one branch (t rows, once per request)
  void Conditioning(const float* prompt_d, const float* mu_d, const float* style_h, std::int64_t t, float* c) {
    constexpr std::int64_t D = 512;
    const auto tt = static_cast<std::size_t>(t);
    float* cp = f1.Ensure(tt * 1536);
    Linear(mu_d, t, cond_proj, cp);
    std::vector<float> st(tt * 192);
    for (std::size_t r = 0; r < tt; ++r) {
      std::memcpy(st.data() + r * 192, style_h, 192 * 4);
    }
    float* sd = f3.Ensure(tt * 1536);
    Check(hipMemcpy(sd, st.data(), st.size() * 4, hipMemcpyHostToDevice), "style");
    float* cat1 = col.Ensure(tt * (80 + D));
    LaunchConcatCols(prompt_d, cp, cat1, tt, 80, D, stream);
    float* cat2 = xin.Ensure(tt * 784);
    LaunchConcatCols(cat1, sd, cat2, tt, 80 + D, 192, stream);
    Linear(cat2, t, merge_rest, c);
  }

  std::vector<float> Vocode(const float* mel, std::int64_t frames) {
    float* a = va.Ensure(static_cast<std::size_t>(frames * 80));
    Check(hipMemcpy(a, mel, static_cast<std::size_t>(frames * 80) * 4, hipMemcpyHostToDevice), "mel");
    std::int64_t t = frames;
    float* cur = vb.Ensure(static_cast<std::size_t>(t * 1536));
    RunConv(pre, a, t, cur);
    std::int64_t ch = 1536;
    for (std::size_t i = 0; i < ups.size(); ++i) {
      const UpDev& u = ups[i];
      const std::int64_t t2 = (t - 1) * u.stride - 2 * u.pad + u.k;
      float* z = vt.Ensure(static_cast<std::size_t>(t * u.k * u.cout));
      Lin all = u.all;
      all.b = nullptr;
      Linear(cur, t, all, z);
      float* nx = va.Ensure(static_cast<std::size_t>(t2 * u.cout));
      LaunchConvTransposeGather(z, u.all.b, nx, static_cast<std::size_t>(t), static_cast<std::size_t>(t2),
                                static_cast<std::size_t>(u.cout), static_cast<std::size_t>(u.k),
                                static_cast<std::size_t>(u.stride), static_cast<std::size_t>(u.pad), stream);
      t = t2;
      ch = u.cout;
      const auto n = static_cast<std::size_t>(t * ch);
      float* acc = vacc.Ensure(n);
      Check(hipMemsetAsync(acc, 0, n * 4, stream), "acc");
      for (int j = 0; j < 3; ++j) {
        const BlockDev& b = blocks[i * 3 + static_cast<std::size_t>(j)];
        float* hh = vh.Ensure(n);
        Check(hipMemcpyAsync(hh, nx, n * 4, hipMemcpyDeviceToDevice, stream), "res copy");
        float* s1 = vb.Ensure(n);
        float* s2 = vc.Ensure(n);
        for (int d = 0; d < 3; ++d) {
          LaunchAliasFreeSnake(hh, s1, b.acts[2 * d].alpha, b.acts[2 * d].inv_beta, b.acts[2 * d].up,
                               b.acts[2 * d].down, static_cast<std::size_t>(t), static_cast<std::size_t>(ch),
                               stream);
          RunConv(b.c1[d], s1, t, s2);
          LaunchAliasFreeSnake(s2, s1, b.acts[2 * d + 1].alpha, b.acts[2 * d + 1].inv_beta, b.acts[2 * d + 1].up,
                               b.acts[2 * d + 1].down, static_cast<std::size_t>(t), static_cast<std::size_t>(ch),
                               stream);
          RunConv(b.c2[d], s1, t, s2);
          ming::LaunchAdd(hh, s2, n, stream);
        }
        ming::LaunchAdd(acc, hh, n, stream);
      }
      cur = vb.Ensure(n);
      LaunchScale(cur, acc, 1.0F / 3.0F, n, stream);
      // nx (va) is free again; keep `cur` in vb
    }
    float* act = vc.Ensure(static_cast<std::size_t>(t * ch));
    LaunchAliasFreeSnake(cur, act, post_act.alpha, post_act.inv_beta, post_act.up, post_act.down,
                         static_cast<std::size_t>(t), static_cast<std::size_t>(ch), stream);
    float* wav = va.Ensure(static_cast<std::size_t>(t));
    RunConv(post, act, t, wav);
    LaunchClamp(wav, -1.0F, 1.0F, static_cast<std::size_t>(t), stream);
    std::vector<float> out_h(static_cast<std::size_t>(t));
    Check(hipMemcpyAsync(out_h.data(), wav, out_h.size() * 4, hipMemcpyDeviceToHost, stream), "wav");
    Check(hipStreamSynchronize(stream), "vocoder sync");
    return out_h;
  }
};

Runtime::~Runtime() {
  if (impl_ != nullptr) {
    for (float* p : impl_->kcache) {
      (void)hipFree(p);
    }
    for (float* p : impl_->vcache) {
      (void)hipFree(p);
    }
    if (impl_->stream != nullptr) {
      (void)hipStreamDestroy(impl_->stream);
    }
  }
}

std::unique_ptr<Runtime> Runtime::Create(const UnifiedVoice& gpt, const S2Mel& s2mel,
                                         const BigVgan& vocoder, std::string* error) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    if (error != nullptr) {
      *error = "no HIP device";
    }
    return nullptr;
  }
  auto rt = std::unique_ptr<Runtime>(new Runtime());
  rt->impl_ = std::make_unique<Impl>();
  try {
    Check(hipStreamCreate(&rt->impl_->stream), "stream");
    rt->impl_->blas = std::make_unique<Blas>(rt->impl_->stream);
    const char* fast = std::getenv("GUFO_INDEXTTS_DIT_FAST");
    rt->impl_->fast = fast == nullptr || fast[0] != '0';
    const char* profile = std::getenv("GUFO_INDEXTTS_PROFILE");
    rt->impl_->prof.on = profile != nullptr && profile[0] == '1';
    rt->impl_->LoadGpt(gpt);
    rt->impl_->LoadDit(s2mel);
    rt->impl_->LoadVocoder(vocoder);
    Check(hipDeviceSynchronize(), "upload sync");
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = e.what();
    }
    return nullptr;
  }
  return rt;
}

void Runtime::GptPrefill(const std::vector<float>& prefix, std::int64_t rows, std::int64_t max_new,
                         std::vector<float>* logits) {
  Impl& m = *impl_;
  const std::int64_t d = m.voice->config().dim;
  const std::int64_t need = rows + 1 + max_new + 1;
  if (need > m.cap) {
    for (float* p : m.kcache) {
      (void)hipFree(p);
    }
    for (float* p : m.vcache) {
      (void)hipFree(p);
    }
    m.kcache.assign(m.gpt.size(), nullptr);
    m.vcache.assign(m.gpt.size(), nullptr);
    for (std::size_t l = 0; l < m.gpt.size(); ++l) {
      Check(hipMalloc(reinterpret_cast<void**>(&m.kcache[l]), static_cast<std::size_t>(need * d) * 4), "kcache");
      Check(hipMalloc(reinterpret_cast<void**>(&m.vcache[l]), static_cast<std::size_t>(need * d) * 4), "vcache");
    }
    m.cap = need;
  }
  m.len = 0;
  std::vector<float> host(prefix.begin(), prefix.begin() + rows * d);
  host.resize(static_cast<std::size_t>((rows + 1) * d));
  const PthArchive& sd = m.voice->archive();
  const float* mel = sd.F32("model.mel_embedding.weight") + static_cast<std::int64_t>(m.voice->config().start_mel) * d;
  const float* pos = sd.F32("model.mel_pos_embedding.emb.weight");
  for (std::int64_t i = 0; i < d; ++i) {
    host[static_cast<std::size_t>(rows * d + i)] = mel[i] + pos[i];
  }
  float* xp = m.x.Ensure(host.size());
  Check(hipMemcpy(xp, host.data(), host.size() * 4, hipMemcpyHostToDevice), "prefix");
  m.GptForward(rows + 1, logits);
  m.mel_tokens = 1;
}

void Runtime::GptStep(std::int32_t token, std::vector<float>* logits) {
  Impl& m = *impl_;
  const std::int64_t d = m.voice->config().dim;
  const std::int64_t position = m.mel_tokens + 1;
  if (m.len + 1 > m.cap || position >= m.voice->config().max_mel_positions) {
    throw std::runtime_error("GPT KV cache / mel positions exhausted");
  }
  const PthArchive& sd = m.voice->archive();
  const float* mel = sd.F32("model.mel_embedding.weight") + static_cast<std::int64_t>(token) * d;
  const float* pos = sd.F32("model.mel_pos_embedding.emb.weight") + position * d;
  std::vector<float> row(static_cast<std::size_t>(d));
  for (std::int64_t i = 0; i < d; ++i) {
    row[static_cast<std::size_t>(i)] = mel[i] + pos[i];
  }
  float* xp = m.x.Ensure(static_cast<std::size_t>(d));
  Check(hipMemcpy(xp, row.data(), row.size() * 4, hipMemcpyHostToDevice), "step row");
  m.GptForward(1, logits);
  ++m.mel_tokens;
}

std::vector<float> Runtime::Velocity(const float* x, const float* prompt, float time, const float* style,
                                     const float* cond, std::int64_t t) {
  Impl& m = *impl_;
  const auto tt = static_cast<std::size_t>(t);
  const std::size_t pr = Blas::PaddedRows(tt);
  m.UploadTime(time);
  float* xd = m.dx.Ensure(tt * 80);
  float* pd = m.dprompt.Ensure(tt * 80);
  float* cd = m.dcond.Ensure(tt * 512);
  Check(hipMemcpy(xd, x, tt * 80 * 4, hipMemcpyHostToDevice), "x");
  Check(hipMemcpy(pd, prompt, tt * 80 * 4, hipMemcpyHostToDevice), "prompt");
  Check(hipMemcpy(cd, cond, tt * 512 * 4, hipMemcpyHostToDevice), "cond");
  float* c = m.c_cond.Ensure(tt * 512);
  m.Conditioning(pd, cd, style, t, c);
  float* v = m.v_c.Ensure(pr * 80);
  m.Estimator(xd, c, t, 1, v);
  std::vector<float> out(tt * 80);
  Check(hipMemcpy(out.data(), v, out.size() * 4, hipMemcpyDeviceToHost), "v");
  return out;
}

std::vector<float> Runtime::CfmSample(const std::vector<float>& noise, const float* mu, std::int64_t t,
                                      const float* prompt_mel, std::int64_t prompt_len, const float* style,
                                      const CfmOptions& options) {
  Impl& m = *impl_;
  const auto tt = static_cast<std::size_t>(t);
  const std::size_t pr2 = Blas::PaddedRows(2 * tt);
  std::vector<float> prompt_x(tt * 80, 0.0F);
  std::copy_n(prompt_mel, prompt_len * 80, prompt_x.data());
  std::vector<float> x0(noise);
  std::fill(x0.begin(), x0.begin() + prompt_len * 80, 0.0F);
  // CFG as one batch of two sequences, as in the reference solve_euler: rows [0, t) conditional,
  // [t, 2t) unconditional (prompt 0, mu 0, style 0); both branches see the same x.
  float* xd = m.dx.Ensure(2 * tt * 80);
  float* pd = m.dprompt.Ensure(tt * 80);
  float* md = m.dcond.Ensure(tt * 512);
  Check(hipMemcpy(xd, x0.data(), tt * 80 * 4, hipMemcpyHostToDevice), "x");
  Check(hipMemcpy(xd + tt * 80, x0.data(), tt * 80 * 4, hipMemcpyHostToDevice), "x");
  Check(hipMemcpy(pd, prompt_x.data(), tt * 80 * 4, hipMemcpyHostToDevice), "prompt");
  Check(hipMemcpy(md, mu, tt * 512 * 4, hipMemcpyHostToDevice), "mu");
  float* c2 = m.c_cond.Ensure(2 * tt * 512);
  m.Conditioning(pd, md, style, t, c2);
  Check(hipMemsetAsync(pd, 0, tt * 80 * 4, m.stream), "zero prompt");
  Check(hipMemsetAsync(md, 0, tt * 512 * 4, m.stream), "zero mu");
  const std::vector<float> zero_style(192, 0.0F);
  m.Conditioning(pd, md, zero_style.data(), t, c2 + tt * 512);
  const std::vector<float> span = TorchLinspace01(options.steps);
  std::vector<float> times;
  float time = span[0];
  for (int step = 1; step <= options.steps; ++step) {
    times.push_back(time);
    time = time + (span[static_cast<std::size_t>(step)] - span[static_cast<std::size_t>(step - 1)]);
  }
  m.PrepareTimeGrid(times);
  float* v = m.v_c.Ensure(pr2 * 80);
  m.prof.Reset();
  const auto t0 = std::chrono::steady_clock::now();
  for (int step = 1; step <= options.steps; ++step) {
    const float dt = span[static_cast<std::size_t>(step)] - span[static_cast<std::size_t>(step - 1)];
    m.SelectGridStep(static_cast<std::size_t>(step - 1));
    m.Estimator(xd, c2, t, 2, v);
    LaunchCfgEulerTM(xd, v, v + tt * 80, options.cfg_rate, dt, tt, 80, static_cast<std::size_t>(prompt_len),
                     m.stream);
    Check(hipMemcpyAsync(xd + tt * 80, xd, tt * 80 * 4, hipMemcpyDeviceToDevice, m.stream), "x copy");
    m.prof.Flush();
  }
  std::vector<float> out(tt * 80);
  Check(hipMemcpyAsync(out.data(), xd, out.size() * 4, hipMemcpyDeviceToHost, m.stream), "mel");
  Check(hipStreamSynchronize(m.stream), "cfm sync");
  if (m.prof.on) {
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    char line[400];
    std::snprintf(line, sizeof(line),
                  "cfm profile (%s, %lld frames x2, %d steps): wall %.1f ms | gemm %.1f, attention %.1f, im2col %.1f, "
                  "other %.1f ms (GPU, incl. per-step sync)",
                  m.fast ? "fast" : "legacy", static_cast<long long>(t), options.steps, wall, m.prof.ms[Prof::kGemm],
                  m.prof.ms[Prof::kAttention], m.prof.ms[Prof::kConv], m.prof.ms[Prof::kOther]);
    m.last_profile = line;
    std::fprintf(stderr, "indextts %s\n", line);
  }
  return out;
}

void Runtime::SetFastDit(bool fast) { impl_->fast = fast; }

std::string Runtime::LastCfmProfile() const { return impl_->last_profile; }

std::vector<float> Runtime::Vocode(const float* mel, std::int64_t frames) { return impl_->Vocode(mel, frames); }

#endif

}  // namespace gufo::models::indextts::hip
