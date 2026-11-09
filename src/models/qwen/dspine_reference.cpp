#include "src/models/qwen/dspine_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen {
namespace {

constexpr float kTiny = 1.1754943508222875e-38f;

/// float -> nearest BF16 (round to nearest even) -> float, as MLX stores
/// every bf16 intermediate.
inline float Bf(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  std::memcpy(&x, &u, 4);
  return x;
}

void BfAll(std::vector<float>& v) {
  for (float& x : v) x = Bf(x);
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Elements per block / bytes per block of the storage types the converter
/// and the 27B target use.
std::pair<std::size_t, std::size_t> BlockOf(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kF32: return {1, 4};
    case core::GgmlType::kBF16: return {1, 2};
    case core::GgmlType::kQ8_0: return {32, 34};
    case core::GgmlType::kQ4_K: return {256, 144};
    case core::GgmlType::kQ6_K: return {256, 210};
    default: throw std::runtime_error("dspine: unsupported tensor type");
  }
}

void Dequant(core::GgmlType type, const void* src, float* dst, std::size_t n) {
  switch (type) {
    case core::GgmlType::kF32:
      std::memcpy(dst, src, n * 4);
      return;
    case core::GgmlType::kBF16: {
      const auto* s = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(s[i]) << 16;
        std::memcpy(&dst[i], &u, 4);
      }
      return;
    }
    case core::GgmlType::kQ8_0: quant::DequantizeQ8_0(src, dst, n); return;
    case core::GgmlType::kQ4_K: quant::DequantizeQ4_K(src, dst, n); return;
    case core::GgmlType::kQ6_K: quant::DequantizeQ6_K(src, dst, n); return;
    default: throw std::runtime_error("dspine: unsupported tensor type");
  }
}

/// RMSNorm with one bf16 rounding of the scaled output (mx.fast.rms_norm).
void RmsNorm(float* x, std::size_t n, const float* w, float eps) {
  double ss = 0;
  for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
  for (std::size_t i = 0; i < n; ++i) x[i] = Bf(x[i] * inv * w[i]);
}

void Rope(float* v, std::size_t head_dim, std::size_t pos, float theta) {
  const std::size_t half = head_dim / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const double inv = std::pow(static_cast<double>(theta),
                                -2.0 * static_cast<double>(i) / head_dim);
    const double a = static_cast<double>(pos) * inv;
    const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
    const float x1 = v[i], x2 = v[i + half];
    v[i] = Bf(x1 * c - x2 * s);
    v[i + half] = Bf(x1 * s + x2 * c);
  }
}

}  // namespace

std::unique_ptr<DSpineReference> DSpineReference::Open(
    const std::filesystem::path& gguf, std::string* error) {
  auto self = std::unique_ptr<DSpineReference>(new DSpineReference());
  self->reader_ = core::GgufReader::OpenFile(gguf, error);
  if (!self->reader_) return nullptr;
  const auto& r = *self->reader_;
  auto cfg = ReadDSpineConfig(r, error);
  if (!cfg) return nullptr;
  self->cfg_ = std::move(*cfg);
  return self;
}

void DSpineReference::Row(std::string_view name, std::size_t row, float* out) const {
  const auto* t = reader_->FindTensor(name);
  if (!t) throw std::runtime_error("dspine: missing tensor " + std::string(name));
  const std::size_t ne0 = t->dimensions[0];
  const auto [vals, bytes] = BlockOf(t->type);
  const std::size_t row_bytes = ne0 / vals * bytes;
  Dequant(t->type, static_cast<const std::uint8_t*>(t->data) + row * row_bytes, out, ne0);
}

std::vector<float> DSpineReference::Matrix(std::string_view name) const {
  const auto* t = reader_->FindTensor(name);
  if (!t) throw std::runtime_error("dspine: missing tensor " + std::string(name));
  const std::size_t n = t->ElementCount(), ne0 = t->dimensions[0], rows = n / ne0;
  std::vector<float> out(n);
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(rows); ++i)
    Row(name, static_cast<std::size_t>(i), out.data() + static_cast<std::size_t>(i) * ne0);
  return out;
}

/// y[rows, out] = bf16(x[rows, in] @ W^T), W stored as `out` GGUF rows of `in`.
std::vector<float> DSpineReference::Linear(std::span<const float> x, std::size_t rows,
                                           std::string_view weight) const {
  const auto* t = reader_->FindTensor(weight);
  if (!t) throw std::runtime_error("dspine: missing tensor " + std::string(weight));
  const std::size_t in = t->dimensions[0], out = t->ElementCount() / in;
  if (x.size() != rows * in) throw std::runtime_error("dspine: Linear shape " + std::string(weight));
  std::vector<float> y(rows * out);
#pragma omp parallel
  {
    std::vector<float> w(in);
#pragma omp for schedule(static)
    for (std::int64_t o = 0; o < static_cast<std::int64_t>(out); ++o) {
      Row(weight, static_cast<std::size_t>(o), w.data());
      for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x.data() + r * in;
        double acc = 0;
        for (std::size_t i = 0; i < in; ++i) acc += static_cast<double>(xr[i]) * w[i];
        y[r * out + static_cast<std::size_t>(o)] = Bf(static_cast<float>(acc));
      }
    }
  }
  return y;
}

DSpineRound DSpineReference::Draft(std::span<const float> context, std::size_t S,
                                   std::int64_t anchor,
                                   const DSpineTargetAccess& target) const {
  const auto& c = cfg_;
  const std::size_t H = c.hidden, T = c.block_size, NH = c.heads, NKV = c.kv_heads,
                    HD = c.head_dim, R = c.rank, C = c.code_size, K = c.head_topk,
                    NL = c.layers, V = c.vocab;
  const float sqc = std::sqrt(static_cast<float>(C));

  // Embedding of [anchor, mask * (T-1)] from the target (bf16 rows).
  std::vector<float> residual(T * H);
  for (std::size_t t = 0; t < T; ++t)
    target.embed_row(t == 0 ? anchor : static_cast<std::int64_t>(c.mask_token), residual.data() + t * H);
  BfAll(residual);

  // Context: hidden_norm(fc(target_hidden)).
  std::vector<float> ctx = Linear(context, S, "fc.weight");
  {
    std::vector<float> w(H);
    Row("enc.output_norm.weight", 0, w.data());
    for (std::size_t s = 0; s < S; ++s) RmsNorm(ctx.data() + s * H, H, w.data(), c.rms_eps);
  }
  std::vector<float> x_in = residual;
  {
    std::vector<float> w(H);
    Row("blk.0.attn_norm.weight", 0, w.data());
    for (std::size_t t = 0; t < T; ++t) RmsNorm(x_in.data() + t * H, H, w.data(), c.rms_eps);
  }

  const auto packed_gate = Matrix("dspine.bridge.gate.weight");             // [R, C]
  const auto packed_pred = Matrix("dspine.bridge.predecessor_gate.weight");  // [R, C]
  const auto packed_prev = Matrix("dspine.bridge.prev.weight");              // [R, C]
  std::vector<float> gate_b(R);
  Row("dspine.bridge.gate.bias", 0, gate_b.data());
  std::vector<float> cb_anchor(C);
  Row("dspine.bridge.codebook", static_cast<std::size_t>(anchor), cb_anchor.data());
  const auto feature_write = Matrix("dspine.bridge.feature_write");  // [NL, H, R]

  std::vector<float> stash_raw, stash_rms, stash_packed;
  for (std::size_t L = 0; L < NL; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    auto q = Linear(x_in, T, p + "attn_q.weight");
    auto kb = Linear(x_in, T, p + "attn_k.weight");
    auto vb = Linear(x_in, T, p + "attn_v.weight");
    auto kc = Linear(ctx, S, p + "attn_k.weight");
    auto vc = Linear(ctx, S, p + "attn_v.weight");
    std::vector<float> qn(HD), kn(HD);
    Row(p + "attn_q_norm.weight", 0, qn.data());
    Row(p + "attn_k_norm.weight", 0, kn.data());
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t h = 0; h < NH; ++h) {
        float* v = q.data() + (t * NH + h) * HD;
        RmsNorm(v, HD, qn.data(), c.rms_eps);
        Rope(v, HD, S + t, c.rope_theta);
      }
      for (std::size_t h = 0; h < NKV; ++h) {
        float* v = kb.data() + (t * NKV + h) * HD;
        RmsNorm(v, HD, kn.data(), c.rms_eps);
        Rope(v, HD, S + t, c.rope_theta);
      }
    }
    for (std::size_t s = 0; s < S; ++s)
      for (std::size_t h = 0; h < NKV; ++h) {
        float* v = kc.data() + (s * NKV + h) * HD;
        RmsNorm(v, HD, kn.data(), c.rms_eps);
        Rope(v, HD, s, c.rope_theta);
      }
    // Attention: context fully visible, causal inside the block, sliding window.
    std::vector<float> att(T * NH * HD);
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
#pragma omp parallel for collapse(2)
    for (std::int64_t t = 0; t < static_cast<std::int64_t>(T); ++t)
      for (std::int64_t h = 0; h < static_cast<std::int64_t>(NH); ++h) {
        const std::size_t kv = static_cast<std::size_t>(h) / (NH / NKV);
        const float* qv = q.data() + (static_cast<std::size_t>(t) * NH + h) * HD;
        const std::size_t nk = S + static_cast<std::size_t>(t) + 1;
        const std::size_t qpos = S + static_cast<std::size_t>(t);
        std::vector<float> sc(nk);
        float mx = -INFINITY;
        for (std::size_t j = 0; j < nk; ++j) {
          const float* kv_ = j < S ? kc.data() + (j * NKV + kv) * HD
                                   : kb.data() + ((j - S) * NKV + kv) * HD;
          const std::size_t kpos = j < S ? j : S + (j - S);
          if (c.sliding_window && qpos - kpos >= c.sliding_window) { sc[j] = -INFINITY; continue; }
          double d = 0;
          for (std::size_t i = 0; i < HD; ++i) d += static_cast<double>(qv[i]) * kv_[i];
          sc[j] = static_cast<float>(d) * scale;
          mx = std::max(mx, sc[j]);
        }
        double den = 0;
        for (auto& v : sc) { v = std::exp(v - mx); den += v; }
        float* o = att.data() + (static_cast<std::size_t>(t) * NH + h) * HD;
        for (std::size_t i = 0; i < HD; ++i) {
          double acc = 0;
          for (std::size_t j = 0; j < nk; ++j) {
            const float* vv = j < S ? vc.data() + (j * NKV + kv) * HD
                                    : vb.data() + ((j - S) * NKV + kv) * HD;
            acc += sc[j] * static_cast<double>(vv[i]);
          }
          o[i] = Bf(static_cast<float>(acc / den));
        }
      }
    auto ao = Linear(att, T, p + "attn_output.weight");
    std::vector<float> h(T * H);
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = Bf(residual[i] + ao[i]);
    std::vector<float> hn = h;
    {
      std::vector<float> w(H);
      Row(p + "ffn_norm.weight", 0, w.data());
      for (std::size_t t = 0; t < T; ++t) RmsNorm(hn.data() + t * H, H, w.data(), c.rms_eps);
    }
    auto g = Linear(hn, T, p + "ffn_gate.weight");
    auto u = Linear(hn, T, p + "ffn_up.weight");
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = Bf((g[i] / (1.0f + std::exp(-g[i]))) * u[i]);
    auto dn = Linear(g, T, p + "ffn_down.weight");
    std::vector<float> out(T * H);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = Bf(h[i] + dn[i]);

    // ---- code bridge (bridge.py __call__) ----
    std::vector<float> rmsv(T), normalized(T * H);
    for (std::size_t t = 0; t < T; ++t) {
      double ss = 0;
      for (std::size_t i = 0; i < H; ++i) ss += static_cast<double>(out[t * H + i]) * out[t * H + i];
      rmsv[t] = std::sqrt(std::max(static_cast<float>(ss / H), kTiny));
      for (std::size_t i = 0; i < H; ++i)
        normalized[t * H + i] = Bf(out[t * H + i] / (rmsv[t] + c.bridge_eps));
    }
    std::vector<float> projected(T * C);
    {
      const bool last = L == NL - 1;
      const auto* rt = reader_->FindTensor(last ? "dspine.bridge.read" : "dspine.bridge.early_read");
      const std::size_t base_row = last ? 0 : L * C;  // early_read: [NL-1][C] rows of H
      (void)rt;
#pragma omp parallel
      {
        std::vector<float> w(H);
#pragma omp for
        for (std::int64_t o = 0; o < static_cast<std::int64_t>(C); ++o) {
          Row(last ? "dspine.bridge.read" : "dspine.bridge.early_read", base_row + static_cast<std::size_t>(o), w.data());
          for (std::size_t t = 0; t < T; ++t) {
            double acc = 0;
            for (std::size_t i = 0; i < H; ++i) acc += static_cast<double>(normalized[t * H + i]) * w[i];
            projected[t * C + static_cast<std::size_t>(o)] = Bf(static_cast<float>(acc));
          }
        }
      }
    }
    std::vector<float> codes(T * C);
    for (std::size_t t = 0; t < T; ++t) {
      double pn = 0;
      for (std::size_t i = 0; i < C; ++i) pn += static_cast<double>(projected[t * C + i]) * projected[t * C + i];
      const float pnf = std::sqrt(static_cast<float>(pn));
      for (std::size_t i = 0; i < C; ++i)
        codes[t * C + i] = t == 0 ? Bf(sqc * cb_anchor[i]) : Bf(sqc * projected[t * C + i] / (pnf + c.bridge_eps));
    }
    std::vector<float> packed(T * 3 * R);
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t part = 0; part < 3; ++part) {
        const auto& W = part == 0 ? packed_gate : part == 1 ? packed_pred : packed_prev;
        for (std::size_t o = 0; o < R; ++o) {
          double acc = 0;
          for (std::size_t i = 0; i < C; ++i) acc += static_cast<double>(codes[t * C + i]) * W[o * C + i];
          packed[t * 3 * R + part * R + o] = Bf(static_cast<float>(acc));
        }
      }
    if (L == NL - 1) { stash_raw = out; stash_rms = rmsv; stash_packed = packed; }
    std::vector<float> message(T * R, 0.0f);
    for (std::size_t t = 1; t < T; ++t)
      for (std::size_t o = 0; o < R; ++o) {
        const float qv = packed[t * 3 * R + o], sv = packed[(t - 1) * 3 * R + R + o],
                    vv = packed[(t - 1) * 3 * R + 2 * R + o];
        message[t * R + o] = Bf(Sigmoid(qv + sv + gate_b[o]) * vv);
      }
    const float* fw = feature_write.data() + L * H * R;  // [H, R]
    std::vector<float> out_w(T * H);
#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(H); ++i)
      for (std::size_t t = 0; t < T; ++t) {
        double acc = 0;
        for (std::size_t o = 0; o < R; ++o) acc += static_cast<double>(message[t * R + o]) * fw[static_cast<std::size_t>(i) * R + o];
        const float delta = Bf(static_cast<float>(acc));
        const float f = out[t * H + static_cast<std::size_t>(i)];
        out_w[t * H + static_cast<std::size_t>(i)] = t == 0 ? Bf(f) : Bf(f + rmsv[t] * delta);
      }
    std::vector<float> nw(H);
    Row(L + 1 < NL ? "blk." + std::to_string(L + 1) + ".attn_norm.weight" : std::string("output_norm.weight"), 0, nw.data());
    for (std::size_t t = 0; t < T; ++t) {
      double ss = 0;
      for (std::size_t i = 0; i < H; ++i) ss += static_cast<double>(out_w[t * H + i]) * out_w[t * H + i];
      const float inv = 1.0f / std::sqrt(static_cast<float>(ss / H) + c.rms_eps);
      for (std::size_t i = 0; i < H; ++i) x_in[t * H + i] = Bf(Bf(out_w[t * H + i] * inv) * nw[i]);
    }
    residual = out_w;
  }

  // ---- greedy refinement tail (propose.py RefinementTail.greedy) ----
  DSpineRound round;
  const std::size_t n = T - 1;
  round.rows.assign(x_in.begin() + H, x_in.end());
  round.stash_raw = stash_raw;
  std::vector<float> logits(n * V);
  target.head_logits(round.rows.data(), n, logits.data());
  round.cand.resize(n * K);
  round.cand_logits.resize(n * K);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::int32_t> ord(V);
    std::iota(ord.begin(), ord.end(), 0);
    const float* l = logits.data() + i * V;
    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](std::int32_t a, std::int32_t b) {
      return l[a] > l[b] || (l[a] == l[b] && a < b);
    });
    std::sort(ord.begin(), ord.begin() + K);
    for (std::size_t k = 0; k < K; ++k) {
      round.cand[i * K + k] = ord[k];
      round.cand_logits[i * K + k] = l[ord[k]];
    }
  }
  std::vector<float> norm_w(H);
  Row("output_norm.weight", 0, norm_w.data());
  const float* write_w = feature_write.data() + (NL - 1) * H * R;  // [H, R]
  std::vector<float> row(H), cb(C);
  auto tg_row = [&](std::int64_t v, float* out) {  // bf16((head*norm)) @ W -> [R]
    target.head_row(v, row.data());
    for (std::size_t i = 0; i < H; ++i) row[i] = Bf(Bf(row[i]) * norm_w[i]);
    for (std::size_t o = 0; o < R; ++o) {
      double acc = 0;
      for (std::size_t i = 0; i < H; ++i) acc += static_cast<double>(row[i]) * write_w[i * R + o];
      out[o] = Bf(static_cast<float>(acc));
    }
  };
  auto tsv_row = [&](std::int64_t v, float* out) {  // bf16(sqrt(C) * codebook) @ [pred; prev]^T -> [2R]
    Row("dspine.bridge.codebook", static_cast<std::size_t>(v), cb.data());
    for (std::size_t i = 0; i < C; ++i) cb[i] = Bf(sqc * cb[i]);
    for (std::size_t part = 0; part < 2; ++part) {
      const auto& W = part == 0 ? packed_pred : packed_prev;
      for (std::size_t o = 0; o < R; ++o) {
        double acc = 0;
        for (std::size_t i = 0; i < C; ++i) acc += static_cast<double>(cb[i]) * W[o * C + i];
        out[part * R + o] = Bf(static_cast<float>(acc));
      }
    }
  };
  std::vector<std::int32_t> idx(n * K);
  std::vector<float> tg(K * R), tsv(K * 2 * R), base(K), hr(H), m(R);
  for (std::size_t i = 0; i < n; ++i) {
    const float* raw = stash_raw.data() + (i + 1) * H;
    const float rms_i = stash_rms[i + 1];
    const float* qg = stash_packed.data() + (i + 1) * 3 * R;
    for (std::size_t k = 0; k < K; ++k) {
      const std::int64_t v = round.cand[i * K + k];
      target.head_row(v, hr.data());
      double acc = 0;
      for (std::size_t j = 0; j < H; ++j) acc += static_cast<double>(raw[j] * norm_w[j]) * Bf(hr[j]);
      base[k] = static_cast<float>(acc);
      tg_row(v, tg.data() + k * R);
      const std::int64_t sid = i == 0 ? anchor : round.cand[(i - 1) * K + k];
      tsv_row(sid, tsv.data() + k * 2 * R);
    }
    for (std::size_t s = 0; s < K; ++s) {  // sender s -> best receiver
      for (std::size_t o = 0; o < R; ++o)
        m[o] = Sigmoid(qg[o] + tsv[s * 2 * R + o] + gate_b[o]) * tsv[s * 2 * R + R + o];
      float best = -INFINITY;
      std::int32_t arg = 0;
      for (std::size_t j = 0; j < K; ++j) {
        double sv = 0;
        for (std::size_t o = 0; o < R; ++o) sv += static_cast<double>(m[o]) * tg[j * R + o];
        const float score = base[j] + rms_i * static_cast<float>(sv);
        if (score > best) { best = score; arg = static_cast<std::int32_t>(j); }
      }
      idx[i * K + s] = arg;
    }
  }
  round.drafts.resize(n);
  std::int32_t j = idx[0];
  for (std::size_t i = 0; i < n; ++i) {
    round.drafts[i] = round.cand[i * K + static_cast<std::size_t>(j)];
    if (i + 1 < n) j = idx[(i + 1) * K + static_cast<std::size_t>(j)];
  }
  return round;
}

}  // namespace gufo::models::qwen
