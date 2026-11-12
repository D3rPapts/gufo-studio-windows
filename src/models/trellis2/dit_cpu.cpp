// TRELLIS.2: flow DiT, FP32 CPU reference with streamed weights (see dit.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include "src/models/trellis2/scheduler.hpp"
#include "src/models/wan22/tensor_file.hpp"

namespace gufo::trellis2 {
namespace {

void linear(const float* in, int tokens, const float* w, const float* b, int rows, int cols, float* out) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float* x = in + static_cast<std::size_t>(t) * cols;
        float* y = out + static_cast<std::size_t>(t) * rows;
        for (int r = 0; r < rows; ++r) {
            const float* wr = w + static_cast<std::size_t>(r) * cols;
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) acc += x[c] * wr[c];
            y[r] = acc + (b ? b[r] : 0.f);
        }
    }
}

void layer_norm(const float* x, int tokens, int width, const float* w, const float* b, float eps, float* out) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float* row = x + static_cast<std::size_t>(t) * width;
        float* y = out + static_cast<std::size_t>(t) * width;
        double sum = 0;
        for (int c = 0; c < width; ++c) sum += row[c];
        const float mean = static_cast<float>(sum / width);
        double sq = 0;
        for (int c = 0; c < width; ++c) sq += static_cast<double>(row[c] - mean) * (row[c] - mean);
        const float rstd = 1.f / std::sqrt(static_cast<float>(sq / width) + eps);
        for (int c = 0; c < width; ++c) {
            const float n = (row[c] - mean) * rstd;
            y[c] = w ? n * w[c] + b[c] : n;
        }
    }
}

// x * (1 + scale) + shift, shift/scale = mod chunks.
void modulate(float* x, int tokens, int width, const float* shift, const float* scale) {
    for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < width; ++c) {
            float& v = x[static_cast<std::size_t>(t) * width + c];
            v = v * (1.f + scale[c]) + shift[c];
        }
}

// F.normalize(x) * gamma * sqrt(d) per head, in place on [tokens, heads * d] (column gamma).
void qk_norm(float* x, int tokens, int width, int heads, const float* gamma) {
    const int d = width / heads;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h) {
            float* v = x + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(h) * d;
            double sq = 0;
            for (int i = 0; i < d; ++i) sq += static_cast<double>(v[i]) * v[i];
            const float inv = static_cast<float>(std::sqrt(static_cast<double>(d)) / std::max(std::sqrt(sq), 1e-12));
            for (int i = 0; i < d; ++i) v[i] = v[i] * inv * gamma[static_cast<std::size_t>(h) * d + i];
        }
}

void attention(const float* q, int tq, int q_stride, const float* k, const float* v, int tk, int kv_stride, int heads,
               int d, float* out, int out_stride) {
    const float scale = 1.f / std::sqrt(static_cast<float>(d));
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < tq; ++i) {
        std::vector<float> s(static_cast<std::size_t>(tk));
        for (int h = 0; h < heads; ++h) {
            const float* qi = q + static_cast<std::size_t>(i) * q_stride + static_cast<std::size_t>(h) * d;
            float m = -INFINITY;
            for (int j = 0; j < tk; ++j) {
                const float* kj = k + static_cast<std::size_t>(j) * kv_stride + static_cast<std::size_t>(h) * d;
                float acc = 0.f;
                for (int c = 0; c < d; ++c) acc += qi[c] * kj[c];
                s[static_cast<std::size_t>(j)] = acc * scale;
                m = std::max(m, s[static_cast<std::size_t>(j)]);
            }
            float sum = 0.f;
            for (int j = 0; j < tk; ++j) { s[static_cast<std::size_t>(j)] = std::exp(s[static_cast<std::size_t>(j)] - m); sum += s[static_cast<std::size_t>(j)]; }
            const float inv = 1.f / sum;
            float* o = out + static_cast<std::size_t>(i) * out_stride + static_cast<std::size_t>(h) * d;
            for (int c = 0; c < d; ++c) o[c] = 0.f;
            for (int j = 0; j < tk; ++j) {
                const float p = s[static_cast<std::size_t>(j)] * inv;
                const float* vj = v + static_cast<std::size_t>(j) * kv_stride + static_cast<std::size_t>(h) * d;
                for (int c = 0; c < d; ++c) o[c] += p * vj[c];
            }
        }
    }
}

std::uint64_t rng_state;
float uniform() {
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    const std::uint64_t r = rng_state * 0x2545F4914F6CDD1DULL;
    return static_cast<float>(static_cast<double>(r >> 11) / 9007199254740992.0) * 2.f - 1.f;
}
void fill(std::vector<float>* v, std::size_t n, float s) {
    v->resize(n);
    for (auto& x : *v) x = uniform() * s;
}

}  // namespace

std::vector<Coord> dense_grid_coords(int res) {
    std::vector<Coord> c;
    c.reserve(static_cast<std::size_t>(res) * res * res);
    for (int x = 0; x < res; ++x)
        for (int y = 0; y < res; ++y)
            for (int z = 0; z < res; ++z) c.push_back({x, y, z});
    return c;
}

bool DitHostWeights::FromFile(const std::filesystem::path& file, DitHostWeights* out, std::string* error) {
    auto tf = std::make_shared<models::wan22::TensorFile>();
    if (!tf->Open(file, error)) return false;
    const auto& W = static_cast<std::size_t>(out->shape.width);
    const auto M = static_cast<std::size_t>(out->shape.mlp), C = static_cast<std::size_t>(out->shape.cond_dim);
    auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
        return tf->ReadF32(name, count, dst, error);
    };
    if (!tf->Has("input_layer.weight")) {
        if (error) *error = "not a converted TRELLIS.2 flow file (input_layer.weight missing): " + file.string();
        return false;
    }
    const auto in_shape = tf->Shape("input_layer.weight"), out_shape = tf->Shape("out_layer.weight");
    out->shape.in_channels = static_cast<int>(in_shape[1]);
    out->shape.out_channels = static_cast<int>(out_shape[0]);
    DitStemWeights& s = out->stem;
    if (!read("t_embedder.mlp.0.weight", W * static_cast<std::size_t>(out->shape.time_freq), &s.t0_w) || !read("t_embedder.mlp.0.bias", W, &s.t0_b) ||
        !read("t_embedder.mlp.2.weight", W * W, &s.t2_w) || !read("t_embedder.mlp.2.bias", W, &s.t2_b) ||
        !read("adaLN_modulation.1.weight", 6 * W * W, &s.ada_w) || !read("adaLN_modulation.1.bias", 6 * W, &s.ada_b) ||
        !read("input_layer.weight", W * static_cast<std::size_t>(out->shape.in_channels), &s.in_w) || !read("input_layer.bias", W, &s.in_b) ||
        !read("out_layer.weight", static_cast<std::size_t>(out->shape.out_channels) * W, &s.out_w) ||
        !read("out_layer.bias", static_cast<std::size_t>(out->shape.out_channels), &s.out_b))
        return false;
    out->load_block = [tf, W, M, C](int layer, DitBlockWeights* b, std::string* err) {
        const std::string p = "blocks." + std::to_string(layer) + ".";
        auto rd = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
            return tf->ReadF32(p + name, count, dst, err);
        };
        return rd("modulation", 6 * W, &b->modulation) && rd("norm2.weight", W, &b->norm2_w) && rd("norm2.bias", W, &b->norm2_b) &&
               rd("self_attn.to_qkv.weight", 3 * W * W, &b->qkv_w) && rd("self_attn.to_qkv.bias", 3 * W, &b->qkv_b) &&
               rd("self_attn.q_rms_norm.gamma", W, &b->q_gamma) && rd("self_attn.k_rms_norm.gamma", W, &b->k_gamma) &&
               rd("self_attn.to_out.weight", W * W, &b->self_out_w) && rd("self_attn.to_out.bias", W, &b->self_out_b) &&
               rd("cross_attn.to_q.weight", W * W, &b->cq_w) && rd("cross_attn.to_q.bias", W, &b->cq_b) &&
               rd("cross_attn.to_kv.weight", 2 * W * C, &b->ckv_w) && rd("cross_attn.to_kv.bias", 2 * W, &b->ckv_b) &&
               rd("cross_attn.q_rms_norm.gamma", W, &b->cq_gamma) && rd("cross_attn.k_rms_norm.gamma", W, &b->ck_gamma) &&
               rd("cross_attn.to_out.weight", W * W, &b->cross_out_w) && rd("cross_attn.to_out.bias", W, &b->cross_out_b) &&
               rd("mlp.mlp.0.weight", M * W, &b->mlp0_w) && rd("mlp.mlp.0.bias", M, &b->mlp0_b) &&
               rd("mlp.mlp.2.weight", W * M, &b->mlp2_w) && rd("mlp.mlp.2.bias", W, &b->mlp2_b);
    };
    return true;
}

DitHostWeights DitHostWeights::Random(const DitShape& shape, std::uint32_t seed, float scale) {
    DitHostWeights w;
    w.shape = shape;
    rng_state = seed * 0x9E3779B97F4A7C15ULL + 7;
    const auto W = static_cast<std::size_t>(shape.width), M = static_cast<std::size_t>(shape.mlp), C = static_cast<std::size_t>(shape.cond_dim);
    fill(&w.stem.t0_w, W * static_cast<std::size_t>(shape.time_freq), scale); fill(&w.stem.t0_b, W, scale);
    fill(&w.stem.t2_w, W * W, scale); fill(&w.stem.t2_b, W, scale);
    fill(&w.stem.ada_w, 6 * W * W, scale); fill(&w.stem.ada_b, 6 * W, scale);
    fill(&w.stem.in_w, W * static_cast<std::size_t>(shape.in_channels), scale); fill(&w.stem.in_b, W, scale);
    fill(&w.stem.out_w, static_cast<std::size_t>(shape.out_channels) * W, scale); fill(&w.stem.out_b, static_cast<std::size_t>(shape.out_channels), scale);
    auto blocks = std::make_shared<std::vector<DitBlockWeights>>(static_cast<std::size_t>(shape.layers));
    for (auto& b : *blocks) {
        fill(&b.modulation, 6 * W, scale);
        b.norm2_w.assign(W, 1.f); fill(&b.norm2_b, W, scale);
        fill(&b.qkv_w, 3 * W * W, scale); fill(&b.qkv_b, 3 * W, scale);
        fill(&b.q_gamma, W, 1.f); fill(&b.k_gamma, W, 1.f);
        fill(&b.self_out_w, W * W, scale); fill(&b.self_out_b, W, scale);
        fill(&b.cq_w, W * W, scale); fill(&b.cq_b, W, scale);
        fill(&b.ckv_w, 2 * W * C, scale); fill(&b.ckv_b, 2 * W, scale);
        fill(&b.cq_gamma, W, 1.f); fill(&b.ck_gamma, W, 1.f);
        fill(&b.cross_out_w, W * W, scale); fill(&b.cross_out_b, W, scale);
        fill(&b.mlp0_w, M * W, scale); fill(&b.mlp0_b, M, scale);
        fill(&b.mlp2_w, W * M, scale); fill(&b.mlp2_b, W, scale);
    }
    w.load_block = [blocks](int layer, DitBlockWeights* out, std::string*) {
        *out = (*blocks)[static_cast<std::size_t>(layer)];
        return true;
    };
    return w;
}

bool dit_forward_cpu(const DitHostWeights& w, const float* x, const std::vector<Coord>& coords, const float* cond,
                     int cond_tokens, float t, std::vector<float>* v, std::map<int, std::vector<float>>* taps,
                     std::string* error, const DitForwardOptions& options) {
    const DitShape& s = w.shape;
    const int N = static_cast<int>(coords.size()), W = s.width, H = s.heads, D = s.head_dim(), M = s.mlp, C = s.cond_dim;
    const std::size_t NW = static_cast<std::size_t>(N) * W;
    if (!w.load_block) {
        if (error) *error = "DiT weights not loaded";
        return false;
    }
    // Timestep embedding and shared modulation: mod = adaLN(SiLU(t_emb)), t_emb = mlp2(SiLU(mlp0(sinus))).
    const std::vector<float> sinus = timestep_embedding(t, s.time_freq);
    std::vector<float> temb(W), temb2(W), mod(6 * static_cast<std::size_t>(W));
    linear(sinus.data(), 1, w.stem.t0_w.data(), w.stem.t0_b.data(), W, s.time_freq, temb.data());
    for (auto& e : temb) e = e / (1.f + std::exp(-e));
    linear(temb.data(), 1, w.stem.t2_w.data(), w.stem.t2_b.data(), W, W, temb2.data());
    for (auto& e : temb2) e = e / (1.f + std::exp(-e));
    linear(temb2.data(), 1, w.stem.ada_w.data(), w.stem.ada_b.data(), 6 * W, W, mod.data());

    std::vector<float> hidden(NW), normed(NW), qkv(static_cast<std::size_t>(N) * 3 * W), attn(NW), proj(NW),
        mlp(static_cast<std::size_t>(N) * M), kv(static_cast<std::size_t>(cond_tokens) * 2 * W), bmod(6 * static_cast<std::size_t>(W));
    linear(x, N, w.stem.in_w.data(), w.stem.in_b.data(), W, s.in_channels, hidden.data());
    // RoPE angle table [N, D/2].
    std::vector<float> angles(static_cast<std::size_t>(N) * (D / 2));
    for (int i = 0; i < N; ++i) rope_angles(coords[static_cast<std::size_t>(i)].x, coords[static_cast<std::size_t>(i)].y, coords[static_cast<std::size_t>(i)].z, angles.data() + static_cast<std::size_t>(i) * (D / 2), D);

    DitBlockWeights b;
    const int layers = options.layer_limit > 0 ? std::min(options.layer_limit, s.layers) : s.layers;
    for (int layer = 0; layer < layers; ++layer) {
        if (!w.load_block(layer, &b, error)) return false;
        for (std::size_t i = 0; i < bmod.size(); ++i) bmod[i] = mod[i] + b.modulation[i];
        const float* shift_msa = bmod.data();
        const float* scale_msa = bmod.data() + W;
        const float* gate_msa = bmod.data() + 2 * W;
        const float* shift_mlp = bmod.data() + 3 * W;
        const float* scale_mlp = bmod.data() + 4 * W;
        const float* gate_mlp = bmod.data() + 5 * W;
        // Self-attention branch.
        layer_norm(hidden.data(), N, W, nullptr, nullptr, DitConfig::block_norm_eps, normed.data());
        modulate(normed.data(), N, W, shift_msa, scale_msa);
        linear(normed.data(), N, b.qkv_w.data(), b.qkv_b.data(), 3 * W, W, qkv.data());
        // q | k | v columns; qk norm + RoPE on q and k.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            float* row = qkv.data() + static_cast<std::size_t>(i) * 3 * W;
            for (int part = 0; part < 2; ++part) {
                float* p = row + static_cast<std::size_t>(part) * W;
                const float* gamma = part == 0 ? b.q_gamma.data() : b.k_gamma.data();
                for (int h = 0; h < H; ++h) {
                    float* vh = p + static_cast<std::size_t>(h) * D;
                    double sq = 0;
                    for (int c = 0; c < D; ++c) sq += static_cast<double>(vh[c]) * vh[c];
                    const float inv = static_cast<float>(std::sqrt(static_cast<double>(D)) / std::max(std::sqrt(sq), 1e-12));
                    for (int c = 0; c < D; ++c) vh[c] = vh[c] * inv * gamma[static_cast<std::size_t>(h) * D + c];
                    apply_rope(vh, angles.data() + static_cast<std::size_t>(i) * (D / 2), D);
                }
            }
        }
        attention(qkv.data(), N, 3 * W, qkv.data() + W, qkv.data() + 2 * W, N, 3 * W, H, D, attn.data(), W);
        linear(attn.data(), N, b.self_out_w.data(), b.self_out_b.data(), W, W, proj.data());
        for (std::size_t i = 0; i < NW; ++i) hidden[i] += gate_msa[i % W] * proj[i];
        // Cross-attention branch (norm2 affine, no modulation).
        layer_norm(hidden.data(), N, W, b.norm2_w.data(), b.norm2_b.data(), DitConfig::block_norm_eps, normed.data());
        linear(normed.data(), N, b.cq_w.data(), b.cq_b.data(), W, W, attn.data());   // q into attn buffer
        qk_norm(attn.data(), N, W, H, b.cq_gamma.data());
        // to_kv rows [0, W) = k, [W, 2W) = v (reshape [cond, 2, H, D]); k gets the per-head gamma.
        float* kd = kv.data();
        float* vd = kv.data() + static_cast<std::size_t>(cond_tokens) * W;
        linear(cond, cond_tokens, b.ckv_w.data(), b.ckv_b.data(), W, C, kd);
        linear(cond, cond_tokens, b.ckv_w.data() + static_cast<std::size_t>(W) * C, b.ckv_b.data() + W, W, C, vd);
        qk_norm(kd, cond_tokens, W, H, b.ck_gamma.data());
        attention(attn.data(), N, W, kd, vd, cond_tokens, W, H, D, proj.data(), W);
        linear(proj.data(), N, b.cross_out_w.data(), b.cross_out_b.data(), W, W, attn.data());
        for (std::size_t i = 0; i < NW; ++i) hidden[i] += attn[i];
        // MLP branch.
        layer_norm(hidden.data(), N, W, nullptr, nullptr, DitConfig::block_norm_eps, normed.data());
        modulate(normed.data(), N, W, shift_mlp, scale_mlp);
        linear(normed.data(), N, b.mlp0_w.data(), b.mlp0_b.data(), M, W, mlp.data());
        for (auto& e : mlp) e = 0.5f * e * (1.f + std::tanh(0.7978845608028654f * (e + 0.044715f * e * e * e)));
        linear(mlp.data(), N, b.mlp2_w.data(), b.mlp2_b.data(), W, M, proj.data());
        for (std::size_t i = 0; i < NW; ++i) hidden[i] += gate_mlp[i % W] * proj[i];
        if (taps && std::find(options.tap_layers.begin(), options.tap_layers.end(), layer) != options.tap_layers.end())
            (*taps)[layer] = hidden;
    }
    layer_norm(hidden.data(), N, W, nullptr, nullptr, DitConfig::final_norm_eps, normed.data());
    v->resize(static_cast<std::size_t>(N) * s.out_channels);
    linear(normed.data(), N, w.stem.out_w.data(), w.stem.out_b.data(), s.out_channels, W, v->data());
    return true;
}

}  // namespace gufo::trellis2
