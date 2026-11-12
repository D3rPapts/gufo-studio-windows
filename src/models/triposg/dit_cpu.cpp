// TripoSG: DiT weight providers and the FP32 CPU reference forward.
// AI-assisted (Claude Code), 2026-11-12.
#include "dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "src/models/triposg/scheduler.hpp"
#include "src/models/wan22/tensor_file.hpp"

namespace gufo::triposg {
namespace {

void linear(const float* in, int tokens, const float* w, const float* b, int rows, int cols, float* out, bool gelu) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float* x = in + static_cast<std::size_t>(t) * static_cast<std::size_t>(cols);
        float* y = out + static_cast<std::size_t>(t) * static_cast<std::size_t>(rows);
        for (int r = 0; r < rows; ++r) {
            const float* wr = w + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) acc += x[c] * wr[c];
            if (b) acc += b[r];
            if (gelu) acc = 0.5f * acc * (1.f + std::erf(acc * 0.70710678118654752f));
            y[r] = acc;
        }
    }
}

void layer_norm(const float* x, const float* w, const float* b, int tokens, int width, float eps, float* out) {
    for (int t = 0; t < tokens; ++t) {
        const float* row = x + static_cast<std::size_t>(t) * static_cast<std::size_t>(width);
        float* y = out + static_cast<std::size_t>(t) * static_cast<std::size_t>(width);
        double sum = 0;
        for (int c = 0; c < width; ++c) sum += row[c];
        const float mean = static_cast<float>(sum / width);
        double sq = 0;
        for (int c = 0; c < width; ++c) {
            const float d = row[c] - mean;
            sq += static_cast<double>(d) * d;
        }
        const float rstd = 1.f / std::sqrt(static_cast<float>(sq / width) + eps);
        for (int c = 0; c < width; ++c) y[c] = (row[c] - mean) * rstd * w[c] + b[c];
    }
}

// RMS norm of every head slice of width `dim` in place (rows of heads*dim).
void rms_norm_heads(float* x, int tokens, int heads, int dim, const float* w, float eps) {
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h) {
            float* v = x + (static_cast<std::size_t>(t) * static_cast<std::size_t>(heads) + static_cast<std::size_t>(h)) * static_cast<std::size_t>(dim);
            double sq = 0;
            for (int d = 0; d < dim; ++d) sq += static_cast<double>(v[d]) * v[d];
            const float r = 1.f / std::sqrt(static_cast<float>(sq / dim) + eps);
            for (int d = 0; d < dim; ++d) v[d] = v[d] * r * w[d];
        }
}

// out[T, H*D] = softmax(q k^T / sqrt(D)) v per head; q [T, H*D], k/v [S, H*D].
void attention(const float* q, int T, const float* k, const float* v, int S, int heads, int dim, float* out) {
    const float scale = 1.f / std::sqrt(static_cast<float>(dim));
    const std::size_t stride = static_cast<std::size_t>(heads) * static_cast<std::size_t>(dim);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < T * heads; ++i) {
        const int t = i / heads, h = i % heads;
        std::vector<float> s(static_cast<std::size_t>(S));
        const float* qi = q + static_cast<std::size_t>(t) * stride + static_cast<std::size_t>(h) * static_cast<std::size_t>(dim);
        float m = -INFINITY;
        for (int j = 0; j < S; ++j) {
            const float* kj = k + static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(h) * static_cast<std::size_t>(dim);
            float acc = 0.f;
            for (int d = 0; d < dim; ++d) acc += qi[d] * kj[d];
            s[static_cast<std::size_t>(j)] = acc * scale;
            m = std::max(m, s[static_cast<std::size_t>(j)]);
        }
        float sum = 0.f;
        for (int j = 0; j < S; ++j) { s[static_cast<std::size_t>(j)] = std::exp(s[static_cast<std::size_t>(j)] - m); sum += s[static_cast<std::size_t>(j)]; }
        const float inv = 1.f / sum;
        float* o = out + static_cast<std::size_t>(t) * stride + static_cast<std::size_t>(h) * static_cast<std::size_t>(dim);
        for (int d = 0; d < dim; ++d) o[d] = 0.f;
        for (int j = 0; j < S; ++j) {
            const float p = s[static_cast<std::size_t>(j)] * inv;
            const float* vj = v + static_cast<std::size_t>(j) * stride + static_cast<std::size_t>(h) * static_cast<std::size_t>(dim);
            for (int d = 0; d < dim; ++d) o[d] += p * vj[d];
        }
    }
}

struct Rng {
    std::uint64_t state;
    float next() {  // xorshift64*, uniform [-1, 1)
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        const std::uint64_t r = state * 0x2545F4914F6CDD1DULL;
        return static_cast<float>(static_cast<double>(r >> 11) / 9007199254740992.0) * 2.f - 1.f;
    }
    void fill(std::vector<float>* v, std::size_t n, float s) {
        v->resize(n);
        for (auto& x : *v) x = next() * s;
    }
};

}  // namespace

bool DitHostWeights::FromFile(const std::filesystem::path& file, DitHostWeights* out, std::string* error) {
    auto tf = std::make_shared<models::wan22::TensorFile>();
    if (!tf->Open(file, error)) return false;
    tf->set_prefix("dit.");
    const DitShape s;
    out->shape = s;
    const std::size_t W = static_cast<std::size_t>(s.width);
    auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
        return tf->ReadF32(name, count, dst, error);
    };
    DitStemWeights& st = out->stem;
    if (!read("proj_in.weight", W * static_cast<std::size_t>(s.in_channels), &st.proj_in_w) || !read("proj_in.bias", W, &st.proj_in_b) ||
        !read("time_proj.linear_1.weight", 4 * W * W, &st.time1_w) || !read("time_proj.linear_1.bias", 4 * W, &st.time1_b) ||
        !read("time_proj.linear_2.weight", W * 4 * W, &st.time2_w) || !read("time_proj.linear_2.bias", W, &st.time2_b) ||
        !read("norm_out.weight", W, &st.norm_out_w) || !read("norm_out.bias", W, &st.norm_out_b) ||
        !read("proj_out.weight", static_cast<std::size_t>(s.in_channels) * W, &st.proj_out_w) ||
        !read("proj_out.bias", static_cast<std::size_t>(s.in_channels), &st.proj_out_b))
        return false;
    out->load_block = [tf, s](int layer, DitBlockWeights* b, std::string* err) {
        const std::size_t W = static_cast<std::size_t>(s.width);
        const std::size_t D = static_cast<std::size_t>(s.head_dim());
        const std::string p = "blocks." + std::to_string(layer) + ".";
        auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
            return tf->ReadF32(p + name, count, dst, err);
        };
        bool ok = read("norm1.weight", W, &b->norm1_w) && read("norm1.bias", W, &b->norm1_b) &&
                  read("norm2.weight", W, &b->norm2_w) && read("norm2.bias", W, &b->norm2_b) &&
                  read("norm3.weight", W, &b->norm3_w) && read("norm3.bias", W, &b->norm3_b) &&
                  read("attn1.to_q.weight", W * W, &b->q1) && read("attn1.to_k.weight", W * W, &b->k1) &&
                  read("attn1.to_v.weight", W * W, &b->v1) && read("attn1.to_out.0.weight", W * W, &b->o1_w) &&
                  read("attn1.to_out.0.bias", W, &b->o1_b) && read("attn1.norm_q.weight", D, &b->nq1) &&
                  read("attn1.norm_k.weight", D, &b->nk1) &&
                  read("attn2.to_q.weight", W * W, &b->q2) && read("attn2.to_k.weight", W * static_cast<std::size_t>(s.cross_dim), &b->k2) &&
                  read("attn2.to_v.weight", W * static_cast<std::size_t>(s.cross_dim), &b->v2) &&
                  read("attn2.to_out.0.weight", W * W, &b->o2_w) && read("attn2.to_out.0.bias", W, &b->o2_b) &&
                  read("attn2.norm_q.weight", D, &b->nq2) && read("attn2.norm_k.weight", D, &b->nk2) &&
                  read("ff.net.0.proj.weight", static_cast<std::size_t>(s.mlp) * W, &b->ff1_w) &&
                  read("ff.net.0.proj.bias", static_cast<std::size_t>(s.mlp), &b->ff1_b) &&
                  read("ff.net.2.weight", W * static_cast<std::size_t>(s.mlp), &b->ff2_w) && read("ff.net.2.bias", W, &b->ff2_b);
        if (ok && s.has_skip(layer))
            ok = read("skip_linear.weight", W * 2 * W, &b->skip_w) && read("skip_linear.bias", W, &b->skip_b) &&
                 read("skip_norm.weight", W, &b->skip_norm_w) && read("skip_norm.bias", W, &b->skip_norm_b);
        return ok;
    };
    return true;
}

DitHostWeights DitHostWeights::Random(const DitShape& shape, std::uint32_t seed, float scale) {
    DitHostWeights w;
    w.shape = shape;
    const std::size_t W = static_cast<std::size_t>(shape.width);
    Rng rng{seed * 0x9E3779B97F4A7C15ULL + 7};
    rng.fill(&w.stem.proj_in_w, W * static_cast<std::size_t>(shape.in_channels), scale);
    rng.fill(&w.stem.proj_in_b, W, scale);
    rng.fill(&w.stem.time1_w, 4 * W * W, scale);
    rng.fill(&w.stem.time1_b, 4 * W, scale);
    rng.fill(&w.stem.time2_w, W * 4 * W, scale);
    rng.fill(&w.stem.time2_b, W, scale);
    w.stem.norm_out_w.assign(W, 1.f);
    w.stem.norm_out_b.assign(W, 0.f);
    rng.fill(&w.stem.proj_out_w, static_cast<std::size_t>(shape.in_channels) * W, scale);
    rng.fill(&w.stem.proj_out_b, static_cast<std::size_t>(shape.in_channels), scale);
    // Blocks are generated deterministically per layer from (seed, layer).
    w.load_block = [shape, seed, scale](int layer, DitBlockWeights* b, std::string*) {
        const std::size_t W = static_cast<std::size_t>(shape.width);
        const std::size_t D = static_cast<std::size_t>(shape.head_dim());
        const std::size_t C = static_cast<std::size_t>(shape.cross_dim);
        const std::size_t M = static_cast<std::size_t>(shape.mlp);
        Rng rng{(static_cast<std::uint64_t>(seed) << 8 | static_cast<std::uint64_t>(layer + 1)) * 0x9E3779B97F4A7C15ULL + 3};
        b->norm1_w.assign(W, 1.f); b->norm1_b.assign(W, 0.f);
        b->norm2_w.assign(W, 1.f); b->norm2_b.assign(W, 0.f);
        b->norm3_w.assign(W, 1.f); b->norm3_b.assign(W, 0.f);
        rng.fill(&b->q1, W * W, scale); rng.fill(&b->k1, W * W, scale); rng.fill(&b->v1, W * W, scale);
        rng.fill(&b->o1_w, W * W, scale); rng.fill(&b->o1_b, W, scale);
        b->nq1.assign(D, 1.f); b->nk1.assign(D, 1.f);
        rng.fill(&b->q2, W * W, scale); rng.fill(&b->k2, W * C, scale); rng.fill(&b->v2, W * C, scale);
        rng.fill(&b->o2_w, W * W, scale); rng.fill(&b->o2_b, W, scale);
        b->nq2.assign(D, 1.f); b->nk2.assign(D, 1.f);
        rng.fill(&b->ff1_w, M * W, scale); rng.fill(&b->ff1_b, M, scale);
        rng.fill(&b->ff2_w, W * M, scale); rng.fill(&b->ff2_b, W, scale);
        if (shape.has_skip(layer)) {
            rng.fill(&b->skip_w, W * 2 * W, scale); rng.fill(&b->skip_b, W, scale);
            b->skip_norm_w.assign(W, 1.f); b->skip_norm_b.assign(W, 0.f);
        }
        return true;
    };
    return w;
}

bool dit_forward_cpu(const DitHostWeights& w, const float* x, int tokens, const float* cond, int cond_tokens, float t,
                     std::vector<float>* v, std::map<int, std::vector<float>>* taps, std::string* error,
                     const DitForwardOptions& options) {
    const DitShape& s = w.shape;
    const int W = s.width, H = s.heads, D = s.head_dim();
    const int N = tokens + 1;  // time token + latents
    const std::size_t NW = static_cast<std::size_t>(N) * static_cast<std::size_t>(W);
    std::vector<float> hidden(NW), normed(NW), q(NW), k(NW), vv(NW), attn(NW), proj(NW),
        mlp(static_cast<std::size_t>(N) * static_cast<std::size_t>(s.mlp));
    // Time token: sinusoid -> linear_1 -> GELU -> linear_2.
    const std::vector<float> sinus = timestep_embedding(t, W);
    std::vector<float> time_hidden(4 * static_cast<std::size_t>(W)), temb(static_cast<std::size_t>(W));
    linear(sinus.data(), 1, w.stem.time1_w.data(), w.stem.time1_b.data(), 4 * W, W, time_hidden.data(), true);
    linear(time_hidden.data(), 1, w.stem.time2_w.data(), w.stem.time2_b.data(), W, 4 * W, temb.data(), false);
    std::copy(temb.begin(), temb.end(), hidden.begin());
    linear(x, tokens, w.stem.proj_in_w.data(), w.stem.proj_in_b.data(), W, s.in_channels, hidden.data() + W, false);
    // Cross keys/values are recomputed per block from cond (weights differ per block).
    std::vector<float> ck(static_cast<std::size_t>(cond_tokens) * static_cast<std::size_t>(W)), cv(ck.size());
    std::vector<std::vector<float>> skips;
    DitBlockWeights b;
    for (int layer = 0; layer < s.layers; ++layer) {
        b = DitBlockWeights{};
        if (!w.load_block(layer, &b, error)) return false;
        if (s.has_skip(layer)) {
            if (skips.empty()) {
                if (error) *error = "skip stack underflow";
                return false;
            }
            const std::vector<float> skip = std::move(skips.back());
            skips.pop_back();
            std::vector<float> cat(static_cast<std::size_t>(N) * 2 * static_cast<std::size_t>(W));
            for (int n = 0; n < N; ++n) {
                std::copy(skip.begin() + static_cast<std::ptrdiff_t>(n) * W, skip.begin() + static_cast<std::ptrdiff_t>(n + 1) * W,
                          cat.begin() + static_cast<std::ptrdiff_t>(n) * 2 * W);
                std::copy(hidden.begin() + static_cast<std::ptrdiff_t>(n) * W, hidden.begin() + static_cast<std::ptrdiff_t>(n + 1) * W,
                          cat.begin() + static_cast<std::ptrdiff_t>(n) * 2 * W + W);
            }
            linear(cat.data(), N, b.skip_w.data(), b.skip_b.data(), W, 2 * W, proj.data(), false);
            layer_norm(proj.data(), b.skip_norm_w.data(), b.skip_norm_b.data(), N, W, DitConfig::norm_eps, hidden.data());
        }
        // Self-attention.
        layer_norm(hidden.data(), b.norm1_w.data(), b.norm1_b.data(), N, W, DitConfig::norm_eps, normed.data());
        linear(normed.data(), N, b.q1.data(), nullptr, W, W, q.data(), false);
        linear(normed.data(), N, b.k1.data(), nullptr, W, W, k.data(), false);
        linear(normed.data(), N, b.v1.data(), nullptr, W, W, vv.data(), false);
        rms_norm_heads(q.data(), N, H, D, b.nq1.data(), DitConfig::qk_norm_eps);
        rms_norm_heads(k.data(), N, H, D, b.nk1.data(), DitConfig::qk_norm_eps);
        attention(q.data(), N, k.data(), vv.data(), N, H, D, attn.data());
        linear(attn.data(), N, b.o1_w.data(), b.o1_b.data(), W, W, proj.data(), false);
        for (std::size_t i = 0; i < NW; ++i) hidden[i] += proj[i];
        // Cross-attention on the image tokens.
        layer_norm(hidden.data(), b.norm2_w.data(), b.norm2_b.data(), N, W, DitConfig::norm_eps, normed.data());
        linear(normed.data(), N, b.q2.data(), nullptr, W, W, q.data(), false);
        linear(cond, cond_tokens, b.k2.data(), nullptr, W, s.cross_dim, ck.data(), false);
        linear(cond, cond_tokens, b.v2.data(), nullptr, W, s.cross_dim, cv.data(), false);
        rms_norm_heads(q.data(), N, H, D, b.nq2.data(), DitConfig::qk_norm_eps);
        rms_norm_heads(ck.data(), cond_tokens, H, D, b.nk2.data(), DitConfig::qk_norm_eps);
        attention(q.data(), N, ck.data(), cv.data(), cond_tokens, H, D, attn.data());
        linear(attn.data(), N, b.o2_w.data(), b.o2_b.data(), W, W, proj.data(), false);
        for (std::size_t i = 0; i < NW; ++i) hidden[i] += proj[i];
        // FFN.
        layer_norm(hidden.data(), b.norm3_w.data(), b.norm3_b.data(), N, W, DitConfig::norm_eps, normed.data());
        linear(normed.data(), N, b.ff1_w.data(), b.ff1_b.data(), s.mlp, W, mlp.data(), true);
        linear(mlp.data(), N, b.ff2_w.data(), b.ff2_b.data(), W, s.mlp, proj.data(), false);
        for (std::size_t i = 0; i < NW; ++i) hidden[i] += proj[i];
        if (layer <= s.skip_last_push()) skips.push_back(hidden);
        if (taps && std::find(options.tap_layers.begin(), options.tap_layers.end(), layer) != options.tap_layers.end())
            (*taps)[layer] = hidden;
    }
    layer_norm(hidden.data(), w.stem.norm_out_w.data(), w.stem.norm_out_b.data(), N, W, DitConfig::out_norm_eps, normed.data());
    v->resize(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(s.in_channels));
    linear(normed.data() + W, tokens, w.stem.proj_out_w.data(), w.stem.proj_out_b.data(), s.in_channels, W, v->data(), false);
    return true;
}

}  // namespace gufo::triposg
