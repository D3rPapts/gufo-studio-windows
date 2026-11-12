// TripoSG: VAE decoder weight providers and the FP32 CPU reference.
// AI-assisted (Claude Code), 2026-11-12.
#include "vae.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

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
    float next() {
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

void vae_embed_points(const float* points, int count, int freqs, std::vector<float>* out) {
    const int per = 1 + 2 * freqs;
    out->resize(static_cast<std::size_t>(count) * 3 * static_cast<std::size_t>(per));
    for (int n = 0; n < count; ++n) {
        float* row = out->data() + static_cast<std::size_t>(n) * 3 * static_cast<std::size_t>(per);
        const float* p = points + static_cast<std::size_t>(n) * 3;
        // Layout: [x, y, z, sin(f0 x) .. sin(f7 x), sin(f0 y) .., sin(.. z), cos(...) likewise].
        row[0] = p[0]; row[1] = p[1]; row[2] = p[2];
        for (int i = 0; i < 3; ++i)
            for (int f = 0; f < freqs; ++f) {
                const float arg = p[i] * static_cast<float>(1 << f);
                row[3 + i * freqs + f] = std::sin(arg);
                row[3 + 3 * freqs + i * freqs + f] = std::cos(arg);
            }
    }
}

bool VaeHostWeights::FromFile(const std::filesystem::path& file, VaeHostWeights* out, std::string* error) {
    auto tf = std::make_shared<models::wan22::TensorFile>();
    if (!tf->Open(file, error)) return false;
    tf->set_prefix("vae.");
    const VaeShape s;
    out->shape = s;
    const std::size_t W = static_cast<std::size_t>(s.width), M = static_cast<std::size_t>(s.mlp);
    auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
        return tf->ReadF32(name, count, dst, error);
    };
    VaeQueryWeights& qw = out->query;
    const std::string q = "decoder.blocks." + std::to_string(s.layers) + ".";
    if (!read("post_quant.weight", W * static_cast<std::size_t>(s.latent_channels), &out->post_quant_w) ||
        !read("post_quant.bias", W, &out->post_quant_b) ||
        !read(q + "norm2.weight", W, &qw.norm2_w) || !read(q + "norm2.bias", W, &qw.norm2_b) ||
        !read(q + "norm3.weight", W, &qw.norm3_w) || !read(q + "norm3.bias", W, &qw.norm3_b) ||
        !read(q + "attn2.norm_cross.weight", W, &qw.norm_cross_w) || !read(q + "attn2.norm_cross.bias", W, &qw.norm_cross_b) ||
        !read(q + "attn2.to_q.weight", W * W, &qw.q) || !read(q + "attn2.to_k.weight", W * W, &qw.k) ||
        !read(q + "attn2.to_v.weight", W * W, &qw.v) || !read(q + "attn2.to_out.0.weight", W * W, &qw.o_w) ||
        !read(q + "attn2.to_out.0.bias", W, &qw.o_b) ||
        !read(q + "ff.net.0.proj.weight", M * W, &qw.ff1_w) || !read(q + "ff.net.0.proj.bias", M, &qw.ff1_b) ||
        !read(q + "ff.net.2.weight", W * M, &qw.ff2_w) || !read(q + "ff.net.2.bias", W, &qw.ff2_b) ||
        !read("decoder.proj_query.weight", W * static_cast<std::size_t>(s.query_channels()), &qw.proj_query_w) ||
        !read("decoder.proj_query.bias", W, &qw.proj_query_b) ||
        !read("decoder.norm_out.weight", W, &qw.norm_out_w) || !read("decoder.norm_out.bias", W, &qw.norm_out_b) ||
        !read("decoder.proj_out.weight", W, &qw.proj_out_w) || !read("decoder.proj_out.bias", 1, &qw.proj_out_b))
        return false;
    out->load_block = [tf, s](int layer, VaeBlockWeights* b, std::string* err) {
        const std::size_t W = static_cast<std::size_t>(s.width), M = static_cast<std::size_t>(s.mlp);
        const std::string p = "decoder.blocks." + std::to_string(layer) + ".";
        auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
            return tf->ReadF32(p + name, count, dst, err);
        };
        return read("norm1.weight", W, &b->norm1_w) && read("norm1.bias", W, &b->norm1_b) &&
               read("norm3.weight", W, &b->norm3_w) && read("norm3.bias", W, &b->norm3_b) &&
               read("attn1.to_q.weight", W * W, &b->q) && read("attn1.to_k.weight", W * W, &b->k) &&
               read("attn1.to_v.weight", W * W, &b->v) && read("attn1.to_out.0.weight", W * W, &b->o_w) &&
               read("attn1.to_out.0.bias", W, &b->o_b) &&
               read("ff.net.0.proj.weight", M * W, &b->ff1_w) && read("ff.net.0.proj.bias", M, &b->ff1_b) &&
               read("ff.net.2.weight", W * M, &b->ff2_w) && read("ff.net.2.bias", W, &b->ff2_b);
    };
    return true;
}

VaeHostWeights VaeHostWeights::Random(const VaeShape& shape, std::uint32_t seed, float scale) {
    VaeHostWeights w;
    w.shape = shape;
    const std::size_t W = static_cast<std::size_t>(shape.width), M = static_cast<std::size_t>(shape.mlp);
    Rng rng{seed * 0x9E3779B97F4A7C15ULL + 11};
    rng.fill(&w.post_quant_w, W * static_cast<std::size_t>(shape.latent_channels), scale);
    rng.fill(&w.post_quant_b, W, scale);
    VaeQueryWeights& q = w.query;
    q.norm2_w.assign(W, 1.f); q.norm2_b.assign(W, 0.f); q.norm3_w.assign(W, 1.f); q.norm3_b.assign(W, 0.f);
    q.norm_cross_w.assign(W, 1.f); q.norm_cross_b.assign(W, 0.f);
    rng.fill(&q.q, W * W, scale); rng.fill(&q.k, W * W, scale); rng.fill(&q.v, W * W, scale);
    rng.fill(&q.o_w, W * W, scale); rng.fill(&q.o_b, W, scale);
    rng.fill(&q.ff1_w, M * W, scale); rng.fill(&q.ff1_b, M, scale); rng.fill(&q.ff2_w, W * M, scale); rng.fill(&q.ff2_b, W, scale);
    rng.fill(&q.proj_query_w, W * static_cast<std::size_t>(shape.query_channels()), scale); rng.fill(&q.proj_query_b, W, scale);
    q.norm_out_w.assign(W, 1.f); q.norm_out_b.assign(W, 0.f);
    rng.fill(&q.proj_out_w, W, scale); rng.fill(&q.proj_out_b, 1, scale);
    w.load_block = [shape, seed, scale](int layer, VaeBlockWeights* b, std::string*) {
        const std::size_t W = static_cast<std::size_t>(shape.width), M = static_cast<std::size_t>(shape.mlp);
        Rng rng{(static_cast<std::uint64_t>(seed) << 8 | static_cast<std::uint64_t>(layer + 1)) * 0x9E3779B97F4A7C15ULL + 5};
        b->norm1_w.assign(W, 1.f); b->norm1_b.assign(W, 0.f); b->norm3_w.assign(W, 1.f); b->norm3_b.assign(W, 0.f);
        rng.fill(&b->q, W * W, scale); rng.fill(&b->k, W * W, scale); rng.fill(&b->v, W * W, scale);
        rng.fill(&b->o_w, W * W, scale); rng.fill(&b->o_b, W, scale);
        rng.fill(&b->ff1_w, M * W, scale); rng.fill(&b->ff1_b, M, scale); rng.fill(&b->ff2_w, W * M, scale); rng.fill(&b->ff2_b, W, scale);
        return true;
    };
    return w;
}

bool vae_kv_cpu(const VaeHostWeights& w, const float* z, int tokens, std::vector<float>* kv, std::string* error) {
    const VaeShape& s = w.shape;
    const int W = s.width, H = s.heads, D = s.head_dim();
    const std::size_t TW = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(W);
    std::vector<float> hidden(TW), normed(TW), q(TW), k(TW), v(TW), attn(TW), proj(TW),
        mlp(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(s.mlp));
    linear(z, tokens, w.post_quant_w.data(), w.post_quant_b.data(), W, s.latent_channels, hidden.data(), false);
    VaeBlockWeights b;
    for (int layer = 0; layer < s.layers; ++layer) {
        b = VaeBlockWeights{};
        if (!w.load_block(layer, &b, error)) return false;
        layer_norm(hidden.data(), b.norm1_w.data(), b.norm1_b.data(), tokens, W, VaeConfig::norm_eps, normed.data());
        linear(normed.data(), tokens, b.q.data(), nullptr, W, W, q.data(), false);
        linear(normed.data(), tokens, b.k.data(), nullptr, W, W, k.data(), false);
        linear(normed.data(), tokens, b.v.data(), nullptr, W, W, v.data(), false);
        attention(q.data(), tokens, k.data(), v.data(), tokens, H, D, attn.data());
        linear(attn.data(), tokens, b.o_w.data(), b.o_b.data(), W, W, proj.data(), false);
        for (std::size_t i = 0; i < TW; ++i) hidden[i] += proj[i];
        layer_norm(hidden.data(), b.norm3_w.data(), b.norm3_b.data(), tokens, W, VaeConfig::norm_eps, normed.data());
        linear(normed.data(), tokens, b.ff1_w.data(), b.ff1_b.data(), s.mlp, W, mlp.data(), true);
        linear(mlp.data(), tokens, b.ff2_w.data(), b.ff2_b.data(), W, s.mlp, proj.data(), false);
        for (std::size_t i = 0; i < TW; ++i) hidden[i] += proj[i];
    }
    *kv = std::move(hidden);
    return true;
}

bool vae_query_cpu(const VaeHostWeights& w, const float* kv, int tokens, const float* points, int count,
                   std::vector<float>* logits, std::string* error) {
    (void)error;
    const VaeShape& s = w.shape;
    const VaeQueryWeights& qw = w.query;
    const int W = s.width, H = s.heads, D = s.head_dim();
    const std::size_t TW = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(W);
    // Keys/values from the normalised cache (independent of the queries).
    std::vector<float> kvn(TW), k(TW), v(TW);
    layer_norm(kv, qw.norm_cross_w.data(), qw.norm_cross_b.data(), tokens, W, 1e-5f, kvn.data());
    linear(kvn.data(), tokens, qw.k.data(), nullptr, W, W, k.data(), false);
    linear(kvn.data(), tokens, qw.v.data(), nullptr, W, W, v.data(), false);
    const std::size_t PW = static_cast<std::size_t>(count) * static_cast<std::size_t>(W);
    std::vector<float> embed, hidden(PW), normed(PW), q(PW), attn(PW), proj(PW),
        mlp(static_cast<std::size_t>(count) * static_cast<std::size_t>(s.mlp));
    vae_embed_points(points, count, s.freqs, &embed);
    linear(embed.data(), count, qw.proj_query_w.data(), qw.proj_query_b.data(), W, s.query_channels(), hidden.data(), false);
    layer_norm(hidden.data(), qw.norm2_w.data(), qw.norm2_b.data(), count, W, VaeConfig::norm_eps, normed.data());
    linear(normed.data(), count, qw.q.data(), nullptr, W, W, q.data(), false);
    attention(q.data(), count, k.data(), v.data(), tokens, H, D, attn.data());
    linear(attn.data(), count, qw.o_w.data(), qw.o_b.data(), W, W, proj.data(), false);
    for (std::size_t i = 0; i < PW; ++i) hidden[i] += proj[i];
    layer_norm(hidden.data(), qw.norm3_w.data(), qw.norm3_b.data(), count, W, VaeConfig::norm_eps, normed.data());
    linear(normed.data(), count, qw.ff1_w.data(), qw.ff1_b.data(), s.mlp, W, mlp.data(), true);
    linear(mlp.data(), count, qw.ff2_w.data(), qw.ff2_b.data(), W, s.mlp, proj.data(), false);
    for (std::size_t i = 0; i < PW; ++i) hidden[i] += proj[i];
    layer_norm(hidden.data(), qw.norm_out_w.data(), qw.norm_out_b.data(), count, W, 1e-5f, normed.data());
    logits->resize(static_cast<std::size_t>(count));
    for (int n = 0; n < count; ++n) {
        float acc = qw.proj_out_b[0];
        const float* row = normed.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(W);
        for (int c = 0; c < W; ++c) acc += row[c] * qw.proj_out_w[static_cast<std::size_t>(c)];
        (*logits)[static_cast<std::size_t>(n)] = -acc;  // positive inside
    }
    return true;
}

}  // namespace gufo::triposg
