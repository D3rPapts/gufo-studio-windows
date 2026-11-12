// TripoSG: DINOv2-L weights loader and FP32 CPU reference encoder.
// AI-assisted (Claude Code), 2026-11-12.
#include "dinov2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "src/models/wan22/tensor_file.hpp"

namespace gufo::triposg {
namespace {

using Cfg = DinoConfig;
constexpr int kPatchValues = 3 * Cfg::patch * Cfg::patch;  // 588

// out[t, r] = sum_c in[t, c] * w[r, c] + b[r]  (optionally exact GELU).
void linear(const float* in, int tokens, const float* w, const float* b, int rows, int cols, float* out,
            bool gelu) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float* x = in + static_cast<std::size_t>(t) * static_cast<std::size_t>(cols);
        float* y = out + static_cast<std::size_t>(t) * static_cast<std::size_t>(rows);
        for (int r = 0; r < rows; ++r) {
            const float* wr = w + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) acc += x[c] * wr[c];
            acc += b[r];
            if (gelu) acc = 0.5f * acc * (1.f + std::erf(acc * 0.70710678118654752f));
            y[r] = acc;
        }
    }
}

void layer_norm(const float* x, const float* w, const float* b, int tokens, float* out) {
    for (int t = 0; t < tokens; ++t) {
        const float* row = x + static_cast<std::size_t>(t) * Cfg::width;
        float* y = out + static_cast<std::size_t>(t) * Cfg::width;
        double sum = 0;
        for (int c = 0; c < Cfg::width; ++c) sum += row[c];
        const float mean = static_cast<float>(sum / Cfg::width);
        double sq = 0;
        for (int c = 0; c < Cfg::width; ++c) {
            const float d = row[c] - mean;
            sq += static_cast<double>(d) * d;
        }
        const float rstd = 1.f / std::sqrt(static_cast<float>(sq / Cfg::width) + Cfg::layer_norm_eps);
        for (int c = 0; c < Cfg::width; ++c) y[c] = (row[c] - mean) * rstd * w[c] + b[c];
    }
}

}  // namespace

void dinov2_patch_rows(const float* pixels, std::vector<float>* rows) {
    rows->resize(static_cast<std::size_t>(Cfg::grid) * Cfg::grid * kPatchValues);
    for (int py = 0; py < Cfg::grid; ++py)
        for (int px = 0; px < Cfg::grid; ++px)
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < Cfg::patch; ++y)
                    for (int x = 0; x < Cfg::patch; ++x)
                        (*rows)[(static_cast<std::size_t>(py) * Cfg::grid + px) * kPatchValues +
                                static_cast<std::size_t>((c * Cfg::patch + y) * Cfg::patch + x)] =
                            pixels[(static_cast<std::size_t>(c) * Cfg::image_size + py * Cfg::patch + y) * Cfg::image_size +
                                   px * Cfg::patch + x];
}

bool Dinov2Weights::Load(const std::filesystem::path& file, Dinov2Weights* out, std::string* error) {
    models::wan22::TensorFile tf;
    if (!tf.Open(file, error)) return false;
    tf.set_prefix("dino.");
    auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
        return tf.ReadF32(name, count, dst, error);
    };
    const std::size_t W = Cfg::width;
    if (!read("embeddings.cls_token", W, &out->cls) ||
        !read("embeddings.patch_embeddings.projection.weight", W * kPatchValues, &out->patch_w) ||
        !read("embeddings.patch_embeddings.projection.bias", W, &out->patch_b) ||
        !read("embeddings.position_embeddings_224", static_cast<std::size_t>(Cfg::tokens) * W, &out->pos224) ||
        !read("layernorm.weight", W, &out->final_ln_w) || !read("layernorm.bias", W, &out->final_ln_b))
        return false;
    out->layers.resize(Cfg::layers);
    for (int i = 0; i < Cfg::layers; ++i) {
        Layer& l = out->layers[static_cast<std::size_t>(i)];
        const std::string p = "encoder.layer." + std::to_string(i) + ".";
        if (!read(p + "norm1.weight", W, &l.ln1_w) || !read(p + "norm1.bias", W, &l.ln1_b) ||
            !read(p + "attention.qkv.weight", 3 * W * W, &l.qkv_w) || !read(p + "attention.qkv.bias", 3 * W, &l.qkv_b) ||
            !read(p + "attention.output.dense.weight", W * W, &l.out_w) ||
            !read(p + "attention.output.dense.bias", W, &l.out_b) ||
            !read(p + "layer_scale1.lambda1", W, &l.ls1) ||
            !read(p + "norm2.weight", W, &l.ln2_w) || !read(p + "norm2.bias", W, &l.ln2_b) ||
            !read(p + "mlp.fc1.weight", static_cast<std::size_t>(Cfg::mlp) * W, &l.fc1_w) ||
            !read(p + "mlp.fc1.bias", Cfg::mlp, &l.fc1_b) ||
            !read(p + "mlp.fc2.weight", W * Cfg::mlp, &l.fc2_w) || !read(p + "mlp.fc2.bias", W, &l.fc2_b) ||
            !read(p + "layer_scale2.lambda1", W, &l.ls2))
            return false;
    }
    return out->Validate(error);
}

Dinov2Weights Dinov2Weights::Random(std::uint32_t seed, float scale, int layers) {
    std::uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1;
    auto uniform = [&]() {  // xorshift64*, in [-1, 1)
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        const std::uint64_t r = state * 0x2545F4914F6CDD1DULL;
        return static_cast<float>(static_cast<double>(r >> 11) / 9007199254740992.0) * 2.f - 1.f;
    };
    auto fill = [&](std::vector<float>* v, std::size_t n, float s) {
        v->resize(n);
        for (auto& x : *v) x = uniform() * s;
    };
    auto ones = [](std::vector<float>* v, std::size_t n) { v->assign(n, 1.f); };
    Dinov2Weights w;
    const std::size_t W = Cfg::width;
    fill(&w.cls, W, scale);
    fill(&w.patch_w, W * kPatchValues, scale);
    fill(&w.patch_b, W, scale);
    fill(&w.pos224, static_cast<std::size_t>(Cfg::tokens) * W, scale);
    ones(&w.final_ln_w, W);
    w.final_ln_b.assign(W, 0.f);
    w.layers.resize(static_cast<std::size_t>(layers));
    for (auto& l : w.layers) {
        ones(&l.ln1_w, W); l.ln1_b.assign(W, 0.f);
        fill(&l.qkv_w, 3 * W * W, scale); fill(&l.qkv_b, 3 * W, scale);
        fill(&l.out_w, W * W, scale); fill(&l.out_b, W, scale);
        fill(&l.ls1, W, 1.f);
        ones(&l.ln2_w, W); l.ln2_b.assign(W, 0.f);
        fill(&l.fc1_w, static_cast<std::size_t>(Cfg::mlp) * W, scale); fill(&l.fc1_b, Cfg::mlp, scale);
        fill(&l.fc2_w, W * Cfg::mlp, scale); fill(&l.fc2_b, W, scale);
        fill(&l.ls2, W, 1.f);
    }
    return w;
}

bool Dinov2Weights::Validate(std::string* error) const {
    const std::size_t W = Cfg::width;
    auto bad = [&](const char* what) {
        if (error) *error = std::string("DINOv2 weights: bad ") + what;
        return false;
    };
    if (cls.size() != W || patch_w.size() != W * kPatchValues || patch_b.size() != W ||
        pos224.size() != static_cast<std::size_t>(Cfg::tokens) * W || final_ln_w.size() != W || final_ln_b.size() != W)
        return bad("embeddings");
    for (const auto& l : layers) {
        if (l.ln1_w.size() != W || l.ln1_b.size() != W || l.qkv_w.size() != 3 * W * W || l.qkv_b.size() != 3 * W ||
            l.out_w.size() != W * W || l.out_b.size() != W || l.ls1.size() != W || l.ln2_w.size() != W ||
            l.ln2_b.size() != W || l.fc1_w.size() != static_cast<std::size_t>(Cfg::mlp) * W || l.fc1_b.size() != Cfg::mlp ||
            l.fc2_w.size() != W * Cfg::mlp || l.fc2_b.size() != W || l.ls2.size() != W)
            return bad("layer");
    }
    return true;
}

std::size_t Dinov2Weights::parameter_count() const {
    std::size_t n = cls.size() + patch_w.size() + patch_b.size() + pos224.size() + final_ln_w.size() + final_ln_b.size();
    for (const auto& l : layers)
        n += l.ln1_w.size() + l.ln1_b.size() + l.qkv_w.size() + l.qkv_b.size() + l.out_w.size() + l.out_b.size() +
             l.ls1.size() + l.ln2_w.size() + l.ln2_b.size() + l.fc1_w.size() + l.fc1_b.size() + l.fc2_w.size() +
             l.fc2_b.size() + l.ls2.size();
    return n;
}

bool dinov2_encode_cpu(const Dinov2Weights& w, const float* pixels, std::vector<float>* tokens,
                       std::string* error) {
    if (!w.Validate(error)) return false;
    const int T = Cfg::tokens;
    const std::size_t W = Cfg::width;
    const std::size_t TW = static_cast<std::size_t>(T) * W;
    std::vector<float> rows;
    dinov2_patch_rows(pixels, &rows);
    std::vector<float> hidden(TW), normed(TW), qkv(static_cast<std::size_t>(T) * 3 * W), attn(TW), proj(TW),
        mlp(static_cast<std::size_t>(T) * Cfg::mlp);
    // Embeddings: [CLS; patches] + baked positions.
    for (std::size_t c = 0; c < W; ++c) hidden[c] = w.cls[c];
    linear(rows.data(), T - 1, w.patch_w.data(), w.patch_b.data(), Cfg::width, kPatchValues, hidden.data() + W, false);
    for (std::size_t i = 0; i < TW; ++i) hidden[i] += w.pos224[i];

    std::vector<float> scores(static_cast<std::size_t>(T) * T);
    const float scale = 1.f / std::sqrt(static_cast<float>(Cfg::head_dim));
    for (const auto& l : w.layers) {
        layer_norm(hidden.data(), l.ln1_w.data(), l.ln1_b.data(), T, normed.data());
        linear(normed.data(), T, l.qkv_w.data(), l.qkv_b.data(), 3 * Cfg::width, Cfg::width, qkv.data(), false);
        for (int h = 0; h < Cfg::heads; ++h) {
            const std::size_t qo = static_cast<std::size_t>(h) * Cfg::head_dim;
            const std::size_t ko = W + qo, vo = 2 * W + qo;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < T; ++i) {
                const float* q = qkv.data() + static_cast<std::size_t>(i) * 3 * W + qo;
                float* s = scores.data() + static_cast<std::size_t>(i) * T;
                float m = -INFINITY;
                for (int j = 0; j < T; ++j) {
                    const float* k = qkv.data() + static_cast<std::size_t>(j) * 3 * W + ko;
                    float acc = 0.f;
                    for (int d = 0; d < Cfg::head_dim; ++d) acc += q[d] * k[d];
                    s[j] = acc * scale;
                    m = std::max(m, s[j]);
                }
                float sum = 0.f;
                for (int j = 0; j < T; ++j) { s[j] = std::exp(s[j] - m); sum += s[j]; }
                const float inv = 1.f / sum;
                float* o = attn.data() + static_cast<std::size_t>(i) * W + qo;
                for (int d = 0; d < Cfg::head_dim; ++d) o[d] = 0.f;
                for (int j = 0; j < T; ++j) {
                    const float p = s[j] * inv;
                    const float* v = qkv.data() + static_cast<std::size_t>(j) * 3 * W + vo;
                    for (int d = 0; d < Cfg::head_dim; ++d) o[d] += p * v[d];
                }
            }
        }
        linear(attn.data(), T, l.out_w.data(), l.out_b.data(), Cfg::width, Cfg::width, proj.data(), false);
        for (std::size_t i = 0; i < TW; ++i) hidden[i] += l.ls1[i % W] * proj[i];
        layer_norm(hidden.data(), l.ln2_w.data(), l.ln2_b.data(), T, normed.data());
        linear(normed.data(), T, l.fc1_w.data(), l.fc1_b.data(), Cfg::mlp, Cfg::width, mlp.data(), true);
        linear(mlp.data(), T, l.fc2_w.data(), l.fc2_b.data(), Cfg::width, Cfg::mlp, proj.data(), false);
        for (std::size_t i = 0; i < TW; ++i) hidden[i] += l.ls2[i % W] * proj[i];
    }
    tokens->resize(TW);
    layer_norm(hidden.data(), w.final_ln_w.data(), w.final_ln_b.data(), T, tokens->data());
    return true;
}

}  // namespace gufo::triposg
