// TRELLIS.2: DINOv3 ViT-L/16 weights loader and FP32 CPU reference encoder.
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/dinov3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "src/models/wan22/tensor_file.hpp"

namespace gufo::trellis2 {
namespace {

using Cfg = DinoV3Config;
constexpr int kPatchValues = 3 * Cfg::patch * Cfg::patch;  // 768
constexpr int kPrefix = 1 + Cfg::registers;                 // 5
constexpr double kPi = 3.14159265358979323846;

void linear(const float* in, int tokens, const float* w, const float* b, int rows, int cols, float* out, bool gelu) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float* x = in + static_cast<std::size_t>(t) * cols;
        float* y = out + static_cast<std::size_t>(t) * rows;
        for (int r = 0; r < rows; ++r) {
            const float* wr = w + static_cast<std::size_t>(r) * cols;
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) acc += x[c] * wr[c];
            if (b) acc += b[r];
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
        for (int c = 0; c < Cfg::width; ++c) {
            const float n = (row[c] - mean) * rstd;
            y[c] = w ? n * w[c] + b[c] : n;
        }
    }
}

}  // namespace

void dinov3_patch_rows(const float* pixels, int image, std::vector<float>* rows) {
    const int grid = image / Cfg::patch;
    rows->resize(static_cast<std::size_t>(grid) * grid * kPatchValues);
    for (int py = 0; py < grid; ++py)
        for (int px = 0; px < grid; ++px)
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < Cfg::patch; ++y)
                    for (int x = 0; x < Cfg::patch; ++x)
                        (*rows)[(static_cast<std::size_t>(py) * grid + px) * kPatchValues +
                                static_cast<std::size_t>((c * Cfg::patch + y) * Cfg::patch + x)] =
                            pixels[(static_cast<std::size_t>(c) * image + py * Cfg::patch + y) * image + px * Cfg::patch + x];
}

void dinov3_rope_angles(int grid, std::vector<float>* angles) {
    constexpr int half = Cfg::head_dim / 2;        // 32
    constexpr int per_axis = Cfg::head_dim / 4;    // 16
    angles->resize(static_cast<std::size_t>(grid) * grid * half);
    float inv_freq[per_axis];
    for (int i = 0; i < per_axis; ++i)
        inv_freq[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(Cfg::rope_theta), static_cast<double>(4 * i) / Cfg::head_dim));
    for (int py = 0; py < grid; ++py) {
        const float cy = 2.f * (static_cast<float>(py) + 0.5f) / static_cast<float>(grid) - 1.f;
        for (int px = 0; px < grid; ++px) {
            const float cx = 2.f * (static_cast<float>(px) + 0.5f) / static_cast<float>(grid) - 1.f;
            float* a = angles->data() + (static_cast<std::size_t>(py) * grid + px) * half;
            for (int i = 0; i < per_axis; ++i) {
                a[i] = static_cast<float>(2.0 * kPi * cy * inv_freq[i]);
                a[per_axis + i] = static_cast<float>(2.0 * kPi * cx * inv_freq[i]);
            }
        }
    }
}

void dinov3_apply_rope(float* v, const float* a) {
    constexpr int half = Cfg::head_dim / 2;
    for (int d = 0; d < half; ++d) {
        const float c = std::cos(a[d]), s = std::sin(a[d]);
        const float x1 = v[d], x2 = v[d + half];
        v[d] = x1 * c - x2 * s;
        v[d + half] = x2 * c + x1 * s;
    }
}

bool Dinov3Weights::Load(const std::filesystem::path& file, Dinov3Weights* out, std::string* error) {
    models::wan22::TensorFile tf;
    if (!tf.Open(file, error)) return false;
    auto read = [&](const std::string& name, std::size_t count, std::vector<float>* dst) {
        return tf.ReadF32(name, count, dst, error);
    };
    const std::size_t W = Cfg::width;
    if (!read("embeddings.cls_token", W, &out->cls) || !read("embeddings.register_tokens", Cfg::registers * W, &out->registers) ||
        !read("embeddings.patch_embeddings.weight", W * kPatchValues, &out->patch_w) ||
        !read("embeddings.patch_embeddings.bias", W, &out->patch_b))
        return false;
    out->layers.resize(Cfg::layers);
    for (int i = 0; i < Cfg::layers; ++i) {
        Layer& l = out->layers[static_cast<std::size_t>(i)];
        const std::string p = "layer." + std::to_string(i) + ".";
        if (!read(p + "norm1.weight", W, &l.ln1_w) || !read(p + "norm1.bias", W, &l.ln1_b) ||
            !read(p + "attention.qkv.weight", 3 * W * W, &l.qkv_w) || !read(p + "attention.qkv.bias", 3 * W, &l.qkv_b) ||
            !read(p + "attention.o_proj.weight", W * W, &l.out_w) || !read(p + "attention.o_proj.bias", W, &l.out_b) ||
            !read(p + "layer_scale1.lambda1", W, &l.ls1) || !read(p + "norm2.weight", W, &l.ln2_w) ||
            !read(p + "norm2.bias", W, &l.ln2_b) || !read(p + "mlp.up_proj.weight", static_cast<std::size_t>(Cfg::mlp) * W, &l.fc1_w) ||
            !read(p + "mlp.up_proj.bias", Cfg::mlp, &l.fc1_b) || !read(p + "mlp.down_proj.weight", W * Cfg::mlp, &l.fc2_w) ||
            !read(p + "mlp.down_proj.bias", W, &l.fc2_b) || !read(p + "layer_scale2.lambda1", W, &l.ls2))
            return false;
    }
    return out->Validate(error);
}

Dinov3Weights Dinov3Weights::Random(std::uint32_t seed, float scale, int layers) {
    std::uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1;
    auto uniform = [&]() {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        const std::uint64_t r = state * 0x2545F4914F6CDD1DULL;
        return static_cast<float>(static_cast<double>(r >> 11) / 9007199254740992.0) * 2.f - 1.f;
    };
    auto fill = [&](std::vector<float>* v, std::size_t n, float s) {
        v->resize(n);
        for (auto& x : *v) x = uniform() * s;
    };
    Dinov3Weights w;
    const std::size_t W = Cfg::width;
    fill(&w.cls, W, scale);
    fill(&w.registers, Cfg::registers * W, scale);
    fill(&w.patch_w, W * kPatchValues, scale);
    fill(&w.patch_b, W, scale);
    w.layers.resize(static_cast<std::size_t>(layers));
    for (auto& l : w.layers) {
        l.ln1_w.assign(W, 1.f); l.ln1_b.assign(W, 0.f);
        fill(&l.qkv_w, 3 * W * W, scale); fill(&l.qkv_b, 3 * W, scale);
        std::fill(l.qkv_b.begin() + static_cast<std::ptrdiff_t>(W), l.qkv_b.begin() + static_cast<std::ptrdiff_t>(2 * W), 0.f);
        fill(&l.out_w, W * W, scale); fill(&l.out_b, W, scale);
        fill(&l.ls1, W, 1.f);
        l.ln2_w.assign(W, 1.f); l.ln2_b.assign(W, 0.f);
        fill(&l.fc1_w, static_cast<std::size_t>(Cfg::mlp) * W, scale); fill(&l.fc1_b, Cfg::mlp, scale);
        fill(&l.fc2_w, W * Cfg::mlp, scale); fill(&l.fc2_b, W, scale);
        fill(&l.ls2, W, 1.f);
    }
    return w;
}

bool Dinov3Weights::Validate(std::string* error) const {
    const std::size_t W = Cfg::width;
    auto bad = [&](const char* what) {
        if (error) *error = std::string("DINOv3 weights: bad ") + what;
        return false;
    };
    if (cls.size() != W || registers.size() != Cfg::registers * W || patch_w.size() != W * kPatchValues || patch_b.size() != W)
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

std::size_t Dinov3Weights::parameter_count() const {
    std::size_t n = cls.size() + registers.size() + patch_w.size() + patch_b.size();
    for (const auto& l : layers)
        n += l.ln1_w.size() + l.ln1_b.size() + l.qkv_w.size() + l.qkv_b.size() + l.out_w.size() + l.out_b.size() +
             l.ls1.size() + l.ln2_w.size() + l.ln2_b.size() + l.fc1_w.size() + l.fc1_b.size() + l.fc2_w.size() +
             l.fc2_b.size() + l.ls2.size();
    return n;
}

bool dinov3_encode_cpu(const Dinov3Weights& w, const float* pixels, int image, std::vector<float>* tokens,
                       std::string* error) {
    if (!w.Validate(error)) return false;
    if (image % Cfg::patch != 0 || image <= 0) {
        if (error) *error = "DINOv3: image size must be a multiple of 16";
        return false;
    }
    const int grid = image / Cfg::patch;
    const int P = grid * grid;
    const int T = kPrefix + P;
    const std::size_t W = Cfg::width;
    const std::size_t TW = static_cast<std::size_t>(T) * W;
    std::vector<float> rows, angles;
    dinov3_patch_rows(pixels, image, &rows);
    dinov3_rope_angles(grid, &angles);
    std::vector<float> hidden(TW), normed(TW), qkv(static_cast<std::size_t>(T) * 3 * W), attn(TW), proj(TW),
        mlp(static_cast<std::size_t>(T) * Cfg::mlp);
    std::copy(w.cls.begin(), w.cls.end(), hidden.begin());
    std::copy(w.registers.begin(), w.registers.end(), hidden.begin() + static_cast<std::ptrdiff_t>(W));
    linear(rows.data(), P, w.patch_w.data(), w.patch_b.data(), Cfg::width, kPatchValues, hidden.data() + kPrefix * W, false);

    std::vector<float> scores(static_cast<std::size_t>(T) * T);
    const float scale = 1.f / std::sqrt(static_cast<float>(Cfg::head_dim));
    constexpr int half = Cfg::head_dim / 2;
    for (const auto& l : w.layers) {
        layer_norm(hidden.data(), l.ln1_w.data(), l.ln1_b.data(), T, normed.data());
        linear(normed.data(), T, l.qkv_w.data(), l.qkv_b.data(), 3 * Cfg::width, Cfg::width, qkv.data(), false);
        // RoPE on q and k of the patch tokens only.
#pragma omp parallel for schedule(static)
        for (int p = 0; p < P; ++p) {
            float* row = qkv.data() + static_cast<std::size_t>(kPrefix + p) * 3 * W;
            const float* a = angles.data() + static_cast<std::size_t>(p) * half;
            for (int h = 0; h < Cfg::heads; ++h) {
                dinov3_apply_rope(row + static_cast<std::size_t>(h) * Cfg::head_dim, a);
                dinov3_apply_rope(row + W + static_cast<std::size_t>(h) * Cfg::head_dim, a);
            }
        }
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
                    const float pj = s[j] * inv;
                    const float* v = qkv.data() + static_cast<std::size_t>(j) * 3 * W + vo;
                    for (int d = 0; d < Cfg::head_dim; ++d) o[d] += pj * v[d];
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
    layer_norm(hidden.data(), nullptr, nullptr, T, tokens->data());   // F.layer_norm without affine
    return true;
}

}  // namespace gufo::trellis2
