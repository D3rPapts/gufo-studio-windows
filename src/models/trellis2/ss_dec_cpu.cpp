// TRELLIS.2: sparse-structure decoder, weights loader and FP32 CPU reference (see ss_dec.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/ss_dec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "src/models/wan22/tensor_file.hpp"

namespace gufo::trellis2 {
namespace {

std::uint64_t rng_state;
float uniform() {
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    const std::uint64_t r = rng_state * 0x2545F4914F6CDD1DULL;
    return static_cast<float>(static_cast<double>(r >> 11) / 9007199254740992.0) * 2.f - 1.f;
}

SsDecConv random_conv(int co, int ci, float scale) {
    SsDecConv c;
    c.co = co; c.ci = ci;
    c.w.resize(static_cast<std::size_t>(co) * 27 * ci);
    for (auto& x : c.w) x = uniform() * scale / std::sqrt(static_cast<float>(27 * ci)) * 5.f;
    c.b.resize(static_cast<std::size_t>(co));
    for (auto& x : c.b) x = uniform() * scale;
    return c;
}
SsDecNorm random_norm(int c) {
    SsDecNorm n;
    n.w.assign(static_cast<std::size_t>(c), 1.f);
    n.b.resize(static_cast<std::size_t>(c));
    for (auto& x : n.b) x = uniform() * 0.1f;
    return n;
}

bool load_conv(models::wan22::TensorFile& tf, const std::string& name, SsDecConv* c, std::string* error) {
    const auto shape = tf.Shape(name + ".weight");
    if (shape.size() != 2 || shape[1] % 27 != 0) {
        if (error) *error = "ss_dec: " + name + ".weight is not [Co, 27*Ci]";
        return false;
    }
    c->co = static_cast<int>(shape[0]);
    c->ci = static_cast<int>(shape[1] / 27);
    return tf.ReadF32(name + ".weight", shape[0] * shape[1], &c->w, error) && tf.ReadF32(name + ".bias", shape[0], &c->b, error);
}
bool load_norm(models::wan22::TensorFile& tf, const std::string& name, int c, SsDecNorm* n, std::string* error) {
    return tf.ReadF32(name + ".weight", static_cast<std::size_t>(c), &n->w, error) &&
           tf.ReadF32(name + ".bias", static_cast<std::size_t>(c), &n->b, error);
}
bool load_res(models::wan22::TensorFile& tf, const std::string& base, int c, SsDecResBlock* r, std::string* error) {
    return load_norm(tf, base + ".norm1", c, &r->norm1, error) && load_norm(tf, base + ".norm2", c, &r->norm2, error) &&
           load_conv(tf, base + ".conv1", &r->conv1, error) && load_conv(tf, base + ".conv2", &r->conv2, error);
}

void res_block(const SsDecResBlock& r, const float* x, int res, int c, float* out, std::vector<float>& h, std::vector<float>& h2) {
    const std::size_t V = static_cast<std::size_t>(res) * res * res;
    h.assign(x, x + V * c);
    channel_norm_silu(h.data(), static_cast<int>(V), c, r.norm1.w.data(), r.norm1.b.data(), SsDecConfig::norm_eps, true);
    h2.resize(V * c);
    dense_conv3(h.data(), res, c, r.conv1.w.data(), r.conv1.b.data(), c, h2.data());
    channel_norm_silu(h2.data(), static_cast<int>(V), c, r.norm2.w.data(), r.norm2.b.data(), SsDecConfig::norm_eps, true);
    dense_conv3(h2.data(), res, c, r.conv2.w.data(), r.conv2.b.data(), c, out);
    for (std::size_t i = 0; i < V * c; ++i) out[i] += x[i];
}

}  // namespace

void dense_conv3(const float* in, int res, int ci, const float* w, const float* b, int co, float* out) {
    const std::size_t k = static_cast<std::size_t>(27) * ci;
#pragma omp parallel for schedule(static)
    for (int vx = 0; vx < res * res * res; ++vx) {
        const int x = vx / (res * res), y = (vx / res) % res, z = vx % res;
        std::vector<float> col(k, 0.f);
        for (int t = 0; t < 27; ++t) {
            int dx, dy, dz;
            tap_offset(t, &dx, &dy, &dz);
            const int nx = x + dx, ny = y + dy, nz = z + dz;
            if (nx < 0 || ny < 0 || nz < 0 || nx >= res || ny >= res || nz >= res) continue;
            std::memcpy(col.data() + static_cast<std::size_t>(t) * ci, in + (static_cast<std::size_t>(nx) * res * res + static_cast<std::size_t>(ny) * res + nz) * ci,
                        sizeof(float) * static_cast<std::size_t>(ci));
        }
        float* o = out + static_cast<std::size_t>(vx) * co;
        for (int oc = 0; oc < co; ++oc) {
            const float* wr = w + static_cast<std::size_t>(oc) * k;
            double acc = b ? b[oc] : 0.0;
            for (std::size_t j = 0; j < k; ++j) acc += static_cast<double>(col[j]) * wr[j];
            o[oc] = static_cast<float>(acc);
        }
    }
}

void channel_norm_silu(float* x, int voxels, int channels, const float* w, const float* b, float eps, bool silu) {
#pragma omp parallel for schedule(static)
    for (int v = 0; v < voxels; ++v) {
        float* row = x + static_cast<std::size_t>(v) * channels;
        double mean = 0;
        for (int c = 0; c < channels; ++c) mean += row[c];
        mean /= channels;
        double var = 0;
        for (int c = 0; c < channels; ++c) var += (row[c] - mean) * (row[c] - mean);
        var /= channels;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < channels; ++c) {
            float y = static_cast<float>((row[c] - mean) * inv) * w[c] + b[c];
            if (silu) y = y / (1.f + std::exp(-y));
            row[c] = y;
        }
    }
}

void pixel_shuffle3(const float* in, int res, int channels, float* out) {
    const int R2 = 2 * res;
    for (int x = 0; x < res; ++x)
        for (int y = 0; y < res; ++y)
            for (int z = 0; z < res; ++z) {
                const float* src = in + ((static_cast<std::size_t>(x) * res + y) * res + z) * 8 * channels;
                for (int s = 0; s < 8; ++s) {
                    const int sx = s >> 2, sy = (s >> 1) & 1, sz = s & 1;
                    float* dst = out + ((static_cast<std::size_t>(2 * x + sx) * R2 + (2 * y + sy)) * R2 + (2 * z + sz)) * channels;
                    for (int c = 0; c < channels; ++c) dst[c] = src[static_cast<std::size_t>(c) * 8 + s];
                }
            }
}

bool SsDecWeights::Load(const std::filesystem::path& file, SsDecWeights* out, std::string* error) {
    models::wan22::TensorFile tf;
    if (!tf.Open(file, error)) return false;
    if (!tf.Has("input_layer.weight") || !tf.Has("out_layer.2.weight")) {
        if (error) *error = "not a converted ss_dec file: " + file.string();
        return false;
    }
    const auto& ch = out->shape.channels;
    if (!load_conv(tf, "input_layer", &out->input, error)) return false;
    out->middle.resize(static_cast<std::size_t>(out->shape.middle_blocks));
    for (int i = 0; i < out->shape.middle_blocks; ++i)
        if (!load_res(tf, "middle_block." + std::to_string(i), ch[0], &out->middle[static_cast<std::size_t>(i)], error)) return false;
    // blocks: 0,1 res ch0 ; 2 up ; 3,4 res ch1 ; 5 up ; 6,7 res ch2
    out->level0.resize(2); out->level1.resize(2); out->level2.resize(2);
    if (!load_res(tf, "blocks.0", ch[0], &out->level0[0], error) || !load_res(tf, "blocks.1", ch[0], &out->level0[1], error) ||
        !load_conv(tf, "blocks.2.conv", &out->up0, error) ||
        !load_res(tf, "blocks.3", ch[1], &out->level1[0], error) || !load_res(tf, "blocks.4", ch[1], &out->level1[1], error) ||
        !load_conv(tf, "blocks.5.conv", &out->up1, error) ||
        !load_res(tf, "blocks.6", ch[2], &out->level2[0], error) || !load_res(tf, "blocks.7", ch[2], &out->level2[1], error) ||
        !load_norm(tf, "out_layer.0", ch[2], &out->out_norm, error) || !load_conv(tf, "out_layer.2", &out->out, error))
        return false;
    if (out->input.ci != out->shape.latent_channels || out->input.co != ch[0] || out->up0.co != 8 * ch[1] || out->up1.co != 8 * ch[2] || out->out.co != 1) {
        if (error) *error = "ss_dec: unexpected channel layout";
        return false;
    }
    return true;
}

SsDecWeights SsDecWeights::Random(const SsDecShape& shape, std::uint32_t seed, float scale) {
    rng_state = seed * 0x9E3779B97F4A7C15ULL + 3;
    SsDecWeights w;
    w.shape = shape;
    const auto& ch = shape.channels;
    auto res = [&](int c) {
        SsDecResBlock r;
        r.norm1 = random_norm(c); r.norm2 = random_norm(c);
        r.conv1 = random_conv(c, c, scale); r.conv2 = random_conv(c, c, scale);
        return r;
    };
    w.input = random_conv(ch[0], shape.latent_channels, scale);
    for (int i = 0; i < shape.middle_blocks; ++i) w.middle.push_back(res(ch[0]));
    for (int i = 0; i < shape.res_blocks; ++i) w.level0.push_back(res(ch[0]));
    w.up0 = random_conv(8 * ch[1], ch[0], scale);
    for (int i = 0; i < shape.res_blocks; ++i) w.level1.push_back(res(ch[1]));
    w.up1 = random_conv(8 * ch[2], ch[1], scale);
    for (int i = 0; i < shape.res_blocks; ++i) w.level2.push_back(res(ch[2]));
    w.out_norm = random_norm(ch[2]);
    w.out = random_conv(1, ch[2], scale);
    return w;
}

std::size_t SsDecWeights::parameter_count() const {
    auto conv = [](const SsDecConv& c) { return c.w.size() + c.b.size(); };
    auto res = [&](const SsDecResBlock& r) { return conv(r.conv1) + conv(r.conv2) + r.norm1.w.size() + r.norm1.b.size() + r.norm2.w.size() + r.norm2.b.size(); };
    std::size_t n = conv(input) + conv(up0) + conv(up1) + conv(out) + out_norm.w.size() + out_norm.b.size();
    for (const auto& r : middle) n += res(r);
    for (const auto& r : level0) n += res(r);
    for (const auto& r : level1) n += res(r);
    for (const auto& r : level2) n += res(r);
    return n;
}

bool ss_dec_forward_cpu(const SsDecWeights& w, const float* latent, std::vector<float>* logits, std::string* error) {
    const SsDecShape& s = w.shape;
    const auto& ch = s.channels;
    int res = s.latent_res;
    std::size_t V = static_cast<std::size_t>(res) * res * res;
    std::vector<float> a(V * ch[0]), b, h, h2;
    dense_conv3(latent, res, s.latent_channels, w.input.w.data(), w.input.b.data(), ch[0], a.data());
    b.resize(V * ch[0]);
    for (const auto& r : w.middle) { res_block(r, a.data(), res, ch[0], b.data(), h, h2); a.swap(b); }
    for (const auto& r : w.level0) { res_block(r, a.data(), res, ch[0], b.data(), h, h2); a.swap(b); }
    // up 512 -> 128 at 32^3
    b.resize(V * 8 * ch[1]);
    dense_conv3(a.data(), res, ch[0], w.up0.w.data(), w.up0.b.data(), 8 * ch[1], b.data());
    res *= 2; V *= 8;
    a.resize(V * ch[1]);
    pixel_shuffle3(b.data(), res / 2, ch[1], a.data());
    b.resize(V * ch[1]);
    for (const auto& r : w.level1) { res_block(r, a.data(), res, ch[1], b.data(), h, h2); a.swap(b); }
    // up 128 -> 32 at 64^3
    b.resize(V * 8 * ch[2]);
    dense_conv3(a.data(), res, ch[1], w.up1.w.data(), w.up1.b.data(), 8 * ch[2], b.data());
    res *= 2; V *= 8;
    a.resize(V * ch[2]);
    pixel_shuffle3(b.data(), res / 2, ch[2], a.data());
    b.resize(V * ch[2]);
    for (const auto& r : w.level2) { res_block(r, a.data(), res, ch[2], b.data(), h, h2); a.swap(b); }
    channel_norm_silu(a.data(), static_cast<int>(V), ch[2], w.out_norm.w.data(), w.out_norm.b.data(), SsDecConfig::norm_eps, true);
    logits->resize(V);
    dense_conv3(a.data(), res, ch[2], w.out.w.data(), w.out.b.data(), 1, logits->data());
    if (error) error->clear();
    return true;
}

std::vector<Coord> occupancy_to_coords(const float* logits, int res64, int target) {
    std::vector<Coord> coords;
    const int ratio = res64 / target;
    for (int x = 0; x < target; ++x)
        for (int y = 0; y < target; ++y)
            for (int z = 0; z < target; ++z) {
                bool active = false;
                for (int dx = 0; dx < ratio && !active; ++dx)
                    for (int dy = 0; dy < ratio && !active; ++dy)
                        for (int dz = 0; dz < ratio && !active; ++dz)
                            active = logits[(static_cast<std::size_t>(x * ratio + dx) * res64 + (y * ratio + dy)) * res64 + (z * ratio + dz)] > 0.f;
                if (active) coords.push_back({x, y, z});
            }
    return coords;
}

}  // namespace gufo::trellis2
