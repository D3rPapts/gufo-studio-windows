// TRELLIS.2: sparse decoders, weights loader and FP32 CPU reference (see sparse_dec.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/sparse_dec.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "src/models/wan22/tensor_file.hpp"

namespace gufo::trellis2 {
namespace {

void linear(const float* in, int rows, const float* w, const float* b, int out_cols, int in_cols, float* out) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* x = in + static_cast<std::size_t>(r) * in_cols;
        float* y = out + static_cast<std::size_t>(r) * out_cols;
        for (int o = 0; o < out_cols; ++o) {
            const float* wr = w + static_cast<std::size_t>(o) * in_cols;
            double acc = b ? b[o] : 0.0;
            for (int c = 0; c < in_cols; ++c) acc += static_cast<double>(x[c]) * wr[c];
            y[o] = static_cast<float>(acc);
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

struct Level {
    std::vector<Coord> coords;
    VoxelHashMap map;
    std::vector<std::int32_t> table;
    void Build(std::vector<Coord> c) {
        coords = std::move(c);
        map.Build(coords);
        table = build_neighbour_table(coords, map);
    }
    int n() const { return static_cast<int>(coords.size()); }
};

}  // namespace

bool SparseDecHostWeights::FromFile(const std::filesystem::path& file, SparseDecHostWeights* out, std::string* error) {
    auto tf = std::make_shared<models::wan22::TensorFile>();
    if (!tf->Open(file, error)) return false;
    if (!tf->Has("from_latent.weight") || !tf->Has("output_layer.weight")) {
        if (error) *error = "not a converted TRELLIS.2 sparse decoder: " + file.string();
        return false;
    }
    SparseDecShape& s = out->shape;
    const auto from = tf->Shape("from_latent.weight"), outs = tf->Shape("output_layer.weight");
    s.latent_channels = static_cast<int>(from[1]);
    s.outputs = static_cast<int>(outs[0]);
    s.pred_subdiv = tf->Has("blocks.0.4.to_subdiv.weight");
    if (!tf->ReadF32("from_latent.weight", from[0] * from[1], &out->from_w, error) || !tf->ReadF32("from_latent.bias", from[0], &out->from_b, error) ||
        !tf->ReadF32("output_layer.weight", outs[0] * outs[1], &out->out_w, error) || !tf->ReadF32("output_layer.bias", outs[0], &out->out_b, error))
        return false;
    const SparseDecShape shape = s;
    out->load_block = [tf, shape](int level, int j, ConvNeXtWeights* b, std::string* err) {
        const std::size_t c = static_cast<std::size_t>(shape.channels[level]);
        const std::string p = "blocks." + std::to_string(level) + "." + std::to_string(j) + ".";
        return tf->ReadF32(p + "conv.weight", c * 27 * c, &b->conv_w, err) && tf->ReadF32(p + "conv.bias", c, &b->conv_b, err) &&
               tf->ReadF32(p + "norm.weight", c, &b->norm_w, err) && tf->ReadF32(p + "norm.bias", c, &b->norm_b, err) &&
               tf->ReadF32(p + "mlp.0.weight", 4 * c * c, &b->mlp0_w, err) && tf->ReadF32(p + "mlp.0.bias", 4 * c, &b->mlp0_b, err) &&
               tf->ReadF32(p + "mlp.2.weight", c * 4 * c, &b->mlp2_w, err) && tf->ReadF32(p + "mlp.2.bias", c, &b->mlp2_b, err);
    };
    out->load_up = [tf, shape](int level, C2SWeights* u, std::string* err) {
        const std::size_t c = static_cast<std::size_t>(shape.channels[level]), c2 = static_cast<std::size_t>(shape.channels[level + 1]);
        const std::string p = "blocks." + std::to_string(level) + "." + std::to_string(shape.blocks[level]) + ".";
        bool ok = tf->ReadF32(p + "norm1.weight", c, &u->norm1_w, err) && tf->ReadF32(p + "norm1.bias", c, &u->norm1_b, err) &&
                  tf->ReadF32(p + "conv1.weight", 8 * c2 * 27 * c, &u->conv1_w, err) && tf->ReadF32(p + "conv1.bias", 8 * c2, &u->conv1_b, err) &&
                  tf->ReadF32(p + "conv2.weight", c2 * 27 * c2, &u->conv2_w, err) && tf->ReadF32(p + "conv2.bias", c2, &u->conv2_b, err);
        if (ok && shape.pred_subdiv)
            ok = tf->ReadF32(p + "to_subdiv.weight", 8 * c, &u->subdiv_w, err) && tf->ReadF32(p + "to_subdiv.bias", 8, &u->subdiv_b, err);
        return ok;
    };
    return true;
}

SparseDecHostWeights SparseDecHostWeights::Random(const SparseDecShape& shape, std::uint32_t seed, float scale) {
    rng_state = seed * 0x9E3779B97F4A7C15ULL + 11;
    SparseDecHostWeights w;
    w.shape = shape;
    fill(&w.from_w, static_cast<std::size_t>(shape.channels[0]) * shape.latent_channels, scale);
    fill(&w.from_b, static_cast<std::size_t>(shape.channels[0]), scale);
    fill(&w.out_w, static_cast<std::size_t>(shape.outputs) * shape.channels[shape.levels - 1], scale);
    fill(&w.out_b, static_cast<std::size_t>(shape.outputs), scale);
    auto blocks = std::make_shared<std::vector<std::vector<ConvNeXtWeights>>>(static_cast<std::size_t>(shape.levels));
    auto ups = std::make_shared<std::vector<C2SWeights>>(static_cast<std::size_t>(shape.levels - 1));
    for (int l = 0; l < shape.levels; ++l) {
        const std::size_t c = static_cast<std::size_t>(shape.channels[l]);
        for (int j = 0; j < shape.blocks[l]; ++j) {
            ConvNeXtWeights b;
            fill(&b.conv_w, c * 27 * c, scale / 5.f); fill(&b.conv_b, c, scale);
            b.norm_w.assign(c, 1.f); fill(&b.norm_b, c, scale);
            fill(&b.mlp0_w, 4 * c * c, scale); fill(&b.mlp0_b, 4 * c, scale);
            fill(&b.mlp2_w, c * 4 * c, scale); fill(&b.mlp2_b, c, scale);
            (*blocks)[static_cast<std::size_t>(l)].push_back(std::move(b));
        }
        if (l + 1 < shape.levels) {
            const std::size_t c2 = static_cast<std::size_t>(shape.channels[l + 1]);
            C2SWeights u;
            u.norm1_w.assign(c, 1.f); fill(&u.norm1_b, c, scale);
            fill(&u.conv1_w, 8 * c2 * 27 * c, scale / 5.f); fill(&u.conv1_b, 8 * c2, scale);
            fill(&u.conv2_w, c2 * 27 * c2, scale / 5.f); fill(&u.conv2_b, c2, scale);
            if (shape.pred_subdiv) { fill(&u.subdiv_w, 8 * c, scale * 4.f); fill(&u.subdiv_b, 8, scale); }
            (*ups)[static_cast<std::size_t>(l)] = std::move(u);
        }
    }
    w.load_block = [blocks](int level, int j, ConvNeXtWeights* b, std::string*) {
        *b = (*blocks)[static_cast<std::size_t>(level)][static_cast<std::size_t>(j)];
        return true;
    };
    w.load_up = [ups](int level, C2SWeights* u, std::string*) {
        *u = (*ups)[static_cast<std::size_t>(level)];
        return true;
    };
    return w;
}

bool sparse_dec_forward_cpu(const SparseDecHostWeights& w, const float* latent, const std::vector<Coord>& coords_in,
                            SparseDecResult* result, std::string* error, const SparseDecOptions& options) {
    const SparseDecShape& s = w.shape;
    if (!s.pred_subdiv && (!options.guide_masks || options.guide_masks->size() < static_cast<std::size_t>(s.levels - 1))) {
        if (error) *error = "sparse decoder without pred_subdiv needs guide masks";
        return false;
    }
    Level level;
    level.Build(coords_in);
    const int C0 = s.channels[0];
    std::vector<float> x(static_cast<std::size_t>(level.n()) * C0);
    linear(latent, level.n(), w.from_w.data(), w.from_b.data(), C0, s.latent_channels, x.data());
    result->masks.clear();
    result->level_feats.clear();
    result->level_coords.clear();
    std::vector<float> h, h2, mlp;
    ConvNeXtWeights b;
    C2SWeights u;
    const int levels = options.level_limit > 0 ? std::min(options.level_limit, s.levels) : s.levels;
    for (int l = 0; l < levels; ++l) {
        const int C = s.channels[l];
        const int N = level.n();
        for (int j = 0; j < s.blocks[l]; ++j) {
            if (!w.load_block(l, j, &b, error)) return false;
            h.resize(static_cast<std::size_t>(N) * C);
            submanifold_conv3(x.data(), N, C, level.table.data(), b.conv_w.data(), b.conv_b.data(), C, h.data(), 65536);
            layer_norm_rows(h.data(), N, C, b.norm_w.data(), b.norm_b.data(), SparseDecConfig::norm_eps);
            mlp.resize(static_cast<std::size_t>(N) * 4 * C);
            linear(h.data(), N, b.mlp0_w.data(), b.mlp0_b.data(), 4 * C, C, mlp.data());
            silu_rows(mlp.data(), mlp.size());
            linear(mlp.data(), N, b.mlp2_w.data(), b.mlp2_b.data(), C, 4 * C, h.data());
            for (std::size_t i = 0; i < x.size(); ++i) x[i] += h[i];
        }
        if (options.capture_levels) {
            result->level_feats.push_back(x);
            result->level_coords.push_back(level.coords);
        }
        if (l + 1 >= s.levels || l + 1 >= levels) break;
        // Up-block C -> C2
        if (!w.load_up(l, &u, error)) return false;
        const int C2 = s.channels[l + 1];
        if (C < 8 || C2 % (C / 8) != 0) {
            if (error) *error = "sparse decoder: channels " + std::to_string(C) + " -> " + std::to_string(C2) + " not supported by the skip path";
            return false;
        }
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(N) * 8);
        if (s.pred_subdiv) {
            std::vector<float> logits(static_cast<std::size_t>(N) * 8);
            linear(x.data(), N, u.subdiv_w.data(), u.subdiv_b.data(), 8, C, logits.data());
            for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = logits[i] > 0.f ? 1 : 0;
        } else {
            mask = (*options.guide_masks)[static_cast<std::size_t>(l)];
            if (mask.size() != static_cast<std::size_t>(N) * 8) {
                if (error) *error = "guide mask size mismatch at level " + std::to_string(l);
                return false;
            }
        }
        result->masks.push_back(mask);
        h = x;
        layer_norm_rows(h.data(), N, C, u.norm1_w.data(), u.norm1_b.data(), SparseDecConfig::norm_eps);
        silu_rows(h.data(), h.size());
        h2.resize(static_cast<std::size_t>(N) * 8 * C2);
        submanifold_conv3(h.data(), N, C, level.table.data(), u.conv1_w.data(), u.conv1_b.data(), 8 * C2, h2.data(), 32768);
        const Children ch = channel2spatial_coords(level.coords, mask);
        const int M = static_cast<int>(ch.coords.size());
        if (M == 0) {
            if (error) *error = "no voxels subdivided at level " + std::to_string(l);
            return false;
        }
        std::vector<float> hc(static_cast<std::size_t>(M) * C2);
        channel2spatial_feats(h2.data(), C2, ch, hc.data());
        layer_norm_rows(hc.data(), M, C2, nullptr, nullptr, SparseDecConfig::norm_eps);
        silu_rows(hc.data(), hc.size());
        Level next;
        next.Build(ch.coords);
        std::vector<float> conv2(static_cast<std::size_t>(M) * C2);
        submanifold_conv3(hc.data(), M, C2, next.table.data(), u.conv2_w.data(), u.conv2_b.data(), C2, conv2.data(), 65536);
        std::vector<float> skip(static_cast<std::size_t>(M) * C2);
        channel2spatial_skip(x.data(), C, C2 / (C / 8), ch, skip.data());
        for (std::size_t i = 0; i < conv2.size(); ++i) conv2[i] += skip[i];
        x.swap(conv2);
        level = std::move(next);
    }
    result->coords = level.coords;
    if (levels < s.levels) {
        result->outputs.clear();
        return true;
    }
    const int Cl = s.channels[s.levels - 1];
    layer_norm_rows(x.data(), level.n(), Cl, nullptr, nullptr, SparseDecConfig::final_norm_eps);
    result->outputs.resize(static_cast<std::size_t>(level.n()) * s.outputs);
    linear(x.data(), level.n(), w.out_w.data(), w.out_b.data(), s.outputs, Cl, result->outputs.data());
    return true;
}

}  // namespace gufo::trellis2
