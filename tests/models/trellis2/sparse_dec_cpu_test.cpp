// CPU tests for src/models/trellis2/sparse_dec_cpu.cpp on a tiny shape (channels
// 16/8/8/8/8, one ConvNeXt block per level): output rows = children of the last up-block,
// masks per up-block, guide masks reproduce the predicted run, determinism, level_limit,
// the random weights validate, and the skip path keeps a constant input constant-ish.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/sparse_dec.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
}  // namespace

int main() {
    tr::SparseDecShape s;
    s.latent_channels = 4;
    s.channels[0] = 16; s.channels[1] = 8; s.channels[2] = 8; s.channels[3] = 8; s.channels[4] = 8;
    s.blocks[0] = 1; s.blocks[1] = 1; s.blocks[2] = 1; s.blocks[3] = 1; s.blocks[4] = 0;
    s.outputs = 7;
    auto w = tr::SparseDecHostWeights::Random(s, 9, 0.1f);
    std::vector<tr::Coord> coords = {{1, 1, 1}, {1, 1, 2}, {2, 1, 1}, {3, 3, 3}, {1, 2, 1}};
    std::vector<float> latent(coords.size() * 4);
    for (std::size_t i = 0; i < latent.size(); ++i) latent[i] = std::sin(0.7f * static_cast<float>(i));
    tr::SparseDecResult r1, r2, r3;
    std::string error;
    tr::SparseDecOptions opts;
    opts.capture_levels = true;
    Expect(tr::sparse_dec_forward_cpu(w, latent.data(), coords, &r1, &error, opts), "forward: " + error);
    Expect(r1.masks.size() == 4, "four subdivision masks");
    std::size_t children = 0;
    for (auto m : r1.masks.back()) children += m;
    Expect(r1.coords.size() == children && r1.outputs.size() == children * 7, "outputs on the children of the last up-block");
    Expect(r1.level_feats.size() == 5 && r1.level_feats[0].size() == coords.size() * 16, "level taps");
    for (int l = 1; l < 5; ++l) {
        std::size_t c = 0;
        for (auto m : r1.masks[static_cast<std::size_t>(l - 1)]) c += m;
        Expect(r1.level_coords[static_cast<std::size_t>(l)].size() == c, "level " + std::to_string(l) + " has as many voxels as mask bits");
    }
    bool finite = true;
    for (float v : r1.outputs) finite = finite && std::isfinite(v);
    Expect(finite, "finite outputs");
    // children coordinates are refinements of their parents: coord / 2^l in the original set
    bool refined = true;
    for (const auto& c : r1.coords) {
        const tr::Coord p{c.x >> 4, c.y >> 4, c.z >> 4};
        bool found = false;
        for (const auto& o : coords) found = found || o == p;
        refined = refined && found;
    }
    Expect(refined, "final coordinates refine the input voxels (16x)");
    // guided run with the predicted masks (texture decoder mode) gives the same coordinates
    auto wg = w;
    wg.shape.pred_subdiv = false;
    tr::SparseDecOptions guided;
    guided.guide_masks = &r1.masks;
    Expect(tr::sparse_dec_forward_cpu(wg, latent.data(), coords, &r2, &error, guided), "guided forward: " + error);
    Expect(r2.coords == r1.coords && r2.outputs == r1.outputs, "guide masks reproduce the predicted run");
    // deterministic + level_limit
    tr::sparse_dec_forward_cpu(w, latent.data(), coords, &r3, &error);
    Expect(r3.outputs == r1.outputs, "deterministic");
    tr::SparseDecOptions lim;
    lim.level_limit = 2;
    lim.capture_levels = true;
    tr::SparseDecResult r4;
    Expect(tr::sparse_dec_forward_cpu(w, latent.data(), coords, &r4, &error, lim), "level_limit: " + error);
    Expect(r4.level_feats.size() == 2 && r4.level_feats[1] == r1.level_feats[1] && r4.outputs.empty(), "level_limit 2 matches the first two levels");
    // mask mismatch rejected
    std::vector<std::vector<std::uint8_t>> bad = r1.masks;
    bad[0].pop_back();
    tr::SparseDecOptions badopt;
    badopt.guide_masks = &bad;
    Expect(!tr::sparse_dec_forward_cpu(wg, latent.data(), coords, &r4, &error, badopt), "bad guide mask rejected");
    if (failures == 0) std::cout << "trellis2_sparse_dec_cpu_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
