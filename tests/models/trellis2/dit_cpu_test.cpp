// CPU tests for src/models/trellis2/dit_cpu.cpp on a tiny shape (width 48, 2 heads x 24,
// 2 blocks, MLP 64, cond 16): shapes, finite/deterministic output, the timestep and the
// condition change the result, taps, layer_limit, dense grid coordinate order, RoPE with
// head_dim 24 (4 frequencies per axis).
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "src/models/trellis2/dit.hpp"
#include "src/models/trellis2/scheduler.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
double Diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::fabs(a[i] - b[i]);
    return d / static_cast<double>(a.size());
}
}  // namespace

int main() {
    tr::DitShape s;
    s.width = 48; s.heads = 2; s.layers = 2; s.mlp = 64; s.cond_dim = 16; s.in_channels = 8; s.out_channels = 8;
    auto w = tr::DitHostWeights::Random(s, 3, 0.05f);
    const auto coords = tr::dense_grid_coords(4);   // 64 tokens
    Expect(coords.size() == 64 && coords[1].z == 1 && coords[4].y == 1 && coords[16].x == 1, "dense grid order x*16+y*4+z");
    const int N = 64, M = 5;
    std::vector<float> x(static_cast<std::size_t>(N) * 8), cond(static_cast<std::size_t>(M) * 16);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.37f * static_cast<float>(i));
    for (std::size_t i = 0; i < cond.size(); ++i) cond[i] = std::cos(0.11f * static_cast<float>(i));
    std::string error;
    std::vector<float> v1, v2, v3;
    std::map<int, std::vector<float>> taps;
    tr::DitForwardOptions opts;
    opts.tap_layers = {0, 1};
    Expect(tr::dit_forward_cpu(w, x.data(), coords, cond.data(), M, 1000.f, &v1, &taps, &error, opts), "forward: " + error);
    Expect(v1.size() == static_cast<std::size_t>(N) * 8, "v shape [64, 8]");
    bool finite = true;
    for (float e : v1) finite = finite && std::isfinite(e);
    Expect(finite, "finite");
    Expect(taps.size() == 2 && taps[0].size() == static_cast<std::size_t>(N) * 48 && Diff(taps[0], taps[1]) > 1e-6, "taps for both blocks differ");
    tr::dit_forward_cpu(w, x.data(), coords, cond.data(), M, 1000.f, &v2, nullptr, &error);
    Expect(v1 == v2, "deterministic");
    tr::dit_forward_cpu(w, x.data(), coords, cond.data(), M, 300.f, &v3, nullptr, &error);
    Expect(Diff(v1, v3) > 1e-6, "timestep changes the velocity (modulation)");
    std::vector<float> zero(cond.size(), 0.f), v4;
    tr::dit_forward_cpu(w, x.data(), coords, zero.data(), M, 1000.f, &v4, nullptr, &error);
    Expect(Diff(v1, v4) > 1e-6, "condition changes the velocity (cross attention)");
    // layer_limit 1 equals taps[0] -> final norm + out
    tr::DitForwardOptions lim;
    lim.layer_limit = 1;
    lim.tap_layers = {0};
    std::map<int, std::vector<float>> taps1;
    std::vector<float> v5;
    tr::dit_forward_cpu(w, x.data(), coords, cond.data(), M, 1000.f, &v5, &taps1, &error, lim);
    Expect(taps1.size() == 1 && taps1[0] == taps[0], "layer_limit 1 reproduces block 0");
    Expect(Diff(v5, v1) > 1e-6, "layer_limit output differs from the full run");
    // RoPE with head_dim 24: 4 per axis, 12 pairs, no padding
    float ang[12];
    tr::rope_angles(2, 3, 5, ang, 24);
    Expect(ang[0] == 2.f && ang[4] == 3.f && ang[8] == 5.f && std::fabs(ang[1] - 2.f / std::pow(10000.f, 0.25f)) < 1e-5f, "rope angles head_dim 24");
    // Positions matter: swapping two token coordinates changes those rows' velocities
    auto coords2 = coords;
    std::swap(coords2[3], coords2[40]);
    std::vector<float> v6;
    tr::dit_forward_cpu(w, x.data(), coords2, cond.data(), M, 1000.f, &v6, nullptr, &error);
    Expect(Diff(v1, v6) > 1e-7, "coordinates (RoPE) influence the output");
    if (failures == 0) std::cout << "trellis2_dit_cpu_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
