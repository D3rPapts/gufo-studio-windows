// CPU tests for src/models/trellis2/sparse_conv_cpu.cpp with random weights:
// hashmap hits/misses, neighbour table (self tap, symmetry), submanifold conv against
// the dense brute-force reference (blocked and unblocked), Channel2Spatial child order,
// skip-path channel repetition, LayerNorm/SiLU basics.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "src/models/trellis2/sparse_conv.hpp"

namespace tr = gufo::trellis2;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<tr::Coord> RandomCoords(std::mt19937& rng, int grid, double density) {
    std::vector<tr::Coord> coords;
    std::bernoulli_distribution keep(density);
    for (int x = 0; x < grid; ++x)
        for (int y = 0; y < grid; ++y)
            for (int z = 0; z < grid; ++z)
                if (keep(rng)) coords.push_back({x, y, z});
    // shuffle so that row order != grid order
    std::shuffle(coords.begin(), coords.end(), rng);
    return coords;
}

std::vector<float> Random(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

void HashMapTest() {
    std::mt19937 rng(1);
    auto coords = RandomCoords(rng, 12, 0.3);
    tr::VoxelHashMap map;
    map.Build(coords);
    Expect(map.size() == coords.size(), "hashmap holds every voxel");
    bool all = true;
    for (std::size_t i = 0; i < coords.size(); ++i)
        all = all && map.Find(coords[i].x, coords[i].y, coords[i].z) == static_cast<std::int64_t>(i);
    Expect(all, "every inserted coordinate maps to its row");
    std::set<std::tuple<int, int, int>> present;
    for (const auto& c : coords) present.emplace(c.x, c.y, c.z);
    bool misses = true;
    for (int x = -1; x <= 12; ++x)
        for (int y = -1; y <= 12; ++y)
            for (int z = -1; z <= 12; ++z)
                if (!present.count({x, y, z})) misses = misses && map.Find(x, y, z) == -1;
    Expect(misses, "absent coordinates (incl. negative / outside) return -1");
    Expect(map.Find(1023, 1023, 1023) == -1 && map.Find(1024, 0, 0) == -1, "range limits");
    bool threw = false;
    try {
        tr::VoxelHashMap dup;
        dup.Build({{1, 2, 3}, {1, 2, 3}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Expect(threw, "duplicate coordinates are rejected");
}

void NeighbourTableTest() {
    std::mt19937 rng(2);
    auto coords = RandomCoords(rng, 10, 0.4);
    tr::VoxelHashMap map;
    map.Build(coords);
    auto table = tr::build_neighbour_table(coords, map);
    Expect(table.size() == coords.size() * 27, "27 taps per voxel");
    bool self = true, sym = true, offsets = true;
    for (std::size_t i = 0; i < coords.size(); ++i) {
        self = self && table[i * 27 + tr::kCenterTap] == static_cast<std::int32_t>(i);
        for (int t = 0; t < 27; ++t) {
            const std::int32_t j = table[i * 27 + t];
            if (j < 0) continue;
            int dx, dy, dz;
            tr::tap_offset(t, &dx, &dy, &dz);
            offsets = offsets && coords[static_cast<std::size_t>(j)].x == coords[i].x + dx &&
                      coords[static_cast<std::size_t>(j)].y == coords[i].y + dy &&
                      coords[static_cast<std::size_t>(j)].z == coords[i].z + dz;
            sym = sym && table[static_cast<std::size_t>(j) * 27 + (26 - t)] == static_cast<std::int32_t>(i);
        }
    }
    Expect(self, "centre tap 13 is the voxel itself");
    Expect(offsets, "tap t points to coord + offset(t)");
    Expect(sym, "neighbour relation is symmetric (tap t <-> 26 - t)");
    Expect(tr::tap_index(0, 0, 0) == 13 && tr::tap_index(-1, -1, -1) == 0 && tr::tap_index(1, 1, 1) == 26 &&
               tr::tap_index(1, 0, 0) == 22 && tr::tap_index(0, 1, 0) == 16 && tr::tap_index(0, 0, 1) == 14,
           "tap index = (dx+1)*9 + (dy+1)*3 + (dz+1)");
}

void ConvTest() {
    std::mt19937 rng(3);
    auto coords = RandomCoords(rng, 8, 0.35);
    const int ci = 5, co = 4, n = static_cast<int>(coords.size());
    auto in = Random(rng, static_cast<std::size_t>(n) * ci, 1.f);
    auto w = Random(rng, static_cast<std::size_t>(co) * 27 * ci, 0.3f);
    auto b = Random(rng, co, 0.5f);
    std::vector<float> ref(static_cast<std::size_t>(n) * co);
    tr::submanifold_conv3_dense_reference(coords, in.data(), ci, w.data(), b.data(), co, ref.data());

    tr::VoxelHashMap map;
    map.Build(coords);
    auto table = tr::build_neighbour_table(coords, map);
    std::vector<float> out(ref.size()), out_blocked(ref.size());
    tr::submanifold_conv3(in.data(), n, ci, table.data(), w.data(), b.data(), co, out.data(), 0);
    tr::submanifold_conv3(in.data(), n, ci, table.data(), w.data(), b.data(), co, out_blocked.data(), 7);
    Expect(MaxAbsDiff(out, ref) < 1e-4, "gather-GEMM conv equals the dense brute force");
    Expect(MaxAbsDiff(out_blocked, out) == 0.0, "row blocking does not change the result");

    // An isolated voxel sees only its centre tap.
    std::vector<tr::Coord> one = {{5, 5, 5}};
    tr::VoxelHashMap m1;
    m1.Build(one);
    auto t1 = tr::build_neighbour_table(one, m1);
    std::vector<float> o1(co);
    tr::submanifold_conv3(in.data(), 1, ci, t1.data(), w.data(), b.data(), co, o1.data(), 0);
    bool centre = true;
    for (int o = 0; o < co; ++o) {
        double acc = b[static_cast<std::size_t>(o)];
        for (int c = 0; c < ci; ++c)
            acc += static_cast<double>(in[static_cast<std::size_t>(c)]) *
                   w[static_cast<std::size_t>(o) * 27 * ci + static_cast<std::size_t>(tr::kCenterTap) * ci + static_cast<std::size_t>(c)];
        centre = centre && std::fabs(o1[static_cast<std::size_t>(o)] - acc) < 1e-5;
    }
    Expect(centre, "isolated voxel uses only tap 13 columns");
}

void Channel2SpatialTest() {
    std::vector<tr::Coord> parents = {{1, 2, 3}, {0, 0, 0}, {4, 4, 4}};
    // parent 0: children k = 1 and 6; parent 1: none; parent 2: k = 0, 7
    std::vector<std::uint8_t> mask(24, 0);
    mask[0 * 8 + 1] = 1; mask[0 * 8 + 6] = 1; mask[2 * 8 + 0] = 1; mask[2 * 8 + 7] = 1;
    auto ch = tr::channel2spatial_coords(parents, mask);
    Expect(ch.coords.size() == 4, "four children");
    Expect(ch.coords[0] == tr::Coord{3, 4, 6} && ch.coords[1] == tr::Coord{2, 5, 7} &&
               ch.coords[2] == tr::Coord{8, 8, 8} && ch.coords[3] == tr::Coord{9, 9, 9},
           "child = 2*parent + (k&1, k>>1&1, k>>2&1), parent-major, k ascending");
    Expect(ch.parent[1] == 0 && ch.parent[2] == 2 && ch.child_index[1] == 6, "parent / child index bookkeeping");
    const int C = 3;
    std::vector<float> feats(parents.size() * 8 * C);
    for (std::size_t i = 0; i < feats.size(); ++i) feats[i] = static_cast<float>(i);
    std::vector<float> out(4 * C);
    tr::channel2spatial_feats(feats.data(), C, ch, out.data());
    Expect(out[0] == (0 * 8 + 1) * C + 0 && out[3] == (0 * 8 + 6) * C + 0 && out[6] == (2 * 8 + 0) * C + 0 &&
               out[11] == (2 * 8 + 7) * C + 2,
           "child features are rows k of the [N*8, C] view");
    // skip: channels_in 16 -> 2 per child, repeated 3x -> 6
    std::vector<float> xin(parents.size() * 16);
    for (std::size_t i = 0; i < xin.size(); ++i) xin[i] = static_cast<float>(i);
    std::vector<float> skip(4 * 6);
    tr::channel2spatial_skip(xin.data(), 16, 3, ch, skip.data());
    Expect(skip[0] == 2 && skip[1] == 2 && skip[2] == 2 && skip[3] == 3 && skip[5] == 3 && skip[6] == 12 && skip[9] == 13,
           "skip = child slice k*Cs..(k+1)*Cs with each channel repeated 3x consecutively");
}

void NormActTest() {
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f, 10.f, 10.f, 10.f, 10.f};
    tr::layer_norm_rows(x.data(), 2, 4, nullptr, nullptr, 1e-6f);
    Expect(std::fabs(x[0] + 1.3416f) < 1e-3f && std::fabs(x[3] - 1.3416f) < 1e-3f, "layer norm row 0");
    Expect(std::fabs(x[4]) < 1e-2f, "constant row normalises to ~0");
    std::vector<float> w = {2.f, 2.f, 2.f, 2.f}, b = {1.f, 1.f, 1.f, 1.f};
    std::vector<float> y = {1.f, 2.f, 3.f, 4.f};
    tr::layer_norm_rows(y.data(), 1, 4, w.data(), b.data(), 1e-6f);
    Expect(std::fabs(y[3] - (2.f * 1.3416f + 1.f)) < 1e-3f, "affine layer norm");
    std::vector<float> s = {0.f, 100.f, -100.f};
    tr::silu_rows(s.data(), s.size());
    Expect(s[0] == 0.f && std::fabs(s[1] - 100.f) < 1e-3f && std::fabs(s[2]) < 1e-3f, "silu");
}

}  // namespace

int main() {
    HashMapTest();
    NeighbourTableTest();
    ConvTest();
    Channel2SpatialTest();
    NormActTest();
    if (failures == 0) std::cout << "trellis2_sparse_conv_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
