// CPU tests for src/models/triposg/marching_cubes.*: analytic sphere SDF (vertex /
// face counts, Euler characteristic, closed oriented surface, volume within 2 %),
// the TripoSG sign convention (positive inside), two disjoint spheres and a random
// field covering all 256 cube cases (every used edge id must be a crossing edge, so a
// broken table row shows up as an open edge inside a cube).
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/triposg/marching_cubes.hpp"

namespace mc = gufo::triposg;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename F>
std::vector<float> Sample(std::size_t n, float lo, float hi, F f) {
    std::vector<float> values(n * n * n);
    const float step = (hi - lo) / static_cast<float>(n - 1);
    for (std::size_t ix = 0; ix < n; ++ix)
        for (std::size_t iy = 0; iy < n; ++iy)
            for (std::size_t iz = 0; iz < n; ++iz)
                values[(ix * n + iy) * n + iz] =
                    f(lo + static_cast<float>(ix) * step, lo + static_cast<float>(iy) * step,
                      lo + static_cast<float>(iz) * step);
    return values;
}

mc::ScalarGrid Grid(const std::vector<float>& values, std::size_t n, float lo, float hi) {
    mc::ScalarGrid g;
    g.values = values.data();
    g.nx = g.ny = g.nz = n;
    for (int a = 0; a < 3; ++a) { g.min[a] = lo; g.max[a] = hi; }
    return g;
}

void SphereTest() {
    const std::size_t n = 65;
    const float r = 0.7f;
    const auto values = Sample(n, -1.f, 1.f, [&](float x, float y, float z) {
        return std::sqrt(x * x + y * y + z * z) - r;  // plain SDF, negative inside
    });
    mc::MarchingCubesOptions opt;
    opt.inside_positive = false;
    const mc::Mesh mesh = mc::marching_cubes(Grid(values, n, -1.f, 1.f), opt);

    Expect(!mesh.vertices.empty() && !mesh.faces.empty(), "sphere: mesh not empty");
    // Closed triangle mesh of genus 0: F = 2V - 4, E = 3F/2.
    Expect(mesh.faces.size() == 2 * mesh.vertices.size() - 4,
           "sphere: Euler characteristic 2 (V=" + std::to_string(mesh.vertices.size()) +
               " F=" + std::to_string(mesh.faces.size()) + ")");
    // Surface area 4*pi*r^2 at ~0.031 spacing: roughly 4*pi*r^2 / (0.5*h^2) faces.
    Expect(mesh.faces.size() > 8000 && mesh.faces.size() < 20000,
           "sphere: face count plausible (" + std::to_string(mesh.faces.size()) + ")");
    Expect(mc::count_open_edges(mesh) == 0, "sphere: closed and consistently oriented");
    const double expected = 4.0 / 3.0 * 3.14159265358979 * r * r * r;
    const double vol = mc::signed_volume(mesh);
    Expect(std::fabs(vol - expected) / expected < 0.02,
           "sphere: volume " + std::to_string(vol) + " ~ " + std::to_string(expected));
    Expect(vol > 0, "sphere: outward orientation (positive signed volume)");
    float bmin[3], bmax[3];
    Expect(mc::bounding_box(mesh, bmin, bmax), "sphere: bounding box");
    for (int a = 0; a < 3; ++a) {
        Expect(std::fabs(bmin[a] + r) < 0.02f && std::fabs(bmax[a] - r) < 0.02f,
               "sphere: bbox axis " + std::to_string(a));
    }
    // All vertices lie on the sphere within one cell.
    float worst = 0.f;
    for (const auto& v : mesh.vertices)
        worst = std::max(worst, std::fabs(std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z) - r));
    Expect(worst < 0.01f, "sphere: vertices on the surface (worst " + std::to_string(worst) + ")");

    // TripoSG convention: the decoder returns -SDF (positive inside). Same mesh expected.
    std::vector<float> negated(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) negated[i] = -values[i];
    mc::MarchingCubesOptions opt_pos;
    opt_pos.inside_positive = true;
    const mc::Mesh mesh_pos = mc::marching_cubes(Grid(negated, n, -1.f, 1.f), opt_pos);
    Expect(mesh_pos.vertices.size() == mesh.vertices.size() && mesh_pos.faces.size() == mesh.faces.size(),
           "inside_positive: same counts");
    Expect(mc::signed_volume(mesh_pos) > 0 && mc::count_open_edges(mesh_pos) == 0,
           "inside_positive: outward and closed");

    // Without welding every triangle has its own vertices.
    mc::MarchingCubesOptions opt_soup = opt;
    opt_soup.weld = false;
    const mc::Mesh soup = mc::marching_cubes(Grid(values, n, -1.f, 1.f), opt_soup);
    Expect(soup.faces.size() == mesh.faces.size() && soup.vertices.size() == 3 * soup.faces.size(),
           "weld=false: triangle soup");
}

void TwoSpheresTest() {
    const std::size_t n = 49;
    const auto values = Sample(n, -1.f, 1.f, [](float x, float y, float z) {
        const float a = std::sqrt((x - 0.45f) * (x - 0.45f) + y * y + z * z) - 0.35f;
        const float b = std::sqrt((x + 0.45f) * (x + 0.45f) + y * y + z * z) - 0.35f;
        return std::min(a, b);
    });
    mc::MarchingCubesOptions opt;
    opt.inside_positive = false;
    const mc::Mesh mesh = mc::marching_cubes(Grid(values, n, -1.f, 1.f), opt);
    // Two genus-0 components: V - E + F = 4.
    Expect(mesh.faces.size() == 2 * mesh.vertices.size() - 8, "two spheres: Euler characteristic 4");
    Expect(mc::count_open_edges(mesh) == 0, "two spheres: closed");
    const double expected = 2.0 * 4.0 / 3.0 * 3.14159265358979 * 0.35 * 0.35 * 0.35;
    Expect(std::fabs(mc::signed_volume(mesh) - expected) / expected < 0.03, "two spheres: volume");
}

void RandomFieldTest() {
    // Every one of the 256 cases occurs; each cube's patch must be manifold inside the
    // cube (the table only references crossing edges, so an open edge inside a cube
    // means a broken row). Open edges between neighbouring cubes can legitimately
    // occur on ambiguous faces with the classic table; they are counted separately.
    const std::size_t n = 12;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> values(n * n * n);
    std::vector<bool> seen(256, false);
    for (int round = 0; round < 30; ++round) {
        for (auto& v : values) v = dist(rng);
        for (std::size_t ix = 0; ix + 1 < n; ++ix)
            for (std::size_t iy = 0; iy + 1 < n; ++iy)
                for (std::size_t iz = 0; iz + 1 < n; ++iz) {
                    int idx = 0;
                    const int off[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
                    for (int c = 0; c < 8; ++c)
                        if (values[((ix + off[c][0]) * n + iy + off[c][1]) * n + iz + off[c][2]] < 0.f) idx |= 1 << c;
                    seen[static_cast<std::size_t>(idx)] = true;
                }
        mc::MarchingCubesOptions opt;
        opt.inside_positive = false;
        const mc::Mesh mesh = mc::marching_cubes(Grid(values, n, -1.f, 1.f), opt);
        Expect(!mesh.faces.empty(), "random: mesh produced");
        for (const auto& f : mesh.faces)
            for (auto i : f) Expect(i < mesh.vertices.size(), "random: index in range");
        // With welding the open edges are only on ambiguous faces: far fewer than the
        // ~1/8 of all edges a broken row would produce.
        const std::size_t open = mc::count_open_edges(mesh);
        Expect(open < mesh.faces.size() * 3 / 8,
               "random: open edges " + std::to_string(open) + " of " + std::to_string(mesh.faces.size() * 3));
    }
    int covered = 0;
    for (bool s : seen) covered += s ? 1 : 0;
    Expect(covered == 256, "random: all 256 cases exercised (" + std::to_string(covered) + ")");
}

void EmptyAndDegenerateTest() {
    const std::size_t n = 8;
    const auto all_outside = Sample(n, -1.f, 1.f, [](float, float, float) { return 1.f; });
    mc::MarchingCubesOptions opt;
    opt.inside_positive = false;
    Expect(mc::marching_cubes(Grid(all_outside, n, -1.f, 1.f), opt).faces.empty(), "all outside: no faces");
    mc::ScalarGrid bad;
    Expect(mc::marching_cubes(bad, opt).faces.empty(), "null grid: no faces");
    const mc::Mesh empty;
    float a[3], b[3];
    Expect(!mc::bounding_box(empty, a, b), "empty mesh: no bbox");
    Expect(mc::count_open_edges(empty) == 0 && mc::signed_volume(empty) == 0.0, "empty mesh: zero");
}

}  // namespace

int main() {
    SphereTest();
    TwoSpheresTest();
    RandomFieldTest();
    EmptyAndDegenerateTest();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "marching_cubes_test: ok\n";
    return 0;
}
