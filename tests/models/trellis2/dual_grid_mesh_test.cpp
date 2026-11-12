// CPU tests for src/models/trellis2/dual_grid_mesh.cpp: a 2x2x2 block of voxels whose six
// flagged edges enclose the central corner gives six quads (twelve triangles) with outward
// winding and no boundary; a missing neighbour becomes a virtual centre vertex instead of a
// hole; the split follows the quad weights; vertex placement; output decoding (sigmoid with
// margin, logit sign, softplus). Sheet splitting: two inside corners diagonal on one voxel
// face give two vertices per shared voxel and no edge with more than two triangles. A parity
// defect (one flag removed) becomes a seam, the mesh stays closed. Random flags: always closed.
// AI-assisted (Claude Code), 2026-11-12.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/dual_grid_mesh.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
double Volume(const tr::Mesh& m) {
    double vol = 0;
    for (const auto& f : m.faces) {
        const tr::MeshVertex& a = m.vertices[f[0]];
        const tr::MeshVertex& b = m.vertices[f[1]];
        const tr::MeshVertex& c = m.vertices[f[2]];
        vol += (static_cast<double>(a.x) * (static_cast<double>(b.y) * c.z - static_cast<double>(b.z) * c.y) -
                static_cast<double>(a.y) * (static_cast<double>(b.x) * c.z - static_cast<double>(b.z) * c.x) +
                static_cast<double>(a.z) * (static_cast<double>(b.x) * c.y - static_cast<double>(b.y) * c.x)) / 6.0;
    }
    return vol;
}
}  // namespace

int main() {
    std::vector<tr::Coord> coords;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) coords.push_back({x, y, z});   // index = x*4 + y*2 + z
    const std::size_t n = coords.size();
    std::vector<float> dual(n * 3, 0.5f);
    std::vector<std::uint8_t> inter(n * 3, 0);
    // The six edges meeting at corner (1,1,1): owner voxel = corner minus one in the other axes.
    inter[0 * 3 + 0] = inter[0 * 3 + 1] = inter[0 * 3 + 2] = 1;   // voxel (0,0,0): edges into the corner
    inter[4 * 3 + 0] = 1;                                         // voxel (1,0,0): x-edge out of the corner
    inter[2 * 3 + 1] = 1;                                         // voxel (0,1,0): y-edge
    inter[1 * 3 + 2] = 1;                                         // voxel (0,0,1): z-edge
    std::vector<float> wgt(n, 1.f);
    tr::DualGridInput in;
    in.coords = &coords; in.dual = dual.data(); in.intersected = inter.data(); in.split_weight = wgt.data();
    in.resolution = 2;
    in.sign_smooth_passes = 0;
    tr::DualGridStats st;
    tr::Mesh mesh = tr::dual_grid_to_mesh(in, &st);
    Expect(st.quads == 6 && mesh.faces.size() == 12, "six quads -> twelve triangles (" + std::to_string(st.quads) + ")");
    Expect(mesh.vertices.size() == 8 && st.extra_vertices == 0 && st.virtual_vertices == 0, "one vertex per voxel");
    Expect(st.open_edges == 0 && st.seam_edges == 0, "closed, no seam (" + std::to_string(st.open_edges) + "/" + std::to_string(st.seam_edges) + ")");
    Expect(Volume(mesh) > 0, "outward winding (volume " + std::to_string(Volume(mesh)) + ")");
    // vertex of voxel (1,0,1) at index 5: (1 + 0.5) * 0.5 - 0.5 = 0.25, (0+0.5)*0.5-0.5 = -0.25
    Expect(std::fabs(mesh.vertices[5].x - 0.25f) < 1e-6f && std::fabs(mesh.vertices[5].y + 0.25f) < 1e-6f && std::fabs(mesh.vertices[5].z - 0.25f) < 1e-6f,
           "vertex = (coord + dual) / res + aabb_min");
    // First quad: x-edge of voxel (0,0,0) (lower corner (0,1,1) outside -> table order): voxels
    // (0,0,0),(0,0,1),(0,1,1),(0,1,0) = indices 0,1,3,2 ; split 0-2 when w0*w2 > w1*w3 (equal -> 1-3)
    Expect(mesh.faces[0][0] == 0 && mesh.faces[0][1] == 1 && mesh.faces[0][2] == 2 && mesh.faces[1][0] == 2 && mesh.faces[1][1] == 1 && mesh.faces[1][2] == 3,
           "x-edge quad 0,1,3,2 split along 1-3 with equal weights");
    wgt[0] = 3.f; wgt[3] = 3.f;   // w0*w2 = 9 > w1*w3 = 1 -> split 0-2
    mesh = tr::dual_grid_to_mesh(in, &st);
    Expect(mesh.faces[0][0] == 0 && mesh.faces[0][1] == 1 && mesh.faces[0][2] == 3 && mesh.faces[1][0] == 0 && mesh.faces[1][1] == 3 && mesh.faces[1][2] == 2,
           "split along 0-2 when w0*w2 > w1*w3");
    // Missing neighbour: drop voxel (0,1,1) -> its corner of the x-edge quad becomes a virtual centre vertex.
    std::vector<tr::Coord> fewer = coords;
    fewer.erase(fewer.begin() + 3);
    std::vector<float> dual2(fewer.size() * 3, 0.5f), wgt2(fewer.size(), 1.f);
    std::vector<std::uint8_t> inter2(fewer.size() * 3, 0);
    inter2[0 * 3 + 0] = inter2[0 * 3 + 1] = inter2[0 * 3 + 2] = 1;
    inter2[3 * 3 + 0] = 1;   // (1,0,0) is now index 3
    inter2[2 * 3 + 1] = 1;
    inter2[1 * 3 + 2] = 1;
    tr::DualGridInput in2 = in;
    in2.coords = &fewer; in2.dual = dual2.data(); in2.intersected = inter2.data(); in2.split_weight = wgt2.data();
    mesh = tr::dual_grid_to_mesh(in2, &st);
    Expect(st.quads == 6 && st.virtual_vertices == 1 && mesh.vertices.size() == 8 && st.open_edges == 0,
           "missing voxel -> virtual vertex, still closed (" + std::to_string(st.quads) + "/" + std::to_string(st.virtual_vertices) + "/" + std::to_string(st.open_edges) + ")");
    Expect(std::fabs(mesh.vertices[7].x + 0.25f) < 1e-6f && std::fabs(mesh.vertices[7].y - 0.25f) < 1e-6f && std::fabs(mesh.vertices[7].z - 0.25f) < 1e-6f,
           "virtual vertex at the voxel centre");
    // Decoding
    std::vector<float> out7 = {0.f, 10.f, -10.f, 1.f, -1.f, 0.f, 0.f};
    std::vector<float> d, sw;
    std::vector<std::uint8_t> it;
    tr::decode_shape_outputs(out7.data(), 1, &d, &it, &sw);
    Expect(std::fabs(d[0] - 0.5f) < 1e-6f && d[1] > 1.49f && d[2] < -0.49f, "dual = 2*sigmoid - 0.5");
    Expect(it[0] == 1 && it[1] == 0 && it[2] == 0, "intersected = logit > 0");
    Expect(std::fabs(sw[0] - std::log(2.f)) < 1e-6f, "softplus(0) = ln 2");

    // Sheet splitting on a 3x3x3 block: the inside corners (2,1,2) and (1,2,2) are diagonal on the
    // face between u = (1,1,1) and u' = (1,1,2); each is enclosed by its six edges (owner voxel =
    // corner minus one in the other axes). Both u and u' contain both corners -> two sheets each.
    std::vector<tr::Coord> block;
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 3; ++z) block.push_back({x, y, z});
    auto row = [](int x, int y, int z) { return static_cast<std::size_t>((x * 3 + y) * 3 + z); };
    std::vector<float> dual3(block.size() * 3, 0.5f), wgt3(block.size(), 1.f);
    std::vector<std::uint8_t> inter3(block.size() * 3, 0);
    auto flag = [&](int cx, int cy, int cz, int axis, int dir) {   // edge from corner c along +axis (dir 1) or -axis (dir -1)
        int c[3] = {cx, cy, cz};
        if (dir < 0) c[axis] -= 1;
        const int v[3] = {c[0] - (axis != 0), c[1] - (axis != 1), c[2] - (axis != 2)};
        inter3[row(v[0], v[1], v[2]) * 3 + static_cast<std::size_t>(axis)] = 1;
    };
    for (int axis = 0; axis < 3; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) { flag(2, 1, 2, axis, dir); flag(1, 2, 2, axis, dir); }
    tr::DualGridInput in3;
    in3.coords = &block; in3.dual = dual3.data(); in3.intersected = inter3.data(); in3.split_weight = wgt3.data();
    in3.resolution = 3;
    in3.sign_smooth_passes = 0;
    mesh = tr::dual_grid_to_mesh(in3, &st);
    Expect(st.quads == 12 && mesh.faces.size() == 24 && st.seam_edges == 0, "two enclosed corners -> twelve quads (" + std::to_string(st.quads) + ")");
    Expect(st.extra_vertices == 2 && mesh.vertices.size() == 29,
           "two sheets in u and u' -> two extra vertices (" + std::to_string(st.extra_vertices) + ")");
    Expect(st.open_edges == 0, "two sheets: closed (" + std::to_string(st.open_edges) + ")");
    Expect(tr::max_edge_multiplicity(mesh) == 2, "no edge with more than two triangles (" + std::to_string(tr::max_edge_multiplicity(mesh)) + ")");
    Expect(Volume(mesh) > 0, "two sheets: outward winding");
    // Vertex ids follow the voxel order with the extra sheet right after its voxel: u = row 13 -> 13, 14; u' = row 14 -> 15, 16.
    const float ux = (1.f + 0.5f) / 3.f - 0.5f, uz2 = (2.f + 0.5f) / 3.f - 0.5f;
    Expect(std::fabs(mesh.vertices[14].x - ux) < 1e-6f && std::fabs(mesh.vertices[14].z - ux) < 1e-6f && std::fabs(mesh.vertices[16].z - uz2) < 1e-6f,
           "sheet vertices sit at their voxel");
    std::size_t uses13 = 0, uses14 = 0;
    for (const auto& f : mesh.faces) for (auto v : f) { uses13 += v == 13; uses14 += v == 14; }
    Expect(uses13 >= 3 && uses14 >= 3 && uses13 + uses14 <= 12, "each sheet vertex of u is used by its three quads (" + std::to_string(uses13) + "/" + std::to_string(uses14) + ")");
    // Parity defect: drop one of the twelve flags -> a seam, but still closed.
    inter3[row(1, 0, 1) * 3 + 0] = 0;   // x-edge (1,1,2)->(2,1,2)
    mesh = tr::dual_grid_to_mesh(in3, &st);
    Expect(st.seam_edges > 0 && st.open_edges == 0, "parity defect -> seam, closed (" + std::to_string(st.seam_edges) + "/" + std::to_string(st.open_edges) + ")");

    // Random flags on a 5x5x5 block: whatever the flags, the mesh is closed.
    std::vector<tr::Coord> big;
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
            for (int z = 0; z < 5; ++z) big.push_back({x, y, z});
    std::vector<float> dualb(big.size() * 3, 0.5f), wgtb(big.size(), 1.f);
    std::vector<std::uint8_t> interb(big.size() * 3, 0);
    std::uint32_t rng = 12345u;
    std::size_t worst = 0, splits = 0, open = 0;
    for (int trial = 0; trial < 40; ++trial) {
        for (auto& v : interb) { rng = rng * 1664525u + 1013904223u; v = (rng >> 24) < (trial < 20 ? 128u : 220u) ? 1 : 0; }
        tr::DualGridInput inb;
        inb.coords = &big; inb.dual = dualb.data(); inb.intersected = interb.data(); inb.split_weight = wgtb.data();
        inb.resolution = 5;
        inb.sign_smooth_passes = 0;
        mesh = tr::dual_grid_to_mesh(inb, &st);
        worst = std::max(worst, tr::max_edge_multiplicity(mesh));
        splits += st.extra_vertices;
        open += st.open_edges;
        if (st.open_edges && open == st.open_edges) {   // first failing trial: show a few boundary edges
            std::vector<std::uint64_t> keys;
            for (const auto& f : mesh.faces)
                for (int e = 0; e < 3; ++e) {
                    const std::uint32_t a = f[static_cast<std::size_t>(e)], b = f[static_cast<std::size_t>((e + 1) % 3)];
                    keys.push_back((static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b));
                }
            std::sort(keys.begin(), keys.end());
            int shown = 0;
            for (std::size_t i = 0; i < keys.size() && shown < 6; ++i) {
                if ((i + 1 < keys.size() && keys[i + 1] == keys[i]) || (i > 0 && keys[i - 1] == keys[i])) continue;
                const std::uint32_t a = static_cast<std::uint32_t>(keys[i] >> 32), b = static_cast<std::uint32_t>(keys[i]);
                const std::size_t nv = big.size() + st.extra_vertices;
                std::cerr << "  open edge " << a << (a >= nv ? "(virtual)" : "") << " - " << b << (b >= nv ? "(virtual)" : "")
                          << " at (" << mesh.vertices[a].x << ',' << mesh.vertices[a].y << ',' << mesh.vertices[a].z << ") - ("
                          << mesh.vertices[b].x << ',' << mesh.vertices[b].y << ',' << mesh.vertices[b].z << ")\n";
                ++shown;
            }
            std::cerr << "  trial " << trial << ": quads " << st.quads << ", extra " << st.extra_vertices << ", virtual " << st.virtual_vertices
                      << ", seams " << st.seam_edges << ", open " << st.open_edges << ", max mult " << tr::max_edge_multiplicity(mesh) << "\n";
        }
    }
    Expect(open == 0, "random flags: always closed (" + std::to_string(open) + " open edges)");
    Expect(worst <= 4, "random flags: at most four triangles per edge (" + std::to_string(worst) + ")");
    Expect(splits > 0, "random flags produce some sheet splits (" + std::to_string(splits) + ")");
    // Majority smoothing: one enclosed corner in a 5x5x5 block is a sliver (1 of 27 inside) and vanishes
    // with one pass; the random-flag meshes stay closed with smoothing on.
    {
        std::vector<std::uint8_t> one(big.size() * 3, 0);
        auto rowb = [](int x, int y, int z) { return static_cast<std::size_t>((x * 5 + y) * 5 + z); };
        auto flagb = [&](int cx, int cy, int cz, int ax, int dir) {
            int c[3] = {cx, cy, cz};
            if (dir < 0) c[ax] -= 1;
            one[rowb(c[0] - (ax != 0), c[1] - (ax != 1), c[2] - (ax != 2)) * 3 + static_cast<std::size_t>(ax)] = 1;
        };
        for (int ax = 0; ax < 3; ++ax) for (int dir = -1; dir <= 1; dir += 2) flagb(2, 2, 2, ax, dir);
        tr::DualGridInput ins;
        ins.coords = &big; ins.dual = dualb.data(); ins.intersected = one.data(); ins.split_weight = wgtb.data();
        ins.resolution = 5;
        ins.sign_smooth_passes = 0;
        mesh = tr::dual_grid_to_mesh(ins, &st);
        Expect(st.quads == 6 && st.smoothed_corners == 0, "lone corner without smoothing: six quads");
        ins.sign_smooth_passes = 1;
        mesh = tr::dual_grid_to_mesh(ins, &st);
        Expect(st.quads == 0 && st.smoothed_corners == 1 && st.open_edges == 0, "one majority pass removes the sliver (" + std::to_string(st.quads) + "/" + std::to_string(st.smoothed_corners) + ")");
        rng = 777u;
        std::size_t open_s = 0;
        for (int trial = 0; trial < 10; ++trial) {
            for (auto& v : interb) { rng = rng * 1664525u + 1013904223u; v = (rng >> 24) < 128u ? 1 : 0; }
            tr::DualGridInput inr;
            inr.coords = &big; inr.dual = dualb.data(); inr.intersected = interb.data(); inr.split_weight = wgtb.data();
            inr.resolution = 5;
            inr.sign_smooth_passes = 1;
            mesh = tr::dual_grid_to_mesh(inr, &st);
            open_s += st.open_edges;
        }
        Expect(open_s == 0, "random flags with smoothing: still closed");
    }
    if (failures == 0) std::cout << "trellis2_dual_grid_mesh_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
