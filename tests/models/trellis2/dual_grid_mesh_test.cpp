// CPU tests for src/models/trellis2/dual_grid_mesh.cpp: a 2x2x2 block of voxels with one
// intersected edge per axis at the origin voxel gives three quads (six triangles) whose
// corners are the four voxels around each edge; missing neighbours skip the edge; the
// split follows the quad weights; vertex placement; boundary edge counting; the output
// decoding (sigmoid with margin, logit sign, softplus).
// AI-assisted (Claude Code), 2026-11-12.
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
}  // namespace

int main() {
    std::vector<tr::Coord> coords;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) coords.push_back({x, y, z});
    const std::size_t n = coords.size();
    std::vector<float> dual(n * 3, 0.5f);
    std::vector<std::uint8_t> inter(n * 3, 0);
    inter[0] = inter[1] = inter[2] = 1;   // voxel (0,0,0): x, y and z edges intersected
    std::vector<float> wgt(n, 1.f);
    tr::DualGridInput in;
    in.coords = &coords; in.dual = dual.data(); in.intersected = inter.data(); in.split_weight = wgt.data();
    in.resolution = 2;
    tr::DualGridStats st;
    tr::Mesh mesh = tr::dual_grid_to_mesh(in, &st);
    Expect(st.quads == 3 && mesh.faces.size() == 6 && st.skipped_edges == 0, "three quads -> six triangles");
    Expect(mesh.vertices.size() == 8, "one vertex per voxel");
    // vertex of voxel (1,0,1) at index 5: (1 + 0.5) * 0.5 - 0.5 = 0.25, (0+0.5)*0.5-0.5 = -0.25
    Expect(std::fabs(mesh.vertices[5].x - 0.25f) < 1e-6f && std::fabs(mesh.vertices[5].y + 0.25f) < 1e-6f && std::fabs(mesh.vertices[5].z - 0.25f) < 1e-6f,
           "vertex = (coord + dual) / res + aabb_min");
    // x-edge quad: voxels (0,0,0),(0,0,1),(0,1,1),(0,1,0) = indices 0,1,3,2 ; split 0-2 when w0*w2 > w1*w3 (equal -> 1-3)
    Expect(mesh.faces[0][0] == 0 && mesh.faces[0][1] == 1 && mesh.faces[0][2] == 2 && mesh.faces[1][0] == 2 && mesh.faces[1][1] == 1 && mesh.faces[1][2] == 3,
           "x-edge quad 0,1,3,2 split along 1-3 with equal weights");
    wgt[0] = 3.f; wgt[3] = 3.f;   // w0*w2 = 9 > w1*w3 = 1 -> split 0-2
    mesh = tr::dual_grid_to_mesh(in, &st);
    Expect(mesh.faces[0][0] == 0 && mesh.faces[0][1] == 1 && mesh.faces[0][2] == 3 && mesh.faces[1][0] == 0 && mesh.faces[1][1] == 3 && mesh.faces[1][2] == 2,
           "split along 0-2 when w0*w2 > w1*w3");
    // open edges: the three quads share vertex 0 pairwise with one edge each (0-1, 0-2, 0-4): 12 - 2*3 = 6
    Expect(st.open_edges == 6, "6 boundary edges for the three quads (" + std::to_string(st.open_edges) + ")");
    // missing neighbour: drop voxel (0,1,1) -> the x-edge of (0,0,0) is skipped; y- and z-edge quads remain
    std::vector<tr::Coord> fewer = coords;
    fewer.erase(fewer.begin() + 3);
    std::vector<float> dual2(fewer.size() * 3, 0.5f), wgt2(fewer.size(), 1.f);
    std::vector<std::uint8_t> inter2(fewer.size() * 3, 0);
    inter2[0] = inter2[1] = inter2[2] = 1;
    tr::DualGridInput in2 = in;
    in2.coords = &fewer; in2.dual = dual2.data(); in2.intersected = inter2.data(); in2.split_weight = wgt2.data();
    mesh = tr::dual_grid_to_mesh(in2, &st);
    Expect(st.quads == 2 && st.skipped_edges == 1, "edges whose four voxels are not all active are skipped (" + std::to_string(st.quads) + "/" + std::to_string(st.skipped_edges) + ")");
    // decoding
    std::vector<float> out7 = {0.f, 10.f, -10.f, 1.f, -1.f, 0.f, 0.f};
    std::vector<float> d, sw;
    std::vector<std::uint8_t> it;
    tr::decode_shape_outputs(out7.data(), 1, &d, &it, &sw);
    Expect(std::fabs(d[0] - 0.5f) < 1e-6f && d[1] > 1.49f && d[2] < -0.49f, "dual = 2*sigmoid - 0.5");
    Expect(it[0] == 1 && it[1] == 0 && it[2] == 0, "intersected = logit > 0");
    Expect(std::fabs(sw[0] - std::log(2.f)) < 1e-6f, "softplus(0) = ln 2");
    if (failures == 0) std::cout << "trellis2_dual_grid_mesh_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
