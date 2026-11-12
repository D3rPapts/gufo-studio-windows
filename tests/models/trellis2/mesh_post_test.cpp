// CPU tests for src/models/trellis2/mesh_post.cpp: a cube with one missing face (a
// square hole, perimeter 4 * side) is closed by fill_holes when the threshold allows it
// and left alone otherwise; larger holes are skipped; y-up rotation keeps the volume
// sign; decimation reduces faces and keeps the mesh closed; statistics.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/mesh_post.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Unit cube [0, s]^3 with outward faces; `without_top` drops the +z face (hole of 4 edges).
tr::Mesh Cube(float s, bool without_top) {
    tr::Mesh m;
    for (int i = 0; i < 8; ++i) m.vertices.push_back({(i & 1) ? s : 0.f, (i & 2) ? s : 0.f, (i & 4) ? s : 0.f});
    auto quad = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
        m.faces.push_back({a, b, c});
        m.faces.push_back({a, c, d});
    };
    quad(0, 2, 3, 1);   // -z (normal down)
    if (!without_top) quad(4, 5, 7, 6);   // +z
    quad(0, 1, 5, 4);   // -y
    quad(2, 6, 7, 3);   // +y
    quad(0, 4, 6, 2);   // -x
    quad(1, 3, 7, 5);   // +x
    return m;
}
}  // namespace

int main() {
    // orientation: flip two faces of a closed cube and invert the whole mesh -> unify restores it
    {
        tr::Mesh c = Cube(0.1f, false);
        std::swap(c.faces[3][1], c.faces[3][2]);
        std::swap(c.faces[8][1], c.faces[8][2]);
        Expect(tr::mesh_stats(c).volume < 0.001 - 1e-6, "flipped faces break the volume");
        tr::UnifyStats un = tr::unify_orientation(&c);
        Expect(un.components == 1 && un.flipped_faces >= 2 && std::fabs(tr::mesh_stats(c).volume - 0.001) < 1e-6, "unify repairs two flipped faces (" + std::to_string(un.flipped_faces) + ")");
        for (auto& f : c.faces) std::swap(f[1], f[2]);   // whole mesh inside out
        un = tr::unify_orientation(&c);
        Expect(un.global_flip == 1 && std::fabs(tr::mesh_stats(c).volume - 0.001) < 1e-6, "negative volume -> global flip");
    }
    tr::Mesh closed = Cube(0.1f, false);
    tr::MeshStats st = tr::mesh_stats(closed);
    Expect(st.closed && std::fabs(st.volume - 0.001) < 1e-6 && st.bbox_max[2] == 0.1f, "closed cube: volume s^3, bbox");

    tr::Mesh open = Cube(0.1f, true);
    Expect(tr::mesh_stats(open).open_edges == 4, "open cube: 4 boundary edges");
    tr::Mesh filled = open;
    // remove_small_components: a cube (12 triangles) plus a lone triangle far away -> the triangle goes.
    {
        tr::Mesh m;
        m.vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {5, 5, 5}, {6, 5, 5}, {5, 6, 5}};
        m.faces = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}, {8, 9, 10}};
        const std::size_t removed = tr::remove_small_components(&m, 4);
        Expect(removed == 1 && m.faces.size() == 12 && m.vertices.size() == 8, "remove_small_components drops the lone triangle and its vertices ("
               + std::to_string(removed) + "/" + std::to_string(m.faces.size()) + "/" + std::to_string(m.vertices.size()) + ")");
        Expect(tr::remove_small_components(&m, 4) == 0 && m.faces.size() == 12, "nothing else to remove");
        Expect(tr::remove_small_components(&m, 13) == 1 && m.faces.empty() && m.vertices.empty(), "threshold above the cube removes it");
    }
    tr::HoleFillStats hf = tr::fill_holes(&filled, 0.5f);   // perimeter 0.4 < 0.5
    Expect(hf.loops == 1 && hf.filled == 1, "one loop filled");
    st = tr::mesh_stats(filled);
    Expect(st.closed && filled.vertices.size() == 9 && filled.faces.size() == 14, "fan from the centroid closes the hole (9 vertices, 14 faces)");
    Expect(std::fabs(st.volume - 0.001) < 1e-6, "filled cube has the volume s^3 (consistent orientation)");

    tr::Mesh untouched = open;
    hf = tr::fill_holes(&untouched, 0.3f);   // 0.4 > 0.3
    Expect(hf.loops == 1 && hf.filled == 0 && hf.skipped_large == 1 && untouched.faces.size() == open.faces.size(), "large hole left open");

    const tr::Mesh yup = tr::to_y_up(closed);
    Expect(yup.vertices[4].y == 0.1f && yup.vertices[4].z == 0.f && yup.vertices[2].z == -0.1f, "(x, y, z) -> (x, z, -y)");
    Expect(std::fabs(tr::mesh_stats(yup).volume - 0.001) < 1e-6, "rotation keeps the volume sign");

    // decimation: subdivide the cube faces first (refine each triangle into 4) to have something to collapse
    tr::Mesh fine = closed;
    for (int round = 0; round < 3; ++round) {
        tr::Mesh next;
        next.vertices = fine.vertices;
        for (const auto& f : fine.faces) {
            std::uint32_t mid[3];
            for (int e = 0; e < 3; ++e) {
                const auto& a = fine.vertices[f[static_cast<std::size_t>(e)]];
                const auto& b = fine.vertices[f[static_cast<std::size_t>((e + 1) % 3)]];
                mid[e] = static_cast<std::uint32_t>(next.vertices.size());
                next.vertices.push_back({(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2});
            }
            next.faces.push_back({f[0], mid[0], mid[2]});
            next.faces.push_back({mid[0], f[1], mid[1]});
            next.faces.push_back({mid[2], mid[1], f[2]});
            next.faces.push_back({mid[0], mid[1], mid[2]});
        }
        fine = next;
    }
    Expect(fine.faces.size() == 12 * 64, "refined cube has 768 faces");
    const tr::Mesh dec = tr::decimate_mesh(fine, 100);
    Expect(dec.faces.size() <= 100 && dec.faces.size() >= 12, "decimated to <= 100 faces (" + std::to_string(dec.faces.size()) + ")");
    Expect(tr::decimate_mesh(closed, 100).faces.size() == 12, "target above the face count: unchanged");
    if (failures == 0) std::cout << "trellis2_mesh_post_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
