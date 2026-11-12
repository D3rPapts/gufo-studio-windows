// TRELLIS.2: flexible dual grid -> mesh (see dual_grid_mesh.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/dual_grid_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "src/models/trellis2/config.hpp"

namespace gufo::trellis2 {

void decode_shape_outputs(const float* out7, std::size_t n, std::vector<float>* dual, std::vector<std::uint8_t>* intersected,
                          std::vector<float>* split_weight, float margin) {
    dual->resize(n * 3);
    intersected->resize(n * 3);
    split_weight->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float* o = out7 + i * 7;
        for (int a = 0; a < 3; ++a) {
            const float s = 1.f / (1.f + std::exp(-o[a]));
            (*dual)[i * 3 + static_cast<std::size_t>(a)] = (1.f + 2.f * margin) * s - margin;
            (*intersected)[i * 3 + static_cast<std::size_t>(a)] = o[3 + a] > 0.f ? 1 : 0;
        }
        const float w = o[6];
        (*split_weight)[i] = w > 20.f ? w : std::log1p(std::exp(w));   // softplus
    }
}

Mesh dual_grid_to_mesh(const DualGridInput& in, DualGridStats* stats) {
    Mesh mesh;
    const auto& coords = *in.coords;
    const std::size_t n = coords.size();
    const float voxel = (in.aabb_max - in.aabb_min) / static_cast<float>(in.resolution);
    mesh.vertices.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        mesh.vertices[i] = {(static_cast<float>(coords[i].x) + in.dual[i * 3]) * voxel + in.aabb_min,
                            (static_cast<float>(coords[i].y) + in.dual[i * 3 + 1]) * voxel + in.aabb_min,
                            (static_cast<float>(coords[i].z) + in.dual[i * 3 + 2]) * voxel + in.aabb_min};
    }
    VoxelHashMap map;
    map.Build(coords);
    DualGridStats st;
    const auto& off = MeshConfig::edge_neighbour_offsets;
    for (std::size_t i = 0; i < n; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!in.intersected[i * 3 + static_cast<std::size_t>(axis)]) continue;
            std::uint32_t q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                const std::int64_t idx = map.Find(coords[i].x + off[axis][k][0], coords[i].y + off[axis][k][1], coords[i].z + off[axis][k][2]);
                if (idx < 0) ok = false;
                else q[k] = static_cast<std::uint32_t>(idx);
            }
            if (!ok) { ++st.skipped_edges; continue; }
            ++st.quads;
            bool split_02;
            if (in.split_weight) {
                split_02 = in.split_weight[q[0]] * in.split_weight[q[2]] > in.split_weight[q[1]] * in.split_weight[q[3]];
            } else {
                auto normal = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c, float* out) {
                    const MeshVertex& A = mesh.vertices[a];
                    const MeshVertex& B = mesh.vertices[b];
                    const MeshVertex& C = mesh.vertices[c];
                    const float u[3] = {B.x - A.x, B.y - A.y, B.z - A.z}, v[3] = {C.x - A.x, C.y - A.y, C.z - A.z};
                    out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
                };
                float n0[3], n1[3], m0[3], m1[3];
                normal(q[0], q[1], q[2], n0); normal(q[1], q[2], q[3], n1);
                normal(q[0], q[1], q[3], m0); normal(q[1], q[2], q[3], m1);
                const float align0 = std::fabs(n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]);
                const float align1 = std::fabs(m0[0] * m1[0] + m0[1] * m1[1] + m0[2] * m1[2]);
                split_02 = align0 > align1;
            }
            if (split_02) {
                mesh.faces.push_back({q[0], q[1], q[2]});
                mesh.faces.push_back({q[0], q[2], q[3]});
            } else {
                mesh.faces.push_back({q[0], q[1], q[3]});
                mesh.faces.push_back({q[3], q[1], q[2]});
            }
        }
    }
    if (stats) {
        st.open_edges = count_boundary_edges(mesh);
        *stats = st;
    }
    return mesh;
}

std::size_t count_boundary_edges(const Mesh& mesh) {
    std::unordered_map<std::uint64_t, int> edges;
    edges.reserve(mesh.faces.size() * 3);
    for (const auto& f : mesh.faces)
        for (int e = 0; e < 3; ++e) {
            const std::uint32_t a = f[static_cast<std::size_t>(e)], b = f[static_cast<std::size_t>((e + 1) % 3)];
            const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
            ++edges[key];
        }
    std::size_t open = 0;
    for (const auto& [k, c] : edges) open += c == 1;
    return open;
}

}  // namespace gufo::trellis2
