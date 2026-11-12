// TRELLIS.2: mesh post-processing (see mesh_post.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/mesh_post.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

#include "src/models/trellis2/dual_grid_mesh.hpp"
#include "src/models/triposg/marching_cubes.hpp"
#include "src/models/triposg/mesh_decimate.hpp"

namespace gufo::trellis2 {

HoleFillStats fill_holes(Mesh* mesh, float max_perimeter) {
    HoleFillStats st;
    // Directed edges a->b of the faces, sorted; an edge is a boundary edge when it occurs once
    // and its reverse b->a is absent. The hole loop runs b->a: next[b] = a. Vertices with more
    // than one outgoing boundary edge are non-manifold -> their loops are skipped.
    auto key = [](std::uint32_t a, std::uint32_t b) { return (static_cast<std::uint64_t>(a) << 32) | b; };
    std::vector<std::uint64_t> directed;
    directed.reserve(mesh->faces.size() * 3);
    for (const auto& f : mesh->faces)
        for (int e = 0; e < 3; ++e) directed.push_back(key(f[static_cast<std::size_t>(e)], f[static_cast<std::size_t>((e + 1) % 3)]));
    std::sort(directed.begin(), directed.end());
    auto present = [&](std::uint64_t k) { return std::binary_search(directed.begin(), directed.end(), k); };
    std::unordered_map<std::uint32_t, std::uint32_t> next;
    std::unordered_map<std::uint32_t, int> out_degree;
    for (std::size_t i = 0; i < directed.size();) {
        std::size_t j = i + 1;
        while (j < directed.size() && directed[j] == directed[i]) ++j;
        if (j - i == 1) {
            const std::uint32_t a = static_cast<std::uint32_t>(directed[i] >> 32), b = static_cast<std::uint32_t>(directed[i] & 0xFFFFFFFFu);
            if (!present(key(b, a))) {
                next[b] = a;
                ++out_degree[b];
            }
        }
        i = j;
    }
    std::unordered_map<std::uint32_t, bool> visited;
    for (const auto& [start, _] : next) {
        if (visited[start]) continue;
        std::vector<std::uint32_t> loop;
        std::uint32_t v = start;
        bool manifold = true;
        while (!visited[v]) {
            visited[v] = true;
            if (out_degree[v] != 1) manifold = false;
            loop.push_back(v);
            auto it = next.find(v);
            if (it == next.end()) { manifold = false; break; }
            v = it->second;
        }
        if (v != start) manifold = false;
        ++st.loops;
        if (!manifold || loop.size() < 3) { ++st.skipped_nonmanifold; continue; }
        double perimeter = 0;
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const MeshVertex& p = mesh->vertices[loop[i]];
            const MeshVertex& q = mesh->vertices[loop[(i + 1) % loop.size()]];
            perimeter += std::sqrt(static_cast<double>(p.x - q.x) * (p.x - q.x) + static_cast<double>(p.y - q.y) * (p.y - q.y) + static_cast<double>(p.z - q.z) * (p.z - q.z));
        }
        if (perimeter > max_perimeter) { ++st.skipped_large; st.skipped_perimeters.push_back(static_cast<float>(perimeter)); continue; }
        if (loop.size() == 3) {
            // loop runs b->a... (opposite to the face side): the triangle (l0, l1, l2) closes it.
            mesh->faces.push_back({loop[0], loop[1], loop[2]});
        } else {
            MeshVertex c{0.f, 0.f, 0.f};
            for (auto idx : loop) { c.x += mesh->vertices[idx].x; c.y += mesh->vertices[idx].y; c.z += mesh->vertices[idx].z; }
            c.x /= static_cast<float>(loop.size()); c.y /= static_cast<float>(loop.size()); c.z /= static_cast<float>(loop.size());
            const auto ci = static_cast<std::uint32_t>(mesh->vertices.size());
            mesh->vertices.push_back(c);
            for (std::size_t i = 0; i < loop.size(); ++i) mesh->faces.push_back({loop[i], loop[(i + 1) % loop.size()], ci});
        }
        ++st.filled;
    }
    std::sort(st.skipped_perimeters.begin(), st.skipped_perimeters.end(), std::greater<float>());
    return st;
}

Mesh decimate_mesh(const Mesh& mesh, std::size_t target_faces) {
    if (target_faces == 0 || mesh.faces.size() <= target_faces) return mesh;
    triposg::Mesh tm;
    tm.vertices.reserve(mesh.vertices.size());
    for (const auto& v : mesh.vertices) tm.vertices.push_back({v.x, v.y, v.z});
    tm.faces = mesh.faces;
    triposg::DecimateOptions d;
    d.target_faces = target_faces;
    const triposg::Mesh out = triposg::compact(triposg::decimate(tm, d));
    Mesh result;
    result.vertices.reserve(out.vertices.size());
    for (const auto& v : out.vertices) result.vertices.push_back({v.x, v.y, v.z});
    result.faces = out.faces;
    return result;
}

Mesh to_y_up(const Mesh& mesh) {
    Mesh out = mesh;
    for (auto& v : out.vertices) v = {v.x, v.z, -v.y};
    return out;
}

MeshStats mesh_stats(const Mesh& mesh) {
    MeshStats s;
    for (int a = 0; a < 3; ++a) { s.bbox_min[a] = std::numeric_limits<float>::max(); s.bbox_max[a] = -std::numeric_limits<float>::max(); }
    for (const auto& v : mesh.vertices) {
        s.bbox_min[0] = std::min(s.bbox_min[0], v.x); s.bbox_max[0] = std::max(s.bbox_max[0], v.x);
        s.bbox_min[1] = std::min(s.bbox_min[1], v.y); s.bbox_max[1] = std::max(s.bbox_max[1], v.y);
        s.bbox_min[2] = std::min(s.bbox_min[2], v.z); s.bbox_max[2] = std::max(s.bbox_max[2], v.z);
    }
    if (mesh.vertices.empty()) for (int a = 0; a < 3; ++a) s.bbox_min[a] = s.bbox_max[a] = 0.f;
    double vol = 0;
    for (const auto& f : mesh.faces) {
        const MeshVertex& a = mesh.vertices[f[0]];
        const MeshVertex& b = mesh.vertices[f[1]];
        const MeshVertex& c = mesh.vertices[f[2]];
        vol += (static_cast<double>(a.x) * (static_cast<double>(b.y) * c.z - static_cast<double>(b.z) * c.y) -
                static_cast<double>(a.y) * (static_cast<double>(b.x) * c.z - static_cast<double>(b.z) * c.x) +
                static_cast<double>(a.z) * (static_cast<double>(b.x) * c.y - static_cast<double>(b.y) * c.x)) / 6.0;
    }
    s.volume = vol;
    s.open_edges = count_boundary_edges(mesh);
    s.closed = s.open_edges == 0;
    return s;
}

}  // namespace gufo::trellis2
