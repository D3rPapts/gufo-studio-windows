// TripoSG: quadric-error edge collapse (see mesh_decimate.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "mesh_decimate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gufo::triposg {
namespace {

// Symmetric 4x4 quadric stored as 10 coefficients.
struct Quadric {
    double q[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    void add_plane(double a, double b, double c, double d, double w) {
        q[0] += w * a * a; q[1] += w * a * b; q[2] += w * a * c; q[3] += w * a * d;
        q[4] += w * b * b; q[5] += w * b * c; q[6] += w * b * d;
        q[7] += w * c * c; q[8] += w * c * d;
        q[9] += w * d * d;
    }
    Quadric& operator+=(const Quadric& o) {
        for (int i = 0; i < 10; ++i) q[i] += o.q[i];
        return *this;
    }
    double eval(double x, double y, double z) const {
        return q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z +
               2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9];
    }
};

struct Candidate {
    double cost;
    std::uint32_t a, b;
    std::uint32_t version;  // sum of vertex versions at push time (lazy invalidation)
    bool operator<(const Candidate& o) const { return cost > o.cost; }  // min-heap
};

std::uint64_t edge_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32) | b;
}

}  // namespace

Mesh compact(const Mesh& mesh) {
    Mesh out;
    std::vector<std::uint32_t> remap(mesh.vertices.size(), UINT32_MAX);
    for (const auto& f : mesh.faces) {
        if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2]) continue;
        std::array<std::uint32_t, 3> g{};
        for (int k = 0; k < 3; ++k) {
            std::uint32_t& r = remap[f[static_cast<std::size_t>(k)]];
            if (r == UINT32_MAX) {
                r = static_cast<std::uint32_t>(out.vertices.size());
                out.vertices.push_back(mesh.vertices[f[static_cast<std::size_t>(k)]]);
            }
            g[static_cast<std::size_t>(k)] = r;
        }
        out.faces.push_back(g);
    }
    return out;
}

Mesh decimate(const Mesh& input, const DecimateOptions& options) {
    if (input.faces.size() <= options.target_faces) return compact(input);
    const std::size_t nv = input.vertices.size();
    std::vector<std::array<double, 3>> pos(nv);
    for (std::size_t i = 0; i < nv; ++i) pos[i] = {input.vertices[i].x, input.vertices[i].y, input.vertices[i].z};
    std::vector<std::array<std::uint32_t, 3>> faces = input.faces;
    std::vector<bool> face_alive(faces.size(), true);
    std::vector<std::vector<std::uint32_t>> vertex_faces(nv);
    std::vector<Quadric> quadric(nv);
    float bmin[3], bmax[3];
    bounding_box(input, bmin, bmax);
    const double diag2 = static_cast<double>(bmax[0] - bmin[0]) * (bmax[0] - bmin[0]) + static_cast<double>(bmax[1] - bmin[1]) * (bmax[1] - bmin[1]) +
                         static_cast<double>(bmax[2] - bmin[2]) * (bmax[2] - bmin[2]);
    const double error_limit = options.max_error > 0 ? options.max_error * diag2 : 1e300;

    auto plane = [&](std::size_t fi, double* n, double* d) -> bool {
        const auto& f = faces[fi];
        const auto& a = pos[f[0]];
        const auto& b = pos[f[1]];
        const auto& c = pos[f[2]];
        const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const double vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        n[0] = uy * vz - uz * vy; n[1] = uz * vx - ux * vz; n[2] = ux * vy - uy * vx;
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len <= 0) return false;
        n[0] /= len; n[1] /= len; n[2] /= len;
        *d = -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2]);
        return true;
    };
    for (std::size_t fi = 0; fi < faces.size(); ++fi) {
        double n[3], d;
        for (auto v : faces[fi]) vertex_faces[v].push_back(static_cast<std::uint32_t>(fi));
        if (!plane(fi, n, &d)) continue;
        for (auto v : faces[fi]) quadric[v].add_plane(n[0], n[1], n[2], d, 1.0);
    }
    std::vector<std::uint32_t> version(nv, 0);
    std::vector<bool> vertex_alive(nv, true);

    auto collapse_cost = [&](std::uint32_t a, std::uint32_t b, std::array<double, 3>* best) {
        Quadric q = quadric[a];
        q += quadric[b];
        const std::array<double, 3> mid = {(pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2, (pos[a][2] + pos[b][2]) / 2};
        const std::array<std::array<double, 3>, 3> options3 = {pos[a], pos[b], mid};
        double best_cost = 1e300;
        for (const auto& p : options3) {
            const double c = q.eval(p[0], p[1], p[2]);
            if (c < best_cost) { best_cost = c; *best = p; }
        }
        return best_cost;
    };

    std::priority_queue<Candidate> heap;
    std::unordered_set<std::uint64_t> seen;
    auto push_edges_of = [&](std::uint32_t v) {
        for (auto fi : vertex_faces[v]) {
            if (!face_alive[fi]) continue;
            for (auto u : faces[fi]) {
                if (u == v) continue;
                const std::uint64_t key = edge_key(v, u);
                if (!seen.insert(key).second) continue;
                std::array<double, 3> p;
                const double c = collapse_cost(v, u, &p);
                heap.push({c, std::min(v, u), std::max(v, u), version[v] + version[u]});
            }
        }
    };
    for (std::uint32_t v = 0; v < nv; ++v) push_edges_of(v);
    seen.clear();

    std::size_t alive_faces = faces.size();
    while (alive_faces > options.target_faces && !heap.empty()) {
        const Candidate c = heap.top();
        heap.pop();
        const std::uint32_t a = c.a, b = c.b;
        if (!vertex_alive[a] || !vertex_alive[b] || version[a] + version[b] != c.version) continue;
        if (c.cost > error_limit) break;
        std::array<double, 3> target;
        collapse_cost(a, b, &target);
        // Link condition (manifold preservation): the vertices adjacent to both a and b
        // must be exactly the third vertices of the faces shared by the edge (a, b).
        {
            std::vector<std::uint32_t> na, nb, shared_third;
            for (auto fi : vertex_faces[a]) {
                if (!face_alive[fi]) continue;
                const auto& f = faces[fi];
                const bool has_b = f[0] == b || f[1] == b || f[2] == b;
                for (auto v : f) {
                    if (v == a) continue;
                    if (has_b && v != b) shared_third.push_back(v);
                    if (v != b) na.push_back(v);
                }
            }
            for (auto fi : vertex_faces[b]) {
                if (!face_alive[fi]) continue;
                for (auto v : faces[fi]) if (v != b && v != a) nb.push_back(v);
            }
            std::sort(na.begin(), na.end()); na.erase(std::unique(na.begin(), na.end()), na.end());
            std::sort(nb.begin(), nb.end()); nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
            std::sort(shared_third.begin(), shared_third.end());
            shared_third.erase(std::unique(shared_third.begin(), shared_third.end()), shared_third.end());
            std::vector<std::uint32_t> common;
            std::set_intersection(na.begin(), na.end(), nb.begin(), nb.end(), std::back_inserter(common));
            if (common != shared_third || shared_third.size() != 2) continue;
        }
        // Reject if any surviving face around a or b would flip (normal dot < 0) or degenerate.
        bool flip = false;
        for (auto v : {a, b}) {
            for (auto fi : vertex_faces[v]) {
                if (!face_alive[fi] || flip) continue;
                const auto& f = faces[fi];
                const bool shared = (f[0] == a || f[1] == a || f[2] == a) && (f[0] == b || f[1] == b || f[2] == b);
                if (shared) continue;  // this face disappears
                double n0[3], d0;
                if (!plane(fi, n0, &d0)) continue;
                std::array<double, 3> p[3] = {pos[f[0]], pos[f[1]], pos[f[2]]};
                for (int k = 0; k < 3; ++k) if (f[static_cast<std::size_t>(k)] == v) p[k] = target;
                const double ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
                const double vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1], vz = p[2][2] - p[0][2];
                const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
                const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len <= 1e-30 || (nx * n0[0] + ny * n0[1] + nz * n0[2]) / len < 0.1) flip = true;
            }
        }
        if (flip) continue;
        // Collapse b into a.
        pos[a] = target;
        quadric[a] += quadric[b];
        vertex_alive[b] = false;
        for (auto fi : vertex_faces[b]) {
            if (!face_alive[fi]) continue;
            auto& f = faces[fi];
            const bool shared = (f[0] == a || f[1] == a || f[2] == a);
            if (shared) {
                face_alive[fi] = false;
                --alive_faces;
                continue;
            }
            for (auto& v : f) if (v == b) v = a;
            vertex_faces[a].push_back(fi);
        }
        vertex_faces[b].clear();
        vertex_faces[b].shrink_to_fit();
        // Drop dead faces from a's list and dedupe.
        auto& fa = vertex_faces[a];
        fa.erase(std::remove_if(fa.begin(), fa.end(), [&](std::uint32_t fi) { return !face_alive[fi]; }), fa.end());
        std::sort(fa.begin(), fa.end());
        fa.erase(std::unique(fa.begin(), fa.end()), fa.end());
        ++version[a];
        push_edges_of(a);
        seen.clear();
    }

    Mesh out;
    std::vector<std::uint32_t> remap(nv, UINT32_MAX);
    for (std::size_t fi = 0; fi < faces.size(); ++fi) {
        if (!face_alive[fi]) continue;
        const auto& f = faces[fi];
        if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2]) continue;
        std::array<std::uint32_t, 3> g{};
        for (int k = 0; k < 3; ++k) {
            std::uint32_t& r = remap[f[static_cast<std::size_t>(k)]];
            if (r == UINT32_MAX) {
                r = static_cast<std::uint32_t>(out.vertices.size());
                const auto& p = pos[f[static_cast<std::size_t>(k)]];
                out.vertices.push_back(MeshVertex{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])});
            }
            g[static_cast<std::size_t>(k)] = r;
        }
        out.faces.push_back(g);
    }
    return out;
}

}  // namespace gufo::triposg
