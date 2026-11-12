// TRELLIS.2 stage 9: UV unwrap and texture baking (see texture_bake.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/texture_bake.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "src/models/trellis2/config.hpp"
#include "xatlas.h"

namespace gufo::trellis2 {

namespace {

int Root(std::vector<std::uint32_t>& parent, std::uint32_t i) {
    while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return static_cast<int>(i);
}

bool UnwrapBox(const Mesh& mesh, int texture_size, UvMesh* out, std::string* error, const UnwrapOptions& o) {
    const std::size_t F = mesh.faces.size(), V = mesh.vertices.size();
    // Face normals.
    std::vector<float> fn(F * 3, 0.f);
    for (std::size_t f = 0; f < F; ++f) {
        const MeshVertex& a = mesh.vertices[mesh.faces[f][0]];
        const MeshVertex& b = mesh.vertices[mesh.faces[f][1]];
        const MeshVertex& c = mesh.vertices[mesh.faces[f][2]];
        const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        float n[3] = {uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 0.f) for (int k = 0; k < 3; ++k) fn[f * 3 + static_cast<std::size_t>(k)] = n[k] / len;
    }
    // Edge adjacency: up to three neighbours per face (first other face on each edge).
    std::vector<std::pair<std::uint64_t, std::uint32_t>> edges;
    edges.reserve(F * 3);
    for (std::uint32_t f = 0; f < F; ++f)
        for (int e = 0; e < 3; ++e) {
            const std::uint32_t a = mesh.faces[f][static_cast<std::size_t>(e)], b = mesh.faces[f][static_cast<std::size_t>((e + 1) % 3)];
            edges.emplace_back((static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b), f);
        }
    std::sort(edges.begin(), edges.end());
    std::vector<std::int32_t> adj(F * 3, -1);
    std::vector<std::uint8_t> adj_n(F, 0);
    auto link = [&](std::uint32_t f, std::uint32_t g) {
        if (adj_n[f] < 3) adj[f * 3 + adj_n[f]++] = static_cast<std::int32_t>(g);
        if (adj_n[g] < 3) adj[g * 3 + adj_n[g]++] = static_cast<std::int32_t>(f);
    };
    for (std::size_t i = 0; i < edges.size();) {
        std::size_t j = i + 1;
        while (j < edges.size() && edges[j].first == edges[i].first) ++j;
        for (std::size_t k = i; k + 1 < j; k += 2) link(edges[k].second, edges[k + 1].second);
        i = j;
    }
    // Normal smoothing over neighbours.
    std::vector<float> sm(fn);
    for (int pass = 0; pass < o.normal_smoothing_passes; ++pass) {
        std::vector<float> next(F * 3);
        for (std::size_t f = 0; f < F; ++f) {
            float n[3] = {sm[f * 3], sm[f * 3 + 1], sm[f * 3 + 2]};
            for (int k = 0; k < adj_n[f]; ++k) {
                const auto g = static_cast<std::size_t>(adj[f * 3 + static_cast<std::size_t>(k)]);
                n[0] += sm[g * 3]; n[1] += sm[g * 3 + 1]; n[2] += sm[g * 3 + 2];
            }
            const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (int k = 0; k < 3; ++k) next[f * 3 + static_cast<std::size_t>(k)] = len > 0.f ? n[k] / len : 0.f;
        }
        sm.swap(next);
    }
    // Bin = dominant axis and sign (0..5).
    std::vector<std::uint8_t> bin(F, 0);
    for (std::size_t f = 0; f < F; ++f) {
        const float* n = sm.data() + f * 3;
        int a = 0;
        if (std::fabs(n[1]) > std::fabs(n[a])) a = 1;
        if (std::fabs(n[2]) > std::fabs(n[a])) a = 2;
        bin[f] = static_cast<std::uint8_t>(a * 2 + (n[a] < 0.f ? 1 : 0));
    }
    // Charts: connected faces of one bin.
    std::vector<std::uint32_t> parent(F);
    for (std::uint32_t f = 0; f < F; ++f) parent[f] = f;
    for (std::uint32_t f = 0; f < F; ++f)
        for (int k = 0; k < adj_n[f]; ++k) {
            const auto g = static_cast<std::uint32_t>(adj[f * 3 + static_cast<std::size_t>(k)]);
            if (bin[g] == bin[f]) {
                const int ra = Root(parent, f), rb = Root(parent, g);
                if (ra != rb) parent[static_cast<std::size_t>(ra)] = static_cast<std::uint32_t>(rb);
            }
        }
    std::vector<std::uint32_t> size(F, 0);
    std::vector<std::uint8_t> axis(F, 0);
    for (std::uint32_t f = 0; f < F; ++f) ++size[static_cast<std::size_t>(Root(parent, f))];
    for (std::uint32_t f = 0; f < F; ++f) if (parent[f] == f) axis[f] = bin[f];
    // Merge small charts into the neighbouring chart they share most edges with (axis of the bigger one).
    for (int pass = 0; pass < 6; ++pass) {
        std::unordered_map<std::uint64_t, std::uint32_t> shared;   // (small root, other root) -> edges
        for (std::uint32_t f = 0; f < F; ++f) {
            const int rf = Root(parent, f);
            if (size[static_cast<std::size_t>(rf)] >= static_cast<std::uint32_t>(o.min_chart_faces)) continue;
            for (int k = 0; k < adj_n[f]; ++k) {
                const int rg = Root(parent, static_cast<std::uint32_t>(adj[f * 3 + static_cast<std::size_t>(k)]));
                if (rg != rf) ++shared[(static_cast<std::uint64_t>(rf) << 32) | static_cast<std::uint32_t>(rg)];
            }
        }
        if (shared.empty()) break;
        std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> best;   // small root -> (other, edges)
        for (const auto& [key, n] : shared) {
            const auto a = static_cast<std::uint32_t>(key >> 32), b = static_cast<std::uint32_t>(key);
            auto it = best.find(a);
            if (it == best.end() || n > it->second.second) best[a] = {b, n};
        }
        bool merged = false;
        for (const auto& [a, target] : best) {
            const int ra = Root(parent, a), rb = Root(parent, target.first);
            if (ra == rb) continue;
            const bool keep_b = size[static_cast<std::size_t>(rb)] >= size[static_cast<std::size_t>(ra)];
            const int keep = keep_b ? rb : ra, drop = keep_b ? ra : rb;
            parent[static_cast<std::size_t>(drop)] = static_cast<std::uint32_t>(keep);
            size[static_cast<std::size_t>(keep)] += size[static_cast<std::size_t>(drop)];
            merged = true;
        }
        if (!merged) break;
    }
    // Chart ids and projections.
    std::vector<std::int32_t> chart_of_root(F, -1);
    struct Chart { std::uint8_t axis; float umin, umax, vmin, vmax; float x0, y0, scale; std::uint32_t faces; };
    std::vector<Chart> charts;
    std::vector<std::uint32_t> face_chart(F);
    for (std::uint32_t f = 0; f < F; ++f) {
        const int r = Root(parent, f);
        if (chart_of_root[static_cast<std::size_t>(r)] < 0) {
            chart_of_root[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(charts.size());
            charts.push_back({axis[static_cast<std::size_t>(r)], 1e30f, -1e30f, 1e30f, -1e30f, 0.f, 0.f, 1.f, 0});
        }
        face_chart[f] = static_cast<std::uint32_t>(chart_of_root[static_cast<std::size_t>(r)]);
        ++charts[face_chart[f]].faces;
    }
    auto project = [](const MeshVertex& p, std::uint8_t ax, float* u, float* v) {
        const float c[3] = {p.x, p.y, p.z};
        const int a = ax / 2, b = (a + 1) % 3, cc = (a + 2) % 3;
        // mirror for the negative side so the texture is not flipped when seen from outside
        *u = (ax & 1) ? -c[b] : c[b];
        *v = c[cc];
    };
    // Vertices per (chart, vertex).
    std::unordered_map<std::uint64_t, std::uint32_t> ids;
    ids.reserve(V + V / 4);
    out->mesh.vertices.clear();
    out->mesh.faces.resize(F);
    out->source.clear();
    std::vector<std::uint32_t> vchart;
    for (std::uint32_t f = 0; f < F; ++f) {
        const std::uint32_t ch = face_chart[f];
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t v = mesh.faces[f][static_cast<std::size_t>(k)];
            const std::uint64_t key = (static_cast<std::uint64_t>(ch) << 32) | v;
            auto it = ids.find(key);
            std::uint32_t id;
            if (it == ids.end()) {
                id = static_cast<std::uint32_t>(out->mesh.vertices.size());
                ids.emplace(key, id);
                out->mesh.vertices.push_back(mesh.vertices[v]);
                out->source.push_back(v);
                vchart.push_back(ch);
                float u, vv;
                project(mesh.vertices[v], charts[ch].axis, &u, &vv);
                Chart& c = charts[ch];
                c.umin = std::min(c.umin, u); c.umax = std::max(c.umax, u); c.vmin = std::min(c.vmin, vv); c.vmax = std::max(c.vmax, vv);
            } else id = it->second;
            out->mesh.faces[f][static_cast<std::size_t>(k)] = id;
        }
    }
    // Shelf packing: charts sorted by height, scale shrunk until everything fits.
    const float S = static_cast<float>(texture_size), pad = static_cast<float>(o.padding);
    std::vector<std::uint32_t> order(charts.size());
    for (std::uint32_t i = 0; i < charts.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return (charts[a].vmax - charts[a].vmin) > (charts[b].vmax - charts[b].vmin);
    });
    double area = 0;
    for (const Chart& c : charts) area += static_cast<double>(c.umax - c.umin) * static_cast<double>(c.vmax - c.vmin);
    float scale = area > 0 ? static_cast<float>(std::sqrt(o.fill_target * static_cast<double>(S) * S / area)) : 1.f;
    bool packed = false;
    for (int attempt = 0; attempt < 40 && !packed; ++attempt) {
        float x = pad, y = pad, shelf = 0.f;
        packed = true;
        for (std::uint32_t i : order) {
            Chart& c = charts[i];
            const float w = (c.umax - c.umin) * scale, h = (c.vmax - c.vmin) * scale;
            if (w + 2 * pad > S || h + 2 * pad > S) { packed = false; break; }
            if (x + w + pad > S) { x = pad; y += shelf + pad; shelf = 0.f; }
            if (y + h + pad > S) { packed = false; break; }
            c.x0 = x; c.y0 = y; c.scale = scale;
            x += w + pad;
            shelf = std::max(shelf, h);
        }
        if (!packed) scale *= 0.92f;
    }
    if (!packed) {
        if (error) *error = "unwrap: charts do not fit into the atlas";
        return false;
    }
    out->uv.resize(out->mesh.vertices.size() * 2);
    for (std::size_t i = 0; i < out->mesh.vertices.size(); ++i) {
        const Chart& c = charts[vchart[i]];
        float u, v;
        project(out->mesh.vertices[i], c.axis, &u, &v);
        out->uv[i * 2] = std::clamp((c.x0 + (u - c.umin) * c.scale) / S, 0.f, 1.f);
        out->uv[i * 2 + 1] = std::clamp((c.y0 + (v - c.vmin) * c.scale) / S, 0.f, 1.f);
    }
    out->charts = charts.size();
    out->atlas_width = out->atlas_height = texture_size;
    return true;
}

}  // namespace

bool unwrap_uv(const Mesh& mesh, int texture_size, UvMesh* out, std::string* error, const UnwrapOptions& options) {
    if (options.box_projection) {
        if (mesh.vertices.empty() || mesh.faces.empty()) { if (error) *error = "unwrap: empty mesh"; return false; }
        if (texture_size < 64) { if (error) *error = "unwrap: texture size below 64"; return false; }
        return UnwrapBox(mesh, texture_size, out, error, options);
    }
    if (mesh.vertices.empty() || mesh.faces.empty()) {
        if (error) *error = "unwrap: empty mesh";
        return false;
    }
    if (texture_size < 64) {
        if (error) *error = "unwrap: texture size below 64";
        return false;
    }
    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::MeshDecl decl;
    decl.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
    decl.vertexPositionData = mesh.vertices.data();
    decl.vertexPositionStride = sizeof(MeshVertex);
    decl.indexCount = static_cast<std::uint32_t>(mesh.faces.size() * 3);
    decl.indexData = mesh.faces.data();
    decl.indexFormat = xatlas::IndexFormat::UInt32;
    const xatlas::AddMeshError add = xatlas::AddMesh(atlas, decl, 1);
    if (add != xatlas::AddMeshError::Success) {
        if (error) *error = std::string("unwrap: ") + xatlas::StringForEnum(add);
        xatlas::Destroy(atlas);
        return false;
    }
    xatlas::ChartOptions chart;
    chart.maxCost = options.max_cost;
    chart.normalDeviationWeight = options.normal_deviation_weight;
    chart.straightnessWeight = options.straightness_weight;
    chart.normalSeamWeight = options.normal_seam_weight;
    chart.roundnessWeight = options.roundness_weight;
    chart.maxIterations = static_cast<std::uint32_t>(std::max(1, options.iterations));
    xatlas::PackOptions pack;
    pack.resolution = static_cast<std::uint32_t>(texture_size);
    pack.padding = static_cast<std::uint32_t>(std::max(0, options.padding));
    pack.bilinear = true;
    pack.blockAlign = false;
    // texelsPerUnit 0 with a resolution lets xatlas estimate the scale for one atlas; when the
    // estimate overflows into a second atlas, shrink the charts and pack again.
    xatlas::ComputeCharts(atlas, chart);
    float texels_per_unit = 0.f;
    for (int attempt = 0; attempt < 6; ++attempt) {
        pack.texelsPerUnit = texels_per_unit;
        xatlas::PackCharts(atlas, pack);
        if (atlas->atlasCount <= 1) break;
        texels_per_unit = (texels_per_unit > 0.f ? texels_per_unit : atlas->texelsPerUnit) * 0.8f;
    }
    if (atlas->meshCount != 1 || atlas->atlasCount != 1 || atlas->width == 0 || atlas->height == 0) {
        if (error) *error = "unwrap: charts do not fit into one atlas (" + std::to_string(atlas->atlasCount) + " atlases)";
        xatlas::Destroy(atlas);
        return false;
    }
    const xatlas::Mesh& m = atlas->meshes[0];
    out->mesh.vertices.resize(m.vertexCount);
    out->uv.resize(static_cast<std::size_t>(m.vertexCount) * 2);
    out->source.resize(m.vertexCount);
    const float sx = 1.f / static_cast<float>(atlas->width), sy = 1.f / static_cast<float>(atlas->height);
    for (std::uint32_t i = 0; i < m.vertexCount; ++i) {
        const xatlas::Vertex& v = m.vertexArray[i];
        out->mesh.vertices[i] = mesh.vertices[v.xref];
        out->source[i] = v.xref;
        out->uv[i * 2] = std::clamp(v.uv[0] * sx, 0.f, 1.f);
        out->uv[i * 2 + 1] = std::clamp(v.uv[1] * sy, 0.f, 1.f);
    }
    out->mesh.faces.resize(m.indexCount / 3);
    for (std::uint32_t f = 0; f < m.indexCount / 3; ++f)
        out->mesh.faces[f] = {m.indexArray[f * 3], m.indexArray[f * 3 + 1], m.indexArray[f * 3 + 2]};
    out->charts = atlas->chartCount;
    out->atlas_width = static_cast<int>(atlas->width);
    out->atlas_height = static_cast<int>(atlas->height);
    xatlas::Destroy(atlas);
    return true;
}

VoxelAttrSampler::VoxelAttrSampler(const std::vector<Coord>& coords, const std::vector<float>& attrs, int channels, int resolution,
                                   int max_radius)
    : coords_(coords), attrs_(attrs), channels_(channels), resolution_(resolution), max_radius_(max_radius) {
    map_.Build(coords);
}

bool VoxelAttrSampler::Sample(float x, float y, float z, float* out) const {
    const float res = static_cast<float>(resolution_);
    const float g[3] = {(x - MeshConfig::aabb_min) * res - 0.5f, (y - MeshConfig::aabb_min) * res - 0.5f, (z - MeshConfig::aabb_min) * res - 0.5f};
    int i0[3];
    float fr[3];
    for (int d = 0; d < 3; ++d) {
        const float fl = std::floor(g[d]);
        i0[d] = static_cast<int>(fl);
        fr[d] = g[d] - fl;
    }
    float acc[16] = {};
    float total = 0.f;
    const int nc = std::min(channels_, 16);
    for (int k = 0; k < 8; ++k) {
        const int cx = i0[0] + (k & 1), cy = i0[1] + ((k >> 1) & 1), cz = i0[2] + (k >> 2);
        if (cx < 0 || cy < 0 || cz < 0) continue;
        const std::int64_t row = map_.Find(cx, cy, cz);
        if (row < 0) continue;
        const float w = ((k & 1) ? fr[0] : 1.f - fr[0]) * (((k >> 1) & 1) ? fr[1] : 1.f - fr[1]) * ((k >> 2) ? fr[2] : 1.f - fr[2]);
        if (w <= 0.f) continue;
        const float* a = attrs_.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(channels_);
        for (int c = 0; c < nc; ++c) acc[c] += w * a[c];
        total += w;
    }
    if (total > 1e-6f) {
        for (int c = 0; c < nc; ++c) out[c] = acc[c] / total;
        return true;
    }
    // Nearest active voxel in growing Chebyshev shells around the rounded position.
    const int rx = static_cast<int>(std::lround(g[0])), ry = static_cast<int>(std::lround(g[1])), rz = static_cast<int>(std::lround(g[2]));
    for (int r = 1; r <= max_radius_; ++r) {
        std::int64_t best = -1;
        float best_d = 0.f;
        for (int dx = -r; dx <= r; ++dx)
            for (int dy = -r; dy <= r; ++dy)
                for (int dz = -r; dz <= r; ++dz) {
                    if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
                    const int cx = rx + dx, cy = ry + dy, cz = rz + dz;
                    if (cx < 0 || cy < 0 || cz < 0) continue;
                    const std::int64_t row = map_.Find(cx, cy, cz);
                    if (row < 0) continue;
                    const float ex = static_cast<float>(cx) - g[0], ey = static_cast<float>(cy) - g[1], ez = static_cast<float>(cz) - g[2];
                    const float d = ex * ex + ey * ey + ez * ez;
                    if (best < 0 || d < best_d) { best = row; best_d = d; }
                }
        if (best >= 0) {
            const float* a = attrs_.data() + static_cast<std::size_t>(best) * static_cast<std::size_t>(channels_);
            for (int c = 0; c < nc; ++c) out[c] = a[c];
            return true;
        }
    }
    return false;
}

BakedTextures bake_textures(const UvMesh& mesh, const VoxelAttrSampler& sampler, int size, int dilate_px) {
    BakedTextures bt;
    bt.size = size;
    const std::size_t texels = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    std::vector<float> attr(texels * 6, 0.f);
    std::vector<std::uint8_t> mask(texels, 0);
    const float S = static_cast<float>(size);
    for (const auto& f : mesh.mesh.faces) {
        float px[3], py[3];
        for (int k = 0; k < 3; ++k) {
            px[k] = mesh.uv[static_cast<std::size_t>(f[static_cast<std::size_t>(k)]) * 2] * S;
            py[k] = mesh.uv[static_cast<std::size_t>(f[static_cast<std::size_t>(k)]) * 2 + 1] * S;
        }
        const float area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
        if (std::fabs(area) < 1e-12f) continue;
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min({px[0], px[1], px[2]}))));
        const int x1 = std::min(size - 1, static_cast<int>(std::ceil(std::max({px[0], px[1], px[2]}))));
        const int y0 = std::max(0, static_cast<int>(std::floor(std::min({py[0], py[1], py[2]}))));
        const int y1 = std::min(size - 1, static_cast<int>(std::ceil(std::max({py[0], py[1], py[2]}))));
        const MeshVertex& A = mesh.mesh.vertices[f[0]];
        const MeshVertex& B = mesh.mesh.vertices[f[1]];
        const MeshVertex& C = mesh.mesh.vertices[f[2]];
        const float inv = 1.f / area;
        // Texels whose centre lies inside the triangle grown by half a texel (so thin triangles still
        // leave a footprint); the barycentrics are clamped for those border texels.
        const float grow = 0.5f / std::sqrt(std::fabs(area));   // ~half texel in barycentric units
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float cx = static_cast<float>(x) + 0.5f, cy = static_cast<float>(y) + 0.5f;
                float w0 = ((px[1] - cx) * (py[2] - cy) - (px[2] - cx) * (py[1] - cy)) * inv;
                float w1 = ((px[2] - cx) * (py[0] - cy) - (px[0] - cx) * (py[2] - cy)) * inv;
                float w2 = 1.f - w0 - w1;
                if (w0 < -grow || w1 < -grow || w2 < -grow) continue;
                w0 = std::max(w0, 0.f); w1 = std::max(w1, 0.f); w2 = std::max(w2, 0.f);
                const float sum = w0 + w1 + w2;
                w0 /= sum; w1 /= sum; w2 /= sum;
                const std::size_t t = static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x);
                if (mask[t]) continue;
                const float p[3] = {w0 * A.x + w1 * B.x + w2 * C.x, w0 * A.y + w1 * B.y + w2 * C.y, w0 * A.z + w1 * B.z + w2 * C.z};
                float a[16] = {0.5f, 0.5f, 0.5f, 0.f, 0.9f, 1.f};
                if (!sampler.Sample(p[0], p[1], p[2], a)) ++bt.unsampled_texels;
                std::memcpy(attr.data() + t * 6, a, 6 * sizeof(float));
                mask[t] = 1;
                ++bt.covered_texels;
            }
    }
    // Dilation: every uncovered texel next to a covered one takes the mean of its covered neighbours.
    std::vector<std::uint8_t> next_mask(texels);
    for (int it = 0; it < dilate_px; ++it) {
        next_mask = mask;
        bool any = false;
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                const std::size_t t = static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x);
                if (mask[t]) continue;
                float acc[6] = {};
                int n = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy) continue;
                        const int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                        const std::size_t u = static_cast<std::size_t>(ny) * static_cast<std::size_t>(size) + static_cast<std::size_t>(nx);
                        if (!mask[u]) continue;
                        for (int c = 0; c < 6; ++c) acc[c] += attr[u * 6 + static_cast<std::size_t>(c)];
                        ++n;
                    }
                if (!n) continue;
                for (int c = 0; c < 6; ++c) attr[t * 6 + static_cast<std::size_t>(c)] = acc[c] / static_cast<float>(n);
                next_mask[t] = 1;
                any = true;
            }
        mask.swap(next_mask);
        if (!any) break;
    }
    bt.base_color.assign(texels * 4, 0);
    bt.metallic_roughness.assign(texels * 3, 0);
    auto to8 = [](float v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f)); };
    for (std::size_t t = 0; t < texels; ++t) {
        const float* a = attr.data() + t * 6;
        bt.base_color[t * 4] = to8(a[0]); bt.base_color[t * 4 + 1] = to8(a[1]); bt.base_color[t * 4 + 2] = to8(a[2]);
        bt.base_color[t * 4 + 3] = mask[t] ? to8(a[5]) : 255;
        bt.metallic_roughness[t * 3] = 0;
        bt.metallic_roughness[t * 3 + 1] = to8(a[4]);
        bt.metallic_roughness[t * 3 + 2] = to8(a[3]);
    }
    return bt;
}

}  // namespace gufo::trellis2
