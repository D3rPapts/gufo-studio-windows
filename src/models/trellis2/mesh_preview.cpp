// TRELLIS.2 (from the TripoSG port): software mesh preview and a dependency-free PNG writer.
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/mesh_preview.hpp"

#include "src/models/trellis2/png.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace gufo::trellis2 {
namespace {

struct Vec3 {
    float x, y, z;
};

Vec3 rotate(const MeshVertex& v, float yaw, float pitch) {
    // Yaw around y (vertical), then pitch around x; camera looks along -z.
    const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
    const float x1 = cy * v.x + sy * v.z;
    const float z1 = -sy * v.x + cy * v.z;
    const float y2 = cp * v.y - sp * z1;
    const float z2 = sp * v.y + cp * z1;
    return {x1, y2, z2};
}


}  // namespace

bool bounding_box(const Mesh& mesh, float* bmin, float* bmax) {
    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();
    for (const auto& v : mesh.vertices) {
        bmin[0] = std::min(bmin[0], v.x); bmax[0] = std::max(bmax[0], v.x);
        bmin[1] = std::min(bmin[1], v.y); bmax[1] = std::max(bmax[1], v.y);
        bmin[2] = std::min(bmin[2], v.z); bmax[2] = std::max(bmax[2], v.z);
    }
    if (mesh.vertices.empty()) {
        bmin[0] = bmin[1] = bmin[2] = bmax[0] = bmax[1] = bmax[2] = 0.f;
        return false;
    }
    return true;
}


std::vector<std::uint8_t> render_preview(const Mesh& mesh, const PreviewOptions& o, int* width, int* height) {
    const int views = static_cast<int>(o.yaw_degrees.size());
    const int S = o.size;
    *width = S * views;
    *height = S;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(*width) * static_cast<std::size_t>(*height) * 3);
    for (std::size_t i = 0; i < rgb.size(); i += 3) { rgb[i] = o.background[0]; rgb[i + 1] = o.background[1]; rgb[i + 2] = o.background[2]; }
    if (mesh.vertices.empty() || mesh.faces.empty()) return rgb;
    float bmin[3], bmax[3];
    bounding_box(mesh, bmin, bmax);
    const float centre[3] = {(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2};
    float radius = 0.f;
    for (const auto& v : mesh.vertices)
        radius = std::max(radius, std::sqrt((v.x - centre[0]) * (v.x - centre[0]) + (v.y - centre[1]) * (v.y - centre[1]) + (v.z - centre[2]) * (v.z - centre[2])));
    if (radius <= 0.f) radius = 1.f;
    const float scale = 0.45f * static_cast<float>(S) / radius;
    const float light[3] = {0.4f, 0.7f, 0.6f};
    const float ll = std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
    std::vector<float> zbuf(static_cast<std::size_t>(S) * static_cast<std::size_t>(S));
    std::vector<Vec3> p(mesh.vertices.size());
    for (int view = 0; view < views; ++view) {
        const float yaw = o.yaw_degrees[static_cast<std::size_t>(view)] * 3.14159265f / 180.f;
        const float pitch = o.pitch_degrees * 3.14159265f / 180.f;
        std::fill(zbuf.begin(), zbuf.end(), -std::numeric_limits<float>::max());
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            MeshVertex c{mesh.vertices[i].x - centre[0], mesh.vertices[i].y - centre[1], mesh.vertices[i].z - centre[2]};
            Vec3 r = rotate(c, yaw, pitch);
            p[i] = {static_cast<float>(S) / 2 + r.x * scale, static_cast<float>(S) / 2 - r.y * scale, r.z};
        }
        for (const auto& f : mesh.faces) {
            const Vec3& a = p[f[0]];
            const Vec3& b = p[f[1]];
            const Vec3& c = p[f[2]];
            // Flat normal in view space (screen y is flipped, so use the original rotated coordinates).
            const float ux = b.x - a.x, uy = -(b.y - a.y), uz = b.z - a.z;
            const float vx = c.x - a.x, vy = -(c.y - a.y), vz = c.z - a.z;
            float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (nl <= 0.f) continue;
            nx /= nl; ny /= nl; nz /= nl;
            float lambert = (nx * light[0] + ny * light[1] + nz * light[2]) / ll;
            if (nz < 0.f) lambert = -lambert;  // two-sided: back faces lit as if flipped
            lambert = std::max(0.f, lambert);
            const float shade = 0.25f + 0.75f * lambert;
            float base[3] = {static_cast<float>(o.colour[0]), static_cast<float>(o.colour[1]), static_cast<float>(o.colour[2])};
            if (o.vertex_colors && o.vertex_colors->size() == mesh.vertices.size() * 4)
                for (int ch = 0; ch < 3; ++ch)
                    base[ch] = 255.f * ((*o.vertex_colors)[f[0] * 4 + static_cast<std::size_t>(ch)] + (*o.vertex_colors)[f[1] * 4 + static_cast<std::size_t>(ch)] +
                                        (*o.vertex_colors)[f[2] * 4 + static_cast<std::size_t>(ch)]) / 3.f;
            const int x0 = std::max(0, static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
            const int x1 = std::min(S - 1, static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
            const int y0 = std::max(0, static_cast<int>(std::floor(std::min({a.y, b.y, c.y}))));
            const int y1 = std::min(S - 1, static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));
            const float det = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
            if (std::fabs(det) < 1e-12f) continue;
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const float px = static_cast<float>(x) + 0.5f, py = static_cast<float>(y) + 0.5f;
                    const float w0 = ((b.x - px) * (c.y - py) - (c.x - px) * (b.y - py)) / det;
                    const float w1 = ((c.x - px) * (a.y - py) - (a.x - px) * (c.y - py)) / det;
                    const float w2 = 1.f - w0 - w1;
                    if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
                    const float z = w0 * a.z + w1 * b.z + w2 * c.z;
                    float& zb = zbuf[static_cast<std::size_t>(y) * static_cast<std::size_t>(S) + static_cast<std::size_t>(x)];
                    if (z <= zb) continue;
                    zb = z;
                    std::uint8_t* px_out = rgb.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(*width) + static_cast<std::size_t>(view * S + x)) * 3;
                    for (int ch = 0; ch < 3; ++ch) px_out[ch] = static_cast<std::uint8_t>(std::min(255.f, base[ch] * shade));
                }
        }
    }
    return rgb;
}

}  // namespace gufo::trellis2
