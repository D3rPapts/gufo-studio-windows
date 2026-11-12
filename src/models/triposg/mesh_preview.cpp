// TripoSG: software mesh preview and a dependency-free PNG writer.
// AI-assisted (Claude Code), 2026-11-12.
#include "mesh_preview.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace gufo::triposg {
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

std::uint32_t crc_table[256];
bool crc_ready = false;

std::uint32_t crc32(const std::uint8_t* data, std::size_t n, std::uint32_t crc = 0xFFFFFFFFu) {
    if (!crc_ready) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            crc_table[i] = c;
        }
        crc_ready = true;
    }
    for (std::size_t i = 0; i < n; ++i) crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

void put_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v >> 24)); out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 8)); out.push_back(static_cast<std::uint8_t>(v));
}

void chunk(std::vector<std::uint8_t>& out, const char* type, const std::vector<std::uint8_t>& data) {
    put_be32(out, static_cast<std::uint32_t>(data.size()));
    std::vector<std::uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    out.insert(out.end(), body.begin(), body.end());
    put_be32(out, crc32(body.data(), body.size()) ^ 0xFFFFFFFFu);
}

}  // namespace

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
                    for (int ch = 0; ch < 3; ++ch) px_out[ch] = static_cast<std::uint8_t>(std::min(255.f, o.colour[ch] * shade));
                }
        }
    }
    return rgb;
}

std::vector<std::uint8_t> encode_png_rgb(const std::uint8_t* rgb, int width, int height) {
    std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<std::uint8_t> ihdr;
    put_be32(ihdr, static_cast<std::uint32_t>(width));
    put_be32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8 bit, RGB, deflate, filter 0, no interlace
    chunk(png, "IHDR", ihdr);
    // Raw scanlines with filter byte 0.
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(height) * (static_cast<std::size_t>(width) * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3,
                   rgb + static_cast<std::size_t>(y + 1) * static_cast<std::size_t>(width) * 3);
    }
    // zlib stream with stored blocks (max 65535 bytes each) and Adler-32.
    std::vector<std::uint8_t> z = {0x78, 0x01};
    std::size_t pos = 0;
    while (pos < raw.size() || raw.empty()) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
        const bool last = pos + n >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<std::uint8_t>(n & 0xFF)); z.push_back(static_cast<std::uint8_t>(n >> 8));
        z.push_back(static_cast<std::uint8_t>(~n & 0xFF)); z.push_back(static_cast<std::uint8_t>((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
        if (raw.empty()) break;
    }
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t byte : raw) { a = (a + byte) % 65521; b = (b + a) % 65521; }
    put_be32(z, (b << 16) | a);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});
    return png;
}

bool write_png_rgb(const std::filesystem::path& path, const std::uint8_t* rgb, int width, int height, std::string* error) {
    const std::vector<std::uint8_t> png = encode_png_rgb(rgb, width, height);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "cannot write " + path.string();
        return false;
    }
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return static_cast<bool>(out);
}

}  // namespace gufo::triposg
