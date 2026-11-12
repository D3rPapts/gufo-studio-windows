// TRELLIS.2: PNG writer and voxel slice mosaic (see png.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/png.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <zlib.h>

namespace gufo::trellis2 {
namespace {

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

std::vector<std::uint8_t> encode_png(const std::uint8_t* pixels, int width, int height, int channels) {
    std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<std::uint8_t> ihdr;
    put_be32(ihdr, static_cast<std::uint32_t>(width));
    put_be32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.insert(ihdr.end(), {8, static_cast<std::uint8_t>(channels == 4 ? 6 : 2), 0, 0, 0});
    chunk(png, "IHDR", ihdr);
    const std::size_t row = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(height) * (row + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);   // filter: none
        raw.insert(raw.end(), pixels + static_cast<std::size_t>(y) * row, pixels + static_cast<std::size_t>(y + 1) * row);
    }
    uLongf dest_len = compressBound(static_cast<uLong>(raw.size()));
    std::vector<std::uint8_t> z(dest_len);
    if (compress2(z.data(), &dest_len, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) dest_len = 0;
    z.resize(dest_len);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});
    return png;
}

std::vector<std::uint8_t> encode_png_rgb(const std::uint8_t* rgb, int width, int height) { return encode_png(rgb, width, height, 3); }
std::vector<std::uint8_t> encode_png_rgba(const std::uint8_t* rgba, int width, int height) { return encode_png(rgba, width, height, 4); }

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

std::vector<std::uint8_t> slice_mosaic(const float* grid, int res, float lo, float hi, int scale, int* width, int* height) {
    const int tiles = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(res))));
    const int tile = res * scale + 1;
    *width = tiles * tile;
    *height = static_cast<int>((res + tiles - 1) / tiles) * tile;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(*width) * static_cast<std::size_t>(*height) * 3, 40);
    const float range = hi > lo ? hi - lo : 1.f;
    for (int z = 0; z < res; ++z) {
        const int tx = (z % tiles) * tile, ty = (z / tiles) * tile;
        for (int y = 0; y < res; ++y)
            for (int x = 0; x < res; ++x) {
                const float v = grid[(static_cast<std::size_t>(x) * res + y) * res + z];
                const float g = std::clamp((v - lo) / range, 0.f, 1.f);
                const std::uint8_t c = static_cast<std::uint8_t>(g * 255.f);
                for (int sy = 0; sy < scale; ++sy)
                    for (int sx = 0; sx < scale; ++sx) {
                        std::uint8_t* p = rgb.data() + (static_cast<std::size_t>(ty + y * scale + sy) * static_cast<std::size_t>(*width) +
                                                        static_cast<std::size_t>(tx + x * scale + sx)) * 3;
                        p[0] = c; p[1] = c; p[2] = c;
                    }
            }
    }
    return rgb;
}

}  // namespace gufo::trellis2
