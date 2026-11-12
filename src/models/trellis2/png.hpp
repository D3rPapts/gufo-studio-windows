// TRELLIS.2: dependency-free PNG writer (RGB8, stored zlib blocks; same as the TripoSG
// preview writer) and a slice mosaic for voxel grids (occupancy / logits as images).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gufo::trellis2 {

std::vector<std::uint8_t> encode_png_rgb(const std::uint8_t* rgb, int width, int height);
bool write_png_rgb(const std::filesystem::path& path, const std::uint8_t* rgb, int width, int height, std::string* error);

// Mosaic of `res` z-slices (each res x res, laid out in a square grid, `scale` pixels per
// voxel) of a [res, res, res] grid indexed (x, y, z): value v -> grey 0..255 via
// (v - lo) / (hi - lo); slice z at tile z, row = y, column = x. Returns RGB8 and its size.
std::vector<std::uint8_t> slice_mosaic(const float* grid, int res, float lo, float hi, int scale, int* width, int* height);

}  // namespace gufo::trellis2
