// TripoSG: software preview of a mesh (z-buffer, flat Lambert shading, orthographic),
// several views side by side, written as PNG (stored deflate, no dependencies).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "src/models/triposg/marching_cubes.hpp"

namespace gufo::triposg {

struct PreviewOptions {
    int size = 384;                  // pixels per view (square)
    std::vector<float> yaw_degrees = {30.f, 120.f, 210.f};  // views around the vertical axis
    float pitch_degrees = 20.f;      // camera elevation
    std::uint8_t background[3] = {245, 245, 245};
    std::uint8_t colour[3] = {200, 160, 110};
};

// RGB8 image of width size * views, height size.
std::vector<std::uint8_t> render_preview(const Mesh& mesh, const PreviewOptions& options, int* width, int* height);

// Minimal PNG writer (RGB8, stored zlib blocks).
bool write_png_rgb(const std::filesystem::path& path, const std::uint8_t* rgb, int width, int height, std::string* error);

}  // namespace gufo::triposg
