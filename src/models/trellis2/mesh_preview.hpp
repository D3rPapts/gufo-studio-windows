// TRELLIS.2 (from the TripoSG port): software preview of a mesh (z-buffer, flat Lambert shading, orthographic),
// several views side by side, written as PNG (stored deflate, no dependencies).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "src/models/trellis2/mesh.hpp"

namespace gufo::trellis2 {

struct PreviewOptions {
    int size = 384;                  // pixels per view (square)
    std::vector<float> yaw_degrees = {30.f, 120.f, 210.f};  // views around the vertical axis
    float pitch_degrees = 20.f;      // camera elevation
    std::uint8_t background[3] = {245, 245, 245};
    std::uint8_t colour[3] = {200, 160, 110};
};

// Axis-aligned bounds of the mesh; false (zeros) for an empty mesh.
bool bounding_box(const Mesh& mesh, float* bmin, float* bmax);

// RGB8 image of width size * views, height size.
std::vector<std::uint8_t> render_preview(const Mesh& mesh, const PreviewOptions& options, int* width, int* height);

}  // namespace gufo::trellis2
