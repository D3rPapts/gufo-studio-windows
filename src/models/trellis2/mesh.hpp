// TRELLIS.2: triangle mesh container shared by the dual-grid extraction, the GLB writer
// and the software preview (same layout as the TripoSG port).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace gufo::trellis2 {

struct MeshVertex {
    float x, y, z;
};

struct Mesh {
    std::vector<MeshVertex> vertices;
    std::vector<std::array<std::uint32_t, 3>> faces;  // counter-clockwise seen from outside
};

}  // namespace gufo::trellis2
