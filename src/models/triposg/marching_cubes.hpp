// TripoSG: CPU marching cubes (classic 256-case table) for SDF grids.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace gufo::triposg {

struct MeshVertex {
    float x, y, z;
};

struct Mesh {
    std::vector<MeshVertex> vertices;
    std::vector<std::array<std::uint32_t, 3>> faces;  // counter-clockwise seen from outside
};

// Regular scalar grid of (nx * ny * nz) samples, index = (ix * ny + iy) * nz + iz
// (same memory order as a C array grid[nx][ny][nz] and as the torch view used by TripoSG).
struct ScalarGrid {
    const float* values = nullptr;
    std::size_t nx = 0, ny = 0, nz = 0;
    // World coordinate of sample 0 and of sample n-1 on each axis.
    float min[3] = {-1.f, -1.f, -1.f};
    float max[3] = {1.f, 1.f, 1.f};

    float at(std::size_t ix, std::size_t iy, std::size_t iz) const {
        return values[(ix * ny + iy) * nz + iz];
    }
};

struct MarchingCubesOptions {
    float iso_level = 0.f;
    // TripoSG's decoder emits -SDF (positive inside). With inside_positive=true the
    // emitted triangles face outward for such a field; false for a plain SDF (negative inside).
    bool inside_positive = true;
    // Merge coincident vertices on shared edges (true gives an indexed, closed mesh).
    bool weld = true;
};

// Extracts the iso-surface. Vertices are placed by linear interpolation along cube
// edges and mapped into world coordinates with grid.min/grid.max.
Mesh marching_cubes(const ScalarGrid& grid, const MarchingCubesOptions& options = {});

// Checks that every edge is used by exactly two faces with opposite direction
// (closed, consistently oriented 2-manifold). Returns the number of offending edges.
std::size_t count_open_edges(const Mesh& mesh);

// Signed volume via the divergence theorem (positive for outward-facing faces).
double signed_volume(const Mesh& mesh);

// Axis-aligned bounding box; returns false for an empty mesh.
bool bounding_box(const Mesh& mesh, float out_min[3], float out_max[3]);

}  // namespace gufo::triposg
