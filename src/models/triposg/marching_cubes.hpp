// TripoSG: CPU marching cubes (classic 256-case table) for SDF grids.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
    const float* values = nullptr;            // FP32 samples, or
    const std::uint16_t* half_values = nullptr;  // IEEE half samples (large grids)
    std::size_t nx = 0, ny = 0, nz = 0;
    // World coordinate of sample 0 and of sample n-1 on each axis.
    float min[3] = {-1.f, -1.f, -1.f};
    float max[3] = {1.f, 1.f, 1.f};

    static float half_to_float(std::uint16_t h) {
        const std::uint32_t sign = (h & 0x8000u) << 16;
        std::uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
        std::uint32_t bits;
        if (exp == 0) {
            if (mant == 0) bits = sign;
            else {  // subnormal
                exp = 127 - 15 + 1;
                while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
                mant &= 0x3FF;
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
        else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    static std::uint16_t float_to_half(float f) {
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
        const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127 + 15;
        std::uint32_t mant = bits & 0x7FFFFFu;
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
        if (exp <= 0) {
            if (exp < -10) return sign;
            mant |= 0x800000u;
            const int shift = 14 - exp;
            std::uint16_t h = static_cast<std::uint16_t>(mant >> shift);
            if ((mant >> (shift - 1)) & 1u) ++h;  // round half up
            return static_cast<std::uint16_t>(sign | h);
        }
        std::uint16_t h = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
        if (mant & 0x1000u) ++h;  // round to nearest (ties up)
        return h;
    }
    float at(std::size_t ix, std::size_t iy, std::size_t iz) const {
        const std::size_t i = (ix * ny + iy) * nz + iz;
        return values ? values[i] : half_to_float(half_values[i]);
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
