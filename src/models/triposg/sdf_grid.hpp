// TripoSG: hierarchical SDF grid extraction (after hierarchical_extract_geometry):
// a dense coarse grid, then per level a band of voxels around the surface
// (|value| < band or sign change, dilated by one voxel) is re-queried at double
// resolution while everything else is interpolated from the coarser level. The
// final level is stored as IEEE half (512^3 = 256 MB) for the marching cubes.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gufo::triposg {

// Evaluates the decoder at `count` points (xyz interleaved); writes `count` values.
using SdfQuery = std::function<bool(const float* points, int count, std::vector<float>* values, std::string* error)>;

struct SdfGridOptions {
    std::vector<int> levels = {256, 512};  // samples per axis, each level double the previous
    float bounds = 1.005f;                 // cube [-bounds, bounds]^3, samples incl. both ends
    float band = 1.f;                      // |value| < band marks a voxel for refinement
    int dilate = 1;                        // neighbourhood (in coarse voxels) added to the band
    std::size_t chunk = 1 << 20;           // points per query batch
    std::function<void(const std::string&)> progress;
};

struct SdfGridResult {
    std::vector<std::uint16_t> half;  // final level, half precision, index (ix * n + iy) * n + iz
    int n = 0;
    std::vector<std::size_t> queried_per_level;  // points evaluated per level
};

bool extract_sdf_grid(const SdfQuery& query, const SdfGridOptions& options, SdfGridResult* out, std::string* error);

}  // namespace gufo::triposg
