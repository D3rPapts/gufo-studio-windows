// TRELLIS.2: mesh post-processing after the dual grid: fill small holes (boundary loops
// with a perimeter below a threshold are fanned from their centroid, like CuMesh
// fill_holes with max_hole_perimeter 0.03 on the unit cube), quadric decimation
// (TripoSG's mesh_decimate), z-up -> y-up for the GLB, and statistics.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstddef>
#include <vector>

#include "src/models/trellis2/mesh.hpp"

namespace gufo::trellis2 {

struct HoleFillStats {
    std::size_t loops = 0, filled = 0, skipped_large = 0, skipped_nonmanifold = 0;
    std::vector<float> skipped_perimeters;   // perimeters of the loops left open (sorted descending)
};

// Fills boundary loops whose perimeter is below `max_perimeter` (object units). Returns stats.
HoleFillStats fill_holes(Mesh* mesh, float max_perimeter = 0.03f);

// Quadric edge-collapse decimation to at most `target_faces` triangles (0 = unchanged).
Mesh decimate_mesh(const Mesh& mesh, std::size_t target_faces);

// (x, y, z) z-up -> (x, z, -y) y-up (a rotation, keeps the winding).
Mesh to_y_up(const Mesh& mesh);

struct MeshStats {
    float bbox_min[3] = {0, 0, 0}, bbox_max[3] = {0, 0, 0};
    double volume = 0;       // signed volume (positive for outward-facing closed meshes)
    std::size_t open_edges = 0;
    bool closed = false;
};
MeshStats mesh_stats(const Mesh& mesh);

}  // namespace gufo::trellis2
