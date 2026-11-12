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

struct UnifyStats {
    std::size_t components = 0, flipped_faces = 0, global_flip = 0;
};
// Makes the winding consistent across every edge-connected component (BFS over shared
// edges: neighbours must traverse the shared edge in opposite directions) and flips the whole
// mesh when its signed volume is negative. The dual grid emits quads in a fixed vertex order,
// so half of the surface comes out inverted (the original relies on CuMesh
// unify_face_orientations at export time).
UnifyStats unify_orientation(Mesh* mesh);

// Removes edge-connected components with fewer than `min_faces` triangles (stray-flag bubbles of
// the sign field: six quads around one corner), unused vertices are dropped. Returns the number
// of removed components.
std::size_t remove_small_components(Mesh* mesh, std::size_t min_faces = 64);

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
