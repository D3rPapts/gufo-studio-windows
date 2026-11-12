// TRELLIS.2: flexible dual grid -> triangle mesh (o_voxel.convert.flexible_dual_grid_to_mesh,
// inference path). Every active voxel carries one dual vertex (offset in [-0.5, 1.5]
// voxel units), three edge-intersection flags (x, y, z edges leaving the voxel) and a
// quad split weight. For an intersected edge the four voxels around that edge (offsets
// MeshConfig::edge_neighbour_offsets, hashmap lookup) form a quad of dual vertices; it
// is split along 0-2 when w0*w2 > w1*w3, else along 1-3. Vertices: (coord + dual) /
// resolution + aabb_min. Unlike the original, a voxel crossed by several surface sheets
// gets one vertex per sheet (see dual_grid_mesh.cpp), so every dual edge carries at most
// two quads and the later orientation/hole filling sees a 2-manifold (with boundaries).
// The flags become an inside/outside sign per grid corner (flood fill with parity), the mesh is
// the boundary between the signs and therefore closed; parity defects turn into seams, not holes.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <vector>

#include "src/models/trellis2/mesh.hpp"
#include "src/models/trellis2/sparse_conv.hpp"

namespace gufo::trellis2 {

struct DualGridInput {
    const std::vector<Coord>* coords = nullptr;   // [N]
    const float* dual = nullptr;                  // [N, 3] vertex offsets (already sigmoid-mapped)
    const std::uint8_t* intersected = nullptr;    // [N, 3] flags per axis
    const float* split_weight = nullptr;          // [N] softplus weights (nullptr: split by normal alignment)
    int resolution = 512;
    float aabb_min = -0.5f;
    float aabb_max = 0.5f;
};

struct DualGridStats {
    std::size_t quads = 0, skipped_edges = 0;   // edges whose 4 voxels are not all active
    std::size_t open_edges = 0;                 // boundary edges of the triangle mesh
    std::size_t extra_vertices = 0;             // vertices beyond one per voxel (sheet splits)
    std::size_t virtual_vertices = 0;           // centre vertices of inactive voxels next to a sign change
    std::size_t seam_edges = 0;                 // edges whose sign change and flag disagree (parity defects)
};

Mesh dual_grid_to_mesh(const DualGridInput& in, DualGridStats* stats = nullptr);

// Boundary edges (used by exactly one triangle) of a mesh.
std::size_t count_boundary_edges(const Mesh& mesh);

// Largest number of triangles sharing one undirected edge (2 for a manifold mesh).
std::size_t max_edge_multiplicity(const Mesh& mesh);

// Decoder output [N, 7] -> the three dual-grid arrays (sigmoid with voxel margin 0.5,
// logits > 0, softplus).
void decode_shape_outputs(const float* out7, std::size_t n, std::vector<float>* dual, std::vector<std::uint8_t>* intersected,
                          std::vector<float>* split_weight, float voxel_margin = 0.5f);

}  // namespace gufo::trellis2
