// TRELLIS.2: flexible dual grid -> triangle mesh (o_voxel.convert.flexible_dual_grid_to_mesh,
// inference path). Every active voxel carries one dual vertex (offset in [-0.5, 1.5]
// voxel units), three edge-intersection flags (x, y, z edges leaving the voxel) and a
// quad split weight. For an intersected edge the four voxels around that edge (offsets
// MeshConfig::edge_neighbour_offsets, hashmap lookup) form a quad of dual vertices; it
// is split along 0-2 when w0*w2 > w1*w3, else along 1-3. Vertices: (coord + dual) /
// resolution + aabb_min. No hole filling yet (the original runs CuMesh fill_holes).
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
};

Mesh dual_grid_to_mesh(const DualGridInput& in, DualGridStats* stats = nullptr);

// Boundary edges (used by exactly one triangle) of a mesh.
std::size_t count_boundary_edges(const Mesh& mesh);

// Decoder output [N, 7] -> the three dual-grid arrays (sigmoid with voxel margin 0.5,
// logits > 0, softplus).
void decode_shape_outputs(const float* out7, std::size_t n, std::vector<float>* dual, std::vector<std::uint8_t>* intersected,
                          std::vector<float>* split_weight, float voxel_margin = 0.5f);

}  // namespace gufo::trellis2
