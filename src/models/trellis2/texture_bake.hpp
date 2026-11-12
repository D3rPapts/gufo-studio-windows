// TRELLIS.2 stage 9: UV unwrap (xatlas, MIT, third_party/xatlas) and texture baking from the
// texture decoder's voxel attributes, replacing the original's CuMesh uv_unwrap + nvdiffrast
// rasterisation + flex_gemm grid_sample_3d + cv2.inpaint (trellis2_texturing.py::postprocess_mesh).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "src/models/trellis2/mesh.hpp"
#include "src/models/trellis2/sparse_conv.hpp"

namespace gufo::trellis2 {

// Mesh with one UV per vertex (vertices are duplicated along chart seams by the unwrap).
struct UvMesh {
    Mesh mesh;
    std::vector<float> uv;              // [vertices * 2] in [0, 1]; v = 0 is the top row of the image (glTF)
    std::vector<std::uint32_t> source;  // original vertex index of every vertex
    std::size_t charts = 0;
    int atlas_width = 0, atlas_height = 0;
};

// xatlas chart growth. The dual-grid surface is bumpy at voxel scale, so the xatlas defaults
// (max_cost 2, normal deviation weight 2, straightness 6, normal seam 4) stop chart growth at
// nearly every triangle (cup: 48 794 charts, 23 % coverage, 185 s); the defaults here let
// charts grow across that noise.
struct UnwrapOptions {
    // Box projection (default): faces binned by their smoothed normal's dominant axis, charts =
    // connected runs of one bin, small charts merged into their biggest neighbour, every chart an
    // orthographic projection along its axis, shelf-packed. Seconds for 500 k triangles; xatlas
    // needed 2-20 minutes and tens of thousands of charts on the voxel-noisy surface.
    bool box_projection = true;
    int normal_smoothing_passes = 3;   // face-normal averaging over edge neighbours before binning
    int min_chart_faces = 1000;        // smaller charts are merged into a neighbour
    float fill_target = 0.75f;         // first packing attempt: chart boxes cover this share of the atlas
    // xatlas settings (box_projection = false)
    float max_cost = 8.f;
    float normal_deviation_weight = 1.f;
    float straightness_weight = 2.f;
    float normal_seam_weight = 2.f;
    float roundness_weight = 0.01f;
    int iterations = 1;
    int padding = 4;
};

// Charts, parametrises and packs the mesh into one atlas of about `texture_size` texels per
// side (xatlas may choose a slightly different aspect) with `padding` texels between charts.
// Fails (false, *error) on empty or broken input.
bool unwrap_uv(const Mesh& mesh, int texture_size, UvMesh* out, std::string* error, const UnwrapOptions& options = {});

// Trilinear sampler over the texture decoder's sparse voxel attributes: one row of `channels`
// floats per active voxel at `coords`, voxel size 1/resolution on [-0.5, 0.5]^3 (z-up frame of
// the dual grid), voxel centres at (c + 0.5) / resolution - 0.5. Positions with no active voxel
// among their eight interpolation corners take the nearest active voxel within `max_radius`
// voxels (Chebyshev shells), which also colours the seam caps of the closed mesh.
class VoxelAttrSampler {
public:
    VoxelAttrSampler(const std::vector<Coord>& coords, const std::vector<float>& attrs, int channels, int resolution,
                     int max_radius = 8);
    // Fills out[channels]; returns false and leaves `out` unchanged when nothing is within reach.
    bool Sample(float x, float y, float z, float* out) const;
    int channels() const { return channels_; }

private:
    const std::vector<Coord>& coords_;
    const std::vector<float>& attrs_;
    int channels_, resolution_, max_radius_;
    VoxelHashMap map_;
};

struct BakedTextures {
    int size = 0;
    std::vector<std::uint8_t> base_color;          // RGBA8 [size * size * 4]
    std::vector<std::uint8_t> metallic_roughness;  // RGB8 [size * size * 3]: G = roughness, B = metallic (glTF)
    std::size_t covered_texels = 0;                // texels inside a triangle (before dilation)
    std::size_t unsampled_texels = 0;              // covered texels without any voxel in reach
};

// Rasterises the UV triangles at `size` x `size` texels, samples the voxel attributes (layout:
// base colour r g b, metallic, roughness, alpha) at every covered texel's 3D position and
// dilates the result `dilate_px` texels into the uncovered area (no seams when mip-mapped).
BakedTextures bake_textures(const UvMesh& mesh, const VoxelAttrSampler& sampler, int size, int dilate_px = 8);

}  // namespace gufo::trellis2
