// TRELLIS.2: the whole image -> GLB pipeline on the GPU (HIP build only):
// preprocess -> DINOv3 (FP32) -> sparse-structure flow (BF16, 12 Euler steps with
// interval CFG) -> ss decoder -> 32^3 active voxels -> shape flow 512 (sparse DiT) ->
// shape decoder (sparse engine) -> flexible dual grid -> fill holes -> decimation ->
// y-up GLB + preview PNG. The texture branch (stage 8) hangs off the same latents; stage 9
// unwraps (xatlas) and bakes the voxel attributes into PBR textures when texture_size > 0.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gufo::trellis2 {

struct GenerateOptions {
    int steps = 12;             // Euler steps for every flow (0 = sampler default 12)
    std::uint64_t seed = 42;
    int resolution = 512;       // only 512 in this stage
    int max_faces = 500000;     // 0 = no decimation
    float hole_perimeter = 0.03f;
    bool preview = true;
    int preview_size = 384;
    bool texture = false;       // stage 8: texture flow + decoder -> vertex colours
    int texture_size = 0;       // stage 9: > 0 bakes PBR textures of this size (UV unwrap), 0 = vertex colours
};

struct GenerateResult {
    std::vector<std::uint8_t> glb;
    std::vector<std::uint8_t> preview_png;
    std::size_t vertices = 0, faces = 0, faces_before_decimation = 0;
    std::size_t structure_voxels = 0, shape_voxels = 0, holes_filled = 0, open_edges = 0, open_loops = 0, flipped_faces = 0;
    std::size_t seam_edges = 0, small_components_removed = 0;   // sign-field seams (parity defects), stray bubbles dropped
    std::size_t uv_charts = 0, texture_size = 0, unsampled_texels = 0;   // stage 9 (0 when vertex colours were written)
    double texel_coverage = 0, seconds_bake = 0;
    std::vector<float> largest_open_loops;   // perimeters of the biggest loops left open (object units)
    float bbox_min[3] = {0, 0, 0}, bbox_max[3] = {0, 0, 0};
    double volume = 0;
    bool closed = false;
    double seconds_encode = 0, seconds_structure = 0, seconds_shape = 0, seconds_decode = 0, seconds_mesh = 0, seconds_texture = 0;
};

// phase: "encode" | "structure" | "shape" | "decode" | "texture" | "mesh" | "export".
using GenerateProgress = std::function<void(const std::string& phase, int completed, int total)>;

class Pipeline {
public:
    ~Pipeline();
    // `weights_dir` holds trellis2-dinov3 / ss_flow / ss_dec / shape_flow_512 / shape_dec
    // (/ tex_flow_512 / tex_dec) .safetensors from tools/trellis2/convert_hf_to_gufo.py.
    static std::unique_ptr<Pipeline> Load(const std::filesystem::path& weights_dir, bool with_texture, std::string* error);
    // rgba: packed RGBA8 (alpha required). One call at a time.
    bool Generate(const std::uint8_t* rgba, int width, int height, const GenerateOptions& options,
                  const GenerateProgress& progress, const std::atomic<bool>* cancel, GenerateResult* result, std::string* error);
    std::size_t weight_bytes() const;
    bool has_texture() const;

private:
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::trellis2
