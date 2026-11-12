// TRELLIS.2: sparse-structure decoder (TRELLIS v1 SparseStructureDecoder
// ss_dec_conv3d_16l8_fp16, MIT): dense Conv3d ResNet 16^3 x 8 -> 64^3 x 1 occupancy logits.
// Layout on the host and device: voxel-major [V, C] (channels last, voxel index
// (x * R + y) * R + z), so LayerNorm over channels is a row norm and the 3^3 convolution
// is an im2col gather [V, 27 * Ci] (tap t = (dx+1)*9 + (dy+1)*3 + (dz+1), zero padding)
// times the converted weight [Co, 27 * Ci]. Upsampling: conv to 8 * Co then pixel shuffle
// (channel c * 8 + sx * 4 + sy * 2 + sz -> voxel (2x + sx, 2y + sy, 2z + sz)).
// FP32 CPU reference + FP32 rocBLAS HIP path.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/trellis2/config.hpp"
#include "src/models/trellis2/sparse_conv.hpp"

namespace gufo::trellis2 {

struct SsDecShape {
    int latent_channels = SsDecConfig::latent_channels;
    int latent_res = SsDecConfig::latent_res;
    int channels[3] = {SsDecConfig::channels[0], SsDecConfig::channels[1], SsDecConfig::channels[2]};
    int res_blocks = SsDecConfig::res_blocks;
    int middle_blocks = SsDecConfig::middle_blocks;
    int out_res() const { return latent_res * 4; }
};

struct SsDecConv {
    std::vector<float> w, b;   // [Co, 27 * Ci], [Co]
    int co = 0, ci = 0;
};
struct SsDecNorm {
    std::vector<float> w, b;   // [C]
};
struct SsDecResBlock {
    SsDecNorm norm1, norm2;
    SsDecConv conv1, conv2;
};

struct SsDecWeights {
    SsDecShape shape;
    SsDecConv input;                        // 8 -> 512
    std::vector<SsDecResBlock> middle;      // 2 x 512
    std::vector<SsDecResBlock> level0;      // 2 x 512
    SsDecConv up0;                          // 512 -> 8 * 128
    std::vector<SsDecResBlock> level1;      // 2 x 128
    SsDecConv up1;                          // 128 -> 8 * 32
    std::vector<SsDecResBlock> level2;      // 2 x 32
    SsDecNorm out_norm;                     // 32
    SsDecConv out;                          // 32 -> 1

    static bool Load(const std::filesystem::path& file, SsDecWeights* out, std::string* error);
    static SsDecWeights Random(const SsDecShape& shape, std::uint32_t seed, float scale = 0.05f);
    std::size_t parameter_count() const;
};

// Dense 3^3 convolution on an R^3 grid, voxel-major [V, Ci] -> [V, Co], zero padding.
void dense_conv3(const float* in, int res, int ci, const float* w, const float* b, int co, float* out);
// Channel LayerNorm (over C per voxel) with affine, eps 1e-5, then optional SiLU, in place.
void channel_norm_silu(float* x, int voxels, int channels, const float* w, const float* b, float eps, bool silu);
// Pixel shuffle 2: [R^3, 8 * C] -> [(2R)^3, C].
void pixel_shuffle3(const float* in, int res, int channels, float* out);

// latent: [16^3, 8] voxel-major. logits: [64^3] occupancy logits (> 0 = active).
bool ss_dec_forward_cpu(const SsDecWeights& w, const float* latent, std::vector<float>* logits, std::string* error);

// Occupancy 64^3 (> 0) -> coordinates at `resolution` (64: as is; 32: max-pool 2 and > 0.5),
// sorted x, y, z ascending like torch.argwhere.
std::vector<Coord> occupancy_to_coords(const float* logits, int res64, int target_res);

#if GUFO_TRELLIS2_HIP
class SsDecHip {
public:
    ~SsDecHip();
    static std::unique_ptr<SsDecHip> Create(const SsDecWeights& weights, std::string* error);
    bool Forward(const float* latent, std::vector<float>* logits, std::string* error);
    std::size_t device_bytes() const;

private:
    struct Impl;
    explicit SsDecHip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::trellis2
