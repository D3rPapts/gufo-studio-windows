// TRELLIS.2: sparse decoders (shape: FlexiDualGridVaeDecoder with 7 outputs and
// predicted subdivision; texture: SparseUnetVaeDecoder with 6 outputs guided by the shape
// masks). Five levels of SparseConvNeXt blocks (SubM conv 3^3 -> LN -> MLP 4x SiLU -> +x)
// joined by Channel2Spatial up-blocks (to_subdiv mask, conv to 8*C', C2S, LN, SiLU, conv,
// + repeated skip). CPU reference (sparse_conv_cpu.cpp primitives, FP32, streamed weights)
// and the HIP path on the qwen_image_21 runtime (gather kernels + hipBLASLt GEMMs, BF16).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/trellis2/config.hpp"
#include "src/models/trellis2/sparse_conv.hpp"

namespace gufo::trellis2 {

struct SparseDecShape {
    int latent_channels = SparseDecConfig::latent_channels;
    int levels = SparseDecConfig::levels;
    int channels[5] = {1024, 512, 256, 128, 64};
    int blocks[5] = {4, 16, 8, 4, 0};
    int outputs = SparseDecConfig::shape_outputs;
    bool pred_subdiv = true;
    int mlp_ratio = SparseDecConfig::mlp_ratio;
};

struct ConvNeXtWeights {
    std::vector<float> conv_w, conv_b;   // [C, 27C], [C]
    std::vector<float> norm_w, norm_b;   // [C]
    std::vector<float> mlp0_w, mlp0_b;   // [4C, C]
    std::vector<float> mlp2_w, mlp2_b;   // [C, 4C]
};
struct C2SWeights {
    std::vector<float> norm1_w, norm1_b;   // [C]
    std::vector<float> conv1_w, conv1_b;   // [8C', 27C]
    std::vector<float> conv2_w, conv2_b;   // [C', 27C']
    std::vector<float> subdiv_w, subdiv_b; // [8, C] (shape decoder only)
};

class SparseDecHostWeights {
public:
    static bool FromFile(const std::filesystem::path& file, SparseDecHostWeights* out, std::string* error);
    static SparseDecHostWeights Random(const SparseDecShape& shape, std::uint32_t seed, float scale = 0.05f);

    SparseDecShape shape;
    std::vector<float> from_w, from_b;     // [C0, latent]
    std::vector<float> out_w, out_b;       // [outputs, C4]
    std::function<bool(int level, int j, ConvNeXtWeights*, std::string*)> load_block;
    std::function<bool(int level, C2SWeights*, std::string*)> load_up;
};

struct SparseDecOptions {
    int level_limit = 0;                                   // > 0: stop after this many levels (taps only)
    const std::vector<std::vector<std::uint8_t>>* guide_masks = nullptr;   // texture decoder: masks per up-block
    bool capture_levels = false;                           // keep the features after each level
};

struct SparseDecResult {
    std::vector<Coord> coords;                 // final voxel coordinates
    std::vector<float> outputs;                // [M, outputs]
    std::vector<std::vector<std::uint8_t>> masks;   // per up-block: [N_level * 8] subdivision mask
    std::vector<std::vector<float>> level_feats;    // if capture_levels: features after each level [N, C]
    std::vector<std::vector<Coord>> level_coords;
};

bool sparse_dec_forward_cpu(const SparseDecHostWeights& w, const float* latent, const std::vector<Coord>& coords,
                            SparseDecResult* result, std::string* error, const SparseDecOptions& options = {});

#if GUFO_TRELLIS2_HIP
class SparseDecHip {
public:
    ~SparseDecHip();
    static std::unique_ptr<SparseDecHip> Create(const std::filesystem::path& file, const SparseDecShape& shape, std::string* error);
    bool Forward(const float* latent, const std::vector<Coord>& coords, SparseDecResult* result, std::string* error,
                 const SparseDecOptions& options = {});
    std::size_t weight_bytes() const;

private:
    struct Impl;
    explicit SparseDecHip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::trellis2
