// TRELLIS.2: flow DiT (SparseStructureFlowModel / SLatFlowModel, one architecture for
// the five checkpoints): 30 blocks of width 1536, shared AdaLN modulation (6 x 1536 from
// the timestep, plus a per-block learned vector), self-attention with per-head qk RMS
// norm and 3-D RoPE on the token coordinates, cross-attention on the DINOv3 tokens,
// GELU(tanh) MLP 8192, final non-affine LayerNorm + out_layer. Dense (ss: the 16^3 grid)
// and sparse (active voxels) differ only in the coordinate list.
// Two paths: FP32 CPU reference with streamed weights (one block in RAM) and the BF16
// HIP path on the qwen_image_21 runtime (dit.hip).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/models/trellis2/config.hpp"
#include "src/models/trellis2/sparse_conv.hpp"

namespace gufo::trellis2 {

struct DitShape {
    int width = DitConfig::width;
    int heads = DitConfig::heads;
    int layers = DitConfig::layers;
    int mlp = DitConfig::mlp;
    int cond_dim = DitConfig::cond_dim;
    int time_freq = DitConfig::time_freq;
    int in_channels = 8;
    int out_channels = 8;
    int head_dim() const { return width / heads; }
};

struct DitBlockWeights {
    std::vector<float> modulation;          // [6 * width]
    std::vector<float> norm2_w, norm2_b;    // [width]
    std::vector<float> qkv_w, qkv_b;        // [3 * width, width], [3 * width]
    std::vector<float> q_gamma, k_gamma;    // [width] (head-major per-head gammas)
    std::vector<float> self_out_w, self_out_b;
    std::vector<float> cq_w, cq_b;          // cross to_q [width, width]
    std::vector<float> ckv_w, ckv_b;        // cross to_kv [2 * width, cond_dim]
    std::vector<float> cq_gamma, ck_gamma;
    std::vector<float> cross_out_w, cross_out_b;
    std::vector<float> mlp0_w, mlp0_b;      // [mlp, width]
    std::vector<float> mlp2_w, mlp2_b;      // [width, mlp]
};

struct DitStemWeights {
    std::vector<float> t0_w, t0_b;          // t_embedder.mlp.0 [width, time_freq]
    std::vector<float> t2_w, t2_b;          // t_embedder.mlp.2 [width, width]
    std::vector<float> ada_w, ada_b;        // adaLN_modulation.1 [6 * width, width]
    std::vector<float> in_w, in_b;          // input_layer [width, in]
    std::vector<float> out_w, out_b;        // out_layer [out, width]
};

class DitHostWeights {
public:
    using BlockLoader = std::function<bool(int layer, DitBlockWeights* out, std::string* error)>;
    // Reads a converted trellis2-<part>.safetensors (unprefixed names); the blocks stream on demand.
    static bool FromFile(const std::filesystem::path& file, DitHostWeights* out, std::string* error);
    static DitHostWeights Random(const DitShape& shape, std::uint32_t seed, float scale = 0.05f);

    DitShape shape;
    DitStemWeights stem;
    BlockLoader load_block;
};

struct DitForwardOptions {
    std::vector<int> tap_layers;   // block outputs to capture ([tokens, width])
    int layer_limit = 0;           // > 0: run only the first N blocks (CPU/GPU comparison)
};

// One velocity prediction. x: [tokens, in]; coords: [tokens] (RoPE); cond: [cond_tokens, cond_dim];
// t: model timestep (0..1000). v: [tokens, out]. taps: layer -> block output.
bool dit_forward_cpu(const DitHostWeights& weights, const float* x, const std::vector<Coord>& coords, const float* cond,
                     int cond_tokens, float t, std::vector<float>* v, std::map<int, std::vector<float>>* taps,
                     std::string* error, const DitForwardOptions& options = {});

// Dense sparse-structure token coordinates: 16^3 grid in x*256 + y*16 + z order.
std::vector<Coord> dense_grid_coords(int res = SsDecConfig::latent_res);

#if GUFO_TRELLIS2_HIP
class DitHip {
public:
    ~DitHip();
    // `file` is one converted part (e.g. trellis2-ss_flow.safetensors); shape names the channels.
    static std::unique_ptr<DitHip> Create(const std::filesystem::path& file, const DitShape& shape, std::string* error);
    bool Forward(const float* x, const std::vector<Coord>& coords, const float* cond, int cond_tokens, float t,
                 std::vector<float>* v, std::map<int, std::vector<float>>* taps, std::string* error,
                 const DitForwardOptions& options = {});
    std::size_t weight_bytes() const;

private:
    struct Impl;
    explicit DitHip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::trellis2
