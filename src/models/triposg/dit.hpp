// TripoSG: rectified-flow DiT (TripoSGDiTModel): 21 pre-norm blocks of width 2048
// (self-attention with per-head RMS q/k norm, cross-attention on the DINOv2 tokens,
// GELU FFN), a timestep token prepended to the 2048 latent tokens, U-Net style long
// skips (blocks 0..9 push, 11..20 pop: h = skip_norm(skip_linear(cat(skip, h)))),
// final LayerNorm + proj_out on the latent rows. Two paths: an FP32 CPU reference
// with streamed weights (one block in memory) and the BF16 HIP path on the
// qwen_image_21 runtime.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/models/triposg/config.hpp"

namespace gufo::triposg {

struct DitShape {
    int width = DitConfig::width;
    int heads = DitConfig::heads;
    int layers = DitConfig::layers;
    int mlp = DitConfig::mlp;
    int cross_dim = DitConfig::cross_dim;
    int in_channels = DitConfig::in_channels;
    int head_dim() const { return width / heads; }
    int skip_last_push() const { return layers / 2 - 1; }
    int skip_first_pop() const { return layers / 2 + 1; }
    bool has_skip(int layer) const { return layer >= skip_first_pop(); }
};

struct DitBlockWeights {
    std::vector<float> norm1_w, norm1_b, norm2_w, norm2_b, norm3_w, norm3_b;  // [width]
    std::vector<float> q1, k1, v1, o1_w, o1_b;      // self: [width, width] x3, out + bias
    std::vector<float> nq1, nk1;                    // [head_dim] RMS norm weights
    std::vector<float> q2, k2, v2, o2_w, o2_b;      // cross: q [width, width], k/v [width, cross_dim]
    std::vector<float> nq2, nk2;
    std::vector<float> ff1_w, ff1_b, ff2_w, ff2_b;  // [mlp, width], [mlp], [width, mlp], [width]
    std::vector<float> skip_w, skip_b, skip_norm_w, skip_norm_b;  // only for layers >= skip_first_pop
};

struct DitStemWeights {
    std::vector<float> proj_in_w, proj_in_b;        // [width, in], [width]
    std::vector<float> time1_w, time1_b;            // [4*width, width]
    std::vector<float> time2_w, time2_b;            // [width, 4*width]
    std::vector<float> norm_out_w, norm_out_b;      // [width]
    std::vector<float> proj_out_w, proj_out_b;      // [in, width], [in]
};

// Host weight provider: the stem is kept, blocks are produced on demand so the
// 1.4 B parameters never sit in RAM at once (the file path streams via TensorFile).
class DitHostWeights {
public:
    using BlockLoader = std::function<bool(int layer, DitBlockWeights* out, std::string* error)>;
    static bool FromFile(const std::filesystem::path& file, DitHostWeights* out, std::string* error);
    static DitHostWeights Random(const DitShape& shape, std::uint32_t seed, float scale = 0.05f);

    DitShape shape;
    DitStemWeights stem;
    BlockLoader load_block;
};

struct DitForwardOptions {
    std::vector<int> tap_layers;  // block outputs to capture ([1 + tokens, width], row 0 = time token)
};

// One velocity prediction. x: [tokens, in_channels]; cond: [cond_tokens, cross_dim];
// t in [0, 1000]. v: [tokens, in_channels]. taps: layer -> block output.
bool dit_forward_cpu(const DitHostWeights& weights, const float* x, int tokens, const float* cond, int cond_tokens,
                     float t, std::vector<float>* v, std::map<int, std::vector<float>>* taps, std::string* error,
                     const DitForwardOptions& options = {});

#if GUFO_TRIPOSG_HIP
class DitHip {
public:
    ~DitHip();
    // Maps the converted triposg.safetensors (names "dit.*"); weights upload lazily.
    static std::unique_ptr<DitHip> Create(const std::filesystem::path& file, std::string* error);
    bool Forward(const float* x, int tokens, const float* cond, int cond_tokens, float t, std::vector<float>* v,
                 std::map<int, std::vector<float>>* taps, std::string* error, const DitForwardOptions& options = {});
    std::size_t weight_bytes() const;

private:
    struct Impl;
    explicit DitHip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::triposg
