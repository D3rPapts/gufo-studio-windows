// TripoSG: shape VAE decoder (TripoSGVAEModel.decode). Latents [2048, 64] ->
// post_quant -> 16 self-attention blocks (8 heads x 128, no qk norm) -> KV cache
// [2048, 1024]. Query points p in [-1.005, 1.005]^3 -> frequency embedding (51) ->
// proj_query -> one cross-attention block over norm_cross(KV) -> norm_out ->
// proj_out -> the decoder returns -logit, which is negative inside the shape and
// positive outside (SDF-like; measured on real latents), level set 0.
// FP32 CPU reference (streamed weights) and BF16 HIP path on the qwen_image_21 runtime.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/triposg/config.hpp"

namespace gufo::triposg {

struct VaeShape {
    int width = VaeConfig::width_decoder;
    int heads = VaeConfig::heads;
    int layers = VaeConfig::layers_decoder;  // self-attention blocks before the query block
    int mlp = VaeConfig::mlp_decoder;
    int latent_channels = VaeConfig::latent_channels;
    int freqs = VaeConfig::embed_freqs;
    int head_dim() const { return width / heads; }
    int query_channels() const { return 3 * (1 + 2 * freqs); }
};

struct VaeBlockWeights {  // self-attention block i (0..15)
    std::vector<float> norm1_w, norm1_b, norm3_w, norm3_b;
    std::vector<float> q, k, v, o_w, o_b;
    std::vector<float> ff1_w, ff1_b, ff2_w, ff2_b;
};

struct VaeQueryWeights {  // block 16 + head
    std::vector<float> norm2_w, norm2_b, norm3_w, norm3_b;
    std::vector<float> norm_cross_w, norm_cross_b;
    std::vector<float> q, k, v, o_w, o_b;
    std::vector<float> ff1_w, ff1_b, ff2_w, ff2_b;
    std::vector<float> proj_query_w, proj_query_b;  // [width, 51], [width]
    std::vector<float> norm_out_w, norm_out_b;
    std::vector<float> proj_out_w, proj_out_b;      // [1, width], [1]
};

class VaeHostWeights {
public:
    using BlockLoader = std::function<bool(int layer, VaeBlockWeights* out, std::string* error)>;
    static bool FromFile(const std::filesystem::path& file, VaeHostWeights* out, std::string* error);
    static VaeHostWeights Random(const VaeShape& shape, std::uint32_t seed, float scale = 0.05f);

    VaeShape shape;
    std::vector<float> post_quant_w, post_quant_b;  // [width, latent], [width]
    VaeQueryWeights query;
    BlockLoader load_block;
};

// Frequency embedding of points [n, 3] -> [n, 51] (x, sin(f x), cos(f x); f = 2^0..2^7).
void vae_embed_points(const float* points, int count, int freqs, std::vector<float>* out);

// KV cache for one latent set: z [tokens, latent] -> kv [tokens, width].
bool vae_kv_cpu(const VaeHostWeights& w, const float* z, int tokens, std::vector<float>* kv, std::string* error);
// Occupancy logits (positive inside) for query points given the KV cache.
bool vae_query_cpu(const VaeHostWeights& w, const float* kv, int tokens, const float* points, int count,
                   std::vector<float>* logits, std::string* error);

#if GUFO_TRIPOSG_HIP
class VaeHip {
public:
    ~VaeHip();
    static std::unique_ptr<VaeHip> Create(const std::filesystem::path& file, std::string* error);
    // Builds the KV cache and the cross-attention keys/values for the latent set.
    bool Prepare(const float* z, int tokens, std::string* error);
    std::vector<float> DownloadKv();
    // Queries up to any count of points in chunks; logits positive inside.
    bool Query(const float* points, int count, std::vector<float>* logits, std::string* error, int chunk = 32768);

private:
    struct Impl;
    explicit VaeHip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::triposg
