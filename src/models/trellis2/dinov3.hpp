// TRELLIS.2: DINOv3 ViT-L/16 image encoder (HF DINOv3ViTModel
// "facebook/dinov3-vitl16-pretrain-lvd1689m", Meta DINOv3 licence) as used by
// Trellis2's DinoV3FeatureExtractor: 512 px -> 1 CLS + 4 register + 1024 patch tokens,
// 24 layers with 2-D RoPE on the patch tokens, LayerScale, exact GELU; the output is
// the hidden state after layer 24 followed by F.layer_norm WITHOUT affine (the model's
// own `norm` is not applied). Weights from tools/trellis2/convert_hf_to_gufo.py
// (part "dinov3": fused qkv with a zero key bias, patch weight flattened to [1024, 768]).
// FP32 CPU reference + FP32 rocBLAS HIP path (template: triposg/dinov2.hip).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/trellis2/config.hpp"

namespace gufo::trellis2 {

struct Dinov3Weights {
    struct Layer {
        std::vector<float> ln1_w, ln1_b;    // [1024]
        std::vector<float> qkv_w, qkv_b;    // [3072, 1024], [3072] (q | k | v; k bias is zero)
        std::vector<float> out_w, out_b;    // o_proj [1024, 1024], [1024]
        std::vector<float> ls1;             // layer_scale1.lambda1 [1024]
        std::vector<float> ln2_w, ln2_b;
        std::vector<float> fc1_w, fc1_b;    // mlp.up_proj [4096, 1024], [4096]
        std::vector<float> fc2_w, fc2_b;    // mlp.down_proj [1024, 4096], [1024]
        std::vector<float> ls2;
    };
    std::vector<float> cls;                 // [1024]
    std::vector<float> registers;           // [4, 1024]
    std::vector<float> patch_w, patch_b;    // [1024, 3*16*16], [1024]
    std::vector<Layer> layers;              // 24

    static bool Load(const std::filesystem::path& file, Dinov3Weights* out, std::string* error);
    static Dinov3Weights Random(std::uint32_t seed, float scale = 0.02f, int layers = DinoV3Config::layers);
    bool Validate(std::string* error) const;
    std::size_t parameter_count() const;
};

// Patch rows [grid*grid, 768] (row-major y, x; values c, y, x) from CHW pixels of
// image x image.
void dinov3_patch_rows(const float* pixels, int image, std::vector<float>* rows);

// 2-D RoPE angle table for a grid x grid patch layout: [grid*grid, 32] angles; the
// 64-dim head rotates pair (d, d + 32) by angle d (d < 16: y, 16 <= d < 32: x).
// angle = 2 pi * c * theta^(-4 d' / 64), c = 2 (i + 0.5) / grid - 1.
void dinov3_rope_angles(int grid, std::vector<float>* angles);

// In place on one head vector [64]: v*cos + rotate_half(v)*sin.
void dinov3_apply_rope(float* v, const float* angles32);

// pixels: [3, image, image] normalised; tokens: [1 + 4 + (image/16)^2, 1024].
bool dinov3_encode_cpu(const Dinov3Weights& weights, const float* pixels, int image, std::vector<float>* tokens,
                       std::string* error);

#if GUFO_TRELLIS2_HIP
class Dinov3Hip {
public:
    ~Dinov3Hip();
    // `image` fixes the token count (512 or 1024).
    static std::unique_ptr<Dinov3Hip> Create(const Dinov3Weights& weights, int image, std::string* error);
    bool Encode(const float* pixels, std::vector<float>* tokens, std::string* error);
    std::size_t device_bytes() const;

private:
    struct Impl;
    explicit Dinov3Hip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::trellis2
