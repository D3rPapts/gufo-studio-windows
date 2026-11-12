// TripoSG: DINOv2-L/14 image encoder (HF Dinov2Model "facebook/dinov2-large") for
// the 224 px conditioning: 257 tokens x 1024, `last_hidden_state` after the final
// LayerNorm. Weights come from the converted triposg.safetensors ("dino." prefix,
// fused qkv, baked 224 px position table). A plain FP32 CPU path serves as the
// reference for the HIP path and for dump comparisons without a GPU.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/triposg/config.hpp"

namespace gufo::triposg {

struct Dinov2Weights {
    struct Layer {
        std::vector<float> ln1_w, ln1_b;    // [1024]
        std::vector<float> qkv_w, qkv_b;    // [3072, 1024], [3072] (q | k | v)
        std::vector<float> out_w, out_b;    // [1024, 1024], [1024]
        std::vector<float> ls1;             // [1024] layer_scale1.lambda1
        std::vector<float> ln2_w, ln2_b;
        std::vector<float> fc1_w, fc1_b;    // [4096, 1024], [4096]
        std::vector<float> fc2_w, fc2_b;    // [1024, 4096], [1024]
        std::vector<float> ls2;
    };
    std::vector<float> cls;                 // [1024]
    std::vector<float> patch_w, patch_b;    // [1024, 3*14*14], [1024]
    std::vector<float> pos224;              // [257, 1024]
    std::vector<float> final_ln_w, final_ln_b;
    std::vector<Layer> layers;              // 24

    // Reads the "dino." tensors of a converted file (BF16 matrices are widened).
    static bool Load(const std::filesystem::path& file, Dinov2Weights* out, std::string* error);
    // Deterministic pseudo-random weights for tests (LayerNorm weights 1, biases 0).
    static Dinov2Weights Random(std::uint32_t seed, float scale = 0.02f, int layers = DinoConfig::layers);
    bool Validate(std::string* error) const;
    std::size_t parameter_count() const;
};

// pixels: [3, 224, 224] normalised; tokens: [257, 1024].
bool dinov2_encode_cpu(const Dinov2Weights& weights, const float* pixels, std::vector<float>* tokens,
                       std::string* error);

// Patch rows [256, 588] from CHW pixels (shared by both paths).
void dinov2_patch_rows(const float* pixels, std::vector<float>* rows);

#if GUFO_TRIPOSG_HIP
class Dinov2Hip {
public:
    ~Dinov2Hip();
    static std::unique_ptr<Dinov2Hip> Create(const Dinov2Weights& weights, std::string* error);
    bool Encode(const float* pixels, std::vector<float>* tokens, std::string* error);
    std::size_t device_bytes() const;

private:
    struct Impl;
    explicit Dinov2Hip(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace gufo::triposg
