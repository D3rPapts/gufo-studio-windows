// TRELLIS.2: tensor inventory of the converted gufo weight files as produced by
// tools/trellis2/convert_hf_to_gufo.py (one file per part, names unchanged except
// where noted). Checked against the real safetensors headers on 2026-11-12:
//   DiT files: 640 tensors, 1 292 253 728 params (32/32 channels), all BF16 upstream
//   shape_dec: 292 tensors, 474 231 463 params, F16 upstream
//   ss_dec (TRELLIS v1): 74 tensors, 73 671 201 params, F16 convs + F32 norms
//
// Converter rules: (a) 2-D matrices with both dims >= 64 are stored BF16, everything
// else F32; (b) 5-D conv weights are flattened to [Co, 27 * Ci] with column
// t * Ci + ci, tap t = (dx+1)*9 + (dy+1)*3 + (dz+1) for the offset (dx,dy,dz) in
// {-1,0,1}^3 (= PyTorch Conv3d cross-correlation order kD,kH,kW over the coordinate
// axes x,y,z); the sparse decoders ship [Co,3,3,3,Ci] (FlexGEMM layout), ss_dec ships
// [Co,Ci,3,3,3] (PyTorch) - both end up identical; (c) the per-head qk RMS gammas
// [12,128] are flattened to [1536] (head-major) so they can scale the columns of the
// [tokens, 1536] q/k matrices.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/trellis2/config.hpp"

namespace gufo::trellis2 {

struct TensorSpec {
    std::string name;                   // name in the gufo file (%d = block index)
    std::array<std::size_t, 2> shape;   // converted shape (1-D: second entry 0)
    std::size_t repeat;                 // number of blocks for %d names
    bool bf16;
    std::size_t elements() const { return shape[0] * (shape[1] ? shape[1] : 1); }
};

inline std::size_t total_params(const std::vector<TensorSpec>& specs) {
    std::size_t n = 0;
    for (const auto& s : specs) n += s.elements() * s.repeat;
    return n;
}

// --- Flow DiT (identical names in all five files) ---------------------------
inline std::vector<TensorSpec> describe_flow(const FlowVariant& v) {
    constexpr std::size_t W = DitConfig::width, M = DitConfig::mlp, C = DitConfig::cond_dim;
    constexpr std::size_t L = DitConfig::layers, F = DitConfig::time_freq;
    const std::size_t in = static_cast<std::size_t>(v.in_channels), out = static_cast<std::size_t>(v.out_channels);
    std::vector<TensorSpec> t = {
        {"t_embedder.mlp.0.weight", {W, F}, 1, true},  {"t_embedder.mlp.0.bias", {W, 0}, 1, false},
        {"t_embedder.mlp.2.weight", {W, W}, 1, true},  {"t_embedder.mlp.2.bias", {W, 0}, 1, false},
        {"adaLN_modulation.1.weight", {6 * W, W}, 1, true}, {"adaLN_modulation.1.bias", {6 * W, 0}, 1, false},
        {"input_layer.weight", {W, in}, 1, in >= 64},  {"input_layer.bias", {W, 0}, 1, false},
        {"blocks.%d.modulation", {6 * W, 0}, L, false},
        {"blocks.%d.norm2.weight", {W, 0}, L, false},   {"blocks.%d.norm2.bias", {W, 0}, L, false},
        {"blocks.%d.self_attn.to_qkv.weight", {3 * W, W}, L, true}, {"blocks.%d.self_attn.to_qkv.bias", {3 * W, 0}, L, false},
        {"blocks.%d.self_attn.q_rms_norm.gamma", {W, 0}, L, false},  // flattened [12,128]
        {"blocks.%d.self_attn.k_rms_norm.gamma", {W, 0}, L, false},
        {"blocks.%d.self_attn.to_out.weight", {W, W}, L, true},      {"blocks.%d.self_attn.to_out.bias", {W, 0}, L, false},
        {"blocks.%d.cross_attn.to_q.weight", {W, W}, L, true},       {"blocks.%d.cross_attn.to_q.bias", {W, 0}, L, false},
        {"blocks.%d.cross_attn.to_kv.weight", {2 * W, C}, L, true},  {"blocks.%d.cross_attn.to_kv.bias", {2 * W, 0}, L, false},
        {"blocks.%d.cross_attn.q_rms_norm.gamma", {W, 0}, L, false},
        {"blocks.%d.cross_attn.k_rms_norm.gamma", {W, 0}, L, false},
        {"blocks.%d.cross_attn.to_out.weight", {W, W}, L, true},     {"blocks.%d.cross_attn.to_out.bias", {W, 0}, L, false},
        {"blocks.%d.mlp.mlp.0.weight", {M, W}, L, true},             {"blocks.%d.mlp.mlp.0.bias", {M, 0}, L, false},
        {"blocks.%d.mlp.mlp.2.weight", {W, M}, L, true},             {"blocks.%d.mlp.mlp.2.bias", {W, 0}, L, false},
        {"out_layer.weight", {out, W}, 1, out >= 64},   {"out_layer.bias", {out, 0}, 1, false},
    };
    return t;
}

// --- Sparse-structure decoder (dense Conv3d ResNet) ---------------------------
inline std::vector<TensorSpec> describe_ss_dec() {
    constexpr std::size_t K = 27;
    auto conv = [](std::string name, std::size_t co, std::size_t ci) {
        return std::vector<TensorSpec>{{name + ".weight", {co, K * ci}, 1, true}, {name + ".bias", {co, 0}, 1, false}};
    };
    auto norm = [](std::string name, std::size_t c) {
        return std::vector<TensorSpec>{{name + ".weight", {c, 0}, 1, false}, {name + ".bias", {c, 0}, 1, false}};
    };
    auto res = [&](std::string base, std::size_t c) {
        std::vector<TensorSpec> v;
        for (auto& e : norm(base + ".norm1", c)) v.push_back(e);
        for (auto& e : norm(base + ".norm2", c)) v.push_back(e);
        for (auto& e : conv(base + ".conv1", c, c)) v.push_back(e);
        for (auto& e : conv(base + ".conv2", c, c)) v.push_back(e);
        return v;
    };
    std::vector<TensorSpec> t;
    auto append = [&](std::vector<TensorSpec> v) { for (auto& e : v) t.push_back(e); };
    const auto& ch = SsDecConfig::channels;
    append(conv("input_layer", ch[0], SsDecConfig::latent_channels));
    append(res("middle_block.0", ch[0]));
    append(res("middle_block.1", ch[0]));
    // blocks: 0,1 res 512 ; 2 up 512->128 ; 3,4 res 128 ; 5 up 128->32 ; 6,7 res 32
    append(res("blocks.0", ch[0]));
    append(res("blocks.1", ch[0]));
    append(conv("blocks.2.conv", 8 * ch[1], ch[0]));
    append(res("blocks.3", ch[1]));
    append(res("blocks.4", ch[1]));
    append(conv("blocks.5.conv", 8 * ch[2], ch[1]));
    append(res("blocks.6", ch[2]));
    append(res("blocks.7", ch[2]));
    append(norm("out_layer.0", ch[2]));
    append(conv("out_layer.2", 1, ch[2]));
    return t;
}

// --- Sparse decoders (shape: pred_subdiv, 7 outputs; tex: guided, 6 outputs) ----
inline std::vector<TensorSpec> describe_sparse_dec(bool shape) {
    constexpr std::size_t K = 27;
    std::vector<TensorSpec> t;
    auto add = [&](std::string name, std::size_t a, std::size_t b, bool bf) { t.push_back({name, {a, b}, 1, bf}); };
    const std::size_t latent = SparseDecConfig::latent_channels;
    const std::size_t out = shape ? SparseDecConfig::shape_outputs : SparseDecConfig::tex_outputs;
    add("from_latent.weight", SparseDecConfig::channels[0], latent, false);   // [1024, 32]: 32 < 64 -> F32
    add("from_latent.bias", SparseDecConfig::channels[0], 0, false);
    for (int level = 0; level < SparseDecConfig::levels; ++level) {
        const std::size_t c = SparseDecConfig::channels[level];
        const int n = SparseDecConfig::blocks[level];
        for (int j = 0; j < n; ++j) {
            const std::string b = "blocks." + std::to_string(level) + "." + std::to_string(j);
            add(b + ".conv.weight", c, K * c, true);           add(b + ".conv.bias", c, 0, false);
            add(b + ".norm.weight", c, 0, false);              add(b + ".norm.bias", c, 0, false);
            add(b + ".mlp.0.weight", 4 * c, c, true);          add(b + ".mlp.0.bias", 4 * c, 0, false);
            add(b + ".mlp.2.weight", c, 4 * c, true);          add(b + ".mlp.2.bias", c, 0, false);
        }
        if (level + 1 < SparseDecConfig::levels) {
            const std::size_t c2 = SparseDecConfig::channels[level + 1];
            const std::string b = "blocks." + std::to_string(level) + "." + std::to_string(n);
            add(b + ".norm1.weight", c, 0, false);             add(b + ".norm1.bias", c, 0, false);
            add(b + ".conv1.weight", 8 * c2, K * c, true);     add(b + ".conv1.bias", 8 * c2, 0, false);
            add(b + ".conv2.weight", c2, K * c2, true);        add(b + ".conv2.bias", c2, 0, false);
            if (shape) {
                add(b + ".to_subdiv.weight", 8, c, false);     add(b + ".to_subdiv.bias", 8, 0, false);
            }
        }
    }
    add("output_layer.weight", out, SparseDecConfig::channels[SparseDecConfig::levels - 1], false);
    add("output_layer.bias", out, 0, false);
    return t;
}

// Expected parameter counts from the HF headers (2026-11-12).
constexpr std::size_t kParamsShapeDec = 474231463;
constexpr std::size_t kParamsTexDec = 474216006;   // shape_dec - 4 to_subdiv (15 392) - 1 output row (65)
constexpr std::size_t kParamsSsDec = 73671201;

// Metadata keys written by the converter.
constexpr std::string_view kMetaFormat = "gufo.trellis2.format";    // "1"
constexpr std::string_view kMetaPart = "gufo.trellis2.part";        // e.g. "ss_flow", "shape_dec"
constexpr std::string_view kMetaSource = "gufo.trellis2.source";    // "microsoft/TRELLIS.2-4B@<rev>"
constexpr std::string_view kMetaLicense = "gufo.trellis2.license";  // "MIT (Microsoft)"
constexpr std::string_view kMetaConvLayout = "gufo.trellis2.conv_layout";  // "co_tap_ci"

}  // namespace gufo::trellis2
