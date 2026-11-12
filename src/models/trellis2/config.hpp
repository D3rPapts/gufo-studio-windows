// TRELLIS.2 (microsoft/TRELLIS.2-4B, MIT): architecture constants as published in the
// HF checkpoint configs (ckpts/*.json, pipeline.json) and read off the model code
// (trellis2/models, trellis2/modules). Nothing here executes upstream code; the
// values were cross-checked against the safetensors headers on 2026-11-12.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gufo::trellis2 {

// --- Image conditioning: DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m) ---
// Gated Meta licence; the weights are NOT part of this port's download. Shapes from the
// HF transformers DINOv3ViT implementation (hidden 1024, 24 layers, 16 heads, patch 16,
// 4 register tokens, 2-D RoPE, LayerScale, GELU MLP 4096, no position table).
struct DinoV3Config {
    static constexpr int patch = 16;
    static constexpr int width = 1024;
    static constexpr int heads = 16;
    static constexpr int head_dim = width / heads;  // 64
    static constexpr int layers = 24;
    static constexpr int mlp = 4096;
    static constexpr int registers = 4;             // CLS + 4 register tokens precede the patches
    static constexpr float layer_norm_eps = 1e-5f;
    static constexpr float rope_theta = 100.f;      // HF config rope_theta for DINOv3 ViT
    static constexpr int tokens(int image) { return 1 + registers + (image / patch) * (image / patch); }
    static constexpr int tokens_512 = 1 + registers + 32 * 32;    // 1029
    static constexpr int tokens_1024 = 1 + registers + 64 * 64;   // 4101
    // Input: PIL LANCZOS resize to image x image, /255, ImageNet normalisation.
    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float std[3] = {0.229f, 0.224f, 0.225f};
    // Output: hidden states after the last layer, then F.layer_norm WITHOUT affine
    // (the model's own final layernorm is not applied). The negative condition is zeros.
};

// --- Flow DiTs (SparseStructureFlowModel / SLatFlowModel): one architecture, five files ---
struct DitConfig {
    static constexpr int width = 1536;
    static constexpr int heads = 12;
    static constexpr int head_dim = width / heads;  // 128 (fits wan22/qwen attention kernels)
    static constexpr int layers = 30;
    static constexpr int mlp = 8192;                // int(1536 * 5.3334)
    static constexpr int cond_dim = 1024;           // DINOv3 width
    static constexpr int time_freq = 256;           // sinusoid: [cos | sin], max period 10000
    static constexpr float block_norm_eps = 1e-6f;  // norm1/norm3 without affine, norm2 with affine
    static constexpr float final_norm_eps = 1e-5f;  // F.layer_norm default, no affine
    static constexpr int modulation = 6 * width;    // shift/scale/gate (attn) + shift/scale/gate (mlp)
    // 3-D RoPE on self-attention q/k: freq_dim = 128/2/3 = 21 angles per axis, 63 used,
    // one complex pair padded with angle 0. Pair i of the 64 pairs: axis i/21, j = i%21,
    // angle = coord[axis] / 10000^(j/21). Pairs are adjacent (x[2i], x[2i+1]).
    static constexpr int rope_freqs_per_axis = head_dim / 2 / 3;  // 21
    static constexpr int rope_pairs = head_dim / 2;               // 64
    static constexpr float rope_base = 10000.f;
    // qk RMS norm: F.normalize(x) * gamma[head] * sqrt(128) (gamma differs per head).
    static constexpr float qk_norm_scale_dim = static_cast<float>(head_dim);
};

enum class Flow { kSparseStructure, kShape512, kShape1024, kTex512, kTex1024 };

struct FlowVariant {
    const char* file;      // ckpts/<file>.safetensors
    int in_channels;       // input_layer columns
    int out_channels;      // out_layer rows
    int resolution;        // token grid: dense 16^3 for ss, sparse 32^3 / 64^3 otherwise
    bool sparse;           // tokens = active voxels (coordinate list) instead of the full grid
    std::size_t params;    // from the safetensors header (2026-11-12)
};

constexpr FlowVariant kFlows[] = {
    {"ss_flow_img_dit_1_3B_64_bf16",           8,  8, 16, false, 1292179976},
    {"slat_flow_img2shape_dit_1_3B_512_bf16",  32, 32, 32, true,  1292253728},
    {"slat_flow_img2shape_dit_1_3B_1024_bf16", 32, 32, 64, true,  1292253728},
    {"slat_flow_imgshape2tex_dit_1_3B_512_bf16",  64, 32, 32, true, 1292302880},
    {"slat_flow_imgshape2tex_dit_1_3B_1024_bf16", 64, 32, 64, true, 1292302880},
};

// --- Sparse-structure decoder (TRELLIS v1 ss_dec_conv3d_16l8_fp16, MIT, 148 MB) ---
// Dense Conv3d ResNet 16^3 x 8 -> 64^3 x 1 occupancy logits (> 0 = active).
struct SsDecConfig {
    static constexpr int latent_channels = 8;
    static constexpr int latent_res = 16;
    static constexpr int out_res = 64;
    static constexpr int channels[3] = {512, 128, 32};
    static constexpr int res_blocks = 2;         // per level
    static constexpr int middle_blocks = 2;      // at 512 before the levels
    static constexpr float norm_eps = 1e-5f;     // ChannelLayerNorm32 (LayerNorm over channels)
    // Block order: input_layer (8->512, 3^3) ; middle_block.0/1 ; blocks.0/1 (512) ;
    // blocks.2 up 512->128 (conv 3^3 to 8*128 + pixel shuffle 2) ; blocks.3/4 (128) ;
    // blocks.5 up 128->32 ; blocks.6/7 (32) ; out_layer: LN + SiLU + conv 32->1.
    // ResBlock: LN -> SiLU -> conv1 -> LN -> SiLU -> conv2 -> + skip (1x1 conv if C changes).
    // For the 512 path the 64^3 occupancy is max-pooled (2) to 32^3 (> 0.5).
};

// --- Sparse decoders (shape: FlexiDualGridVaeDecoder 7 outputs, tex: SparseUnetVaeDecoder 6) ---
struct SparseDecConfig {
    static constexpr int latent_channels = 32;
    static constexpr int levels = 5;
    static constexpr int channels[levels] = {1024, 512, 256, 128, 64};
    static constexpr int blocks[levels] = {4, 16, 8, 4, 0};   // SparseConvNeXtBlock3d per level
    static constexpr int mlp_ratio = 4;
    static constexpr float norm_eps = 1e-6f;                  // LayerNorm32 in blocks
    static constexpr float final_norm_eps = 1e-5f;            // F.layer_norm, no affine
    static constexpr int upsamples = levels - 1;              // 4 x C2S (factor 2) = 16x
    static constexpr int shape_outputs = 7;   // dual vertex 3 (sigmoid), 3 edge-intersection logits, 1 quad weight
    static constexpr int tex_outputs = 6;     // base colour 3, metallic, roughness, alpha (x0.5+0.5)
    static constexpr float voxel_margin = 0.5f;  // vertex = (1 + 2m) * sigmoid(v) - m
    // ConvNeXt block: SubM conv 3^3 (C->C) -> LN(affine) -> Linear(C,4C) -> SiLU -> Linear(4C,C) -> + x.
    // C2S block (C -> C'): subdiv = to_subdiv(x) > 0 ([N,8] logits on the pre-norm input);
    //   h = conv1(SiLU(norm1(x))) with 8*C' outputs -> Channel2Spatial(subdiv) -> LN(no affine)
    //   -> SiLU -> conv2 (C'->C') -> + skip, skip = Channel2Spatial(x) (C/8 per child) with each
    //   channel repeated C' / (C/8) times consecutively.
    // Texture decoder: pred_subdiv = false, the masks come from the shape decoder (guide_subs).
};

// --- Samplers (FlowEulerGuidanceIntervalSampler, pipeline.json) ---
struct SamplerParams {
    int steps;
    float guidance;        // g: pred = g * pos + (1 - g) * neg; 1.0 = no CFG pass
    float rescale;         // CFG rescale on x0 (0 = off)
    float interval_lo, interval_hi;  // CFG only when lo <= t <= hi (rescaled t)
    float rescale_t;       // t = r*s / (1 + (r-1)*s), s = linspace(1, 0, steps+1)
    float sigma_min = 1e-5f;
};
constexpr SamplerParams kSsSampler{12, 7.5f, 0.7f, 0.6f, 1.0f, 5.0f};
constexpr SamplerParams kShapeSampler{12, 7.5f, 0.5f, 0.6f, 1.0f, 3.0f};
constexpr SamplerParams kTexSampler{12, 1.0f, 0.0f, 0.6f, 0.9f, 3.0f};

// Latent normalisation (pipeline.json): sampled slat = z * std + mean; the texture flow
// receives the shape latent as (slat - mean) / std concatenated behind its 32 noise channels.
constexpr float kShapeMean[32] = {
    0.781296f, 0.018091f, -0.495192f, -0.558457f, 1.060530f, 0.093252f, 1.518149f, -0.933218f,
    -0.732996f, 2.604095f, -0.118341f, -2.143904f, 0.495076f, -2.179512f, -2.130751f, -0.996944f,
    0.261421f, -2.217463f, 1.260067f, -0.150213f, 3.790713f, 1.481266f, -1.046058f, -1.523667f,
    -0.059621f, 2.220780f, 1.621212f, 0.877230f, 0.567247f, -3.175944f, -3.186688f, 1.578665f};
constexpr float kShapeStd[32] = {
    5.972266f, 4.706852f, 5.445010f, 5.209927f, 5.320220f, 4.547237f, 5.020802f, 5.444004f,
    5.226681f, 5.683095f, 4.831436f, 5.286469f, 5.652043f, 5.367606f, 5.525084f, 4.730578f,
    4.805265f, 5.124013f, 5.530808f, 5.619001f, 5.103930f, 5.417670f, 5.269677f, 5.547194f,
    5.634698f, 5.235274f, 6.110351f, 5.511298f, 6.237273f, 4.879207f, 5.347008f, 5.405691f};
constexpr float kTexMean[32] = {
    3.501659f, 2.212398f, 2.226094f, 0.251093f, -0.026248f, -0.687364f, 0.439898f, -0.928075f,
    0.029398f, -0.339596f, -0.869527f, 1.038479f, -0.972385f, 0.126042f, -1.129303f, 0.455149f,
    -1.209521f, 2.069067f, 0.544735f, 2.569128f, -0.323407f, 2.293000f, -1.925608f, -1.217717f,
    1.213905f, 0.971588f, -0.023631f, 0.106750f, 2.021786f, 0.250524f, -0.662387f, -0.768862f};
constexpr float kTexStd[32] = {
    2.665652f, 2.743913f, 2.765121f, 2.595319f, 3.037293f, 2.291316f, 2.144656f, 2.911822f,
    2.969419f, 2.501689f, 2.154811f, 3.163343f, 2.621215f, 2.381943f, 3.186697f, 3.021588f,
    2.295916f, 3.234985f, 3.233086f, 2.260140f, 2.874801f, 2.810596f, 3.292720f, 2.674999f,
    2.680878f, 2.372054f, 2.451546f, 2.353556f, 2.995195f, 2.379849f, 2.786195f, 2.775190f};

// --- Mesh extraction (o-voxel flexible_dual_grid_to_mesh, inference path) ---
struct MeshConfig {
    static constexpr float aabb_min = -0.5f;   // object cube [-0.5, 0.5]^3
    static constexpr float aabb_max = 0.5f;
    static constexpr float max_hole_perimeter = 3e-2f;  // Mesh.fill_holes
    // vertex = (coord + dual_vertex) * voxel_size + aabb_min, voxel_size = 1 / resolution.
    // For each voxel and axis a with intersected[a]: the 4 voxels around the edge
    // (offsets below, in order) form a quad of dual vertices; it is split 0-2 if
    // w0*w2 > w1*w3 (w = softplus quad weight) else 1-3.
    static constexpr int edge_neighbour_offsets[3][4][3] = {
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},   // x-axis edge
        {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},   // y-axis edge
        {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},   // z-axis edge
    };
    static constexpr int quad_split_1[6] = {0, 1, 2, 0, 2, 3};
    static constexpr int quad_split_2[6] = {0, 1, 3, 3, 1, 2};
};

// Pipeline resolutions: '512' uses ss 32^3, shape/tex flow 512, decoders at 512.
struct PipelineConfig {
    static constexpr int default_resolution = 512;
    static constexpr int max_tokens_cascade = 49152;   // 1024 cascade token cap (later stage)
    static constexpr int image_cond_512 = 512;         // DINOv3 input size for the 512 path
    static constexpr float preprocess_alpha_threshold = 0.8f * 255.f;  // bbox from alpha > 204
};

}  // namespace gufo::trellis2
