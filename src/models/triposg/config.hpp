// TripoSG (VAST-AI/TripoSG, MIT weights): architecture constants as published in the
// HF config.json files (transformer/, vae/, image_encoder_dinov2/, scheduler/).
// Values were read from the HF raw files on 2026-11-12; nothing is downloaded here.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gufo::triposg {

// --- Image conditioning: HF Dinov2Model "facebook/dinov2-large" ---------------
struct DinoConfig {
    static constexpr int image_size = 224;        // BitImageProcessor: shortest edge 256, centre crop 224
    static constexpr int patch = 14;
    static constexpr int grid = image_size / patch;  // 16
    static constexpr int tokens = 1 + grid * grid;   // 257 (CLS + patches, no register tokens)
    static constexpr int width = 1024;
    static constexpr int heads = 16;
    static constexpr int head_dim = width / heads;   // 64
    static constexpr int layers = 24;
    static constexpr int mlp = 4096;
    static constexpr float layer_norm_eps = 1e-6f;
    static constexpr int trained_grid = 37;          // position table 1 + 37*37 = 1370 rows (518 px)
    static constexpr int trained_positions = 1 + trained_grid * trained_grid;
    // Normalisation (ImageNet), applied after rescale 1/255.
    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float std[3] = {0.229f, 0.224f, 0.225f};
    // Output: last_hidden_state after the final `layernorm`, [257, 1024]. The
    // unconditional branch for CFG is an all-zero [257, 1024] tensor.
};

// --- Rectified-flow DiT: TripoSGDiTModel ------------------------------------
struct DitConfig {
    static constexpr int in_channels = 64;      // latent channels (= out_channels)
    static constexpr int latent_tokens = 2048;  // pipeline default num_tokens
    static constexpr int width = 2048;
    static constexpr int heads = 16;
    static constexpr int head_dim = width / heads;  // 128 (matches gufo's wan22 attention kernel)
    static constexpr int layers = 21;
    static constexpr int mlp = width * 4;           // 8192, exact (erf) GELU
    static constexpr int cross_dim = 1024;          // DINOv2 width
    static constexpr float norm_eps = 1e-5f;        // FP32LayerNorm in blocks
    static constexpr float qk_norm_eps = 1e-6f;     // RMSNorm over q/k per head ([128] weights)
    static constexpr float out_norm_eps = 1e-5f;    // norm_out (LayerNorm default)
    // Timestep: sinusoidal 2048-d (sin first, then cos, no flip, shift 0, max period 10000),
    // then linear_1 (2048 -> 8192) + GELU + linear_2 (8192 -> 2048). Prepended as token 0,
    // so the DiT sequence length is 1 + latent_tokens = 2049.
    static constexpr int time_tokens = 1;
    static constexpr int sequence = time_tokens + latent_tokens;
    // Long skips: blocks 0..9 push their output, blocks 11..20 pop (LIFO): block 11 gets
    // block 9's output, ..., block 20 gets block 0's. Block 10 (the middle) has no skip.
    static constexpr int skip_first_push = 0;
    static constexpr int skip_last_push = layers / 2 - 1;  // 9
    static constexpr int skip_first_pop = layers / 2 + 1;  // 11
    // Skip combine (skip_concat_front, skip_norm_last): h = skip_norm(skip_linear(cat(skip, h))).
};

// --- Shape VAE: TripoSGVAEModel (only the decoder is needed for inference) ---
struct VaeConfig {
    static constexpr int latent_channels = 64;
    static constexpr int latent_tokens = DitConfig::latent_tokens;
    static constexpr int width_decoder = 1024;
    static constexpr int width_encoder = 512;
    static constexpr int heads = 8;
    static constexpr int head_dim = width_decoder / heads;  // 128
    static constexpr int layers_decoder = 16;  // self-attention blocks (kv cache), then 1 cross block
    static constexpr int layers_encoder = 8;
    static constexpr int mlp_decoder = width_decoder * 4;  // 4096
    static constexpr float norm_eps = 1e-5f;
    // Query embedding: FrequencyPositionalEmbedding, 8 log-spaced freqs 2^0..2^7 (no pi),
    // include_input -> 3 * (1 + 2*8) = 51 channels; proj_query 51 -> 1024.
    static constexpr int embed_freqs = 8;
    static constexpr int query_channels = 3 * (1 + 2 * embed_freqs);  // 51
    static constexpr int encoder_in_channels = 3 + query_channels;      // 54 (encoder proj_in)
    // The last block is cross attention: q from the embedded query points, k/v from the
    // kv cache with `norm_cross` LayerNorm(1024); then norm_out + proj_out (1024 -> 1).
    // The decoder returns -logit. Measured on real latents (mug, 12.11.): this value is
    // NEGATIVE inside the shape and positive outside (SDF-like), level set 0.
};

// --- Scheduler: RectifiedFlowScheduler ---------------------------------------
struct SchedulerConfig {
    static constexpr int train_timesteps = 1000;
    static constexpr float shift = 1.f;  // no shift: sigma_i = 1 - i/N, i = 0..N-1, then 0
    // t passed to the DiT is sigma * 1000; Euler step x += (sigma_i - sigma_{i+1}) * v.
    static constexpr int default_steps = 50;
    static constexpr float default_guidance = 7.f;  // CFG with zero image tokens as negative
};

// --- Geometry extraction defaults (hierarchical_extract_geometry) -------------
struct GeometryConfig {
    static constexpr float bounds = 1.005f;        // cube [-1.005, 1.005]^3
    static constexpr int dense_octree_depth = 8;   // 256^3 dense query grid
    static constexpr int hierarchical_depth = 9;   // refine the |logit| < 1 band at 512^3
    static constexpr float band_threshold = 1.f;
    static constexpr int query_chunk = 50000;      // decoder query points per launch
};

}  // namespace gufo::triposg
