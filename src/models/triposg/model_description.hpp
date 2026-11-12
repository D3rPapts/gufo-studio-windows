// TripoSG: tensor inventory of the converted gufo weight file (one safetensors,
// names below) as produced by tools/triposg/convert_hf_to_gufo.py from the three HF
// files transformer/diffusion_pytorch_model.safetensors (554 tensors, 1.44 B params),
// vae/diffusion_pytorch_model.safetensors (356 tensors, 243 M) and
// image_encoder_dinov2/model.safetensors (439 tensors, 304 M); all F32 upstream.
//
// The converter (a) keeps HF names under the prefixes "dit.", "vae." and "dino.",
// (b) un-interleaves the head-major q/k/v layout of TripoSGAttnProcessor2_0 (see
// kHeadInterleaved), (c) bakes the 224-px position table for DINOv2 and (d) stores
// BF16 for matrices and F32 for norms/biases/tables.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "src/models/triposg/config.hpp"

namespace gufo::triposg {

struct TensorSpec {
    std::string_view name;   // name in the gufo file (printf-style %d for the block index)
    std::array<std::size_t, 4> shape;  // trailing zeros unused
    std::size_t repeat;      // number of blocks for %d names (1 = single tensor)
    bool bf16;               // stored as BF16 (true) or F32 (false)
};

// Upstream TripoSG stacks q|k|v (self) or k|v (cross) and views the result as
// [tokens, heads, 3*head_dim] / [tokens, heads, 2*head_dim]. Hence per head h the
// query rows of the *stacked* matrix are [h*3d, h*3d+d), the key rows
// [h*3d+d, h*3d+2d) and the value rows [h*3d+2d, h*3d+3d). The converter rebuilds
// plain to_q/to_k/to_v with rows [h*d, (h+1)*d) so gufo's attention sees the usual
// layout. Cross-attention queries are NOT interleaved upstream and are kept as is.
constexpr bool kHeadInterleaved = true;

// clang-format off
constexpr TensorSpec kDitTensors[] = {
    {"dit.proj_in.weight",               {DitConfig::width, DitConfig::in_channels, 0, 0}, 1, true},
    {"dit.proj_in.bias",                 {DitConfig::width, 0, 0, 0},                      1, false},
    {"dit.time_proj.linear_1.weight",    {DitConfig::mlp, DitConfig::width, 0, 0},         1, true},
    {"dit.time_proj.linear_1.bias",      {DitConfig::mlp, 0, 0, 0},                        1, false},
    {"dit.time_proj.linear_2.weight",    {DitConfig::width, DitConfig::mlp, 0, 0},         1, true},
    {"dit.time_proj.linear_2.bias",      {DitConfig::width, 0, 0, 0},                      1, false},
    {"dit.blocks.%d.norm1.weight",       {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.norm1.bias",         {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.attn1.to_q.weight",  {DitConfig::width, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn1.to_k.weight",  {DitConfig::width, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn1.to_v.weight",  {DitConfig::width, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn1.norm_q.weight",{DitConfig::head_dim, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.attn1.norm_k.weight",{DitConfig::head_dim, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.attn1.to_out.0.weight", {DitConfig::width, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn1.to_out.0.bias",{DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.norm2.weight",       {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.norm2.bias",         {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.attn2.to_q.weight",  {DitConfig::width, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn2.to_k.weight",  {DitConfig::width, DitConfig::cross_dim, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn2.to_v.weight",  {DitConfig::width, DitConfig::cross_dim, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn2.norm_q.weight",{DitConfig::head_dim, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.attn2.norm_k.weight",{DitConfig::head_dim, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.attn2.to_out.0.weight", {DitConfig::width, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.attn2.to_out.0.bias",{DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.norm3.weight",       {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.norm3.bias",         {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.ff.net.0.proj.weight", {DitConfig::mlp, DitConfig::width, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.ff.net.0.proj.bias", {DitConfig::mlp, 0, 0, 0}, DitConfig::layers, false},
    {"dit.blocks.%d.ff.net.2.weight",    {DitConfig::width, DitConfig::mlp, 0, 0}, DitConfig::layers, true},
    {"dit.blocks.%d.ff.net.2.bias",      {DitConfig::width, 0, 0, 0}, DitConfig::layers, false},
    // Only blocks 11..20 carry skip_linear/skip_norm (10 blocks); %d runs 11..20.
    {"dit.blocks.%d.skip_linear.weight", {DitConfig::width, 2 * DitConfig::width, 0, 0}, DitConfig::layers - DitConfig::skip_first_pop, true},
    {"dit.blocks.%d.skip_linear.bias",   {DitConfig::width, 0, 0, 0}, DitConfig::layers - DitConfig::skip_first_pop, false},
    {"dit.blocks.%d.skip_norm.weight",   {DitConfig::width, 0, 0, 0}, DitConfig::layers - DitConfig::skip_first_pop, false},
    {"dit.blocks.%d.skip_norm.bias",     {DitConfig::width, 0, 0, 0}, DitConfig::layers - DitConfig::skip_first_pop, false},
    {"dit.norm_out.weight",              {DitConfig::width, 0, 0, 0}, 1, false},
    {"dit.norm_out.bias",                {DitConfig::width, 0, 0, 0}, 1, false},
    {"dit.proj_out.weight",              {DitConfig::in_channels, DitConfig::width, 0, 0}, 1, true},
    {"dit.proj_out.bias",                {DitConfig::in_channels, 0, 0, 0}, 1, false},
};

// VAE decoder (the encoder, quant and FPS sampling are training/point-cloud only and
// are not converted by default; the converter keeps them with --with-encoder).
constexpr TensorSpec kVaeTensors[] = {
    {"vae.post_quant.weight",            {VaeConfig::width_decoder, VaeConfig::latent_channels, 0, 0}, 1, true},
    {"vae.post_quant.bias",              {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    // Blocks 0..15: self attention (norm1/attn1, norm3/ff); no qk norm, no qkv bias.
    {"vae.decoder.blocks.%d.norm1.weight", {VaeConfig::width_decoder, 0, 0, 0}, VaeConfig::layers_decoder, false},
    {"vae.decoder.blocks.%d.norm1.bias",   {VaeConfig::width_decoder, 0, 0, 0}, VaeConfig::layers_decoder, false},
    {"vae.decoder.blocks.%d.attn1.to_q.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, VaeConfig::layers_decoder, true},
    {"vae.decoder.blocks.%d.attn1.to_k.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, VaeConfig::layers_decoder, true},
    {"vae.decoder.blocks.%d.attn1.to_v.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, VaeConfig::layers_decoder, true},
    {"vae.decoder.blocks.%d.attn1.to_out.0.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, VaeConfig::layers_decoder, true},
    {"vae.decoder.blocks.%d.attn1.to_out.0.bias", {VaeConfig::width_decoder, 0, 0, 0}, VaeConfig::layers_decoder, false},
    {"vae.decoder.blocks.%d.norm3.weight", {VaeConfig::width_decoder, 0, 0, 0}, VaeConfig::layers_decoder + 1, false},
    {"vae.decoder.blocks.%d.norm3.bias",   {VaeConfig::width_decoder, 0, 0, 0}, VaeConfig::layers_decoder + 1, false},
    {"vae.decoder.blocks.%d.ff.net.0.proj.weight", {VaeConfig::mlp_decoder, VaeConfig::width_decoder, 0, 0}, VaeConfig::layers_decoder + 1, true},
    {"vae.decoder.blocks.%d.ff.net.0.proj.bias", {VaeConfig::mlp_decoder, 0, 0, 0}, VaeConfig::layers_decoder + 1, false},
    {"vae.decoder.blocks.%d.ff.net.2.weight", {VaeConfig::width_decoder, VaeConfig::mlp_decoder, 0, 0}, VaeConfig::layers_decoder + 1, true},
    {"vae.decoder.blocks.%d.ff.net.2.bias", {VaeConfig::width_decoder, 0, 0, 0}, VaeConfig::layers_decoder + 1, false},
    // Block 16: cross attention from the query points (norm2/attn2 with norm_cross).
    {"vae.decoder.blocks.16.norm2.weight",  {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.blocks.16.norm2.bias",    {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.blocks.16.attn2.norm_cross.weight", {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.blocks.16.attn2.norm_cross.bias",   {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.blocks.16.attn2.to_q.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, 1, true},
    {"vae.decoder.blocks.16.attn2.to_k.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, 1, true},
    {"vae.decoder.blocks.16.attn2.to_v.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, 1, true},
    {"vae.decoder.blocks.16.attn2.to_out.0.weight", {VaeConfig::width_decoder, VaeConfig::width_decoder, 0, 0}, 1, true},
    {"vae.decoder.blocks.16.attn2.to_out.0.bias", {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.proj_query.weight",    {VaeConfig::width_decoder, VaeConfig::query_channels, 0, 0}, 1, false},
    {"vae.decoder.proj_query.bias",      {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.norm_out.weight",      {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.norm_out.bias",        {VaeConfig::width_decoder, 0, 0, 0}, 1, false},
    {"vae.decoder.proj_out.weight",      {1, VaeConfig::width_decoder, 0, 0}, 1, false},
    {"vae.decoder.proj_out.bias",        {1, 0, 0, 0}, 1, false},
};

// DINOv2-L (HF Dinov2Model names kept; q/k/v are fused into qkv by the converter so
// the layout matches wan22/moge.hip's `attn.qkv`).
constexpr TensorSpec kDinoTensors[] = {
    {"dino.embeddings.cls_token",        {1, 1, DinoConfig::width, 0}, 1, false},
    {"dino.embeddings.patch_embeddings.projection.weight", {DinoConfig::width, 3, DinoConfig::patch, DinoConfig::patch}, 1, false},
    {"dino.embeddings.patch_embeddings.projection.bias",   {DinoConfig::width, 0, 0, 0}, 1, false},
    // Original 1370-row table and the bicubic (A=-0.75, scale (16+0.1)/37, HF
    // transformers 4.45 semantics) 257-row table for 224 px, baked by the converter.
    {"dino.embeddings.position_embeddings", {1, DinoConfig::trained_positions, DinoConfig::width, 0}, 1, false},
    {"dino.embeddings.position_embeddings_224", {DinoConfig::tokens, DinoConfig::width, 0, 0}, 1, false},
    {"dino.encoder.layer.%d.norm1.weight", {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.norm1.bias",   {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.attention.qkv.weight", {3 * DinoConfig::width, DinoConfig::width, 0, 0}, DinoConfig::layers, true},
    {"dino.encoder.layer.%d.attention.qkv.bias",   {3 * DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.attention.output.dense.weight", {DinoConfig::width, DinoConfig::width, 0, 0}, DinoConfig::layers, true},
    {"dino.encoder.layer.%d.attention.output.dense.bias",   {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.layer_scale1.lambda1", {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.norm2.weight", {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.norm2.bias",   {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.mlp.fc1.weight", {DinoConfig::mlp, DinoConfig::width, 0, 0}, DinoConfig::layers, true},
    {"dino.encoder.layer.%d.mlp.fc1.bias",   {DinoConfig::mlp, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.mlp.fc2.weight", {DinoConfig::width, DinoConfig::mlp, 0, 0}, DinoConfig::layers, true},
    {"dino.encoder.layer.%d.mlp.fc2.bias",   {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.encoder.layer.%d.layer_scale2.lambda1", {DinoConfig::width, 0, 0, 0}, DinoConfig::layers, false},
    {"dino.layernorm.weight",            {DinoConfig::width, 0, 0, 0}, 1, false},
    {"dino.layernorm.bias",              {DinoConfig::width, 0, 0, 0}, 1, false},
};
// clang-format on

// Metadata keys written into the safetensors header by the converter.
constexpr std::string_view kMetaFormat = "gufo.triposg.format";   // "1"
constexpr std::string_view kMetaSource = "gufo.triposg.source";   // "VAST-AI/TripoSG@<revision>"
constexpr std::string_view kMetaLicense = "gufo.triposg.license"; // "MIT (VAST-AI-Research)"

}  // namespace gufo::triposg
