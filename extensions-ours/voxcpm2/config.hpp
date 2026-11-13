// VoxCPM2 (OpenBMB, Apache-2.0) model configuration.
//
// VoxCPM2 is a tokenizer-free diffusion-autoregressive TTS model: a MiniCPM-4
// text-semantic LM (28 layers) and a residual acoustic LM (8 layers, no RoPE)
// drive a local DiT flow-matching head (12 layers) that samples one patch of
// four 64-dim audio-VAE latents per step; a local encoder (12 layers) feeds
// each patch back. AudioVAE V2 (causal convolutions with Snake activations)
// encodes 16 kHz reference audio and decodes 48 kHz speech.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_CONFIG_HPP_
#define GUFO_MODELS_VOXCPM2_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::models::voxcpm2 {

// One MiniCPM-4 transformer stack (all four stacks share the layer design).
struct StackConfig {
  std::int32_t hidden_size{2048};
  std::int32_t intermediate_size{6144};
  std::int32_t num_layers{28};
  std::int32_t num_heads{16};
  std::int32_t num_kv_heads{2};
  std::int32_t head_dim{128};
  bool rope{true};
};

struct AudioVaeConfig {
  std::int32_t encoder_dim{128};
  std::vector<std::int32_t> encoder_rates{2, 5, 8, 8};
  std::int32_t latent_dim{64};
  std::int32_t decoder_dim{2048};
  std::vector<std::int32_t> decoder_rates{8, 6, 5, 2, 2, 2};
  std::vector<std::int32_t> sr_bin_boundaries{20000, 30000, 40000};
  std::int32_t sample_rate{16000};
  std::int32_t out_sample_rate{48000};

  [[nodiscard]] std::int32_t chunk_size() const;         // encoder hop (640)
  [[nodiscard]] std::int32_t decode_chunk_size() const;  // decoder hop (1920)
};

struct ModelConfig {
  StackConfig base_lm;      // text-semantic LM (with token embeddings)
  StackConfig residual_lm;  // residual acoustic LM
  StackConfig encoder;      // local encoder (LocEnc)
  StackConfig dit;          // local DiT estimator
  std::int32_t vocab_size{73448};
  float rms_norm_eps{1e-5F};
  float rope_theta{10000.0F};
  std::vector<float> rope_short_factor;  // LongRoPE (head_dim / 2 values)
  std::vector<float> rope_long_factor;
  std::int32_t max_position_embeddings{32768};
  std::int32_t original_max_position_embeddings{32768};
  bool use_mup{false};
  float scale_emb{12.0F};
  float scale_depth{1.4F};
  std::int32_t patch_size{4};
  std::int32_t feat_dim{64};
  std::int32_t fsq_latent_dim{512};
  std::int32_t fsq_scale{9};
  float sigma_min{1e-6F};
  float inference_cfg_rate{2.0F};
  bool dit_mean_mode{false};
  AudioVaeConfig vae;
  std::int32_t max_length{8192};

  // LongRoPE: the factors picked for max_position_embeddings and the cos/sin
  // scaling factor sqrt(1 + log(scale) / log(original)).
  [[nodiscard]] const std::vector<float>& rope_factors() const;
  [[nodiscard]] float rope_scaling() const;
};

// Special token ids of the VoxCPM2 prompt layout.
inline constexpr std::int64_t kAudioStartToken = 101;
inline constexpr std::int64_t kAudioEndToken = 102;
inline constexpr std::int64_t kRefAudioStartToken = 103;
inline constexpr std::int64_t kRefAudioEndToken = 104;

[[nodiscard]] std::optional<ModelConfig> ParseModelConfig(const std::string& json_text,
                                                          std::string* error = nullptr);
[[nodiscard]] std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir, std::string* error = nullptr);

// True for a directory with a VoxCPM2 config.json ("architecture": "voxcpm2"),
// model.safetensors and audiovae.pth (or audiovae.safetensors).
[[nodiscard]] bool LooksLikeVoxCpm2(const std::string& model_dir);

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_CONFIG_HPP_
