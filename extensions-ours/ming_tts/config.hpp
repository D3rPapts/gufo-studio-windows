// Ming-3.0-tts-0.5B (inclusionAI, MIT) model configuration.
//
// The checkpoint is a Qwen2.5-0.5B backbone driving a continuous-latent TTS
// head: a DiT flow-matching head samples one 64-dim audio-VAE latent per step,
// an aggregator projects it back into the backbone, and a Qwen2-style VAE
// decoder with an ISTFT head renders 44.1 kHz audio. Written with AI
// assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_CONFIG_HPP_
#define GUFO_MODELS_MING_TTS_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>

namespace gufo::models::ming_tts {

// One Qwen2 decoder stack (backbone or VAE decoder).
struct Qwen2Config {
  std::int32_t hidden_size{896};
  std::int32_t intermediate_size{4864};
  std::int32_t num_layers{24};
  std::int32_t num_heads{14};
  std::int32_t num_kv_heads{2};
  std::int32_t head_dim{64};
  float rms_norm_eps{1e-6F};
  float rope_theta{1000000.0F};
  // Keys j with j <= i - sliding_window are masked (transformers SDPA mask);
  // 0 disables the window.
  std::int32_t sliding_window{0};
  std::int32_t vocab_size{151936};
};

// DiT flow head and aggregator share one block design.
struct DitConfig {
  std::int32_t hidden_size{1024};
  std::int32_t depth{8};
  std::int32_t num_heads{16};
  std::int32_t mlp_ratio{4};
  std::int32_t patch_size{1};
  std::int32_t history_patch_size{1};
};

struct ModelConfig {
  Qwen2Config llm;
  Qwen2Config vae_decoder;
  Qwen2Config vae_encoder;            // enc_kwargs.backbone
  std::int32_t vae_encoder_aggregator_layers{4};
  DitConfig dit;
  DitConfig aggregator;
  std::int32_t latent_dim{64};
  std::int32_t vae_patch_size{4};    // latent -> VAE frames upsampling
  std::int32_t vae_hop_size{882};    // samples per VAE frame
  std::int32_t sample_rate{44100};
  std::int32_t speaker_dim{192};
};

[[nodiscard]] std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir, std::string* error = nullptr);

// True for a Ming-3.0-tts dense (0.5B) checkpoint directory: config.json with
// architectures BailingMMNativeForConditionalGeneration, model_type "dense",
// plus model.safetensors and campplus.onnx.
[[nodiscard]] bool LooksLikeMingTts(const std::string& model_dir);

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_CONFIG_HPP_
