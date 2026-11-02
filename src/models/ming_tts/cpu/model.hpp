// CPU reference runtime for Ming-3.0-tts-0.5B: Qwen2 backbone with KV cache,
// DiT flow-matching head, aggregator, stop/speaker heads and the audio-VAE
// decoder with its ISTFT head. FP32 math on BF16 weights borrowed from the
// mapped checkpoint. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_CPU_MODEL_HPP_
#define GUFO_MODELS_MING_TTS_CPU_MODEL_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/ming_tts/config.hpp"
#include "src/models/ming_tts/cpu/ops.hpp"
#include "src/models/ming_tts/safetensors.hpp"

namespace gufo::models::ming_tts::cpu {

struct Qwen2Layer {
  Linear q, k, v, o, gate, up, down;
  std::vector<float> input_norm;
  std::vector<float> post_norm;
};

struct Qwen2Stack {
  Qwen2Config config;
  std::vector<Qwen2Layer> layers;
  std::vector<float> final_norm;
};

struct DitBlock {
  Linear q, k, v, out, ff_in, ff_out;
  std::vector<float> norm1;
  std::vector<float> norm2;
};

struct DitStack {
  DitConfig config;
  std::vector<DitBlock> blocks;
  std::vector<float> inv_freq;
  std::vector<float> final_norm;
  Linear final_linear;
};

struct FlowHead {  // flowloss.cfm.model
  DitStack stack;
  Linear time_mlp0, time_mlp2, x_embedder, c_embedder;
};

struct Aggregator {  // linear_proj_audio
  DitStack stack;
  Linear x_embedder;
  std::vector<float> cls;  // word_embedder row 0
};

struct VaeEncoder {  // audio.encoder (prompt latents)
  Qwen2Stack backbone;
  Qwen2Stack aggregator;  // 4 layers over [4 frames + CLS] groups
  Linear fc1, fc2, fc3;
  std::vector<float> cls;
};

struct VaeDecoder {  // audio.decoder
  Qwen2Stack stack;
  Linear fc1, head_out;
  std::vector<float> window;  // head.istft.window (BF16 values)
};

// Per-step sampler options (FlowLoss.sample / CFM.sample defaults).
struct FlowOptions {
  float cfg{2.0F};
  float sigma{0.25F};
  float temperature{0.0F};
  std::int32_t steps{10};
};

class Model {
public:
  [[nodiscard]] static std::unique_ptr<Model> Load(const std::string& model_dir,
                                                   std::string* error);

  [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }

  // Token embedding rows -> out[ids.size(), hidden].
  void Embed(std::span<const std::int64_t> ids, float* out) const;
  // spk_head: 192 -> hidden.
  void ProjectSpeaker(std::span<const float> embedding, float* out) const;

  // Backbone with KV cache. Appends `rows` positions; writes the final
  // RMS-normed hidden state of every row to `normed` [rows, hidden].
  // Optional per-layer capture (layer outputs before the final norm).
  void ResetCache();
  [[nodiscard]] std::int64_t cache_length() const noexcept {
    return cache_length_;
  }
  void BackboneForward(const float* embeddings, std::int64_t rows,
                       float* normed,
                       std::vector<std::vector<float>>* layer_outputs = nullptr);

  // softmax(stop_head(z))[1]
  [[nodiscard]] float StopProbability(const float* z) const;

  // One DiT velocity evaluation with classifier-free guidance batch:
  // returns cond and uncond velocities [2, latent_dim].
  void FlowVelocity(const float* x, float t, const float* cond,
                    const float* history, float* cond_uncond) const;
  // Full sampler: integrates from `noise` [latent_dim] to a latent.
  // `sde_noise` ([steps, latent_dim]) is only read when temperature > 0.
  void SampleLatent(const float* cond, const float* history, const float* noise,
                    const float* sde_noise, const FlowOptions& options,
                    float* latent) const;
  [[nodiscard]] static std::vector<float> FlowTimesteps(std::int32_t steps);

  // Aggregator: latent [latent_dim] -> next backbone input [hidden].
  void Aggregate(const float* latent, float* out) const;

  // VAE decoder on [frames, latent_dim] latents -> 44.1 kHz waveform.
  struct VaeTrace {
    std::vector<float> fc1, upsampled, hidden_l0, hidden_last, head_out;
  };
  [[nodiscard]] std::vector<float> DecodeLatents(const float* latents,
                                                 std::int64_t frames,
                                                 VaeTrace* trace = nullptr) const;

  // VAE encoder on a 44.1 kHz prompt waveform (front-padded to a multiple of
  // hop * patch): returns the posterior parameters [patches, 2 * latent_dim]
  // (mean | scale) like AudioVAE.encoder + fc3.
  [[nodiscard]] std::vector<float> EncodePosterior(const float* wave,
                                                   std::int64_t samples,
                                                   std::int64_t* patches) const;
  // mean + (softplus(scale) + 1e-4) * noise, noise [patches, latent_dim].
  [[nodiscard]] std::vector<float> SamplePosterior(
      const std::vector<float>& params, const float* noise) const;

private:
  void DitForward(const DitStack& stack, float* x, std::int64_t batch,
                  std::int64_t seq) const;
  void Qwen2Forward(const Qwen2Stack& stack, float* hidden, std::int64_t rows,
                    std::int64_t first_position, bool use_cache,
                    std::vector<std::vector<float>>* layer_outputs) const;

  ModelConfig config_;
  std::unique_ptr<SafetensorsFile> weights_;
  const std::uint16_t* embed_tokens_{nullptr};
  Qwen2Stack llm_;
  Linear spk_head_, stop_head_;
  FlowHead flow_;
  Aggregator aggregator_;
  VaeDecoder vae_;
  VaeEncoder encoder_;

  // Backbone KV cache: per layer [positions, kv_heads, head_dim].
  mutable std::vector<std::vector<float>> cache_k_;
  mutable std::vector<std::vector<float>> cache_v_;
  std::int64_t cache_length_{0};
};

}  // namespace gufo::models::ming_tts::cpu

#endif  // GUFO_MODELS_MING_TTS_CPU_MODEL_HPP_
