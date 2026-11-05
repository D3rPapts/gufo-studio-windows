// CPU reference runtime for VoxCPM2: MiniCPM-4 base LM and residual LM with
// KV caches, local encoder, local DiT with the UnifiedCFM Euler sampler (CFG
// zero-star), FSQ, fusion and stop heads. FP32 math on BF16 weights borrowed
// from the mapped checkpoint. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_CPU_MODEL_HPP_
#define GUFO_MODELS_VOXCPM2_CPU_MODEL_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/ming_tts/cpu/ops.hpp"
#include "src/models/ming_tts/safetensors.hpp"
#include "src/models/voxcpm2/backend.hpp"
#include "src/models/voxcpm2/config.hpp"

namespace gufo::models::voxcpm2::cpu {

using ming_tts::cpu::Linear;

struct Layer {
  Linear q, k, v, o, gate, up, down;
  std::vector<float> input_norm;
  std::vector<float> post_norm;
};

struct Stack {
  StackConfig config;
  std::vector<Layer> layers;
  std::vector<float> final_norm;
};

// Static KV cache of one LM: per layer [positions, kv_heads, head_dim].
struct KvCache {
  std::vector<std::vector<float>> k;
  std::vector<std::vector<float>> v;
  std::int64_t length{0};
};

struct LocalEncoder {
  Linear in_proj;
  std::vector<float> special_token;
  Stack stack;
};

struct LocalDit {
  Linear in_proj, cond_proj, out_proj;
  Linear time_mlp1, time_mlp2, delta_mlp1, delta_mlp2;
  Stack stack;
};

struct Fsq {
  Linear in_proj, out_proj;
};

// Optional captures for the oracle test.
struct PrefillTrace {
  std::vector<float> locenc_l0_row0;  // [patch + 1, enc_hidden] of row 0
  std::vector<float> locenc_out;      // [rows, enc_hidden]
  std::vector<float> feat_embed;      // [rows, hidden]
  std::vector<float> combined;        // [rows, hidden]
  std::vector<float> base_l0;         // [rows, hidden]
  std::vector<float> base_out;        // [rows, hidden] (final norm)
  std::vector<float> enc_outputs;     // [rows, hidden] after FSQ mixing
  std::vector<float> residual_inputs; // [rows, hidden]
  std::vector<float> residual_out;    // [rows, hidden]
};

class Model final : public Backend {
public:
  [[nodiscard]] static std::unique_ptr<Model> Load(const std::string& model_dir,
                                                   std::string* error);

  // Backend
  [[nodiscard]] const char* name() const noexcept override { return "cpu"; }
  void Prefill(const std::int64_t* ids, const float* feats, const std::uint8_t* audio_mask,
               std::int64_t rows, float* lm_hidden, float* residual_hidden) override {
    PrefillTraced(ids, feats, audio_mask, rows, lm_hidden, residual_hidden, nullptr);
  }
  void SamplePatch(const float* lm_hidden, const float* residual_hidden,
                   const float* prev_patch, const float* noise, const FlowOptions& options,
                   float* patch) override;
  void StopLogits(const float* lm_hidden, float* logits) override;
  void Advance(const float* patch, float* lm_hidden, float* residual_hidden) override;

  void PrefillTraced(const std::int64_t* ids, const float* feats,
                     const std::uint8_t* audio_mask, std::int64_t rows, float* lm_hidden,
                     float* residual_hidden, PrefillTrace* trace);

  [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }

  // Building blocks (exposed for the oracle test and the GPU upload).
  void Embed(std::span<const std::int64_t> ids, float* out) const;
  // Local encoder on `count` patches [count, patch, latent] -> CLS outputs
  // [count, enc_hidden]; optional capture of layer 0 for the first patch.
  void LocalEncode(const float* patches, std::int64_t count, float* out,
                   std::vector<float>* layer0_row0 = nullptr) const;
  // dit_hidden = [lm_to_dit(lm_hidden), res_to_dit(residual_hidden)] [2 * dit_hidden]
  void DitCondition(const float* lm_hidden, const float* residual_hidden, float* mu) const;
  // One CFG estimator call (batch 2: conditional, mu = 0): x, cond [patch,
  // latent], scalar time t -> velocities [2, patch, latent].
  void Velocity(const float* x, const float* mu, const float* cond, float t,
                float* out) const;
  // UnifiedCFM t_span (linspace(1, 0, n + 1) + sway).
  [[nodiscard]] static std::vector<float> TimeSpan(std::int32_t timesteps, float sway);
  // Full Euler sampler from `noise` [patch, latent].
  void SampleFlow(const float* mu, const float* cond, const float* noise,
                  const FlowOptions& options, float* patch) const;
  void FsqForward(const float* x, std::int64_t rows, float* out) const;

  [[nodiscard]] const Stack& base_lm() const noexcept { return base_; }
  [[nodiscard]] const Stack& residual_lm() const noexcept { return residual_; }
  [[nodiscard]] const LocalEncoder& local_encoder() const noexcept { return locenc_; }
  [[nodiscard]] const LocalDit& local_dit() const noexcept { return dit_; }
  [[nodiscard]] const Fsq& fsq() const noexcept { return fsq_; }
  [[nodiscard]] const Linear& enc_to_lm() const noexcept { return enc_to_lm_; }
  [[nodiscard]] const Linear& lm_to_dit() const noexcept { return lm_to_dit_; }
  [[nodiscard]] const Linear& res_to_dit() const noexcept { return res_to_dit_; }
  [[nodiscard]] const Linear& fusion() const noexcept { return fusion_; }
  [[nodiscard]] const Linear& stop_proj() const noexcept { return stop_proj_; }
  [[nodiscard]] const Linear& stop_head() const noexcept { return stop_head_; }
  [[nodiscard]] const std::uint16_t* embed_tokens() const noexcept { return embed_; }
  // RoPE tables [positions, head_dim / 2] (cos / sin, LongRoPE scaled).
  [[nodiscard]] const std::vector<float>& rope_cos() const noexcept { return rope_cos_; }
  [[nodiscard]] const std::vector<float>& rope_sin() const noexcept { return rope_sin_; }
  [[nodiscard]] std::int64_t rope_positions() const noexcept { return rope_positions_; }
  // Grows the RoPE tables to at least `positions` rows.
  void PrepareRope(std::int64_t positions) const { EnsureRope(positions); }
  // delta_time_mlp(sinusoidal(0)) (dt is always 0 without mean mode) [dit_hidden].
  [[nodiscard]] const std::vector<float>& delta_time() const noexcept { return delta_time_; }
  // SinusoidalPosEmb(t) [dit_hidden].
  void TimeEmbedding(float t, float* out) const;

  [[nodiscard]] std::int64_t base_length() const noexcept { return base_cache_.length; }

private:
  void StackForward(const Stack& stack, float* x, std::int64_t batch, std::int64_t seq,
                    bool causal, KvCache* cache, std::vector<float>* layer0) const;
  void ApplyRope(float* x, std::int64_t seq, std::int64_t heads, std::int64_t head_dim,
                 std::int64_t first_position) const;
  void EnsureRope(std::int64_t positions) const;

  ModelConfig config_;
  std::unique_ptr<ming_tts::SafetensorsFile> weights_;
  const std::uint16_t* embed_{nullptr};
  Stack base_, residual_;
  LocalEncoder locenc_;
  LocalDit dit_;
  Fsq fsq_;
  Linear enc_to_lm_, lm_to_dit_, res_to_dit_, fusion_, stop_proj_, stop_head_;
  std::vector<float> delta_time_;
  mutable std::vector<float> rope_cos_, rope_sin_;
  mutable std::int64_t rope_positions_{0};
  KvCache base_cache_, residual_cache_;
};

}  // namespace gufo::models::voxcpm2::cpu

#endif  // GUFO_MODELS_VOXCPM2_CPU_MODEL_HPP_
