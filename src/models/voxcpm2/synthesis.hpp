// VoxCPM2 synthesis pipeline (VoxCPM.generate -> VoxCPM2Model
// _generate_with_prompt_cache -> _inference): text (+ "(instruction)" style
// prefix) -> prompt layout (zero-shot, reference voice, continuation or
// both) -> AR loop (base LM, residual LM, DiT flow head, local encoder, stop
// head) -> streaming AudioVAE decoder at 48 kHz.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_SYNTHESIS_HPP_
#define GUFO_MODELS_VOXCPM2_SYNTHESIS_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/voxcpm2/backend.hpp"

namespace gufo::models::voxcpm2 {

struct SynthesisRequest {
  std::string text;
  // Style / emotion instruction; sent as "(instruction)text" (VoxCPM2
  // "controllable cloning"). Empty: the text is used as is (it may already
  // start with a bracket instruction).
  std::string instruction;
  // Reference voice (isolated prefix, VoxCPM2 reference mode), mono.
  std::vector<float> reference_audio;
  std::uint32_t reference_sample_rate{0};
  // Continuation prompt (audio + its transcript), mono.
  std::vector<float> prompt_audio;
  std::uint32_t prompt_sample_rate{0};
  std::string prompt_text;

  float cfg{2.0F};
  std::int32_t timesteps{10};
  std::uint64_t seed{4242};
  std::int32_t min_len{2};
  std::int32_t max_len{4096};
  bool retry_badcase{true};
  std::int32_t retry_badcase_max_times{3};
  float retry_badcase_ratio_threshold{6.0F};

  // Called with the 48 kHz audio of every AR step (160 ms) while generating;
  // false cancels. Disables the bad-case retry (like voxcpm streaming).
  std::function<bool(std::span<const float>)> on_audio;
  // Test hook: initial flow noise [patch, latent] of AR step `step` in
  // attempt `attempt` (both 0-based) instead of the internal RNG.
  std::function<void(std::int32_t attempt, std::int64_t step, float* noise)> noise_source;
};

struct SynthesisResult {
  std::uint32_t sample_rate{48000};
  std::vector<float> samples;
  std::int64_t steps{0};       // generated patches (without prompt context)
  std::int32_t attempts{1};    // 1 + bad-case retries
  std::vector<std::int64_t> input_ids;
  std::vector<float> patches;  // generated latents [steps, patch, latent]
};

class Engine {
public:
  [[nodiscard]] static std::unique_ptr<Engine> Create(const std::string& model_dir,
                                                      std::string* error);
  ~Engine();

  [[nodiscard]] bool Generate(const SynthesisRequest& request,
                              const std::function<bool()>& is_cancelled,
                              SynthesisResult* result, std::string* error);

  [[nodiscard]] std::uint32_t sample_rate() const noexcept;
  // "hip" or "cpu": where the AR loop runs (the AudioVAE runs on the CPU).
  [[nodiscard]] const char* backend_name() const noexcept;

  // AudioVAE encoder patches [patches, patch, latent] for a mono waveform
  // (resampled to 16 kHz, padded to whole patches on the right or left).
  [[nodiscard]] std::vector<float> EncodeAudio(std::span<const float> audio,
                                               std::uint32_t sample_rate, bool pad_left);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

// Whitespace clean-up of VoxCPM.generate ("\n" -> " ", runs of whitespace -> " ").
[[nodiscard]] std::string NormalizeWhitespace(const std::string& text);

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_SYNTHESIS_HPP_
