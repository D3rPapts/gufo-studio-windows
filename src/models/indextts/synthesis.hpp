// IndexTTS-2.5 (German fine-tune) synthesis engine: IndexTTS2.infer_generator
// with the settings of indextts_de.py (lang "de", prepare_german, text
// normalization off, 8-value emotion vector, num_beams 1, temperature 0.7,
// top_p 0.8, top_k 30, repetition_penalty 10), reference conditioning cached
// per voice, output 22.05 kHz int16 (clamp(32767 * wav) truncated).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_SYNTHESIS_HPP_
#define GUFO_MODELS_INDEXTTS_SYNTHESIS_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/indextts/gpt.hpp"

namespace gufo::models::indextts {

inline constexpr std::int64_t kOutputRate = 22050;

// True for an IndexTTS-2.5 bundle directory (config.yaml "version: 2.5",
// gpt.pth / german/gpt.pth, s2mel.pth, codec.pth, w2v-bert-2.0, campplus,
// bigvgan_v2_22khz_80band_256x, multilingual tiktoken).
[[nodiscard]] bool LooksLikeIndexTts25(const std::filesystem::path& dir);

// Emotion names (English and German, case-insensitive) -> index in
// [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]; -1 if unknown.
[[nodiscard]] int EmotionIndex(const std::string& name);

// Emotion spec of the serve API: "" -> none; eight comma-separated numbers
// (emo_vector order); one name ("Wut", "sad") -> weight 1.0; or
// "name:weight, name:weight". Weights must be finite and in [0, 1.2].
// Returns false (with `error`) for anything else.
[[nodiscard]] bool ParseEmotionSpec(const std::string& spec,
                                    std::optional<std::array<float, 8>>* out,
                                    std::string* error);

struct SynthesisRequest {
  std::string text;
  std::vector<float> reference;  // mono
  std::int64_t reference_rate{0};
  std::optional<std::array<float, 8>> emotion;  // emo_vector
  float emo_alpha{1.0F};
  SamplingOptions sampling;
  std::uint64_t seed{1234};
  std::string language{"de"};
  float duration_factor{1.0F};
  std::int32_t interval_silence_ms{200};
  std::int32_t max_text_tokens_per_segment{120};
  // Test hooks: fixed uniforms for the AR sampler and CFM noise [t, 80].
  const std::vector<double>* forced_uniforms{nullptr};
  const std::vector<float>* forced_noise{nullptr};
};

struct SynthesisResult {
  std::vector<std::int16_t> samples;  // 22.05 kHz
  std::vector<std::int32_t> codes;    // last segment
  double reference_seconds{0.0};
  double gpt_seconds{0.0};
  double s2mel_seconds{0.0};
  double vocoder_seconds{0.0};
  double first_audio_seconds{0.0};
  std::int32_t segments{0};
  std::int32_t gpt_steps{0};
};

class Engine {
public:
  [[nodiscard]] static std::unique_ptr<Engine> Create(const std::filesystem::path& bundle,
                                                      std::string* error);
  ~Engine();

  using CancelFn = std::function<bool()>;
  // Called per finished segment (int16 samples, silence included between segments).
  using ChunkFn = std::function<void(std::span<const std::int16_t>)>;
  [[nodiscard]] bool Generate(const SynthesisRequest& request, const CancelFn& cancelled,
                              const ChunkFn& on_chunk, SynthesisResult* result,
                              std::string* error);
  [[nodiscard]] const char* backend_name() const noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_SYNTHESIS_HPP_
