// Ming-3.0-tts-0.5B synthesis pipeline (MingTTS.synthesize for the dense
// checkpoint): text fragments -> prompt -> AR loop (backbone, DiT flow head,
// stop head, aggregator) -> VAE decoder -> trailing-silence trim.
// Runs on the CPU reference runtime. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_SYNTHESIS_HPP_
#define GUFO_MODELS_MING_TTS_SYNTHESIS_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ming_tts {

struct SynthesisRequest {
  std::string text;
  // Target language code or name ("de", "German", ...); empty: no
  // "Target language" line.
  std::string language{"de"};
  // EMOTION task: label or name (高兴/happy/fröhlich, 悲伤/sad, 愤怒/angry,
  // 恐惧/fear, 厌恶/disgust, 惊讶/surprise, 中性/neutral).
  std::string emotion;
  // FREEFORM task: free-text caption (e.g. a whisper description). Takes
  // precedence over `emotion`; must not be combined with the controls below.
  std::string instruction;
  // BASIC task: speaking rate (慢速/中速/快速), pitch and volume (低/中/高),
  // as values or names accepted by SpeedLabel/PitchLabel/VolumeLabel. Any
  // subset may be set; unset ones stay null in the caption.
  std::string speed;
  std::string pitch;
  std::string volume;
  // DIALECT task: "cantonese" or "chuanyu dialect" (DialectLabel names). As
  // in the reference, a dialect request drops the "Target language" line.
  std::string dialect;
  // Reference voice (mono) for the CAM++ speaker embedding.
  std::vector<float> reference_audio;
  std::uint32_t reference_sample_rate{0};
  // Transcript of reference_audio: enables prompt latents from the VAE
  // encoder (the reference ZERO_SHOT layout).
  std::string reference_text;
  // Test hook: posterior noise for the prompt latents ([patches, 64]).
  std::vector<float> prompt_noise;
  // Precomputed 192-dim CAM++ embedding; overrides reference_audio.
  std::vector<float> speaker_embedding;

  float cfg{2.0F};
  float sigma{0.25F};
  float temperature{0.0F};
  std::uint64_t seed{4242};
  std::int32_t max_decode_steps{1000};
  std::size_t max_fragment_length{24};
  bool trim_silence{true};
  // Send the text as one fragment (MingTTS raw_text=True).
  bool raw_text{false};

  // Called with each finished fragment's 44.1 kHz audio; false cancels.
  std::function<bool(std::span<const float>)> on_audio;
  // Test hook: fills the initial flow noise (latent_dim values) for an AR
  // step instead of the internal RNG. `fragment` and `step` are 0-based.
  std::function<void(std::size_t fragment, std::int64_t step, float* noise)>
      noise_source;
};

// BASIC/DIALECT caption from request.speed/pitch/volume/dialect (zh schema
// for BASIC, en schema for DIALECT, as tts_tasks.build_task_instruction).
// Leaves *instruction empty when none of them is set. Fails on unknown
// values and, like the reference (one task per request), when BASIC and
// DIALECT are mixed or combined with `emotion` or `instruction`.
[[nodiscard]] bool ResolveTaskInstruction(const SynthesisRequest& request,
                                          std::optional<std::string>* instruction,
                                          std::string* error);

struct SynthesisResult {
  std::uint32_t sample_rate{44100};
  std::vector<float> samples;
  std::vector<std::int64_t> fragment_steps;  // AR steps per fragment
  std::vector<std::string> fragments;
  std::vector<float> speaker_embedding;      // the 192 values used
  std::vector<float> latents;                // last fragment, [steps, 64]
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
  // "hip" or "cpu": where the AR loop and the VAE decoder run.
  [[nodiscard]] const char* backend_name() const noexcept;

  // Prompt latents [patches, 64] for a mono reference waveform (resampled to
  // 44.1 kHz, front padded, VAE encoder, posterior sample with `noise` when it
  // has the right size, otherwise drawn from `rng`).
  [[nodiscard]] std::vector<float> EncodePrompt(std::span<const float> audio,
                                                std::uint32_t sample_rate,
                                                std::span<const float> noise,
                                                std::mt19937_64* rng);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

// ming_tts.trim_trailing_silence (0.1 s frames, threshold 1e-3, 0.3 s tail).
[[nodiscard]] std::vector<float> TrimTrailingSilence(std::span<const float> speech,
                                                     std::int64_t sample_rate);

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_SYNTHESIS_HPP_
