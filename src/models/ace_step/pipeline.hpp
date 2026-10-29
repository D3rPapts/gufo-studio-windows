#ifndef GUFO_MODELS_ACE_STEP_PIPELINE_HPP_
#define GUFO_MODELS_ACE_STEP_PIPELINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::ace_step {

/// Files of one ACE-Step 1.5 XL turbo installation.
struct ModelPaths {
  std::filesystem::path dit;        // acestep_v1.5_xl_turbo_bf16.safetensors
  std::filesystem::path text;       // qwen_0.6b_ace15.safetensors
  std::filesystem::path vae;        // Diffusers Oobleck vae safetensors
  std::filesystem::path tokenizer;  // Qwen3 tokenizer.json
  std::filesystem::path silence;    // silence_latent.safetensors [frames, 64]
  std::filesystem::path lm;         // optional qwen_4b_ace15.safetensors

  /// Resolves the files below `root`: `dit.safetensors`,
  /// `text_encoder.safetensors`, `vae.safetensors`, `tokenizer.json` and
  /// `silence_latent.safetensors` (links into the ComfyUI / Hugging Face
  /// downloads are fine), and the optional 5 Hz LM `lm.safetensors`. False
  /// and `error` when a required file is missing.
  [[nodiscard]] static bool FromRoot(const std::filesystem::path& root,
                                     ModelPaths* paths, std::string* error);
};

/// One text-to-music request (official ACE-Step 1.5 prompt format).
struct MusicRequest {
  std::string caption;         // style tags / description
  std::string lyrics;          // "[Instrumental]" when empty
  std::string language{"en"};  // lyric language code
  std::string bpm;             // "" -> N/A
  std::string keyscale;        // e.g. "A minor", "" -> N/A
  std::string timesignature;   // "4", "3", ... "" -> N/A
  double duration{30.0};       // seconds, 1..600
  std::uint64_t seed{42};
  int steps{8};  // turbo: 8
  float shift{3.0F};
  /// Plan the song with the 5 Hz LM first (when the LM is loaded), as
  /// ComfyUI's `generate_audio_codes` default does.
  bool audio_codes{true};
  float lm_cfg{2.0F};
  float lm_temperature{0.85F};
  float lm_top_p{0.9F};
};

struct MusicTelemetry {
  double text_seconds{0};
  double condition_seconds{0};
  double diffusion_seconds{0};
  double vae_seconds{0};
  double lm_seconds{0};
  int codes{0};
  int frames{0};
  int text_tokens{0};
  int lyric_tokens{0};
};

/// Progress callback: fraction 0..1 and a short phase name. Returning false
/// cancels the generation.
using MusicProgress = std::function<bool(double, std::string_view)>;

/// The text-to-music pipeline: tokenizer, Qwen3-0.6B text encoder, condition
/// encoder, turbo DiT sampler (Euler, no CFG) and the Oobleck VAE. All parts
/// stay loaded (about 12 GB of device memory).
class MusicPipeline {
public:
  [[nodiscard]] static std::unique_ptr<MusicPipeline> Load(
      const ModelPaths& paths, std::string* error);
  ~MusicPipeline();
  MusicPipeline(const MusicPipeline&) = delete;
  MusicPipeline& operator=(const MusicPipeline&) = delete;

  /// Planar stereo audio [2][samples] at 48 kHz, peak-normalized to <= 1.
  bool Generate(const MusicRequest& request, std::vector<float>* audio,
                MusicTelemetry* telemetry, const MusicProgress& progress,
                std::string* error);

private:
  struct Impl;
  explicit MusicPipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Text prompts of the official pipeline (exposed for tests).
std::string MusicTextPrompt(const MusicRequest& request);
std::string MusicLyricPrompt(const MusicRequest& request);
/// 5 Hz LM prompts: open assistant turn after the metadata "think" block;
/// the unconditional prompt is "NO USER INPUT" with an empty think block.
std::string MusicLmPrompt(const MusicRequest& request);
std::string MusicLmUnconditionalPrompt();

/// Standard normal noise [count] from `seed` (gufo's own generator).
std::vector<float> MusicNoise(std::uint64_t seed, std::size_t count);

/// 16-bit PCM WAV of planar stereo audio.
std::string EncodeStereoWav(std::span<const float> planar, int sample_rate);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_PIPELINE_HPP_
