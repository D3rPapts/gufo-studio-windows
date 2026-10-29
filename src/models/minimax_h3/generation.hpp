#ifndef GUFO_MODELS_MINIMAX_H3_GENERATION_HPP_
#define GUFO_MODELS_MINIMAX_H3_GENERATION_HPP_

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/media.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::string_view kH3GenerationSchema =
    "gufo.minimax-h3-text-generation.v1";

struct GenerationParameters {
  std::string preset;
  int internal_width{0};
  int internal_height{0};
  int output_width{0};
  int output_height{0};
  int frames{0};
  int evaluations{0};
  int active_blocks{0};
  int reuse_interval{0};
  bool row_parallel_attention{true};
  bool decode_audio{true};
  bool mux{true};
  std::vector<int> selected_frames;
};

/// "WxH" canvas of the "exact-WxH" preset: multiples of 32, 256..1344 per
/// side and at most 768*1344 pixels (e.g. 832x512).
[[nodiscard]] std::optional<std::pair<int, int>> ParseCanvasSize(
    std::string_view size);
[[nodiscard]] std::optional<GenerationParameters> ResolveGenerationPreset(
    std::string_view name, std::string* error = nullptr);
/// HTTP duration at 24 fps, snapped upward to the official VAE frame grid.
/// The aligned result must fit in 5..15 seconds. "1" selects the existing
/// 22-frame diagnostic extension.
[[nodiscard]] std::optional<int> ResolveDurationFrames(
    std::string_view seconds, std::string* error = nullptr);
[[nodiscard]] bool ValidateGenerationParameters(
    const GenerationParameters& parameters, std::string* error = nullptr);
[[nodiscard]] std::string GenerationParametersJson(
    const GenerationParameters& parameters, std::uint64_t seed,
    std::string_view prompt_sha256);

// Publishes a complete diagnostic PPM frame set by renaming one sibling
// staging directory. Existing output is preserved and never replaced.
[[nodiscard]] bool WriteDiagnosticFrames(const std::filesystem::path& directory,
                                         const VideoFrames& frames,
                                         const CancellationToken* cancellation,
                                         std::string* error = nullptr);
[[nodiscard]] bool WriteDiagnosticLatents(
    const std::filesystem::path& directory, const GenerationGeometry& geometry,
    std::span<const float> video, std::span<const float> audio,
    const CancellationToken* cancellation, std::string* error = nullptr);

/// `GenerationKeyframe::frame` value that places the keyframe by its anchor.
inline constexpr int kKeyframeAtAnchor = std::numeric_limits<int>::min();

/// An FL2VA keyframe or guide frame as decoded RGB8 at its own size;
/// generation fits it to the canvas (cover and centre crop).
struct GenerationKeyframe {
  KeyframeAnchor anchor{KeyframeAnchor::kFirst};
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgb;
  // Pixel frame the image is anchored at (MiniMaxH3AddGuide frame_idx);
  // negative values count from the end (-1 = last frame). kKeyframeAtAnchor
  // uses `anchor` (first or last frame).
  int frame{kKeyframeAtAnchor};
  // Also shown to the text encoder as a "<Picture i>" block when a vision
  // projector is configured (MiniMaxH3ImageToVideo first/last frames). Guide
  // frames of MiniMaxH3AddGuide condition the latents only.
  bool vision{true};
};

/// An audio guide (MiniMaxH3AddGuide audio): 32 kHz stereo PCM anchored at a
/// pixel frame and cropped to the video's remaining audio track.
struct GenerationAudioGuide {
  int frame{0};
  AudioWaveform waveform;
};

struct GenerationRequest {
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path output_path;
  std::filesystem::path frames_directory;
  std::filesystem::path latents_directory;
  std::string prompt;
  std::uint64_t seed{0};
  GenerationParameters parameters;
  // FL2VA keyframes and guide frames (at most kH3MaximumKeyframes, one per
  // pixel frame). Empty: text-to-video.
  std::vector<GenerationKeyframe> keyframes;
  // Audio guides (at most kH3MaximumAudioGuides).
  std::vector<GenerationAudioGuide> audio_guides;
  // Optional transformer LoRA (original/ComfyUI layout), merged at load.
  std::filesystem::path lora_path;
  float lora_strength{1.0F};
  // Qwen3-VL vision tower (mmproj GGUF) for the keyframes' "<Picture i>"
  // blocks in the prompt. Empty: keyframes condition the latents only.
  std::filesystem::path vision_projector;
  // Wan 2.2 runner only (MiniMax H3 ignores them): image-to-video first
  // frame as RGB8 at its own size (empty: text-to-video), negative prompt
  // and classifier-free guidance scale.
  int first_frame_width{0};
  int first_frame_height{0};
  std::vector<std::uint8_t> first_frame;
  std::string negative_prompt;
  float guidance{5.0F};
  // Wan 2.1 camera runner only: "gufo.camera" JSON and an uploaded guide
  // video file.
  std::string camera_json;
  std::filesystem::path guide_video_path;
};

struct GenerationTelemetry {
  double inventory_ms{0.0};
  double tokenizer_ms{0.0};
  double prompt_ms{0.0};
  double keyframe_ms{0.0};
  double audio_guide_ms{0.0};
  std::size_t audio_guides{0};
  std::size_t audio_guide_frames{0};
  double vision_ms{0.0};
  std::size_t vision_tokens{0};
  double denoiser_ms{0.0};
  double video_vae_ms{0.0};
  double first_preview_ms{0.0};
  double audio_vae_ms{0.0};
  double media_ms{0.0};
  double total_ms{0.0};
  std::size_t prompt_tokens{0};
  InspectionTelemetry inventory;
  PromptEncoderTelemetry prompt_encoder;
  VideoVaeEncoderTelemetry keyframe_encoder;
  std::size_t keyframes{0};
  std::size_t lora_modules{0};
  DenoiserTelemetry denoiser;
  VideoVaeTelemetry video_vae;
  AudioVaeTelemetry audio_vae;
  MediaMuxTelemetry media;
};

using GenerationProgress = void (*)(std::string_view phase, int completed,
                                    int total, void* opaque);

[[nodiscard]] bool GenerateTextVideo(const GenerationRequest& request,
                                     const CancellationToken* cancellation,
                                     GenerationProgress progress,
                                     void* progress_opaque,
                                     GenerationTelemetry* telemetry,
                                     std::string* error = nullptr);

[[nodiscard]] std::string GenerationTelemetryJson(
    const GenerationTelemetry& telemetry);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_GENERATION_HPP_
