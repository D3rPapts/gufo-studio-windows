#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/ext/image.hpp"  // LatentPreview, PreviewSink (live previews)

namespace gufo::ext {

struct VideoParameters {
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

/// A named binary input of a video request: an uploaded multipart file, or
/// bytes a provider decoded from the request body (images, audio, video).
/// The bytes stay in their uploaded container format; providers decode them.
struct VideoAttachment {
  std::string name;
  std::vector<std::uint8_t> bytes;
};

struct VideoRequest {
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path output_path;
  std::filesystem::path frames_directory;
  std::filesystem::path latents_directory;
  std::string prompt;
  std::uint64_t seed{0};
  VideoParameters parameters;
  /// Provider-owned conditioning and options as a JSON object, produced by
  /// the provider's `prepare_request` (empty for providers without one).
  std::string options_json;
  /// Binary inputs referenced by `options_json` (images, audio, video).
  std::vector<VideoAttachment> attachments;
  /// Live preview of this job (gufo-studio-windows, changes/0.2/
  /// live-previews.md); empty when the provider publishes none.
  PreviewSink preview;
};

/// One `POST /v1/videos` request as the server received it, for providers
/// that parse their own conditioning (`VideoProvider::prepare_request`).
struct VideoRequestInput {
  /// Canonical model name (aliases already mapped to the provider id).
  std::string_view model;
  /// The request body as a JSON object; multipart text fields appear as
  /// strings, a multipart `gufo` field as its parsed object.
  std::string_view body_json;
  /// Multipart file parts by field name, in upload order.
  std::span<const VideoAttachment> files;
};

/// A request the provider accepted: what the job service queues and later
/// hands back to the runner in `VideoRequest`.
struct VideoPreparedRequest {
  std::string prompt;
  /// Echoed in the job object (`size`, `seconds`).
  std::string size;
  std::string seconds;
  std::string output_format{"mp4"};
  std::uint64_t seed{42};
  VideoParameters parameters;
  std::string options_json;
  std::vector<VideoAttachment> attachments;
};

/// A rejected request: HTTP status (400 by default), OpenAI-style error code
/// and message.
struct VideoRequestError {
  int status{400};
  std::string code{"invalid_request"};
  std::string message;
};

struct VideoTelemetry {
  double inventory_ms{0.0};
  double tokenizer_ms{0.0};
  double prompt_ms{0.0};
  double denoise_ms{0.0};
  double video_vae_ms{0.0};
  double first_preview_ms{0.0};
  double audio_vae_ms{0.0};
  double media_ms{0.0};
  double total_ms{0.0};
  std::size_t prompt_tokens{0};
  std::string extra_json;
};

struct VideoResult {
  bool success{false};
  VideoTelemetry telemetry;
  std::string error;
};

using VideoProgress = void (*)(std::string_view phase, int completed, int total,
                               void* opaque);
using CancellationCheck = std::function<bool()>;

/// Abstract capability runner for video generation models.
class VideoRunner {
public:
  virtual ~VideoRunner() = default;

  [[nodiscard]] virtual bool Generate(const VideoRequest& request,
                                      const CancellationCheck& is_cancelled,
                                      VideoProgress progress,
                                      void* progress_opaque,
                                      VideoTelemetry* telemetry,
                                      std::string* error) = 0;
};

using VideoProbe = bool (*)(std::string_view model_root);
using VideoArtifactResolver = bool (*)(std::string_view model_root,
                                       std::string_view source_manifest,
                                       std::string* error);
using VideoRunnerFactory = std::unique_ptr<VideoRunner> (*)(
    const std::string& model_root, std::string* error);

struct VideoProvider {
  std::string id;
  VideoProbe probe = nullptr;
  VideoArtifactResolver resolve = nullptr;
  VideoRunnerFactory create = nullptr;
  std::optional<VideoParameters> (*resolve_preset)(
      std::string_view name, std::string* error) = nullptr;
  std::optional<int> (*resolve_duration_frames)(std::string_view seconds,
                                                std::string* error) = nullptr;
  bool (*validate_parameters)(const VideoParameters& parameters,
                              std::string* error) = nullptr;
  std::string (*parameters_json)(const VideoParameters& parameters,
                                 std::uint64_t seed,
                                 std::string_view prompt_sha256) = nullptr;
  std::string (*telemetry_json)(const VideoTelemetry& telemetry) = nullptr;
  /// Optional: parses and validates a whole request, including image, audio
  /// and video conditioning the core does not model. When set, the server
  /// accepts the provider id (plus configured aliases) as the model name and
  /// skips its own preset and duration handling for this provider.
  bool (*prepare_request)(const VideoRequestInput& input,
                          VideoPreparedRequest* prepared,
                          VideoRequestError* error) = nullptr;
  /// Probe order when several providers accept a model directory: the
  /// highest priority wins, ties keep registration order.
  int probe_priority{0};
  /// Latent space of `VideoRequest::preview` latents for the host's
  /// latent->RGB map ("wan22": 48 channels, "wan21": 16 channels, "minimax_h3")
  /// and the preview loop's frame rate. Null: no previews.
  const char* preview_latent = nullptr;
  int preview_fps{24};
};

void RegisterVideoProvider(VideoProvider provider);
const VideoProvider* FindVideoProvider(std::string_view id);
const VideoProvider* ProbeVideoProvider(std::string_view model_root);

}  // namespace gufo::ext
