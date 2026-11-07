#ifndef GUFO_SERVER_VIDEO_JOBS_HPP_
#define GUFO_SERVER_VIDEO_JOBS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/media/video_postprocess.hpp"
#include "src/ext/video.hpp"

namespace gufo::server {

enum class VideoJobStatus : std::uint8_t {
  kQueued,
  kInProgress,
  kCompleted,
  kFailed,
  kCancelled,
};

[[nodiscard]] std::string_view ToString(VideoJobStatus status) noexcept;

enum class VideoJobResult : std::uint8_t {
  kOk,
  kInvalid,
  kQueueFull,
  kNotFound,
  kNotReady,
  kExpired,
  kIoError,
};

struct VideoJobRequest {
  std::string model{"minimax-h3"};
  std::string prompt;
  std::string size{"512x512"};
  std::string seconds{"1"};
  std::string output_format{"mp4"};
  ext::VideoParameters parameters;
  std::uint64_t seed{42};
  /// Provider-owned options and inputs (`VideoProvider::prepare_request`).
  std::string options_json;
  std::vector<ext::VideoAttachment> attachments;
  /// gufo-studio-windows post-steps after the runner (RIFE, Real-ESRGAN);
  /// need VideoJobServiceOptions::postprocessors.
  VideoPostProcess post;
  /// Post-processing only (MP4 bytes, e.g. ComfyUI LoadVideo -> RIFE VFI):
  /// nothing is generated, the upload is interpolated and/or upscaled. Needs
  /// a post-step; prompt, size and parameters are not used.
  std::vector<std::uint8_t> source_video;
};

struct VideoJobSnapshot {
  std::string id;
  VideoJobStatus status{VideoJobStatus::kQueued};
  int progress{0};
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  std::int64_t expires_at{0};
  std::string model;
  std::string size;
  std::string seconds;
  std::string output_format;
  std::string error_code;
  std::string error_message;
};

struct VideoJobContent {
  std::filesystem::path path;
  std::string media_type;
  std::uint64_t bytes{0};
};

struct VideoJobServiceOptions {
  /// Video provider id; empty: the provider whose probe accepts
  /// `model_root` (highest `probe_priority`), else `minimax-h3`.
  std::string provider_id;
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path storage_root;
  std::size_t queue_capacity{1};
  std::chrono::seconds artifact_ttl{std::chrono::hours(1)};
  /// Additional IDs accepted from requests. They behave as the root
  /// `minimax-h3` name; the preset-selecting variants stay fixed (#957).
  std::vector<std::string> model_aliases;
  bool validate_model_inventory{true};
  std::function<std::string()> id_factory;
  std::function<std::int64_t()> now;
  std::function<bool(const ext::VideoRequest&, const ext::CancellationCheck&,
                     ext::VideoProgress, void*, ext::VideoTelemetry*,
                     std::string*)>
      runner;
  /// gufo-studio-windows: RIFE / Real-ESRGAN post-steps; empty members
  /// reject requests that ask for them.
  VideoPostProcessors postprocessors;
};

struct VideoJobCreateResult {
  VideoJobResult result{VideoJobResult::kInvalid};
  std::optional<VideoJobSnapshot> job;
  std::string error;
};

struct VideoJobLookupResult {
  VideoJobResult result{VideoJobResult::kNotFound};
  std::optional<VideoJobSnapshot> job;
  std::string error;
};

struct VideoJobContentResult {
  VideoJobResult result{VideoJobResult::kNotFound};
  std::optional<VideoJobContent> content;
  std::string error;
};

class VideoJobService {
public:
  explicit VideoJobService(VideoJobServiceOptions options);
  ~VideoJobService();

  VideoJobService(const VideoJobService&) = delete;
  VideoJobService& operator=(const VideoJobService&) = delete;
  VideoJobService(VideoJobService&&) = delete;
  VideoJobService& operator=(VideoJobService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  /// The provider this service runs (nullptr when initialization failed).
  [[nodiscard]] const ext::VideoProvider* provider() const noexcept;
  /// Canonical model name: the provider id of a provider with
  /// `prepare_request`, else `minimax-h3`.
  [[nodiscard]] std::string model_id() const;
  /// True for a canonical `minimax-h3` name or a configured alias (#957).
  [[nodiscard]] bool serves_model(std::string_view name) const;
  /// The request name as a canonical model: aliases map to the root
  /// `minimax-h3`, unknown names are returned unchanged for rejection.
  [[nodiscard]] std::string normalize_model(std::string_view name) const;
  [[nodiscard]] const std::vector<std::string>& model_aliases() const;
  [[nodiscard]] VideoJobCreateResult Create(const VideoJobRequest& request);
  [[nodiscard]] VideoJobLookupResult Get(std::string_view id);
  [[nodiscard]] VideoJobContentResult Content(std::string_view id);
  [[nodiscard]] VideoJobLookupResult Delete(std::string_view id);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_JOBS_HPP_
