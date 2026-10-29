#ifndef GUFO_SERVER_VIDEO_JOBS_HPP_
#define GUFO_SERVER_VIDEO_JOBS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/preview.hpp"
#include "src/models/minimax_h3/generation.hpp"

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

// Which model family a video service runs. MiniMax H3 is validated against
// its pinned source manifest; Wan 2.2 TI2V-5B is a Diffusers directory
// (model_index.json names WanPipeline) and reuses GenerationParameters with
// preset "wan22": output size, frames and evaluations (= sampling steps).
enum class VideoModelFamily : std::uint8_t {
  kMinimaxH3,
  kWan22,
};

inline constexpr std::string_view kWan22ModelId = "wan2.2-ti2v-5b";
inline constexpr std::string_view kWan22Preset = "wan22";

// True when `root/model_index.json` describes a Wan 2.2 TI2V-5B pipeline.
[[nodiscard]] bool IsWan22ModelDirectory(const std::filesystem::path& root);

// Wan 2.2 request shape: multiples of 32, 256..1280 per side and at most
// 1280 * 704 pixels; 4k + 1 frames up to 121; 1..100 steps.
[[nodiscard]] bool ValidateWan22Parameters(
    const minimax_h3::GenerationParameters& parameters,
    std::string* error = nullptr);
// Frames at 24 fps for an HTTP duration of 1..5 seconds, snapped to 4k + 1.
[[nodiscard]] std::optional<int> ResolveWan22DurationFrames(
    std::string_view seconds, std::string* error = nullptr);
// Parses "WxH".
[[nodiscard]] bool ParseVideoSize(std::string_view size, int* width,
                                  int* height);

struct VideoJobRequest {
  std::string model{"minimax-h3"};
  std::string prompt;
  std::string size{"512x512"};
  std::string seconds{"1"};
  std::string output_format{"mp4"};
  minimax_h3::GenerationParameters parameters;
  std::uint64_t seed{42};
  // FL2VA keyframes and guide frames (decoded RGB8), one per pixel frame.
  std::vector<minimax_h3::GenerationKeyframe> keyframes;
  // Audio guides (decoded 32 kHz stereo), e.g. the voice at frame 0.
  std::vector<minimax_h3::GenerationAudioGuide> audio_guides;
  // Name of a LoRA registered with the service; empty for none.
  std::string lora;
  std::optional<double> lora_strength;
  // False lets a registered LoRA choose its own step count.
  bool steps_explicit{false};
  // Wan 2.2 only: image-to-video first frame (decoded RGB8 at its own size,
  // empty for text-to-video), negative prompt and guidance scale.
  int first_frame_width{0};
  int first_frame_height{0};
  std::vector<std::uint8_t> first_frame;
  std::string negative_prompt;
  double guidance{5.0};
  // RIFE frame multiplier for the finished MP4 (1 = off, 2..8); needs
  // VideoJobServiceOptions::interpolator.
  int interpolate{1};
  // Live preview (GET /v1/previews/{job id}): a loop of all latent frames,
  // every second step by default ("gufo": {"preview": {...}}).
  PreviewOptions preview{.always = false, .every = 2, .max_side = 320};
};

/// A LoRA the operator registered under a name (`--lora NAME=PATH[,...]`).
struct VideoLora {
  std::filesystem::path path;
  int steps{0};  // 0 keeps the preset's evaluations
  float strength{1.0F};
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
  VideoModelFamily family{VideoModelFamily::kMinimaxH3};
  // Public model id of a Wan 2.2 service (empty: kWan22ModelId).
  std::string model_id;
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path storage_root;
  std::size_t queue_capacity{1};
  std::chrono::seconds artifact_ttl{std::chrono::hours(1)};
  bool validate_model_inventory{true};
  std::function<std::string()> id_factory;
  std::function<std::int64_t()> now;
  std::function<bool(const minimax_h3::GenerationRequest&,
                     const minimax_h3::CancellationToken*,
                     minimax_h3::GenerationProgress, void*,
                     minimax_h3::GenerationTelemetry*, std::string*)>
      runner;
  std::map<std::string, VideoLora, std::less<>> loras;
  // Qwen3-VL vision tower for keyframe prompt blocks; empty disables them.
  std::filesystem::path vision_projector;
  // Frame interpolation of a finished MP4 (input, output, multiplier); empty
  // rejects requests with interpolate > 1. Progress phase "interpolate".
  std::function<bool(const std::filesystem::path&, const std::filesystem::path&,
                     int, const minimax_h3::CancellationToken*,
                     minimax_h3::GenerationProgress, void*, std::string*)>
      interpolator;
  // MiniMax H3 worker recycling. After a large job the process keeps GPU
  // memory that a following large job (e.g. 832x512, 124 frames) cannot do
  // without, and that job fails with a device fault. When a completed job's
  // denoiser had at least `recycle_rows` video rows and no other job waits,
  // `recycle` is called once (with the job id) after the result is
  // persisted; new requests are then refused with 503 until the process
  // exits. Completed jobs are recovered from storage by the next process.
  // 0 or an empty callback disables recycling.
  std::size_t recycle_rows{0};
  std::function<void(const std::string& job_id)> recycle;
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
  [[nodiscard]] VideoModelFamily family() const noexcept;
  // Public model id: "minimax-h3" or the Wan 2.2 id.
  [[nodiscard]] std::string model_id() const;
  [[nodiscard]] std::string initialization_error() const;
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
