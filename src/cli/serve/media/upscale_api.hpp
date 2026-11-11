// Real-ESRGAN image and video upscaling behind `gufo serve upscale
// --device npu|gpu|cpu`:
//   POST /v1/images/upscale                 sync image (or async job)
//   GET  /v1/images/upscale/{id}[/content]  image job status / result
//   POST /v1/videos/upscale                 async video job (frames through
//                                           FFmpeg, audio copied)
//   GET  /v1/videos/{id}[/content]          video job status / MP4
//   DELETE /v1/videos/{id}                  cancel / forget a job
// Written with AI assistance (Claude Code).
// gufo-studio-windows; no upstream capability (see media_service.hpp).
#ifndef GUFO_SERVER_MEDIA_UPSCALE_API_HPP_
#define GUFO_SERVER_MEDIA_UPSCALE_API_HPP_

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/media/media_service.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

inline constexpr std::string_view kUpscaleImagePath = "/v1/images/upscale";
inline constexpr std::string_view kUpscaleVideoPath = "/v1/videos/upscale";

struct UpscaleModelArg {
  std::string name;
  std::filesystem::path path;
  /// "general" | "wdn" | "anime" (from the file name when empty).
  std::string variant;
};

/// `gufo serve upscale` options.
struct UpscaleServeOptions {
  std::string device{"npu"};  // npu | cpu | gpu
  /// npu / cpu: body ONNX files (several sizes may share a name);
  /// gpu: converted Real-ESRGAN safetensors.
  std::vector<UpscaleModelArg> models;
  std::filesystem::path onnxruntime;   // onnxruntime.dll override
  std::filesystem::path npu_ep;        // onnxruntime_vitisai_ep.dll override
  std::filesystem::path npu_cache;     // VitisAI compile cache
  std::filesystem::path npu_config;    // VitisAI config_file
  int threads{0};                      // CPU provider threads
  bool fp32{false};                    // gpu: FP32 validation path
};

/// Result of one upscale.
struct UpscaleOutcome {
  std::vector<std::uint8_t> rgb24;
  int width{0};
  int height{0};
  double body_seconds{0};
  double post_seconds{0};
  std::string model_file;
};

/// One served weight file.
struct UpscaleModelInfo {
  std::string name;     // public model id
  std::string variant;  // general | wdn | anime | "" (ONNX bodies: by name)
  std::filesystem::path path;
  int convolutions{0};  // gpu: from the safetensors header
  int scale{4};
  std::size_t parameters{0};
};

/// Device-specific engine: ONNX Runtime (npu / cpu) or HIP (gpu).
struct UpscaleEngine {
  std::string device;
  /// Public model names, the first is the default.
  std::vector<std::string> model_names;
  /// Every served file (name + variant); ResolveUpscaleModel picks one.
  std::vector<UpscaleModelInfo> models;
  /// Runtime description for /v1/models and logs.
  std::string runtime;
  /// rgb24 w x h -> out_w x out_h (0 x 0: native x4) with fit (crop | pad |
  /// stretch) through the file named (model, variant). Called under the
  /// service lock.
  std::function<bool(const std::string& model, const std::string& variant,
                     const std::uint8_t* rgb24, int w, int h, int out_w,
                     int out_h, const std::string& fit, UpscaleOutcome* out,
                     std::string* error)>
      run;
};

/// Builds the engine for options.device; nullptr + *error on failure.
[[nodiscard]] std::unique_ptr<UpscaleEngine> CreateUpscaleEngine(
    const UpscaleServeOptions& options, std::string* error);

/// Index into engine.models for a request's `model` ("" = the default) and
/// `variant` ("" = the first file of that model); a variant name alone
/// ("wdn") also selects its file. -1 with *error when nothing matches.
[[nodiscard]] int ResolveUpscaleModel(const UpscaleEngine& engine,
                                      std::string_view model,
                                      std::string_view variant,
                                      std::string* error);

/// Real-ESRGAN upscaling service, one device at a time.
///
/// POST /v1/images/upscale
///   {model, variant ("general" | "wdn" | "anime"), image (data URL /
///    base64 / https / file:), scale (2 | 4, default 4; 2 is the x4 picture
///    resized to half) or size "WxH" | width + height, fit ("crop" default |
///    "pad" | "stretch"), output ("png" default | "jpg"), quality (JPEG,
///    default 90), response_format ("b64_json" default | "png" | "jpg" for
///    the bytes), async (false)}
///   -> {created, model, variant, device, width, height, format, data:
///       [{b64_json}], cost: {...}} or the image bytes; async: 202 + job.
/// GET /v1/images/upscale/{id}            job status
/// GET /v1/images/upscale/{id}/content    the image (bytes or b64_json)
///
/// POST /v1/videos/upscale
///   {model, variant, video (MP4 data URL / base64 / https / file:), scale
///    (2 | 4, default 4) or size "WxH" | width + height (even), fit ("pad"
///    default), crf (18), keep_audio (true), async (true)}
///   -> 202 + a video-job object {id, object: "video", status, progress,
///      ...}; the router records the id like a /v1/videos job.
/// GET    /v1/videos/{id}           status; GET /v1/videos/{id}/content: MP4
/// DELETE /v1/videos/{id}           cancels a running job / drops a result
///
/// Sync image requests run on the request thread (at most kMaxWaiting
/// wait); jobs run one after another on a worker thread, frame by frame
/// between the sync requests. Finished jobs are kept for an hour (at most
/// kKeptJobs results).
class UpscaleService final : public MediaService {
public:
  static constexpr int kMaxWaiting = 8;
  static constexpr std::size_t kQueueCapacity = 8;
  static constexpr std::size_t kKeptJobs = 8;

  explicit UpscaleService(std::unique_ptr<UpscaleEngine> engine);
  ~UpscaleService();
  UpscaleService(const UpscaleService&) = delete;
  UpscaleService& operator=(const UpscaleService&) = delete;

  [[nodiscard]] bool ready() const override { return engine_ != nullptr; }
  [[nodiscard]] std::string model_id() const override {
    return engine_->model_names.front();
  }
  [[nodiscard]] std::vector<std::string> model_ids() const override {
    return engine_->model_names;
  }
  [[nodiscard]] const UpscaleEngine& engine() const { return *engine_; }
  [[nodiscard]] std::string_view capability() const override {
    return "upscale";
  }
  [[nodiscard]] std::string_view owned_by() const override {
    return "operator-supplied-realesrgan";
  }
  /// /v1/models: device and runtime; /ready: device.
  void Describe(json::Value* entry, bool ready) const override;
  [[nodiscard]] bool HandlesPath(std::string_view path) const override;
  [[nodiscard]] std::string_view unavailable_message() const override {
    return "upscaler is not ready";
  }
  [[nodiscard]] std::string_view unavailable_code() const override {
    return "upscale_service_unavailable";
  }

  HttpResponse Handle(const HttpRequest& request) override;

private:
  struct Job;

  HttpResponse HandleImage(const HttpRequest& request);
  HttpResponse SubmitVideo(const HttpRequest& request);
  HttpResponse JobStatus(const std::string& id, bool content);
  HttpResponse CancelJob(const std::string& id);
  HttpResponse ImageResponse(const Job& job);

  /// Parses the fields shared by images and videos into the job (model,
  /// variant, scale / size, fit); an error response when they are invalid.
  std::optional<HttpResponse> ParseCommon(const json::Value& body, Job* job,
                                          bool video);
  bool RunImage(Job* job, bool sync, const HttpRequest* request);
  void RunVideo(Job* job);
  bool Enqueue(const std::shared_ptr<Job>& job, std::string* error);
  void Worker();
  void ExpireLocked();

  std::unique_ptr<UpscaleEngine> engine_;
  std::mutex run_mutex_;
  std::mutex count_mutex_;
  int waiting_{0};

  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::uint64_t next_id_{1};
  bool stop_{false};
  std::thread worker_;
};

/// /v1/images/upscale[/...], /v1/videos/upscale and /v1/videos/{id}[/...]
/// (the job reads of this service; `gufo serve upscale` has no other video
/// service).
[[nodiscard]] bool IsUpscaleApiPath(std::string_view path) noexcept;

/// "1920x1080" -> 1920, 1080.
[[nodiscard]] bool ParseUpscaleSize(std::string_view text, int* w, int* h);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_UPSCALE_API_HPP_
