// MMAudio video / text to audio ("foley") behind /v1/audio/foley.
// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_FOLEY_API_HPP_
#define GUFO_SERVER_FOLEY_API_HPP_

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "src/cli/serve/http_server.hpp"
#include "src/models/mmaudio/pipeline.hpp"

namespace gufo::server {

inline constexpr std::string_view kFoleyApiPath = "/v1/audio/foley";

/// MMAudio large_44k_v2 behind /v1/audio/foley:
///   POST /v1/audio/foley  {prompt, negative_prompt, duration (8), steps (25),
///        cfg_strength (4.5), seed, video | image (data URL, base64 or
///        https), mask_away_clip, frame_mode ("resample" | "consecutive"),
///        force_rate, skip_first_frames, select_every_nth, frame_load_cap,
///        async} -> audio/wav (44.1 kHz mono; async: 202 + job JSON)
///   GET  /v1/audio/foley/{id}          job status
///   GET  /v1/audio/foley/{id}/content  audio/wav
/// Without video / image it is text to audio. frame_mode "consecutive"
/// reproduces ComfyUI-MMAudio's MMAudioSampler (frames at the source rate,
/// after the VHS_LoadVideo options); "resample" is MMAudio's own 8 / 25 fps
/// sampling. One worker runs the jobs in order; results are kept for an hour.
class FoleyService {
public:
  static constexpr std::size_t kQueueCapacity = 8;
  static constexpr std::size_t kKeptJobs = 16;

  FoleyService(const std::filesystem::path& model_root, std::string model_id);
  ~FoleyService();
  FoleyService(const FoleyService&) = delete;
  FoleyService& operator=(const FoleyService&) = delete;

  [[nodiscard]] bool ready() const { return pipeline_ != nullptr; }
  [[nodiscard]] const std::string& initialization_error() const {
    return init_error_;
  }
  [[nodiscard]] const std::string& model_id() const { return model_id_; }

  HttpResponse Handle(const HttpRequest& request);

private:
  struct Job;
  HttpResponse Submit(const HttpRequest& request);
  HttpResponse Status(const std::string& id, bool content);
  void Worker();
  void ExpireLocked();

  std::string model_id_;
  std::string init_error_;
  std::unique_ptr<models::mmaudio::FoleyPipeline> pipeline_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::uint64_t next_id_{1};
  bool stop_{false};
  std::thread worker_;
};

[[nodiscard]] bool IsFoleyApiPath(std::string_view path) noexcept;

}  // namespace gufo::server

#endif  // GUFO_SERVER_FOLEY_API_HPP_
