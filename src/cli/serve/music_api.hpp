#ifndef GUFO_SERVER_MUSIC_API_HPP_
#define GUFO_SERVER_MUSIC_API_HPP_

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
#include "src/models/ace_step/pipeline.hpp"

namespace gufo::server {

inline constexpr std::string_view kMusicApiPath = "/v1/audio/music";

/// ACE-Step 1.5 text-to-music behind /v1/audio/music:
///   POST /v1/audio/music               {prompt|tags, lyrics, duration, seed,
///        steps, bpm, keyscale, timesignature, language, audio_codes,
///        lm_cfg, lm_temperature, lm_top_p, async} -> audio/wav
///        (async: 202 + job JSON)
///   GET  /v1/audio/music/{id}          job status
///   GET  /v1/audio/music/{id}/content  audio/wav
/// One worker generates jobs in order; finished audio is kept for an hour
/// (at most kKeptJobs results).
class MusicService {
public:
  static constexpr std::size_t kQueueCapacity = 8;
  static constexpr std::size_t kKeptJobs = 16;

  MusicService(const std::filesystem::path& model_root, std::string model_id);
  ~MusicService();
  MusicService(const MusicService&) = delete;
  MusicService& operator=(const MusicService&) = delete;

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
  std::unique_ptr<models::ace_step::MusicPipeline> pipeline_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::uint64_t next_id_{1};
  bool stop_{false};
  std::thread worker_;
};

[[nodiscard]] bool IsMusicApiPath(std::string_view path) noexcept;

}  // namespace gufo::server

#endif  // GUFO_SERVER_MUSIC_API_HPP_
