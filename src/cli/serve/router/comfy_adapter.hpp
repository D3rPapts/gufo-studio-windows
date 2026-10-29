#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_

// ComfyUI-compatible front for the router (`gufo router --comfy-dir DIR`):
// POST /prompt takes an API-format workflow, comfy_graph maps it to gufo
// endpoints, and one worker thread runs the jobs in order through the
// router's own OpenAI-style API. Results are served the ComfyUI way:
// /history/{id}, /view?filename=.., /queue, /upload/image, /free,
// /interrupt, /system_stats, /object_info and the /ws progress socket.
//
// Files live below DIR: input/ (uploads), output/ (Save* nodes), temp/
// (Preview* nodes and work files) and history.json (finished prompts).

#include <atomic>
#include <chrono>
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

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/core/json.hpp"

namespace gufo::router::comfy {

struct FetchResult {
  int status{0};
  std::string body;
  std::string content_type;
};

/// Sends one request to the router's own model API and returns the whole
/// response. `request.is_cancelled` aborts the wait.
using Fetch = std::function<FetchResult(const server::HttpRequest&)>;

struct AdapterOptions {
  std::filesystem::path root;
  /// ffmpeg for FLAC output, PNG frames from videos and non-WAV voice
  /// references; empty = WAV output, no frame extraction.
  std::filesystem::path ffmpeg;
  ModelIds models;
  Fetch fetch;
  /// POST /free: unloads idle workers. Optional.
  std::function<void()> unload;
  std::chrono::milliseconds video_poll{std::chrono::seconds(2)};
  std::size_t history_limit{2000};
};

class Adapter {
public:
  explicit Adapter(AdapterOptions options);
  ~Adapter();
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  void Start();
  /// Cancels the running job and joins the worker.
  void Stop();

  /// The ComfyUI response for `request`, or nullopt when the path is not
  /// one of ComfyUI's (the router then routes it as usual).
  std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& request);

  struct Job;

private:
  server::HttpResponse Submit(const server::HttpRequest& request);
  server::HttpResponse QueueGet();
  server::HttpResponse QueuePost(const server::HttpRequest& request);
  server::HttpResponse HistoryGet(const server::HttpRequest& request,
                                  std::string_view id);
  server::HttpResponse HistoryPost(const server::HttpRequest& request);
  server::HttpResponse Interrupt(const server::HttpRequest& request);
  server::HttpResponse Free(const server::HttpRequest& request);
  server::HttpResponse View(const server::HttpRequest& request);
  server::HttpResponse Upload(const server::HttpRequest& request);
  server::HttpResponse SystemStats();
  server::HttpResponse ObjectInfo(std::string_view class_type);
  server::HttpResponse Socket(const server::HttpRequest& request);

  void WorkerMain();
  void Execute(const std::shared_ptr<Job>& job);
  json::Value RunTask(const std::shared_ptr<Job>& job, const Task& task);
  json::Value RunImage(const std::shared_ptr<Job>& job, const Task& task,
                       const ImageTask& spec);
  json::Value RunSpeech(const std::shared_ptr<Job>& job, const Task& task,
                        const SpeechTask& spec);
  json::Value RunDialogue(const std::shared_ptr<Job>& job, const Task& task,
                          const DialogueTask& spec);
  std::string Speak(const std::shared_ptr<Job>& job, const SpeechTask& spec);
  json::Value SaveSpeech(const Task& task,
                         const std::vector<std::string>& wavs);
  json::Value RunVideo(const std::shared_ptr<Job>& job, const Task& task,
                       const VideoTask& spec);  json::Value RunMusic(const std::shared_ptr<Job>& job, const Task& task,
                       const MusicTask& spec);

  std::filesystem::path FilePath(const FileRef& ref) const;
  std::string ReadFileRef(const FileRef& ref) const;
  json::Value SaveBytes(const OutputTarget& target, std::string_view extension,
                        const std::string& bytes);
  json::Value SaveVideoFrames(const OutputTarget& target,
                              const std::filesystem::path& video);
  json::Value SaveAudioTrack(const OutputTarget& target,
                             const std::filesystem::path& media);
  std::filesystem::path WorkFile(std::string_view extension);
  bool RunFfmpeg(const std::vector<std::filesystem::path>& args,
                 std::string* error);
  FetchResult Call(const std::shared_ptr<Job>& job, std::string method,
                   std::string path, std::string body,
                   std::string content_type);

  void Emit(const std::string& client, std::string_view type, json::Value data);
  void EmitStatusLocked();
  json::Value QueueItemLocked(const Job& job) const;
  int QueueRemainingLocked() const;
  void PersistLocked();
  void LoadHistory();

  AdapterOptions options_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Job>> pending_;
  std::shared_ptr<Job> running_;
  std::map<std::string, json::Value> history_;
  std::vector<std::string> history_order_;
  long long next_number_{0};
  bool stopping_{false};
  std::thread worker_;

  struct Event {
    std::uint64_t seq;
    std::string client;
    std::string text;
  };
  std::deque<Event> events_;
  std::uint64_t next_event_{1};
  std::uint64_t work_counter_{0};
};

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_
