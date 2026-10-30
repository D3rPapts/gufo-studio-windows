#include "src/cli/serve/router/router.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/cli/serve/router/registry.hpp"
#include "src/cli/serve/router/residency.hpp"
#include "src/cli/serve/router/workers.hpp"
#include "src/cli/serve/router/ws_relay.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/serve/serve_options.hpp"
#include "src/cli/serve/video_api.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/json.hpp"

namespace gufo::cli {
namespace {

namespace json = gufo::json;
namespace router = gufo::router;
using server::HttpRequest;
using server::HttpResponse;
using server::HttpServer;
using server::HttpServerOptions;
using server::Logger;

constexpr std::string_view kPresetGrammar =
    "Preset file format:\n"
    "  Plain UTF-8 text. Lines starting with '#' are comments; blank lines\n"
    "  are ignored. Each model is one section:\n"
    "\n"
    "    [llm/<model-id>]      modality is one of llm|image|video|tts|asr;\n"
    "                          <model-id> is the routing key served under\n"
    "                          /v1/models and must be unique per file.\n"
    "    key = value           maps to the worker argv '--key value'. Keys\n"
    "                          are exactly the long options of\n"
    "                          'gufo serve <modality>' shown below; boolean\n"
    "                          options take true|false.\n"
    "    host, port, api-key   are router-managed and rejected in "
    "sections.\n\n";

constexpr std::chrono::milliseconds kAdmissionPoll{50};
constexpr std::chrono::milliseconds kSupervisionTick{250};
constexpr int kSpawnAttempts = 3;
constexpr std::chrono::seconds kShutdownDrain{10};

HttpResponse Err(int status, std::string_view reason, std::string_view message,
                 std::string_view type, std::string_view code) {
  json::Value detail = json::Value::object();
  detail["message"] = std::string(message);
  detail["type"] = std::string(type);
  detail["code"] = std::string(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return HttpResponse{
      .status = status, .reason = std::string(reason), .body = body.dump()};
}

/// Static file under --pages-dir: GET /<page>/ serves <dir>/<page>/index.html,
/// GET /<page>/<file> the file itself, and /<page> redirects to /<page>/.
/// Same-origin pages need no CORS or private-network preflight from browsers.
std::optional<HttpResponse> ServePage(const std::filesystem::path& dir,
                                      const HttpRequest& req) {
  if (dir.empty() || req.method != "GET" || req.path.size() < 2 ||
      req.path.starts_with("/v1/"))
    return std::nullopt;
  std::string_view rel = std::string_view(req.path).substr(1);
  if (const auto query = rel.find('?'); query != std::string_view::npos)
    rel = rel.substr(0, query);
  for (const char c : rel)
    if (c == '\\' || c == ':' || c == '\0')
      return std::nullopt;
  std::filesystem::path file = dir;
  std::size_t segments = 0;
  for (std::size_t start = 0; start <= rel.size();) {
    const auto end = std::min(rel.find('/', start), rel.size());
    const auto part = rel.substr(start, end - start);
    if (part == "." || part == "..")
      return std::nullopt;
    if (!part.empty()) {
      file /= std::string(part);
      ++segments;
    }
    start = end + 1;
  }
  std::error_code ec;
  if (segments == 1 && !rel.ends_with('/') &&
      std::filesystem::is_directory(file, ec))
    return HttpResponse{.status = 301,
                        .reason = "Moved Permanently",
                        .headers = {{"Location", req.path + "/"}}};
  if (std::filesystem::is_directory(file, ec))
    file /= "index.html";
  if (!std::filesystem::is_regular_file(file, ec))
    return std::nullopt;
  std::ifstream in(file, std::ios::binary);
  std::string body((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  const auto ext = file.extension().string();
  const std::string type = ext == ".html"  ? "text/html; charset=utf-8"
                           : ext == ".js"  ? "text/javascript; charset=utf-8"
                           : ext == ".css" ? "text/css; charset=utf-8"
                           : ext == ".png" ? "image/png"
                           : ext == ".svg" ? "image/svg+xml"
                           : ext == ".jpg" ? "image/jpeg"
                           : ext == ".mp4" ? "video/mp4"
                           : ext == ".wav" ? "audio/wav"
                           : ext == ".txt" ? "text/plain; charset=utf-8"
                           : ext == ".json" ? "application/json"
                                            : "application/octet-stream";
  return HttpResponse{.status = 200,
                      .reason = "OK",
                      .body = std::move(body),
                      .headers = {{"Content-Type", type},
                                  {"Cache-Control", "no-cache"}}};
}

/// Router-managed paths answered by the front itself. Returns nullopt when
/// the path belongs to the model-routing table instead.
std::optional<HttpResponse> RouteFront(const router::Registry& registry,
                                       const HttpRequest& req) {
  if (req.method == "GET" && (req.path == "/health" || req.path == "/healthz" ||
                              req.path == "/v1/health")) {
    json::Value body = json::Value::object();
    body["status"] = "ok";
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  }
  // Preset validation finished before the listener started, so answering is
  // itself the readiness condition for the front.
  if (req.method == "GET" && (req.path == "/ready" || req.path == "/readyz" ||
                              req.path == "/v1/ready")) {
    json::Value body = json::Value::object();
    body["status"] = "ready";
    return HttpResponse{.status = 200, .reason = "OK", .body = body.dump()};
  }
  if (req.method == "GET" &&
      (req.path == "/v1/models" || req.path == "/models")) {
    json::Value resp = json::Value::object();
    resp["object"] = "list";
    json::Value data = json::Value::array();
    // llama.cpp's router shape (status.value / status.args), so clients
    // written for it list and pick models unchanged.
    for (const auto& [preset, status] : registry.Statuses()) {
      json::Value model = json::Value::object();
      model["id"] = preset->id;
      model["object"] = "model";
      model["owned_by"] = "gufo";
      model["modality"] = preset->modality;
      model["loaded"] = status == "loaded";
      json::Value args = json::Value::array();
      for (const std::string& arg : preset->argv)
        args.push_back(arg);
      json::Value state = json::Value::object();
      state["value"] = std::string(status);
      state["args"] = std::move(args);
      model["status"] = std::move(state);
      data.push_back(std::move(model));
    }
    resp["data"] = std::move(data);
    return HttpResponse{.status = 200, .reason = "OK", .body = resp.dump()};
  }
  return std::nullopt;
}

/// Drains a (possibly streamed) response into one buffer for the ComfyUI
/// adapter.
router::comfy::FetchResult CollectResponse(HttpResponse response) {
  router::comfy::FetchResult result;
  result.status = response.status;
  for (const auto& [name, value] : response.headers) {
    if (name.size() == 12 && (name[0] == 'C' || name[0] == 'c') &&
        (name == "Content-Type" || name == "content-type"))
      result.content_type = value;
  }
  if (response.streaming_body) {
    response.streaming_body([&result](std::string_view chunk) {
      result.body.append(chunk);
      return true;
    });
  } else {
    result.body = std::move(response.body);
  }
  return result;
}

std::optional<int> ParseEnvPort(const char* value) {
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  const std::string_view text(value);
  int parsed = 0;
  const auto [ptr, ec] =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (ec != std::errc{} || ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return parsed;
}

void PrintRouterHelp(const ArgParser& parser) {
  std::cout << parser.FormatHelp() << "\n" << kPresetGrammar;
  for (const std::string_view modality :
       {"llm", "image", "video", "tts", "asr"}) {
    PrintServeHelp("gufo", modality);
  }
}

std::string MultipartModel(const std::string& body) {
  const std::string needle = "name=\"model\"";
  const auto field = body.find(needle);
  if (field == std::string::npos) {
    return "";
  }
  const auto start = body.find("\r\n\r\n", field);
  if (start == std::string::npos) {
    return "";
  }
  const auto end = body.find("\r\n", start + 4);
  return body.substr(start + 4, end == std::string::npos ? std::string::npos
                                                         : end - (start + 4));
}

bool IsJsonModelPost(const HttpRequest& req) {
  if (req.method != "POST") {
    return false;
  }
  static const std::vector<std::string> kPaths = {
      "/v1/chat/completions", "/v1/completions",        "/v1/responses",
      "/v1/messages",         "/v1/images/generations", "/v1/images/edits",
      "/v1/audio/speech",     "/v1/audio/music"};
  if (std::find(kPaths.begin(), kPaths.end(), req.path) != kPaths.end()) {
    return true;
  }
  return server::IsVideoApiPath(req.path);
}

std::string JsonBodyModel(const HttpRequest& req) {
  try {
    const json::Value parsed = json::parse(req.body);
    if (const auto* value = parsed.find("model");
        value != nullptr && value->is_string()) {
      return value->str();
    }
  } catch (const std::exception&) {
    // A malformed body is a missing model as far as routing is concerned.
  }
  return "";
}

/// Model resolution table from the design spec. `missing` marks a family
/// endpoint whose model source (JSON field, form field, query param) is
/// absent, which is a 400 rather than an unresolvable path.
struct Resolution {
  std::string model_id;
  bool missing{false};
};

bool IsIntrospectionPath(const std::string& path) {
  return path == "/v1/slots" || path == "/slots" || path == "/v1/metrics" ||
         path == "/metrics";
}

/// The WebSocket-routed paths from the spec resolution table; their model
/// comes from the `model` query parameter (ResolveModel).
bool IsWebSocketPath(const HttpRequest& req) {
  return req.method == "GET" &&
         (req.path == "/v1/realtime" || req.path == "/v1/audio/speech/stream");
}

/// Job id from `/v1/videos/<id>` or `/v1/videos/<id>/content`, matching the
/// shape video_api.cpp serves (ParseVideoPath).
std::string VideoJobId(const std::string& path) {
  constexpr std::string_view prefix = "/v1/videos/";
  std::string_view remainder(path);
  if (!remainder.starts_with(prefix)) {
    return "";
  }
  remainder.remove_prefix(prefix.size());
  constexpr std::string_view suffix = "/content";
  if (remainder.ends_with(suffix)) {
    remainder.remove_suffix(suffix.size());
  }
  if (remainder.empty() || remainder.find('/') != std::string_view::npos) {
    return "";
  }
  return std::string(remainder);
}

Resolution ResolveModel(const HttpRequest& req) {
  Resolution resolution;
  if (IsJsonModelPost(req)) {
    // Image edits and video first/last frames may arrive as multipart forms.
    resolution.model_id =
        req.header("content-type").starts_with("multipart/form-data")
            ? MultipartModel(req.body)
            : JsonBodyModel(req);
    resolution.missing = resolution.model_id.empty();
    return resolution;
  }
  if (req.method == "POST" && req.path == "/v1/audio/transcriptions") {
    resolution.model_id = MultipartModel(req.body);
    resolution.missing = resolution.model_id.empty();
    return resolution;
  }
  if (IsWebSocketPath(req)) {
    resolution.model_id = req.query_param("model");
    resolution.missing = resolution.model_id.empty();
    return resolution;
  }
  if (req.method == "GET" && IsIntrospectionPath(req.path)) {
    resolution.model_id = req.query_param("model");
    resolution.missing = resolution.model_id.empty();
    return resolution;
  }
  return resolution;
}

/// Holds the front-request busy count until released exactly once; releasing
/// from the streaming relay keeps the model busy across the whole stream.
/// `Arm` runs with the router mutex already held; `Release` takes it itself
/// because the stream completion (or response destruction) happens on a
/// request thread with the lock free.
class HeldBusy : public std::enable_shared_from_this<HeldBusy> {
public:
  HeldBusy(std::mutex& mutex, router::Registry& registry, std::string model_id)
      : mutex_(mutex), registry_(registry), model_id_(std::move(model_id)) {}
  HeldBusy(const HeldBusy&) = delete;
  HeldBusy& operator=(const HeldBusy&) = delete;
  ~HeldBusy() { Release(); }

  std::shared_ptr<HeldBusy> Arm() {
    registry_.RequestStarted(model_id_);
    armed_ = true;
    return shared_from_this();
  }

  void Release() {
    if (!armed_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    registry_.RequestFinished(model_id_);
  }

private:
  std::mutex& mutex_;
  router::Registry& registry_;
  std::string model_id_;
  bool armed_{false};
};

/// Holds a model's open-socket count for one relayed WebSocket. `Arm` runs
/// with the router mutex held so the worker is eviction- and idle-unload-
/// exempt before the 101 reaches the client; the front releases it when the
/// relay returns, and the destructor covers a response that is dropped
/// without the callback ever being served (failed 101 write). `Release` takes
/// the mutex itself because the relay finishes on a request thread with the
/// state lock free.
class HeldWebSocket : public std::enable_shared_from_this<HeldWebSocket> {
public:
  HeldWebSocket(std::mutex& mutex, router::Registry& registry,
                std::string model_id)
      : mutex_(mutex), registry_(registry), model_id_(std::move(model_id)) {}
  HeldWebSocket(const HeldWebSocket&) = delete;
  HeldWebSocket& operator=(const HeldWebSocket&) = delete;
  ~HeldWebSocket() { Release(); }

  std::shared_ptr<HeldWebSocket> Arm() {
    registry_.WebSocketOpened(model_id_);
    armed_ = true;
    return shared_from_this();
  }

  void Release() {
    if (!armed_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    registry_.WebSocketClosed(model_id_);
  }

private:
  std::mutex& mutex_;
  router::Registry& registry_;
  std::string model_id_;
  bool armed_{false};
};

bool IsVideoCreate(const HttpRequest& req) {
  return req.method == "POST" && req.path == "/v1/videos";
}

/// Job id of the status resource `/v1/videos/<id>`; empty for `/content` or
/// any deeper path.
std::string VideoStatusJobId(const std::string& path) {
  constexpr std::string_view prefix = "/v1/videos/";
  if (!path.starts_with(prefix)) {
    return "";
  }
  const std::string_view id(path.data() + prefix.size(),
                            path.size() - prefix.size());
  if (id.empty() || id.find('/') != std::string_view::npos) {
    return "";
  }
  return std::string(id);
}

/// Sniffs a relayed worker body so the router can keep its video-job
/// bookkeeping aligned with the worker's authoritative response. Bytes
/// accumulate up to the cap; an oversized or malformed body still relays
/// untouched and is simply not recorded.
class VideoBodySniffer {
public:
  enum class Kind { kTrack, kStatus, kDelete };

  VideoBodySniffer(std::mutex& mutex, router::Registry& registry, Kind kind,
                   std::string model_id, std::string job_id)
      : mutex_(mutex),
        registry_(registry),
        kind_(kind),
        model_id_(std::move(model_id)),
        job_id_(std::move(job_id)) {}
  VideoBodySniffer(const VideoBodySniffer&) = delete;
  VideoBodySniffer& operator=(const VideoBodySniffer&) = delete;

  std::string_view Observe(std::string_view chunk) {
    if (!discarded_) {
      if (buffer_.size() + chunk.size() > kBodySniffBytes) {
        discarded_ = true;
        buffer_.clear();
      } else {
        buffer_.append(chunk);
      }
    }
    return chunk;
  }

  /// Applies the bookkeeping once the relay has finished (or failed; a
  /// partial body never parses and records nothing).
  void Commit() {
    if (discarded_) {
      return;
    }
    if (kind_ == Kind::kDelete) {
      const std::lock_guard<std::mutex> lock(mutex_);
      registry_.CompleteVideoJob(job_id_);
      return;
    }
    json::Value parsed;
    try {
      parsed = json::parse(buffer_);
    } catch (const std::exception&) {
      return;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (kind_ == Kind::kTrack) {
      const auto* id = parsed.find("id");
      if (id != nullptr && id->is_string() && !id->str().empty()) {
        registry_.TrackVideoJob(id->str(), model_id_);
      }
      return;
    }
    const auto* status = parsed.find("status");
    if (status == nullptr || !status->is_string()) {
      return;
    }
    const std::string value = status->str();
    if (value == "completed" || value == "failed" || value == "cancelled") {
      registry_.CompleteVideoJob(job_id_);
    }
  }

private:
  static constexpr std::size_t kBodySniffBytes = 1 << 20;

  std::mutex& mutex_;
  router::Registry& registry_;
  Kind kind_;
  std::string model_id_;
  std::string job_id_;
  std::string buffer_;
  bool discarded_{false};
};

/// Routes requests, owns live workers and the shared admission state.
class RouterState {
public:
  RouterState(std::vector<router::PresetModel> presets,
              router::RegistryOptions options)
      : registry(std::move(presets), options), options(options) {}

  router::Registry registry;
  router::RegistryOptions options;
  std::mutex mutex;
  std::map<std::string, std::shared_ptr<router::Worker>> workers;
  std::filesystem::path exe_path;
  std::filesystem::path pages_dir;
  std::atomic<bool> stop_requested{false};
  /// Memory probes, learned peaks and (with --memory-guard) admission by
  /// live free memory instead of a model count.
  std::unique_ptr<router::Residency> residency;
  /// Workers being stopped right now (status "unloading").
  std::set<std::string> stopping;
  /// ComfyUI-compatible front (--comfy-dir); null when disabled.
  std::unique_ptr<router::comfy::Adapter> comfy;

  /// Worker::Spawn must run on the long-lived spawner thread: PR_SET_PDEATHSIG
  /// binds the worker to the thread that forked it, and request-handling
  /// connection threads exit as soon as their client disconnects. The spawner
  /// lives for the whole routing lifetime, so workers die only with the
  /// router.
  void StartSpawner() {
    spawner_ = std::thread([this] { SpawnThreadMain(); });
  }

  void StopSpawner() {
    {
      const std::lock_guard<std::mutex> lock(spawn_mutex_);
      spawn_shutdown_ = true;
    }
    spawn_cv_.notify_all();
    if (spawner_.joinable()) {
      spawner_.join();
    }
  }

  /// Runs one Worker::Spawn on the spawner thread and waits for its result.
  /// The caller holds the state mutex; the spawner never touches it.
  std::unique_ptr<router::Worker> SpawnViaSpawnerLocked(
      const router::WorkerSpec& spec, std::string* error) {
    SpawnJob job;
    job.spec = spec;
    {
      const std::lock_guard<std::mutex> lock(spawn_mutex_);
      pending_ = &job;
    }
    spawn_cv_.notify_all();
    {
      std::unique_lock<std::mutex> done_lock(job.done_mutex);
      job.done_cv.wait(done_lock, [&job] { return job.done; });
    }
    *error = job.error;
    return std::move(job.result);
  }

  HttpResponse Dispatch(const HttpRequest& req) {
    // Every registry touch, including RouteFront's Listing, is serialized on
    // the same mutex the supervision tick holds.
    if (req.method == "GET" && (req.path == "/router/status" ||
                                req.path.starts_with("/router/status?")))
      return RouterStatus();
    if (comfy != nullptr) {
      if (auto answer = comfy->Handle(req))
        return std::move(*answer);
    }
    if (auto page = ServePage(pages_dir, req))
      return std::move(*page);
    std::unique_lock<std::mutex> lock(mutex);
    if (const auto front = RouteFront(registry, req)) {
      return *front;
    }
    if (req.path.starts_with("/v1/videos/") && req.method != "GET" &&
        req.method != "DELETE") {
      // Job subpaths have no verbs beyond the status/content reads the
      // front's video API answers: everything else is 405 regardless of job
      // state or body, mirroring HandleVideoApiRequest's precedence.
      return Err(405, "Method Not Allowed", "method is not allowed",
                 "invalid_request_error", "method_not_allowed");
    }
    if (req.path.starts_with("/v1/previews/"))
      return PreviewLocked(req, lock);
    const Resolution resolution = ResolveModel(req);
    if (resolution.missing) {
      return Err(400, "Bad Request", "missing 'model' field",
                 "invalid_request_error", "missing_model");
    }
    Resolution resolved = resolution;
    if (resolved.model_id.empty() && server::IsVideoApiPath(req.path) &&
        (req.method == "GET" || req.method == "DELETE")) {
      const std::string job_id = VideoJobId(req.path);
      if (req.method == "DELETE" && req.path.ends_with("/content") &&
          !job_id.empty()) {
        // The front only serves content reads with GET; shape and method
        // decide before any job lookup.
        return Err(405, "Method Not Allowed", "method is not allowed",
                   "invalid_request_error", "method_not_allowed");
      }
      // Status/content reads for a recorded job route to its owner model and
      // reload that worker if it died (persisted job state answers again).
      resolved.model_id = registry.JobOwner(job_id);
    }
    if (resolved.model_id.empty()) {
      // Nothing resolved the path to a model. Video paths mirror the front's
      // video API precedence: shape and method first (405 above), then the
      // unrecorded or malformed job id (404 video_not_found without leaking
      // preset ids, 405 for a collection read). A genuinely unknown model id
      // on a non-video route keeps the model_not_found listing and an
      // unknown route stays 404 not_found.
      if (server::IsVideoApiPath(req.path) &&
          (req.method == "GET" || req.method == "DELETE")) {
        if (req.path == "/v1/videos") {
          return Err(405, "Method Not Allowed", "method is not allowed",
                     "invalid_request_error", "method_not_allowed");
        }
        return Err(404, "Not Found", "video job not found",
                   "invalid_request_error", "video_not_found");
      }
      return Err(404, "Not Found", "no route for this path",
                 "invalid_request_error", "not_found");
    }
    const std::string& model_id = resolved.model_id;
    const router::PresetModel* preset = registry.Find(model_id);
    if (preset == nullptr) {
      return ModelNotFoundError();
    }
    const bool ws_relay =
        IsWebSocketPath(req) && server::IsWebSocketUpgrade(req);
    if (ws_relay) {
      // Validate the upgrade handshake before admission so a malformed
      // WebSocket request never spawns or wakes a worker. The identical
      // UpgradeWebSocket call below is guaranteed to yield the 101 head.
      const HttpResponse handshake = server::UpgradeWebSocket(req, nullptr);
      if (handshake.status != 101) {
        return handshake;
      }
    }
    if (IsIntrospectionPath(req.path) && req.method == "GET" &&
        !LoadedLocked(model_id)) {
      return Err(409, "Conflict", "model '" + model_id + "' is not loaded",
                 "invalid_request_error", "model_not_loaded");
    }
    // Shutdown order per spec: the drain refuses NEW loads but keeps proxying
    // already-loaded models until the final worker stop.
    if (stop_requested.load() && !LoadedLocked(model_id)) {
      return Err(502, "Bad Gateway", "router is shutting down", "server_error",
                 "upstream_unavailable");
    }
    auto admitted = AdmitLocked(model_id, *preset);
    if (const auto* failed = std::get_if<HttpResponse>(&admitted)) {
      return std::move(*failed);
    }
    const int worker_port = std::get<int>(admitted);
    router::UpstreamTarget target;
    target.host = "127.0.0.1";
    target.port = worker_port;
    if (ws_relay) {
      // Same lock discipline as the HTTP relay below: the socket count is
      // armed here (worker stays resident and eviction-exempt), while
      // RelayWebSocket itself runs from the 101 callback after this return
      // has released the state mutex, and releases it when either side
      // closes.
      auto sockets = std::make_shared<HeldWebSocket>(mutex, registry, model_id);
      const std::string ws_model = model_id;
      HttpResponse upgrade = server::UpgradeWebSocket(
          req, [target, ws_model, sockets](server::WebSocket& front) {
            std::string error;
            if (!router::RelayWebSocket(target, front, &error)) {
              Logger::Warn("router", "event=ws_relay_end model=" + ws_model +
                                         " error=" + error);
            }
            sockets->Release();
          });
      // Arm as the last fallible step: anything throwing above unwinds an
      // unarmed guard (Release is a no-op), so no throw path can destroy an
      // armed guard and self-lock the still-held state mutex.
      sockets->Arm();
      return upgrade;
    }
    auto busy = std::make_shared<HeldBusy>(mutex, registry, model_id)->Arm();
    // The relay blocks while connecting and reading upstream headers, so it
    // runs with the lock free; the busy count keeps the worker resident.
    lock.unlock();
    HttpResponse response = router::ProxyToWorker(target, req);
    std::shared_ptr<VideoBodySniffer> sniffer;
    if (response.status >= 200 && response.status < 300 &&
        response.streaming_body) {
      sniffer = MakeVideoSniffer(req, model_id);
    }
    {
      const std::lock_guard<std::mutex> relock(mutex);
      if (response.status == 502 && !response.streaming_body) {
        HandleUpstreamFailureLocked(model_id);
      }
    }
    if (response.streaming_body) {
      // The busy count lives until the streamed body completes or cancels.
      HttpResponse streamed = std::move(response);
      streamed.streaming_body = [inner = std::move(streamed.streaming_body),
                                 busy, sniffer](
                                    const HttpResponse::BodyWriter& writer) {
        const HttpResponse::BodyWriter sink =
            sniffer == nullptr ? writer
                               : [&writer, sniffer](std::string_view chunk) {
                                   return writer(sniffer->Observe(chunk));
                                 };
        try {
          inner(sink);
        } catch (...) {
          busy->Release();
          throw;
        }
        busy->Release();
        if (sniffer != nullptr) {
          sniffer->Commit();
        }
      };
      return streamed;
    }
    busy->Release();
    return response;
  }

  /// Live previews (/v1/previews/{id}[/events]) of a running image or video
  /// job: the model comes from `?model=` or, for a video job, its owner.
  /// Only a loaded worker is asked; a preview never starts or keeps one.
  HttpResponse PreviewLocked(const HttpRequest& req,
                             std::unique_lock<std::mutex>& lock) {
    std::string model_id = req.query_param("model");
    if (model_id.empty()) {
      std::string_view id(req.path);
      id.remove_prefix(std::string_view("/v1/previews/").size());
      if (id.ends_with("/events"))
        id.remove_suffix(std::string_view("/events").size());
      model_id = registry.JobOwner(id);
    }
    const auto worker = model_id.empty() || !LoadedLocked(model_id)
                            ? nullptr
                            : FindWorkerLocked(model_id);
    if (worker == nullptr)
      return Err(404, "Not Found",
                 "no loaded image or video model runs this preview "
                 "(pass ?model= for images)",
                 "invalid_request_error", "preview_not_found");
    router::UpstreamTarget target;
    target.host = "127.0.0.1";
    target.port = worker->port();
    lock.unlock();
    return router::ProxyToWorker(target, req);
  }

  /// Stops every loaded worker that serves no request (ComfyUI /free).
  void UnloadIdle() {
    std::unique_lock<std::mutex> lock(mutex);
    // Loaded, not loading, not busy (requests, sockets, open video jobs).
    std::vector<std::string> idle;
    for (const auto& entry : registry.Residents()) {
      if (!entry.loading && !entry.busy)
        idle.push_back(entry.id);
    }
    for (const auto& id : idle) {
      // Each stop releases the lock: skip a model that got busy meanwhile.
      if (registry.Busy(id) || FindWorkerLocked(id) == nullptr)
        continue;
      Logger::Info("router", "event=free_unload model=" + id);
      if (residency != nullptr)
        residency->Event("unloaded " + id + " (/free)");
      // StopLocked records the release, so the next admission waits until
      // the memory is back.
      StopLocked(id);
    }
  }

  /// One supervision tick: reap dead workers, unload idle ones, free a slot
  /// for queued loads, and kill wedged loads past their deadline.
  void SuperviseOnce() {
    std::unique_lock<std::mutex> lock(mutex);
    for (auto it = workers.begin(); it != workers.end();) {
      int exit_status = 0;
      if (it->second->PollExit(&exit_status)) {
        Logger::Warn("router", "event=worker_exit model=" + it->first +
                                   " status=" + std::to_string(exit_status));
        registry.MarkUnloaded(it->first);
        it = workers.erase(it);
      } else {
        ++it;
      }
    }
    for (const auto& id : registry.DueIdleUnloads()) {
      // An earlier stop in this loop released the lock: a request may have
      // made this model busy meanwhile.
      if (registry.Busy(id)) {
        continue;
      }
      Logger::Info("router", "event=idle_unload model=" + id);
      if (residency != nullptr) {
        residency->Event("idle unload " + id);
      }
      StopLocked(id);
    }
    if (registry.HasQueued()) {
      const std::string candidate = registry.EvictionCandidate();
      if (!candidate.empty()) {
        Logger::Info("router", "event=evict model=" + candidate);
        if (residency != nullptr) {
          residency->Event("evicted " + candidate + " (--models-max)");
        }
        StopLocked(candidate);
      }
    }
    SuperviseMemoryLocked(lock);
    for (const auto& id : registry.DueLoadTimeouts()) {
      Logger::Warn("router", "event=load_timeout_expired model=" + id);
      StopLocked(id);
    }
  }

  /// GET /router/status: live memory, resident workers with their usage and
  /// idle timers, and the last residency decisions.
  HttpResponse RouterStatus() {
    std::vector<router::Registry::ResidentEntry> residents;
    std::set<std::string> unloading;
    router::Registry::Clock::time_point now;
    {
      const std::lock_guard<std::mutex> lock(mutex);
      residents = registry.Residents();
      unloading = stopping;
      now = registry.NowForStatus();
    }
    json::Value body = residency != nullptr
                           ? residency->Status(residents, unloading, now)
                           : json::Value::object();
    body["models_max"] = static_cast<long long>(options.max_active);
    return HttpResponse{.status = 200,
                        .reason = "OK",
                        .body = body.dump(),
                        .headers = {{"Content-Type", "application/json"},
                                    {"Cache-Control", "no-cache"}}};
  }

private:
  using AdmissionResult = std::variant<int, HttpResponse>;

  /// Worker pids for the memory probes (state mutex held).
  router::Residency::WorkerPids PidsLocked() {
    router::Residency::WorkerPids pids;
    for (const auto& [id, worker] : workers) {
      pids.emplace_back(id, worker->pid());
    }
    return pids;
  }

  /// Samples memory every second (every tick, 250 ms, while a worker loads)
  /// and, with the guard on, runs the watchdog: memory critical during a load
  /// kills the newest load; otherwise the least recently used idle worker
  /// goes. Busy workers are never touched.
  void SuperviseMemoryLocked(std::unique_lock<std::mutex>& lock) {
    if (residency == nullptr) {
      return;
    }
    const auto residents = registry.Residents();
    const bool loading =
        std::any_of(residents.begin(), residents.end(),
                    [](const auto& entry) { return entry.loading; });
    if (++sample_ticks_ < (loading ? 1 : 4)) {
      return;
    }
    sample_ticks_ = 0;
    const auto pids = PidsLocked();
    lock.unlock();
    residency->Sample(pids);
    lock.lock();
    if (!residency->enabled()) {
      return;
    }
    const router::WatchdogAction action =
        residency->Watchdog(registry.Residents());
    if (!action.abort_load.empty()) {
      Logger::Warn("router", "event=memory_abort model=" + action.abort_load +
                                 " reason=\"" + action.reason + "\"");
      residency->OnAborted(action.abort_load);
      residency->Event("aborted load of " + action.abort_load + ": " +
                       action.reason);
      StopLocked(action.abort_load);
    } else if (!action.evict.empty()) {
      Logger::Warn("router", "event=memory_evict model=" + action.evict +
                                 " reason=\"" + action.reason + "\"");
      residency->Event("evicted " + action.evict + ": " + action.reason);
      StopLocked(action.evict);
    } else if (!action.reason.empty() &&
               std::chrono::steady_clock::now() - last_critical_log_ >
                   std::chrono::seconds(10)) {
      last_critical_log_ = std::chrono::steady_clock::now();
      Logger::Warn("router", "event=memory_critical reason=\"" + action.reason +
                                 "\" action=none");
      residency->Event(action.reason + " (nothing idle to stop)");
    }
  }

  enum class Gate { kSpawn, kWait, kEvicted };

  /// Memory admission for a cold start of `model_id` (state mutex held; it
  /// is released while evicted workers stop and their memory returns).
  Gate MemoryGateLocked(const std::string& model_id, std::string* last_wait) {
    const auto residents = registry.Residents();
    const router::AdmissionPlan plan =
        residency->PlanFor(model_id, residents, PidsLocked());
    if (plan.action == router::AdmissionPlan::Action::kWait) {
      // One event per kind of wait, not one per changed number.
      const std::string kind = plan.reason.substr(0, plan.reason.find(';'));
      if (kind != *last_wait) {
        *last_wait = kind;
        residency->Event("waiting to load " + model_id + ": " + plan.reason);
      }
      return Gate::kWait;
    }
    if (plan.action == router::AdmissionPlan::Action::kSpawn) {
      std::string beside;
      for (const auto& entry : residents) {
        if (entry.id != model_id) {
          beside += (beside.empty() ? " beside " : ", ") + entry.id;
        }
      }
      residency->Event("loading " + model_id + beside + " (" + plan.reason +
                       ")");
      return Gate::kSpawn;
    }
    for (const auto& victim : plan.evict) {
      // The plan was made under the lock, but each stop releases it: a
      // victim that became busy or reloads meanwhile stays; the next plan
      // decides again.
      bool idle_ready = false;
      for (const auto& entry : registry.Residents()) {
        idle_ready =
            idle_ready || (entry.id == victim && !entry.busy && !entry.loading);
      }
      if (!idle_ready) {
        continue;
      }
      Logger::Info("router",
                   "event=memory_evict model=" + victim + " for=" + model_id);
      residency->Event("evicted " + victim + " to fit " + model_id + " (" +
                       plan.reason + ")");
      StopLocked(victim);
    }
    return Gate::kEvicted;
  }

  int sample_ticks_{0};
  std::chrono::steady_clock::time_point last_critical_log_{};

  struct SpawnJob {
    router::WorkerSpec spec;
    std::unique_ptr<router::Worker> result;
    std::string error;
    bool done{false};
    std::mutex done_mutex;
    std::condition_variable done_cv;
  };

  void SpawnThreadMain() {
    for (;;) {
      SpawnJob* job = nullptr;
      {
        std::unique_lock<std::mutex> lock(spawn_mutex_);
        spawn_cv_.wait(
            lock, [this] { return spawn_shutdown_ || pending_ != nullptr; });
        if (spawn_shutdown_) {
          return;
        }
        job = pending_;
        pending_ = nullptr;
      }
      job->result = router::Worker::Spawn(job->spec, exe_path, &job->error);
      {
        const std::lock_guard<std::mutex> done_lock(job->done_mutex);
        job->done = true;
      }
      job->done_cv.notify_all();
    }
  }

  std::thread spawner_;
  std::mutex spawn_mutex_;
  std::condition_variable spawn_cv_;
  SpawnJob* pending_{nullptr};
  bool spawn_shutdown_{false};

  bool LoadedLocked(const std::string& model_id) {
    for (const auto& [id, loaded] : registry.Listing()) {
      if (id == model_id) {
        return loaded;
      }
    }
    return false;
  }

  HttpResponse ModelNotFoundError() {
    std::string ids;
    for (const auto& [id, loaded] : registry.Listing()) {
      (void)loaded;
      if (!ids.empty()) {
        ids += ", ";
      }
      ids += id;
    }
    return Err(404, "Not Found", "unknown model; preset ids: " + ids,
               "invalid_request_error", "model_not_found");
  }

  std::shared_ptr<router::Worker> FindWorkerLocked(const std::string& id) {
    const auto it = workers.find(id);
    return it == workers.end() ? nullptr : it->second;
  }

  /// Stops and unloads `id`. The mutex is released around Stop(); the local
  /// shared_ptr keeps the worker object alive until it is reaped.
  void StopLocked(const std::string& id) {
    auto worker = FindWorkerLocked(id);
    bool was_ready = false;
    for (const auto& entry : registry.Residents()) {
      was_ready = was_ready || (entry.id == id && !entry.loading);
    }
    registry.MarkUnloaded(id);
    workers.erase(id);
    if (worker == nullptr) {
      return;
    }
    stopping.insert(id);
    if (residency != nullptr) {
      residency->NoteStopping(id);
    }
    mutex.unlock();
    try {
      worker->Stop();
    } catch (...) {
      mutex.lock();
      stopping.erase(id);
      throw;
    }
    mutex.lock();
    stopping.erase(id);
    if (residency != nullptr) {
      residency->OnUnloaded(id, was_ready);
    }
  }

  /// A 502 relayed before any upstream response byte means the worker failed
  /// for this request. Reap it when it exited; a live-but-refusing worker is
  /// killed and unloaded so the next request spawns a fresh backend instead
  /// of retrying the same dead one until the idle timeout.
  void HandleUpstreamFailureLocked(const std::string& id) {
    auto worker = FindWorkerLocked(id);
    if (worker == nullptr) {
      return;
    }
    int exit_status = 0;
    if (worker->PollExit(&exit_status)) {
      Logger::Warn("router", "event=worker_exit model=" + id +
                                 " status=" + std::to_string(exit_status));
      registry.MarkUnloaded(id);
      workers.erase(id);
      return;
    }
    Logger::Warn("router",
                 "event=worker_unresponsive model=" + id + " action=restart");
    StopLocked(id);
  }

  /// Video-job bookkeeping for proxied 2xx responses: the create response
  /// records the worker-assigned id against the owning model; a terminal
  /// status read or a DELETE releases it. Returns nullptr for non-video
  /// requests.
  std::shared_ptr<VideoBodySniffer> MakeVideoSniffer(
      const HttpRequest& req, const std::string& model_id) {
    if (IsVideoCreate(req)) {
      return std::make_shared<VideoBodySniffer>(
          mutex, registry, VideoBodySniffer::Kind::kTrack, model_id, "");
    }
    if (req.method != "GET" && req.method != "DELETE") {
      return nullptr;
    }
    const std::string job_id = VideoStatusJobId(req.path);
    if (job_id.empty()) {
      return nullptr;
    }
    return std::make_shared<VideoBodySniffer>(
        mutex, registry,
        req.method == "DELETE" ? VideoBodySniffer::Kind::kDelete
                               : VideoBodySniffer::Kind::kStatus,
        model_id, job_id);
  }

  /// Arms a fresh spawn with a freshly reserved port. Returns the fatal 502
  /// (port exhaustion / spawn failure) or nullopt once the worker is spawned.
  std::optional<HttpResponse> TrySpawnLocked(const std::string& model_id,
                                             const router::PresetModel& preset,
                                             int attempt) {
    std::string spawn_error;
    const int worker_port = router::ReserveLoopbackPort(&spawn_error);
    if (worker_port < 0) {
      registry.MarkUnloaded(model_id);
      return Err(502, "Bad Gateway", "no loopback port: " + spawn_error,
                 "server_error", "upstream_unavailable");
    }
    registry.MarkLoading(model_id);
    router::WorkerSpec spec;
    spec.model_id = model_id;
    spec.modality = preset.modality;
    spec.args = preset.argv;
    spec.port = worker_port;
    spec.command = preset.command;
    auto worker = SpawnViaSpawnerLocked(spec, &spawn_error);
    if (worker == nullptr) {
      Logger::Warn("router", "event=spawn_failed model=" + model_id +
                                 " attempt=" + std::to_string(attempt) +
                                 " error=" + spawn_error);
      registry.MarkUnloaded(model_id);
      return Err(
          502, "Bad Gateway",
          "failed to spawn worker for '" + model_id + "': " + spawn_error,
          "server_error", "upstream_unavailable");
    }
    Logger::Info("router", "event=spawn model=" + model_id +
                               " port=" + std::to_string(worker_port) +
                               " attempt=" + std::to_string(attempt));
    workers[model_id] = std::move(worker);
    if (residency != nullptr) {
      residency->OnSpawn(model_id);
    }
    return std::nullopt;
  }

  /// Polls `registry.RequestLoad` until the model is ready, following the
  /// spawn/await/queue decisions. State transitions happen in short critical
  /// sections; the mutex stays free while waiting for worker readiness so
  /// supervision ticks and concurrent waiters are never starved.
  AdmissionResult AdmitLocked(const std::string& model_id,
                              const router::PresetModel& preset) {
    int attempt = 0;
    // The whole held request (queue + spawn + readiness) is bound by
    // load_timeout even if supervision or a spawn retry clears the registry
    // deadline state mid-flight. MarkUnloaded disarms the registry clock, so
    // the front tracks the outer bound itself.
    const auto began = std::chrono::steady_clock::now();
    std::string last_wait;
    for (;;) {
      if (options.load_timeout > std::chrono::seconds(0) &&
          std::chrono::steady_clock::now() - began >= options.load_timeout) {
        return LoadTimeoutLocked(model_id);
      }
      std::shared_ptr<router::Worker> worker;
      int port = 0;
      {
        const router::LoadDecision decision = registry.RequestLoad(model_id);
        if (decision == router::LoadDecision::kLoadTimeout) {
          return LoadTimeoutLocked(model_id);
        }
        if (decision == router::LoadDecision::kSpawn && residency != nullptr &&
            residency->enabled()) {
          if (residency->AbortedSince(model_id, began)) {
            registry.MarkUnloaded(model_id);
            return Err(
                507, "Insufficient Storage",
                "loading '" + model_id + "' was stopped because memory ran low",
                "server_error", "insufficient_memory");
          }
          if (stop_requested.load()) {
            return Err(502, "Bad Gateway", "router is shutting down",
                       "server_error", "upstream_unavailable");
          }
          const Gate gate = MemoryGateLocked(model_id, &last_wait);
          if (gate == Gate::kWait) {
            mutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            mutex.lock();
            continue;
          }
          if (gate == Gate::kEvicted) {
            continue;
          }
        }
        if (decision == router::LoadDecision::kSpawn) {
          ++attempt;
          if (attempt > kSpawnAttempts) {
            registry.MarkUnloaded(model_id);
            return Err(502, "Bad Gateway",
                       "worker for '" + model_id + "' failed to start",
                       "server_error", "upstream_unavailable");
          }
          if (const auto failure = TrySpawnLocked(model_id, preset, attempt)) {
            return *failure;
          }
        }
        worker = FindWorkerLocked(model_id);
        if (worker == nullptr) {
          // A load is supposedly underway but its worker vanished; the next
          // RequestLoad re-decides.
          port = -1;
        } else {
          port = worker->port();
        }
      }
      if (port < 0) {
        SleepUnlocked();
        continue;
      }
      // One admission iteration against `worker` with the lock released
      // around both the health probe and the poll sleep. The mutex is held
      // at each loop entry and on every exit path.
      bool redecide = false;
      while (!redecide) {
        int exit_status = 0;
        const bool exited = worker->PollExit(&exit_status);
        mutex.unlock();
        bool healthy = false;
        try {
          healthy = worker->Healthy();
        } catch (...) {
          mutex.lock();
          throw;
        }
        mutex.lock();
        if (exited) {
          if (FindWorkerLocked(model_id) == worker) {
            Logger::Warn("router",
                         "event=worker_exit model=" + model_id + " status=" +
                             std::to_string(exit_status) + " phase=load");
            registry.MarkUnloaded(model_id);
            workers.erase(model_id);
          }
          redecide = true;
          continue;
        }
        if (healthy) {
          // The load deadline may have expired (and a supervised stop erased
          // this worker) while the probe ran unlocked. Arming ready without
          // a worker behind it would strand the model loaded until the next
          // request's own timeout, so re-decide instead.
          if (FindWorkerLocked(model_id) != worker) {
            redecide = true;
            continue;
          }
          registry.MarkReady(model_id);
          return port;
        }
        if (stop_requested.load()) {
          return Err(502, "Bad Gateway", "router is shutting down",
                     "server_error", "upstream_unavailable");
        }
        const router::LoadDecision recheck = registry.RequestLoad(model_id);
        if (recheck == router::LoadDecision::kLoadTimeout) {
          return LoadTimeoutLocked(model_id);
        }
        SleepUnlocked();
      }
    }
  }

  /// Sleeps for one admission poll with the state mutex released (the caller
  /// must hold it; it is held again on return).
  void SleepUnlocked() {
    mutex.unlock();
    std::this_thread::sleep_for(kAdmissionPoll);
    mutex.lock();
  }

  /// Turns an expired load deadline into the 504 the held request receives,
  /// killing the wedged worker first.
  HttpResponse LoadTimeoutLocked(const std::string& model_id) {
    Logger::Warn("router", "event=load_timeout model=" + model_id);
    auto worker = FindWorkerLocked(model_id);
    registry.MarkUnloaded(model_id);
    workers.erase(model_id);
    if (worker != nullptr) {
      mutex.unlock();
      try {
        worker->Stop();
      } catch (...) {
        mutex.lock();
        throw;
      }
      mutex.lock();
    }
    return Err(504, "Gateway Timeout",
               "model '" + model_id + "' did not load before the timeout",
               "server_error", "load_timeout");
  }
};

}  // namespace

int RunRouter(std::span<const char* const> args) {
  (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)std::setvbuf(stderr, nullptr, _IONBF, 0);

  std::string host = "127.0.0.1";
  if (const char* env_host = std::getenv("HOST");
      env_host != nullptr && *env_host != '\0') {
    host = env_host;
  } else if (const char* env_host = std::getenv("GUFO_HOST");
             env_host != nullptr && *env_host != '\0') {
    host = env_host;
  }
  int port = 8080;
  if (const auto env_port = ParseEnvPort(std::getenv("PORT"));
      env_port.has_value()) {
    port = *env_port;
  } else if (const auto env_port = ParseEnvPort(std::getenv("GUFO_PORT"));
             env_port.has_value()) {
    port = *env_port;
  }

  std::filesystem::path preset_path;
  std::filesystem::path models_dir;
  std::filesystem::path pages_dir;
  std::filesystem::path comfy_dir;
  std::filesystem::path comfy_ffmpeg;
  std::size_t models_max = 2;
  std::size_t sleep_idle_seconds = 900;
  std::size_t load_timeout_seconds = 600;
  bool autoload = false;
  bool memory_guard = false;
  bool memory_report = false;
  std::filesystem::path memory_file;
  router::MemoryPolicy memory_policy;
  double memory_factor = memory_policy.factor;
  double memory_learned_factor = memory_policy.learned_factor;
  double reserve_gpu_gb = router::ToGb(memory_policy.reserve_gpu);
  double reserve_host_gb = router::ToGb(memory_policy.reserve_host);
  double abort_host_gb = router::ToGb(memory_policy.abort_host);
  double abort_gpu_gb = router::ToGb(memory_policy.abort_gpu);
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  ServerLogOptions log_options;

  ArgParser parser("gufo router",
                   "Serve multiple preset models with on-demand loading");
  parser.AddOption("", "--models-preset", "FILE",
                   "Preset file listing the models to multiplex (required)",
                   "Router", &preset_path);
  parser.AddOption("", "--models-dir", "DIR",
                   "Base directory that resolves relative model paths",
                   "Router", &models_dir);
  parser.AddOption("", "--pages-dir", "DIR",
                   "Serve static pages: GET /<name>/ returns "
                   "DIR/<name>/index.html",
                   "Router", &pages_dir);
  parser.AddOption("", "--comfy-dir", "DIR",
                   "Serve a ComfyUI-compatible API (/prompt, /history, "
                   "/view, /upload/image, /ws); DIR holds input/, output/, "
                   "temp/ and history.json",
                   "Router", &comfy_dir);
  parser.AddOption("", "--comfy-ffmpeg", "FILE",
                   "ffmpeg for the ComfyUI API: FLAC audio, PNG frames from "
                   "videos, non-WAV voice references",
                   "Router", &comfy_ffmpeg);
  parser.AddOption("", "--models-max", "N",
                   "Maximum concurrently loaded workers (default: 2; "
                   "0 = unlimited)",
                   "Router", &models_max);
  parser.AddOption("", "--sleep-idle-seconds", "SEC",
                   "Unload a worker idle for this long (default: 900; "
                   "0 disables)",
                   "Router", &sleep_idle_seconds);
  parser.AddOption("", "--load-timeout-seconds", "SEC",
                   "Bound for one full cold load (default: 600; "
                   "0 = unlimited)",
                   "Router", &load_timeout_seconds);
  parser.AddFlag("", "--autoload",
                 "Preload preset models at startup in section order "
                 "(with --memory-guard only the first one)",
                 "Router", &autoload);
  parser.AddFlag("", "--memory-guard",
                 "Admit cold loads by live free GPU/host memory: keep loaded "
                 "models beside a new one while it fits, evict idle ones "
                 "least recently used first, wait for busy ones, and stop a "
                 "load when memory runs critically low. Combine with "
                 "--models-max 0. Preset keys router-gpu-gb, router-host-gb, "
                 "router-exclusive and router-idle-seconds tune single models",
                 "Memory", &memory_guard);
  parser.AddFlag("", "--memory-report",
                 "Print one memory probe sample as JSON and exit", "Memory",
                 &memory_report);
  parser.AddOption("", "--memory-file", "FILE",
                   "Learned per-model memory peaks (JSON; default: the preset "
                   "path with .memory.json)",
                   "Memory", &memory_file);
  parser.AddOption("", "--memory-factor", "X",
                   "Safety factor on estimates and learned peaks (default: "
                   "1.15)",
                   "Memory", &memory_factor);
  parser.AddOption("", "--memory-learned-factor", "X",
                   "Safety factor on measured (learned) peaks (default: "
                   "1.05)",
                   "Memory", &memory_learned_factor);
  parser.AddOption("", "--memory-reserve-gpu-gb", "GB",
                   "GPU memory always kept free (default: 3)", "Memory",
                   &reserve_gpu_gb);
  parser.AddOption("", "--memory-reserve-host-gb", "GB",
                   "Host RAM always kept free (default: 4)", "Memory",
                   &reserve_host_gb);
  parser.AddOption("", "--memory-abort-host-gb", "GB",
                   "Watchdog: stop the newest load below this free host RAM "
                   "(default: 1.5)",
                   "Memory", &abort_host_gb);
  parser.AddOption("", "--memory-abort-gpu-gb", "GB",
                   "Watchdog: stop the newest load below this free GPU memory "
                   "(dedicated + shared; default: 0.5)",
                   "Memory", &abort_gpu_gb);
  AddServerOptions(parser, &host, &port, nullptr, &max_connections,
                   &max_request_body_bytes, &api_key, &log_options);

  // A bare `help` argument (e.g. `gufo router help`) prints the router help,
  // matching sibling commands; ArgParser only understands --help/-h.
  if (args.size() == 1 && args[0] == std::string_view("help")) {
    PrintRouterHelp(parser);
    return 0;
  }

  std::string parse_error;
  if (!parser.Parse(args, &parse_error)) {
    std::cerr << "Error: " << parse_error << "\n";
    parser.PrintHelp(std::cerr);
    return 2;
  }
  if (parser.IsHelpRequested()) {
    PrintRouterHelp(parser);
    return 0;
  }
  if (log_options.verbose && log_options.level.has_value()) {
    std::cerr << "Error: -v cannot combine with --log-level\n";
    return 2;
  }
  if (log_options.verbose) {
    Logger::SetLevel(server::LogLevel::kDebug);
  } else if (log_options.level.has_value()) {
    Logger::SetLevel(*log_options.level);
  }

  if (memory_report) {
    // One probe sample as JSON, for checking the counters on a machine.
    const auto probe = router::MakeSystemMemoryProbe();
    const router::SystemMemory memory = probe->System();
    json::Value report = json::Value::object();
    report["gpu_valid"] = memory.gpu_valid;
    report["gpu_dedicated_total_gb"] = router::ToGb(memory.gpu_dedicated_total);
    report["gpu_dedicated_used_gb"] = router::ToGb(memory.gpu_dedicated_used);
    report["gpu_shared_total_gb"] = router::ToGb(memory.gpu_shared_total);
    report["gpu_shared_used_gb"] = router::ToGb(memory.gpu_shared_used);
    report["host_total_gb"] = router::ToGb(memory.host_total);
    report["host_available_gb"] = router::ToGb(memory.host_available);
    report["probe"] = probe->Describe();
    std::cout << report.dump() << "\n";
    return 0;
  }
  if (preset_path.empty()) {
    std::cerr << "Error: --models-preset <FILE> is required\n";
    return 2;
  }
  std::vector<router::PresetModel> models;
  std::string preset_error;
  if (!router::LoadPreset(preset_path, models_dir, &models, &preset_error)) {
    std::cerr << "Error: " << preset_error << "\n";
    return 2;
  }

  router::RegistryOptions registry_options;
  registry_options.max_active = models_max;
  registry_options.idle_timeout = std::chrono::seconds(sleep_idle_seconds);
  registry_options.load_timeout = std::chrono::seconds(load_timeout_seconds);
  auto state = std::make_shared<RouterState>(models, registry_options);
  state->exe_path = router::ExecutablePath();
  state->pages_dir = pages_dir;
  {
    router::ResidencyOptions residency_options;
    residency_options.enabled = memory_guard;
    residency_options.policy.factor = memory_factor;
    residency_options.policy.learned_factor = memory_learned_factor;
    residency_options.policy.reserve_gpu = router::FromGb(reserve_gpu_gb);
    residency_options.policy.reserve_host = router::FromGb(reserve_host_gb);
    residency_options.policy.abort_host = router::FromGb(abort_host_gb);
    residency_options.policy.abort_gpu = router::FromGb(abort_gpu_gb);
    residency_options.idle_default = registry_options.idle_timeout;
    if (memory_file.empty()) {
      memory_file = preset_path;
      memory_file.replace_extension(".memory.json");
    }
    residency_options.memory_file = memory_file;
    state->residency = std::make_unique<router::Residency>(
        residency_options, router::MakeSystemMemoryProbe(), models);
    std::string memory_error;
    if (!state->residency->Init(&memory_error)) {
      std::cerr << "Error: " << memory_error << "\n";
      return 2;
    }
    if (memory_guard) {
      Logger::Info("router",
                   "event=memory_guard file=" + memory_file.string() +
                       " factor=" + std::to_string(memory_factor) +
                       " reserve_gpu_gb=" + std::to_string(reserve_gpu_gb) +
                       " reserve_host_gb=" + std::to_string(reserve_host_gb));
    }
  }
  if (!comfy_dir.empty()) {
    router::comfy::AdapterOptions comfy_options;
    comfy_options.root = comfy_dir;
    comfy_options.ffmpeg = comfy_ffmpeg;
    RouterState* const raw = state.get();
    if (api_key.empty()) {
      // Jobs go through the front like any client, so they show up in the
      // request log and count as busy for idle checks.
      const std::string self_host =
          host.empty() || host == "0.0.0.0" || host == "::" || host == "[::]"
              ? "127.0.0.1"
              : host;
      comfy_options.fetch = [self_host, port](const HttpRequest& request) {
        router::UpstreamTarget target;
        target.host = self_host;
        target.port = port;
        return CollectResponse(router::ProxyToWorker(target, request));
      };
    } else {
      // The loopback relay drops Authorization; dispatch in process instead.
      comfy_options.fetch = [raw](const HttpRequest& request) {
        return CollectResponse(raw->Dispatch(request));
      };
    }
    comfy_options.unload = [raw] { raw->UnloadIdle(); };
    try {
      state->comfy =
          std::make_unique<router::comfy::Adapter>(std::move(comfy_options));
    } catch (const std::exception& error) {
      std::cerr << "Error: --comfy-dir: " << error.what() << '\n';
      return 2;
    }
  }

  // --autoload preloads in section order up to --models-max, waiting for each
  // worker's health endpoint within --load-timeout-seconds.
  if (autoload) {
    for (const auto& model : models) {
      if ((registry_options.max_active > 0 &&
           state->workers.size() >= registry_options.max_active) ||
          (memory_guard && !state->workers.empty())) {
        break;
      }
      std::string spawn_error;
      const int worker_port = router::ReserveLoopbackPort(&spawn_error);
      if (worker_port < 0) {
        std::cerr << "Error: model '" << model.id << "': " << spawn_error
                  << "\n";
        return 2;
      }
      const router::WorkerSpec spec{.model_id = model.id,
                                    .modality = model.modality,
                                    .args = model.argv,
                                    .port = worker_port,
                                    .command = model.command};
      auto worker = router::Worker::Spawn(spec, state->exe_path, &spawn_error);
      if (worker == nullptr) {
        std::cerr << "Error: model '" << model.id << "': " << spawn_error
                  << "\n";
        return 2;
      }
      Logger::Info("router", "event=autoload_spawn model=" + model.id +
                                 " port=" + std::to_string(worker_port));
      const auto started = std::chrono::steady_clock::now();
      while (!worker->Healthy()) {
        int exit_status = 0;
        if (worker->PollExit(&exit_status)) {
          std::cerr << "Error: model '" << model.id
                    << "' worker exited with status " << exit_status << "\n";
          return 2;
        }
        if (load_timeout_seconds > 0 &&
            std::chrono::steady_clock::now() - started >=
                std::chrono::seconds(load_timeout_seconds)) {
          std::cerr << "Error: model '" << model.id << "' load timed out\n";
          return 2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
      state->registry.MarkReady(model.id);
      Logger::Info("router", "event=autoload_ready model=" + model.id);
      state->workers[model.id] = std::move(worker);
      state->residency->OnSpawn(model.id);
      state->residency->Event("autoload " + model.id);
    }
  }

  HttpServerOptions options;
  options.max_request_body_bytes = max_request_body_bytes;
  options.max_connections = max_connections;
  options.api_key = api_key;
  // SIGTERM only stops the listener here; the drain below keeps established
  // connections (and their relays) alive until they finish or the drain
  // window ends, instead of the front severing them on the signal.
  options.graceful_shutdown = true;
  options.dispatcher = [state](const HttpRequest& request) {
    return state->Dispatch(request);
  };
  HttpServer server(host, port, nullptr, nullptr, nullptr, nullptr,
                    std::move(options));
  std::string start_error;
  if (!server.start(&start_error)) {
    std::cerr << "Error: " << start_error << "\n";
    return 2;
  }
  state->StartSpawner();
  if (state->comfy != nullptr) {
    state->comfy->Start();
    Logger::Info("router", "event=comfy_api dir=" + comfy_dir.string() +
                               " ffmpeg=" +
                               (comfy_ffmpeg.empty() ? std::string("none")
                                                     : comfy_ffmpeg.string()));
  }
  std::thread supervision([&state] {
    while (!state->stop_requested.load()) {
      state->SuperviseOnce();
      std::this_thread::sleep_for(kSupervisionTick);
    }
  });
  server.run(true);

  // Graceful shutdown: stop accepting, refuse new loads, drain in-flight
  // work for up to 10 s, then terminate every worker.
  state->stop_requested.store(true);
  if (state->comfy != nullptr)
    state->comfy->Stop();
  const auto drain_deadline = std::chrono::steady_clock::now() + kShutdownDrain;
  while (std::chrono::steady_clock::now() < drain_deadline) {
    bool any_busy = false;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (const auto& [id, worker] : state->workers) {
        (void)worker;
        any_busy = any_busy || state->registry.Busy(id);
      }
    }
    if (!any_busy) {
      break;
    }
    // The lock is free while sleeping so completing requests can decrement
    // their busy count and end the drain early.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  {
    std::vector<std::shared_ptr<router::Worker>> live;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (auto& [id, worker] : state->workers) {
        (void)id;
        live.push_back(std::move(worker));
      }
      state->workers.clear();
    }
    // Stop runs with the state mutex free; the local shared_ptrs keep the
    // worker objects alive until each has been terminated.
    for (auto& worker : live) {
      worker->Stop();
    }
  }
  supervision.join();
  state->StopSpawner();
  return 0;
}

}  // namespace gufo::cli
