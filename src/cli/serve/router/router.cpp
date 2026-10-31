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
#include "src/cli/serve/router/jobs.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/cli/serve/router/registry.hpp"
#include "src/cli/serve/router/remote.hpp"
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
    "sections.\n"
    "\n"
    "    [remote/<model-id>]   a model another gufo router (1.2 or newer)\n"
    "                          serves under the same id; nothing is started\n"
    "                          here. Keys:\n"
    "      url = http://<IPv4>:<port>   the remote router; must match\n"
    "                                   --remote-allow (required)\n"
    "      modality = image             llm|image|video|tts|asr|embedding|\n"
    "                                   omni, as listed (required)\n"
    "      api-key-env = NAME           environment variable with the remote\n"
    "                                   router's --api-key\n"
    "      connect-timeout-seconds = 3  unreachable -> 503 remote_unavailable\n"
    "      head-timeout-seconds = 0     response head deadline (0 = none)\n"
    "      gpu-slot = NAME              remote models sharing NAME run one\n"
    "                                   GPU request (or video/music job) at\n"
    "                                   a time; others wait\n"
    "      queue-timeout-seconds = 900  longest wait, then 503 remote_busy\n"
    "      busy-check-url = http://<IPv4>:<port>/queue  optional ComfyUI\n"
    "                                   queue; wait while it runs anything\n"
    "      max-request-bytes = N        413 above N bytes\n\n";

constexpr std::chrono::milliseconds kAdmissionPoll{50};
constexpr std::chrono::milliseconds kSupervisionTick{250};
constexpr int kSpawnAttempts = 3;
constexpr std::chrono::seconds kShutdownDrain{10};
/// Remote routers: model list refresh period and the bound of one probe.
constexpr std::chrono::seconds kRemoteProbeEvery{5};
constexpr std::chrono::milliseconds kRemoteProbeTimeout{2000};
/// Busy-check url: poll period while it reports work.
constexpr std::chrono::seconds kBusyCheckEvery{2};

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

/// llama.cpp's error body (format_error_response): a numeric `code` equal to
/// the HTTP status and llama.cpp's `type` names, for the router-mode
/// /models/load and /models/unload routes clients written for it parse.
HttpResponse LlamaErr(int status, std::string_view reason,
                      std::string_view message, std::string_view type) {
  json::Value detail = json::Value::object();
  detail["code"] = status;
  detail["message"] = std::string(message);
  detail["type"] = std::string(type);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return HttpResponse{.status = status,
                      .reason = std::string(reason),
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"}}};
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

/// A failed --autoload of one model: the router stays up, lists the model as
/// failed and retries after `retry_at` (a request may load it before).
struct AutoloadFailure {
  std::string error;
  int attempts{0};
  std::chrono::steady_clock::time_point retry_at{};
};
using AutoloadFailures = std::map<std::string, AutoloadFailure>;

long long RetryInSeconds(const AutoloadFailure& failure) {
  const auto left = std::chrono::duration_cast<std::chrono::seconds>(
      failure.retry_at - std::chrono::steady_clock::now());
  return std::max<long long>(0, left.count());
}

json::Value AutoloadFailureJson(const AutoloadFailure& failure) {
  json::Value out = json::Value::object();
  out["error"] = failure.error;
  out["attempts"] = failure.attempts;
  out["retry_in_s"] = RetryInSeconds(failure);
  return out;
}

/// Router-managed paths answered by the front itself. Returns nullopt when
/// the path belongs to the model-routing table instead.
std::optional<HttpResponse> RouteFront(const router::Registry& registry,
                                       const AutoloadFailures& failures,
                                       const router::RemoteHub* remotes,
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
    for (const auto& [preset, local_status] : registry.Statuses()) {
      // A remote model shows its router's last listed state (cached by the
      // remote thread, never asked live) and no args.
      std::string_view status = local_status;
      std::string remote_status;
      json::Value model = json::Value::object();
      model["id"] = preset->id;
      model["object"] = "model";
      model["owned_by"] = "gufo";
      model["modality"] = preset->modality;
      if (preset->remote.has_value()) {
        const router::RemoteEndpoint& url = preset->remote->url;
        std::string error;
        json::Value remote = json::Value::object();
        remote["url"] = "http://" + router::EndpointKey(url);
        const bool reachable =
            remotes == nullptr || remotes->Reachable(url, &error);
        remote["reachable"] = reachable;
        if (!reachable)
          remote["error"] = error;
        model["remote"] = std::move(remote);
        if (remotes != nullptr)
          remote_status = remotes->ModelStatus(url, preset->id);
        status = remote_status.empty() ? std::string_view("unloaded")
                                       : std::string_view(remote_status);
      }
      model["loaded"] = status == "loaded";
      json::Value args = json::Value::array();
      for (const std::string& arg : preset->argv)
        args.push_back(arg);
      json::Value state = json::Value::object();
      state["value"] = std::string(status);
      state["args"] = std::move(args);
      // llama.cpp marks a failed instance with status.failed; gufo adds the
      // error and the next --autoload retry.
      if (const auto failed = failures.find(preset->id);
          failed != failures.end() && std::string_view(status) == "unloaded") {
        state["failed"] = true;
        state["error"] = failed->second.error;
        state["attempts"] = failed->second.attempts;
        state["retry_in_s"] = RetryInSeconds(failed->second);
      }
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

/// Positive whole seconds separated by commas ("30,60,120").
bool ParseSecondsList(std::string_view text,
                      std::vector<std::chrono::seconds>* out) {
  out->clear();
  for (std::size_t start = 0; start <= text.size();) {
    const auto end = std::min(text.find(',', start), text.size());
    const std::string_view part = text.substr(start, end - start);
    long long value = 0;
    const auto [ptr, ec] =
        std::from_chars(part.data(), part.data() + part.size(), value);
    if (ec != std::errc{} || ptr != part.data() + part.size() || value < 1)
      return false;
    out->emplace_back(value);
    start = end + 1;
  }
  return true;
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
  // The worker POST routes that take a JSON `model` field, including the
  // llama-server (/completion, /apply-template) and embedding
  // (/embeddings, /embedding) aliases.
  static const std::vector<std::string> kPaths = {
      "/v1/chat/completions",      "/v1/completions",
      "/v1/responses",             "/v1/messages",
      "/v1/messages/count_tokens", "/v1/images/generations",
      "/v1/images/edits",          "/v1/audio/speech",
      "/v1/audio/music",           "/v1/embeddings",
      "/embeddings",               "/embedding",
      "/completion",               "/apply-template"};
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

/// Worker GET routes that describe one model (voices, llama-server props,
/// image LoRAs). Their model comes from the `model` query parameter like
/// the introspection paths, but they load the model like any request.
bool IsModelQueryGet(const HttpRequest& req) {
  return req.method == "GET" &&
         (req.path == "/v1/audio/voices" || req.path == "/props" ||
          req.path == "/sdapi/v1/loras");
}

/// Job id from `/v1/audio/music/<id>` or `/v1/audio/music/<id>/content`,
/// the async music reads music_api.cpp serves; empty for anything else.
std::string MusicJobId(const std::string& path) {
  constexpr std::string_view prefix = "/v1/audio/music/";
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
  if (req.method == "GET" &&
      (IsIntrospectionPath(req.path) || IsModelQueryGet(req))) {
    resolution.model_id = req.query_param("model");
    resolution.missing = resolution.model_id.empty();
    return resolution;
  }
  if (req.method == "GET" && !MusicJobId(req.path).empty()) {
    // Optional: without ?model= the job's recorded owner answers.
    resolution.model_id = req.query_param("model");
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

/// Everything one remote relay holds until its response is done: the busy
/// count (job reads route back, /models/unload sees it), the in-flight
/// count the shutdown drain waits for and, with gpu-slot, the remote GPU
/// slot unless a started job takes it over. Release is idempotent and takes
/// the state mutex itself (through HeldBusy).
class RemoteHold {
public:
  RemoteHold(std::shared_ptr<HeldBusy> busy, std::atomic<int>& inflight)
      : busy_(std::move(busy)), inflight_(inflight) {
    ++inflight_;
  }
  RemoteHold(const RemoteHold&) = delete;
  RemoteHold& operator=(const RemoteHold&) = delete;
  ~RemoteHold() { Release(); }

  void AttachSlot(std::shared_ptr<router::GpuSlots::Hold> slot) {
    slot_ = std::move(slot);
  }
  /// Hands the slot to a started job (nullptr without one).
  std::shared_ptr<router::GpuSlots::Hold> TakeSlot() {
    return std::move(slot_);
  }

  void Release() {
    if (released_.exchange(true)) {
      return;
    }
    busy_->Release();
    --inflight_;
    if (slot_ != nullptr) {
      slot_->Release();
    }
  }

private:
  std::shared_ptr<HeldBusy> busy_;
  std::atomic<int>& inflight_;
  std::shared_ptr<router::GpuSlots::Hold> slot_;
  std::atomic<bool> released_{false};
};

/// True for remote requests that start GPU work there (generation,
/// transcription, chat, a job, a load); reads never wait for the slot.
bool StartsRemoteGpuWork(const HttpRequest& req) {
  return req.method == "POST";
}

/// A remote video or async music job that holds its model's GPU slot until
/// it ends. The remote thread polls its status; a client read that sees it
/// end releases it as well.
struct RemoteJob {
  std::string model_id;
  router::RemoteSpec spec;
  bool music{false};
  std::shared_ptr<router::GpuSlots::Hold> slot;
};

bool IsVideoCreate(const HttpRequest& req) {
  return req.method == "POST" && req.path == "/v1/videos";
}

bool IsMusicCreate(const HttpRequest& req) {
  return req.method == "POST" && req.path == "/v1/audio/music";
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

/// Sniffs a relayed worker body so the router can keep its video- and
/// async-music-job bookkeeping aligned with the worker's authoritative
/// response. Bytes accumulate up to the cap; an oversized or malformed body
/// still relays untouched and is simply not recorded. kContent (a music
/// result download) only needs the 2xx and buffers nothing.
class JobBodySniffer {
public:
  enum class Kind { kTrack, kStatus, kDelete, kContent };

  JobBodySniffer(std::mutex& mutex, router::Registry& registry, Kind kind,
                 std::string model_id, std::string job_id)
      : mutex_(mutex),
        registry_(registry),
        kind_(kind),
        model_id_(std::move(model_id)),
        job_id_(std::move(job_id)) {}
  JobBodySniffer(const JobBodySniffer&) = delete;
  JobBodySniffer& operator=(const JobBodySniffer&) = delete;

  std::string_view Observe(std::string_view chunk) {
    if (!discarded_ && kind_ != Kind::kContent) {
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
    if (kind_ == Kind::kDelete || kind_ == Kind::kContent) {
      const std::lock_guard<std::mutex> lock(mutex_);
      CompleteOwnedLocked();
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
        tracked_ = id->str();
      }
      return;
    }
    const auto* status = parsed.find("status");
    if (status == nullptr || !status->is_string()) {
      return;
    }
    const std::string value = status->str();
    if (value == "completed" || value == "failed" || value == "cancelled") {
      CompleteOwnedLocked();
    }
  }

  /// The job id a create response recorded (after Commit); "" otherwise.
  const std::string& tracked() const { return tracked_; }
  /// The job a status, content or DELETE read is about; "" for a create.
  const std::string& job_id() const { return job_id_; }

private:
  /// Releases the job only for the model that answered: music ids are
  /// numbered per worker, so a `?model=` read of another worker's
  /// `music_1` must not close the recorded owner's job.
  void CompleteOwnedLocked() {
    if (registry_.JobOwner(job_id_) == model_id_) {
      registry_.CompleteVideoJob(job_id_);
    }
  }

  static constexpr std::size_t kBodySniffBytes = 1 << 20;

  std::mutex& mutex_;
  router::Registry& registry_;
  Kind kind_;
  std::string model_id_;
  std::string job_id_;
  std::string buffer_;
  std::string tracked_;
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
  /// --autoload models whose load failed and wait for a retry.
  AutoloadFailures autoload_failed;
  /// ComfyUI-compatible front (--comfy-dir); null when disabled.
  std::unique_ptr<router::comfy::Adapter> comfy;
  /// Background jobs with webhooks (--jobs-dir); null when disabled.
  std::unique_ptr<router::jobs::JobQueue> jobs;
  /// Relays to remote routers still running (the shutdown drain waits).
  std::atomic<int> remote_inflight{0};
  /// Reachability and model states of the remote routers; null without
  /// [remote/<id>] sections.
  std::unique_ptr<router::RemoteHub> remotes;
  /// Remote GPU slots (preset gpu-slot).
  router::GpuSlots gpu_slots;
  /// Remote jobs holding a GPU slot, by job id (own mutex, never nested
  /// inside the state mutex).
  std::mutex remote_jobs_mutex_;
  std::map<std::string, RemoteJob> remote_jobs_;

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
    if (jobs != nullptr) {
      if (auto answer = jobs->Handle(req))
        return std::move(*answer);
    }
    if (comfy != nullptr) {
      if (auto answer = comfy->Handle(req))
        return std::move(*answer);
    }
    if (auto page = ServePage(pages_dir, req))
      return std::move(*page);
    std::unique_lock<std::mutex> lock(mutex);
    if (req.method == "POST" &&
        (req.path == "/models/load" || req.path == "/models/unload")) {
      return ModelsControlLocked(req, lock);
    }
    if (const auto front =
            RouteFront(registry, autoload_failed, remotes.get(), req)) {
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
    if (req.path.starts_with("/v1/audio/music/") && req.method != "GET") {
      // Music job reads are GET-only, as in music_api.cpp.
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
    const std::string music_job = req.method == "GET"
                                      ? MusicJobId(req.path)
                                      : std::string();
    if (resolved.model_id.empty() && !music_job.empty()) {
      // Async music reads (status, /content) go to the worker that answered
      // the POST, like video jobs.
      resolved.model_id = registry.JobOwner(music_job);
      if (resolved.model_id.empty()) {
        return Err(404, "Not Found", "unknown music job " + music_job,
                   "invalid_request_error", "not_found");
      }
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
    if (preset->remote.has_value()) {
      // Another router owns the model: no admission, load or memory here.
      return DispatchRemoteLocked(req, model_id, *preset->remote, lock);
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
    if (!music_job.empty() && !LoadedLocked(model_id)) {
      // Music jobs live only in the worker's memory: a stopped worker took
      // them along, so a read never starts one. Drop the open job so the
      // model no longer counts as busy.
      if (registry.JobOwner(music_job) == model_id)
        registry.CompleteVideoJob(music_job);
      return Err(404, "Not Found", "unknown music job " + music_job,
                 "invalid_request_error", "not_found");
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
    std::shared_ptr<JobBodySniffer> sniffer;
    if (response.status >= 200 && response.status < 300 &&
        response.streaming_body) {
      sniffer = MakeJobSniffer(req, model_id);
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

  /// Relays a request for a `[remote/<id>]` model to its router. The model
  /// stays "unloaded" in the registry (never admitted, measured, evicted or
  /// idle-unloaded); only the busy count and the job owners track it, so
  /// the drain waits for it and job reads route back. The relay runs with
  /// the state mutex released. A connect failure is a 503
  /// remote_unavailable: nothing local is restarted.
  HttpResponse DispatchRemoteLocked(const HttpRequest& req,
                                    const std::string& model_id,
                                    const router::RemoteSpec& spec,
                                    std::unique_lock<std::mutex>& lock) {
    if (spec.max_request_bytes > 0 &&
        req.body.size() > spec.max_request_bytes) {
      return Err(413, "Payload Too Large",
                 "request body exceeds " +
                     std::to_string(spec.max_request_bytes) +
                     " bytes for remote model '" + model_id + "'",
                 "invalid_request_error", "request_too_large");
    }
    if (stop_requested.load()) {
      return Err(502, "Bad Gateway", "router is shutting down", "server_error",
                 "upstream_unavailable");
    }
    auto hold = std::make_shared<RemoteHold>(
        std::make_shared<HeldBusy>(mutex, registry, model_id)->Arm(),
        remote_inflight);
    lock.unlock();
    if (StartsRemoteGpuWork(req)) {
      if (auto refused = WaitForRemoteGpu(req, model_id, spec, *hold)) {
        hold->Release();
        return std::move(*refused);
      }
    }
    HttpResponse response = RelayRemote(model_id, spec, req);
    if (response.status == 503 && !response.streaming_body) {
      hold->Release();
      return response;
    }
    std::shared_ptr<JobBodySniffer> sniffer;
    if (response.status >= 200 && response.status < 300 &&
        response.streaming_body) {
      sniffer = MakeJobSniffer(req, model_id);
    }
    if (!response.streaming_body) {
      hold->Release();
      return response;
    }
    // The hold lives until the streamed body completes or cancels; a video
    // or async music job keeps the GPU slot until it ends.
    HttpResponse streamed = std::move(response);
    const bool music = IsMusicCreate(req);
    streamed.streaming_body = [this, inner = std::move(streamed.streaming_body),
                               hold, sniffer, model_id, spec, music](
                                  const HttpResponse::BodyWriter& writer) {
      const HttpResponse::BodyWriter sink =
          sniffer == nullptr ? writer
                             : [&writer, sniffer](std::string_view chunk) {
                                 return writer(sniffer->Observe(chunk));
                               };
      try {
        inner(sink);
      } catch (...) {
        hold->Release();
        throw;
      }
      if (sniffer != nullptr) {
        sniffer->Commit();
        if (!sniffer->tracked().empty()) {
          if (auto slot = hold->TakeSlot()) {
            HoldSlotForJob(sniffer->tracked(), model_id, spec, music,
                           std::move(slot));
          }
        } else if (!sniffer->job_id().empty()) {
          ReleaseJobSlotIfDone(sniffer->job_id());
        }
      }
      hold->Release();
    };
    return streamed;
  }

  /// Hands the GPU slot of the request that started a remote job to the
  /// job, until the remote thread or a client read sees it end.
  void HoldSlotForJob(const std::string& job_id, const std::string& model_id,
                      const router::RemoteSpec& spec, bool music,
                      std::shared_ptr<router::GpuSlots::Hold> slot) {
    slot->Describe(std::string(music ? "music" : "video") + " job " + job_id +
                   " on " + model_id);
    const std::lock_guard<std::mutex> lock(remote_jobs_mutex_);
    auto& job = remote_jobs_[job_id];
    if (job.slot != nullptr && job.slot != slot) {
      job.slot->Release();
    }
    job = RemoteJob{model_id, spec, music, std::move(slot)};
  }

  /// Releases the slot of a job the registry no longer counts as open.
  void ReleaseJobSlotIfDone(const std::string& job_id) {
    bool open = false;
    {
      const std::lock_guard<std::mutex> lock(mutex);
      open = registry.JobOpen(job_id);
    }
    if (!open) {
      ReleaseJobSlot(job_id, "finished");
    }
  }

  void ReleaseJobSlot(const std::string& job_id, std::string_view why) {
    std::shared_ptr<router::GpuSlots::Hold> slot;
    std::string model_id;
    {
      const std::lock_guard<std::mutex> lock(remote_jobs_mutex_);
      const auto it = remote_jobs_.find(job_id);
      if (it == remote_jobs_.end()) {
        return;
      }
      slot = std::move(it->second.slot);
      model_id = it->second.model_id;
      remote_jobs_.erase(it);
    }
    Logger::Info("router", "event=remote_job_end job=" + job_id +
                               " model=" + model_id + " reason=" +
                               std::string(why));
    if (slot != nullptr) {
      slot->Release();
    }
  }

  /// Before a GPU-starting remote request (state mutex free): takes the
  /// model's gpu slot, then waits while the busy-check url reports work,
  /// both within queue-timeout-seconds. Returns the 503 remote_busy when
  /// the wait ran out (or the client left), nullopt to go ahead.
  std::optional<HttpResponse> WaitForRemoteGpu(const HttpRequest& req,
                                               const std::string& model_id,
                                               const router::RemoteSpec& spec,
                                               RemoteHold& hold) {
    const auto began = std::chrono::steady_clock::now();
    const auto deadline =
        began + std::chrono::seconds(spec.queue_timeout_seconds);
    const auto busy = [&](const std::string& what) {
      Logger::Warn("router", "event=remote_busy model=" + model_id +
                                 " waited_ms=" +
                                 std::to_string(
                                     std::chrono::duration_cast<
                                         std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() -
                                         began)
                                         .count()) +
                                 " reason=\"" + what + "\"");
      HttpResponse refused =
          Err(503, "Service Unavailable",
              "remote model '" + model_id + "' is busy: " + what,
              "server_error", "remote_busy");
      refused.headers.emplace_back("Retry-After", "30");
      return refused;
    };
    if (!spec.gpu_slot.empty()) {
      auto slot = gpu_slots.Acquire(
          spec.gpu_slot, req.method + " " + req.path + " " + model_id,
          std::chrono::milliseconds(std::chrono::seconds(
              spec.queue_timeout_seconds)),
          req.is_cancelled);
      if (slot == nullptr) {
        return busy("gpu slot '" + spec.gpu_slot + "' stayed taken");
      }
      hold.AttachSlot(std::move(slot));
    }
    if (!spec.busy_check.has_value()) {
      return std::nullopt;
    }
    const router::RemoteEndpoint& check = *spec.busy_check;
    for (;;) {
      router::UpstreamTarget target;
      target.host = check.host;
      target.port = check.port;
      target.connect_timeout = kRemoteProbeTimeout;
      target.head_timeout = kRemoteProbeTimeout;
      HttpRequest probe;
      probe.method = "GET";
      const auto query = check.path.find('?');
      probe.path = check.path.substr(0, query);
      if (query != std::string::npos)
        probe.query = check.path.substr(query + 1);
      const auto probe_deadline =
          std::chrono::steady_clock::now() + 2 * kRemoteProbeTimeout;
      probe.is_cancelled = [probe_deadline] {
        return std::chrono::steady_clock::now() >= probe_deadline;
      };
      const router::comfy::FetchResult answer =
          CollectResponse(router::ProxyToWorker(target, probe));
      std::string result = "free";
      if (answer.status == 502 && answer.body.find("upstream_unavailable") !=
                                      std::string::npos) {
        // Nothing answers (ComfyUI paused while gufo runs): free.
        result = "unreachable (counts as free)";
      } else if (answer.status == 200 &&
                 router::BusyCheckSaysBusy(answer.body)) {
        result = "busy";
      }
      remotes->NoteBusyCheck(check, result);
      if (result != "busy") {
        return std::nullopt;
      }
      if (req.is_cancelled && req.is_cancelled()) {
        return busy("client left while the busy check reported work");
      }
      if (std::chrono::steady_clock::now() + kBusyCheckEvery >= deadline) {
        return busy("busy check http://" + router::EndpointKey(check) +
                    check.path + " kept reporting work");
      }
      std::this_thread::sleep_for(kBusyCheckEvery);
    }
  }

  /// One relay to a remote router (state mutex free). A connect failure or
  /// a remote that closed before answering becomes 503 remote_unavailable;
  /// every remote answer, errors included, passes through unchanged.
  HttpResponse RelayRemote(const std::string& model_id,
                           const router::RemoteSpec& spec,
                           const HttpRequest& req) {
    std::string known_error;
    if (remotes != nullptr && !remotes->Reachable(spec.url, &known_error)) {
      // The last probe or relay failed: answer at once instead of letting
      // every client wait out the connect timeout.
      return router::RemoteUnavailable(model_id, known_error);
    }
    HttpResponse response =
        router::ProxyToWorker(router::RemoteTarget(spec), req);
    if (response.status != 502 || response.streaming_body) {
      return response;
    }
    std::string detail = response.log_details;
    constexpr std::string_view kPrefix = "upstream_unavailable detail=";
    if (std::string_view(detail).starts_with(kPrefix)) {
      detail.erase(0, kPrefix.size());
    }
    if (remotes != nullptr && (detail.starts_with("connect") ||
                               detail.starts_with("resolve"))) {
      // Only a failed connect says the router is gone; a lost answer may
      // be one bad request.
      remotes->MarkUnreachable(spec.url, detail);
    }
    Logger::Warn("router", "event=remote_unavailable model=" + model_id +
                               " url=" + spec.url.host + ":" +
                               std::to_string(spec.url.port) + " detail=\"" +
                               detail + "\"");
    return router::RemoteUnavailable(model_id, detail);
  }

  /// Live previews (/v1/previews/{id}[/events]) of a running image or video
  /// job: the model comes from `?model=` or, for a video job, its owner.
  /// Only a loaded worker is asked; a preview never starts or keeps one.
  /// A remote model's preview goes to its router.
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
    if (const router::PresetModel* preset =
            model_id.empty() ? nullptr : registry.Find(model_id);
        preset != nullptr && preset->remote.has_value()) {
      // The remote router finds the job itself: path and query go unchanged
      // and the SSE stream relays as it arrives.
      const router::RemoteSpec spec = *preset->remote;
      lock.unlock();
      return RelayRemote(model_id, spec, req);
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

  /// llama.cpp router-mode model control (state mutex held):
  ///   POST /models/unload {"model": id} stops the worker -> {"success": true}
  ///   POST /models/load   {"model": id} loads it (synchronously, unlike
  ///                       llama.cpp, which answers before the load ends)
  /// Errors use llama.cpp's messages and shape: unknown model -> "model is
  /// not found" (unload 400 invalid_request_error, load 404 not_found_error),
  /// unload of an unloaded model -> 400 "model is not running", load of a
  /// loaded/loading one -> 400 "model is already running". gufo addition:
  /// unloading a model that serves requests, sockets or video jobs is a 409
  /// "model is busy" instead of cutting the work off. A remote model's
  /// load/unload goes to its router with the state mutex released; that
  /// router answers (and decides about its memory) itself.
  HttpResponse ModelsControlLocked(const HttpRequest& req,
                                   std::unique_lock<std::mutex>& lock) {
    const bool unload = req.path == "/models/unload";
    std::string model_id;
    try {
      const json::Value body = json::parse(req.body);
      model_id = body.member_str("model");
    } catch (const std::exception&) {
      return LlamaErr(400, "Bad Request", "request body must be JSON",
                      "invalid_request_error");
    }
    const router::PresetModel* preset = registry.Find(model_id);
    if (preset == nullptr) {
      return unload ? LlamaErr(400, "Bad Request", "model is not found",
                               "invalid_request_error")
                    : LlamaErr(404, "Not Found", "model is not found",
                               "not_found_error");
    }
    if (preset->remote.has_value()) {
      const router::RemoteSpec spec = *preset->remote;
      Logger::Info("router", std::string("event=models_") +
                                 (unload ? "unload" : "load") +
                                 " model=" + model_id + " remote=" +
                                 spec.url.host + ":" +
                                 std::to_string(spec.url.port));
      auto hold = std::make_shared<RemoteHold>(
          std::make_shared<HeldBusy>(mutex, registry, model_id)->Arm(),
          remote_inflight);
      lock.unlock();
      if (!unload) {
        // A load fills the remote GPU like a request does.
        if (auto refused = WaitForRemoteGpu(req, model_id, spec, *hold)) {
          hold->Release();
          return std::move(*refused);
        }
      }
      HttpResponse answer = RelayRemote(model_id, spec, req);
      if (!answer.streaming_body) {
        hold->Release();
        return answer;
      }
      answer.streaming_body = [inner = std::move(answer.streaming_body),
                               hold](const HttpResponse::BodyWriter& writer) {
        try {
          inner(writer);
        } catch (...) {
          hold->Release();
          throw;
        }
        hold->Release();
      };
      return answer;
    }
    bool resident = false;
    bool busy = false;
    for (const auto& entry : registry.Residents()) {
      if (entry.id == model_id) {
        resident = true;
        busy = entry.busy;
      }
    }
    json::Value ok = json::Value::object();
    ok["success"] = true;
    const HttpResponse success{
        .status = 200,
        .reason = "OK",
        .body = ok.dump(),
        .headers = {{"Content-Type", "application/json"}}};
    if (unload) {
      if (!resident) {
        return LlamaErr(400, "Bad Request", "model is not running",
                        "invalid_request_error");
      }
      if (busy || registry.Busy(model_id)) {
        return LlamaErr(409, "Conflict", "model is busy",
                        "invalid_request_error");
      }
      Logger::Info("router", "event=models_unload model=" + model_id);
      if (residency != nullptr) {
        residency->Event("unloaded " + model_id + " (/models/unload)");
      }
      StopLocked(model_id);
      return success;
    }
    if (resident) {
      return LlamaErr(400, "Bad Request", "model is already running",
                      "invalid_request_error");
    }
    if (stop_requested.load()) {
      return Err(502, "Bad Gateway", "router is shutting down", "server_error",
                 "upstream_unavailable");
    }
    Logger::Info("router", "event=models_load model=" + model_id);
    auto admitted = AdmitLocked(model_id, *preset);
    if (auto* failed = std::get_if<HttpResponse>(&admitted)) {
      return std::move(*failed);
    }
    return success;
  }

  /// --autoload in the background, after the front listens: models load in
  /// section order through the normal admission (memory guard, load
  /// timeout) with one spawn per attempt. A failed load never ends the
  /// router: the model is listed as failed and retried after `delays`
  /// (the last delay repeats); a request may load it any time before.
  /// Stops at --models-max loaded workers, with --memory-guard after the
  /// first one.
  void AutoloadMain(std::vector<std::string> ids, bool memory_guard,
                    std::vector<std::chrono::seconds> delays) {
    if (delays.empty())
      delays.push_back(std::chrono::seconds(300));
    for (const std::string& id : ids) {
      {
        const std::lock_guard<std::mutex> lock(mutex);
        if (stop_requested.load())
          return;
        if ((options.max_active > 0 && workers.size() >= options.max_active) ||
            (memory_guard && !workers.empty()))
          return;
      }
      for (int attempt = 0;; ++attempt) {
        std::unique_lock<std::mutex> lock(mutex);
        if (stop_requested.load())
          return;
        if (LoadedLocked(id)) {
          autoload_failed.erase(id);
          break;
        }
        const router::PresetModel* preset = registry.Find(id);
        if (preset == nullptr)
          break;
        Logger::Info("router", "event=autoload_start model=" + id +
                                   " attempt=" + std::to_string(attempt + 1));
        auto admitted = AdmitLocked(id, *preset, /*max_spawns=*/1);
        if (std::holds_alternative<int>(admitted)) {
          autoload_failed.erase(id);
          Logger::Info("router", "event=autoload_ready model=" + id);
          if (residency != nullptr)
            residency->Event("autoload " + id);
          break;
        }
        if (stop_requested.load())
          return;
        std::string error = "load failed";
        try {
          const json::Value body =
              json::parse(std::get<HttpResponse>(admitted).body);
          if (const auto* detail = body.find("error"))
            error = detail->member_str("message", error);
        } catch (const std::exception&) {
          // Keep the generic text.
        }
        const auto delay =
            delays[std::min<std::size_t>(attempt, delays.size() - 1)];
        AutoloadFailure& failure = autoload_failed[id];
        failure.error = error;
        failure.attempts = attempt + 1;
        failure.retry_at = std::chrono::steady_clock::now() + delay;
        Logger::Warn("router", "event=autoload_failed model=" + id +
                                   " attempt=" + std::to_string(attempt + 1) +
                                   " retry_in_s=" +
                                   std::to_string(delay.count()) +
                                   " error=\"" + error + "\"");
        if (residency != nullptr)
          residency->Event("autoload of " + id + " failed (" + error +
                           "); retry in " + std::to_string(delay.count()) +
                           " s");
        const auto retry_at = failure.retry_at;
        bool loaded = false;
        while (!loaded && std::chrono::steady_clock::now() < retry_at) {
          lock.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
          lock.lock();
          if (stop_requested.load())
            return;
          loaded = LoadedLocked(id);
        }
        if (loaded) {
          autoload_failed.erase(id);
          break;
        }
      }
    }
  }

  /// Remote thread: asks every remote router for its model list every few
  /// seconds (no lock held while asking) and logs ids the remote does not
  /// serve. Runs until the router stops.
  void RemoteMain() {
    while (!stop_requested.load()) {
      for (const auto& target : remotes->Targets()) {
        if (stop_requested.load())
          return;
        router::RemoteProbe probe = router::ProbeRemoteModels(
            target.url, target.api_key, kRemoteProbeTimeout);
        const std::string key = router::EndpointKey(target.url);
        std::string reachable_before;
        const bool was_reachable =
            remotes->Reachable(target.url, &reachable_before);
        if (probe.reachable != was_reachable) {
          Logger::Info("router", "event=remote_" +
                                     std::string(probe.reachable
                                                     ? "reachable"
                                                     : "unreachable") +
                                     " url=" + key +
                                     (probe.error.empty()
                                          ? std::string()
                                          : " detail=\"" + probe.error + "\""));
        }
        for (const auto& id : remotes->Record(target.url, std::move(probe))) {
          Logger::Warn("router", "event=remote_model_missing model=" + id +
                                     " url=" + key);
        }
      }
      PollRemoteJobs();
      const auto next = std::chrono::steady_clock::now() + kRemoteProbeEvery;
      while (!stop_requested.load() && std::chrono::steady_clock::now() < next)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  /// Asks the remote router how each slot-holding job is doing (no lock
  /// held while asking). A terminal status, an unknown job or an
  /// unreachable router ends the job here: its slot goes to the next
  /// request and the registry stops counting it.
  void PollRemoteJobs() {
    std::vector<std::pair<std::string, RemoteJob>> jobs;
    {
      const std::lock_guard<std::mutex> lock(remote_jobs_mutex_);
      for (const auto& [id, job] : remote_jobs_)
        jobs.emplace_back(id, RemoteJob{job.model_id, job.spec, job.music,
                                        nullptr});
    }
    for (const auto& [job_id, job] : jobs) {
      if (stop_requested.load())
        return;
      router::UpstreamTarget target = router::RemoteTarget(job.spec);
      target.connect_timeout = kRemoteProbeTimeout;
      target.head_timeout = kRemoteProbeTimeout;
      HttpRequest status;
      status.method = "GET";
      status.path = (job.music ? "/v1/audio/music/" : "/v1/videos/") + job_id;
      if (job.music)
        status.query = "model=" + job.model_id;
      const auto deadline =
          std::chrono::steady_clock::now() + 2 * kRemoteProbeTimeout;
      status.is_cancelled = [deadline] {
        return std::chrono::steady_clock::now() >= deadline;
      };
      const router::comfy::FetchResult answer =
          CollectResponse(router::ProxyToWorker(target, status));
      std::string end;
      if (answer.status == 502 &&
          answer.body.find("upstream_unavailable") != std::string::npos) {
        end = "remote unreachable";
      } else if (answer.status == 404) {
        end = "unknown on the remote";
      } else if (answer.status >= 200 && answer.status < 300) {
        try {
          const std::string value =
              json::parse(answer.body).member_str("status");
          if (value == "completed" || value == "failed" ||
              value == "cancelled")
            end = value;
        } catch (const std::exception&) {
          // Not a job object: ask again next round.
        }
      }
      if (end.empty())
        continue;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        if (registry.JobOwner(job_id) == job.model_id &&
            registry.JobOpen(job_id))
          registry.CompleteVideoJob(job_id);
      }
      ReleaseJobSlot(job_id, end);
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
    json::Value failed = json::Value::object();
    {
      const std::lock_guard<std::mutex> lock(mutex);
      for (const auto& [id, failure] : autoload_failed)
        failed[id] = AutoloadFailureJson(failure);
    }
    body["autoload_failed"] = std::move(failed);
    if (remotes != nullptr) {
      // Remote routers (cached probe) and, per remote model, whether a
      // relay or job of it is open here.
      json::Value remote = json::Value::object();
      remote["routers"] = remotes->Status();
      json::Value models = json::Value::object();
      {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto& [preset, status] : registry.Statuses()) {
          (void)status;
          if (!preset->remote.has_value())
            continue;
          json::Value model = json::Value::object();
          model["url"] = "http://" + router::EndpointKey(preset->remote->url);
          model["busy"] = registry.Busy(preset->id);
          model["gpu_slot"] = preset->remote->gpu_slot;
          models[preset->id] = std::move(model);
        }
      }
      remote["models"] = std::move(models);
      remote["inflight"] = static_cast<long long>(remote_inflight.load());
      remote["gpu_slots"] = gpu_slots.Status();
      remote["busy_checks"] = remotes->BusyCheckStatus();
      json::Value jobs_held = json::Value::object();
      {
        const std::lock_guard<std::mutex> lock(remote_jobs_mutex_);
        for (const auto& [id, job] : remote_jobs_) {
          json::Value entry = json::Value::object();
          entry["model"] = job.model_id;
          entry["kind"] = job.music ? "music" : "video";
          entry["gpu_slot"] = job.spec.gpu_slot;
          jobs_held[id] = std::move(entry);
        }
      }
      remote["jobs"] = std::move(jobs_held);
      body["remote"] = std::move(remote);
    }
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

  /// Job bookkeeping for proxied 2xx responses: the create response of a
  /// video or async music job records the worker-assigned id against the
  /// owning model; a terminal status read, a video DELETE or a music result
  /// download releases it. Returns nullptr for every other request.
  std::shared_ptr<JobBodySniffer> MakeJobSniffer(const HttpRequest& req,
                                                 const std::string& model_id) {
    if (IsVideoCreate(req) || IsMusicCreate(req)) {
      // A synchronous music answer is a WAV body: it never parses and
      // records nothing.
      return std::make_shared<JobBodySniffer>(
          mutex, registry, JobBodySniffer::Kind::kTrack, model_id, "");
    }
    if (req.method == "GET") {
      if (const std::string music_id = MusicJobId(req.path);
          !music_id.empty()) {
        return std::make_shared<JobBodySniffer>(
            mutex, registry,
            req.path.ends_with("/content") ? JobBodySniffer::Kind::kContent
                                           : JobBodySniffer::Kind::kStatus,
            model_id, music_id);
      }
    }
    if (req.method != "GET" && req.method != "DELETE") {
      return nullptr;
    }
    const std::string job_id = VideoStatusJobId(req.path);
    if (job_id.empty()) {
      return nullptr;
    }
    return std::make_shared<JobBodySniffer>(
        mutex, registry,
        req.method == "DELETE" ? JobBodySniffer::Kind::kDelete
                               : JobBodySniffer::Kind::kStatus,
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
                              const router::PresetModel& preset,
                              int max_spawns = kSpawnAttempts) {
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
          if (attempt > max_spawns) {
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
  std::filesystem::path jobs_dir;
  std::size_t jobs_parallel = 1;
  std::string jobs_webhook_allow;
  std::string remote_allow;
  std::string jobs_webhook_secret;
  std::string jobs_webhook_retry = "5,30,120,600";
  std::size_t jobs_webhook_timeout_seconds = 10;
  std::size_t models_max = 2;
  std::size_t sleep_idle_seconds = 900;
  std::size_t load_timeout_seconds = 600;
  bool autoload = false;
  std::string autoload_retry = "30,60,120,300";
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
  parser.AddOption("", "--jobs-dir", "DIR",
                   "Serve the job API (/v1/jobs): wrapped image, video, "
                   "audio and chat requests run in the background; DIR holds "
                   "jobs.json and the results",
                   "Jobs", &jobs_dir);
  parser.AddOption("", "--jobs-parallel", "N",
                   "Jobs running at the same time (default: 1)", "Jobs",
                   &jobs_parallel);
  parser.AddOption("", "--jobs-webhook-allow", "LIST",
                   "Webhook targets jobs may name, comma separated: host "
                   "names, IPv4 addresses or a.b.c.d/n ranges, each with an "
                   "optional :port, or * (any but link-local/multicast). "
                   "Default: none, webhooks off",
                   "Jobs", &jobs_webhook_allow);
  parser.AddOption("", "--jobs-webhook-secret", "SECRET",
                   "Sign webhooks like Standard Webhooks (HMAC-SHA256; "
                   "whsec_<base64> or raw text)",
                   "Jobs", &jobs_webhook_secret);
  parser.AddOption("", "--jobs-webhook-retry-seconds", "LIST",
                   "Waits before retrying a failed webhook, comma separated "
                   "(default: 5,30,120,600)",
                   "Jobs", &jobs_webhook_retry);
  parser.AddOption("", "--jobs-webhook-timeout-seconds", "SEC",
                   "Time one webhook attempt may take (default: 10)", "Jobs",
                   &jobs_webhook_timeout_seconds);
  parser.AddOption("", "--remote-allow", "LIST",
                   "Remote routers [remote/<id>] sections may name, comma "
                   "separated: IPv4 addresses or a.b.c.d/n ranges, each "
                   "with an optional :port. Default: none, remote sections "
                   "are refused",
                   "Router", &remote_allow);
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
                 "Preload preset models in the background after startup, "
                 "in section order (with --memory-guard only the first "
                 "one); a failed load is listed as failed and retried",
                 "Router", &autoload);
  parser.AddOption("", "--autoload-retry-seconds", "LIST",
                   "Waits before retrying a failed --autoload, comma "
                   "separated; the last one repeats (default: "
                   "30,60,120,300)",
                   "Router", &autoload_retry);
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

  // Remote models reach only the routers --remote-allow names.
  std::vector<router::jobs::AllowEntry> remote_allow_list;
  if (std::string allow_error; !router::jobs::ParseAllowList(
          remote_allow, &remote_allow_list, &allow_error)) {
    std::cerr << "Error: --remote-allow: " << allow_error << "\n";
    return 2;
  }
  for (const auto& model : models) {
    std::string remote_error;
    if (model.remote.has_value() &&
        !router::CheckRemoteAllowed(*model.remote, remote_allow_list,
                                    &remote_error)) {
      std::cerr << "Error: [remote/" << model.id << "] " << remote_error
                << "\n";
      return 2;
    }
  }

  router::RegistryOptions registry_options;
  registry_options.max_active = models_max;
  registry_options.idle_timeout = std::chrono::seconds(sleep_idle_seconds);
  registry_options.load_timeout = std::chrono::seconds(load_timeout_seconds);
  std::vector<std::chrono::seconds> autoload_retry_delays;
  if (!ParseSecondsList(autoload_retry, &autoload_retry_delays)) {
    std::cerr << "Error: --autoload-retry-seconds wants positive whole "
                 "seconds separated by commas, not '"
              << autoload_retry << "'\n";
    return 2;
  }
  std::vector<std::chrono::seconds> webhook_retry_delays;
  if (!ParseSecondsList(jobs_webhook_retry, &webhook_retry_delays)) {
    std::cerr << "Error: --jobs-webhook-retry-seconds wants positive whole "
                 "seconds separated by commas, not '"
              << jobs_webhook_retry << "'\n";
    return 2;
  }
  std::vector<router::jobs::AllowEntry> webhook_allow;
  if (std::string allow_error; !router::jobs::ParseAllowList(
          jobs_webhook_allow, &webhook_allow, &allow_error)) {
    std::cerr << "Error: --jobs-webhook-allow: " << allow_error << "\n";
    return 2;
  }
  auto state = std::make_shared<RouterState>(models, registry_options);
  state->exe_path = router::ExecutablePath();
  if (auto hub = std::make_unique<router::RemoteHub>(models); !hub->empty())
    state->remotes = std::move(hub);
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
    // Remote models use the remote machine's memory: no estimates, no
    // learned peaks, no memory.json entries.
    std::vector<router::PresetModel> local_models;
    for (const auto& model : models) {
      if (!model.remote.has_value())
        local_models.push_back(model);
    }
    state->residency = std::make_unique<router::Residency>(
        residency_options, router::MakeSystemMemoryProbe(), local_models);
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
  // The ComfyUI adapter and the job API call the router's own API.
  RouterState* const raw = state.get();
  router::comfy::Fetch self_fetch;
  if (api_key.empty()) {
    // Jobs go through the front like any client, so they show up in the
    // request log and count as busy for idle checks.
    const std::string self_host =
        host.empty() || host == "0.0.0.0" || host == "::" || host == "[::]"
            ? "127.0.0.1"
            : host;
    self_fetch = [self_host, port](const HttpRequest& request) {
      router::UpstreamTarget target;
      target.host = self_host;
      target.port = port;
      return CollectResponse(router::ProxyToWorker(target, request));
    };
  } else {
    // The loopback relay drops Authorization; dispatch in process instead.
    self_fetch = [raw](const HttpRequest& request) {
      return CollectResponse(raw->Dispatch(request));
    };
  }
  if (!comfy_dir.empty()) {
    router::comfy::AdapterOptions comfy_options;
    comfy_options.root = comfy_dir;
    comfy_options.ffmpeg = comfy_ffmpeg;
    // Graphs that need a model the preset lacks (Qwen-Image Turbo) fail
    // validation instead of running on another model.
    for (const auto& model : models)
      comfy_options.models.available.push_back(model.id);
    comfy_options.fetch = self_fetch;
    comfy_options.unload = [raw] { raw->UnloadIdle(); };
    try {
      state->comfy =
          std::make_unique<router::comfy::Adapter>(std::move(comfy_options));
    } catch (const std::exception& error) {
      std::cerr << "Error: --comfy-dir: " << error.what() << '\n';
      return 2;
    }
  }
  if (!jobs_dir.empty()) {
    router::jobs::Options jobs_options;
    jobs_options.root = jobs_dir;
    jobs_options.fetch = self_fetch;
    for (const auto& model : models)
      jobs_options.models.push_back(model.id);
    jobs_options.parallel = jobs_parallel;
    jobs_options.webhook_allow = std::move(webhook_allow);
    jobs_options.webhook_secret = jobs_webhook_secret;
    jobs_options.webhook_retry.assign(webhook_retry_delays.begin(),
                                      webhook_retry_delays.end());
    jobs_options.webhook_timeout = std::chrono::seconds(
        std::max<std::size_t>(1, jobs_webhook_timeout_seconds));
    try {
      state->jobs =
          std::make_unique<router::jobs::JobQueue>(std::move(jobs_options));
    } catch (const std::exception& error) {
      std::cerr << "Error: --jobs-dir: " << error.what() << '\n';
      return 2;
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
  if (state->jobs != nullptr) {
    state->jobs->Start();
    Logger::Info("router", "event=jobs_api dir=" + jobs_dir.string() +
                               " parallel=" + std::to_string(jobs_parallel) +
                               " webhooks=" +
                               (jobs_webhook_allow.empty()
                                    ? std::string("off")
                                    : jobs_webhook_allow));
  }
  std::thread supervision([&state] {
    while (!state->stop_requested.load()) {
      state->SuperviseOnce();
      std::this_thread::sleep_for(kSupervisionTick);
    }
  });
  std::thread remote_thread;
  if (state->remotes != nullptr)
    remote_thread = std::thread([&state] { state->RemoteMain(); });
  // --autoload runs behind the listening front so a failed load (e.g. "bad
  // allocation" under memory pressure) never ends the router and other
  // models and pages stay reachable meanwhile.
  std::thread autoloader;
  if (autoload) {
    std::vector<std::string> ids;
    for (const auto& model : models) {
      if (!model.remote.has_value())
        ids.push_back(model.id);
    }
    autoloader = std::thread(
        [&state, ids = std::move(ids), memory_guard,
         delays = std::move(autoload_retry_delays)]() mutable {
          state->AutoloadMain(std::move(ids), memory_guard, std::move(delays));
        });
  }
  server.run(true);

  // Graceful shutdown: stop accepting, refuse new loads, drain in-flight
  // work for up to 10 s, then terminate every worker.
  state->stop_requested.store(true);
  if (state->comfy != nullptr)
    state->comfy->Stop();
  if (state->jobs != nullptr)
    state->jobs->Stop();
  const auto drain_deadline = std::chrono::steady_clock::now() + kShutdownDrain;
  while (std::chrono::steady_clock::now() < drain_deadline) {
    bool any_busy = false;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (const auto& [id, worker] : state->workers) {
        (void)worker;
        any_busy = any_busy || state->registry.Busy(id);
      }
      // Remote relays in flight; remote jobs run on, so they do not wait.
      any_busy = any_busy || state->remote_inflight.load() > 0;
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
  if (autoloader.joinable())
    autoloader.join();
  if (remote_thread.joinable())
    remote_thread.join();
  state->StopSpawner();
  return 0;
}

}  // namespace gufo::cli
