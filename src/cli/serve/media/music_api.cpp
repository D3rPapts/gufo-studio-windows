#include "src/cli/serve/media/music_api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/music_encode.hpp"
#include "src/core/json.hpp"
#include "src/models/ace_step/vae.hpp"

namespace gufo::server {
namespace {

namespace ace = models::ace_step;

std::int64_t Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

HttpResponse Json(int status, std::string reason, const json::Value& body) {
  HttpResponse response;
  response.status = status;
  response.reason = std::move(reason);
  response.body = body.dump();
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return Json(status, std::move(reason), root);
}

// Reads a member that may be given as a string or a number.
std::string Text(const json::Value& body, const char* key) {
  const json::Value* value = body.find(key);
  if (value == nullptr)
    return {};
  if (value->is_string())
    return value->str();
  if (value->is_number()) {
    const double number = value->as_double();
    if (number == std::floor(number))
      return std::to_string(static_cast<long long>(number));
    return std::to_string(number);
  }
  return {};
}

}  // namespace

struct MusicService::Job {
  std::string id;
  ace::MusicRequest request;
  std::string status{"queued"};
  double progress{0};
  std::string error;
  MusicOutputFormat format;
  std::string output;  // the WAV, or its mp3 / m4a transcode
  ace::MusicTelemetry telemetry;
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  bool done{false};
  // Live preview: requested explicitly ("preview": true) or watched (a GET
  // of /preview within kPreviewWatchWindow); the newest preview is kept.
  bool preview_always{false};
  int preview_every{2};
  bool preview_watched{false};
  std::chrono::steady_clock::time_point preview_watched_at{};
  ace::MusicPreview preview;  // wav empty before the first one
  std::int64_t preview_at{0};
};

namespace {

// "preview": true | false | "always" | {"always": bool, "every": 1..50},
// also as "gufo": {"preview": {...}} (the form the pages use for images; an
// "id" there is ignored, the preview lives under the music job id). Returns
// an error message (empty when fine).
std::string ParseMusicPreview(const json::Value& body, bool* always,
                              int* every) {
  const json::Value* value = body.find("preview");
  if (value == nullptr) {
    if (const json::Value* gufo = body.find("gufo");
        gufo != nullptr && gufo->is_object())
      value = gufo->find("preview");
  }
  if (value == nullptr || value->is_null())
    return {};
  if (value->is_bool()) {
    *always = value->as_bool();
    return {};
  }
  if (value->is_string()) {
    if (value->str() == "always") {
      *always = true;
      return {};
    }
    return "preview must be true, false, \"always\" or an object";
  }
  if (!value->is_object())
    return "preview must be true, false, \"always\" or an object";
  if (const json::Value* flag = value->find("always"); flag != nullptr) {
    if (!flag->is_bool())
      return "preview.always must be a boolean";
    *always = flag->as_bool();
  }
  if (const json::Value* n = value->find("every"); n != nullptr) {
    if (!n->is_number() || n->as_double() < 1 || n->as_double() > 50 ||
        n->as_double() != std::floor(n->as_double()))
      return "preview.every must be an integer 1..50";
    *every = static_cast<int>(n->as_double());
  }
  return {};
}

}  // namespace

bool IsMusicApiPath(std::string_view path) noexcept {
  return path == kMusicApiPath || (path.size() > kMusicApiPath.size() &&
                                   path.substr(0, kMusicApiPath.size() + 1) ==
                                       std::string(kMusicApiPath) + "/");
}

bool MusicService::HandlesPath(std::string_view path) const {
  return IsMusicApiPath(path);
}

MusicService::MusicService(const std::filesystem::path& model_root,
                           std::string model_id)
    : model_id_(std::move(model_id)) {
  ace::ModelPaths paths;
  if (!ace::ModelPaths::FromRoot(model_root, &paths, &init_error_))
    return;
#if defined(ENGINE_ENABLE_HIP)
  pipeline_ = ace::MusicPipeline::Load(paths, &init_error_);
  if (pipeline_)
    worker_ = std::thread([this] { Worker(); });
#else
  init_error_ = "ACE-Step music needs a ROCm HIP build of gufo";
#endif
}

MusicService::~MusicService() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  changed_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

void MusicService::ExpireLocked() {
  const std::int64_t now = Now();
  std::vector<std::pair<std::int64_t, std::string>> finished;
  for (const auto& [id, job] : jobs_)
    if (job->done)
      finished.emplace_back(job->completed_at, id);
  std::sort(finished.begin(), finished.end());
  const std::size_t surplus =
      finished.size() > kKeptJobs ? finished.size() - kKeptJobs : 0;
  for (std::size_t i = 0; i < finished.size(); ++i)
    if (i < surplus || now - finished[i].first > 3600)
      jobs_.erase(finished[i].second);
}

void MusicService::Worker() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_)
        return;
      job = queue_.front();
      queue_.pop_front();
      job->status = "in_progress";
    }
    std::vector<float> audio;
    std::string error;
    ace::MusicTelemetry telemetry;
#if defined(ENGINE_ENABLE_HIP)
    ace::MusicPreviewSink sink;
    sink.every = job->preview_every;
    sink.wanted = [this, job] {
      std::lock_guard lock(mutex_);
      return job->preview_always ||
             (job->preview_watched &&
              std::chrono::steady_clock::now() - job->preview_watched_at <=
                  kPreviewWatchWindow);
    };
    sink.publish = [this, job](ace::MusicPreview preview) {
      Logger::Info("music",
                   "job=" + job->id + " preview step=" +
                       std::to_string(preview.step) + "/" +
                       std::to_string(preview.steps) + " window=" +
                       std::to_string(preview.start_seconds) + "s+" +
                       std::to_string(preview.window_seconds) + "s bytes=" +
                       std::to_string(preview.wav.size()) + " decode_ms=" +
                       std::to_string(std::lround(preview.decode_ms)));
      std::lock_guard lock(mutex_);
      job->preview = std::move(preview);
      job->preview_at = Now();
    };
    const bool ok = pipeline_->Generate(
        job->request, &audio, &telemetry,
        [this, job](double fraction, std::string_view) {
          std::lock_guard lock(mutex_);
          job->progress = fraction;
          return !stop_;
        },
        &error, &sink);
#else
    // Not reached: without HIP the service never becomes ready.
    const bool ok = false;
#endif
    std::string output;
    bool encoded = ok;
    if (ok) {
      const std::string wav = ace::EncodeStereoWav(audio, ace::kSampleRate);
      // mp3 / m4a run through FFmpeg; a missing or failing FFmpeg fails the
      // job (HTTP 500 on the content read) and keeps the WAV on disk.
      encoded = EncodeMusic(wav, job->format, &output, &error);
      if (!encoded)
        Logger::Error("music", "job=" + job->id + " format=" +
                                   job->format.name + " " + error);
    }
    {
      std::lock_guard lock(mutex_);
      job->telemetry = telemetry;
      job->status = encoded ? "completed" : "failed";
      job->error = error;
      job->output = std::move(output);
      job->progress = 1.0;
      job->completed_at = Now();
      job->done = true;
    }
    changed_.notify_all();
  }
}

HttpResponse MusicService::Handle(const HttpRequest& request) {
  if (request.path == kMusicApiPath) {
    if (request.method != "POST")
      return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
    return Submit(request);
  }
  if (request.method != "GET")
    return Error(405, "Method Not Allowed", "use GET", "method_not_allowed");
  std::string rest(request.path.substr(kMusicApiPath.size() + 1));
  bool content = false;
  if (const auto slash = rest.find('/'); slash != std::string::npos) {
    const std::string verb = rest.substr(slash);
    rest.resize(slash);
    if (verb == "/preview")
      return Preview(rest);
    if (verb != "/content")
      return Error(404, "Not Found", "unknown music path", "not_found");
    content = true;
  }
  return Status(rest, content);
}

HttpResponse MusicService::Submit(const HttpRequest& request) {
  json::Value body;
  try {
    // Tolerate a UTF-8 byte order mark (Windows PowerShell writes one).
    const std::string_view text = request.body;
    body = json::parse(std::string(
        text.substr(0, 3) == "\xEF\xBB\xBF" ? text.substr(3) : text));
  } catch (const std::exception&) {
    return Error(400, "Bad Request", "request body must be valid JSON",
                 "parse_error");
  }
  if (!body.is_object())
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_request");
  auto job = std::make_shared<Job>();
  ace::MusicRequest& r = job->request;
  r.caption = body.member_str("prompt");
  if (r.caption.empty())
    r.caption = body.member_str("tags", body.member_str("caption"));
  if (r.caption.empty())
    return Error(400, "Bad Request", "prompt (style tags) is required",
                 "missing_prompt");
  r.lyrics = body.member_str("lyrics");
  r.language = body.member_str("language", "en");
  r.bpm = Text(body, "bpm");
  r.keyscale = body.member_str("keyscale", body.member_str("key"));
  r.timesignature = Text(body, "timesignature");
  if (r.timesignature.empty())
    r.timesignature = Text(body, "time_signature");
  if (r.timesignature.size() > 2 &&
      r.timesignature.substr(r.timesignature.size() - 2) == "/4")
    r.timesignature.resize(r.timesignature.size() - 2);
  r.duration = body.member_double("duration", 30.0);
  if (const json::Value* seed = body.find("seed");
      seed != nullptr && seed->is_number())
    r.seed = static_cast<std::uint64_t>(seed->as_double());
  else
    r.seed = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  r.steps = static_cast<int>(body.member_double("steps", 8));
  if (const json::Value* codes = body.find("audio_codes");
      codes != nullptr && codes->is_bool())
    r.audio_codes = codes->as_bool();
  r.lm_cfg = static_cast<float>(body.member_double("lm_cfg", r.lm_cfg));
  r.lm_temperature = static_cast<float>(
      body.member_double("lm_temperature", r.lm_temperature));
  r.lm_top_p = static_cast<float>(body.member_double("lm_top_p", r.lm_top_p));
  {
    std::string message, code;
    if (!ParseMusicOutputFormat(body, &job->format, &message, &code))
      return Error(400, "Bad Request", std::move(message), std::move(code));
  }
  if (std::string message =
          ParseMusicPreview(body, &job->preview_always, &job->preview_every);
      !message.empty())
    return Error(400, "Bad Request", std::move(message), "invalid_preview");
  if (!(r.duration >= 1.0 && r.duration <= 600.0))
    return Error(400, "Bad Request", "duration must be 1..600 seconds",
                 "invalid_duration");
  if (r.steps < 1 || r.steps > 50)
    return Error(400, "Bad Request", "steps must be 1..50", "invalid_steps");
  const json::Value* async_value = body.find("async");
  const bool async = async_value != nullptr && async_value->as_bool();
  {
    std::lock_guard lock(mutex_);
    ExpireLocked();
    if (queue_.size() >= kQueueCapacity)
      return Error(429, "Too Many Requests", "music queue is full",
                   "queue_full");
    job->id = "music_" + std::to_string(next_id_++);
    job->created_at = Now();
    jobs_[job->id] = job;
    queue_.push_back(job);
  }
  changed_.notify_all();
  if (async)
    return Status(job->id, false);
  {
    std::unique_lock lock(mutex_);
    while (!job->done && !(request.is_cancelled && request.is_cancelled()))
      changed_.wait_for(lock, std::chrono::seconds(1));
    if (!job->done)
      return Error(499, "Client Closed Request", "request cancelled",
                   "cancelled");
  }
  return Status(job->id, true);
}

HttpResponse MusicService::Status(const std::string& id, bool content) {
  std::lock_guard lock(mutex_);
  const auto it = jobs_.find(id);
  if (it == jobs_.end())
    return Error(404, "Not Found", "unknown music job " + id, "not_found");
  const Job& job = *it->second;
  if (content) {
    if (job.status == "failed")
      return Error(500, "Internal Server Error", job.error,
                   "generation_failed");
    if (!job.done)
      return Error(409, "Conflict", "music job is not finished", "not_ready");
    HttpResponse response;
    response.body = job.output;
    response.headers = {
        {"Content-Type", job.format.content_type},
        {"Content-Disposition",
         "attachment; filename=\"" + job.format.filename + "\""},
        {"X-Gufo-Music-Id", job.id},
        {"X-Gufo-Seed", std::to_string(job.request.seed)},
        {"X-Gufo-Generation-Seconds",
         std::to_string(job.telemetry.lm_seconds + job.telemetry.text_seconds +
                        job.telemetry.condition_seconds +
                        job.telemetry.diffusion_seconds +
                        job.telemetry.vae_seconds)}};
    return response;
  }
  json::Value body = json::Value::object();
  body["id"] = job.id;
  body["object"] = "audio.music";
  body["model"] = model_id_;
  body["status"] = job.status;
  body["progress"] = static_cast<double>(std::lround(job.progress * 100));
  body["created_at"] = static_cast<double>(job.created_at);
  if (job.done)
    body["completed_at"] = static_cast<double>(job.completed_at);
  body["duration"] = job.request.duration;
  body["seed"] = std::to_string(job.request.seed);
  body["response_format"] = job.format.name;
  if (job.format.needs_ffmpeg())
    body["bitrate_kbps"] = static_cast<double>(job.format.bitrate_kbps);
  if (!job.error.empty())
    body["error"] = job.error;
  // Live preview (relative, like a router job's preview_url): offered while
  // the job runs and, once there is one, until the job is forgotten.
  if (job.status == "in_progress" || !job.preview.wav.empty())
    body["preview_url"] = "/v1/audio/music/" + job.id + "/preview";
  else
    body["preview_url"] = nullptr;
  if (!job.preview.wav.empty()) {
    body["preview_step"] = static_cast<double>(job.preview.step);
    body["preview_steps"] = static_cast<double>(job.preview.steps);
    body["preview_seconds"] = job.preview.start_seconds;
    body["preview_window"] = job.preview.window_seconds;
    body["preview_at"] = static_cast<double>(job.preview_at);
  } else {
    body["preview_step"] = nullptr;
  }
  return Json(job.done || job.status != "queued" ? 200 : 202,
              job.done ? "OK" : "Accepted", body);
}

HttpResponse MusicService::Preview(const std::string& id) {
  std::lock_guard lock(mutex_);
  const auto it = jobs_.find(id);
  if (it == jobs_.end())
    return Error(404, "Not Found", "unknown music job " + id, "not_found");
  Job& job = *it->second;
  // A fetch counts as watching: the worker renders the next previews.
  job.preview_watched = true;
  job.preview_watched_at = std::chrono::steady_clock::now();
  HttpResponse response;
  response.headers = {
      {"Cache-Control", "no-store"},
      {"X-Gufo-Music-Id", job.id},
      {"X-Gufo-Preview-State",
       job.done ? (job.status == "failed" ? "failed" : "done")
                : (job.status == "queued" ? "pending" : "running")}};
  if (job.preview.wav.empty()) {
    response.status = 204;
    response.reason = "No Content";
    return response;
  }
  response.status = 200;
  response.reason = "OK";
  response.body = job.preview.wav;
  response.headers.emplace_back("Content-Type", "audio/wav");
  response.headers.emplace_back("X-Gufo-Preview-Step",
                                std::to_string(job.preview.step));
  response.headers.emplace_back("X-Gufo-Preview-Steps",
                                std::to_string(job.preview.steps));
  response.headers.emplace_back("X-Gufo-Preview-Seconds",
                                std::to_string(job.preview.start_seconds));
  response.headers.emplace_back("X-Gufo-Preview-Window",
                                std::to_string(job.preview.window_seconds));
  return response;
}

}  // namespace gufo::server
