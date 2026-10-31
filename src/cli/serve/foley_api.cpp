// MMAudio video / text to audio behind /v1/audio/foley (see foley_api.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/foley_api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <utility>
#include <vector>

#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/models/mmaudio/preprocess.hpp"
#include "src/models/mmaudio/video_input.hpp"

namespace gufo::server {
namespace {

namespace mm = models::mmaudio;

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

bool Flag(const json::Value& body, const char* key, bool fallback) {
  const json::Value* value = body.find(key);
  return value != nullptr && value->is_bool() ? value->as_bool() : fallback;
}

}  // namespace

struct FoleyService::Job {
  std::string id;
  mm::FoleyRequest request;
  mm::FrameOptions frames;
  std::vector<std::uint8_t> video;  // encoded, decoded by the worker
  std::vector<std::uint8_t> image;
  std::string status{"queued"};
  double progress{0};
  std::string error;
  std::string wav;
  mm::FoleyTelemetry telemetry;
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  bool done{false};
};

bool IsFoleyApiPath(std::string_view path) noexcept {
  return path == kFoleyApiPath ||
         (path.size() > kFoleyApiPath.size() &&
          path.substr(0, kFoleyApiPath.size()) == kFoleyApiPath &&
          path[kFoleyApiPath.size()] == '/');
}

FoleyService::FoleyService(const std::filesystem::path& model_root,
                           std::string model_id)
    : model_id_(std::move(model_id)) {
  mm::ModelPaths paths;
  if (!mm::ModelPaths::FromRoot(model_root, &paths, &init_error_))
    return;
  pipeline_ = mm::FoleyPipeline::Load(paths, &init_error_);
  if (pipeline_)
    worker_ = std::thread([this] { Worker(); });
}

FoleyService::~FoleyService() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  changed_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

void FoleyService::ExpireLocked() {
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

void FoleyService::Worker() {
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
    std::string error;
    bool ok = true;
    mm::FoleyRequest request = job->request;
    // Media -> preprocessed frames.
    if (!job->video.empty()) {
      std::error_code ec;
      const auto path = std::filesystem::temp_directory_path(ec) /
                        ("gufo-foley-" + job->id + ".mp4");
      {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(job->video.data()),
                  static_cast<std::streamsize>(job->video.size()));
        ok = static_cast<bool>(out);
      }
      if (!ok)
        error = "cannot write the video to a temporary file";
      else
        ok = mm::LoadVideoFrames(path, job->frames, &request, &error);
      std::filesystem::remove(path, ec);
    } else if (!job->image.empty()) {
      try {
        const core::Image image = core::DecodeImage(job->image);
        mm::LoadImageFrame(image.pixels, static_cast<int>(image.width),
                           static_cast<int>(image.height), &request);
      } catch (const std::exception& e) {
        ok = false;
        error = std::string("image: ") + e.what();
      }
    }
    std::vector<float> audio;
    mm::FoleyTelemetry telemetry;
    if (ok)
      ok = pipeline_->Generate(
          request, &audio, &telemetry,
          [this, job](double fraction, std::string_view) {
            std::lock_guard lock(mutex_);
            job->progress = fraction;
            return !stop_;
          },
          &error);
    std::string wav;
    if (ok)
      wav = mm::EncodeMonoWav(audio, mm::kSampleRate);
    {
      std::lock_guard lock(mutex_);
      job->telemetry = telemetry;
      job->request.duration = request.duration;
      job->status = ok ? "completed" : "failed";
      job->error = error;
      job->wav = std::move(wav);
      job->video.clear();
      job->video.shrink_to_fit();
      job->image.clear();
      job->progress = 1.0;
      job->completed_at = Now();
      job->done = true;
    }
    changed_.notify_all();
  }
}

HttpResponse FoleyService::Handle(const HttpRequest& request) {
  if (request.path == kFoleyApiPath) {
    if (request.method != "POST")
      return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
    return Submit(request);
  }
  if (request.method != "GET")
    return Error(405, "Method Not Allowed", "use GET", "method_not_allowed");
  std::string rest(request.path.substr(kFoleyApiPath.size() + 1));
  bool content = false;
  if (const auto slash = rest.find('/'); slash != std::string::npos) {
    if (rest.substr(slash) != "/content")
      return Error(404, "Not Found", "unknown foley path", "not_found");
    content = true;
    rest.resize(slash);
  }
  return Status(rest, content);
}

HttpResponse FoleyService::Submit(const HttpRequest& request) {
  json::Value body;
  try {
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
  mm::FoleyRequest& r = job->request;
  r.prompt = body.member_str("prompt");
  r.negative_prompt = body.member_str("negative_prompt");
  r.duration = body.member_double("duration", 8.0);
  r.steps = static_cast<int>(body.member_double("steps", 25));
  r.cfg_strength = static_cast<float>(
      body.member_double("cfg_strength", body.member_double("cfg", 4.5)));
  if (const json::Value* seed = body.find("seed");
      seed != nullptr && seed->is_number())
    r.seed = static_cast<std::uint64_t>(seed->as_double());
  else
    r.seed = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
  r.mask_away_clip = Flag(body, "mask_away_clip", false);
  const std::string mode = body.member_str("frame_mode", "resample");
  if (mode != "resample" && mode != "consecutive")
    return Error(400, "Bad Request",
                 "frame_mode must be resample or consecutive",
                 "invalid_frame_mode");
  job->frames.consecutive = mode == "consecutive";
  job->frames.force_rate = body.member_double("force_rate", 0);
  job->frames.skip_first_frames =
      static_cast<int>(body.member_double("skip_first_frames", 0));
  job->frames.select_every_nth =
      static_cast<int>(body.member_double("select_every_nth", 1));
  job->frames.frame_load_cap =
      static_cast<int>(body.member_double("frame_load_cap", 0));
  if (body.member_str("response_format", "wav") != "wav")
    return Error(400, "Bad Request", "response_format must be wav",
                 "unsupported_format");
  if (!(r.duration >= 1.0 && r.duration <= 120.0))
    return Error(400, "Bad Request", "duration must be 1..120 seconds",
                 "invalid_duration");
  if (r.steps < 1 || r.steps > 200)
    return Error(400, "Bad Request", "steps must be 1..200", "invalid_steps");
  if (!(r.cfg_strength >= 0.0F && r.cfg_strength <= 30.0F))
    return Error(400, "Bad Request", "cfg_strength must be 0..30",
                 "invalid_cfg");
  const std::string video = body.member_str("video");
  const std::string image = body.member_str("image");
  if (!video.empty() && !image.empty())
    return Error(400, "Bad Request", "send video or image, not both",
                 "invalid_media");
  try {
    core::ImageReadBudget budget;
    budget.remaining_bytes = 512U << 20U;
    budget.deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    if (!video.empty())
      job->video = core::ReadVideoUrl(video, budget);
    else if (!image.empty())
      job->image = core::ReadImageUrl(image, budget);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", std::string("media: ") + e.what(),
                 "invalid_media");
  }
  const json::Value* async_value = body.find("async");
  const bool async = async_value != nullptr && async_value->as_bool();
  {
    std::lock_guard lock(mutex_);
    ExpireLocked();
    if (queue_.size() >= kQueueCapacity)
      return Error(429, "Too Many Requests", "foley queue is full",
                   "queue_full");
    job->id = "foley_" + std::to_string(next_id_++);
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

HttpResponse FoleyService::Status(const std::string& id, bool content) {
  std::lock_guard lock(mutex_);
  const auto it = jobs_.find(id);
  if (it == jobs_.end())
    return Error(404, "Not Found", "unknown foley job " + id, "not_found");
  const Job& job = *it->second;
  if (content) {
    if (job.status == "failed")
      return Error(500, "Internal Server Error", job.error,
                   "generation_failed");
    if (!job.done)
      return Error(409, "Conflict", "foley job is not finished", "not_ready");
    HttpResponse response;
    response.body = job.wav;
    response.headers = {
        {"Content-Type", "audio/wav"},
        {"X-Gufo-Foley-Id", job.id},
        {"X-Gufo-Seed", std::to_string(job.request.seed)},
        {"X-Gufo-Duration", std::to_string(job.request.duration)},
        {"X-Gufo-Generation-Seconds",
         std::to_string(job.telemetry.features_seconds +
                        job.telemetry.sampling_seconds +
                        job.telemetry.decode_seconds)}};
    return response;
  }
  json::Value body = json::Value::object();
  body["id"] = job.id;
  body["object"] = "audio.foley";
  body["model"] = model_id_;
  body["status"] = job.status;
  body["progress"] = static_cast<double>(std::lround(job.progress * 100));
  body["created_at"] = static_cast<double>(job.created_at);
  if (job.done)
    body["completed_at"] = static_cast<double>(job.completed_at);
  body["duration"] = job.request.duration;
  body["seed"] = std::to_string(job.request.seed);
  if (!job.error.empty())
    body["error"] = job.error;
  return Json(job.done || job.status != "queued" ? 200 : 202,
              job.done ? "OK" : "Accepted", body);
}

}  // namespace gufo::server
