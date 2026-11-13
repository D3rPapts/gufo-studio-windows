// gufo serve objekt: job service and HTTP handling (see objekt_service.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/cli/serve/media/objekt_service.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <thread>

#include "src/core/json.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace gufo::server {
namespace {

using Clock = std::chrono::system_clock;

HttpResponse Error(int status, std::string reason, std::string message, std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return {.status = status, .reason = std::move(reason), .body = root.dump(), .headers = {}, .streaming_body = {}};
}

// A JSON number read as an int only inside [lo, hi]: the double is checked
// first, because converting an out-of-range double (e.g. 1e300) to int is
// undefined behaviour that a range check after the cast cannot catch.
bool IntInRange(const json::Value& body, const char* name, double fallback, double lo, double hi, int* out) {
  const double value = body.member_double(name, fallback);
  if (!std::isfinite(value) || value < lo || value > hi) return false;
  *out = static_cast<int>(value);
  return true;
}

std::optional<std::vector<std::uint8_t>> DecodeBase64(std::string_view text) {
  std::vector<std::uint8_t> out;
  out.reserve(text.size() * 3 / 4);
  std::uint32_t buffer = 0;
  int bits = 0;
  for (char c : text) {
    int v;
    if (c >= 'A' && c <= 'Z') v = c - 'A';
    else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
    else if (c >= '0' && c <= '9') v = c - '0' + 52;
    else if (c == '+' || c == '-') v = 62;
    else if (c == '/' || c == '_') v = 63;
    else if (c == '=') break;
    else if (c == '\r' || c == '\n' || c == ' ') continue;
    else return std::nullopt;
    buffer = (buffer << 6U) | static_cast<std::uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<std::uint8_t>((buffer >> bits) & 0xFFU));
    }
  }
  return out;
}

std::string RandomId() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "obj_";
  for (int i = 0; i < 24; ++i) id.push_back(kHex[rng() % 16]);
  return id;
}

bool IsObjektId(std::string_view id) {
  if (id.size() != 28 || !id.starts_with("obj_")) return false;
  return std::all_of(id.begin() + 4, id.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

long long UnixNow() { return static_cast<long long>(Clock::to_time_t(Clock::now())); }

enum class Status { kQueued, kInProgress, kCompleted, kFailed, kCancelled };

std::string_view ToString(Status s) {
  switch (s) {
    case Status::kQueued: return "queued";
    case Status::kInProgress: return "in_progress";
    case Status::kCompleted: return "completed";
    case Status::kFailed: return "failed";
    case Status::kCancelled: return "cancelled";
  }
  return "unknown";
}

struct Job {
  std::string id;
  Status status = Status::kQueued;
  int progress = 0;
  std::string phase;
  long long created_at = 0, completed_at = 0, expires_at = 0;
  std::string model;
  std::shared_ptr<triposg::ObjektRunner> runner;   // the model's runner
  triposg::ObjektRequest request;  // rgba is cleared after the run
  triposg::ObjektResult result;    // glb/preview cleared after writing
  std::string error_code, error_message;
  std::filesystem::path directory;
  std::atomic<bool> cancel{false};
  bool deleted = false;
};

bool WriteAll(const std::filesystem::path& path, const void* data, std::size_t bytes, std::string* error) {
  const std::filesystem::path partial = path.string() + ".partial";
  {
    std::ofstream out(partial, std::ios::binary);
    if (!out) {
      if (error) *error = "cannot write " + path.string();
      return false;
    }
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
      if (error) *error = "write failed for " + path.string();
      return false;
    }
  }
  std::error_code ec;
  std::filesystem::rename(partial, path, ec);
  if (ec) {
    if (error) *error = "rename failed for " + path.string() + ": " + ec.message();
    return false;
  }
  return true;
}

int ProgressPercent(const std::string& phase, int completed, int total) {
  const double f = total > 0 ? std::clamp(static_cast<double>(completed) / total, 0.0, 1.0) : 0.0;
  double lo = 0, hi = 100;
  if (phase == "encode") { lo = 1; hi = 4; }
  else if (phase == "denoise") { lo = 4; hi = 55; }
  else if (phase == "decode") { lo = 55; hi = 90; }
  else if (phase == "mesh") { lo = 90; hi = 97; }
  else if (phase == "export") { lo = 97; hi = 99; }
  return static_cast<int>(lo + (hi - lo) * f);
}

}  // namespace

struct ObjektService::Impl {
  explicit Impl(ObjektServiceOptions o) : options(std::move(o)) {
    if (!options.runner) {
      init_error = "no TripoSG runner";
      return;
    }
    std::error_code ec;
    if (options.storage_root.empty()) options.storage_root = std::filesystem::temp_directory_path(ec) / "gufo-objekt";
    std::filesystem::create_directories(options.storage_root, ec);
    if (ec) {
      init_error = "cannot create " + options.storage_root.string() + ": " + ec.message();
      return;
    }
    Recover();
    worker = std::jthread([this](std::stop_token st) { Run(st); });
  }
  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      for (auto& [id, job] : jobs) job->cancel.store(true);
    }
    wake.notify_all();
  }

  ObjektServiceOptions options;
  std::string init_error;
  std::mutex mutex;
  std::condition_variable wake;
  std::deque<std::shared_ptr<Job>> queue;
  std::map<std::string, std::shared_ptr<Job>> jobs;
  std::shared_ptr<Job> active;
  bool stopping = false;
  std::jthread worker;

  json::Value Snapshot(const Job& j) const {
    json::Value root = json::Value::object();
    root["id"] = j.id;
    root["object"] = "3d.generation";
    root["schema"] = std::string(kObjektApiSchema);
    root["model"] = j.model;
    root["status"] = std::string(ToString(j.status));
    root["progress"] = j.progress;
    root["phase"] = j.phase;
    root["steps"] = j.request.steps;
    root["guidance"] = static_cast<double>(j.request.guidance);
    root["seed"] = static_cast<unsigned long long>(j.request.seed);
    root["resolution"] = j.request.resolution;
    root["max_faces"] = j.request.max_faces;
    root["texture"] = j.request.texture;
    root["texture_size"] = j.request.texture_size;
    root["created_at"] = j.created_at;
    root["completed_at"] = j.completed_at > 0 ? json::Value(j.completed_at) : json::Value();
    root["expires_at"] = j.expires_at > 0 ? json::Value(j.expires_at) : json::Value();
    if (j.error_code.empty()) {
      root["error"] = json::Value();
    } else {
      json::Value e = json::Value::object();
      e["code"] = j.error_code;
      e["message"] = j.error_message;
      root["error"] = std::move(e);
    }
    json::Value data = json::Value::array();
    if (j.status == Status::kCompleted) {
      json::Value item = json::Value::object();
      item["format"] = "glb";
      item["url"] = std::string(kObjektApiPath) + "/" + j.id + "/content";
      item["preview_url"] = std::string(kObjektApiPath) + "/" + j.id + "/preview";
      item["vertices"] = static_cast<unsigned long long>(j.result.vertices);
      item["faces"] = static_cast<unsigned long long>(j.result.faces);
      item["faces_before_decimation"] = static_cast<unsigned long long>(j.result.faces_before_decimation);
      item["closed"] = j.result.closed;
      item["volume"] = j.result.volume;
      json::Value bbox = json::Value::object(), bmin = json::Value::array(), bmax = json::Value::array();
      for (int a = 0; a < 3; ++a) { bmin.push_back(static_cast<double>(j.result.bbox_min[a])); bmax.push_back(static_cast<double>(j.result.bbox_max[a])); }
      bbox["min"] = bmin;
      bbox["max"] = bmax;
      item["bbox"] = bbox;
      json::Value t = json::Value::object();
      t["encode"] = j.result.seconds_encode;
      t["denoise"] = j.result.seconds_denoise;
      t["decode"] = j.result.seconds_decode;
      t["mesh"] = j.result.seconds_mesh;
      item["seconds"] = t;
      data.push_back(std::move(item));
    }
    root["data"] = data;
    return root;
  }

  void PersistLocked(const Job& j) {
    std::error_code ec;
    std::filesystem::create_directories(j.directory, ec);
    const std::string text = Snapshot(j).dump();
    std::string error;
    (void)WriteAll(j.directory / "job.json", text.data(), text.size(), &error);
  }

  void Recover() {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(options.storage_root, ec)) {
      if (!entry.is_directory()) continue;
      const std::string id = entry.path().filename().string();
      if (!IsObjektId(id)) continue;
      std::ifstream in(entry.path() / "job.json", std::ios::binary);
      if (!in) continue;
      std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      try {
        const json::Value v = json::parse(text);
        if (v.member_str("status") != "completed") {
          std::filesystem::remove_all(entry.path(), ec);
          continue;
        }
        auto job = std::make_shared<Job>();
        job->id = id;
        job->status = Status::kCompleted;
        job->progress = 100;
        job->model = v.member_str("model");
        job->created_at = static_cast<long long>(v.member_double("created_at"));
        job->completed_at = static_cast<long long>(v.member_double("completed_at"));
        job->expires_at = static_cast<long long>(v.member_double("expires_at"));
        job->request.steps = static_cast<int>(v.member_double("steps", 50));
        job->request.guidance = static_cast<float>(v.member_double("guidance", 7));
        job->request.seed = static_cast<std::uint64_t>(v.member_double("seed", 42));
        job->request.resolution = static_cast<int>(v.member_double("resolution", 512));
        job->request.max_faces = static_cast<int>(v.member_double("max_faces", 200000));
        job->request.texture_size = static_cast<int>(v.member_double("texture_size", 1024));
        if (const json::Value* data = v.find("data"); data && data->is_array() && !data->empty()) {
          const json::Value& item = data->items()[0];
          job->result.vertices = item.member_size("vertices");
          job->result.faces = item.member_size("faces");
          job->result.faces_before_decimation = item.member_size("faces_before_decimation");
          job->result.closed = item.find("closed") && item.find("closed")->as_bool();
          job->result.volume = item.member_double("volume");
          if (const json::Value* bbox = item.find("bbox"); bbox && bbox->find("min") && bbox->find("max"))
            for (int a = 0; a < 3; ++a) {
              job->result.bbox_min[a] = static_cast<float>(bbox->find("min")->items()[static_cast<std::size_t>(a)].as_double());
              job->result.bbox_max[a] = static_cast<float>(bbox->find("max")->items()[static_cast<std::size_t>(a)].as_double());
            }
        }
        job->directory = entry.path();
        jobs.emplace(id, std::move(job));
      } catch (const std::exception&) {
        std::filesystem::remove_all(entry.path(), ec);
      }
    }
  }

  void SweepExpiredLocked() {
    const long long now = UnixNow();
    for (auto it = jobs.begin(); it != jobs.end();) {
      const auto& job = it->second;
      const bool finished = job->status == Status::kCompleted || job->status == Status::kFailed || job->status == Status::kCancelled;
      if (finished && job->expires_at > 0 && job->expires_at <= now && job != active) {
        std::error_code ec;
        std::filesystem::remove_all(job->directory, ec);
        it = jobs.erase(it);
      } else {
        ++it;
      }
    }
  }

  void Run(std::stop_token st) {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] { return stopping || st.stop_requested() || !queue.empty(); });
        if (stopping || st.stop_requested()) return;
        job = queue.front();
        queue.pop_front();
        active = job;
        job->status = Status::kInProgress;
        job->progress = 1;
        job->phase = "encode";
        PersistLocked(*job);
      }
      triposg::ObjektResult result;
      std::string error;
      const auto progress = [&](const std::string& phase, int completed, int total) {
        const std::lock_guard<std::mutex> lock(mutex);
        job->phase = phase;
        job->progress = std::clamp(ProgressPercent(phase, completed, total), 1, 99);
      };
      const bool ok = job->runner->Run(job->request, progress, &job->cancel, &result, &error);
      std::string write_error;
      bool written = false;
      // A job cancelled or deleted while it ran writes no result files.
      if (ok && !job->cancel.load()) {
        std::error_code ec;
        std::filesystem::create_directories(job->directory, ec);
        written = WriteAll(job->directory / "model.glb", result.glb.data(), result.glb.size(), &write_error);
        if (written && !result.preview_png.empty())
          (void)WriteAll(job->directory / "preview.png", result.preview_png.data(), result.preview_png.size(), &write_error);
      }
      {
        const std::lock_guard<std::mutex> lock(mutex);
        job->request.rgba.clear();
        job->request.rgba.shrink_to_fit();
        job->completed_at = UnixNow();
        job->expires_at = job->completed_at + static_cast<long long>(options.artifact_ttl.count());
        if (ok && written) {
          result.glb.clear();
          result.preview_png.clear();
          job->result = result;
          job->status = Status::kCompleted;
          job->progress = 100;
          job->phase = "done";
        } else if (job->cancel.load()) {
          job->status = Status::kCancelled;
          job->error_code = "cancelled";
          job->error_message = "the job was cancelled";
        } else {
          job->status = Status::kFailed;
          job->error_code = ok ? "io_error" : "generation_failed";
          job->error_message = ok ? write_error : error;
        }
        if (job->deleted) {
          // DELETE while it ran: the job is already gone from `jobs`, so no
          // sweep would ever remove its directory.
          std::error_code ec;
          std::filesystem::remove_all(job->directory, ec);
        } else {
          PersistLocked(*job);
        }
        active.reset();
      }
    }
  }
};

ObjektService::ObjektService(ObjektServiceOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
ObjektService::~ObjektService() = default;
bool ObjektService::ready() const noexcept { return impl_->init_error.empty() && impl_->options.runner != nullptr; }
std::string ObjektService::model_id() const { return impl_->options.model_id; }
std::vector<std::string> ObjektService::model_ids() const {
  std::vector<std::string> ids{impl_->options.model_id};
  for (const auto& [id, runner] : impl_->options.runners)
    if (id != impl_->options.model_id && runner) ids.push_back(id);
  return ids;
}
std::string ObjektService::capability_note(const std::string& id) const {
  if (id == impl_->options.model_id && impl_->options.runner) return impl_->options.runner->capability_note();
  const auto it = impl_->options.runners.find(id);
  return it != impl_->options.runners.end() && it->second ? it->second->capability_note() : "";
}
std::string ObjektService::initialization_error() const { return impl_->init_error; }
std::string ObjektService::attribution(const std::string& id) const {
  if (id == impl_->options.model_id && impl_->options.runner) return impl_->options.runner->attribution();
  const auto it = impl_->options.runners.find(id);
  return it != impl_->options.runners.end() && it->second ? it->second->attribution() : "";
}
std::size_t ObjektService::weight_bytes() const {
  std::size_t total = impl_->options.runner ? impl_->options.runner->weight_bytes() : 0;
  for (const auto& [id, runner] : impl_->options.runners)
    if (runner && runner != impl_->options.runner) total += runner->weight_bytes();
  return total;
}

bool IsObjektApiPath(std::string_view path) noexcept {
  return path == kObjektApiPath || path.starts_with(std::string(kObjektApiPath) + "/");
}

bool ObjektService::HandlesPath(std::string_view path) const { return IsObjektApiPath(path); }

void ObjektService::Describe(json::Value* entry, bool ready) const {
  if (ready) {
    // /ready named the capability in 1.4.23.
    (*entry)["capability"] = std::string(capability());
    return;
  }
  // /v1/models, one entry per model_ids() item (HttpServer set "id" first):
  // owned_by names the model, "note" its capability (shape / pbr), and the
  // attribution its licence asks for.
  const std::string id = entry->member_str("id", model_id());
  (*entry)["owned_by"] = "operator-supplied-" + id;
  (*entry)["note"] = capability_note(id);
  if (const std::string credit = attribution(id); !credit.empty()) (*entry)["attribution"] = credit;
}

HttpResponse ObjektService::Handle(const HttpRequest& request) {
  Impl& m = *impl_;
  if (!ready()) return Error(503, "Service Unavailable", "3D service is not ready", "objekt_service_unavailable");
  // Path: /v1/3d/generations[/<id>[/content|/preview]]
  std::string id, sub;
  if (request.path.size() > kObjektApiPath.size() + 1) {
    std::string rest = request.path.substr(kObjektApiPath.size() + 1);
    const std::size_t slash = rest.find('/');
    id = rest.substr(0, slash);
    if (slash != std::string::npos) sub = rest.substr(slash + 1);
    if (!IsObjektId(id)) return Error(404, "Not Found", "unknown 3D generation", "objekt_not_found");
  }
  if (request.method == "POST" && id.empty()) {
    json::Value body;
    try {
      body = json::parse(request.body);
    } catch (const std::exception&) {
      return Error(400, "Bad Request", "request body must be valid JSON", "parse_error");
    }
    if (!body.is_object()) return Error(400, "Bad Request", "request body must be a JSON object", "invalid_body");
    const json::Value* image = body.find("image");
    if (!image || !image->is_string() || image->str().empty())
      return Error(400, "Bad Request", "field 'image' (base64 PNG with alpha or data URL) is required", "missing_image");
    std::string_view text = image->str();
    if (text.starts_with("data:")) {
      const std::size_t comma = text.find(',');
      if (comma == std::string_view::npos) return Error(400, "Bad Request", "malformed data URL", "invalid_image");
      text = text.substr(comma + 1);
    }
    if (text.size() > m.options.max_image_bytes * 4 / 3 + 4) return Error(413, "Payload Too Large", "image too large", "image_too_large");
    const auto bytes = DecodeBase64(text);
    if (!bytes.has_value() || bytes->size() < 8) return Error(400, "Bad Request", "image is not valid base64", "invalid_image");
    auto job = std::make_shared<Job>();
    try {
      const models::qwen_image_21::Image decoded = models::qwen_image_21::DecodeImage(*bytes);
      job->request.rgba = decoded.rgba;
      job->request.width = decoded.width;
      job->request.height = decoded.height;
    } catch (const std::exception& e) {
      return Error(400, "Bad Request", std::string("cannot decode image: ") + e.what(), "invalid_image");
    }
    job->model = body.member_str("model", m.options.model_id);
    job->runner = job->model == m.options.model_id ? m.options.runner : nullptr;
    if (!job->runner) {
      const auto it = m.options.runners.find(job->model);
      if (it != m.options.runners.end()) job->runner = it->second;
    }
    if (!job->runner) return Error(400, "Bad Request", "unknown model '" + job->model + "'", "unknown_model");
    // Numbers are range-checked as doubles before any conversion.
    if (!IntInRange(body, "steps", job->runner->default_steps(), 1, 200, &job->request.steps)) return Error(400, "Bad Request", "steps must be 1..200", "invalid_steps");
    const double guidance = body.member_double("guidance", 7.0);
    if (!(guidance >= 0.0 && guidance <= 30.0)) return Error(400, "Bad Request", "guidance must be 0..30", "invalid_guidance");
    job->request.guidance = static_cast<float>(guidance);
    const double seed = body.member_double("seed", 42);
    if (!(seed >= 0.0 && seed < 18446744073709551616.0)) return Error(400, "Bad Request", "seed must be 0..2^64-1", "invalid_seed");
    job->request.seed = static_cast<std::uint64_t>(seed);
    if (!IntInRange(body, "resolution", 512, 256, 512, &job->request.resolution) ||
        (job->request.resolution != 256 && job->request.resolution != 512))
      return Error(400, "Bad Request", "resolution must be 256 or 512", "invalid_resolution");
    if (!IntInRange(body, "max_faces", 200000, 0, 5000000, &job->request.max_faces)) return Error(400, "Bad Request", "max_faces must be 0..5000000", "invalid_max_faces");
    if (const json::Value* p = body.find("preview"); p && p->is_bool()) job->request.preview = p->as_bool();
    if (const json::Value* p = body.find("texture"); p && p->is_bool()) job->request.texture = p->as_bool();
    if (!IntInRange(body, "texture_size", 1024, 0, 4096, &job->request.texture_size)) return Error(400, "Bad Request", "texture_size must be 0..4096", "invalid_texture_size");
    job->created_at = UnixNow();
    {
      const std::lock_guard<std::mutex> lock(m.mutex);
      m.SweepExpiredLocked();
      if (m.queue.size() >= m.options.queue_capacity)
        return Error(503, "Service Unavailable", "3D queue is full", "queue_full");
      do { job->id = RandomId(); } while (m.jobs.contains(job->id));
      job->directory = m.options.storage_root / job->id;
      m.jobs.emplace(job->id, job);
      m.queue.push_back(job);
      m.PersistLocked(*job);
    }
    m.wake.notify_one();
    const std::lock_guard<std::mutex> lock(m.mutex);
    HttpResponse response{.status = 202, .reason = "Accepted", .body = m.Snapshot(*job).dump(), .headers = {}, .streaming_body = {}};
    response.headers.emplace_back("Location", std::string(kObjektApiPath) + "/" + job->id);
    response.log_details = "job=" + job->id + " steps=" + std::to_string(job->request.steps) + " res=" + std::to_string(job->request.resolution);
    return response;
  }
  if (id.empty()) return Error(405, "Method Not Allowed", "use POST /v1/3d/generations", "method_not_allowed");
  std::shared_ptr<Job> job;
  {
    const std::lock_guard<std::mutex> lock(m.mutex);
    m.SweepExpiredLocked();
    const auto it = m.jobs.find(id);
    if (it != m.jobs.end()) job = it->second;
  }
  if (!job) return Error(404, "Not Found", "unknown 3D generation", "objekt_not_found");
  if (request.method == "GET" && sub.empty()) {
    const std::lock_guard<std::mutex> lock(m.mutex);
    return {.status = 200, .reason = "OK", .body = m.Snapshot(*job).dump(), .headers = {}, .streaming_body = {}};
  }
  if (request.method == "GET" && (sub == "content" || sub == "preview")) {
    {
      const std::lock_guard<std::mutex> lock(m.mutex);
      if (job->status != Status::kCompleted) return Error(409, "Conflict", "3D generation is not finished", "objekt_not_ready");
    }
    const std::filesystem::path file = job->directory / (sub == "content" ? "model.glb" : "preview.png");
    std::ifstream in(file, std::ios::binary);
    if (!in) return Error(404, "Not Found", "result file is missing", "objekt_file_missing");
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    HttpResponse response{.status = 200, .reason = "OK", .body = std::move(data), .headers = {}, .streaming_body = {}};
    response.headers.emplace_back("Content-Type", sub == "content" ? "model/gltf-binary" : "image/png");
    response.headers.emplace_back("Content-Disposition", sub == "content" ? "attachment; filename=\"" + id + ".glb\"" : "inline; filename=\"" + id + ".png\"");
    response.log_details = "job=" + id + " file=" + sub;
    return response;
  }
  if (request.method == "DELETE" && sub.empty()) {
    bool active_now = false;
    {
      const std::lock_guard<std::mutex> lock(m.mutex);
      job->deleted = true;
      job->cancel.store(true);
      active_now = (m.active == job);
      m.queue.erase(std::remove(m.queue.begin(), m.queue.end(), job), m.queue.end());
      m.jobs.erase(id);
      if (!active_now) {
        std::error_code ec;
        std::filesystem::remove_all(job->directory, ec);
      }
    }
    json::Value body = json::Value::object();
    body["id"] = id;
    body["object"] = "3d.generation.deleted";
    body["deleted"] = true;
    return {.status = 200, .reason = "OK", .body = body.dump(), .headers = {}, .streaming_body = {}};
  }
  return Error(405, "Method Not Allowed", "unsupported method for this path", "method_not_allowed");
}

}  // namespace gufo::server
