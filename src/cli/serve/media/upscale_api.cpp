// Real-ESRGAN image / video upscaling behind /v1/images/upscale and
// /v1/videos/upscale (see upscale_api.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/upscale_api.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>
#include <utility>

#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/preview.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include <png.h>
#include "src/models/realesrgan/srvgg_weights.hpp"
#include "src/models/rife/video_io.hpp"

#if defined(GUFO_HAVE_ONNXRUNTIME)
#include "src/models/realesrgan/ort_upscaler.hpp"
#endif
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/realesrgan/upscaler.hpp"
#endif

namespace gufo::server {
namespace {

constexpr int kMaxOutputSide = 8192;
constexpr std::size_t kMaxVideoBytes = 512U << 20U;
constexpr std::int64_t kKeepSeconds = 3600;

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

// 8-bit RGB PNG (libpng). 1.x used Qwen-Image's RGBA encoder; on 2.0
// Qwen-Image is an extension the server must not link.
std::vector<std::uint8_t> EncodePngRgb(const std::vector<std::uint8_t>& rgb,
                                       int w, int h) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = static_cast<png_uint_32>(w);
  image.height = static_cast<png_uint_32>(h);
  image.format = PNG_FORMAT_RGB;
  png_alloc_size_t size = 0;
  if (!png_image_write_to_memory(&image, nullptr, &size, 0, rgb.data(), 0,
                                 nullptr))
    return {};
  std::vector<std::uint8_t> png(size);
  if (!png_image_write_to_memory(&image, png.data(), &size, 0, rgb.data(), 0,
                                 nullptr))
    return {};
  png.resize(size);
  return png;
}

std::string Ms(double seconds) {
  return std::to_string(std::lround(seconds * 1000.0));
}

std::int64_t Now() {
  return static_cast<std::int64_t>(std::time(nullptr));
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       since)
      .count();
}

/// Strips a UTF-8 byte order mark (Windows PowerShell writes one).
std::optional<json::Value> ParseBody(const HttpRequest& request,
                                     HttpResponse* error) {
  json::Value body;
  try {
    const std::string_view text = request.body;
    body = json::parse(
        text.substr(0, 3) == "\xEF\xBB\xBF" ? text.substr(3) : text);
  } catch (const std::exception&) {
    *error = Error(400, "Bad Request", "request body must be valid JSON",
                   "parse_error");
    return std::nullopt;
  }
  if (!body.is_object()) {
    *error = Error(400, "Bad Request", "request body must be a JSON object",
                   "invalid_request");
    return std::nullopt;
  }
  return body;
}

/// Lower-cased image format name: "png", "jpg" (also "jpeg"); "" otherwise.
std::string ImageFormat(std::string text) {
  std::ranges::transform(text, text.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  if (text == "png")
    return "png";
  if (text == "jpg" || text == "jpeg")
    return "jpg";
  return "";
}

std::filesystem::path TempFile(const std::string& name) {
  std::error_code ec;
  return std::filesystem::temp_directory_path(ec) / name;
}

bool WriteFile(const std::filesystem::path& path,
               const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

bool ReadFile(const std::filesystem::path& path, std::string* bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  bytes->assign(std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
  return true;
}

}  // namespace

struct UpscaleService::Job {
  std::string id;
  std::string kind;  // "image" | "video"
  // Request.
  int model_index{0};
  std::string model;    // public name of the resolved file
  std::string variant;  // its variant
  int scale{4};         // 2 | 4; 0 when size / width + height was given
  int out_w{0};         // 0 x 0: native x4
  int out_h{0};
  std::string fit{"crop"};
  std::string format{"png"};  // image: png | jpg
  bool binary{false};         // image: bytes instead of b64_json
  int quality{90};            // image: JPEG quality
  int crf{18};                // video: H.264 CRF
  bool keep_audio{true};      // video
  std::vector<std::uint8_t> input;  // encoded image or MP4 bytes
  // State.
  std::string status{"queued"};
  double progress{0.0};
  std::string error;
  std::string error_code;
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  bool done{false};
  std::atomic<bool> cancel{false};
  // Result.
  std::string output;  // encoded image or MP4
  int width{0};
  int height{0};
  int in_width{0};
  int in_height{0};
  int frames{0};
  int frames_done{0};
  int fps_num{0};
  int fps_den{1};
  bool audio{false};
  double queue_seconds{0};
  double body_seconds{0};
  double post_seconds{0};
  double encode_seconds{0};
  double total_seconds{0};
  std::string model_file;
};

bool IsUpscaleApiPath(std::string_view path) noexcept {
  return path == kUpscaleImagePath ||
         (path.starts_with(kUpscaleImagePath) &&
          path.size() > kUpscaleImagePath.size() &&
          path[kUpscaleImagePath.size()] == '/') ||
         path == kUpscaleVideoPath || path == "/v1/videos" ||
         path.starts_with("/v1/videos/");
}

bool UpscaleService::HandlesPath(std::string_view path) const {
  return IsUpscaleApiPath(path);
}

void UpscaleService::Describe(json::Value* entry, bool ready) const {
  (*entry)["device"] = engine_->device;
  if (ready)
    return;
  (*entry)["runtime"] = engine_->runtime;
  // The weight files behind the name: "general" | "wdn" | "anime"
  // (request field `variant`), with the architecture when known.
  const std::string name =
      entry->contains("id") && (*entry)["id"].is_string()
          ? (*entry)["id"].str()
          : model_id();
  json::Value variants = json::Value::array();
  for (const UpscaleModelInfo& file : engine_->models) {
    if (file.name != name)
      continue;
    json::Value variant = json::Value::object();
    variant["variant"] = file.variant;
    variant["file"] = file.path.filename().string();
    if (file.convolutions > 0) {
      variant["convolutions"] = file.convolutions;
      variant["scale"] = file.scale;
    }
    variants.push_back(std::move(variant));
  }
  (*entry)["variants"] = std::move(variants);
  json::Value endpoints = json::Value::array();
  endpoints.push_back(std::string(kUpscaleImagePath));
  endpoints.push_back(std::string(kUpscaleVideoPath));
  (*entry)["endpoints"] = std::move(endpoints);
}

bool ParseUpscaleSize(std::string_view text, int* w, int* h) {
  const std::size_t x = text.find_first_of("xX*");
  if (x == std::string_view::npos || x == 0 || x + 1 >= text.size())
    return false;
  int values[2] = {0, 0};
  const std::string_view parts[2] = {text.substr(0, x), text.substr(x + 1)};
  for (int i = 0; i < 2; ++i) {
    for (char c : parts[i]) {
      if (c < '0' || c > '9' || values[i] > 100000)
        return false;
      values[i] = values[i] * 10 + (c - '0');
    }
  }
  if (values[0] <= 0 || values[1] <= 0)
    return false;
  *w = values[0];
  *h = values[1];
  return true;
}

int ResolveUpscaleModel(const UpscaleEngine& engine, std::string_view model,
                        std::string_view variant, std::string* error) {
  if (engine.models.empty()) {
    *error = "no upscale model is served";
    return -1;
  }
  const std::string name =
      model.empty() ? engine.model_names.front() : std::string(model);
  std::vector<int> candidates;
  for (std::size_t i = 0; i < engine.models.size(); ++i)
    if (engine.models[i].name == name)
      candidates.push_back(static_cast<int>(i));
  if (candidates.empty()) {
    // A variant name alone ("wdn") selects its file.
    for (std::size_t i = 0; i < engine.models.size(); ++i)
      if (!engine.models[i].variant.empty() &&
          engine.models[i].variant == name)
        candidates.push_back(static_cast<int>(i));
    if (candidates.empty()) {
      *error = "model '" + name + "' is not served here";
      return -1;
    }
    const UpscaleModelInfo& only =
        engine.models[static_cast<std::size_t>(candidates.front())];
    if (!variant.empty() && only.variant != variant) {
      *error = "model '" + name + "' is the variant '" + only.variant +
               "', not '" + std::string(variant) + "'";
      return -1;
    }
    return candidates.front();
  }
  if (variant.empty())
    return candidates.front();
  for (int index : candidates)
    if (engine.models[static_cast<std::size_t>(index)].variant == variant)
      return index;
  std::string available;
  for (int index : candidates) {
    const std::string& v = engine.models[static_cast<std::size_t>(index)].variant;
    if (!v.empty())
      available += (available.empty() ? "" : ", ") + v;
  }
  *error = "variant '" + std::string(variant) + "' is not served for model '" +
           name + "'" + (available.empty() ? "" : " (" + available + ")");
  return -1;
}

std::unique_ptr<UpscaleEngine> CreateUpscaleEngine(
    const UpscaleServeOptions& options, std::string* error) {
  if (options.models.empty()) {
    *error = "--model NAME=FILE is required";
    return nullptr;
  }
  auto engine = std::make_unique<UpscaleEngine>();
  engine->device = options.device;
  for (const UpscaleModelArg& m : options.models) {
    if (std::find(engine->model_names.begin(), engine->model_names.end(),
                  m.name) == engine->model_names.end())
      engine->model_names.push_back(m.name);
    UpscaleModelInfo info;
    info.name = m.name;
    info.path = m.path;
    info.variant = m.variant.empty()
                       ? models::realesrgan::VariantFromFileName(
                             m.path.stem().string())
                       : m.variant;
    engine->models.push_back(std::move(info));
  }
  if (options.device == "npu" || options.device == "cpu") {
#if defined(GUFO_HAVE_ONNXRUNTIME)
    namespace re = models::realesrgan;
    re::OrtUpscalerOptions ort;
    (void)re::ParseOrtDevice(options.device, &ort.device);
    ort.runtime_library = options.onnxruntime;
    ort.ep_library = options.npu_ep;
    ort.cache_dir = options.npu_cache;
    ort.ep_config_file = options.npu_config;
    ort.threads = options.threads;
    std::vector<re::OrtModelSpec> specs;
    for (const UpscaleModelArg& m : options.models)
      specs.push_back({m.name, m.path, {}});
    std::shared_ptr<re::OrtUpscaler> upscaler =
        re::OrtUpscaler::Load(ort, specs, error);
    if (upscaler == nullptr)
      return nullptr;
    engine->runtime = "onnxruntime " + upscaler->runtime_version() + " (" +
                      upscaler->runtime_library().string() + ")";
    if (upscaler->device() == re::OrtDevice::kNpu)
      engine->runtime += "; VitisAI EP " +
                         (upscaler->ep_version().empty()
                              ? std::string("?")
                              : upscaler->ep_version()) +
                         " (" + upscaler->ep_library().string() + ")";
    for (const re::OrtModelInfo& info : upscaler->models())
      engine->runtime += "; " + info.name + "=" + info.path.filename().string() +
                         " " + std::to_string(info.width) + " wide, " +
                         std::to_string(info.tile_rows) + "-row passes, " +
                         "load " + Ms(info.load_seconds) + " ms";
    // ONNX bodies are chosen by name and frame width; the variant is a label.
    engine->run = [upscaler](const std::string& model, const std::string&,
                             const std::uint8_t* rgb24, int w, int h,
                             int out_w, int out_h, const std::string& fit_name,
                             UpscaleOutcome* out, std::string* err) {
      re::AreaFit fit = re::AreaFit::kCrop;
      if (!re::ParseAreaFit(fit_name, &fit)) {
        *err = "fit must be crop, pad or stretch";
        return false;
      }
      const re::OrtModelInfo* info = upscaler->Find(model, w, h);
      re::UpscaleTimings timings;
      if (!upscaler->Upscale(model, rgb24, w, h, out_w, out_h, fit,
                             &out->rgb24, &timings, err))
        return false;
      out->width = out_w == 0 ? w * 4 : out_w;
      out->height = out_h == 0 ? h * 4 : out_h;
      out->body_seconds = timings.body_seconds;
      out->post_seconds = timings.post_seconds;
      out->model_file = info != nullptr ? info->path.filename().string() : "";
      return true;
    };
    return engine;
#else
    *error = "this gufo was built without GUFO_ENABLE_ONNXRUNTIME";
    return nullptr;
#endif
  }
  if (options.device == "gpu") {
#if defined(ENGINE_ENABLE_HIP)
    namespace re = models::realesrgan;
    // The header is checked on the host first: a clear message for a wrong
    // file before the device loads anything, and the architecture for
    // /v1/models.
    for (UpscaleModelInfo& info : engine->models) {
      re::SafetensorsHeader header;
      re::SrvggArchitecture arch;
      if (!re::ReadSafetensorsHeader(info.path, &header, error) ||
          !re::DescribeSrvgg(header, &arch, error)) {
        *error = info.path.string() + ": " + *error;
        return nullptr;
      }
      info.convolutions = arch.convolutions;
      info.scale = arch.scale;
      info.parameters = arch.parameters;
    }
    auto upscalers =
        std::make_shared<std::vector<std::shared_ptr<re::Upscaler>>>();
    for (const UpscaleModelInfo& info : engine->models) {
      std::shared_ptr<re::Upscaler> u = re::Upscaler::Load(
          info.path, options.fp32 ? re::Precision::kFp32 : re::Precision::kFp16,
          error);
      if (u == nullptr)
        return nullptr;
      upscalers->push_back(std::move(u));
    }
    engine->runtime = std::string("HIP Real-ESRGAN ") +
                      (options.fp32 ? "FP32" : "FP16");
    for (const UpscaleModelInfo& info : engine->models)
      engine->runtime += "; " + info.name +
                         (info.variant.empty() ? "" : "/" + info.variant) +
                         "=" + info.path.filename().string() + " " +
                         std::to_string(info.convolutions) + " conv x" +
                         std::to_string(info.scale);
    const std::vector<UpscaleModelInfo> files = engine->models;
    engine->run = [upscalers, files](const std::string& model,
                                     const std::string& variant,
                                     const std::uint8_t* rgb24, int w, int h,
                                     int out_w, int out_h,
                                     const std::string& fit_name,
                                     UpscaleOutcome* out, std::string* err) {
      re::Fit fit = re::Fit::kCrop;
      if (!re::ParseFit(fit_name, &fit)) {
        *err = "fit must be crop, pad or stretch";
        return false;
      }
      std::size_t index = 0;
      for (std::size_t i = 0; i < files.size(); ++i)
        if (files[i].name == model &&
            (variant.empty() || files[i].variant == variant)) {
          index = i;
          break;
        }
      re::Upscaler& u = *(*upscalers)[index];
      const int ow = out_w == 0 ? w * u.scale() : out_w;
      const int oh = out_h == 0 ? h * u.scale() : out_h;
      out->rgb24.resize(static_cast<std::size_t>(ow) * oh * 3);
      const auto start = std::chrono::steady_clock::now();
      if (!u.Upscale(rgb24, w, h, ow, oh, fit, out->rgb24.data(), err))
        return false;
      out->width = ow;
      out->height = oh;
      out->body_seconds = Seconds(start);
      out->model_file = files[index].path.filename().string();
      return true;
    };
    return engine;
#else
    *error = "--device gpu needs a HIP build of gufo";
    return nullptr;
#endif
  }
  *error = "--device must be npu, gpu or cpu";
  return nullptr;
}

UpscaleService::UpscaleService(std::unique_ptr<UpscaleEngine> engine)
    : engine_(std::move(engine)) {
  if (engine_ != nullptr)
    worker_ = std::thread([this] { Worker(); });
}

UpscaleService::~UpscaleService() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
    for (const auto& [id, job] : jobs_)
      job->cancel = true;
  }
  changed_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

HttpResponse UpscaleService::Handle(const HttpRequest& request) {
  if (request.path == kUpscaleImagePath) {
    if (request.method != "POST")
      return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
    return HandleImage(request);
  }
  if (request.path == kUpscaleVideoPath) {
    if (request.method != "POST")
      return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
    return SubmitVideo(request);
  }
  if (request.path == "/v1/videos")
    return Error(405, "Method Not Allowed",
                 "this server upscales: POST /v1/videos/upscale",
                 "method_not_allowed");
  // /v1/images/upscale/{id}[/content] and /v1/videos/{id}[/content].
  std::string rest;
  if (request.path.starts_with(kUpscaleImagePath))
    rest = request.path.substr(kUpscaleImagePath.size() + 1);
  else
    rest = request.path.substr(std::string_view("/v1/videos/").size());
  bool content = false;
  if (const auto slash = rest.find('/'); slash != std::string::npos) {
    if (rest.substr(slash) != "/content")
      return Error(404, "Not Found", "unknown upscale path", "not_found");
    content = true;
    rest.resize(slash);
  }
  if (rest.empty())
    return Error(404, "Not Found", "unknown upscale path", "not_found");
  if (request.method == "DELETE" && !content)
    return CancelJob(rest);
  if (request.method != "GET")
    return Error(405, "Method Not Allowed", "use GET", "method_not_allowed");
  return JobStatus(rest, content);
}

std::optional<HttpResponse> UpscaleService::ParseCommon(
    const json::Value& body, Job* job, bool video) {
  std::string error;
  job->model_index = ResolveUpscaleModel(
      *engine_, body.member_str("model"), body.member_str("variant"), &error);
  if (job->model_index < 0)
    return Error(404, "Not Found", error, "model_not_found");
  job->model = engine_->models[static_cast<std::size_t>(job->model_index)].name;
  job->variant =
      engine_->models[static_cast<std::size_t>(job->model_index)].variant;
  int out_w = 0, out_h = 0;
  const std::string size = body.member_str("size");
  if (!size.empty()) {
    if (!ParseUpscaleSize(size, &out_w, &out_h))
      return Error(400, "Bad Request", "size must be WIDTHxHEIGHT",
                   "invalid_size");
  } else {
    out_w = static_cast<int>(body.member_double("width", 0));
    out_h = static_cast<int>(body.member_double("height", 0));
    if ((out_w == 0) != (out_h == 0) || out_w < 0 || out_h < 0)
      return Error(400, "Bad Request", "send width and height together",
                   "invalid_size");
  }
  if (out_w > kMaxOutputSide || out_h > kMaxOutputSide)
    return Error(400, "Bad Request", "output is limited to 8192x8192",
                 "invalid_size");
  if (video && out_w > 0 && (out_w % 2 != 0 || out_h % 2 != 0))
    return Error(400, "Bad Request", "video width and height must be even",
                 "invalid_size");
  job->out_w = out_w;
  job->out_h = out_h;
  if (out_w > 0) {
    job->scale = 0;
  } else {
    const json::Value* scale = body.find("scale");
    job->scale = 4;
    if (scale != nullptr) {
      double value = 0;
      if (scale->is_number())
        value = scale->as_double();
      else if (scale->is_string())
        value = scale->str() == "2" ? 2 : scale->str() == "4" ? 4 : 0;
      if (value != 2 && value != 4)
        return Error(400, "Bad Request", "scale must be 2 or 4",
                     "invalid_scale");
      job->scale = static_cast<int>(value);
    }
  }
  job->fit = body.member_str("fit", video ? "pad" : "crop");
  if (job->fit != "crop" && job->fit != "pad" && job->fit != "stretch")
    return Error(400, "Bad Request", "fit must be crop, pad or stretch",
                 "invalid_fit");
  return std::nullopt;
}

bool UpscaleService::Enqueue(const std::shared_ptr<Job>& job,
                             std::string* error) {
  {
    std::lock_guard lock(mutex_);
    ExpireLocked();
    if (queue_.size() >= kQueueCapacity) {
      *error = "upscale queue is full";
      return false;
    }
    job->created_at = Now();
    job->id = "upscale_" + std::to_string(job->created_at) + "_" +
              std::to_string(next_id_++);
    jobs_[job->id] = job;
    queue_.push_back(job);
  }
  changed_.notify_all();
  return true;
}

HttpResponse UpscaleService::HandleImage(const HttpRequest& request) {
  HttpResponse parse_error;
  const std::optional<json::Value> parsed = ParseBody(request, &parse_error);
  if (!parsed)
    return parse_error;
  const json::Value& body = *parsed;
  auto job = std::make_shared<Job>();
  job->kind = "image";
  if (auto error = ParseCommon(body, job.get(), false))
    return *error;
  // Format: `output` names the encoding (png | jpg); `response_format`
  // "b64_json" (default) wraps it in JSON, "png" / "jpg" return the bytes.
  const std::string response_format =
      body.member_str("response_format", "b64_json");
  const std::string output = body.member_str("output");
  if (response_format == "b64_json") {
    job->binary = false;
    job->format = output.empty() ? "png" : ImageFormat(output);
  } else if (!ImageFormat(response_format).empty()) {
    job->binary = true;
    job->format = ImageFormat(response_format);
    if (!output.empty() && ImageFormat(output) != job->format)
      return Error(400, "Bad Request",
                   "output and response_format name different formats",
                   "unsupported_format");
  } else {
    return Error(400, "Bad Request",
                 "response_format must be b64_json, png or jpg",
                 "unsupported_format");
  }
  if (job->format.empty())
    return Error(400, "Bad Request", "output must be png or jpg",
                 "unsupported_format");
  job->quality = static_cast<int>(body.member_double("quality", 90));
  if (job->quality < 1 || job->quality > 100)
    return Error(400, "Bad Request", "quality must be 1..100",
                 "invalid_quality");
  const std::string image_url = body.member_str("image");
  if (image_url.empty())
    return Error(400, "Bad Request", "image is required", "invalid_media");
  try {
    core::ImageReadBudget budget;
    budget.deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    job->input = core::ReadImageUrl(image_url, budget);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", std::string("image: ") + e.what(),
                 "invalid_media");
  }
  const json::Value* async_value = body.find("async");
  if (async_value != nullptr && async_value->as_bool()) {
    std::string error;
    if (!Enqueue(job, &error))
      return Error(429, "Too Many Requests", error, "queue_full");
    return JobStatus(job->id, false);
  }
  job->created_at = Now();
  if (!RunImage(job.get(), true, &request)) {
    if (job->error_code == "cancelled")
      return Error(499, "Client Closed Request", job->error, "cancelled");
    if (job->error_code == "queue_full")
      return Error(429, "Too Many Requests", job->error, "queue_full");
    return Error(400, "Bad Request", job->error,
                 job->error_code.empty() ? "upscale_failed" : job->error_code);
  }
  return ImageResponse(*job);
}

bool UpscaleService::RunImage(Job* job, bool sync, const HttpRequest* request) {
  const auto started = std::chrono::steady_clock::now();
  core::Image image;
  try {
    image = core::DecodeImage(job->input);
  } catch (const std::exception& e) {
    job->error = std::string("image: ") + e.what();
    job->error_code = "invalid_media";
    return false;
  }
  const int w = static_cast<int>(image.width);
  const int h = static_cast<int>(image.height);
  int out_w = job->out_w, out_h = job->out_h;
  if (out_w == 0 && job->scale == 2) {
    out_w = w * 2;
    out_h = h * 2;
  }
  if (out_w > kMaxOutputSide || out_h > kMaxOutputSide ||
      (out_w == 0 && (w * 4 > kMaxOutputSide || h * 4 > kMaxOutputSide))) {
    job->error = "output is limited to 8192x8192";
    job->error_code = "invalid_size";
    return false;
  }
  if (sync) {
    std::lock_guard lock(count_mutex_);
    if (waiting_ >= kMaxWaiting) {
      job->error = "upscale queue is full";
      job->error_code = "queue_full";
      return false;
    }
    ++waiting_;
  }
  UpscaleOutcome out;
  std::string error;
  bool ok = false;
  {
    std::lock_guard run(run_mutex_);
    if (sync) {
      std::lock_guard lock(count_mutex_);
      --waiting_;
    }
    job->queue_seconds = Seconds(started);
    if ((request != nullptr && request->is_cancelled &&
         request->is_cancelled()) ||
        job->cancel) {
      job->error = "request cancelled";
      job->error_code = "cancelled";
      return false;
    }
    ok = engine_->run(job->model, job->variant, image.pixels.data(), w, h,
                      out_w, out_h, job->fit, &out, &error);
  }
  if (!ok) {
    job->error = error;
    job->error_code = "upscale_failed";
    return false;
  }
  const auto encode_start = std::chrono::steady_clock::now();
  if (job->format == "jpg") {
    PreviewRgb picture;
    picture.width = out.width;
    picture.height = out.height;
    picture.rgb = std::move(out.rgb24);
    job->output = EncodePreviewJpeg(picture, job->quality);
  } else {
    const std::vector<std::uint8_t> png =
        EncodePngRgb(out.rgb24, out.width, out.height);
    if (png.empty()) {
      job->error = "PNG encoding failed";
      job->error_code = "encode_failed";
      return false;
    }
    job->output.assign(png.begin(), png.end());
  }
  job->encode_seconds = Seconds(encode_start);
  job->in_width = w;
  job->in_height = h;
  job->width = out.width;
  job->height = out.height;
  job->body_seconds = out.body_seconds;
  job->post_seconds = out.post_seconds;
  job->model_file = out.model_file;
  job->total_seconds = Seconds(started);
  std::vector<std::uint8_t>().swap(job->input);
  return true;
}

HttpResponse UpscaleService::ImageResponse(const Job& job) {
  const std::string timing =
      "queue;dur=" + Ms(job.queue_seconds) + ", body;dur=" +
      Ms(job.body_seconds) + ", post;dur=" + Ms(job.post_seconds) +
      ", encode;dur=" + Ms(job.encode_seconds);
  const std::string content_type =
      job.format == "jpg" ? "image/jpeg" : "image/png";
  if (job.binary) {
    HttpResponse response;
    response.body = job.output;
    response.headers = {{"Content-Type", content_type},
                        {"Server-Timing", timing},
                        {"X-Gufo-Device", engine_->device},
                        {"X-Gufo-Model-File", job.model_file}};
    return response;
  }
  json::Value root = json::Value::object();
  root["created"] = static_cast<double>(job.created_at);
  if (!job.id.empty())
    root["id"] = job.id;
  root["model"] = job.model;
  root["variant"] = job.variant;
  root["device"] = engine_->device;
  root["model_file"] = job.model_file;
  root["width"] = job.width;
  root["height"] = job.height;
  root["input"] = std::to_string(job.in_width) + "x" +
                  std::to_string(job.in_height);
  root["scale"] = job.scale;
  root["fit"] = job.fit;
  root["format"] = job.format;
  json::Value item = json::Value::object();
  item["b64_json"] = EncodeAudioBase64(job.output);
  json::Value data = json::Value::array();
  data.push_back(std::move(item));
  root["data"] = std::move(data);
  json::Value cost = json::Value::object();
  cost["seconds"] = job.total_seconds;
  cost["queue_seconds"] = job.queue_seconds;
  cost["body_seconds"] = job.body_seconds;
  cost["post_seconds"] = job.post_seconds;
  cost["encode_seconds"] = job.encode_seconds;
  root["cost"] = std::move(cost);
  HttpResponse response = Json(200, "OK", root);
  response.headers.emplace_back("Server-Timing", timing);
  return response;
}

HttpResponse UpscaleService::SubmitVideo(const HttpRequest& request) {
  HttpResponse parse_error;
  const std::optional<json::Value> parsed = ParseBody(request, &parse_error);
  if (!parsed)
    return parse_error;
  const json::Value& body = *parsed;
  auto job = std::make_shared<Job>();
  job->kind = "video";
  if (auto error = ParseCommon(body, job.get(), true))
    return *error;
  job->crf = static_cast<int>(body.member_double("crf", 18));
  if (job->crf < 0 || job->crf > 51)
    return Error(400, "Bad Request", "crf must be 0..51", "invalid_crf");
  if (const json::Value* keep = body.find("keep_audio");
      keep != nullptr && keep->is_bool())
    job->keep_audio = keep->as_bool();
  const std::string output_format = body.member_str("output_format", "mp4");
  if (output_format != "mp4")
    return Error(400, "Bad Request", "output_format must be mp4",
                 "unsupported_format");
  const std::string video = body.member_str("video");
  if (video.empty())
    return Error(400, "Bad Request", "video is required", "invalid_media");
  try {
    core::ImageReadBudget budget;
    budget.remaining_bytes = kMaxVideoBytes;
    budget.deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    job->input = core::ReadVideoUrl(video, budget);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", std::string("video: ") + e.what(),
                 "invalid_media");
  }
  if (job->input.empty())
    return Error(400, "Bad Request", "video is empty", "invalid_media");
  const json::Value* async_value = body.find("async");
  const bool async =
      async_value == nullptr || !async_value->is_bool() || async_value->as_bool();
  std::string error;
  if (!Enqueue(job, &error))
    return Error(429, "Too Many Requests", error, "queue_full");
  Logger::Info("upscale", "job=" + job->id + " video " +
                              std::to_string(job->input.size()) +
                              " bytes model=" + job->model +
                              (job->variant.empty() ? "" : "/" + job->variant) +
                              (job->scale > 0
                                   ? " scale=x" + std::to_string(job->scale)
                                   : " size=" + std::to_string(job->out_w) +
                                         "x" + std::to_string(job->out_h)));
  if (async) {
    HttpResponse response = JobStatus(job->id, false);
    response.status = 202;
    response.reason = "Accepted";
    response.headers.emplace_back("Location", "/v1/videos/" + job->id);
    return response;
  }
  {
    std::unique_lock lock(mutex_);
    while (!job->done && !(request.is_cancelled && request.is_cancelled()))
      changed_.wait_for(lock, std::chrono::seconds(1));
    if (!job->done) {
      job->cancel = true;
      return Error(499, "Client Closed Request", "request cancelled",
                   "cancelled");
    }
  }
  return JobStatus(job->id, true);
}

void UpscaleService::RunVideo(Job* job) {
  namespace io = models::rife;
  const auto started = std::chrono::steady_clock::now();
  const std::filesystem::path input = TempFile("gufo-" + job->id + "-in.mp4");
  const std::filesystem::path output =
      TempFile("gufo-" + job->id + "-out.mp4");
  std::error_code ec;
  const auto cleanup = [&] {
    std::filesystem::remove(input, ec);
    std::filesystem::remove(output, ec);
  };
  const auto fail = [&](std::string message, std::string code) {
    job->error = std::move(message);
    job->error_code = std::move(code);
    cleanup();
  };
  if (!WriteFile(input, job->input)) {
    fail("cannot write the video to a temporary file", "io_error");
    return;
  }
  std::vector<std::uint8_t>().swap(job->input);
  io::VideoInfo info;
  std::string error;
  if (!io::ProbeVideo(input, &info, &error)) {
    fail("video: " + error, "invalid_media");
    return;
  }
  if (info.width < 2 || info.height < 2) {
    fail("video has no picture", "invalid_media");
    return;
  }
  int out_w = job->out_w, out_h = job->out_h;
  if (out_w == 0) {
    out_w = info.width * job->scale;
    out_h = info.height * job->scale;
  }
  if (out_w > kMaxOutputSide || out_h > kMaxOutputSide) {
    fail("output is limited to 8192x8192 (" + std::to_string(out_w) + "x" +
             std::to_string(out_h) + ")",
         "invalid_size");
    return;
  }
  {
    std::lock_guard lock(mutex_);
    job->in_width = info.width;
    job->in_height = info.height;
    job->width = out_w;
    job->height = out_h;
    job->frames = std::max(0, info.frames);
    job->fps_num = info.fps_num > 0 ? info.fps_num : 24;
    job->fps_den = info.fps_num > 0 ? info.fps_den : 1;
    job->audio = info.has_audio && job->keep_audio;
  }
  io::FrameReader reader;
  io::FrameWriter writer;
  writer.SetCrf(job->crf);
  if (!reader.Open(input, info.width, info.height, &error)) {
    fail("video: " + error, "invalid_media");
    return;
  }
  if (!writer.Open(output, out_w, out_h, job->fps_num, job->fps_den,
                   job->audio ? input : std::filesystem::path{}, &error)) {
    reader.Close();
    fail("encoder: " + error, "encode_failed");
    return;
  }
  std::vector<std::uint8_t> frame(static_cast<std::size_t>(info.width) *
                                  info.height * 3);
  int done = 0;
  double body_seconds = 0, post_seconds = 0;
  std::string model_file;
  for (;;) {
    bool end = false;
    if (!reader.Read(frame.data(), &end, &error)) {
      if (end)
        break;
      writer.Abort();
      reader.Close();
      fail("decoder: " + error, "invalid_media");
      return;
    }
    if (job->cancel) {
      writer.Abort();
      reader.Close();
      fail("job cancelled", "cancelled");
      return;
    }
    UpscaleOutcome out;
    bool ok = false;
    {
      std::lock_guard run(run_mutex_);
      ok = engine_->run(job->model, job->variant, frame.data(), info.width,
                        info.height, out_w, out_h, job->fit, &out, &error);
    }
    if (!ok || !writer.Write(out.rgb24.data(), &error)) {
      writer.Abort();
      reader.Close();
      fail((ok ? "encoder: " : "upscale: ") + error,
           ok ? "encode_failed" : "upscale_failed");
      return;
    }
    body_seconds += out.body_seconds;
    post_seconds += out.post_seconds;
    model_file = out.model_file;
    ++done;
    {
      std::lock_guard lock(mutex_);
      job->frames_done = done;
      job->progress = job->frames > 0
                          ? std::min(0.99, static_cast<double>(done) /
                                               static_cast<double>(job->frames))
                          : 0.0;
    }
  }
  reader.Close();
  if (done == 0) {
    writer.Abort();
    fail("video has no frames", "invalid_media");
    return;
  }
  if (!writer.Finish(&error)) {
    fail("encoder: " + error, "encode_failed");
    return;
  }
  std::string bytes;
  if (!ReadFile(output, &bytes)) {
    fail("cannot read the encoded video", "io_error");
    return;
  }
  cleanup();
  std::lock_guard lock(mutex_);
  job->output = std::move(bytes);
  job->frames = done;
  job->body_seconds = body_seconds;
  job->post_seconds = post_seconds;
  job->model_file = model_file;
  job->total_seconds = Seconds(started);
}

void UpscaleService::Worker() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_)
        return;
      job = queue_.front();
      queue_.pop_front();
      if (job->cancel) {
        job->status = "cancelled";
        job->error = "job cancelled";
        job->error_code = "cancelled";
        job->done = true;
        job->completed_at = Now();
        continue;
      }
      job->status = "in_progress";
    }
    const auto started = std::chrono::steady_clock::now();
    bool ok = false;
    if (job->kind == "video") {
      RunVideo(job.get());
      ok = job->error_code.empty();
    } else {
      ok = RunImage(job.get(), false, nullptr);
    }
    {
      std::lock_guard lock(mutex_);
      job->status = ok ? "completed"
                       : job->error_code == "cancelled" ? "cancelled"
                                                        : "failed";
      job->progress = ok ? 1.0 : job->progress;
      job->completed_at = Now();
      job->done = true;
      if (!ok)
        std::string().swap(job->output);
    }
    Logger::Info("upscale",
                 "job=" + job->id + " " + job->kind + " " + job->status +
                     (ok ? " " + std::to_string(job->width) + "x" +
                               std::to_string(job->height) +
                               (job->kind == "video"
                                    ? " frames=" + std::to_string(job->frames)
                                    : "") +
                               " seconds=" + Ms(Seconds(started)) + "ms"
                         : " " + job->error));
    changed_.notify_all();
  }
}

void UpscaleService::ExpireLocked() {
  const std::int64_t now = Now();
  std::vector<std::pair<std::int64_t, std::string>> finished;
  for (const auto& [id, job] : jobs_)
    if (job->done)
      finished.emplace_back(job->completed_at, id);
  std::sort(finished.begin(), finished.end());
  const std::size_t surplus =
      finished.size() > kKeptJobs ? finished.size() - kKeptJobs : 0;
  for (std::size_t i = 0; i < finished.size(); ++i)
    if (i < surplus || now - finished[i].first > kKeepSeconds)
      jobs_.erase(finished[i].second);
}

HttpResponse UpscaleService::JobStatus(const std::string& id, bool content) {
  std::lock_guard lock(mutex_);
  const auto it = jobs_.find(id);
  if (it == jobs_.end())
    return Error(404, "Not Found", "unknown upscale job " + id,
                 "video_not_found");
  const Job& job = *it->second;
  if (content) {
    if (job.status == "failed" || job.status == "cancelled")
      return Error(job.status == "failed" ? 500 : 409,
                   job.status == "failed" ? "Internal Server Error"
                                          : "Conflict",
                   job.error, job.error_code.empty() ? "upscale_failed"
                                                     : job.error_code);
    if (!job.done)
      return Error(409, "Conflict", "upscale job is not finished",
                   "not_ready");
    if (job.kind == "image")
      return ImageResponse(job);
    HttpResponse response;
    response.body = job.output;
    response.headers = {
        {"Content-Type", "video/mp4"},
        {"Content-Disposition", "attachment; filename=\"" + job.id + ".mp4\""},
        {"X-Gufo-Video-Id", job.id},
        {"X-Gufo-Device", engine_->device},
        {"X-Gufo-Model-File", job.model_file}};
    return response;
  }
  json::Value body = json::Value::object();
  body["id"] = job.id;
  body["object"] = job.kind == "video" ? "video" : "image.upscale";
  body["schema"] = "gufo.upscale-api.v1";
  body["created_at"] = static_cast<double>(job.created_at);
  body["status"] = job.status;
  body["progress"] = static_cast<double>(std::lround(job.progress * 100));
  body["model"] = job.model;
  body["variant"] = job.variant;
  body["device"] = engine_->device;
  body["size"] = job.width > 0 ? std::to_string(job.width) + "x" +
                                     std::to_string(job.height)
                               : std::string();
  if (job.in_width > 0)
    body["input"] = std::to_string(job.in_width) + "x" +
                    std::to_string(job.in_height);
  body["scale"] = job.scale;
  body["fit"] = job.fit;
  if (job.kind == "video") {
    body["output_format"] = "mp4";
    body["seconds"] = job.fps_num > 0 && job.frames > 0
                          ? std::to_string(static_cast<double>(job.frames) *
                                           job.fps_den / job.fps_num)
                          : std::string();
    json::Value upscale = json::Value::object();
    upscale["frames"] = job.frames;
    upscale["frames_done"] = job.frames_done;
    upscale["fps"] = job.fps_num > 0 ? std::to_string(job.fps_num) + "/" +
                                           std::to_string(job.fps_den)
                                     : std::string();
    upscale["audio"] = job.audio;
    upscale["crf"] = job.crf;
    body["upscale"] = std::move(upscale);
  } else {
    body["format"] = job.format;
  }
  if (job.done) {
    body["completed_at"] = static_cast<double>(job.completed_at);
    body["expires_at"] = static_cast<double>(job.completed_at + kKeepSeconds);
  } else {
    body["completed_at"] = json::Value();
    body["expires_at"] = json::Value();
  }
  if (job.done && job.error_code.empty()) {
    json::Value cost = json::Value::object();
    cost["seconds"] = job.total_seconds;
    cost["body_seconds"] = job.body_seconds;
    cost["post_seconds"] = job.post_seconds;
    body["cost"] = std::move(cost);
    body["model_file"] = job.model_file;
  }
  if (job.error.empty()) {
    body["error"] = json::Value();
  } else {
    json::Value error = json::Value::object();
    error["code"] = job.error_code;
    error["message"] = job.error;
    body["error"] = std::move(error);
  }
  HttpResponse response = Json(200, "OK", body);
  response.log_details = "job=" + job.id + " state=" + job.status +
                         " model=" + job.model + " size=" +
                         std::to_string(job.width) + "x" +
                         std::to_string(job.height);
  return response;
}

HttpResponse UpscaleService::CancelJob(const std::string& id) {
  std::shared_ptr<Job> job;
  {
    std::lock_guard lock(mutex_);
    const auto it = jobs_.find(id);
    if (it == jobs_.end())
      return Error(404, "Not Found", "unknown upscale job " + id,
                   "video_not_found");
    job = it->second;
    if (job->done) {
      // A finished job: forget the result.
      jobs_.erase(it);
      json::Value body = json::Value::object();
      body["id"] = id;
      body["object"] = job->kind == "video" ? "video" : "image.upscale";
      body["status"] = job->status;
      body["deleted"] = true;
      return Json(200, "OK", body);
    }
    job->cancel = true;
  }
  changed_.notify_all();
  return JobStatus(id, false);
}

}  // namespace gufo::server
