// Real-ESRGAN image upscaling behind /v1/images/upscale (see upscale_api.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/upscale_api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <utility>

#include "src/cli/serve/audio_stream.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/model.hpp"

#if defined(GUFO_HAVE_ONNXRUNTIME)
#include "src/models/realesrgan/ort_upscaler.hpp"
#endif
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/realesrgan/upscaler.hpp"
#endif

namespace gufo::server {
namespace {

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

std::vector<std::uint8_t> EncodePngRgb(const std::vector<std::uint8_t>& rgb,
                                       int w, int h) {
  models::qwen_image_21::Image image;
  image.width = w;
  image.height = h;
  image.rgba.resize(static_cast<std::size_t>(w) * h * 4);
  for (std::size_t i = 0, n = static_cast<std::size_t>(w) * h; i < n; ++i) {
    image.rgba[i * 4] = rgb[i * 3];
    image.rgba[i * 4 + 1] = rgb[i * 3 + 1];
    image.rgba[i * 4 + 2] = rgb[i * 3 + 2];
    image.rgba[i * 4 + 3] = 255;
  }
  return models::qwen_image_21::EncodePng(image);
}

std::string Ms(double seconds) {
  return std::to_string(std::lround(seconds * 1000.0));
}

}  // namespace

bool IsUpscaleApiPath(std::string_view path) noexcept {
  return path == kUpscaleImagePath;
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

std::unique_ptr<UpscaleEngine> CreateUpscaleEngine(
    const UpscaleServeOptions& options, std::string* error) {
  if (options.models.empty()) {
    *error = "--model NAME=FILE is required";
    return nullptr;
  }
  auto engine = std::make_unique<UpscaleEngine>();
  engine->device = options.device;
  for (const UpscaleModelArg& m : options.models)
    if (std::find(engine->model_names.begin(), engine->model_names.end(),
                  m.name) == engine->model_names.end())
      engine->model_names.push_back(m.name);
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
    engine->run = [upscaler](const std::string& model,
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
    auto upscalers = std::make_shared<
        std::vector<std::pair<std::string, std::shared_ptr<re::Upscaler>>>>();
    for (const UpscaleModelArg& m : options.models) {
      std::shared_ptr<re::Upscaler> u = re::Upscaler::Load(
          m.path, options.fp32 ? re::Precision::kFp32 : re::Precision::kFp16,
          error);
      if (u == nullptr)
        return nullptr;
      upscalers->emplace_back(m.name, std::move(u));
    }
    engine->runtime = std::string("HIP Real-ESRGAN ") +
                      (options.fp32 ? "FP32" : "FP16");
    const std::vector<UpscaleModelArg> files = options.models;
    engine->run = [upscalers, files](const std::string& model,
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
      for (std::size_t i = 0; i < upscalers->size(); ++i)
        if ((*upscalers)[i].first == model)
          index = i;
      re::Upscaler& u = *(*upscalers)[index].second;
      const int ow = out_w == 0 ? w * u.scale() : out_w;
      const int oh = out_h == 0 ? h * u.scale() : out_h;
      out->rgb24.resize(static_cast<std::size_t>(ow) * oh * 3);
      const auto start = std::chrono::steady_clock::now();
      if (!u.Upscale(rgb24, w, h, ow, oh, fit, out->rgb24.data(), err))
        return false;
      out->width = ow;
      out->height = oh;
      out->body_seconds = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - start)
                              .count();
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
    : engine_(std::move(engine)) {}

HttpResponse UpscaleService::Handle(const HttpRequest& request) {
  if (request.method != "POST")
    return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
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
  std::string model = body.member_str("model", engine_->model_names.front());
  if (std::find(engine_->model_names.begin(), engine_->model_names.end(),
                model) == engine_->model_names.end())
    return Error(404, "Not Found", "model '" + model + "' is not served here",
                 "model_not_found");
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
  if (out_w > 8192 || out_h > 8192)
    return Error(400, "Bad Request", "output is limited to 8192x8192",
                 "invalid_size");
  const std::string fit = body.member_str("fit", "crop");
  const std::string format = body.member_str("response_format", "b64_json");
  if (format != "b64_json" && format != "png")
    return Error(400, "Bad Request", "response_format must be b64_json or png",
                 "unsupported_format");
  const std::string image_url = body.member_str("image");
  if (image_url.empty())
    return Error(400, "Bad Request", "image is required", "invalid_media");
  core::Image image;
  try {
    core::ImageReadBudget budget;
    budget.deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    image = core::DecodeImage(core::ReadImageUrl(image_url, budget));
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", std::string("image: ") + e.what(),
                 "invalid_media");
  }
  {
    std::lock_guard lock(count_mutex_);
    if (waiting_ >= kMaxWaiting)
      return Error(429, "Too Many Requests", "upscale queue is full",
                   "queue_full");
    ++waiting_;
  }
  UpscaleOutcome out;
  std::string error;
  bool ok = false;
  const auto queued = std::chrono::steady_clock::now();
  double wait_seconds = 0;
  {
    std::lock_guard run(run_mutex_);
    {
      std::lock_guard lock(count_mutex_);
      --waiting_;
    }
    wait_seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - queued)
                       .count();
    if (request.is_cancelled && request.is_cancelled())
      return Error(499, "Client Closed Request", "request cancelled",
                   "cancelled");
    ok = engine_->run(model, image.pixels.data(),
                      static_cast<int>(image.width),
                      static_cast<int>(image.height), out_w, out_h, fit, &out,
                      &error);
  }
  if (!ok)
    return Error(400, "Bad Request", error, "upscale_failed");
  const auto encode_start = std::chrono::steady_clock::now();
  const std::vector<std::uint8_t> png =
      EncodePngRgb(out.rgb24, out.width, out.height);
  const double encode_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    encode_start)
                                    .count();
  const std::string timing =
      "queue;dur=" + Ms(wait_seconds) + ", body;dur=" + Ms(out.body_seconds) +
      ", post;dur=" + Ms(out.post_seconds) + ", png;dur=" + Ms(encode_seconds);
  if (format == "png") {
    HttpResponse response;
    response.body.assign(png.begin(), png.end());
    response.headers = {{"Content-Type", "image/png"},
                        {"Server-Timing", timing},
                        {"X-Gufo-Device", engine_->device},
                        {"X-Gufo-Model-File", out.model_file}};
    return response;
  }
  json::Value root = json::Value::object();
  root["created"] = static_cast<double>(std::time(nullptr));
  root["model"] = model;
  root["device"] = engine_->device;
  root["model_file"] = out.model_file;
  root["width"] = out.width;
  root["height"] = out.height;
  root["fit"] = fit;
  json::Value item = json::Value::object();
  item["b64_json"] = EncodeAudioBase64(std::string_view(
      reinterpret_cast<const char*>(png.data()), png.size()));
  json::Value data = json::Value::array();
  data.push_back(std::move(item));
  root["data"] = std::move(data);
  json::Value cost = json::Value::object();
  cost["seconds"] = wait_seconds + out.body_seconds + out.post_seconds +
                    encode_seconds;
  cost["queue_seconds"] = wait_seconds;
  cost["body_seconds"] = out.body_seconds;
  cost["post_seconds"] = out.post_seconds;
  cost["png_seconds"] = encode_seconds;
  root["cost"] = std::move(cost);
  HttpResponse response = Json(200, "OK", root);
  response.headers.emplace_back("Server-Timing", timing);
  return response;
}

}  // namespace gufo::server
