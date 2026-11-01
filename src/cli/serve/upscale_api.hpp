// Real-ESRGAN image upscaling behind POST /v1/images/upscale
// (`gufo serve upscale --device npu|gpu|cpu`).
// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_UPSCALE_API_HPP_
#define GUFO_SERVER_UPSCALE_API_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {

inline constexpr std::string_view kUpscaleImagePath = "/v1/images/upscale";

struct UpscaleModelArg {
  std::string name;
  std::filesystem::path path;
};

/// `gufo serve upscale` options.
struct UpscaleServeOptions {
  std::string device{"npu"};  // npu | cpu | gpu
  /// npu / cpu: body ONNX files (several sizes may share a name);
  /// gpu: converted Real-ESRGAN safetensors.
  std::vector<UpscaleModelArg> models;
  std::filesystem::path onnxruntime;   // onnxruntime.dll override
  std::filesystem::path npu_ep;        // onnxruntime_vitisai_ep.dll override
  std::filesystem::path npu_cache;     // VitisAI compile cache
  std::filesystem::path npu_config;    // VitisAI config_file
  int threads{0};                      // CPU provider threads
  bool fp32{false};                    // gpu: FP32 validation path
};

/// Result of one upscale.
struct UpscaleOutcome {
  std::vector<std::uint8_t> rgb24;
  int width{0};
  int height{0};
  double body_seconds{0};
  double post_seconds{0};
  std::string model_file;
};

/// Device-specific engine: ONNX Runtime (npu / cpu) or HIP (gpu).
struct UpscaleEngine {
  std::string device;
  /// Public model names, the first is the default.
  std::vector<std::string> model_names;
  /// Runtime description for /v1/models and logs.
  std::string runtime;
  /// rgb24 w x h -> out_w x out_h (0 x 0: x4) with fit (crop | pad |
  /// stretch). Called under the service lock.
  std::function<bool(const std::string& model, const std::uint8_t* rgb24,
                     int w, int h, int out_w, int out_h,
                     const std::string& fit, UpscaleOutcome* out,
                     std::string* error)>
      run;
};

/// Builds the engine for options.device; nullptr + *error on failure.
[[nodiscard]] std::unique_ptr<UpscaleEngine> CreateUpscaleEngine(
    const UpscaleServeOptions& options, std::string* error);

/// POST /v1/images/upscale
///   {model, image (data URL / base64 / https), size "WxH" | width + height
///    (default: x4), fit ("crop" default | "pad" | "stretch"),
///    response_format ("b64_json" default | "png")}
/// -> {created, model, device, width, height, data: [{b64_json}],
///     cost: {seconds, body_seconds, post_seconds}} or image/png.
/// One request at a time on the device; at most kMaxWaiting wait.
class UpscaleService {
public:
  static constexpr int kMaxWaiting = 8;

  explicit UpscaleService(std::unique_ptr<UpscaleEngine> engine);

  [[nodiscard]] bool ready() const { return engine_ != nullptr; }
  [[nodiscard]] const std::string& model_id() const {
    return engine_->model_names.front();
  }
  [[nodiscard]] const UpscaleEngine& engine() const { return *engine_; }

  HttpResponse Handle(const HttpRequest& request);

private:
  std::unique_ptr<UpscaleEngine> engine_;
  std::mutex run_mutex_;
  std::mutex count_mutex_;
  int waiting_{0};
};

[[nodiscard]] bool IsUpscaleApiPath(std::string_view path) noexcept;

/// "1920x1080" -> 1920, 1080.
[[nodiscard]] bool ParseUpscaleSize(std::string_view text, int* w, int* h);

}  // namespace gufo::server

#endif  // GUFO_SERVER_UPSCALE_API_HPP_
