// Real-ESRGAN "body" ONNX models (int8 QDQ, fixed input size, e.g. Take
// One's body896q_h288_h3.onnx) in ONNX Runtime: on the AMD NPU through the
// VitisAI execution provider from Windows ML, or on the ONNX Runtime CPU
// provider. ONNX Runtime is loaded at run time (LoadLibrary of
// onnxruntime.dll); nothing else in gufo depends on it.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_REALESRGAN_ORT_UPSCALER_HPP_
#define GUFO_MODELS_REALESRGAN_ORT_UPSCALER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::realesrgan {

enum class OrtDevice : std::uint8_t { kNpu, kCpu };

[[nodiscard]] bool ParseOrtDevice(std::string_view text, OrtDevice* device);
[[nodiscard]] std::string_view OrtDeviceName(OrtDevice device);

struct OrtUpscalerOptions {
  OrtDevice device{OrtDevice::kNpu};
  /// onnxruntime.dll (1.23+). Empty: the newest Windows ML framework package
  /// (WindowsWorkload.WinMLShared.*), then GUFO_ONNXRUNTIME_DLL.
  std::filesystem::path runtime_library;
  /// VitisAI EP library (onnxruntime_vitisai_ep.dll). Empty: asked from the
  /// Windows ML execution provider catalog.
  std::filesystem::path ep_library;
  /// VitisAI compile cache; empty: <model dir>/npu-cache.
  std::filesystem::path cache_dir;
  /// Optional VitisAI config_file (vaiml_config.json).
  std::filesystem::path ep_config_file;
  /// CPU provider intra-op threads (0: ONNX Runtime default).
  int threads{0};
  /// ONNX Runtime log severity (0 verbose .. 4 fatal).
  int log_level{2};
};

/// One body model file under a public name.
struct OrtModelSpec {
  std::string name;
  std::filesystem::path path;
  /// VitisAI cache key; empty: the file stem (Take One: body896q_h288_h3).
  std::string cache_key;
};

/// Information about a loaded body model.
struct OrtModelInfo {
  std::string name;
  std::filesystem::path path;
  std::string cache_key;
  int width{0};       // input frame width
  int tile_rows{0};   // input rows per pass
  double load_seconds{0};
};

struct UpscaleTimings {
  double body_seconds{0};   // ONNX Runtime passes
  double post_seconds{0};   // shuffle + residual + fit
  int passes{0};
};

/// How the x4 picture is fitted into out_w x out_h.
enum class AreaFit : std::uint8_t { kCrop, kPad, kStretch };
[[nodiscard]] bool ParseAreaFit(std::string_view text, AreaFit* fit);
[[nodiscard]] std::string_view AreaFitName(AreaFit fit);

class OrtUpscaler {
public:
  /// Loads ONNX Runtime, registers the NPU provider (kNpu) and opens every
  /// model (the first NPU start compiles; later starts read the cache).
  [[nodiscard]] static std::unique_ptr<OrtUpscaler> Load(
      const OrtUpscalerOptions& options,
      const std::vector<OrtModelSpec>& models, std::string* error);
  ~OrtUpscaler();
  OrtUpscaler(const OrtUpscaler&) = delete;
  OrtUpscaler& operator=(const OrtUpscaler&) = delete;

  /// Model `name` whose input matches w x h (a 512-row frame with a 288-row
  /// body: two tiles). nullptr when none fits.
  [[nodiscard]] const OrtModelInfo* Find(std::string_view name, int w,
                                         int h) const;

  /// Network only: RGB24 [h, w, 3] -> RGB24 [4h, 4w, 3] (Take One's `up`).
  bool UpscaleX4(std::string_view name, const std::uint8_t* rgb24, int w,
                 int h, std::vector<std::uint8_t>* up24,
                 UpscaleTimings* timings, std::string* error,
                 std::vector<float>* body_out = nullptr);

  /// UpscaleX4, then INTER_AREA fit into out_w x out_h (0 x 0: x4 as is).
  bool Upscale(std::string_view name, const std::uint8_t* rgb24, int w, int h,
               int out_w, int out_h, AreaFit fit,
               std::vector<std::uint8_t>* out24, UpscaleTimings* timings,
               std::string* error);

  [[nodiscard]] OrtDevice device() const;
  [[nodiscard]] const std::vector<OrtModelInfo>& models() const;
  /// "1.23.20260331.3.510bad9" etc.
  [[nodiscard]] const std::string& runtime_version() const;
  [[nodiscard]] const std::filesystem::path& runtime_library() const;
  /// VitisAI EP version from the Windows ML catalog ("" for CPU / unknown).
  [[nodiscard]] const std::string& ep_version() const;
  [[nodiscard]] const std::filesystem::path& ep_library() const;

private:
  struct Impl;
  explicit OrtUpscaler(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::realesrgan

#endif  // GUFO_MODELS_REALESRGAN_ORT_UPSCALER_HPP_
