#ifndef GUFO_MODELS_WAN22_MOGE_HPP_
#define GUFO_MODELS_WAN22_MOGE_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// Metric geometry of one image from MoGe-2 (`infer` with its defaults:
/// resolution level 9 = 3600 base tokens, force_projection, apply_mask).
struct MogeResult {
  int width{0};
  int height{0};
  /// Camera-space metric points [height, width, 3] (OpenCV axes: x right, y
  /// down, z forward) and depth [height, width]; +inf where the mask is off.
  std::vector<float> points;
  std::vector<float> depth;
  std::vector<std::uint8_t> mask;
  /// Intrinsics in pixels (principal point at the image centre).
  double fx{0}, fy{0}, cx{0}, cy{0};
  double focal{0};  // relative to half the image diagonal
  double shift{0};
  double metric_scale{0};
};

/// Raw network outputs for oracle checks (before focal/shift recovery).
struct MogeRaw {
  int base_height{0}, base_width{0};
  std::vector<float> points;  // [height, width, 3], affine, after remap
  std::vector<float> mask;    // [height, width], sigmoid
  double metric_scale{0};
};

/// MoGe-2 ViT-L (Microsoft, MIT): DINOv2 ViT-L/14 encoder (layers 5, 11,
/// 17, 23 projected and summed), a 5-level conv neck with view-plane UV
/// inputs, point and mask heads and a metric scale MLP; weights from
/// tools-local/moge_to_safetensors.py. Everything runs in FP32 (one image
/// per clip; precision over speed).
class Moge {
public:
  [[nodiscard]] static std::unique_ptr<Moge> Load(
      const std::filesystem::path& file, std::string* error);
  ~Moge();
  Moge(const Moge&) = delete;
  Moge& operator=(const Moge&) = delete;

  /// `rgb` is RGB8 [height, width, 3].
  bool Infer(std::span<const std::uint8_t> rgb, int width, int height,
             MogeResult* result, MogeRaw* raw, std::string* error);

private:
  struct Impl;
  explicit Moge(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_MOGE_HPP_
