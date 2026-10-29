#ifndef GUFO_MODELS_RIFE_IFNET_HPP_
#define GUFO_MODELS_RIFE_IFNET_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace gufo::models::rife {

/// Activation precision of the IFNet convolutions. kFp16 runs every 3x3
/// convolution (and the transposed ones, rewritten as 3x3 convolutions plus
/// depth-to-space) as FP16 WMMA with FP32 accumulation; warps, resizes, flow
/// and mask stay FP32. kFp32 is a direct FP32 path for validation.
enum class Precision : std::uint8_t { kFp16, kFp32 };

/// RIFE 4.7 IFNet (Practical-RIFE / ComfyUI-Frame-Interpolation arch "4.7",
/// checkpoints rife47.pth and rife49.pth) for one frame pair at a time,
/// exactly as ComfyUI's "RIFE VFI" node calls it: frames zero-padded to
/// multiples of 64, scale_list = [8, 4, 2, 1] / scale_factor, no ensemble
/// (the node passes `ensemble` into the `fastmode` slot), output clamped to
/// [0, 1]. The per-frame encoder features are computed once per frame slot,
/// so x4 runs the encoder once per input frame and the four IFBlocks once per
/// interpolated frame.
class Interpolator {
public:
  static constexpr int kSlots = 2;

  /// Loads an FP32 safetensors state_dict (tools/rife/convert_rife_pth.py).
  [[nodiscard]] static std::unique_ptr<Interpolator> Load(
      const std::filesystem::path& weights, Precision precision,
      std::string* error);
  ~Interpolator();
  Interpolator(const Interpolator&) = delete;
  Interpolator& operator=(const Interpolator&) = delete;

  /// Allocates the workspace for width x height frames. scale_factor is the
  /// node's option (0.5, 1, 2 or 4; 1 by default).
  bool Configure(int width, int height, float scale_factor, std::string* error);

  /// Uploads an RGB24 frame [height, width, 3] into `slot` and encodes it.
  bool SetFrame(int slot, const std::uint8_t* rgb24, std::string* error);

  /// Interpolates between slots at timestep t in (0, 1). Writes RGB24
  /// (round(255 * x)) and/or FP32 RGB in [0, 1]; either may be null.
  bool Interpolate(int slot0, int slot1, float t, std::uint8_t* rgb24,
                   float* rgb, std::string* error);

  [[nodiscard]] Precision precision() const;
  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::size_t workspace_bytes() const;

private:
  struct Impl;
  explicit Interpolator(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::rife

#endif  // GUFO_MODELS_RIFE_IFNET_HPP_
