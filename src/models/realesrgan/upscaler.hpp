#ifndef GUFO_MODELS_REALESRGAN_UPSCALER_HPP_
#define GUFO_MODELS_REALESRGAN_UPSCALER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace gufo::models::realesrgan {

/// Activation precision of the convolutions. kFp16 runs every 3x3
/// convolution as FP16 WMMA with FP32 accumulation; kFp32 is a direct FP32
/// path for validation. The pixel shuffle, the nearest-neighbour base and the
/// final resize are FP32 in both.
enum class Precision : std::uint8_t { kFp16, kFp32 };

/// How the upscaled picture is fitted into the requested output size.
/// kPad keeps the aspect ratio and the whole picture (black bars), kCrop
/// keeps the aspect ratio and fills the output (centre crop), kStretch
/// scales each axis on its own.
enum class Fit : std::uint8_t { kPad, kCrop, kStretch };

[[nodiscard]] bool ParseFit(std::string_view text, Fit* fit);
[[nodiscard]] std::string_view FitName(Fit fit);

/// Output geometry of an upscale to out_w x out_h: the picture is resized to
/// content_w x content_h and placed at (offset_x, offset_y); negative offsets
/// crop. Even sizes and offsets keep 4:2:0 chroma aligned.
struct Placement {
  int content_w{0};
  int content_h{0};
  int offset_x{0};
  int offset_y{0};
};
[[nodiscard]] Placement PlaceContent(int in_w, int in_h, int out_w, int out_h,
                                     Fit fit);

/// Real-ESRGAN SRVGGNetCompact (realesr-animevideov3: 16 body convolutions,
/// realesr-general-x4v3 / -wdn-x4v3: 32; 64 features, PReLU, x4 pixel
/// shuffle plus the nearest-neighbour upsampled input), as
/// basicsr/archs/srvgg_arch.py computes it on RGB in [0, 1], output clamped
/// to [0, 1]. Frames are processed in tiles whose halo equals the number of
/// convolutions, so tiling does not change the result. The x4 picture is
/// then resized to the requested size with an antialiased Lanczos-3 filter.
class Upscaler {
public:
  /// Loads an FP32 safetensors file (tools/upscale/convert_realesrgan_pth.py).
  [[nodiscard]] static std::unique_ptr<Upscaler> Load(
      const std::filesystem::path& weights, Precision precision,
      std::string* error);
  ~Upscaler();
  Upscaler(const Upscaler&) = delete;
  Upscaler& operator=(const Upscaler&) = delete;

  /// Largest tile (input pixels including the halo) per network pass; the
  /// default (0) is 512 Ki pixels, so 832x480 runs as one tile.
  void SetTilePixels(std::size_t pixels);

  /// Network only: RGB24 [h, w, 3] -> FP32 RGB [h * scale, w * scale, 3] in
  /// [0, 1] (validation against the PyTorch model).
  bool UpscaleNetwork(const std::uint8_t* rgb24, int w, int h, float* out,
                      std::string* error);

  /// Network, then the resize into out_w x out_h with `fit`: RGB24 [out_h,
  /// out_w, 3].
  bool Upscale(const std::uint8_t* rgb24, int w, int h, int out_w, int out_h,
               Fit fit, std::uint8_t* out24, std::string* error);

  [[nodiscard]] int scale() const;
  [[nodiscard]] int convolutions() const;
  [[nodiscard]] Precision precision() const;
  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::size_t workspace_bytes() const;

private:
  struct Impl;
  explicit Upscaler(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::realesrgan

#endif  // GUFO_MODELS_REALESRGAN_UPSCALER_HPP_
