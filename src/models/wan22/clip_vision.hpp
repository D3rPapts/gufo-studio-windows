#ifndef GUFO_MODELS_WAN22_CLIP_VISION_HPP_
#define GUFO_MODELS_WAN22_CLIP_VISION_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// The image encoder of Wan 2.1 image-to-video: OpenCLIP ViT-H/14 (Diffusers
/// `image_encoder/`, transformers CLIPVisionModel, BF16 weights), returning
/// `hidden_states[-2]`: the residual stream after 31 of the 32 layers,
/// [257, 1280], without the post layer norm. Patch and position embeddings
/// and the pre layer norm run on the host in FP32; the layers use BF16 GEMMs
/// with FP32 accumulation, an FP32 residual stream and FP32 attention.
class ClipVision {
public:
  static constexpr int kImageSize = 224;
  static constexpr int kTokens = 257;
  static constexpr int kWidth = 1280;

  /// Loads `<diffusers root>/image_encoder`.
  [[nodiscard]] static std::unique_ptr<ClipVision> Load(
      const std::filesystem::path& dir, std::string* error);
  ~ClipVision();
  ClipVision(const ClipVision&) = delete;
  ClipVision& operator=(const ClipVision&) = delete;

  /// CLIPImageProcessor of the pipeline: bicubic resize (PIL-style, with
  /// antialiasing and 8-bit intermediates) of interleaved RGB8 to 224x224,
  /// then (x / 255 - mean) / std. Returns pixel values [3, 224, 224].
  static std::vector<float> Preprocess(std::span<const std::uint8_t> rgb,
                                       int width, int height);

  /// `pixels` [3, 224, 224] -> `states` [257, 1280].
  bool Encode(std::span<const float> pixels, std::vector<float>* states,
              std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit ClipVision(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_CLIP_VISION_HPP_
