// MMAudio for gufo: the DFN5B CLIP ViT-H-14-384 image and text towers.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_CLIP_HPP_
#define GUFO_MODELS_MMAUDIO_CLIP_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/mmaudio/common.hpp"

namespace gufo::models::mmaudio {

/// apple/DFN5B-CLIP-ViT-H-14-384 in OpenCLIP naming (Kijai's
/// apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors): QuickGELU, pre-norm.
/// Image tower: 384 x 384 input (27 x 27 patches of 14, the last 6 pixel rows
/// / columns unused), 32 layers of 1280, CLS -> ln_post -> proj (1024),
/// L2-normalized (encode_image(normalize=True)). Text tower: 77 tokens,
/// causal, 24 layers of 1024, ln_final, every token L2-normalized (MMAudio's
/// patched encode_text returns the whole sequence, no projection).
class ClipEncoder {
public:
  static constexpr int kImageSize = 384;
  static constexpr int kFeatures = 1024;
  static constexpr int kTextTokens = 77;
  static constexpr int kTextWidth = 1024;

  [[nodiscard]] static std::unique_ptr<ClipEncoder> Load(
      const std::filesystem::path& path, Precision precision,
      std::string* error);
  ~ClipEncoder();
  ClipEncoder(const ClipEncoder&) = delete;
  ClipEncoder& operator=(const ClipEncoder&) = delete;

  /// `pixels` [frames][3][384][384], already normalized with the CLIP
  /// mean / std -> `features` [frames][1024].
  bool EncodeImages(const std::vector<float>& pixels, int frames,
                    std::vector<float>* features, std::string* error);
  /// 77 token ids -> [77][1024].
  bool EncodeText(const std::vector<int>& tokens, std::vector<float>* features,
                  std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit ClipEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_CLIP_HPP_
