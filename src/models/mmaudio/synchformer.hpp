// MMAudio for gufo: the Synchformer visual feature extractor.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_SYNCHFORMER_HPP_
#define GUFO_MODELS_MMAUDIO_SYNCHFORMER_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/models/mmaudio/common.hpp"

namespace gufo::models::mmaudio {

/// The video branch of Synchformer (`vfeat_extractor.*` of
/// mmaudio_synchformer_fp16.safetensors): MotionFormer ViT-B/16 with divided
/// space-time attention over 16-frame segments of 224 x 224 (tubelets of
/// 2 x 16 x 16 -> 8 x 14 x 14 tokens + CLS, 12 blocks of 768), final
/// LayerNorm, then a spatial TransformerEncoderLayer with its own CLS per
/// frame. Each segment gives 8 features of 768.
class Synchformer {
public:
  static constexpr int kSize = 224;
  static constexpr int kSegmentFrames = 16;
  static constexpr int kSegmentStep = 8;
  static constexpr int kFeaturesPerSegment = 8;
  static constexpr int kWidth = 768;

  [[nodiscard]] static std::unique_ptr<Synchformer> Load(
      const std::filesystem::path& path, Precision precision,
      std::string* error);
  ~Synchformer();
  Synchformer(const Synchformer&) = delete;
  Synchformer& operator=(const Synchformer&) = delete;

  /// `frames` [count][3][224][224], normalized to (x - 0.5) / 0.5; segments
  /// start every 8 frames ((count - 16) / 8 + 1 of them) -> `features`
  /// [segments * 8][768].
  bool Encode(const std::vector<float>& frames, int count,
              std::vector<float>* features, std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit Synchformer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_SYNCHFORMER_HPP_
