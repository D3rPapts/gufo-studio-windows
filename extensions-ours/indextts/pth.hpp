// Read-only, zero-copy reader for torch.save() checkpoints (.pth/.pt/.bin zip
// archives with an uncompressed pickle) as used by IndexTTS-2.5: gpt.pth,
// s2mel.pth, codec.pth, feat1/feat2.pt, wav2vec2bert_stats.pt, CAM++ and
// BigVGAN. Nested dicts are flattened to dotted keys ("net.cfm.estimator...").
// Tensor data stays in the memory-mapped file; FP32 tensors can be used in
// place. The zip/pickle reader follows src/models/voxcpm2/vae_weights.cpp.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_PTH_HPP_
#define GUFO_MODELS_INDEXTTS_PTH_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gufo/ext/platform/mapped_file.hpp>

namespace gufo::models::indextts {

enum class PthDType : std::uint8_t { kF32, kF16, kBF16, kF64, kI64, kI32, kBool, kUnknown };

struct PthTensor {
  PthDType dtype{PthDType::kUnknown};
  std::vector<std::int64_t> shape;
  const std::uint8_t* data{nullptr};  // first element (storage offset applied)
  std::int64_t numel{0};
  bool contiguous{true};
};

class PthArchive {
public:
  [[nodiscard]] static std::unique_ptr<PthArchive> Open(
      const std::filesystem::path& path, std::string* error = nullptr);

  [[nodiscard]] const PthTensor* Find(std::string_view key) const;
  // Throws std::runtime_error when missing.
  [[nodiscard]] const PthTensor& Require(std::string_view key) const;
  // Zero-copy FP32 view; throws unless the tensor is contiguous F32 with
  // `expect_numel` elements (-1: any).
  [[nodiscard]] const float* F32(std::string_view key, std::int64_t expect_numel = -1) const;
  // Copy converted to FP32 (F32/F16/BF16/F64 sources).
  [[nodiscard]] std::vector<float> ToF32(std::string_view key) const;
  [[nodiscard]] bool Has(std::string_view key) const { return Find(key) != nullptr; }
  [[nodiscard]] const std::map<std::string, PthTensor, std::less<>>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
  platform::ReadOnlyMappedFile file_;
  std::map<std::string, PthTensor, std::less<>> tensors_;
  std::filesystem::path path_;
};

// Folded weight norm: w[o, ...] = g[o] * v[o, ...] / ||v[o, ...]|| (dim 0).
[[nodiscard]] std::vector<float> FoldWeightNorm(const PthArchive& archive,
                                                const std::string& prefix);

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_PTH_HPP_
