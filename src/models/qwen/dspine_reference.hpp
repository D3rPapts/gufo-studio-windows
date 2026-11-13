#ifndef GUFO_MODELS_QWEN_DSPINE_REFERENCE_HPP_
#define GUFO_MODELS_QWEN_DSPINE_REFERENCE_HPP_

// CPU reference of one DSpine draft round (tencent/Qwen3.8-27B-DSpine-b16,
// GGUF architecture "dspine", converted by tools/dspine/convert_dspine_gguf.py):
// a DFlash-style block draft (fc + hidden_norm context, five Qwen3 layers with
// a causal block mask, sliding window) plus the DSpine code bridge after every
// layer, and the greedy serial refinement tail. Mirrors dspine_mlx
// (model.py, bridge.py, propose.py RefinementTail.greedy) including its BF16
// rounding points. fp32 math, OpenMP; meant as the oracle for the HIP runner.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>
#include "src/models/qwen/dspine_config.hpp"

namespace gufo::models::qwen {

/// The target model pieces a draft round needs, all as dense fp32 rows.
struct DSpineTargetAccess {
  /// Token embedding row (hidden values).
  std::function<void(std::int64_t token, float* out)> embed_row;
  /// lm_head row (hidden values).
  std::function<void(std::int64_t token, float* out)> head_row;
  /// logits[i, v] = rows[i] . head_row(v) for all v (n x vocab, row-major).
  std::function<void(const float* rows, std::size_t n, float* logits)> head_logits;
};

struct DSpineRound {
  std::vector<float> rows;           // [T-1, hidden] final normed block rows
  std::vector<float> stash_raw;      // [T, hidden] last layer pre-write stream
  std::vector<std::int32_t> cand;    // [T-1, K] ascending ids
  std::vector<float> cand_logits;    // [T-1, K]
  std::vector<std::int32_t> drafts;  // [T-1]
};

class DSpineReference {
 public:
  [[nodiscard]] static std::unique_ptr<DSpineReference> Open(
      const std::filesystem::path& gguf, std::string* error);
  [[nodiscard]] const DSpineConfig& config() const { return cfg_; }

  /// One block of `T = block_size` (anchor + mask tokens) after `context`
  /// target-hidden rows (S x (targets * hidden), positions 0..S-1).
  [[nodiscard]] DSpineRound Draft(std::span<const float> context,
                                  std::size_t context_rows,
                                  std::int64_t anchor,
                                  const DSpineTargetAccess& target) const;

 private:
  DSpineReference() = default;
  [[nodiscard]] std::vector<float> Matrix(std::string_view name) const;
  void Row(std::string_view name, std::size_t row, float* out) const;
  [[nodiscard]] std::vector<float> Linear(std::span<const float> x,
                                          std::size_t rows,
                                          std::string_view weight) const;

  std::unique_ptr<core::GgufReader> reader_;
  DSpineConfig cfg_;
};

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_DSPINE_REFERENCE_HPP_
