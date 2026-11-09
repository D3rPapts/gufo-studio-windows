#ifndef GUFO_MODELS_QWEN_DSPINE_CONFIG_HPP_
#define GUFO_MODELS_QWEN_DSPINE_CONFIG_HPP_

// Topology of a DSpine draft GGUF (architecture "dspine", written by
// tools/dspine/convert_dspine_gguf.py from tencent/Qwen3.8-27B-DSpine-b16).
// Shared by the CPU reference and the HIP runner.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen {

struct DSpineConfig {
  std::uint32_t layers{0}, hidden{0}, ffn{0}, heads{0}, kv_heads{0};
  std::uint32_t head_dim{0}, block_size{0}, sliding_window{0};
  std::uint32_t mask_token{0}, code_size{0}, rank{0}, head_topk{0};
  std::uint32_t vocab{0};
  float rms_eps{1e-6f}, bridge_eps{1e-6f}, rope_theta{1e7f};
  /// Zero-based target layer OUTPUTS the draft reads (dspine_mlx hooks
  /// `layers[id]` and stores its return value), concatenated in this order.
  /// gufo's capture uses the same numbering, so no -1 as for DFlash-2.
  std::vector<std::uint32_t> target_layers;

  [[nodiscard]] std::size_t FeatureWidth() const noexcept {
    return target_layers.size() * static_cast<std::size_t>(hidden);
  }
};

/// Reads and validates the dspine.* metadata. nullopt + message on error.
[[nodiscard]] std::optional<DSpineConfig> ReadDSpineConfig(
    const core::GgufReader& reader, std::string* error);

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_DSPINE_CONFIG_HPP_
