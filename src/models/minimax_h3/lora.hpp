#ifndef GUFO_MODELS_MINIMAX_H3_LORA_HPP_
#define GUFO_MODELS_MINIMAX_H3_LORA_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::minimax_h3 {

/// One low-rank update W += scale * up(out x rank) * down(rank x in).
struct LoraFactor {
  std::uint64_t rank{0};
  std::uint64_t input{0};
  std::uint64_t output{0};
  float scale{0.0F};
  std::uint64_t down_offset{0};
  std::uint64_t up_offset{0};
  std::uint8_t down_dtype{0};
  std::uint8_t up_dtype{0};
};

/// A transformer LoRA in the original H3 / ComfyUI layout
/// (`diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight`, `.lora_B.weight`,
/// optional `.alpha`). Kohya `lora_down`/`lora_up` names are accepted too.
/// Module names are the checkpoint's own, e.g. `blocks.3.mlp.fc1`. The scale
/// of a factor is strength * alpha / rank (alpha defaults to rank).
class LoraAdapter {
public:
  [[nodiscard]] static std::shared_ptr<const LoraAdapter> Open(
      const std::filesystem::path& path, float strength,
      std::string* error = nullptr);

  [[nodiscard]] const LoraFactor* Find(std::string_view module) const noexcept;
  /// Reads one factor as F32: down is rank x input, up is output x rank.
  /// Safe to call concurrently.
  [[nodiscard]] bool Read(const LoraFactor& factor, std::vector<float>* down,
                          std::vector<float>* up,
                          std::string* error = nullptr) const;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }
  [[nodiscard]] float strength() const noexcept { return strength_; }
  [[nodiscard]] std::size_t modules() const noexcept { return factors_.size(); }

private:
  std::filesystem::path path_;
  float strength_{1.0F};
  std::uint64_t payload_offset_{0};
  std::unordered_map<std::string, LoraFactor> factors_;
};

/// Adds the factor to a row-major BF16 weight (output x input) in device
/// memory. `stream` is a hipStream_t. Synchronizes the stream.
[[nodiscard]] bool HipMergeLoraBf16(void* weight, const LoraFactor& factor,
                                    const std::vector<float>& down,
                                    const std::vector<float>& up, void* stream,
                                    std::string* error = nullptr);

/// Reads and merges the named module's factor when the adapter has one.
/// Returns true without changes when `adapter` is null or lacks the module.
[[nodiscard]] bool MergeLoraModule(const LoraAdapter* adapter,
                                   std::string_view module, void* weight,
                                   std::uint64_t output, std::uint64_t input,
                                   void* stream, std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_LORA_HPP_
