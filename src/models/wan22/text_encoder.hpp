#ifndef GUFO_MODELS_WAN22_TEXT_ENCODER_HPP_
#define GUFO_MODELS_WAN22_TEXT_ENCODER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::wan22 {

/// umT5-XXL encoder of Wan 2.2 (Diffusers `text_encoder/`, BF16). Every layer
/// carries its own relative position bias; attention is unscaled. The layer
/// weights live in device memory; the 2 GiB token embedding stays in its
/// shard and only the rows of a prompt are read. Arithmetic follows the BF16
/// Transformers reference (SDPA attention) except for the residual stream,
/// which stays FP32; GEMMs, norms and softmax accumulate in FP32.
class TextEncoder {
public:
  static constexpr std::size_t kModelWidth = 4096;
  static constexpr std::size_t kMaxTokens = 512;

  /// Loads `<diffusers root>/text_encoder`. Returns null and sets `error` on
  /// failure.
  [[nodiscard]] static std::unique_ptr<TextEncoder> Load(
      const std::filesystem::path& text_encoder_dir, std::string* error);
  ~TextEncoder();
  TextEncoder(const TextEncoder&) = delete;
  TextEncoder& operator=(const TextEncoder&) = delete;

  /// Writes the last hidden state of `ids` (1..kMaxTokens, including `</s>`)
  /// as BF16 rows [ids.size(), kModelWidth] to the device buffer `output`,
  /// ordered on the encoder's stream; returns once the rows are complete.
  bool EncodeToDevice(std::span<const std::int32_t> ids, void* output,
                      std::string* error);
  /// As EncodeToDevice, copied to the host as FP32.
  bool Encode(std::span<const std::int32_t> ids, std::vector<float>* states,
              std::string* error);

  /// Device bytes held for weights and activations.
  [[nodiscard]] std::size_t device_bytes() const;

private:
  struct Impl;
  explicit TextEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TEXT_ENCODER_HPP_
