// Minimal memory-mapped safetensors reader for Ming-3.0-tts (weights) and its
// reference fixtures (F32/I64 tensors). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_SAFETENSORS_HPP_
#define GUFO_MODELS_MING_TTS_SAFETENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/platform/mapped_file.hpp"

namespace gufo::models::ming_tts {

enum class DType : std::uint8_t { kF32, kBF16, kF16, kI64, kI32, kUnknown };

struct TensorView {
  std::string name;
  DType dtype{DType::kUnknown};
  std::vector<std::int64_t> shape;
  const std::uint8_t* data{nullptr};
  std::size_t byte_size{0};

  [[nodiscard]] std::int64_t NumElements() const;
};

class SafetensorsFile {
public:
  [[nodiscard]] static std::unique_ptr<SafetensorsFile> Open(
      const std::filesystem::path& path, std::string* error = nullptr);

  [[nodiscard]] const TensorView* Find(std::string_view name) const;
  // Throws std::runtime_error when the tensor is missing.
  [[nodiscard]] const TensorView& Require(std::string_view name) const;
  // Any float dtype converted to F32; integer dtypes converted by value.
  [[nodiscard]] std::vector<float> ToF32(std::string_view name) const;
  [[nodiscard]] std::vector<std::int64_t> ToI64(std::string_view name) const;
  [[nodiscard]] const std::unordered_map<std::string, TensorView>& tensors()
      const noexcept {
    return tensors_;
  }

private:
  platform::ReadOnlyMappedFile file_;
  std::unordered_map<std::string, TensorView> tensors_;
};

[[nodiscard]] inline float Bf16ToF32(std::uint16_t bits) noexcept {
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
  float value;
  static_assert(sizeof(value) == sizeof(widened));
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_SAFETENSORS_HPP_
