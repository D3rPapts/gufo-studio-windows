// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_TENSOR_FILE_HPP_
#define GUFO_MODELS_WAN22_TENSOR_FILE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/core/platform/mapped_file.hpp"

namespace gufo::models::wan22 {

/// Host reader for one small safetensors file (audio encoder, adapter
/// patches): F16, BF16 and F32 tensors read as FP32 or rounded to BF16.
class TensorFile {
public:
  TensorFile() = default;
  ~TensorFile();
  TensorFile(const TensorFile&) = delete;
  TensorFile& operator=(const TensorFile&) = delete;

  bool Open(const std::filesystem::path& path, std::string* error);
  /// `prefix` is tried in front of `name` first (e.g. "wav2vec2.").
  [[nodiscard]] bool Has(const std::string& name) const;
  [[nodiscard]] std::vector<std::size_t> Shape(const std::string& name) const;
  /// Exactly `count` values (0 = any count) as FP32.
  bool ReadF32(const std::string& name, std::size_t count,
               std::vector<float>* out, std::string* error) const;
  /// Exactly `count` values as BF16 bits (round to nearest even).
  bool ReadBf16(const std::string& name, std::size_t count,
                std::vector<std::uint16_t>* out, std::string* error) const;
  [[nodiscard]] std::string Metadata(const std::string& key) const;
  void set_prefix(std::string prefix) { prefix_ = std::move(prefix); }

private:
  struct Entry {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::uint64_t offset{0};
    std::uint64_t count{0};
  };
  const Entry* Find(const std::string& name) const;

  platform::PlatformFile file_{platform::kInvalidFile};
  std::filesystem::path path_;
  std::map<std::string, Entry, std::less<>> entries_;
  std::map<std::string, std::string, std::less<>> metadata_;
  std::string prefix_;
};

/// Host conversions.
float HalfBitsToFloat(std::uint16_t h);
float Bf16BitsToFloat(std::uint16_t h);
std::uint16_t FloatToBf16Bits(float value);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TENSOR_FILE_HPP_
