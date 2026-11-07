// Small safetensors reader for the CCIP extension: F32 and U8 tensors, copied
// into owned, aligned buffers (the files are 12-150 MB; no mapping needed).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_TENSORS_HPP_
#define GUFO_MODELS_CCIP_TENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::ccip {

struct Tensor {
  std::vector<std::int64_t> shape;
  std::vector<float> f32;         // F32 tensors
  std::vector<std::uint8_t> u8;   // U8 tensors
  [[nodiscard]] std::int64_t numel() const;
};

class TensorFile {
public:
  // Reads every tensor. False with *error set on any problem.
  bool Load(const std::filesystem::path& path, std::string* error);
  [[nodiscard]] const Tensor* Find(std::string_view name) const;
  // Throws std::runtime_error when missing or not F32 / wrong element count.
  [[nodiscard]] const std::vector<float>& F32(std::string_view name,
                                              std::int64_t expect_numel = -1) const;
  [[nodiscard]] const std::map<std::string, Tensor, std::less<>>& all() const {
    return tensors_;
  }

private:
  std::map<std::string, Tensor, std::less<>> tensors_;
};

}  // namespace gufo::models::ccip

#endif  // GUFO_MODELS_CCIP_TENSORS_HPP_
