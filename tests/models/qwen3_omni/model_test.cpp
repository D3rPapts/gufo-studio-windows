#include "src/models/qwen3_omni/model.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "tests/support/temporary_directory.hpp"

namespace {

namespace omni = gufo::models::qwen3_omni;
using gufo::core::GgmlType;
using gufo::core::GgufReader;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

/// A GGUF v3 file with no metadata and the given tensors; payload bytes
/// count up from `seed` so every offset is distinguishable.
struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  GgmlType type;
  std::uint64_t offset;
  std::size_t bytes;
};

std::vector<std::uint8_t> BuildGguf(const std::vector<Tensor>& tensors,
                                    std::size_t payload) {
  std::vector<std::uint8_t> out;
  const auto append = [&](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    out.insert(out.end(), bytes, bytes + size);
  };
  const auto pod = [&](auto value) { append(&value, sizeof(value)); };
  append("GGUF", 4);
  pod(std::uint32_t{3});
  pod(static_cast<std::uint64_t>(tensors.size()));
  pod(std::uint64_t{0});
  for (const Tensor& tensor : tensors) {
    pod(static_cast<std::uint64_t>(tensor.name.size()));
    append(tensor.name.data(), tensor.name.size());
    pod(static_cast<std::uint32_t>(tensor.dims.size()));
    for (const std::uint64_t dimension : tensor.dims) {
      pod(dimension);
    }
    pod(static_cast<std::uint32_t>(tensor.type));
    pod(tensor.offset);
  }
  out.resize((out.size() + 31U) / 32U * 32U, 0);
  const std::size_t start = out.size();
  out.resize(start + payload);
  for (std::size_t index = 0; index < payload; ++index) {
    out[start + index] = static_cast<std::uint8_t>((index * 131U + 7U) >> 1U);
  }
  return out;
}

omni::Weight View(const GgufReader& reader, std::string_view name) {
  const auto* info = reader.FindTensor(name);
  Expect(info != nullptr, "tensor present");
  omni::Weight weight;
  weight.data = info->data;
  weight.type = info->type;
  weight.cols = info->dimensions[0];
  weight.rows = info->dimensions.size() > 1 ? info->dimensions[1] : 1;
  weight.row_bytes = omni::RowBytes(info->type, weight.cols);
  Expect(weight.Bytes() == info->size_bytes, "weight view covers the tensor");
  return weight;
}

}  // namespace

int main() {
  // Offsets straddle 4096-byte sectors; the last tensor ends at the end of
  // the file, inside a partial sector.
  const std::vector<Tensor> tensors = {
      {"a", {1280, 7}, GgmlType::kF16, 0, 1280 * 7 * 2},
      {"b", {96, 3}, GgmlType::kF32, 17952, 96 * 3 * 4},
      {"c", {5120, 3}, GgmlType::kF16, 19104, 5120 * 3 * 2},
  };
  const std::size_t payload = 19104 + 5120 * 3 * 2;
  const auto image = BuildGguf(tensors, payload);

  TemporaryDirectory directory("qwen3-omni-model-test");
  const auto path = directory.path() / "weights.gguf";
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(image.data()),
               static_cast<std::streamsize>(image.size()));
    Expect(file.good(), "fixture written");
  }

  std::string error;
  {
    const auto reader = GgufReader::OpenFile(path, &error);
    Expect(reader != nullptr, "fixture opens: " + error);
    std::vector<std::byte> scratch;
    for (const Tensor& tensor : tensors) {
      const omni::Weight weight = View(*reader, tensor.name);
      const auto bytes = omni::ReadWeightBytes(*reader, weight, &scratch);
      Expect(bytes.size() == tensor.bytes, tensor.name + ": size");
      Expect(bytes.data() != weight.data,
             tensor.name + ": read from the file, not the mapping");
      Expect(reinterpret_cast<std::uintptr_t>(bytes.data()) % 4U == 0U,
             tensor.name + ": element aligned");
      Expect(std::memcmp(bytes.data(), weight.data, bytes.size()) == 0,
             tensor.name + ": bytes match the mapping");
    }
  }

  // A memory image has no file: the mapped bytes are returned as they are.
  const auto memory = GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(memory != nullptr, "memory image opens: " + error);
  std::vector<std::byte> scratch;
  const omni::Weight weight = View(*memory, "c");
  const auto bytes = omni::ReadWeightBytes(*memory, weight, &scratch);
  Expect(bytes.data() == weight.data && bytes.size() == weight.Bytes(),
         "memory image falls back to the mapped view");

  std::cout << "qwen3_omni_model_test passed\n";
  return 0;
}
