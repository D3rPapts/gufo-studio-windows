#include "src/models/minimax_h3/lora.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <unordered_set>

#include "src/models/minimax_h3/json.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr std::uint8_t kBf16 = 1;
constexpr std::uint8_t kF16 = 2;
constexpr std::uint8_t kF32 = 3;
constexpr std::uint64_t kMaximumHeaderBytes = 64ULL << 20U;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::uint8_t ParseDtype(const std::string& dtype) {
  if (dtype == "BF16") {
    return kBf16;
  }
  if (dtype == "F16") {
    return kF16;
  }
  if (dtype == "F32") {
    return kF32;
  }
  return 0;
}

std::size_t DtypeBytes(std::uint8_t dtype) {
  return dtype == kF32 ? 4U : 2U;
}

float HalfToFloat(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
  const std::uint32_t exponent = (half >> 10U) & 0x1FU;
  const std::uint32_t mantissa = half & 0x3FFU;
  if (exponent == 0) {
    const float value = std::ldexp(static_cast<float>(mantissa), -24);
    return sign != 0 ? -value : value;
  }
  if (exponent == 31) {
    return std::bit_cast<float>(sign | 0x7F800000U | (mantissa << 13U));
  }
  return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) |
                              (mantissa << 13U));
}

void Widen(const std::vector<unsigned char>& bytes, std::uint8_t dtype,
           std::vector<float>* output) {
  const std::size_t count = bytes.size() / DtypeBytes(dtype);
  output->resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (dtype == kF32) {
      std::memcpy(&(*output)[index], bytes.data() + index * 4U, 4U);
    } else {
      std::uint16_t value = 0;
      std::memcpy(&value, bytes.data() + index * 2U, 2U);
      (*output)[index] =
          dtype == kBf16
              ? std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U)
              : HalfToFloat(value);
    }
  }
}

bool ReadAt(const std::filesystem::path& path, std::uint64_t offset,
            std::size_t bytes, std::vector<unsigned char>* output) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  input.seekg(static_cast<std::streamoff>(offset));
  output->resize(bytes);
  input.read(reinterpret_cast<char*>(output->data()),
             static_cast<std::streamsize>(bytes));
  return static_cast<std::size_t>(input.gcount()) == bytes;
}

struct RawTensor {
  std::uint8_t dtype{0};
  std::vector<std::uint64_t> shape;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

bool StripSuffix(std::string_view* name, std::string_view suffix) {
  if (name->ends_with(suffix)) {
    name->remove_suffix(suffix.size());
    return true;
  }
  return false;
}

}  // namespace

std::shared_ptr<const LoraAdapter> LoraAdapter::Open(
    const std::filesystem::path& path, float strength, std::string* error) {
  if (!std::isfinite(strength)) {
    SetError(error, "MiniMax H3 LoRA strength must be finite");
    return nullptr;
  }
  std::vector<unsigned char> length_bytes;
  if (!ReadAt(path, 0, 8, &length_bytes)) {
    SetError(error, "cannot read MiniMax H3 LoRA " + path.string());
    return nullptr;
  }
  std::uint64_t header_bytes = 0;
  for (int index = 7; index >= 0; --index) {
    header_bytes =
        (header_bytes << 8U) | length_bytes[static_cast<std::size_t>(index)];
  }
  std::vector<unsigned char> header;
  if (header_bytes == 0 || header_bytes > kMaximumHeaderBytes ||
      !ReadAt(path, 8, static_cast<std::size_t>(header_bytes), &header)) {
    SetError(error, "MiniMax H3 LoRA has an invalid safetensors header: " +
                        path.string());
    return nullptr;
  }
  auto adapter = std::make_shared<LoraAdapter>();
  adapter->path_ = path;
  adapter->strength_ = strength;
  adapter->payload_offset_ = 8U + header_bytes;
  std::unordered_map<std::string, RawTensor> downs;
  std::unordered_map<std::string, RawTensor> ups;
  std::unordered_map<std::string, RawTensor> alphas;
  try {
    const json::Value root = json::Parse(std::string_view(
        reinterpret_cast<const char*>(header.data()), header.size()));
    for (const auto& [key, value] : root.AsObject()) {
      if (key == "__metadata__") {
        continue;
      }
      const json::Value* dtype = value.Find("dtype");
      const json::Value* shape = value.Find("shape");
      const json::Value* data_offsets = value.Find("data_offsets");
      if (dtype == nullptr || shape == nullptr || data_offsets == nullptr) {
        throw json::Error("incomplete tensor entry " + key);
      }
      RawTensor tensor;
      tensor.dtype = ParseDtype(dtype->AsString());
      for (const json::Value& dimension : shape->AsArray()) {
        tensor.shape.push_back(dimension.AsUint64());
      }
      const auto& offsets = data_offsets->AsArray();
      if (offsets.size() != 2 || tensor.dtype == 0) {
        throw json::Error("unsupported tensor " + key);
      }
      tensor.begin = offsets[0].AsUint64();
      tensor.end = offsets[1].AsUint64();
      std::string_view module = key;
      for (const std::string_view prefix :
           {"model.diffusion_model.", "diffusion_model.", "transformer."}) {
        if (module.starts_with(prefix)) {
          module.remove_prefix(prefix.size());
          break;
        }
      }
      if (StripSuffix(&module, ".lora_A.weight") ||
          StripSuffix(&module, ".lora_down.weight")) {
        downs.emplace(std::string(module), std::move(tensor));
      } else if (StripSuffix(&module, ".lora_B.weight") ||
                 StripSuffix(&module, ".lora_up.weight")) {
        ups.emplace(std::string(module), std::move(tensor));
      } else if (StripSuffix(&module, ".alpha")) {
        alphas.emplace(std::string(module), std::move(tensor));
      } else {
        throw json::Error("unsupported LoRA tensor " + key +
                          " (only lora_A/lora_B/alpha factors are applied)");
      }
    }
  } catch (const std::exception& exception) {
    SetError(error,
             "MiniMax H3 LoRA header failed: " + std::string(exception.what()));
    return nullptr;
  }
  for (auto& [module, down] : downs) {
    const auto up = ups.find(module);
    if (up == ups.end() || down.shape.size() != 2 ||
        up->second.shape.size() != 2 || down.shape[0] != up->second.shape[1] ||
        down.shape[0] == 0 ||
        (down.end - down.begin) !=
            down.shape[0] * down.shape[1] * DtypeBytes(down.dtype) ||
        (up->second.end - up->second.begin) !=
            up->second.shape[0] * up->second.shape[1] *
                DtypeBytes(up->second.dtype)) {
      SetError(error, "MiniMax H3 LoRA factor pair is malformed: " + module);
      return nullptr;
    }
    LoraFactor factor;
    factor.rank = down.shape[0];
    factor.input = down.shape[1];
    factor.output = up->second.shape[0];
    factor.down_offset = adapter->payload_offset_ + down.begin;
    factor.up_offset = adapter->payload_offset_ + up->second.begin;
    factor.down_dtype = down.dtype;
    factor.up_dtype = up->second.dtype;
    float alpha = static_cast<float>(factor.rank);
    if (const auto found = alphas.find(module); found != alphas.end()) {
      std::vector<unsigned char> bytes;
      std::vector<float> value;
      if (found->second.end - found->second.begin !=
              DtypeBytes(found->second.dtype) ||
          !ReadAt(path, adapter->payload_offset_ + found->second.begin,
                  DtypeBytes(found->second.dtype), &bytes)) {
        SetError(error, "MiniMax H3 LoRA alpha is malformed: " + module);
        return nullptr;
      }
      Widen(bytes, found->second.dtype, &value);
      alpha = value.front();
    }
    factor.scale = strength * alpha / static_cast<float>(factor.rank);
    adapter->factors_.emplace(module, factor);
  }
  if (adapter->factors_.empty() || adapter->factors_.size() != ups.size()) {
    SetError(error, "MiniMax H3 LoRA has no complete lora_A/lora_B pairs: " +
                        path.string());
    return nullptr;
  }
  return adapter;
}

const LoraFactor* LoraAdapter::Find(std::string_view module) const noexcept {
  const auto found = factors_.find(std::string(module));
  return found == factors_.end() ? nullptr : &found->second;
}

bool LoraAdapter::Read(const LoraFactor& factor, std::vector<float>* down,
                       std::vector<float>* up, std::string* error) const {
  std::vector<unsigned char> bytes;
  const std::size_t down_bytes =
      factor.rank * factor.input * DtypeBytes(factor.down_dtype);
  const std::size_t up_bytes =
      factor.output * factor.rank * DtypeBytes(factor.up_dtype);
  if (!ReadAt(path_, factor.down_offset, down_bytes, &bytes)) {
    SetError(error, "short read in MiniMax H3 LoRA " + path_.string());
    return false;
  }
  Widen(bytes, factor.down_dtype, down);
  if (!ReadAt(path_, factor.up_offset, up_bytes, &bytes)) {
    SetError(error, "short read in MiniMax H3 LoRA " + path_.string());
    return false;
  }
  Widen(bytes, factor.up_dtype, up);
  return true;
}

bool MergeLoraModule(const LoraAdapter* adapter, std::string_view module,
                     void* weight, std::uint64_t output, std::uint64_t input,
                     void* stream, std::string* error) {
  if (adapter == nullptr) {
    return true;
  }
  const LoraFactor* factor = adapter->Find(module);
  if (factor == nullptr) {
    return true;
  }
  if (factor->output != output || factor->input != input) {
    SetError(error, "MiniMax H3 LoRA shape differs for " + std::string(module));
    return false;
  }
  std::vector<float> down;
  std::vector<float> up;
  return adapter->Read(*factor, &down, &up, error) &&
         HipMergeLoraBf16(weight, *factor, down, up, stream, error);
}

#if !defined(ENGINE_ENABLE_HIP)
bool HipMergeLoraBf16(void*, const LoraFactor&, const std::vector<float>&,
                      const std::vector<float>&, void*, std::string* error) {
  SetError(error, "MiniMax H3 LoRA merging requires ENGINE_ENABLE_HIP");
  return false;
}
#endif

}  // namespace gufo::minimax_h3
