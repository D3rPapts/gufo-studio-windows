#include "ming_tts/safetensors.hpp"

#include <cmath>
#include <stdexcept>

#include <gufo/ext/json.hpp>

namespace gufo::models::ming_tts {
namespace {

namespace json = gufo::json;

std::size_t ElementSize(DType dtype) {
  switch (dtype) {
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kBF16:
    case DType::kF16:
      return 2;
    case DType::kI64:
      return 8;
    case DType::kUnknown:
      break;
  }
  return 0;
}

DType ParseDType(const std::string& raw) {
  if (raw == "F32") {
    return DType::kF32;
  }
  if (raw == "BF16") {
    return DType::kBF16;
  }
  if (raw == "F16") {
    return DType::kF16;
  }
  if (raw == "I64") {
    return DType::kI64;
  }
  if (raw == "I32") {
    return DType::kI32;
  }
  return DType::kUnknown;
}

bool ToU64(const json::Value& value, std::uint64_t* out) {
  if (!value.is_number()) {
    return false;
  }
  const double number = value.as_double();
  if (!std::isfinite(number) || number < 0.0 || std::floor(number) != number) {
    return false;
  }
  *out = static_cast<std::uint64_t>(number);
  return true;
}

float HalfToF32(std::uint16_t bits) {
  const std::uint32_t sign = (bits >> 15U) & 1U;
  const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
  const std::uint32_t mantissa = bits & 0x3FFU;
  float value = 0.0F;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(static_cast<float>(mantissa | 0x400U),
                       static_cast<int>(exponent) - 25);
  }
  return sign != 0U ? -value : value;
}

}  // namespace

std::int64_t TensorView::NumElements() const {
  std::int64_t count = 1;
  for (const std::int64_t dim : shape) {
    count *= dim;
  }
  return count;
}

std::unique_ptr<SafetensorsFile> SafetensorsFile::Open(
    const std::filesystem::path& path, std::string* error) {
  auto set_error = [error](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
  };
  auto result = std::make_unique<SafetensorsFile>();
  std::string map_error;
  result->file_ = platform::ReadOnlyMappedFile::Open(path, &map_error);
  if (!result->file_.valid() || result->file_.size() < 8) {
    set_error("cannot map " + path.string() + ": " + map_error);
    return nullptr;
  }
  const std::uint8_t* base = result->file_.data();
  std::uint64_t header_bytes = 0;
  for (int index = 0; index < 8; ++index) {
    header_bytes |= static_cast<std::uint64_t>(base[index]) << (8 * index);
  }
  if (header_bytes == 0 || header_bytes > result->file_.size() - 8) {
    set_error("invalid safetensors header in " + path.string());
    return nullptr;
  }
  json::Value root;
  try {
    root = json::parse(std::string_view(reinterpret_cast<const char*>(base + 8),
                                        static_cast<std::size_t>(header_bytes)));
  } catch (const std::exception& exception) {
    set_error("invalid safetensors header JSON: " +
              std::string(exception.what()));
    return nullptr;
  }
  if (!root.is_object()) {
    set_error("safetensors header must be an object");
    return nullptr;
  }
  const std::size_t data_offset = 8 + static_cast<std::size_t>(header_bytes);
  const std::size_t data_bytes = result->file_.size() - data_offset;
  for (const auto& [name, entry] : root.members()) {
    if (name == "__metadata__") {
      continue;
    }
    const auto* dtype = entry.find("dtype");
    const auto* shape = entry.find("shape");
    const auto* offsets = entry.find("data_offsets");
    if (dtype == nullptr || shape == nullptr || !shape->is_array() ||
        offsets == nullptr || !offsets->is_array() || offsets->size() != 2) {
      set_error("malformed safetensors entry " + name);
      return nullptr;
    }
    TensorView view;
    view.name = name;
    view.dtype = ParseDType(dtype->str());
    for (const auto& dim : shape->items()) {
      std::uint64_t value = 0;
      if (!ToU64(dim, &value)) {
        set_error("malformed shape for " + name);
        return nullptr;
      }
      view.shape.push_back(static_cast<std::int64_t>(value));
    }
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    if (!ToU64(offsets->items()[0], &begin) ||
        !ToU64(offsets->items()[1], &end) || end < begin || end > data_bytes) {
      set_error("malformed offsets for " + name);
      return nullptr;
    }
    view.data = base + data_offset + begin;
    view.byte_size = static_cast<std::size_t>(end - begin);
    const std::size_t element = ElementSize(view.dtype);
    if (element != 0 &&
        static_cast<std::size_t>(view.NumElements()) * element !=
            view.byte_size) {
      set_error("size mismatch for " + name);
      return nullptr;
    }
    result->tensors_.emplace(name, std::move(view));
  }
  return result;
}

const TensorView* SafetensorsFile::Find(std::string_view name) const {
  const auto found = tensors_.find(std::string(name));
  return found == tensors_.end() ? nullptr : &found->second;
}

const TensorView& SafetensorsFile::Require(std::string_view name) const {
  const TensorView* view = Find(name);
  if (view == nullptr) {
    throw std::runtime_error("missing tensor " + std::string(name));
  }
  return *view;
}

std::vector<float> SafetensorsFile::ToF32(std::string_view name) const {
  const TensorView& view = Require(name);
  const auto count = static_cast<std::size_t>(view.NumElements());
  std::vector<float> out(count);
  for (std::size_t index = 0; index < count; ++index) {
    switch (view.dtype) {
      case DType::kF32: {
        std::memcpy(&out[index], view.data + index * 4, 4);
        break;
      }
      case DType::kBF16: {
        std::uint16_t bits = 0;
        std::memcpy(&bits, view.data + index * 2, 2);
        out[index] = Bf16ToF32(bits);
        break;
      }
      case DType::kF16: {
        std::uint16_t bits = 0;
        std::memcpy(&bits, view.data + index * 2, 2);
        out[index] = HalfToF32(bits);
        break;
      }
      case DType::kI64: {
        std::int64_t value = 0;
        std::memcpy(&value, view.data + index * 8, 8);
        out[index] = static_cast<float>(value);
        break;
      }
      case DType::kI32: {
        std::int32_t value = 0;
        std::memcpy(&value, view.data + index * 4, 4);
        out[index] = static_cast<float>(value);
        break;
      }
      case DType::kUnknown:
        throw std::runtime_error("unsupported dtype for " + std::string(name));
    }
  }
  return out;
}

std::vector<std::int64_t> SafetensorsFile::ToI64(std::string_view name) const {
  const TensorView& view = Require(name);
  const auto count = static_cast<std::size_t>(view.NumElements());
  std::vector<std::int64_t> out(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (view.dtype == DType::kI64) {
      std::memcpy(&out[index], view.data + index * 8, 8);
    } else if (view.dtype == DType::kI32) {
      std::int32_t value = 0;
      std::memcpy(&value, view.data + index * 4, 4);
      out[index] = value;
    } else {
      throw std::runtime_error("integer tensor expected: " + std::string(name));
    }
  }
  return out;
}

}  // namespace gufo::models::ming_tts
