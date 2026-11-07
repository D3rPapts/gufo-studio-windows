// Written with AI assistance (Claude Code).
#include "src/models/wan22/tensor_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::models::wan22 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

// ReadFileAt until `bytes` are in or the file ends.
bool ReadAll(platform::PlatformFile file, void* output, std::size_t bytes,
             std::uint64_t offset) {
  auto* target = static_cast<unsigned char*>(output);
  while (bytes > 0) {
    const std::size_t piece = std::min<std::size_t>(bytes, 64U << 20U);
    const std::size_t done = platform::ReadFileAt(file, target, piece, offset);
    if (done == 0)
      return false;
    target += done;
    offset += done;
    bytes -= done;
  }
  return true;
}

}  // namespace

float HalfBitsToFloat(std::uint16_t h) {
  const int sign = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float value;
  if (e == 0)
    value = std::ldexp(static_cast<float>(m), -24);
  else if (e == 31)
    value = m != 0 ? NAN : INFINITY;
  else
    value = std::ldexp(static_cast<float>(1024 + m), e - 25);
  return sign != 0 ? -value : value;
}

float Bf16BitsToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

std::uint16_t FloatToBf16Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

TensorFile::~TensorFile() {
  if (platform::FileValid(file_))
    platform::CloseFile(file_);
}

bool TensorFile::Open(const std::filesystem::path& path, std::string* error) {
  path_ = path;
  std::string open_error;
  file_ = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file_)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return false;
  }
  const auto size = platform::FileSizeBytes(file_);
  std::uint64_t header_size = 0;
  if (!size || platform::ReadFileAt(file_, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file_, text.data(), text.size(), 8) != text.size()) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__") {
        if (entry.is_object())
          for (const auto& [key, value] : entry.members())
            if (value.is_string())
              metadata_[key] = value.str();
        continue;
      }
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (shape == nullptr || offsets == nullptr || offsets->size() != 2) {
        SetError(error, "invalid tensor entry " + name + " in " +
                            path.string());
        return false;
      }
      Entry e;
      e.dtype = entry.member_str("dtype");
      std::size_t width = 0;
      if (e.dtype == "F32")
        width = 4;
      else if (e.dtype == "F16" || e.dtype == "BF16")
        width = 2;
      e.count = 1;
      for (const auto& dim : shape->items()) {
        e.shape.push_back(dim.as_size());
        e.count *= dim.as_size();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (width != 0 &&
          (end - begin != e.count * width || 8 + header_size + end > *size)) {
        SetError(error, "invalid tensor extent " + name + " in " +
                            path.string());
        return false;
      }
      e.offset = 8 + header_size + begin;
      entries_[name] = std::move(e);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return false;
  }
  return true;
}

const TensorFile::Entry* TensorFile::Find(const std::string& name) const {
  if (!prefix_.empty()) {
    const auto it = entries_.find(prefix_ + name);
    if (it != entries_.end())
      return &it->second;
  }
  const auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

bool TensorFile::Has(const std::string& name) const {
  return Find(name) != nullptr;
}

std::vector<std::size_t> TensorFile::Shape(const std::string& name) const {
  const Entry* e = Find(name);
  return e == nullptr ? std::vector<std::size_t>{} : e->shape;
}

std::string TensorFile::Metadata(const std::string& key) const {
  const auto it = metadata_.find(key);
  return it == metadata_.end() ? "" : it->second;
}

bool TensorFile::ReadF32(const std::string& name, std::size_t count,
                         std::vector<float>* out, std::string* error) const {
  const Entry* e = Find(name);
  if (e == nullptr || (count != 0 && e->count != count)) {
    SetError(error, "missing or misshaped tensor " + name + " in " +
                        path_.string());
    return false;
  }
  out->resize(e->count);
  if (e->dtype == "F32") {
    if (!ReadAll(file_, out->data(), e->count * 4, e->offset)) {
      SetError(error, "short read of tensor " + name);
      return false;
    }
    return true;
  }
  if (e->dtype != "F16" && e->dtype != "BF16") {
    SetError(error, "unsupported dtype " + e->dtype + " of tensor " + name);
    return false;
  }
  std::vector<std::uint16_t> half(e->count);
  if (!ReadAll(file_, half.data(), e->count * 2, e->offset)) {
    SetError(error, "short read of tensor " + name);
    return false;
  }
  const bool bf16 = e->dtype == "BF16";
  for (std::size_t i = 0; i < half.size(); ++i)
    (*out)[i] = bf16 ? Bf16BitsToFloat(half[i]) : HalfBitsToFloat(half[i]);
  return true;
}

bool TensorFile::ReadBf16(const std::string& name, std::size_t count,
                          std::vector<std::uint16_t>* out,
                          std::string* error) const {
  const Entry* e = Find(name);
  if (e == nullptr || (count != 0 && e->count != count)) {
    SetError(error, "missing or misshaped tensor " + name + " in " +
                        path_.string());
    return false;
  }
  if (e->dtype == "BF16") {
    out->resize(e->count);
    if (!ReadAll(file_, out->data(), e->count * 2, e->offset)) {
      SetError(error, "short read of tensor " + name);
      return false;
    }
    return true;
  }
  std::vector<float> values;
  if (!ReadF32(name, count, &values, error))
    return false;
  out->resize(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    (*out)[i] = FloatToBf16Bits(values[i]);
  return true;
}

}  // namespace gufo::models::wan22
