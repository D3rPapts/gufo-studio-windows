// Written with AI assistance (Claude Code).
#include "ccip/tensors.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include <gufo/ext/json.hpp>

namespace gufo::models::ccip {

std::int64_t Tensor::numel() const {
  std::int64_t n = 1;
  for (const auto d : shape) n *= d;
  return n;
}

bool TensorFile::Load(const std::filesystem::path& path, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = path.string() + ": " + message;
    return false;
  };
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open");
  in.seekg(0, std::ios::end);
  const auto file_size = static_cast<std::uint64_t>(in.tellg());
  in.seekg(0);
  std::uint64_t header_size = 0;
  if (file_size < 8 || !in.read(reinterpret_cast<char*>(&header_size), 8)) {
    return fail("short file");
  }
  if (header_size > file_size - 8 || header_size > (64u << 20)) {
    return fail("bad header size");
  }
  std::string header(static_cast<std::size_t>(header_size), '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(header_size))) {
    return fail("short header");
  }
  const std::uint64_t data_start = 8 + header_size;
  json::Value root;
  try {
    root = json::parse(header);
  } catch (const std::exception& e) {
    return fail(std::string("header: ") + e.what());
  }
  if (!root.is_object()) return fail("header is not an object");
  for (const auto& [name, info] : root.members()) {
    if (name == "__metadata__") continue;
    const std::string dtype = info.member_str("dtype");
    const json::Value* shape = info.find("shape");
    const json::Value* offsets = info.find("data_offsets");
    if (shape == nullptr || offsets == nullptr || !shape->is_array() ||
        !offsets->is_array() || offsets->size() != 2) {
      return fail("bad entry " + name);
    }
    Tensor t;
    for (const auto& d : shape->items()) t.shape.push_back(static_cast<std::int64_t>(d.as_size()));
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    if (end < begin || data_start + end > file_size) return fail("bad offsets " + name);
    const std::uint64_t bytes = end - begin;
    const auto n = static_cast<std::uint64_t>(t.numel());
    in.seekg(static_cast<std::streamoff>(data_start + begin));
    if (dtype == "F32") {
      if (bytes != n * 4) return fail("size mismatch " + name);
      t.f32.resize(static_cast<std::size_t>(n));
      if (!in.read(reinterpret_cast<char*>(t.f32.data()), static_cast<std::streamsize>(bytes))) {
        return fail("short data " + name);
      }
    } else if (dtype == "U8") {
      if (bytes != n) return fail("size mismatch " + name);
      t.u8.resize(static_cast<std::size_t>(n));
      if (!in.read(reinterpret_cast<char*>(t.u8.data()), static_cast<std::streamsize>(bytes))) {
        return fail("short data " + name);
      }
    } else {
      return fail("unsupported dtype " + dtype + " for " + name);
    }
    tensors_.emplace(name, std::move(t));
  }
  return true;
}

const Tensor* TensorFile::Find(std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const std::vector<float>& TensorFile::F32(std::string_view name,
                                          std::int64_t expect_numel) const {
  const Tensor* t = Find(name);
  if (t == nullptr) throw std::runtime_error("missing tensor " + std::string(name));
  if (t->f32.empty() && t->numel() != 0) {
    throw std::runtime_error("tensor is not F32: " + std::string(name));
  }
  if (expect_numel >= 0 && t->numel() != expect_numel) {
    throw std::runtime_error("unexpected size for " + std::string(name) + ": " +
                             std::to_string(t->numel()) + " != " + std::to_string(expect_numel));
  }
  return t->f32;
}

}  // namespace gufo::models::ccip
