// TRELLIS.2: minimal NumPy .npy (format 1.0) writer/reader for float32 arrays, used by
// the probe tool and the comparison against the PyTorch reference dumps.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gufo::trellis2::npy {

inline bool write_f32(const std::filesystem::path& path, const float* data, const std::vector<std::size_t>& shape,
                      std::string* error) {
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) dict += std::to_string(shape[i]) + (shape.size() == 1 ? "," : i + 1 < shape.size() ? ", " : "");
    dict += "), }";
    while ((10 + dict.size() + 1) % 64 != 0) dict.push_back(' ');
    dict.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "cannot write " + path.string();
        return false;
    }
    const char magic[] = "\x93NUMPY\x01\x00";
    out.write(magic, 8);
    const auto len = static_cast<std::uint16_t>(dict.size());
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(dict.data(), static_cast<std::streamsize>(dict.size()));
    std::size_t count = 1;
    for (auto s : shape) count *= s;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * 4));
    if (!out) {
        if (error) *error = "write failed for " + path.string();
        return false;
    }
    return true;
}

// Reads a C-order little-endian float32 array; shape returned in *shape.
inline bool read_f32(const std::filesystem::path& path, std::vector<float>* data, std::vector<std::size_t>* shape,
                     std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = "cannot read " + path.string();
        return false;
    }
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        if (error) *error = "not a .npy file: " + path.string();
        return false;
    }
    const int major = magic[6];
    std::uint32_t len = 0;
    if (major == 1) {
        std::uint16_t l16 = 0;
        in.read(reinterpret_cast<char*>(&l16), 2);
        len = l16;
    } else {
        in.read(reinterpret_cast<char*>(&len), 4);
    }
    std::string dict(len, '\0');
    in.read(dict.data(), len);
    if (dict.find("'<f4'") == std::string::npos || dict.find("'fortran_order': False") == std::string::npos) {
        if (error) *error = "unsupported .npy dtype/order in " + path.string() + ": " + dict;
        return false;
    }
    const std::size_t open = dict.find("'shape': (");
    const std::size_t close = dict.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        if (error) *error = "bad .npy header: " + dict;
        return false;
    }
    shape->clear();
    std::string nums = dict.substr(open + 10, close - open - 10);
    std::size_t pos = 0;
    while (pos < nums.size()) {
        while (pos < nums.size() && (nums[pos] == ' ' || nums[pos] == ',')) ++pos;
        if (pos >= nums.size()) break;
        std::size_t end = pos;
        while (end < nums.size() && nums[end] >= '0' && nums[end] <= '9') ++end;
        shape->push_back(static_cast<std::size_t>(std::stoull(nums.substr(pos, end - pos))));
        pos = end;
    }
    std::size_t count = 1;
    for (auto s : *shape) count *= s;
    data->resize(count);
    in.read(reinterpret_cast<char*>(data->data()), static_cast<std::streamsize>(count * 4));
    if (!in) {
        if (error) *error = "short .npy data in " + path.string();
        return false;
    }
    return true;
}

}  // namespace gufo::trellis2::npy
