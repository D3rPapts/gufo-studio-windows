#include "src/models/voxcpm2/vae_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <string_view>
#include <unordered_map>

#include "src/core/platform/mapped_file.hpp"
#include "src/core/platform/prefault.hpp"
#include "src/models/ming_tts/safetensors.hpp"

namespace gufo::models::voxcpm2 {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

std::uint16_t U16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8U));
}
std::uint32_t U32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8U) |
         (static_cast<std::uint32_t>(p[2]) << 16U) | (static_cast<std::uint32_t>(p[3]) << 24U);
}
std::uint64_t U64(const std::uint8_t* p) {
  return static_cast<std::uint64_t>(U32(p)) | (static_cast<std::uint64_t>(U32(p + 4)) << 32U);
}

struct ZipEntry {
  const std::uint8_t* data{nullptr};
  std::size_t size{0};
};

// Central directory of a stored (uncompressed) zip, ZIP64 aware.
bool ParseZip(const std::uint8_t* base, std::size_t total,
              std::unordered_map<std::string, ZipEntry>* out) {
  if (total < 22) {
    return false;
  }
  std::size_t eocd = total;
  const std::size_t search = std::min<std::size_t>(total, 65557U);
  for (std::size_t back = 22; back <= search; ++back) {
    if (U32(base + total - back) == 0x06054b50U) {
      eocd = total - back;
      break;
    }
  }
  if (eocd == total) {
    return false;
  }
  std::uint64_t cd_offset = U32(base + eocd + 16);
  std::uint64_t cd_size = U32(base + eocd + 12);
  if (eocd >= 20 && U32(base + eocd - 20) == 0x07064b50U) {
    const std::uint64_t z64 = U64(base + eocd - 12);
    if (z64 + 56 <= total && U32(base + z64) == 0x06064b50U) {
      cd_size = U64(base + z64 + 40);
      cd_offset = U64(base + z64 + 48);
    }
  }
  if (cd_offset + cd_size > total) {
    return false;
  }
  std::size_t pos = static_cast<std::size_t>(cd_offset);
  const std::size_t end = pos + static_cast<std::size_t>(cd_size);
  while (pos + 46 <= end && U32(base + pos) == 0x02014b50U) {
    const std::uint16_t method = U16(base + pos + 10);
    std::uint64_t size = U32(base + pos + 24);
    const std::uint16_t name_len = U16(base + pos + 28);
    const std::uint16_t extra_len = U16(base + pos + 30);
    const std::uint16_t comment_len = U16(base + pos + 32);
    std::uint64_t header = U32(base + pos + 42);
    if (pos + 46 + name_len + extra_len + comment_len > end) {
      return false;
    }
    const std::string name(reinterpret_cast<const char*>(base + pos + 46), name_len);
    const std::uint8_t* extra = base + pos + 46 + name_len;
    for (std::size_t e = 0; e + 4 <= extra_len;) {
      const std::uint16_t id = U16(extra + e);
      const std::uint16_t data_size = U16(extra + e + 2);
      if (e + 4 + data_size > extra_len) {
        break;
      }
      if (id == 0x0001U) {
        std::size_t z = e + 4;
        if (size == 0xFFFFFFFFU && z + 8 <= e + 4 + data_size) {
          size = U64(extra + z);
          z += 8;
        }
        if (U32(base + pos + 20) == 0xFFFFFFFFU && z + 8 <= e + 4 + data_size) {
          z += 8;  // compressed size
        }
        if (header == 0xFFFFFFFFU && z + 8 <= e + 4 + data_size) {
          header = U64(extra + z);
        }
      }
      e += 4 + data_size;
    }
    if (method == 0 && header + 30 <= total && U32(base + header) == 0x04034b50U) {
      const std::uint64_t data =
          header + 30 + U16(base + header + 26) + U16(base + header + 28);
      if (data + size <= total) {
        (*out)[name] = ZipEntry{base + data, static_cast<std::size_t>(size)};
      }
    }
    pos += 46 + name_len + extra_len + comment_len;
  }
  return !out->empty();
}

struct RawTensor {
  std::string dtype;
  std::string storage_key;
  std::int64_t offset{0};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride;
};

// Minimal pickle VM for torch.save state dicts (protocol 2).
bool ParsePickle(const std::uint8_t* pkl, std::size_t len,
                 std::unordered_map<std::string, RawTensor>* tensors) {
  enum class Kind { kMark, kInt, kStr, kTuple, kStorage, kTensor, kOther };
  struct Val {
    Kind kind{Kind::kOther};
    std::int64_t i{0};
    std::string s;
    std::vector<Val> tuple;
    RawTensor tensor;
  };
  auto make = [](Kind kind) {
    Val v;
    v.kind = kind;
    return v;
  };
  std::vector<Val> stack;
  std::unordered_map<std::uint32_t, Val> memo;
  std::size_t pos = 0;
  auto pop_mark = [&](std::vector<Val>* items) {
    while (!stack.empty() && stack.back().kind != Kind::kMark) {
      items->push_back(std::move(stack.back()));
      stack.pop_back();
    }
    if (stack.empty()) {
      return false;
    }
    stack.pop_back();
    std::reverse(items->begin(), items->end());
    return true;
  };
  auto set_item = [&](const Val& key, const Val& value) {
    if (key.kind == Kind::kStr && value.kind == Kind::kTensor) {
      (*tensors)[key.s] = value.tensor;
    }
  };
  while (pos < len) {
    const std::uint8_t op = pkl[pos++];
    switch (op) {
      case 0x80:  // PROTO
        ++pos;
        break;
      case '(':
        stack.push_back(make(Kind::kMark));
        break;
      case '.':
        return !tensors->empty();
      case ')':
        stack.push_back(make(Kind::kTuple));
        break;
      case '}':
      case ']':
      case 0x88:
      case 0x89:
      case 'N':
        stack.push_back(make(Kind::kOther));
        break;
      case 'K': {
        Val v = make(Kind::kInt);
        v.i = pkl[pos++];
        stack.push_back(std::move(v));
        break;
      }
      case 'M': {
        Val v = make(Kind::kInt);
        v.i = U16(pkl + pos);
        pos += 2;
        stack.push_back(std::move(v));
        break;
      }
      case 'J': {
        Val v = make(Kind::kInt);
        v.i = static_cast<std::int32_t>(U32(pkl + pos));
        pos += 4;
        stack.push_back(std::move(v));
        break;
      }
      case 0x8a: {  // LONG1
        const std::uint8_t n = pkl[pos++];
        std::int64_t value = 0;
        for (std::uint8_t b = 0; b < n && b < 8; ++b) {
          value |= static_cast<std::int64_t>(pkl[pos + b]) << (8 * b);
        }
        if (n > 0 && n < 8 && (pkl[pos + n - 1] & 0x80U) != 0) {
          value -= static_cast<std::int64_t>(1) << (8 * n);
        }
        pos += n;
        Val v = make(Kind::kInt);
        v.i = value;
        stack.push_back(std::move(v));
        break;
      }
      case 'G':  // BINFLOAT
        pos += 8;
        stack.push_back(make(Kind::kOther));
        break;
      case 'X': {
        const std::uint32_t n = U32(pkl + pos);
        pos += 4;
        if (pos + n > len) {
          return false;
        }
        Val v = make(Kind::kStr);
        v.s.assign(reinterpret_cast<const char*>(pkl + pos), n);
        pos += n;
        stack.push_back(std::move(v));
        break;
      }
      case 'c': {
        std::size_t nl1 = pos;
        while (nl1 < len && pkl[nl1] != '\n') {
          ++nl1;
        }
        std::size_t nl2 = nl1 + 1;
        while (nl2 < len && pkl[nl2] != '\n') {
          ++nl2;
        }
        if (nl2 >= len) {
          return false;
        }
        Val v = make(Kind::kStr);
        v.s = std::string(reinterpret_cast<const char*>(pkl + pos), nl1 - pos) + " " +
              std::string(reinterpret_cast<const char*>(pkl + nl1 + 1), nl2 - nl1 - 1);
        pos = nl2 + 1;
        stack.push_back(std::move(v));
        break;
      }
      case 'q':
        memo[pkl[pos++]] = stack.back();
        break;
      case 'r':
        memo[U32(pkl + pos)] = stack.back();
        pos += 4;
        break;
      case 'h':
        stack.push_back(memo[pkl[pos++]]);
        break;
      case 'j':
        stack.push_back(memo[U32(pkl + pos)]);
        pos += 4;
        break;
      case 't': {
        std::vector<Val> items;
        if (!pop_mark(&items)) {
          return false;
        }
        Val v = make(Kind::kTuple);
        v.tuple = std::move(items);
        stack.push_back(std::move(v));
        break;
      }
      case 0x85:
      case 0x86:
      case 0x87: {
        const std::size_t n = op == 0x85 ? 1U : (op == 0x86 ? 2U : 3U);
        if (stack.size() < n) {
          return false;
        }
        Val v = make(Kind::kTuple);
        v.tuple.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
          v.tuple[n - 1 - i] = std::move(stack.back());
          stack.pop_back();
        }
        stack.push_back(std::move(v));
        break;
      }
      case 'Q': {  // BINPERSID ('storage', type, key, location, size)
        if (stack.empty() || stack.back().kind != Kind::kTuple ||
            stack.back().tuple.size() < 5) {
          return false;
        }
        Val v = make(Kind::kStorage);
        v.tensor.dtype = stack.back().tuple[1].s;
        v.tensor.storage_key = stack.back().tuple[2].s;
        stack.pop_back();
        stack.push_back(std::move(v));
        break;
      }
      case 'R': {
        if (stack.size() < 2) {
          return false;
        }
        Val args = std::move(stack.back());
        stack.pop_back();
        Val fn = std::move(stack.back());
        stack.pop_back();
        if (fn.kind == Kind::kStr && fn.s == "torch._utils _rebuild_tensor_v2" &&
            args.kind == Kind::kTuple && args.tuple.size() >= 4 &&
            args.tuple[0].kind == Kind::kStorage && args.tuple[2].kind == Kind::kTuple &&
            args.tuple[3].kind == Kind::kTuple) {
          Val v = make(Kind::kTensor);
          v.tensor = args.tuple[0].tensor;
          v.tensor.offset = args.tuple[1].i;
          for (const auto& dim : args.tuple[2].tuple) {
            v.tensor.shape.push_back(dim.i);
          }
          for (const auto& dim : args.tuple[3].tuple) {
            v.tensor.stride.push_back(dim.i);
          }
          stack.push_back(std::move(v));
        } else {
          stack.push_back(make(Kind::kOther));
        }
        break;
      }
      case 's': {
        if (stack.size() < 3) {
          return false;
        }
        Val value = std::move(stack.back());
        stack.pop_back();
        Val key = std::move(stack.back());
        stack.pop_back();
        set_item(key, value);
        break;
      }
      case 'u': {
        std::vector<Val> items;
        if (!pop_mark(&items)) {
          return false;
        }
        for (std::size_t i = 0; i + 1 < items.size(); i += 2) {
          set_item(items[i], items[i + 1]);
        }
        break;
      }
      case 'a':  // APPEND
        if (!stack.empty()) {
          stack.pop_back();
        }
        break;
      case 'e': {  // APPENDS
        std::vector<Val> items;
        if (!pop_mark(&items)) {
          return false;
        }
        break;
      }
      case 'b':
        if (stack.size() < 2) {
          return false;
        }
        stack.pop_back();
        break;
      default:
        return false;
    }
  }
  return !tensors->empty();
}

bool IsContiguous(const RawTensor& t) {
  std::int64_t expect = 1;
  for (std::size_t d = t.shape.size(); d-- > 0;) {
    if (t.shape[d] != 1 && t.stride.size() == t.shape.size() && t.stride[d] != expect) {
      return false;
    }
    expect *= t.shape[d];
  }
  return true;
}

bool LoadPth(const std::filesystem::path& path, VaeWeights* raw, std::string* error) {
  std::string map_error;
  auto mapped = platform::ReadOnlyMappedFile::Open(path, &map_error);
  if (!mapped.valid()) {
    SetError(error, "cannot map " + path.string() + ": " + map_error);
    return false;
  }
  std::unordered_map<std::string, ZipEntry> entries;
  if (!ParseZip(mapped.data(), mapped.size(), &entries)) {
    SetError(error, path.string() + " is not an uncompressed torch zip archive");
    return false;
  }
  const ZipEntry* pkl = nullptr;
  std::string prefix;
  for (const auto& [name, entry] : entries) {
    if (name.ends_with("/data.pkl")) {
      pkl = &entry;
      prefix = name.substr(0, name.size() - 8);
      break;
    }
  }
  std::unordered_map<std::string, RawTensor> tensors;
  if (pkl == nullptr || !ParsePickle(pkl->data, pkl->size, &tensors)) {
    SetError(error, "cannot parse the pickle of " + path.string());
    return false;
  }
  for (const auto& [name, tensor] : tensors) {
    if (tensor.dtype != "torch FloatStorage") {
      continue;  // sr_bin_boundaries (int) comes from config.json
    }
    const auto found = entries.find(prefix + "data/" + tensor.storage_key);
    std::int64_t count = 1;
    for (const std::int64_t dim : tensor.shape) {
      count *= dim;
    }
    if (found == entries.end() || !IsContiguous(tensor) ||
        static_cast<std::size_t>((tensor.offset + count) * 4) > found->second.size) {
      SetError(error, "bad storage for " + name + " in " + path.string());
      return false;
    }
    VaeTensor out;
    out.shape = tensor.shape;
    out.data.resize(static_cast<std::size_t>(count));
    std::memcpy(out.data.data(), found->second.data + tensor.offset * 4,
                static_cast<std::size_t>(count) * 4);
    // Copied: drop the mapped pages so the file and the copy are not both
    // resident (#829).
    platform::ReleaseMappedRange(mapped.data(), mapped.size(),
                                 found->second.data + tensor.offset * 4,
                                 static_cast<std::size_t>(count) * 4);
    (*raw)[name] = std::move(out);
  }
  return true;
}

bool LoadSafetensors(const std::filesystem::path& path, VaeWeights* raw, std::string* error) {
  auto file = ming_tts::SafetensorsFile::Open(path, error);
  if (file == nullptr) {
    return false;
  }
  for (const auto& [name, view] : file->tensors()) {
    if (view.dtype == ming_tts::DType::kI32 || view.dtype == ming_tts::DType::kI64) {
      continue;
    }
    VaeTensor out;
    out.shape = view.shape;
    out.data = file->ToF32(name);
    (*raw)[name] = std::move(out);
  }
  return true;
}

}  // namespace

bool LoadVaeWeights(const std::filesystem::path& model_dir, VaeWeights* out,
                    std::string* error) {
  VaeWeights raw;
  std::error_code ec;
  const auto st = model_dir / "audiovae.safetensors";
  const auto pth = model_dir / "audiovae.pth";
  if (std::filesystem::is_regular_file(st, ec)) {
    if (!LoadSafetensors(st, &raw, error)) {
      return false;
    }
  } else if (!LoadPth(pth, &raw, error)) {
    return false;
  }
  // Fold weight norm: w = g * v / ||v|| (norm over all dims but 0).
  out->clear();
  std::set<std::string> consumed;
  for (auto& [name, g] : raw) {
    static constexpr std::string_view kG = ".weight_g";
    if (!name.ends_with(kG)) {
      continue;
    }
    const std::string base = name.substr(0, name.size() - kG.size());
    const auto v = raw.find(base + ".weight_v");
    if (v == raw.end() || g.data.empty() || v->second.data.size() % g.data.size() != 0) {
      SetError(error, "weight norm pair incomplete for " + base);
      return false;
    }
    const std::size_t rows = g.data.size();
    const std::size_t slice = v->second.data.size() / rows;
    VaeTensor folded;
    folded.shape = v->second.shape;
    folded.data.resize(v->second.data.size());
    for (std::size_t r = 0; r < rows; ++r) {
      const float* vr = v->second.data.data() + r * slice;
      double sum = 0.0;
      for (std::size_t j = 0; j < slice; ++j) {
        sum += static_cast<double>(vr[j]) * vr[j];
      }
      const float scale = g.data[r] / static_cast<float>(std::sqrt(sum));
      for (std::size_t j = 0; j < slice; ++j) {
        folded.data[r * slice + j] = vr[j] * scale;
      }
    }
    (*out)[base + ".weight"] = std::move(folded);
    // Free the raw pair now, not after the loop: the folded copy replaces it.
    std::vector<float>().swap(v->second.data);
    std::vector<float>().swap(g.data);
    consumed.insert(name);
    consumed.insert(base + ".weight_v");
  }
  for (auto& [name, tensor] : raw) {
    if (!consumed.contains(name)) {
      (*out)[name] = std::move(tensor);
    }
  }
  return true;
}

}  // namespace gufo::models::voxcpm2
