#include "indextts/pth.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <gufo/ext/platform/prefault.hpp>

namespace gufo::models::indextts {
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

// ------------------------------------------------------------- pickle ----

struct Node;
using NodePtr = std::shared_ptr<Node>;

enum class Kind { kMark, kInt, kFloat, kStr, kTuple, kList, kDict, kStorage, kTensor, kGlobal, kOther };

struct RawTensor {
  std::string dtype;
  std::string storage_key;
  std::int64_t offset{0};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride;
};

struct Node {
  Kind kind{Kind::kOther};
  std::int64_t i{0};
  std::string s;
  std::vector<NodePtr> items;                       // tuple / list
  std::vector<std::pair<NodePtr, NodePtr>> entries;  // dict (insertion order)
  RawTensor tensor;
};

NodePtr Make(Kind kind) {
  auto n = std::make_shared<Node>();
  n->kind = kind;
  return n;
}

void SetItem(const NodePtr& dict, NodePtr key, NodePtr value) {
  if (dict == nullptr || dict->kind != Kind::kDict) {
    return;
  }
  for (auto& [k, v] : dict->entries) {
    if (k->kind == key->kind && k->kind == Kind::kStr && k->s == key->s) {
      v = std::move(value);
      return;
    }
  }
  dict->entries.emplace_back(std::move(key), std::move(value));
}

NodePtr ParsePickle(const std::uint8_t* pkl, std::size_t len) {
  std::vector<NodePtr> stack;
  std::unordered_map<std::uint32_t, NodePtr> memo;
  std::size_t pos = 0;
  std::uint32_t next_memo = 0;
  auto need = [&](std::size_t n) {
    if (pos + n > len) {
      throw std::runtime_error("truncated pickle");
    }
  };
  auto pop = [&]() {
    if (stack.empty()) {
      throw std::runtime_error("pickle stack underflow");
    }
    NodePtr v = std::move(stack.back());
    stack.pop_back();
    return v;
  };
  auto pop_mark = [&]() {
    std::vector<NodePtr> items;
    while (!stack.empty() && stack.back()->kind != Kind::kMark) {
      items.push_back(std::move(stack.back()));
      stack.pop_back();
    }
    if (stack.empty()) {
      throw std::runtime_error("pickle mark missing");
    }
    stack.pop_back();
    std::reverse(items.begin(), items.end());
    return items;
  };
  auto push_str = [&](std::size_t n) {
    need(n);
    NodePtr v = Make(Kind::kStr);
    v->s.assign(reinterpret_cast<const char*>(pkl + pos), n);
    pos += n;
    stack.push_back(std::move(v));
  };
  auto push_int = [&](std::int64_t value) {
    NodePtr v = Make(Kind::kInt);
    v->i = value;
    stack.push_back(std::move(v));
  };
  auto reduce = [&](NodePtr fn, NodePtr args) -> NodePtr {
    // torch._tensor._rebuild_from_type_v2(func, new_type, args, state): tensors saved as a subclass
    // (e.g. nn.Parameter in s2mel.pth) -> apply the inner rebuild function to its args, drop the type/state.
    if (fn->kind == Kind::kGlobal && fn->s == "torch._tensor _rebuild_from_type_v2" &&
        args->kind == Kind::kTuple && args->items.size() >= 3 && args->items[0]->kind == Kind::kGlobal &&
        args->items[2]->kind == Kind::kTuple) {
      NodePtr inner_fn = args->items[0];
      NodePtr inner_args = args->items[2];
      fn = std::move(inner_fn);
      args = std::move(inner_args);
    }
    if (fn->kind == Kind::kGlobal) {
      if (fn->s == "collections OrderedDict" || fn->s == "builtins dict") {
        return Make(Kind::kDict);
      }
      if ((fn->s == "torch._utils _rebuild_tensor_v2" || fn->s == "torch._utils _rebuild_tensor") &&
          args->kind == Kind::kTuple && args->items.size() >= 4 &&
          args->items[0]->kind == Kind::kStorage) {
        NodePtr v = Make(Kind::kTensor);
        v->tensor = args->items[0]->tensor;
        v->tensor.offset = args->items[1]->i;
        for (const auto& dim : args->items[2]->items) {
          v->tensor.shape.push_back(dim->i);
        }
        for (const auto& dim : args->items[3]->items) {
          v->tensor.stride.push_back(dim->i);
        }
        return v;
      }
      if (fn->s == "torch._utils _rebuild_parameter" && args->kind == Kind::kTuple &&
          !args->items.empty()) {
        return args->items[0];
      }
    }
    return Make(Kind::kOther);
  };
  while (pos < len) {
    const std::uint8_t op = pkl[pos++];
    switch (op) {
      case 0x80:  // PROTO
        need(1);
        ++pos;
        break;
      case 0x95:  // FRAME
        need(8);
        pos += 8;
        break;
      case '(':
        stack.push_back(Make(Kind::kMark));
        break;
      case '.':
        return stack.empty() ? nullptr : stack.back();
      case ')':
        stack.push_back(Make(Kind::kTuple));
        break;
      case '}':
        stack.push_back(Make(Kind::kDict));
        break;
      case ']':
        stack.push_back(Make(Kind::kList));
        break;
      case 'd': {
        auto items = pop_mark();
        NodePtr d = Make(Kind::kDict);
        for (std::size_t k = 0; k + 1 < items.size(); k += 2) {
          SetItem(d, items[k], items[k + 1]);
        }
        stack.push_back(std::move(d));
        break;
      }
      case 'l': {
        NodePtr l = Make(Kind::kList);
        l->items = pop_mark();
        stack.push_back(std::move(l));
        break;
      }
      case 0x88:
      case 0x89:
      case 'N':
        stack.push_back(Make(Kind::kOther));
        break;
      case 'K':
        need(1);
        push_int(pkl[pos++]);
        break;
      case 'M':
        need(2);
        push_int(U16(pkl + pos));
        pos += 2;
        break;
      case 'J':
        need(4);
        push_int(static_cast<std::int32_t>(U32(pkl + pos)));
        pos += 4;
        break;
      case 0x8a: {  // LONG1
        need(1);
        const std::uint8_t n = pkl[pos++];
        need(n);
        std::int64_t value = 0;
        for (std::uint8_t b = 0; b < n && b < 8; ++b) {
          value |= static_cast<std::int64_t>(pkl[pos + b]) << (8 * b);
        }
        if (n > 0 && n < 8 && (pkl[pos + n - 1] & 0x80U) != 0) {
          value -= static_cast<std::int64_t>(1) << (8 * n);
        }
        pos += n;
        push_int(value);
        break;
      }
      case 'G':  // BINFLOAT
        need(8);
        pos += 8;
        stack.push_back(Make(Kind::kFloat));
        break;
      case 'X':
      case 'T':
      case 'B': {
        need(4);
        const std::uint32_t n = U32(pkl + pos);
        pos += 4;
        push_str(n);
        break;
      }
      case 0x8c:
      case 'U':
      case 'C': {
        need(1);
        const std::uint8_t n = pkl[pos++];
        push_str(n);
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
          throw std::runtime_error("bad GLOBAL");
        }
        NodePtr v = Make(Kind::kGlobal);
        v->s = std::string(reinterpret_cast<const char*>(pkl + pos), nl1 - pos) + " " +
               std::string(reinterpret_cast<const char*>(pkl + nl1 + 1), nl2 - nl1 - 1);
        pos = nl2 + 1;
        stack.push_back(std::move(v));
        break;
      }
      case 0x93: {  // STACK_GLOBAL
        NodePtr name = pop();
        NodePtr module = pop();
        NodePtr v = Make(Kind::kGlobal);
        v->s = module->s + " " + name->s;
        stack.push_back(std::move(v));
        break;
      }
      case 'q':
        need(1);
        memo[pkl[pos++]] = stack.back();
        break;
      case 'r':
        need(4);
        memo[U32(pkl + pos)] = stack.back();
        pos += 4;
        break;
      case 0x94:  // MEMOIZE
        memo[next_memo++] = stack.back();
        break;
      case 'h':
        need(1);
        stack.push_back(memo.at(pkl[pos++]));
        break;
      case 'j':
        need(4);
        stack.push_back(memo.at(U32(pkl + pos)));
        pos += 4;
        break;
      case '0':
        pop();
        break;
      case '2':
        stack.push_back(stack.back());
        break;
      case 't': {
        NodePtr v = Make(Kind::kTuple);
        v->items = pop_mark();
        stack.push_back(std::move(v));
        break;
      }
      case 0x85:
      case 0x86:
      case 0x87: {
        const std::size_t n = op == 0x85 ? 1U : (op == 0x86 ? 2U : 3U);
        if (stack.size() < n) {
          throw std::runtime_error("pickle tuple underflow");
        }
        NodePtr v = Make(Kind::kTuple);
        v->items.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
          v->items[n - 1 - k] = pop();
        }
        stack.push_back(std::move(v));
        break;
      }
      case 'Q': {  // BINPERSID ('storage', type, key, location, size)
        NodePtr pid = pop();
        if (pid->kind != Kind::kTuple || pid->items.size() < 5) {
          throw std::runtime_error("unexpected persistent id");
        }
        NodePtr v = Make(Kind::kStorage);
        v->tensor.dtype = pid->items[1]->s;
        v->tensor.storage_key = pid->items[2]->s;
        stack.push_back(std::move(v));
        break;
      }
      case 'R': {
        NodePtr args = pop();
        NodePtr fn = pop();
        stack.push_back(reduce(fn, args));
        break;
      }
      case 0x81: {  // NEWOBJ
        NodePtr args = pop();
        NodePtr cls = pop();
        stack.push_back(reduce(cls, args));
        break;
      }
      case 's': {
        NodePtr value = pop();
        NodePtr key = pop();
        if (!stack.empty()) {
          SetItem(stack.back(), std::move(key), std::move(value));
        }
        break;
      }
      case 'u': {
        auto items = pop_mark();
        if (!stack.empty()) {
          for (std::size_t k = 0; k + 1 < items.size(); k += 2) {
            SetItem(stack.back(), items[k], items[k + 1]);
          }
        }
        break;
      }
      case 'a': {
        NodePtr value = pop();
        if (!stack.empty() && stack.back()->kind == Kind::kList) {
          stack.back()->items.push_back(std::move(value));
        }
        break;
      }
      case 'e': {
        auto items = pop_mark();
        if (!stack.empty() && stack.back()->kind == Kind::kList) {
          for (auto& item : items) {
            stack.back()->items.push_back(std::move(item));
          }
        }
        break;
      }
      case 'b':  // BUILD: drop the state (e.g. OrderedDict _metadata)
        pop();
        break;
      default:
        throw std::runtime_error("unsupported pickle opcode " + std::to_string(op));
    }
  }
  return stack.empty() ? nullptr : stack.back();
}

PthDType DTypeOf(const std::string& storage) {
  if (storage == "torch FloatStorage") return PthDType::kF32;
  if (storage == "torch HalfStorage") return PthDType::kF16;
  if (storage == "torch BFloat16Storage") return PthDType::kBF16;
  if (storage == "torch DoubleStorage") return PthDType::kF64;
  if (storage == "torch LongStorage") return PthDType::kI64;
  if (storage == "torch IntStorage") return PthDType::kI32;
  if (storage == "torch BoolStorage") return PthDType::kBool;
  return PthDType::kUnknown;
}

std::size_t ElementSize(PthDType dtype) {
  switch (dtype) {
    case PthDType::kF32:
    case PthDType::kI32:
      return 4;
    case PthDType::kF16:
    case PthDType::kBF16:
      return 2;
    case PthDType::kF64:
    case PthDType::kI64:
      return 8;
    case PthDType::kBool:
      return 1;
    default:
      return 0;
  }
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

float HalfToF32(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000U) << 16U;
  std::uint32_t exp = (h >> 10U) & 0x1FU;
  std::uint32_t mant = h & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400U) == 0) {
        mant <<= 1U;
        --exp;
      }
      mant &= 0x3FFU;
      bits = sign | (exp << 23U) | (mant << 13U);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | (mant << 13U);
  } else {
    bits = sign | ((exp + 127 - 15) << 23U) | (mant << 13U);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

}  // namespace

std::unique_ptr<PthArchive> PthArchive::Open(const std::filesystem::path& path,
                                             std::string* error) {
  auto archive = std::unique_ptr<PthArchive>(new PthArchive());
  archive->path_ = path;
  std::string map_error;
  archive->file_ = platform::ReadOnlyMappedFile::Open(path, &map_error);
  if (!archive->file_.valid()) {
    SetError(error, "cannot map " + path.string() + ": " + map_error);
    return nullptr;
  }
  std::unordered_map<std::string, ZipEntry> entries;
  if (!ParseZip(archive->file_.data(), archive->file_.size(), &entries)) {
    SetError(error, path.string() + " is not an uncompressed torch zip archive");
    return nullptr;
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
  if (pkl == nullptr) {
    SetError(error, path.string() + ": no data.pkl");
    return nullptr;
  }
  NodePtr root;
  try {
    root = ParsePickle(pkl->data, pkl->size);
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return nullptr;
  }
  if (root == nullptr) {
    SetError(error, path.string() + ": empty pickle");
    return nullptr;
  }
  // Flatten nested dicts; a bare tensor becomes the key "".
  std::vector<std::pair<std::string, NodePtr>> work{{"", root}};
  while (!work.empty()) {
    auto [key, node] = std::move(work.back());
    work.pop_back();
    if (node->kind == Kind::kDict) {
      for (const auto& [k, v] : node->entries) {
        std::string sub;
        if (k->kind == Kind::kStr) {
          sub = k->s;
        } else if (k->kind == Kind::kInt) {
          sub = std::to_string(k->i);
        } else {
          continue;
        }
        work.emplace_back(key.empty() ? sub : key + "." + sub, v);
      }
    } else if (node->kind == Kind::kTensor) {
      const RawTensor& t = node->tensor;
      PthTensor out;
      out.dtype = DTypeOf(t.dtype);
      out.shape = t.shape;
      out.numel = 1;
      for (const std::int64_t d : t.shape) {
        out.numel *= d;
      }
      out.contiguous = IsContiguous(t);
      const auto found = entries.find(prefix + "data/" + t.storage_key);
      const std::size_t es = ElementSize(out.dtype);
      if (found == entries.end() || es == 0) {
        continue;
      }
      if (static_cast<std::size_t>(t.offset + (out.contiguous ? out.numel : 0)) * es >
          found->second.size) {
        SetError(error, "bad storage for " + key + " in " + path.string());
        return nullptr;
      }
      out.data = found->second.data + static_cast<std::size_t>(t.offset) * es;
      archive->tensors_[key] = std::move(out);
    }
  }
  return archive;
}

const PthTensor* PthArchive::Find(std::string_view key) const {
  const auto found = tensors_.find(key);
  return found == tensors_.end() ? nullptr : &found->second;
}

const PthTensor& PthArchive::Require(std::string_view key) const {
  const PthTensor* t = Find(key);
  if (t == nullptr) {
    throw std::runtime_error(path_.filename().string() + ": tensor missing: " + std::string(key));
  }
  return *t;
}

const float* PthArchive::F32(std::string_view key, std::int64_t expect_numel) const {
  const PthTensor& t = Require(key);
  if (t.dtype != PthDType::kF32 || !t.contiguous ||
      (expect_numel >= 0 && t.numel != expect_numel)) {
    throw std::runtime_error(path_.filename().string() + ": unexpected layout for " +
                             std::string(key) + " (numel " + std::to_string(t.numel) +
                             ", want " + std::to_string(expect_numel) + ")");
  }
  return reinterpret_cast<const float*>(t.data);
}

std::vector<float> PthArchive::ToF32(std::string_view key) const {
  const PthTensor& t = Require(key);
  if (!t.contiguous) {
    throw std::runtime_error("non-contiguous tensor " + std::string(key));
  }
  std::vector<float> out(static_cast<std::size_t>(t.numel));
  for (std::int64_t i = 0; i < t.numel; ++i) {
    const std::uint8_t* p = t.data;
    switch (t.dtype) {
      case PthDType::kF32:
        std::memcpy(&out[static_cast<std::size_t>(i)], p + i * 4, 4);
        break;
      case PthDType::kF64: {
        double d;
        std::memcpy(&d, p + i * 8, 8);
        out[static_cast<std::size_t>(i)] = static_cast<float>(d);
        break;
      }
      case PthDType::kBF16: {
        std::uint16_t b;
        std::memcpy(&b, p + i * 2, 2);
        const std::uint32_t w = static_cast<std::uint32_t>(b) << 16U;
        std::memcpy(&out[static_cast<std::size_t>(i)], &w, 4);
        break;
      }
      case PthDType::kF16: {
        std::uint16_t h;
        std::memcpy(&h, p + i * 2, 2);
        out[static_cast<std::size_t>(i)] = HalfToF32(h);
        break;
      }
      case PthDType::kI64: {
        std::int64_t v;
        std::memcpy(&v, p + i * 8, 8);
        out[static_cast<std::size_t>(i)] = static_cast<float>(v);
        break;
      }
      default:
        throw std::runtime_error("unsupported dtype for " + std::string(key));
    }
  }
  // The copy is all callers read: drop the source pages (#829).
  ReleaseResidentPages(t.data, static_cast<std::size_t>(t.numel) * ElementSize(t.dtype));
  return out;
}

void PthArchive::ReleaseTensorPages(std::string_view prefix) const noexcept {
  for (auto it = tensors_.lower_bound(prefix);
       it != tensors_.end() && std::string_view(it->first).starts_with(prefix); ++it) {
    const PthTensor& t = it->second;
    if (t.contiguous) {
      ReleaseResidentPages(t.data, static_cast<std::size_t>(t.numel) * ElementSize(t.dtype));
    }
  }
}

void PthArchive::ReleaseResidentPages() const noexcept {
  platform::ReleaseMappedPages(file_.data(), file_.size());
}

void PthArchive::ReleaseResidentPages(const void* data, std::size_t bytes) const noexcept {
  platform::ReleaseMappedRange(file_.data(), file_.size(), data, bytes);
}

std::vector<float> FoldWeightNorm(const PthArchive& archive, const std::string& prefix) {
  std::string g_key = prefix + ".weight_g";
  std::string v_key = prefix + ".weight_v";
  if (!archive.Has(g_key)) {
    g_key = prefix + ".parametrizations.weight.original0";
    v_key = prefix + ".parametrizations.weight.original1";
  }
  if (!archive.Has(g_key)) {
    return archive.ToF32(prefix + ".weight");  // plain weight
  }
  const std::vector<float> g = archive.ToF32(g_key);
  const std::vector<float> v = archive.ToF32(v_key);
  if (g.empty() || v.size() % g.size() != 0) {
    throw std::runtime_error("weight norm pair mismatch for " + prefix);
  }
  const std::size_t rows = g.size();
  const std::size_t slice = v.size() / rows;
  std::vector<float> w(v.size());
  for (std::size_t r = 0; r < rows; ++r) {
    const float* vr = v.data() + r * slice;
    // torch: v * (g / norm_except_dim(v, 2, 0)) computed in float32.
    float sum = 0.0F;
    double dsum = 0.0;
    for (std::size_t j = 0; j < slice; ++j) {
      dsum += static_cast<double>(vr[j]) * vr[j];
    }
    sum = static_cast<float>(std::sqrt(dsum));
    const float scale = g[r] / sum;
    for (std::size_t j = 0; j < slice; ++j) {
      w[r * slice + j] = vr[j] * scale;
    }
  }
  return w;
}

}  // namespace gufo::models::indextts
