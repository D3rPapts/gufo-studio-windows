#include "voxcpm2/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <gufo/ext/json.hpp>

namespace gufo::models::voxcpm2 {
namespace {

namespace json = gufo::json;

constexpr std::string_view kSpace = "\xE2\x96\x81";  // U+2581

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

// Length of the UTF-8 sequence starting at text[pos] (1 for invalid bytes).
std::size_t CharLength(std::string_view text, std::size_t pos) {
  const auto lead = static_cast<unsigned char>(text[pos]);
  std::size_t length = 1;
  if (lead >= 0xF0 && lead < 0xF8) {
    length = 4;
  } else if (lead >= 0xE0) {
    length = lead < 0xF0 ? 3 : 1;
  } else if (lead >= 0xC0) {
    length = 2;
  }
  if (pos + length > text.size()) {
    return 1;
  }
  for (std::size_t i = 1; i < length; ++i) {
    if ((static_cast<unsigned char>(text[pos + i]) & 0xC0) != 0x80) {
      return 1;
    }
  }
  return length;
}

std::uint32_t CodePoint(std::string_view ch) {
  const auto b0 = static_cast<unsigned char>(ch[0]);
  switch (ch.size()) {
    case 2:
      return ((b0 & 0x1FU) << 6U) | (static_cast<unsigned char>(ch[1]) & 0x3FU);
    case 3:
      return ((b0 & 0x0FU) << 12U) | ((static_cast<unsigned char>(ch[1]) & 0x3FU) << 6U) |
             (static_cast<unsigned char>(ch[2]) & 0x3FU);
    case 4:
      return ((b0 & 0x07U) << 18U) | ((static_cast<unsigned char>(ch[1]) & 0x3FU) << 12U) |
             ((static_cast<unsigned char>(ch[2]) & 0x3FU) << 6U) |
             (static_cast<unsigned char>(ch[3]) & 0x3FU);
    default:
      return b0;
  }
}

std::vector<std::string_view> SplitChars(std::string_view text) {
  std::vector<std::string_view> chars;
  for (std::size_t pos = 0; pos < text.size();) {
    const std::size_t length = CharLength(text, pos);
    chars.push_back(text.substr(pos, length));
    pos += length;
  }
  return chars;
}

bool IsMulticharCjk(std::string_view token) {
  const auto chars = SplitChars(token);
  if (chars.size() < 2) {
    return false;
  }
  for (const auto ch : chars) {
    const std::uint32_t cp = CodePoint(ch);
    if (ch.size() != 3 || cp < 0x4E00 || cp > 0x9FFF) {
      return false;
    }
  }
  return true;
}

std::uint64_t PairKey(std::int32_t left, std::int32_t right) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
         static_cast<std::uint32_t>(right);
}

}  // namespace

bool Tokenizer::Load(const std::string& model_dir, Tokenizer* out, std::string* error) {
  const std::filesystem::path path = std::filesystem::path(model_dir) / "tokenizer.json";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    SetError(error, "cannot read " + path.string());
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  json::Value root;
  try {
    root = json::parse(buffer.str());
  } catch (const std::exception& exception) {
    SetError(error, std::string("invalid tokenizer.json: ") + exception.what());
    return false;
  }
  const json::Value* model = root.find("model");
  if (model == nullptr || model->member_str("type") != "BPE") {
    SetError(error, "VoxCPM2 tokenizer.json: BPE model expected");
    return false;
  }
  if (const json::Value* pre = root.find("pre_tokenizer"); pre != nullptr && !pre->is_null()) {
    SetError(error, "VoxCPM2 tokenizer.json: unexpected pre_tokenizer");
    return false;
  }
  const json::Value* vocab = model->find("vocab");
  const json::Value* merges = model->find("merges");
  if (vocab == nullptr || !vocab->is_object() || merges == nullptr || !merges->is_array()) {
    SetError(error, "VoxCPM2 tokenizer.json: vocab/merges missing");
    return false;
  }
  Tokenizer t;
  std::int32_t max_id = -1;
  for (const auto& [token, id] : vocab->members()) {
    max_id = std::max(max_id, static_cast<std::int32_t>(id.as_double()));
  }
  if (const json::Value* added = root.find("added_tokens"); added != nullptr) {
    for (const auto& entry : added->items()) {
      max_id = std::max(max_id, static_cast<std::int32_t>(entry.member_double("id")));
    }
  }
  t.tokens_.resize(static_cast<std::size_t>(max_id + 1));
  for (const auto& [token, id] : vocab->members()) {
    const auto value = static_cast<std::int32_t>(id.as_double());
    t.tokens_[static_cast<std::size_t>(value)] = token;
    t.vocab_.emplace(token, value);
    if (IsMulticharCjk(token)) {
      t.multichar_cjk_.insert(token);
    }
  }
  if (const json::Value* added = root.find("added_tokens"); added != nullptr) {
    for (const auto& entry : added->items()) {
      const std::string content = entry.member_str("content");
      const auto id = static_cast<std::int32_t>(entry.member_double("id"));
      if (content.empty()) {
        continue;
      }
      t.tokens_[static_cast<std::size_t>(id)] = content;
      t.vocab_.emplace(content, id);
      t.added_.emplace_back(content, id);
    }
  }
  std::ranges::sort(t.added_, [](const auto& a, const auto& b) {
    return a.first.size() > b.first.size();
  });
  std::int32_t rank = 0;
  for (const auto& merge : merges->items()) {
    std::string left;
    std::string right;
    if (merge.is_string()) {
      const std::string& text = merge.str();
      const std::size_t space = text.find(' ');
      if (space == std::string::npos) {
        SetError(error, "VoxCPM2 tokenizer.json: malformed merge " + text);
        return false;
      }
      left = text.substr(0, space);
      right = text.substr(space + 1);
    } else if (merge.is_array() && merge.size() == 2) {
      left = merge.items()[0].str();
      right = merge.items()[1].str();
    } else {
      SetError(error, "VoxCPM2 tokenizer.json: malformed merge");
      return false;
    }
    const auto l = t.vocab_.find(left);
    const auto r = t.vocab_.find(right);
    const auto m = t.vocab_.find(left + right);
    if (l == t.vocab_.end() || r == t.vocab_.end() || m == t.vocab_.end()) {
      SetError(error, "VoxCPM2 tokenizer.json: merge with unknown token " + left + " " + right);
      return false;
    }
    t.merges_.try_emplace(PairKey(l->second, r->second), rank, m->second);
    ++rank;
  }
  for (int byte = 0; byte < 256; ++byte) {
    char name[8];
    std::snprintf(name, sizeof(name), "<0x%02X>", byte);
    const auto found = t.vocab_.find(name);
    if (found == t.vocab_.end()) {
      SetError(error, std::string("VoxCPM2 tokenizer.json: byte token missing ") + name);
      return false;
    }
    t.byte_ids_[byte] = found->second;
  }
  const std::string unk = model->member_str("unk_token", "<unk>");
  if (const auto found = t.vocab_.find(unk); found != t.vocab_.end()) {
    t.unk_id_ = found->second;
  }
  *out = std::move(t);
  return true;
}

void Tokenizer::EncodeSegment(std::string_view segment, std::vector<std::int32_t>* ids) const {
  if (segment.empty()) {
    return;
  }
  // Normalizer: Prepend("▁") then Replace(" ", "▁").
  std::string normalized(kSpace);
  for (const char c : segment) {
    if (c == ' ') {
      normalized += kSpace;
    } else {
      normalized += c;
    }
  }
  std::vector<std::int32_t> symbols;
  for (const auto ch : SplitChars(normalized)) {
    const auto found = vocab_.find(std::string(ch));
    if (found != vocab_.end()) {
      symbols.push_back(found->second);
    } else {
      for (const char byte : ch) {
        symbols.push_back(byte_ids_[static_cast<unsigned char>(byte)]);
      }
    }
  }
  // Merge the lowest-ranked pair (leftmost on ties) until none applies.
  while (symbols.size() > 1) {
    std::int32_t best_rank = std::numeric_limits<std::int32_t>::max();
    std::size_t best = 0;
    std::int32_t merged = 0;
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto found = merges_.find(PairKey(symbols[i], symbols[i + 1]));
      if (found != merges_.end() && found->second.first < best_rank) {
        best_rank = found->second.first;
        best = i;
        merged = found->second.second;
      }
    }
    if (best_rank == std::numeric_limits<std::int32_t>::max()) {
      break;
    }
    symbols[best] = merged;
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best) + 1);
  }
  ids->insert(ids->end(), symbols.begin(), symbols.end());
}

std::vector<std::string> Tokenizer::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> ids;
  std::size_t start = 0;
  std::size_t pos = 0;
  while (pos < text.size()) {
    bool matched = false;
    for (const auto& [content, id] : added_) {
      if (text.substr(pos, content.size()) == content) {
        EncodeSegment(text.substr(start, pos - start), &ids);
        ids.push_back(id);
        pos += content.size();
        start = pos;
        matched = true;
        break;
      }
    }
    if (!matched) {
      ++pos;
    }
  }
  EncodeSegment(text.substr(start), &ids);
  std::vector<std::string> tokens;
  tokens.reserve(ids.size());
  for (const std::int32_t id : ids) {
    tokens.push_back(tokens_[static_cast<std::size_t>(id)]);
  }
  return tokens;
}

std::vector<std::int64_t> Tokenizer::Encode(std::string_view text) const {
  std::vector<std::int64_t> ids;
  for (const std::string& token : Tokenize(text)) {
    std::string clean;
    for (std::size_t pos = 0; pos < token.size();) {
      if (token.compare(pos, kSpace.size(), kSpace) == 0) {
        pos += kSpace.size();
      } else {
        clean += token[pos++];
      }
    }
    if (multichar_cjk_.contains(clean)) {
      for (const auto ch : SplitChars(clean)) {
        const auto found = vocab_.find(std::string(ch));
        ids.push_back(found != vocab_.end() ? found->second : unk_id_);
      }
      continue;
    }
    const auto found = vocab_.find(token);
    ids.push_back(found != vocab_.end() ? found->second : unk_id_);
  }
  return ids;
}

}  // namespace gufo::models::voxcpm2
