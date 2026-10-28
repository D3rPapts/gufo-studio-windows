#include "src/models/wan22/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::models::wan22 {
namespace {

constexpr std::uintmax_t kMaximumTokenizerBytes = 256ULL << 20U;
constexpr std::size_t kMaximumPieces = 1U << 24U;
// tokenizers' Unigram scores an unknown character min_score - 10.
constexpr double kUnknownPenalty = 10.0;
constexpr std::string_view kSpaceSymbol = "\xE2\x96\x81";

std::size_t Utf8Length(unsigned char lead) {
  if (lead < 0x80U)
    return 1;
  if ((lead & 0xE0U) == 0xC0U)
    return 2;
  if ((lead & 0xF0U) == 0xE0U)
    return 3;
  if ((lead & 0xF8U) == 0xF0U)
    return 4;
  return 1;
}

// Length of the character at text[pos], never past the end.
std::size_t CharLength(std::string_view text, std::size_t pos) {
  return std::min(Utf8Length(static_cast<unsigned char>(text[pos])),
                  text.size() - pos);
}

// Python's str.isspace for the code point at text[pos]; returns its length
// in bytes, 0 when it is not whitespace.
std::size_t SpaceLength(std::string_view text, std::size_t pos) {
  const auto c = static_cast<unsigned char>(text[pos]);
  if (c == ' ' || (c >= 0x09U && c <= 0x0DU) || (c >= 0x1CU && c <= 0x1FU))
    return 1;
  const std::size_t length = CharLength(text, pos);
  if (length == 1)
    return 0;
  std::uint32_t code = c & (0xFFU >> (length + 1));
  for (std::size_t i = 1; i < length; ++i)
    code = (code << 6U) | (static_cast<unsigned char>(text[pos + i]) & 0x3FU);
  const bool space = code == 0x85U || code == 0xA0U || code == 0x1680U ||
                     (code >= 0x2000U && code <= 0x200AU) || code == 0x2028U ||
                     code == 0x2029U || code == 0x202FU || code == 0x205FU ||
                     code == 0x3000U;
  return space ? length : 0;
}

bool ReadFile(const std::filesystem::path& path, std::string* contents,
              std::string* error) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    *error = "cannot stat " + path.string() + ": " + ec.message();
    return false;
  }
  if (size > kMaximumTokenizerBytes) {
    *error = "unexpectedly large " + path.string();
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  contents->resize(static_cast<std::size_t>(size));
  if (!input ||
      !input.read(contents->data(), static_cast<std::streamsize>(size))) {
    *error = "cannot read " + path.string();
    return false;
  }
  return true;
}

const json::Value& Member(const json::Value& object, const char* key) {
  static const json::Value kNull;
  const json::Value* value = object.find(key);
  return value != nullptr ? *value : kNull;
}

bool IsReplaceSpaces(const json::Value& normalizer) {
  if (normalizer.member_str("type") != "Replace" ||
      normalizer.member_str("content") != " ")
    return false;
  return Member(normalizer, "pattern").member_str("Regex") == " {2,}";
}

// The checkpoint's normalizer, pre-tokenizer and post-processor; anything
// else would tokenize differently.
bool CheckPipeline(const json::Value& root, std::string* error) {
  const json::Value& normalizer = Member(root, "normalizer");
  bool normalizer_ok = IsReplaceSpaces(normalizer);
  if (!normalizer_ok && normalizer.member_str("type") == "Sequence") {
    const json::Value& list = Member(normalizer, "normalizers");
    normalizer_ok = list.is_array() && list.size() == 1 &&
                    IsReplaceSpaces(list.items()[0]);
  }
  if (!normalizer_ok) {
    *error = "unsupported umT5 normalizer";
    return false;
  }
  const json::Value& pre = Member(root, "pre_tokenizer");
  if (pre.member_str("type") != "Metaspace" ||
      pre.member_str("replacement") != kSpaceSymbol ||
      pre.member_str("prepend_scheme") != "always" ||
      !Member(pre, "split").as_bool(false)) {
    *error = "unsupported umT5 pre-tokenizer";
    return false;
  }
  const json::Value& post = Member(root, "post_processor");
  const json::Value& single = Member(post, "single");
  if (post.member_str("type") != "TemplateProcessing" || !single.is_array() ||
      single.size() != 2 ||
      Member(Member(single.items()[0], "Sequence"), "id").str() != "A" ||
      Member(Member(single.items()[1], "SpecialToken"), "id").str() !=
          "</s>") {
    *error = "unsupported umT5 post-processor";
    return false;
  }
  const json::Value& model = Member(root, "model");
  if (model.member_str("type") != "Unigram" ||
      Member(model, "byte_fallback").as_bool(false)) {
    *error = "expected a Unigram model without byte fallback";
    return false;
  }
  return true;
}

}  // namespace

std::string CleanPrompt(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  bool pending_space = false;
  for (std::size_t pos = 0; pos < text.size();) {
    if (const std::size_t space = SpaceLength(text, pos)) {
      pending_space = !out.empty();
      pos += space;
      continue;
    }
    if (pending_space)
      out.push_back(' ');
    pending_space = false;
    const std::size_t length = CharLength(text, pos);
    out.append(text.substr(pos, length));
    pos += length;
  }
  return out;
}

std::optional<Tokenizer> Tokenizer::Load(const std::filesystem::path& path,
                                         std::string* error) {
  std::string local_error;
  std::string* message = error != nullptr ? error : &local_error;
  std::string contents;
  if (!ReadFile(path, &contents, message))
    return std::nullopt;
  json::Value root;
  try {
    root = json::parse(contents);
  } catch (const std::exception& e) {
    *message = path.string() + ": " + e.what();
    return std::nullopt;
  }
  contents.clear();
  contents.shrink_to_fit();
  if (!CheckPipeline(root, message))
    return std::nullopt;
  const json::Value& model = Member(root, "model");
  const json::Value& vocab = Member(model, "vocab");
  if (!vocab.is_array() || vocab.empty() || vocab.size() > kMaximumPieces) {
    *message = "invalid umT5 vocabulary";
    return std::nullopt;
  }
  Tokenizer t;
  t.pieces_.reserve(vocab.size());
  t.scores_.reserve(vocab.size());
  double min_score = std::numeric_limits<double>::infinity();
  for (const json::Value& entry : vocab.items()) {
    if (!entry.is_array() || entry.size() != 2 ||
        !entry.items()[0].is_string() || !entry.items()[1].is_number()) {
      *message = "invalid umT5 vocabulary entry";
      return std::nullopt;
    }
    const double score = entry.items()[1].as_double();
    if (!std::isfinite(score)) {
      *message = "invalid umT5 piece score";
      return std::nullopt;
    }
    t.pieces_.push_back(entry.items()[0].str());
    t.scores_.push_back(score);
    min_score = std::min(min_score, score);
  }
  t.ids_.reserve(t.pieces_.size());
  for (std::size_t id = 0; id < t.pieces_.size(); ++id) {
    const std::string& piece = t.pieces_[id];
    if (piece.empty() ||
        !t.ids_.emplace(piece, static_cast<std::int32_t>(id)).second) {
      *message = "empty or duplicate umT5 piece";
      return std::nullopt;
    }
    t.max_piece_bytes_ = std::max(t.max_piece_bytes_, piece.size());
  }
  const std::size_t unk = Member(model, "unk_id").as_size(t.pieces_.size());
  if (unk >= t.pieces_.size() || t.pieces_.size() <= kEosId ||
      t.pieces_[kEosId] != "</s>" || t.pieces_[kPadId] != "<pad>") {
    *message = "unexpected umT5 special pieces";
    return std::nullopt;
  }
  t.unk_id_ = static_cast<std::int32_t>(unk);
  t.unk_score_ = min_score - kUnknownPenalty;
  const json::Value& added = Member(root, "added_tokens");
  if (!added.is_array()) {
    *message = "missing umT5 added tokens";
    return std::nullopt;
  }
  for (const json::Value& token : added.items()) {
    const std::size_t id = Member(token, "id").as_size(t.pieces_.size());
    const std::string content = token.member_str("content");
    if (id >= t.pieces_.size() || content.empty() ||
        Member(token, "normalized").as_bool(true) ||
        Member(token, "lstrip").as_bool(true) ||
        Member(token, "rstrip").as_bool(true) ||
        Member(token, "single_word").as_bool(true)) {
      *message = "unsupported umT5 added token " + content;
      return std::nullopt;
    }
    t.added_first_byte_[static_cast<unsigned char>(content[0])] = true;
    t.added_.push_back({content, static_cast<std::int32_t>(id)});
  }
  std::stable_sort(t.added_.begin(), t.added_.end(),
                   [](const AddedToken& a, const AddedToken& b) {
                     return a.content.size() > b.content.size();
                   });
  return t;
}

std::size_t Tokenizer::MatchAddedToken(std::string_view text, std::size_t pos,
                                       std::int32_t* id) const {
  if (!added_first_byte_[static_cast<unsigned char>(text[pos])])
    return 0;
  for (const AddedToken& token : added_) {
    if (text.substr(pos).starts_with(token.content)) {
      *id = token.id;
      return token.content.size();
    }
  }
  return 0;
}

std::vector<std::int32_t> Tokenizer::Encode(std::string_view text,
                                            std::size_t max_tokens) const {
  std::vector<std::int32_t> ids;
  std::size_t segment = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    std::int32_t id = 0;
    if (const std::size_t length = MatchAddedToken(text, pos, &id)) {
      EncodeSegment(text.substr(segment, pos - segment), &ids);
      ids.push_back(id);
      pos += length;
      segment = pos;
    } else {
      pos += CharLength(text, pos);
    }
  }
  EncodeSegment(text.substr(segment), &ids);
  if (max_tokens > 0 && ids.size() > max_tokens - 1)
    ids.resize(max_tokens - 1);
  ids.push_back(kEosId);
  return ids;
}

void Tokenizer::EncodeSegment(std::string_view text,
                              std::vector<std::int32_t>* out) const {
  if (text.empty())
    return;
  // Replace(" {2,}" -> " "), then Metaspace: spaces become U+2581 and one is
  // prepended unless the text already starts with it.
  std::string normalized;
  normalized.reserve(text.size() * 3 + 3);
  for (std::size_t pos = 0; pos < text.size(); ++pos) {
    if (text[pos] == ' ') {
      normalized.append(kSpaceSymbol);
      while (pos + 1 < text.size() && text[pos + 1] == ' ')
        ++pos;
    } else {
      normalized.push_back(text[pos]);
    }
  }
  if (!normalized.starts_with(kSpaceSymbol))
    normalized.insert(0, kSpaceSymbol);
  // Split before every U+2581, which stays with the word that follows.
  const std::string_view view(normalized);
  std::size_t begin = 0;
  for (std::size_t pos = kSpaceSymbol.size(); pos <= view.size();) {
    if (pos == view.size() || view.substr(pos).starts_with(kSpaceSymbol)) {
      EncodeWord(view.substr(begin, pos - begin), out);
      begin = pos;
      if (pos == view.size())
        break;
      pos += kSpaceSymbol.size();
    } else {
      pos += CharLength(view, pos);
    }
  }
}

void Tokenizer::EncodeWord(std::string_view word,
                           std::vector<std::int32_t>* out) const {
  if (word.empty())
    return;
  struct Node {
    double score{0};
    std::size_t start{0};
    std::int32_t id{-1};
  };
  // best[end] is the best segmentation of word[0, end); tokenizers' Viterbi
  // visits candidate pieces shortest first and keeps the first of equal
  // scores.
  std::vector<Node> best(word.size() + 1);
  best[0].id = 0;
  for (std::size_t start = 0; start < word.size();) {
    const std::size_t char_length = CharLength(word, start);
    if (best[start].id < 0) {
      start += char_length;
      continue;
    }
    const double base = best[start].score;
    bool single = false;
    const std::size_t limit = std::min(max_piece_bytes_, word.size() - start);
    for (std::size_t length = char_length; length <= limit;) {
      const auto it = ids_.find(word.substr(start, length));
      if (it != ids_.end()) {
        Node& target = best[start + length];
        const double score = base + scores_[it->second];
        if (target.id < 0 || score > target.score)
          target = {score, start, it->second};
        single = single || length == char_length;
      }
      if (start + length >= word.size())
        break;
      length += CharLength(word, start + length);
    }
    if (!single) {
      Node& target = best[start + char_length];
      const double score = base + unk_score_;
      if (target.id < 0 || score > target.score)
        target = {score, start, unk_id_};
    }
    start += char_length;
  }
  std::vector<std::int32_t> reversed;
  for (std::size_t end = word.size(); end > 0; end = best[end].start)
    reversed.push_back(best[end].id);
  for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
    // Adjacent unknown characters fuse into one `<unk>`.
    if (*it == unk_id_ && !out->empty() && out->back() == unk_id_ &&
        it != reversed.rbegin())
      continue;
    out->push_back(*it);
  }
}

}  // namespace gufo::models::wan22
