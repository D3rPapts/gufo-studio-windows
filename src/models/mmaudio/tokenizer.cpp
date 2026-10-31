// MMAudio for gufo: the CLIP BPE tokenizer (see tokenizer.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/tokenizer.hpp"

#include <unicode/locid.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace gufo::models::mmaudio {
namespace {

bool ReadText(const std::filesystem::path& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return true;
}

void AppendUtf8(std::uint32_t code, std::string* out) {
  if (code < 0x80) {
    out->push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (code >> 6)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (code >> 12)));
    out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (code >> 18)));
    out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

// html.unescape for the common named entities and numeric references.
std::string Unescape(std::string_view text) {
  static const std::pair<std::string_view, std::string_view> kNamed[] = {
      {"&amp;", "&"},  {"&lt;", "<"},    {"&gt;", ">"},
      {"&quot;", "\""}, {"&#39;", "'"},   {"&apos;", "'"},
      {"&nbsp;", "\xC2\xA0"}};
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    if (text[i] != '&') {
      out.push_back(text[i++]);
      continue;
    }
    bool replaced = false;
    for (const auto& [entity, value] : kNamed) {
      if (text.substr(i, entity.size()) == entity) {
        out += value;
        i += entity.size();
        replaced = true;
        break;
      }
    }
    if (replaced)
      continue;
    if (i + 2 < text.size() && text[i + 1] == '#') {
      std::size_t j = i + 2;
      const bool hex = j < text.size() && (text[j] == 'x' || text[j] == 'X');
      if (hex)
        ++j;
      std::uint32_t code = 0;
      const std::size_t digits_start = j;
      while (j < text.size() && j - digits_start < 8 &&
             (hex ? std::isxdigit(static_cast<unsigned char>(text[j]))
                  : std::isdigit(static_cast<unsigned char>(text[j])))) {
        const char c = text[j];
        code = code * (hex ? 16 : 10) +
               static_cast<std::uint32_t>(
                   std::isdigit(static_cast<unsigned char>(c))
                       ? c - '0'
                       : std::tolower(static_cast<unsigned char>(c)) - 'a' +
                             10);
        ++j;
      }
      if (j > digits_start && code > 0 && code < 0x110000) {
        if (j < text.size() && text[j] == ';')
          ++j;
        AppendUtf8(code, &out);
        i = j;
        continue;
      }
    }
    out.push_back(text[i++]);
  }
  return out;
}

}  // namespace

std::string CleanClipText(std::string_view text) {
  const std::string unescaped = Unescape(Unescape(text));
  icu::UnicodeString unicode = icu::UnicodeString::fromUTF8(unescaped);
  // Collapse whitespace runs (Python's \s) to one space and trim.
  icu::UnicodeString collapsed;
  bool pending_space = false;
  for (int32_t i = 0; i < unicode.length();) {
    const UChar32 c = unicode.char32At(i);
    i += U16_LENGTH(c);
    if (u_isUWhiteSpace(c) || c == 0x1C || c == 0x1D || c == 0x1E ||
        c == 0x1F) {
      pending_space = !collapsed.isEmpty();
      continue;
    }
    if (pending_space)
      collapsed.append(static_cast<UChar>(' '));
    pending_space = false;
    collapsed.append(c);
  }
  collapsed.toLower(icu::Locale::getRoot());
  std::string out;
  collapsed.toUTF8String(out);
  return out;
}

bool ClipTokenizer::Load(const std::filesystem::path& dir,
                         std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return false;
  };
  std::string vocab_text, merges_text;
  if (!ReadText(dir / "vocab.json", &vocab_text))
    return fail("cannot read " + (dir / "vocab.json").string());
  if (!ReadText(dir / "merges.txt", &merges_text))
    return fail("cannot read " + (dir / "merges.txt").string());
  try {
    const json::Value vocab = json::parse(vocab_text);
    for (const auto& [word, id] : vocab.members())
      vocab_[word] = static_cast<int>(id.as_double());
  } catch (const std::exception& e) {
    return fail(std::string("vocab.json: ") + e.what());
  }
  std::istringstream lines(merges_text);
  std::string line;
  int rank = 0;
  bool first = true;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (first && line.starts_with("#version")) {
      first = false;
      continue;
    }
    first = false;
    if (line.empty())
      continue;
    ranks_.emplace(line, rank++);
  }
  // GPT-2's byte -> printable code point table.
  std::vector<int> printable;
  for (int b = '!'; b <= '~'; ++b)
    printable.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b)
    printable.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b)
    printable.push_back(b);
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    const bool direct =
        std::find(printable.begin(), printable.end(), b) != printable.end();
    const std::uint32_t code = direct ? b : 256 + extra++;
    byte_encoder_[b].clear();
    AppendUtf8(code, &byte_encoder_[b]);
  }
  if (vocab_.size() < 49000 || ranks_.size() < 48000 ||
      vocab_.find("<|startoftext|>") == vocab_.end() ||
      vocab_.at("<|startoftext|>") != kStartOfText ||
      vocab_.at("<|endoftext|>") != kEndOfText)
    return fail("unexpected CLIP vocabulary in " + dir.string());
  return true;
}

void ClipTokenizer::Bpe(const std::string& token, std::vector<int>* ids) const {
  // Symbols: the byte-encoded characters, the last one with "</w>".
  std::vector<std::string> word;
  for (const unsigned char byte : token)
    word.push_back(byte_encoder_[byte]);
  if (word.empty())
    return;
  word.back() += "</w>";
  while (word.size() > 1) {
    int best = INT_MAX;
    std::size_t at = 0;
    for (std::size_t i = 0; i + 1 < word.size(); ++i) {
      const auto it = ranks_.find(word[i] + " " + word[i + 1]);
      if (it != ranks_.end() && it->second < best) {
        best = it->second;
        at = i;
      }
    }
    if (best == INT_MAX)
      break;
    // Merge every occurrence of the best pair, left to right (as CLIP does).
    const std::string left = word[at], right = word[at + 1];
    std::vector<std::string> merged;
    for (std::size_t i = 0; i < word.size();) {
      if (i + 1 < word.size() && word[i] == left && word[i + 1] == right) {
        merged.push_back(left + right);
        i += 2;
      } else {
        merged.push_back(word[i]);
        ++i;
      }
    }
    word = std::move(merged);
  }
  for (const std::string& piece : word) {
    const auto it = vocab_.find(piece);
    if (it != vocab_.end())
      ids->push_back(it->second);
  }
}

std::vector<int> ClipTokenizer::EncodeWords(std::string_view text) const {
  std::vector<int> ids;
  const std::string clean = CleanClipText(text);
  UErrorCode status = U_ZERO_ERROR;
  static const char* kPattern =
      "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|"
      "[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";
  const std::unique_ptr<icu::RegexPattern> pattern(icu::RegexPattern::compile(
      icu::UnicodeString::fromUTF8(kPattern), UREGEX_CASE_INSENSITIVE,
      status));
  if (U_FAILURE(status) || !pattern)
    throw std::runtime_error("invalid CLIP tokenizer pattern");
  const icu::UnicodeString unicode = icu::UnicodeString::fromUTF8(clean);
  const std::unique_ptr<icu::RegexMatcher> matcher(
      pattern->matcher(unicode, status));
  if (U_FAILURE(status) || !matcher)
    throw std::runtime_error("CLIP tokenizer matcher failed");
  while (matcher->find(status)) {
    std::string piece;
    matcher->group(status).toUTF8String(piece);
    if (piece == "<|startoftext|>") {
      ids.push_back(kStartOfText);
      continue;
    }
    if (piece == "<|endoftext|>") {
      ids.push_back(kEndOfText);
      continue;
    }
    Bpe(piece, &ids);
  }
  if (U_FAILURE(status))
    throw std::runtime_error("CLIP tokenization failed");
  return ids;
}

std::vector<int> ClipTokenizer::Encode(std::string_view text) const {
  std::vector<int> words = EncodeWords(text);
  std::vector<int> ids;
  ids.reserve(kContext);
  ids.push_back(kStartOfText);
  ids.insert(ids.end(), words.begin(), words.end());
  ids.push_back(kEndOfText);
  if (ids.size() > static_cast<std::size_t>(kContext)) {
    ids.resize(kContext);
    ids.back() = kEndOfText;
  }
  ids.resize(kContext, 0);
  return ids;
}

}  // namespace gufo::models::mmaudio
