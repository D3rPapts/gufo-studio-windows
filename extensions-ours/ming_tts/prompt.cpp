#include "ming_tts/prompt.hpp"

#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <utility>

namespace gufo::models::ming_tts {
namespace {

constexpr std::string_view kDefaultTtsPrompt =
    "Please generate speech based on the following description.\n";
constexpr std::string_view kSystemPrefix =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
constexpr std::string_view kUserPrefix = "<|im_start|>user\n";
constexpr std::string_view kUserSuffix = "<|im_end|>\n";
constexpr std::string_view kAssistantPrefix = "<|im_start|>assistant\n";
constexpr std::string_view kSpeakerStart = "<|vision_start|>";
constexpr std::string_view kSpeakerPad = "<|vision_pad|>";
constexpr std::string_view kSpeakerEnd = "<|vision_end|>\n";
constexpr std::string_view kInstructionEos = "<|im_end|>";
constexpr std::string_view kTextInputPrefix = " Text input:\n";
constexpr std::string_view kAudioStart = "<audio>";

std::string ToLower(std::string_view text) {
  icu::UnicodeString value = icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<std::int32_t>(text.size())));
  value.toLower();
  std::string out;
  value.toUTF8String(out);
  return out;
}

std::string Trim(std::string_view text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && static_cast<unsigned char>(text[begin]) <= ' ') {
    ++begin;
  }
  while (end > begin && static_cast<unsigned char>(text[end - 1]) <= ' ') {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::u32string ToU32(std::string_view text) {
  icu::UnicodeString value = icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<std::int32_t>(text.size())));
  std::u32string out;
  for (std::int32_t i = 0; i < value.length();) {
    const UChar32 c = value.char32At(i);
    out.push_back(static_cast<char32_t>(c));
    i += U16_LENGTH(c);
  }
  return out;
}

std::string ToUtf8(const std::u32string& text) {
  icu::UnicodeString value;
  for (const char32_t c : text) {
    value.append(static_cast<UChar32>(c));
  }
  std::string out;
  value.toUTF8String(out);
  return out;
}

std::string Nfc(std::string_view text) {
  UErrorCode status = U_ZERO_ERROR;
  const icu::Normalizer2* normalizer = icu::Normalizer2::getNFCInstance(status);
  const icu::UnicodeString source = icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<std::int32_t>(text.size())));
  if (U_FAILURE(status) || normalizer == nullptr) {
    return std::string(text);
  }
  icu::UnicodeString normalized;
  normalizer->normalize(source, normalized, status);
  if (U_FAILURE(status)) {
    return std::string(text);
  }
  std::string out;
  normalized.toUTF8String(out);
  return out;
}

// Python str.isspace()
bool IsSpace(char32_t c) {
  return u_isUWhiteSpace(static_cast<UChar32>(c)) != 0 ||
         (c >= 0x1C && c <= 0x1F);
}

bool IsDecimal(char32_t c) {
  return u_charType(static_cast<UChar32>(c)) == U_DECIMAL_DIGIT_NUMBER;
}

std::u32string Strip(const std::u32string& text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && IsSpace(text[begin])) {
    ++begin;
  }
  while (end > begin && IsSpace(text[end - 1])) {
    --end;
  }
  return text.substr(begin, end - begin);
}

bool EndsWithAny(const std::u32string& text, std::u32string_view chars) {
  return !text.empty() && chars.find(text.back()) != std::u32string_view::npos;
}

// string.punctuation + string.whitespace (ASCII)
bool IsAsciiPunctOrSpace(char32_t c) {
  constexpr std::string_view kPunct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  constexpr std::string_view kSpace = " \t\n\r\x0b\x0c";
  return c < 0x80 && (kPunct.find(static_cast<char>(c)) != std::string_view::npos ||
                      kSpace.find(static_cast<char>(c)) != std::string_view::npos);
}

bool HasValidContent(const std::u32string& text) {
  return std::ranges::any_of(text, [](char32_t c) { return !IsAsciiPunctOrSpace(c); });
}

bool IsCjkUnit(char32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3041 && c <= 0x3096) ||
         (c >= 0x309D && c <= 0x309F) || (c >= 0x30A1 && c <= 0x30FA) ||
         (c >= 0x30FC && c <= 0x30FF) || (c >= 0xAC00 && c <= 0xD7A3);
}

bool IsLatinLetter(char32_t c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= 0xC0 && c <= 0xD6) || (c >= 0xD8 && c <= 0xF6) ||
         (c >= 0xF8 && c <= 0x24F) || (c >= 0x1E00 && c <= 0x1EFF);
}

std::size_t SemanticLengthU32(const std::u32string& text) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < text.size();) {
    if (IsCjkUnit(text[i])) {
      ++count;
      ++i;
      continue;
    }
    if (IsLatinLetter(text[i])) {
      // [L]+(?:['’][L]+)*
      while (i < text.size() && IsLatinLetter(text[i])) {
        ++i;
      }
      while (i + 1 < text.size() && (text[i] == U'\'' || text[i] == U'’') &&
             IsLatinLetter(text[i + 1])) {
        ++i;
        while (i < text.size() && IsLatinLetter(text[i])) {
          ++i;
        }
      }
      ++count;
      continue;
    }
    ++i;
  }
  return count;
}

bool IsAsciiAlnum(char32_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

void AppendTextFragment(std::vector<std::u32string>* fragments,
                        std::u32string new_text, std::size_t max_len,
                        std::size_t min_tail_length) {
  // new_text.lstrip("，,:;" + string.whitespace)
  constexpr std::u32string_view kLead = U"，,:; \t\n\r\x0b\x0c";
  std::size_t start = 0;
  while (start < new_text.size() &&
         kLead.find(new_text[start]) != std::u32string_view::npos) {
    ++start;
  }
  new_text = new_text.substr(start);
  if (!HasValidContent(new_text)) {
    return;
  }
  if (fragments->empty()) {
    fragments->push_back(std::move(new_text));
    return;
  }
  std::u32string& last = fragments->back();
  const std::size_t last_len = SemanticLengthU32(last);
  const std::size_t new_len = SemanticLengthU32(new_text);
  if (last_len + new_len <= max_len) {
    if (EndsWithAny(last, U"。！？") && new_len < min_tail_length) {
      fragments->push_back(std::move(new_text));
    } else {
      std::u32string separator;
      if (!last.empty() && last.back() != U' ' && !new_text.empty() &&
          IsAsciiAlnum(new_text.front())) {
        separator = U" ";
      }
      last += separator + new_text;
    }
  } else {
    fragments->push_back(std::move(new_text));
  }
}

// split_long_fragment: re.split(r'([，、；。！？,;!?]|\s+[-•·])') with the
// separators folded back into the preceding clause.
std::vector<std::u32string> SplitLongFragment(const std::u32string& text,
                                              std::size_t max_len) {
  if (SemanticLengthU32(text) <= max_len) {
    return {text};
  }
  constexpr std::u32string_view kPunct = U"，、；。！？,;!?";
  constexpr std::u32string_view kBullet = U"-•·";
  std::vector<std::pair<std::u32string, bool>> parts;  // (text, is_separator)
  std::u32string current;
  for (std::size_t i = 0; i < text.size();) {
    std::size_t match = 0;
    if (kPunct.find(text[i]) != std::u32string_view::npos) {
      match = 1;
    } else if (IsSpace(text[i])) {
      std::size_t j = i;
      while (j < text.size() && IsSpace(text[j])) {
        ++j;
      }
      if (j < text.size() && kBullet.find(text[j]) != std::u32string_view::npos) {
        match = j + 1 - i;
      }
    }
    if (match == 0) {
      current.push_back(text[i]);
      ++i;
      continue;
    }
    parts.emplace_back(current, false);
    current.clear();
    parts.emplace_back(text.substr(i, match), true);
    i += match;
  }
  parts.emplace_back(current, false);

  std::vector<std::u32string> merged;
  for (const auto& [part, separator] : parts) {
    if (part.empty()) {
      continue;
    }
    if (!merged.empty() && separator) {
      merged.back() += part;
    } else {
      merged.push_back(part);
    }
  }
  std::vector<std::u32string> fragments;
  std::u32string fragment;
  for (const std::u32string& clause : merged) {
    const std::size_t clause_len = SemanticLengthU32(clause);
    if (clause_len > max_len) {
      if (!fragment.empty()) {
        fragments.push_back(fragment);
        fragment.clear();
      }
      fragments.push_back(clause);
      continue;
    }
    if (SemanticLengthU32(fragment) + clause_len <= max_len) {
      fragment += clause;
    } else {
      if (!fragment.empty()) {
        fragments.push_back(fragment);
      }
      fragment = clause;
    }
  }
  if (!fragment.empty()) {
    fragments.push_back(fragment);
  }
  return fragments;
}

std::u32string ReplaceAll(std::u32string text, std::u32string_view from,
                          std::u32string_view to) {
  std::size_t position = 0;
  while ((position = text.find(from, position)) != std::u32string::npos) {
    text.replace(position, from.size(), to);
    position += to.size();
  }
  return text;
}

std::vector<std::u32string> CutTextBySemanticLength(const std::u32string& text,
                                                    std::size_t max_len,
                                                    std::size_t min_tail) {
  if (!HasValidContent(text)) {
    return {};
  }
  constexpr std::u32string_view kPlaceholder = U"##DOT##";
  // (\d)\.(\d) -> placeholder, non-overlapping left-to-right like re.sub
  auto protect = [&](const std::u32string& input, auto&& is_left,
                     auto&& is_right) {
    std::u32string out;
    for (std::size_t i = 0; i < input.size();) {
      if (i + 2 < input.size() && is_left(input[i]) && input[i + 1] == U'.' &&
          is_right(input[i + 2])) {
        out.push_back(input[i]);
        out += kPlaceholder;
        out.push_back(input[i + 2]);
        i += 3;
      } else {
        out.push_back(input[i]);
        ++i;
      }
    }
    return out;
  };
  auto is_upper = [](char32_t c) { return c >= U'A' && c <= U'Z'; };
  std::u32string processed = protect(text, IsDecimal, IsDecimal);
  for (int round = 0; round < 3; ++round) {
    processed = protect(processed, is_upper, is_upper);
  }
  processed = ReplaceAll(processed, U"\n", U" ");
  processed = ReplaceAll(processed, U"。，", U"。");
  if (SemanticLengthU32(processed) <= max_len) {
    return {ReplaceAll(processed, kPlaceholder, U".")};
  }
  std::u32string normalized = processed;
  for (char32_t& c : normalized) {
    switch (c) {
      case U'.':
        c = U'。';
        break;
      case U'!':
        c = U'！';
        break;
      case U'?':
        c = U'？';
        break;
      case U',':
        c = U'，';
        break;
      default:
        break;
    }
  }
  constexpr std::u32string_view kSentenceEnd = U"。！？";
  std::vector<std::u32string> sentences;
  std::u32string current;
  for (const char32_t c : normalized) {
    current.push_back(c);
    if (kSentenceEnd.find(c) != std::u32string_view::npos) {
      std::u32string sentence = Strip(current);
      if (!sentence.empty()) {
        sentences.push_back(std::move(sentence));
      }
      current.clear();
    }
  }
  if (!current.empty()) {
    sentences.push_back(Strip(current));
    if (!EndsWithAny(sentences.back(), kSentenceEnd)) {
      sentences.back() += U"。";
    }
  }

  constexpr std::u32string_view kClauseEnd = U"，;；";
  std::vector<std::u32string> result;
  for (const std::u32string& sentence : sentences) {
    std::vector<std::u32string> clauses;
    std::u32string clause_buffer;
    auto flush = [&]() {
      const std::u32string clause = Strip(clause_buffer);
      if (!clause.empty() && HasValidContent(clause)) {
        clauses.push_back(clause);
      } else if (!clause.empty() && !clauses.empty()) {
        clauses.back() += clause;
      }
      clause_buffer.clear();
    };
    for (const char32_t c : sentence) {
      clause_buffer.push_back(c);
      if (kClauseEnd.find(c) != std::u32string_view::npos) {
        flush();
      }
    }
    if (!clause_buffer.empty()) {
      flush();
    }
    for (std::size_t i = 0; i < clauses.size();) {
      const std::u32string& clause = clauses[i];
      const std::size_t clause_len = SemanticLengthU32(clause);
      if (clause_len < min_tail && i + 1 < clauses.size()) {
        const std::u32string combined = clause + clauses[i + 1];
        if (SemanticLengthU32(combined) <= max_len) {
          AppendTextFragment(&result, combined, max_len, min_tail);
          i += 2;
          continue;
        }
      }
      if (clause_len > max_len) {
        for (const std::u32string& piece : SplitLongFragment(clause, max_len)) {
          AppendTextFragment(&result, piece, max_len, min_tail);
        }
      } else {
        AppendTextFragment(&result, clause, max_len, min_tail);
      }
      ++i;
    }
  }
  for (std::u32string& fragment : result) {
    fragment = ReplaceAll(fragment, kPlaceholder, U".");
  }
  return result;
}

std::string StripMarkdownCode(std::string_view input) {
  // re.sub(r'``(.+?)``', r'\1'), re.sub(r'`(.+?)`', r'\1'), drop stray '`'.
  auto unwrap = [](std::string text, std::string_view fence) {
    std::string out;
    std::size_t i = 0;
    while (i < text.size()) {
      const std::size_t open = text.find(fence, i);
      if (open == std::string::npos) {
        break;
      }
      // .+? : at least one char, no newline, shortest
      const std::size_t search_from = open + fence.size() + 1;
      std::size_t close = std::string::npos;
      if (search_from <= text.size()) {
        close = text.find(fence, search_from);
      }
      const std::size_t newline = text.find('\n', open + fence.size());
      if (close == std::string::npos ||
          (newline != std::string::npos && newline < close)) {
        out.append(text, i, open + 1 - i);
        i = open + 1;
        continue;
      }
      out.append(text, i, open - i);
      out.append(text, open + fence.size(), close - open - fence.size());
      i = close + fence.size();
    }
    out.append(text, i, std::string::npos);
    return out;
  };
  std::string text = unwrap(std::string(input), "``");
  text = unwrap(text, "`");
  text.erase(std::remove(text.begin(), text.end(), '`'), text.end());
  return text;
}

std::string JsonString(std::string_view value) {
  std::string out = "\"";
  for (const char raw : value) {
    const auto c = static_cast<unsigned char>(raw);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          out += buffer;
        } else {
          out.push_back(raw);
        }
    }
  }
  out.push_back('"');
  return out;
}

std::string JsonOptional(const std::optional<std::string>& value) {
  return value.has_value() ? JsonString(*value) : "null";
}

}  // namespace

std::string BuildTtsPrompt(std::string_view language) {
  const std::string trimmed = Trim(language);
  if (trimmed.empty()) {
    return std::string(kDefaultTtsPrompt);
  }
  static const std::array<std::pair<std::string_view, std::string_view>, 18>
      kNames = {{{"zh", "Chinese"},
                 {"zh-cn", "Chinese"},
                 {"chinese", "Chinese"},
                 {"en", "English"},
                 {"english", "English"},
                 {"ja", "Japanese"},
                 {"japanese", "Japanese"},
                 {"ko", "Korean"},
                 {"korean", "Korean"},
                 {"de", "German"},
                 {"german", "German"},
                 {"fr", "French"},
                 {"french", "French"},
                 {"es", "Spanish"},
                 {"spanish", "Spanish"},
                 {"yue", "Chinese"},
                 {"zh-yue", "Chinese"},
                 {"cantonese", "Chinese"}}};
  const std::string lower = ToLower(trimmed);
  std::string name = trimmed;
  for (const auto& [key, value] : kNames) {
    if (lower == key) {
      name = std::string(value);
    }
  }
  return std::string(kDefaultTtsPrompt) + "Target language: " + name + ".\n";
}

std::optional<std::string> EmotionLabel(std::string_view name) {
  static const std::array<std::pair<std::string_view, std::string_view>, 34>
      kLabels = {{
          {"高兴", "高兴"},        {"happy", "高兴"},     {"joy", "高兴"},
          {"fröhlich", "高兴"},    {"freude", "高兴"},    {"glücklich", "高兴"},
          {"triumph", "高兴"},     {"悲伤", "悲伤"},      {"sad", "悲伤"},
          {"traurig", "悲伤"},     {"trauer", "悲伤"},    {"verzweiflung", "悲伤"},
          {"愤怒", "愤怒"},        {"angry", "愤怒"},     {"anger", "愤怒"},
          {"wut", "愤怒"},         {"wütend", "愤怒"},    {"恐惧", "恐惧"},
          {"fear", "恐惧"},        {"angst", "恐惧"},     {"ängstlich", "恐惧"},
          {"厌恶", "厌恶"},        {"disgust", "厌恶"},   {"ekel", "厌恶"},
          {"angewidert", "厌恶"},  {"惊讶", "惊讶"},      {"surprise", "惊讶"},
          {"surprised", "惊讶"},   {"überraschung", "惊讶"}, {"überrascht", "惊讶"},
          {"中性", "中性"},        {"neutral", "中性"},   {"sachlich", "中性"},
          {"ruhig", "中性"},
      }};
  const std::string lower = ToLower(Trim(name));
  for (const auto& [key, value] : kLabels) {
    if (lower == key) {
      return std::string(value);
    }
  }
  return std::nullopt;
}

namespace {

template <std::size_t N>
std::optional<std::string> LookupLabel(
    const std::array<std::pair<std::string_view, std::string_view>, N>& table,
    std::string_view name) {
  const std::string lower = ToLower(Trim(name));
  for (const auto& [key, value] : table) {
    if (lower == key) {
      return std::string(value);
    }
  }
  return std::nullopt;
}

}  // namespace

std::optional<std::string> SpeedLabel(std::string_view name) {
  static const std::array<std::pair<std::string_view, std::string_view>, 11>
      kLabels = {{
          {"慢速", "慢速"}, {"slow", "慢速"},   {"langsam", "慢速"},
          {"中速", "中速"}, {"medium", "中速"}, {"normal", "中速"},
          {"mittel", "中速"}, {"快速", "快速"}, {"fast", "快速"},
          {"schnell", "快速"}, {"zügig", "快速"},
      }};
  return LookupLabel(kLabels, name);
}

std::optional<std::string> PitchLabel(std::string_view name) {
  static const std::array<std::pair<std::string_view, std::string_view>, 11>
      kLabels = {{
          {"低", "低"},     {"low", "低"},      {"tief", "低"},
          {"niedrig", "低"}, {"中", "中"},       {"medium", "中"},
          {"normal", "中"}, {"mittel", "中"},   {"高", "高"},
          {"high", "高"},   {"hoch", "高"},
      }};
  return LookupLabel(kLabels, name);
}

std::optional<std::string> VolumeLabel(std::string_view name) {
  static const std::array<std::pair<std::string_view, std::string_view>, 15>
      kLabels = {{
          {"低", "低"},      {"low", "低"},     {"quiet", "低"},
          {"soft", "低"},    {"leise", "低"},   {"中", "中"},
          {"medium", "中"},  {"normal", "中"},  {"mittel", "中"},
          {"高", "高"},      {"high", "高"},    {"loud", "高"},
          {"laut", "高"},    {"hoch", "高"},    {"niedrig", "低"},
      }};
  return LookupLabel(kLabels, name);
}

std::optional<std::string> DialectLabel(std::string_view name) {
  static const std::array<std::pair<std::string_view, std::string_view>, 10>
      kLabels = {{
          {"cantonese", "cantonese"},
          {"yue", "cantonese"},
          {"kantonesisch", "cantonese"},
          {"粤语", "cantonese"},
          {"chuanyu dialect", "chuanyu dialect"},
          {"chuanyu", "chuanyu dialect"},
          {"sichuanese", "chuanyu dialect"},
          {"sichuan", "chuanyu dialect"},
          {"sichuanisch", "chuanyu dialect"},
          {"川渝话", "chuanyu dialect"},
      }};
  return LookupLabel(kLabels, name);
}

std::string BuildStructuredInstruction(const StructuredControls& c,
                                       InstructionSchema schema) {
  if (schema == InstructionSchema::kEnglish) {
    std::string out = "{\"audio_sequence\": [{\"id\": 1, \"speaker\": \"speaker_1\", ";
    out += "\"dialect\": " + JsonOptional(c.dialect) + ", ";
    out += "\"style\": " + JsonOptional(c.style) + ", ";
    out += "\"speed\": " + JsonOptional(c.speed) + ", ";
    out += "\"pitch\": " + JsonOptional(c.pitch) + ", ";
    out += "\"volume\": " + JsonOptional(c.volume) + ", ";
    out += "\"emotion\": " + JsonOptional(c.emotion) + ", ";
    out += "\"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, "
           "\"Theme\": null, \"ENV\": null, \"SNR\": null}, ";
    out += "\"ip\": " + JsonOptional(c.ip);
    out += "}]}";
    return out;
  }
  std::string out = "{\"audio_sequence\": [{";
  out += "\"\xE5\xBA\x8F\xE5\x8F\xB7\": 1, ";                  // 序号
  out += "\"\xE8\xAF\xB4\xE8\xAF\x9D\xE4\xBA\xBA\": \"speaker_1\", ";  // 说话人
  out += "\"\xE6\x96\xB9\xE8\xA8\x80\": " + JsonOptional(c.dialect) + ", ";  // 方言
  out += "\"\xE9\xA3\x8E\xE6\xA0\xBC\": " + JsonOptional(c.style) + ", ";    // 风格
  out += "\"\xE8\xAF\xAD\xE9\x80\x9F\": " + JsonOptional(c.speed) + ", ";    // 语速
  out += "\"\xE5\x9F\xBA\xE9\xA2\x91\": " + JsonOptional(c.pitch) + ", ";    // 基频
  out += "\"\xE9\x9F\xB3\xE9\x87\x8F\": " + JsonOptional(c.volume) + ", ";   // 音量
  out += "\"\xE6\x83\x85\xE6\x84\x9F\": " + JsonOptional(c.emotion) + ", ";  // 情感
  out += "\"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, "
         "\"Theme\": null, \"ENV\": null, \"SNR\": null}, ";
  out += "\"IP\": " + JsonOptional(c.ip);
  out += "}]}";
  return out;
}

bool BuildInputIds(const Tokenizer& tokenizer, std::string_view prompt,
                   std::string_view text, std::size_t num_speakers,
                   const std::optional<std::string>& instruction,
                   InputLayout* layout, std::string* error,
                   const std::optional<std::string>& prompt_text,
                   std::size_t prompt_latent_steps) {
  layout->ids.clear();
  layout->speaker_positions.clear();
  layout->latent_position = -1;
  const bool with_latents = prompt_text.has_value() && prompt_latent_steps > 0;
  auto append = [&](std::string_view piece) {
    std::vector<std::uint32_t> ids;
    if (!tokenizer.Encode(piece, &ids, error)) {
      return false;
    }
    layout->ids.insert(layout->ids.end(), ids.begin(), ids.end());
    return true;
  };
  if (!append(kSystemPrefix) || !append(kUserPrefix) || !append(prompt)) {
    return false;
  }
  for (std::size_t index = 0; index < num_speakers; ++index) {
    const std::string label = "  speaker_" + std::to_string(index + 1) + ":";
    if (!append(label) || !append(kSpeakerStart) || !append(kSpeakerPad) ||
        !append(kSpeakerEnd)) {
      return false;
    }
  }
  constexpr std::array<std::string_view, 5> kBgmMarkers = {
      "Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "};
  const bool bgm = std::ranges::all_of(kBgmMarkers, [&](std::string_view m) {
    return text.find(m) != std::string_view::npos;
  });
  if (!bgm && !append(kTextInputPrefix)) {
    return false;
  }
  if (with_latents && !append(*prompt_text)) {
    return false;
  }
  if (!append(text) || !append(kUserSuffix) || !append(kAssistantPrefix)) {
    return false;
  }
  if (instruction.has_value() && (!append(*instruction) || !append(kInstructionEos))) {
    return false;
  }
  if (!append(kAudioStart)) {
    return false;
  }
  for (std::size_t i = 0; i + 1 < layout->ids.size(); ++i) {
    if (layout->ids[i] == kVisionStartId) {
      if (layout->ids[i + 1] != kVisionPadId) {
        if (error != nullptr) {
          *error = "speaker slot is not followed by its pad token";
        }
        return false;
      }
      layout->speaker_positions.push_back(i + 1);
    }
  }
  if (layout->speaker_positions.size() != num_speakers) {
    if (error != nullptr) {
      *error = "speaker slot count mismatch (text must not contain vision tokens)";
    }
    return false;
  }
  if (with_latents) {
    layout->latent_position = static_cast<std::int64_t>(layout->ids.size());
    layout->ids.insert(layout->ids.end(), prompt_latent_steps, kAudioPatchId);
  }
  return true;
}

std::vector<std::string> SplitTextFragments(std::string_view text,
                                            std::size_t max_length) {
  const std::u32string prepared =
      Strip(ToU32(Nfc(StripMarkdownCode(text))));
  std::vector<std::string> out;
  for (const std::u32string& fragment :
       CutTextBySemanticLength(prepared, max_length, 5)) {
    if (!Strip(fragment).empty()) {
      out.push_back(ToUtf8(fragment));
    }
  }
  return out;
}

std::size_t SemanticLength(std::string_view text) {
  return SemanticLengthU32(ToU32(Nfc(text)));
}

}  // namespace gufo::models::ming_tts
