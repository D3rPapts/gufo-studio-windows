// Text frontend of IndexTTS-2.5-German: prepare_german() of the fine-tune
// (german_text.py: character cleanup, German number/date/currency/unit/
// abbreviation expansion, lower case, pronunciation annotations), the
// cleanup steps of IndexTTS2.infer_generator for lang "de" (text
// normalization off), the token-budget segmentation and the multilingual
// tiktoken BPE (multilingual_zh_ja_yue_char_del.tiktoken + Whisper-style
// special tokens). Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_TEXT_HPP_
#define GUFO_MODELS_INDEXTTS_TEXT_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gufo::models::indextts {

// german_text.prepare_german (UTF-8 in/out).
[[nodiscard]] std::string PrepareGerman(const std::string& text);
// german_text.number_to_words for non-negative values below 2^63.
[[nodiscard]] std::string GermanNumberToWords(std::uint64_t number);

class Tokenizer {
public:
  [[nodiscard]] static std::unique_ptr<Tokenizer> Load(const std::filesystem::path& tiktoken_file,
                                                       std::string* error);
  // tiktoken Encoding.encode(text, allowed_special="all").
  [[nodiscard]] std::vector<std::int32_t> Encode(const std::string& text) const;
  [[nodiscard]] std::int32_t SpecialId(const std::string& token) const;  // -1 if unknown
  [[nodiscard]] std::size_t rank_count() const noexcept { return ranks_.size(); }

private:
  void EncodeOrdinary(const std::string& text, std::vector<std::int32_t>* out) const;
  void BytePairEncode(const std::string& piece, std::vector<std::int32_t>* out) const;
  std::unordered_map<std::string, std::int32_t> ranks_;
  std::vector<std::pair<std::string, std::int32_t>> specials_;  // longest first
  std::unordered_map<std::string, std::int32_t> special_ids_;
};

// LANGUAGE_DICT index of indextts.utils.tokenizer (lang_to_token); "common"
// for unknown codes.
[[nodiscard]] std::int32_t LanguageToken(const std::string& lang);

struct PreparedText {
  std::string prepared;               // after prepare_german
  std::string final_text;             // what infer_generator tokenizes
  std::vector<std::string> segments;  // split_text_by_tokens
  std::vector<std::vector<std::int32_t>> ids;  // "<|de|> " + segment, + stop token 1
};

// Full German path of indextts_de.py + IndexTTS2.infer_generator.
[[nodiscard]] PreparedText PrepareForSynthesis(const Tokenizer& tokenizer, const std::string& text,
                                               const std::string& lang = "de",
                                               std::int32_t max_text_tokens_per_segment = 120,
                                               std::int32_t text_pos_capacity = 602);

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_TEXT_HPP_
