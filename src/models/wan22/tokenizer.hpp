#ifndef GUFO_MODELS_WAN22_TOKENIZER_HPP_
#define GUFO_MODELS_WAN22_TOKENIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::wan22 {

/// Collapses every run of whitespace to one space and trims both ends, as
/// the Wan pipeline's `whitespace_clean` does before tokenizing. ftfy and
/// HTML unescaping of `basic_clean` are not applied.
[[nodiscard]] std::string CleanPrompt(std::string_view text);

/// umT5 Unigram tokenizer. It reproduces the Hugging Face `tokenizers`
/// output for the Wan 2.2 `tokenizer/tokenizer.json`: added tokens split
/// first (leftmost, longest), every other segment collapses runs of spaces,
/// becomes U+2581-separated words with a leading U+2581 (Metaspace, prepend
/// "always", split), and each word is segmented by Viterbi over the piece
/// scores. Characters no piece covers become `<unk>`, adjacent ones fused.
/// `</s>` ends the result.
class Tokenizer {
public:
  static constexpr std::int32_t kPadId = 0;
  static constexpr std::int32_t kEosId = 1;

  /// Reads and validates `tokenizer.json`. On failure it returns nullopt and,
  /// when `error` is not null, describes the problem.
  [[nodiscard]] static std::optional<Tokenizer> Load(
      const std::filesystem::path& path, std::string* error = nullptr);

  Tokenizer(Tokenizer&&) noexcept = default;
  Tokenizer& operator=(Tokenizer&&) noexcept = default;
  // `ids_` keys view the strings in `pieces_`.
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  ~Tokenizer() = default;

  /// Encodes UTF-8 `text` and appends `</s>`. Content tokens beyond
  /// `max_tokens - 1` are dropped, as `truncation=True` does.
  [[nodiscard]] std::vector<std::int32_t> Encode(
      std::string_view text, std::size_t max_tokens = 512) const;

  [[nodiscard]] std::size_t vocab_size() const { return pieces_.size(); }

private:
  Tokenizer() = default;

  struct AddedToken {
    std::string content;
    std::int32_t id{0};
  };

  void EncodeSegment(std::string_view text,
                     std::vector<std::int32_t>* out) const;
  void EncodeWord(std::string_view word, std::vector<std::int32_t>* out) const;
  [[nodiscard]] std::size_t MatchAddedToken(std::string_view text,
                                            std::size_t pos,
                                            std::int32_t* id) const;

  std::vector<std::string> pieces_;
  std::vector<double> scores_;
  std::unordered_map<std::string_view, std::int32_t> ids_;
  std::vector<AddedToken> added_;  // longest first
  bool added_first_byte_[256]{};
  std::size_t max_piece_bytes_{0};
  std::int32_t unk_id_{0};
  double unk_score_{0};
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TOKENIZER_HPP_
