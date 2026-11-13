// VoxCPM2 text tokenizer: the checkpoint's tokenizer.json (LlamaTokenizerFast,
// SentencePiece-style BPE with byte fallback, "▁" prepended and replacing
// spaces, no pre-tokenizer) wrapped like voxcpm's mask_multichar_chinese_tokens
// (multi-character CJK tokens are split into single characters). No BOS/EOS is
// added, matching VoxCPM2Model.text_tokenizer(text).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_TOKENIZER_HPP_
#define GUFO_MODELS_VOXCPM2_TOKENIZER_HPP_

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gufo::models::voxcpm2 {

class Tokenizer {
public:
  [[nodiscard]] static bool Load(const std::string& model_dir, Tokenizer* out,
                                 std::string* error);

  // Token ids of `text` (VoxCPM2Model.text_tokenizer).
  [[nodiscard]] std::vector<std::int64_t> Encode(std::string_view text) const;
  // The BPE tokens before the CJK split (LlamaTokenizerFast.tokenize).
  [[nodiscard]] std::vector<std::string> Tokenize(std::string_view text) const;

  [[nodiscard]] std::size_t vocab_size() const noexcept { return tokens_.size(); }

private:
  void EncodeSegment(std::string_view segment, std::vector<std::int32_t>* ids) const;

  std::vector<std::string> tokens_;                      // id -> token
  std::unordered_map<std::string, std::int32_t> vocab_;  // token -> id
  std::unordered_map<std::uint64_t, std::pair<std::int32_t, std::int32_t>>
      merges_;  // (left, right) -> (rank, merged id)
  std::vector<std::pair<std::string, std::int32_t>> added_;  // longest first
  std::int32_t byte_ids_[256]{};
  std::int32_t unk_id_{0};
  std::unordered_set<std::string> multichar_cjk_;
};

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_TOKENIZER_HPP_
