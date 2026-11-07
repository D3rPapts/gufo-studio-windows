// MMAudio for gufo: the CLIP BPE tokenizer of the DFN5B text tower.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_TOKENIZER_HPP_
#define GUFO_MODELS_MMAUDIO_TOKENIZER_HPP_

#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::mmaudio {

/// OpenCLIP's SimpleTokenizer (the one ViT-H-14 / DFN5B use) from the
/// Hugging Face files of the same vocabulary: `vocab.json` and
/// `merges.txt` (e.g. laion/CLIP-ViT-H-14-laion2B-s32B-b79K). Text is
/// HTML-unescaped, whitespace-collapsed and lower-cased, split with CLIP's
/// pattern, byte-encoded and BPE-merged; the result is wrapped in
/// <|startoftext|> / <|endoftext|>, cut to 77 tokens (the last one kept as
/// <|endoftext|>) and padded with 0. ftfy's mojibake repair is not done.
class ClipTokenizer {
public:
  static constexpr int kContext = 77;
  static constexpr int kStartOfText = 49406;
  static constexpr int kEndOfText = 49407;

  /// Reads `dir`/vocab.json and `dir`/merges.txt.
  bool Load(const std::filesystem::path& dir, std::string* error);
  /// 77 token ids.
  [[nodiscard]] std::vector<int> Encode(std::string_view text) const;
  /// The ids without start/end tokens and without the 77 limit.
  [[nodiscard]] std::vector<int> EncodeWords(std::string_view text) const;

private:
  void Bpe(const std::string& token, std::vector<int>* ids) const;

  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> ranks_;  // "left right" -> rank
  std::array<std::string, 256> byte_encoder_;
};

/// The text cleaning of SimpleTokenizer before the split: HTML entities
/// unescaped (twice), whitespace runs collapsed to one space, trimmed,
/// lower-cased (Unicode).
std::string CleanClipText(std::string_view text);

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_TOKENIZER_HPP_
