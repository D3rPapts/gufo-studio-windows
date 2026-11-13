// Written with AI assistance (Claude Code).
#include "indextts/bundle.hpp"

#include <fstream>
#include <string>
#include <system_error>

namespace gufo::models::indextts {

std::filesystem::path GptPath(const std::filesystem::path& dir) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(dir / "german" / "gpt.pth", ec)) {
    return dir / "german" / "gpt.pth";
  }
  return dir / "gpt.pth";
}

bool LooksLikeIndexTts25(const std::filesystem::path& dir) {
  std::error_code ec;
  const auto exists = [&](const std::filesystem::path& p) {
    return std::filesystem::is_regular_file(p, ec);
  };
  if (!exists(dir / "config.yaml") || !exists(GptPath(dir)) || !exists(dir / "s2mel.pth") ||
      !exists(dir / "codec.pth") || !exists(dir / "multilingual_zh_ja_yue_char_del.tiktoken") ||
      !exists(dir / "w2v-bert-2.0" / "model.safetensors") ||
      !exists(dir / "campplus" / "campplus_cn_common.bin") ||
      !exists(dir / "bigvgan_v2_22khz_80band_256x" / "bigvgan_generator.pt")) {
    return false;
  }
  std::ifstream in(dir / "config.yaml");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("version:", 0) == 0 && line.find("2.5") != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace gufo::models::indextts
