// IndexTTS-2.5 extension registration (metadata, probe, artifact check).
// Written with AI assistance (Claude Code).
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <filesystem>
#include <string>
#include <string_view>

#include "indextts/bundle.hpp"

namespace gufo::models::indextts {

bool ProbeIndexTts(std::string_view model_root) {
  return LooksLikeIndexTts25(std::filesystem::path(model_root));
}

bool ResolveIndexTtsArtifacts(std::string_view model_root, std::string* error) {
  if (!LooksLikeIndexTts25(std::filesystem::path(model_root))) {
    if (error != nullptr) {
      *error = "no IndexTTS-2.5 bundle (config.yaml version 2.5, german/gpt.pth or "
               "gpt.pth, s2mel.pth, codec.pth, multilingual_zh_ja_yue_char_del.tiktoken, "
               "w2v-bert-2.0/model.safetensors, campplus/campplus_cn_common.bin, "
               "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt) in " +
               std::string(model_root);
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::indextts

void indextts_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::indextts;
  registry.AddTtsProvider({
      .id = "indextts",
      .probe = &ProbeIndexTts,
      .resolve = &ResolveIndexTtsArtifacts,
      .create = nullptr,
  });
}
