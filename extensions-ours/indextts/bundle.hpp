// IndexTTS-2.5 bundle layout: detection (the extension probe) and the GPT
// checkpoint path. Kept out of synthesis.cpp so that the registration target
// can probe without the engine. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_BUNDLE_HPP_
#define GUFO_MODELS_INDEXTTS_BUNDLE_HPP_

#include <filesystem>

namespace gufo::models::indextts {

// True for an IndexTTS-2.5 bundle directory (config.yaml "version: 2.5",
// gpt.pth / german/gpt.pth, s2mel.pth, codec.pth, w2v-bert-2.0, campplus,
// bigvgan_v2_22khz_80band_256x, multilingual tiktoken).
[[nodiscard]] bool LooksLikeIndexTts25(const std::filesystem::path& dir);

// german/gpt.pth (the German fine-tune) when present, else gpt.pth.
[[nodiscard]] std::filesystem::path GptPath(const std::filesystem::path& dir);

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_BUNDLE_HPP_
