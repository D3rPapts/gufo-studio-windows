// Ming-3.0-tts extension registration (metadata, probe, artifact check).
// Written with AI assistance (Claude Code).
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <string>
#include <string_view>

#include "ming_tts/config.hpp"

namespace gufo::models::ming_tts {

bool ProbeMingTts(std::string_view model_root) {
  return LooksLikeMingTts(std::string(model_root));
}

bool ResolveMingTtsArtifacts(std::string_view model_root, std::string* error) {
  if (!LooksLikeMingTts(std::string(model_root))) {
    if (error != nullptr) {
      *error = "no Ming-3.0-tts checkpoint (config.json, model.safetensors, "
               "campplus.onnx) in " +
               std::string(model_root);
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::ming_tts

void ming_tts_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::ming_tts;
  registry.AddTtsProvider({
      .id = "ming-tts",
      .probe = &ProbeMingTts,
      .resolve = &ResolveMingTtsArtifacts,
      .create = nullptr,
  });
}
