// VoxCPM2 extension registration (metadata, probe, artifact check).
// Written with AI assistance (Claude Code).
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include <string>
#include <string_view>

#include "voxcpm2/config.hpp"

namespace gufo::models::voxcpm2 {

bool ProbeVoxCpm2(std::string_view model_root) {
  return LooksLikeVoxCpm2(std::string(model_root));
}

bool ResolveVoxCpm2Artifacts(std::string_view model_root, std::string* error) {
  if (!LooksLikeVoxCpm2(std::string(model_root))) {
    if (error != nullptr) {
      *error = "no VoxCPM2 checkpoint (config.json, model.safetensors, "
               "tokenizer.json, audiovae.pth or audiovae.safetensors) in " +
               std::string(model_root);
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::voxcpm2

void voxcpm2_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::voxcpm2;
  registry.AddTtsProvider({
      .id = "voxcpm2",
      .probe = &ProbeVoxCpm2,
      .resolve = &ResolveVoxCpm2Artifacts,
      .create = nullptr,
  });
}
