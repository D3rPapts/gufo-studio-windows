// Fish Audio Drama-3 registration without the HIP runtime (probe and artifact
// check only); provider.cpp adds the runner factory in HIP builds.
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>
#include <string>
#include <string_view>

#include "config.hpp"
#include "probe.hpp"

namespace gufo::models::fish_drama3 {

bool ProbeFishDrama3(std::string_view model_root) {
  return LooksLikeFishDrama3(std::string(model_root));
}

bool ResolveFishDrama3Artifacts(std::string_view model_root,
                                std::string* error) {
  const auto config = LoadModelConfigFromPath(std::string(model_root));
  if (!config.has_value()) {
    if (error != nullptr) {
      *error = "cannot load config.json from " + std::string(model_root);
    }
    return false;
  }
  if (!IsSupportedModelConfig(*config)) {
    if (error != nullptr) {
      *error = "unsupported Fish Drama-3 model in " + std::string(model_root);
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::fish_drama3

void fish_drama3_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::fish_drama3;
  registry.AddTtsProvider({
      .id = "fish-drama-3",
      .probe = &ProbeFishDrama3,
      .resolve = &ResolveFishDrama3Artifacts,
      .create = nullptr,
  });
}
