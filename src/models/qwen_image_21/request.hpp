#ifndef GUFO_MODELS_QWEN_IMAGE_21_REQUEST_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_REQUEST_HPP_

#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "src/ext/image.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace gufo::models::qwen_image_21 {

/// What the provider registered at startup (--lora, --lora-default,
/// --controlnet) and needs to accept a request.
struct ProviderState {
  std::set<std::string, std::less<>> loras;
  std::vector<Lora> default_loras;
  bool controlnet{false};
};

/// The 1.4.9 request of an ext::ImageRequest: the host's prompt, references,
/// size, steps and seed plus the extension ("gufo": negative_prompt,
/// guidance, strength, init_image, mask*, regions, base_strength, control_*,
/// preview; "loras") and the named image parts (init_image, mask,
/// control_image, region_image[]). Throws std::invalid_argument (HTTP 400)
/// for what the 1.4.9 image API rejected. Validates with ValidateRequest.
/// "preview" is read by the host (live previews, changes/0.2/live-previews).
Request BuildRequest(const ext::ImageRequest& request,
                     const ProviderState& state);

/// Converts the provider's `serve image` manifest options (extension.json,
/// delivered as ext::ParsedOptions) to the 1.4.9 startup inputs:
/// --lora NAME=PATH[;NAME=PATH...], --lora-default NAME:SCALE[;...] and
/// --controlnet FILE, plus the --keep-text-encoder switch. The lists are ';'-separated because a manifest option
/// cannot repeat. Throws std::invalid_argument.
struct StartupOptions {
  std::vector<std::pair<std::string, std::string>> loras;  // name, path
  std::vector<Lora> default_loras;
  std::string controlnet;
  bool keep_text_encoder{false};
};
StartupOptions ParseStartupOptions(const ext::ParsedOptions& options);

/// `GET /sdapi/v1/loras` body of the 1.4.9 image API.
std::string LorasJson(
    const std::vector<std::pair<std::string, std::string>>& files,
    const std::vector<LoraInfo>& infos, const std::vector<Lora>& defaults,
    const std::vector<Lora>& active);

/// "name:scale,..." or "none" for the request log.
std::string LoraList(const std::vector<Lora>& loras);

}  // namespace gufo::models::qwen_image_21
#endif
