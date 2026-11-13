// MiniMax H3 requests as a gufo video provider (gufo-studio-windows, 2.0):
// the 1.4.9 /v1/videos rules for H3 (keyframes, guide frames and clips, audio
// anchor, LoRA, steps, presets) as `VideoProvider::prepare_request`, and the
// decoding of the prepared job back into a GenerationRequest.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MINIMAX_H3_REQUEST_HPP_
#define GUFO_MODELS_MINIMAX_H3_REQUEST_HPP_

#include <filesystem>
#include <gufo/ext/video.hpp>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/generation.hpp"

namespace gufo::minimax_h3 {

/// A LoRA the operator registered with `--lora NAME=PATH[,steps=N][,strength=S]`.
struct RegisteredLora {
  std::filesystem::path path;
  int steps{0};  // 0 keeps the preset's evaluations
  float strength{1.0F};
};

/// Provider options of `serve video` (in order): "lora" and
/// "vision-projector". Unknown names fail.
struct ProviderOptions {
  std::map<std::string, RegisteredLora, std::less<>> loras;
  std::filesystem::path vision_projector;
};
[[nodiscard]] bool ParseProviderOptions(
    const std::vector<std::pair<std::string, std::string>>& options,
    ProviderOptions* parsed, std::string* error);

/// The process-wide options of the running provider (one `serve video` per
/// process): prepare_request checks LoRA names and their default steps.
void SetActiveProviderOptions(ProviderOptions options);
[[nodiscard]] ProviderOptions ActiveProviderOptions();

/// `VideoProvider::prepare_request` for minimax-h3.
bool PrepareRequest(const ext::VideoRequestInput& input,
                    ext::VideoPreparedRequest* prepared,
                    ext::VideoRequestError* error);

/// Fills keyframes, audio guides, guide clips and the LoRA of `generation`
/// from a prepared job (`options_json` + `attachments`). Guide clips are
/// decoded with FFmpeg into `work_directory`.
[[nodiscard]] bool DecodePreparedRequest(const ext::VideoRequest& request,
                                         const ProviderOptions& options,
                                         const std::filesystem::path& work_directory,
                                         GenerationRequest* generation,
                                         std::string* error);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_REQUEST_HPP_
