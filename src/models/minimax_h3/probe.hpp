#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/ext/video.hpp"

namespace gufo::models::minimax_h3 {

[[nodiscard]] bool ProbeMinimaxH3(std::string_view model_root);
[[nodiscard]] bool ResolveMinimaxH3Artifacts(
    std::string_view model_root, std::string_view source_manifest = {},
    std::string* error = nullptr);

#if defined(ENGINE_ENABLE_HIP)
[[nodiscard]] std::unique_ptr<ext::VideoRunner> CreateMinimaxH3Runner(
    const std::string& model_root, std::string* error = nullptr);
/// gufo-studio-windows: `serve video --lora/--vision-projector` (1.4.9).
[[nodiscard]] std::unique_ptr<ext::VideoRunner> CreateMinimaxH3RunnerWithOptions(
    const ext::VideoRunnerOptions& options, std::string* error);
#endif

/// The 1.4.9 H3 request rules (keyframes, guide frames/clips, audio anchor,
/// LoRA, steps) as `VideoProvider::prepare_request`.
bool PrepareMinimaxH3Request(const ext::VideoRequestInput& input,
                             ext::VideoPreparedRequest* prepared,
                             ext::VideoRequestError* error);
/// `gufo video --h3-prompt-child <request> <response>` (vision + text
/// encoding in a short-lived child process on Windows); -1 for other args.
int MinimaxH3ChildEntry(std::span<const char* const> args);

[[nodiscard]] std::optional<ext::VideoParameters> ResolvePreset(
    std::string_view name, std::string* error = nullptr);
[[nodiscard]] std::optional<int> ResolveDuration(std::string_view seconds,
                                                 std::string* error = nullptr);
[[nodiscard]] bool ValidateParameters(const ext::VideoParameters& parameters,
                                      std::string* error = nullptr);
[[nodiscard]] std::string ParametersJson(const ext::VideoParameters& parameters,
                                         std::uint64_t seed,
                                         std::string_view prompt_sha256);
[[nodiscard]] std::string TelemetryJson(const ext::VideoTelemetry& telemetry);

}  // namespace gufo::models::minimax_h3
