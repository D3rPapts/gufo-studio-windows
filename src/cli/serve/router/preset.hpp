#ifndef GUFO_CLI_SERVE_ROUTER_PRESET_HPP_
#define GUFO_CLI_SERVE_ROUTER_PRESET_HPP_

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::router {

/// Router-only keys (`router-*`), never passed to the worker:
/// router-gpu-gb / router-host-gb: peak memory estimate (GB, < 0 = unset),
/// router-exclusive: nothing else may stay loaded beside this model,
/// router-idle-seconds: idle unload override (< 0 = router default).
struct PresetHints {
  double gpu_gb{-1.0};
  double host_gb{-1.0};
  int exclusive{-1};  // -1 unset, 0 false, 1 true
  long long idle_seconds{-1};
};

struct PresetModel {
  std::string modality;  // "llm"|"video"|"image"|"tts"|"asr"|"embedding"|
                         // "external"
  std::string id;        // routing key from the section header
  std::vector<std::string> argv;  // worker args, e.g. {"--model", "/x.gguf"}
  /// external only: the program to start instead of `gufo serve`. It gets
  /// argv plus `--host 127.0.0.1 --port N` and must answer GET /health with
  /// 200 once ready, as llama-server does. An `args` key puts leading
  /// positional arguments first (`args = serve llm` for another gufo build).
  std::string command;
  PresetHints hints{};
};

/// Parse + validate a preset file. `models_dir` (may be empty) resolves
/// relative path values; when empty, values are passed through unchanged.
/// Returns false and sets `error` (section-prefixed, e.g.
/// "[llm/a] Unknown option: --bogus") on any problem.
[[nodiscard]] bool LoadPreset(const std::filesystem::path& file,
                              const std::filesystem::path& models_dir,
                              std::vector<PresetModel>* models,
                              std::string* error);

/// True when `key` is a router-managed option rejected in sections.
[[nodiscard]] bool IsReservedKey(std::string_view key);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_PRESET_HPP_
