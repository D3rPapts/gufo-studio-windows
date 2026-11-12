#ifndef GUFO_CLI_SERVE_ROUTER_PRESET_HPP_
#define GUFO_CLI_SERVE_ROUTER_PRESET_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
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

/// `http://<IPv4>:<port>[/path]`: a remote gufo router or a busy-check
/// endpoint. No host names, no https, no user info.
struct RemoteEndpoint {
  std::string host;          // dotted IPv4
  std::uint32_t address{0};  // host byte order
  int port{0};
  std::string path;  // "" for a router url, "/queue" for a busy check
};

/// `[remote/<id>]`: the model runs on another gufo router under the same id.
/// The router never starts, measures or unloads it; requests are relayed.
struct RemoteSpec {
  RemoteEndpoint url;
  /// Environment variable holding the remote router's --api-key ("" = none)
  /// and its value, read when the preset loads.
  std::string api_key_env;
  std::string api_key;
  int connect_timeout_seconds{3};
  /// Response head deadline once the request is sent; 0 = none (a long
  /// image or a non-streamed chat answers only when done).
  int head_timeout_seconds{0};
  /// Remote models sharing a name share one GPU: one GPU-starting request
  /// at a time ("" = no slot).
  std::string gpu_slot;
  /// Optional GET that reports the remote GPU busy (ComfyUI /queue).
  std::optional<RemoteEndpoint> busy_check;
  /// Longest wait for the gpu slot (and the busy check) before 503.
  int queue_timeout_seconds{900};
  /// Request body limit for this model (0 = only --max-request-bytes).
  std::size_t max_request_bytes{0};
};

/// Where a model's requests go: the local worker, the remote router, or
/// (kPool, switchable models only) per request to whichever of the two is
/// free; see pool.hpp.
enum class Placement { kLocal, kRemote, kPool };

[[nodiscard]] inline const char* PlacementName(Placement placement) {
  return placement == Placement::kRemote ? "remote"
         : placement == Placement::kPool ? "pool"
                                         : "local";
}

struct PresetModel {
  std::string modality;  // "llm"|"video"|"image"|"tts"|"asr"|"embedding"|
                         // "external"; a remote section's `modality` key
  std::string id;        // routing key from the section header
  std::vector<std::string> argv;  // worker args, e.g. {"--model", "/x.gguf"}
  /// external only: the program to start instead of `gufo serve`. It gets
  /// argv plus `--host 127.0.0.1 --port N` and must answer GET /health with
  /// 200 once ready, as llama-server does. An `args` key puts leading
  /// positional arguments first (`args = serve llm` for another gufo build).
  std::string command;
  PresetHints hints{};
  /// Set for `[remote/<id>]` sections, and for a local section with a
  /// `[remote/<id>]` twin of the same id.
  std::optional<RemoteSpec> remote;
  /// A local section with a `[remote/<id>]` twin: POST /router/placement
  /// moves it between the local worker and the remote router at runtime.
  bool switchable{false};
  /// Start placement: kRemote for `[remote/<id>]` alone, kLocal for a local
  /// section; a twin's `placement = local|remote|pool` key (default local).
  Placement placement{Placement::kLocal};
};

/// The worker capability of an external preset derived from its command line
/// ("serve objekt" -> "image_to_3d"); "" when none is known.
std::string CapabilityOf(const PresetModel& preset);

/// Parse + validate a preset file. `models_dir` (may be empty) resolves
/// relative path values; when empty, values are passed through unchanged.
/// Returns false and sets `error` (section-prefixed, e.g.
/// "[llm/a] Unknown option: --bogus") on any problem.
[[nodiscard]] bool LoadPreset(const std::filesystem::path& file,
                              const std::filesystem::path& models_dir,
                              std::vector<PresetModel>* models,
                              std::string* error);

/// Parses `http://<IPv4>:<port>` (with `allow_path`, an optional path
/// after it). Returns false and sets `error` otherwise.
[[nodiscard]] bool ParseRemoteUrl(std::string_view text, bool allow_path,
                                  RemoteEndpoint* out, std::string* error);

/// True when `key` is a router-managed option rejected in sections.
[[nodiscard]] bool IsReservedKey(std::string_view key);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_PRESET_HPP_
