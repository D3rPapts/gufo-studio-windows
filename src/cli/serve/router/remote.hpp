#ifndef GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_
#define GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_

// Remote models (`[remote/<id>]` preset sections): the allow-list check,
// and the shared state the router keeps about the remote routers.

#include <chrono>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/jobs.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/core/json.hpp"

namespace gufo::router {

/// Whether --remote-allow covers `endpoint`: an IPv4 address or a.b.c.d/n
/// range entry (port 0 = any port) or `*`. Host-name entries never match,
/// since remote urls are IPv4 only.
[[nodiscard]] bool RemoteEndpointAllowed(
    const RemoteEndpoint& endpoint, const std::vector<jobs::AllowEntry>& allow);

/// Checks a remote section's url and busy-check url against --remote-allow.
/// Returns false with the reason in `error`.
[[nodiscard]] bool CheckRemoteAllowed(const RemoteSpec& spec,
                                      const std::vector<jobs::AllowEntry>& allow,
                                      std::string* error);

/// Relay target for a remote model: its router's address, the preset
/// timeouts, keepalive and the remote router's key as Authorization.
[[nodiscard]] UpstreamTarget RemoteTarget(const RemoteSpec& spec);

/// 503 remote_unavailable with Retry-After: 30, for a remote router that
/// cannot be reached (never a reason to stop anything locally).
[[nodiscard]] server::HttpResponse RemoteUnavailable(
    const std::string& model_id, const std::string& detail);

/// "host:port" of an endpoint, the key remote routers are known by.
[[nodiscard]] std::string EndpointKey(const RemoteEndpoint& endpoint);

/// One GET /v1/models against a remote router.
struct RemoteProbe {
  /// The connection worked and an HTTP answer came back.
  bool reachable{false};
  /// Why it failed, or a non-200 answer ("HTTP 401 ...").
  std::string error;
  /// Model id -> llama.cpp status value ("loaded", "loading", "unloaded").
  std::map<std::string, std::string> models;
};

/// Asks `url` for its model list with `key` as bearer token; connect, head
/// and the whole answer each bounded by `timeout`. Blocking: call it with
/// no router lock held.
[[nodiscard]] RemoteProbe ProbeRemoteModels(const RemoteEndpoint& url,
                                            const std::string& key,
                                            std::chrono::milliseconds timeout);

/// What the router knows about each remote router (one entry per url):
/// reachability and the model states it listed. The remote thread refreshes
/// it every few seconds; a relay that cannot connect marks the router
/// unreachable at once. Thread-safe and never calls back into the router,
/// so it may be used with the router's state mutex held.
class RemoteHub {
public:
  using Clock = std::chrono::steady_clock;

  explicit RemoteHub(const std::vector<PresetModel>& presets);

  [[nodiscard]] bool empty() const { return routers_.empty(); }

  struct Target {
    RemoteEndpoint url;
    std::string api_key;
  };
  /// The routers to probe, with the key of the first section naming each.
  [[nodiscard]] std::vector<Target> Targets() const;

  /// Applies a probe. Returns the preset ids for this router that the
  /// remote does not list (once per change, for a warning).
  std::vector<std::string> Record(const RemoteEndpoint& url,
                                  RemoteProbe probe);
  /// A relay could not connect: refuse further relays until a probe works.
  void MarkUnreachable(const RemoteEndpoint& url, const std::string& error);
  /// False after a failed probe or connect (never probed counts as
  /// reachable, so the first request tries); `error` gets the reason.
  [[nodiscard]] bool Reachable(const RemoteEndpoint& url,
                               std::string* error) const;
  /// The status value the remote last listed for `id`; "" when unknown.
  [[nodiscard]] std::string ModelStatus(const RemoteEndpoint& url,
                                        const std::string& id) const;
  /// {"<host:port>": {reachable, error, checked_s_ago, models}}.
  [[nodiscard]] json::Value Status() const;

private:
  struct RouterInfo {
    RemoteEndpoint url;
    std::string api_key;
    std::vector<std::string> ids;  // preset ids served there
    bool checked{false};
    bool reachable{true};
    std::string error;
    Clock::time_point checked_at{};
    std::map<std::string, std::string> models;
    std::vector<std::string> missing;  // last warned set
  };
  mutable std::mutex mutex_;
  std::map<std::string, RouterInfo> routers_;
};

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_
