#ifndef GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_
#define GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_

// Remote models (`[remote/<id>]` preset sections): the allow-list check,
// and the shared state the router keeps about the remote routers.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
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

/// Header a router sets on everything it relays to a remote router, with
/// its own router id. A router that receives it never relays the request on
/// to another remote (loop guard); local models serve it normally.
inline constexpr const char* kForwardedHeader = "X-Gufo-Forwarded";

/// Relay target for a remote model: its router's address, the preset
/// timeouts, keepalive, the remote router's key as Authorization and
/// `X-Gufo-Forwarded: <router_id>` (omitted when `router_id` is empty).
[[nodiscard]] UpstreamTarget RemoteTarget(const RemoteSpec& spec,
                                          const std::string& router_id = {});

/// Random router id for kForwardedHeader, fixed for the process lifetime.
[[nodiscard]] std::string NewRouterId();

/// 508 remote_loop: a request another router already relayed here names a
/// model that is remote on this router as well.
[[nodiscard]] server::HttpResponse RemoteLoop(const std::string& model_id,
                                              const std::string& forwarded_by);

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
  /// Model id -> llama.cpp status value ("loaded", "loading", "unloaded"),
  /// or "remote" for a model the remote router itself relays elsewhere.
  std::map<std::string, std::string> models;
};

/// Asks `url` for its model list with `key` as bearer token; connect, head
/// and the whole answer each bounded by `timeout`. Blocking: call it with
/// no router lock held.
[[nodiscard]] RemoteProbe ProbeRemoteModels(const RemoteEndpoint& url,
                                            const std::string& key,
                                            std::chrono::milliseconds timeout,
                                            const std::string& router_id = {});

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
  /// remote does not list or itself lists as remote (once per change, for a
  /// warning).
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

  /// Last answer of a busy-check url ("free", "busy", "unreachable: ...").
  void NoteBusyCheck(const RemoteEndpoint& url, const std::string& result);
  /// {"http://<host:port><path>": {result, checked_s_ago}}.
  [[nodiscard]] json::Value BusyCheckStatus() const;

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
  std::map<std::string, std::pair<std::string, Clock::time_point>>
      busy_checks_;
};

/// Remote GPU slots: per gpu-slot name, one GPU-starting request (or a
/// video/music job it started) at a time; the others wait in arrival order
/// up to their queue timeout. Thread-safe; a Hold outlives nothing it needs
/// (it shares the slot table).
class GpuSlots {
public:
  using Clock = std::chrono::steady_clock;

  GpuSlots();

  class Hold {
  public:
    ~Hold() { Release(); }
    Hold(const Hold&) = delete;
    Hold& operator=(const Hold&) = delete;
    /// Frees the slot (idempotent) and wakes the next waiter.
    void Release();
    /// What holds the slot, for /router/status ("job video_x on m").
    void Describe(const std::string& holder);
    [[nodiscard]] const std::string& slot() const { return slot_; }

  private:
    friend class GpuSlots;
    struct Table;
    Hold(std::shared_ptr<Table> table, std::string slot)
        : table_(std::move(table)), slot_(std::move(slot)) {}
    std::shared_ptr<Table> table_;
    std::string slot_;
    bool released_{false};
  };

  /// Waits until `slot` is free and takes it for `holder`. Returns nullptr
  /// when `wait` passed or `cancelled()` turned true first.
  [[nodiscard]] std::shared_ptr<Hold> Acquire(
      const std::string& slot, const std::string& holder,
      std::chrono::milliseconds wait, const std::function<bool()>& cancelled);

  /// {"<slot>": {held, holder, held_s, waiting}} for the slots ever used.
  [[nodiscard]] json::Value Status() const;

private:
  std::shared_ptr<Hold::Table> table_;
};

/// Whether a busy-check answer says the remote GPU is busy: a ComfyUI
/// /queue object with a non-empty queue_running or queue_pending. Anything
/// else (other JSON, not JSON) counts as free.
[[nodiscard]] bool BusyCheckSaysBusy(const std::string& body);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_
