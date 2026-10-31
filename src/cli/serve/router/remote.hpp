#ifndef GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_
#define GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_

// Remote models (`[remote/<id>]` preset sections): the allow-list check,
// and the shared state the router keeps about the remote routers.

#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/jobs.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/proxy.hpp"

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

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_REMOTE_HPP_
