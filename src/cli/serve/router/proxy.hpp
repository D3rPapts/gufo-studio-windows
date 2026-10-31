#ifndef GUFO_CLI_SERVE_ROUTER_PROXY_HPP_
#define GUFO_CLI_SERVE_ROUTER_PROXY_HPP_

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"

namespace gufo::router {

struct UpstreamTarget {
  std::string host{"127.0.0.1"};
  int port{0};
  /// The defaults fit a loopback worker. A remote router sets them:
  /// connect bound (0 = blocking connect), response-head deadline after the
  /// request is sent (0 = wait while the client stays), headers sent after
  /// the forwarded ones (e.g. its Authorization), and TCP keepalive so a
  /// pulled cable ends a silent wait.
  std::chrono::milliseconds connect_timeout{0};
  std::chrono::milliseconds head_timeout{0};
  std::vector<std::pair<std::string, std::string>> extra_headers;
  bool keepalive{false};
};

/// Forward one buffered front request to the worker and produce a front
/// response. The returned response streams the worker body through
/// `HttpResponse::streaming_body`, checking `request.is_cancelled` between
/// chunks. Connect failure / EOF before headers → 502 `upstream_unavailable`
/// JSON error. Hop-by-hop headers (Connection, Keep-Alive, Transfer-Encoding,
/// Upgrade) are stripped both ways; `Authorization`, `Host`, `Connection`
/// are not forwarded, and client headers named like an `extra_headers`
/// entry are replaced by it.
gufo::server::HttpResponse ProxyToWorker(
    const UpstreamTarget& target, const gufo::server::HttpRequest& request);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_PROXY_HPP_
