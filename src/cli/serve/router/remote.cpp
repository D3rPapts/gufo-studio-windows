#include "src/cli/serve/router/remote.hpp"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router {

bool RemoteEndpointAllowed(const RemoteEndpoint& endpoint,
                           const std::vector<jobs::AllowEntry>& allow) {
  for (const auto& entry : allow) {
    if (entry.port != 0 && entry.port != endpoint.port) {
      continue;
    }
    if (entry.kind == jobs::AllowEntry::Kind::kAny) {
      return true;
    }
    if (entry.kind == jobs::AllowEntry::Kind::kNetwork &&
        (endpoint.address & entry.mask) == entry.network) {
      return true;
    }
  }
  return false;
}

bool CheckRemoteAllowed(const RemoteSpec& spec,
                        const std::vector<jobs::AllowEntry>& allow,
                        std::string* error) {
  if (allow.empty()) {
    *error = "remote sections need --remote-allow (e.g. --remote-allow " +
             spec.url.host + ":" + std::to_string(spec.url.port) + ")";
    return false;
  }
  if (!RemoteEndpointAllowed(spec.url, allow)) {
    *error = "url http://" + spec.url.host + ":" +
             std::to_string(spec.url.port) + " is not in --remote-allow";
    return false;
  }
  if (spec.busy_check.has_value() &&
      !RemoteEndpointAllowed(*spec.busy_check, allow)) {
    *error = "busy-check-url http://" + spec.busy_check->host + ":" +
             std::to_string(spec.busy_check->port) +
             " is not in --remote-allow";
    return false;
  }
  return true;
}

UpstreamTarget RemoteTarget(const RemoteSpec& spec) {
  UpstreamTarget target;
  target.host = spec.url.host;
  target.port = spec.url.port;
  target.connect_timeout = std::chrono::seconds(spec.connect_timeout_seconds);
  target.head_timeout = std::chrono::seconds(spec.head_timeout_seconds);
  target.keepalive = true;
  if (!spec.api_key.empty()) {
    target.extra_headers.emplace_back("Authorization",
                                      "Bearer " + spec.api_key);
  }
  return target;
}

server::HttpResponse RemoteUnavailable(const std::string& model_id,
                                       const std::string& detail) {
  json::Value error = json::Value::object();
  error["message"] = "remote model '" + model_id +
                     "' is unreachable: " + detail;
  error["type"] = "server_error";
  error["code"] = "remote_unavailable";
  json::Value body = json::Value::object();
  body["error"] = std::move(error);
  server::HttpResponse response;
  response.status = 503;
  response.reason = "Service Unavailable";
  response.body = body.dump();
  response.headers = {{"Content-Type", "application/json"},
                      {"Retry-After", "30"}};
  response.log_details = "remote_unavailable model=" + model_id;
  return response;
}

}  // namespace gufo::router
