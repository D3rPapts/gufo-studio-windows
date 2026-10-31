#include "src/cli/serve/router/remote.hpp"

#include <string>
#include <vector>

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

}  // namespace gufo::router
