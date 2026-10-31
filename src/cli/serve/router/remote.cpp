#include "src/cli/serve/router/remote.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
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

std::string EndpointKey(const RemoteEndpoint& endpoint) {
  return endpoint.host + ":" + std::to_string(endpoint.port);
}

RemoteProbe ProbeRemoteModels(const RemoteEndpoint& url,
                              const std::string& key,
                              std::chrono::milliseconds timeout) {
  RemoteSpec spec;
  spec.url = url;
  spec.api_key = key;
  UpstreamTarget target = RemoteTarget(spec);
  target.connect_timeout = timeout;
  target.head_timeout = timeout;
  server::HttpRequest request;
  request.method = "GET";
  request.path = "/v1/models";
  // The proxy polls the cancel check while it waits: a remote that sends
  // its head and then stalls ends here as well.
  const auto deadline = std::chrono::steady_clock::now() + 2 * timeout;
  request.is_cancelled = [deadline] {
    return std::chrono::steady_clock::now() >= deadline;
  };
  RemoteProbe probe;
  server::HttpResponse response = ProxyToWorker(target, request);
  if (response.status == 502 && !response.streaming_body) {
    probe.error = response.log_details;
    constexpr std::string_view kPrefix = "upstream_unavailable detail=";
    if (std::string_view(probe.error).starts_with(kPrefix)) {
      probe.error.erase(0, kPrefix.size());
    }
    return probe;
  }
  probe.reachable = true;
  std::string body = std::move(response.body);
  bool complete = true;
  if (response.streaming_body) {
    constexpr std::size_t kMaxListing = 4 * 1024 * 1024;
    response.streaming_body([&body, &complete](std::string_view chunk) {
      if (body.size() + chunk.size() > kMaxListing) {
        complete = false;
        return false;
      }
      body.append(chunk);
      return true;
    });
  }
  if (response.status != 200) {
    probe.error = "HTTP " + std::to_string(response.status) + " " +
                  response.reason + " from GET /v1/models";
    return probe;
  }
  try {
    const json::Value listing = json::parse(body);
    const json::Value* data = listing.find("data");
    if (data == nullptr || !data->is_array()) {
      throw std::runtime_error("no data array");
    }
    for (const json::Value& model : data->items()) {
      const json::Value* id = model.find("id");
      if (id == nullptr || !id->is_string()) {
        continue;
      }
      std::string status;
      if (const json::Value* state = model.find("status");
          state != nullptr && state->is_object()) {
        status = state->member_str("value");
      }
      if (status.empty()) {
        const json::Value* loaded = model.find("loaded");
        status = loaded != nullptr && loaded->as_bool()
                     ? "loaded"
                     : "unloaded";
      }
      probe.models[id->str()] = status;
    }
  } catch (const std::exception& error) {
    probe.error = std::string(complete ? "" : "truncated ") +
                  "model list did not parse: " + error.what();
  }
  return probe;
}

RemoteHub::RemoteHub(const std::vector<PresetModel>& presets) {
  for (const auto& preset : presets) {
    if (!preset.remote.has_value()) {
      continue;
    }
    RouterInfo& info = routers_[EndpointKey(preset.remote->url)];
    if (info.ids.empty()) {
      info.url = preset.remote->url;
      info.api_key = preset.remote->api_key;
    }
    info.ids.push_back(preset.id);
  }
}

std::vector<RemoteHub::Target> RemoteHub::Targets() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Target> targets;
  for (const auto& [key, info] : routers_) {
    (void)key;
    targets.push_back({info.url, info.api_key});
  }
  return targets;
}

std::vector<std::string> RemoteHub::Record(const RemoteEndpoint& url,
                                           RemoteProbe probe) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = routers_.find(EndpointKey(url));
  if (it == routers_.end()) {
    return {};
  }
  RouterInfo& info = it->second;
  info.checked = true;
  info.checked_at = Clock::now();
  info.reachable = probe.reachable;
  info.error = std::move(probe.error);
  if (!probe.reachable || !info.error.empty()) {
    // Keep the last known model states; they are only shown.
    return {};
  }
  info.models = std::move(probe.models);
  std::vector<std::string> missing;
  for (const auto& id : info.ids) {
    if (info.models.find(id) == info.models.end()) {
      missing.push_back(id);
    }
  }
  if (missing == info.missing) {
    return {};
  }
  info.missing = missing;
  return missing;
}

void RemoteHub::MarkUnreachable(const RemoteEndpoint& url,
                                const std::string& error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = routers_.find(EndpointKey(url));
  if (it == routers_.end()) {
    return;
  }
  it->second.checked = true;
  it->second.checked_at = Clock::now();
  it->second.reachable = false;
  it->second.error = error;
}

bool RemoteHub::Reachable(const RemoteEndpoint& url,
                          std::string* error) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = routers_.find(EndpointKey(url));
  if (it == routers_.end() || it->second.reachable) {
    return true;
  }
  *error = it->second.error;
  return false;
}

std::string RemoteHub::ModelStatus(const RemoteEndpoint& url,
                                   const std::string& id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = routers_.find(EndpointKey(url));
  if (it == routers_.end() || !it->second.reachable) {
    return "";
  }
  const auto model = it->second.models.find(id);
  return model == it->second.models.end() ? std::string() : model->second;
}

json::Value RemoteHub::Status() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto now = Clock::now();
  json::Value out = json::Value::object();
  for (const auto& [key, info] : routers_) {
    json::Value entry = json::Value::object();
    entry["url"] = "http://" + key;
    entry["checked"] = info.checked;
    entry["reachable"] = info.reachable;
    entry["error"] = info.error;
    if (info.checked) {
      entry["checked_s_ago"] = static_cast<long long>(
          std::chrono::duration_cast<std::chrono::seconds>(now -
                                                           info.checked_at)
              .count());
    }
    json::Value models = json::Value::object();
    for (const auto& id : info.ids) {
      const auto model = info.models.find(id);
      models[id] = model == info.models.end() ? std::string("missing")
                                              : model->second;
    }
    entry["models"] = std::move(models);
    out[key] = std::move(entry);
  }
  return out;
}

}  // namespace gufo::router
