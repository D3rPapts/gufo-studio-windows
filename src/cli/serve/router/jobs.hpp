#ifndef GUFO_CLI_SERVE_ROUTER_JOBS_HPP_
#define GUFO_CLI_SERVE_ROUTER_JOBS_HPP_

// Job API for the router (`gufo router --jobs-dir DIR`): long requests
// (videos, large images, music, speech, chat) run in the background instead
// of on one open HTTP connection.
//
//   POST   /v1/jobs                    {"request": {"path": "/v1/videos",
//                                       "body": {...}}, "webhook_url": "...",
//                                       "metadata": {...}} -> 202 + the job
//                                      (Idempotency-Key: a repeated submit
//                                      returns the first job)
//   GET    /v1/jobs[?limit=N&status=S] newest first
//   GET    /v1/jobs/{id}               status, progress, result links
//   DELETE /v1/jobs/{id}               cancel (also POST /v1/jobs/{id}/cancel)
//   GET    /v1/jobs/{id}/files/{name}  one result file
//
// Jobs run through the router's own API, like the ComfyUI adapter's prompts:
// a /v1/videos job creates the worker's video job and polls it (progress,
// cancel = DELETE), an image job's progress is its live preview's step.
// Results land in DIR/<id>/ and DIR/jobs.json keeps every job across router
// restarts: queued jobs run again, a job that was running is failed.
//
// Webhooks: with "webhook_url" the router POSTs the finished job (completed,
// failed or cancelled) there, signed like Standard Webhooks (webhook-id,
// webhook-timestamp, webhook-signature) when --jobs-webhook-secret is set,
// with a timeout per attempt and growing waits between retries. Only plain
// http targets the router allows (--jobs-webhook-allow: host names, IPv4
// addresses or CIDR ranges, each with an optional :port, or *) are accepted,
// so a request cannot make the router call arbitrary hosts (SSRF). The
// router connects to the address it checked and follows no redirects.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <stop_token>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/core/json.hpp"

namespace gufo::router::jobs {

using comfy::Fetch;
using comfy::FetchResult;

/// One --jobs-webhook-allow entry.
struct AllowEntry {
  enum class Kind { kAny, kHost, kNetwork };
  Kind kind{Kind::kAny};
  /// kHost: the host name, lower case.
  std::string host;
  /// kNetwork: IPv4 network and mask, host byte order.
  std::uint32_t network{0};
  std::uint32_t mask{0};
  /// 0 = any port.
  int port{0};
};

/// Parses a comma separated allow list such as
/// "192.168.1.0/24,studio.local:8090,10.0.0.5,*". Empty text is an empty
/// list (webhooks off).
bool ParseAllowList(std::string_view text, std::vector<AllowEntry>* out,
                    std::string* error);

/// A webhook target: http only, no user info, no IPv6 literal.
struct WebhookUrl {
  /// Lower case host name or dotted IPv4 address.
  std::string host;
  int port{80};
  /// Path and query ("/" at least), without the fragment.
  std::string target;
};

bool ParseWebhookUrl(std::string_view url, WebhookUrl* out,
                     std::string* error);

/// Strict dotted-quad IPv4 (four decimal parts 0..255), host byte order.
bool ParseIpv4(std::string_view text, std::uint32_t* out);
std::string FormatIpv4(std::uint32_t address);

/// IPv4 addresses (host byte order) of a host name or address.
using Resolver =
    std::function<std::vector<std::uint32_t>(const std::string& host)>;
std::vector<std::uint32_t> ResolveIpv4(const std::string& host);

/// Whether the router may POST to `url`: the IPv4 address to connect to, or
/// nullopt with the reason in *error. An empty allow list allows nothing;
/// "*" allows any address except 0.0.0.0/8, link-local 169.254.0.0/16
/// (cloud metadata) and multicast/reserved 224.0.0.0/3, which only an
/// explicit entry opens.
std::optional<std::uint32_t> CheckWebhookTarget(
    const WebhookUrl& url, const std::vector<AllowEntry>& allow,
    const Resolver& resolve, std::string* error);

/// Standard Webhooks signature "v1,<base64 HMAC-SHA256>" of
/// "<id>.<timestamp>.<body>". A "whsec_<base64>" secret is decoded, any
/// other secret is used as raw bytes.
std::string SignWebhook(std::string_view secret, std::string_view id,
                        long long timestamp, std::string_view body);

/// Sends one webhook POST to `address`; returns the HTTP status, or 0 with
/// *error when no response arrived (connect failure, timeout).
using Deliver = std::function<int(std::uint32_t address, const WebhookUrl& url,
                                  const server::HttpRequest& request,
                                  std::chrono::milliseconds timeout,
                                  std::string* error)>;
/// Default Deliver: plain HTTP/1.1 through the router's upstream client.
int DeliverHttp(std::uint32_t address, const WebhookUrl& url,
                const server::HttpRequest& request,
                std::chrono::milliseconds timeout, std::string* error);

struct Options {
  /// Holds jobs.json and one folder per job.
  std::filesystem::path root;
  /// The router's own API (the same fetch the ComfyUI adapter uses).
  Fetch fetch;
  /// Preset model ids; a job naming another model is rejected at submit.
  /// Empty = no check.
  std::vector<std::string> models;
  /// Jobs that run at the same time (each holds one request to the router).
  std::size_t parallel{1};
  std::vector<AllowEntry> webhook_allow;
  std::string webhook_secret;
  /// Waits before the 2nd, 3rd, ... attempt; attempts = size + 1.
  std::vector<std::chrono::milliseconds> webhook_retry{
      std::chrono::seconds(5), std::chrono::seconds(30),
      std::chrono::seconds(120), std::chrono::seconds(600)};
  std::chrono::milliseconds webhook_timeout{std::chrono::seconds(10)};
  /// Empty = ResolveIpv4 / DeliverHttp.
  Resolver resolve;
  Deliver deliver;
  std::chrono::milliseconds video_poll{std::chrono::seconds(2)};
  /// Preview poll for the progress of image jobs.
  std::chrono::milliseconds progress_poll{std::chrono::seconds(1)};
};

class JobQueue {
public:
  explicit JobQueue(Options options);
  ~JobQueue();
  JobQueue(const JobQueue&) = delete;
  JobQueue& operator=(const JobQueue&) = delete;

  void Start();
  /// Stops the runners (a running job ends as failed "router_shutdown") and
  /// the webhook sender; undelivered webhooks are sent after a restart.
  void Stop();

  /// The answer for a /v1/jobs path, or nullopt for any other path.
  std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& request);

  struct Job;

private:
  server::HttpResponse Submit(const server::HttpRequest& request);
  server::HttpResponse List(const server::HttpRequest& request);
  server::HttpResponse Get(const std::string& id);
  server::HttpResponse Cancel(const std::string& id);
  server::HttpResponse File(const std::string& id, const std::string& name);

  void RunnerMain();
  void Execute(const std::shared_ptr<Job>& job);
  void RunVideo(const std::shared_ptr<Job>& job, const FetchResult& created);
  FetchResult Call(const std::shared_ptr<Job>& job, std::string method,
                   std::string path, std::string query, std::string body,
                   std::string content_type);
  void PollPreview(const std::shared_ptr<Job>& job, const std::string& model,
                   std::stop_token stop);
  json::Value StoreResult(Job& job, const FetchResult& result);
  void Finish(const std::shared_ptr<Job>& job, std::string status,
              json::Value result, json::Value error);
  void WebhookMain();
  void DeliverOne(const std::shared_ptr<Job>& job);

  json::Value ViewLocked(const Job& job) const;
  json::Value RecordLocked(const Job& job) const;
  void PersistLocked();
  void Load();
  std::filesystem::path JobDir(const std::string& id) const;

  Options options_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable webhook_wake_;
  /// Every job in submit order (the index of jobs.json).
  std::vector<std::shared_ptr<Job>> order_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::map<std::string, std::string> idempotency_;
  std::deque<std::shared_ptr<Job>> pending_;
  bool stopping_{false};
  std::vector<std::thread> runners_;
  std::thread webhook_thread_;
};

}  // namespace gufo::router::jobs

#endif  // GUFO_CLI_SERVE_ROUTER_JOBS_HPP_
