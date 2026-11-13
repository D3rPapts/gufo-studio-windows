// CPU tests for the router job API (src/cli/serve/router/jobs.*): webhook
// target checks (SSRF), Standard Webhooks signatures, the job life cycle
// against a fake router API and a fake webhook receiver, and the result
// store: keep flag, expiry, delete (single and bulk), list filters, music
// progress, the estimated progress and the video concat helpers (URL to
// job file, ffprobe parsing, copy vs re-encode; no ffmpeg runs here), the
// pool relay and the shared queue (capabilities, pull) against a fake
// partner.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/jobs.hpp"
#include "src/cli/serve/router/pool.hpp"
#include "src/core/json.hpp"

namespace json = gufo::json;
namespace jobs = gufo::router::jobs;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

namespace {

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

std::uint32_t Ip(const char* text) {
  std::uint32_t address = 0;
  if (!jobs::ParseIpv4(text, &address))
    std::abort();
  return address;
}

std::vector<jobs::AllowEntry> Allow(const char* text) {
  std::vector<jobs::AllowEntry> entries;
  std::string error;
  if (!jobs::ParseAllowList(text, &entries, &error)) {
    std::cerr << "allow list '" << text << "': " << error << "\n";
    std::abort();
  }
  return entries;
}

/// Fixed DNS: studio.lan -> 192.168.1.56, meta.evil -> 169.254.169.254,
/// rebind.evil -> 127.0.0.1, everything else unknown.
std::vector<std::uint32_t> FakeResolve(const std::string& host) {
  std::uint32_t literal = 0;
  if (jobs::ParseIpv4(host, &literal))
    return {literal};
  if (host == "studio.lan")
    return {Ip("192.168.1.56")};
  if (host == "meta.evil")
    return {Ip("169.254.169.254")};
  if (host == "rebind.evil")
    return {Ip("127.0.0.1")};
  return {};
}

bool Allowed(const char* url, const char* allow, std::string* error = nullptr) {
  jobs::WebhookUrl parsed;
  std::string why;
  if (!jobs::ParseWebhookUrl(url, &parsed, &why)) {
    if (error != nullptr)
      *error = why;
    return false;
  }
  const bool ok = jobs::CheckWebhookTarget(parsed, Allow(allow), FakeResolve,
                                           &why)
                      .has_value();
  if (error != nullptr)
    *error = why;
  return ok;
}

void TestParseWebhookUrl() {
  jobs::WebhookUrl url;
  std::string error;
  CHECK(jobs::ParseWebhookUrl("http://Studio.LAN:8090/hook?x=1#frag", &url,
                              &error));
  CHECK(url.host == "studio.lan");
  CHECK(url.port == 8090);
  CHECK(url.target == "/hook?x=1");
  CHECK(jobs::ParseWebhookUrl("http://10.0.0.5", &url, &error));
  CHECK(url.port == 80 && url.target == "/");
  CHECK(jobs::ParseWebhookUrl("http://10.0.0.5?a=b", &url, &error));
  CHECK(url.target == "/?a=b");
  CHECK(!jobs::ParseWebhookUrl("https://10.0.0.5/h", &url, &error));
  CHECK(error.find("https") != std::string::npos);
  CHECK(!jobs::ParseWebhookUrl("ftp://10.0.0.5/h", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("file:///etc/passwd", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("gopher://10.0.0.5", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://user:pw@10.0.0.5/", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://[::1]:80/", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://10.0.0.5/a b", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://10.0.0.5/a\r\nX-Evil: 1", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://10.0.0.5:0/", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://10.0.0.5:70000/", &url, &error));
  CHECK(!jobs::ParseWebhookUrl("http://:80/", &url, &error));
}

void TestAllowList() {
  std::vector<jobs::AllowEntry> entries;
  std::string error;
  CHECK(jobs::ParseAllowList("", &entries, &error) && entries.empty());
  CHECK(jobs::ParseAllowList(" 192.168.1.0/24 , studio.lan:8090,*,10.0.0.5",
                             &entries, &error));
  CHECK(entries.size() == 4);
  CHECK(entries[0].kind == jobs::AllowEntry::Kind::kNetwork);
  CHECK(entries[1].kind == jobs::AllowEntry::Kind::kHost &&
        entries[1].port == 8090);
  CHECK(entries[2].kind == jobs::AllowEntry::Kind::kAny);
  CHECK(!jobs::ParseAllowList("10.0.0.0/33", &entries, &error));
  CHECK(!jobs::ParseAllowList("10.0.0/8", &entries, &error));
  CHECK(!jobs::ParseAllowList("host:99999", &entries, &error));
  CHECK(!jobs::ParseAllowList("bad host", &entries, &error));
  std::uint32_t address = 0;
  CHECK(!jobs::ParseIpv4("1.2.3", &address));
  CHECK(!jobs::ParseIpv4("1.2.3.256", &address));
  CHECK(!jobs::ParseIpv4("0x7f.0.0.1", &address));
  CHECK(jobs::FormatIpv4(Ip("192.168.1.56")) == "192.168.1.56");
}

void TestCheckTarget() {
  std::string error;
  // No allow list: webhooks are off.
  CHECK(!Allowed("http://192.168.1.56:8090/h", "", &error));
  CHECK(error.find("--jobs-webhook-allow") != std::string::npos);
  CHECK(Allowed("http://192.168.1.56:8090/h", "192.168.1.0/24"));
  CHECK(!Allowed("http://192.168.2.1/h", "192.168.1.0/24"));
  CHECK(!Allowed("http://127.0.0.1:8081/v1/models", "192.168.1.0/24"));
  // Ports.
  CHECK(Allowed("http://192.168.1.56:8090/h", "192.168.1.56:8090"));
  CHECK(!Allowed("http://192.168.1.56:8081/h", "192.168.1.56:8090"));
  // Host names: named entries trust the name; ranges check the address.
  CHECK(Allowed("http://studio.lan:8090/h", "studio.lan"));
  CHECK(Allowed("http://studio.lan:8090/h", "192.168.1.0/24"));
  CHECK(!Allowed("http://rebind.evil/h", "192.168.1.0/24"));
  CHECK(!Allowed("http://unknown.host/h", "*", &error));
  CHECK(error.find("resolve") != std::string::npos);
  // "*" keeps link-local (cloud metadata) and multicast closed.
  CHECK(Allowed("http://192.168.1.56/h", "*"));
  CHECK(!Allowed("http://169.254.169.254/latest/meta-data", "*"));
  CHECK(!Allowed("http://meta.evil/h", "*"));
  CHECK(!Allowed("http://224.0.0.1/h", "*"));
  CHECK(!Allowed("http://0.0.0.0:8081/h", "*"));
  CHECK(Allowed("http://169.254.1.1/h", "169.254.0.0/16"));
  CHECK(!Allowed("https://192.168.1.56/h", "*"));
}

void TestSignature() {
  // Test vector of the Standard Webhooks reference libraries.
  CHECK(jobs::SignWebhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw",
                          "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330,
                          "{\"test\": 2432232314}") ==
        "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=");
  // A plain secret is used as raw bytes (python: hmac.new(b"geheim", ...)).
  CHECK(jobs::SignWebhook("geheim", "msg_1", 10, "{}") ==
        "v1,Y4iJN5MGxhxIrjYC/dpUVx3ulCjbHUy5F0jl/7k824k=");
}

// ------------------------------------------------------------ job queue

const std::string kPng = std::string("\x89PNG\r\n\x1a\n", 8) + "fake-png";

/// Fake router API: images, videos, speech; "SLOW" prompts wait until the
/// job is cancelled or `release` is set; "FAIL" prompts answer 500.
struct FakeApi {
  std::mutex mutex;
  std::vector<std::string> calls;
  std::atomic<bool> release{false};
  int video_polls{0};
  bool video_deleted{false};
  bool objekt_deleted = false;
  int objekt_polls = 0;
  /// 3D status polls answered 503 before the real answer (a reloading
  /// worker), and the X-Gufo-Job header of every 3D poll and cancel.
  int objekt_flaky = 0;
  std::vector<std::string> objekt_job_headers;
  int unreachable_calls{0};
  /// Music: the async job stays in_progress while set.
  std::atomic<bool> music_hold{false};
  bool music_async{false};
  int music_polls{0};
  /// X-Gufo-Pool of every POST, by image prompt / speech input.
  std::map<std::string, std::string> sides;
  /// X-Gufo-Priority of every POST, by prompt / input ("" = none).
  std::map<std::string, std::string> priorities;
  /// The prompts / inputs of every POST, in arrival order.
  std::vector<std::string> prompts;
  /// Image requests with "WAIT" in the prompt stay here (as the router's
  /// pool does while both machines are busy) until released or redirected
  /// (then 503 pool_redirected); the job ids come from X-Gufo-Job.
  std::set<std::string> waiting;
  std::set<std::string> redirected;
  int image_posts{0};

  std::string SideOf(const std::string& prompt) {
    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = sides.find(prompt);
    return it == sides.end() ? "(none)" : it->second;
  }

  /// What the router reports as Options::pool_states.
  std::map<std::string, std::string> PoolStates() {
    const std::lock_guard<std::mutex> lock(mutex);
    std::map<std::string, std::string> states;
    for (const std::string& id : waiting)
      states[id] = "wait";
    return states;
  }

  /// Options::redirect_waiting: only a waiting request can be redirected.
  bool Redirect(const std::string& id) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (!waiting.contains(id))
      return false;
    redirected.insert(id);
    return true;
  }

  bool Waiting(const std::string& id) {
    const std::lock_guard<std::mutex> lock(mutex);
    return waiting.contains(id);
  }

  jobs::FetchResult operator()(const HttpRequest& request) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      calls.push_back(request.method + " " + request.path +
                      (request.query.empty() ? "" : "?" + request.query));
      if (request.method == "POST") {
        try {
          const json::Value body = json::parse(request.body);
          const std::string key = body.member_str("prompt").empty()
                                      ? body.member_str("input")
                                      : body.member_str("prompt");
          if (!key.empty()) {
            sides[key] = request.header(jobs::kPoolSideHeader);
            priorities[key] = request.header(gufo::router::kPriorityHeader);
            prompts.push_back(key);
          }
        } catch (const std::exception&) {
        }
      }
    }
    jobs::FetchResult result;
    result.status = 200;
    result.content_type = "application/json";
    if (request.path == "/v1/images/generations") {
      const json::Value body = json::parse(request.body);
      const std::string prompt = body.member_str("prompt");
      {
        const std::lock_guard<std::mutex> lock(mutex);
        ++image_posts;
      }
      if (prompt.find("WAIT") != std::string::npos &&
          request.header(jobs::kPoolSideHeader) != "remote") {
        const std::string id = request.header(jobs::kJobHeader);
        {
          const std::lock_guard<std::mutex> lock(mutex);
          waiting.insert(id);
        }
        for (;;) {
          {
            const std::lock_guard<std::mutex> lock(mutex);
            if (redirected.contains(id)) {
              redirected.erase(id);
              waiting.erase(id);
              result.status = 503;
              result.body =
                  R"({"error":{"message":"handed to the partner","code":"pool_redirected"}})";
              return result;
            }
            if (release.load() ||
                (request.is_cancelled && request.is_cancelled())) {
              waiting.erase(id);
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
      }
      if (prompt.find("SLOW") != std::string::npos) {
        while (!release.load() &&
               !(request.is_cancelled && request.is_cancelled()))
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (prompt.find("UNREACHABLE") != std::string::npos) {
        // The router's 503 for a remote that went away (ONCE: only the
        // first call).
        const std::lock_guard<std::mutex> lock(mutex);
        if (++unreachable_calls == 1 ||
            prompt.find("ONCE") == std::string::npos) {
          result.status = 503;
          result.body =
              R"({"error":{"message":"remote model 'qwen-image' is )"
              R"(unreachable: connect","code":"remote_unavailable"}})";
          return result;
        }
      }
      if (prompt.find("FAIL") != std::string::npos) {
        result.status = 500;
        result.body = R"({"error":{"message":"fake failure","code":"boom"}})";
        return result;
      }
      json::Value item = json::Value::object();
      item["b64_json"] = gufo::router::comfy::Base64Encode(kPng);
      item["revised_prompt"] = prompt;
      json::Value data = json::Value::array();
      data.push_back(item);
      data.push_back(item);
      json::Value reply = json::Value::object();
      reply["created"] = 1;
      reply["data"] = std::move(data);
      result.body = reply.dump();
      return result;
    }
    if (request.path.starts_with("/v1/previews/")) {
      result.body = R"({"state":"running","seq":1,"step":5,"steps":20})";
      return result;
    }
    if (request.path == "/v1/audio/speech") {
      try {
        const std::string input = json::parse(request.body).member_str("input");
        if (input.find("SLOW") != std::string::npos) {
          while (!release.load() &&
                 !(request.is_cancelled && request.is_cancelled()))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
      } catch (const std::exception&) {
      }
      result.content_type = "audio/wav";
      result.body = "RIFFfake-wav";
      return result;
    }
    if (request.method == "POST" && request.path == "/v1/audio/music") {
      const json::Value body = json::parse(request.body);
      const json::Value* async_flag = body.find("async");
      if (async_flag == nullptr || !async_flag->as_bool()) {
        result.content_type = "audio/wav";
        result.body = "RIFFmusic";
        return result;
      }
      const std::lock_guard<std::mutex> lock(mutex);
      music_async = true;
      result.status = 202;
      result.body = R"({"id":"music_1","object":"audio.music","status":"queued","progress":0})";
      return result;
    }
    if (request.method == "GET" && request.path == "/v1/audio/music/music_1") {
      const std::lock_guard<std::mutex> lock(mutex);
      ++music_polls;
      result.body =
          music_hold.load()
              ? R"({"id":"music_1","status":"in_progress","progress":40})"
              : R"({"id":"music_1","status":"completed","progress":100,"duration":30})";
      return result;
    }
    if (request.path == "/v1/audio/music/music_1/content") {
      result.content_type = "audio/wav";
      result.body = "RIFFmusic";
      return result;
    }
    if (request.method == "POST" && request.path == "/v1/videos") {
      result.status = 202;
      result.body = R"({"id":"video_1","status":"queued","progress":0})";
      return result;
    }
    if (request.method == "DELETE" && request.path == "/v1/videos/video_1") {
      const std::lock_guard<std::mutex> lock(mutex);
      video_deleted = true;
      result.body = R"({"id":"video_1","deleted":true})";
      return result;
    }
    if (request.method == "GET" && request.path == "/v1/videos/video_1") {
      const std::lock_guard<std::mutex> lock(mutex);
      ++video_polls;
      // `release` here means "never finish" (for the cancel test).
      result.body =
          release.load() || video_polls < 3
              ? R"({"id":"video_1","status":"in_progress","progress":40})"
              : R"({"id":"video_1","status":"completed","progress":100})";
      return result;
    }
    if (request.path == "/v1/videos/video_1/content") {
      result.content_type = "video/mp4";
      result.body = "FAKEMP4";
      return result;
    }
    // gufo serve objekt (TripoSG): 202, two polls, then GLB + preview.
    if (request.method == "POST" && request.path == "/v1/3d/generations") {
      result.status = 202;
      result.body = R"({"id":"obj_1","object":"3d.generation","status":"queued","progress":0})";
      return result;
    }
    // The router can only route these by the model query (no body): a
    // missing query is the 1.4.23 bug ("no route for this path").
    if (request.path.starts_with("/v1/3d/generations/obj_1") &&
        request.query != "model=triposg") {
      result.status = 404;
      result.body = R"({"error":{"message":"no route for this path","code":"not_found"}})";
      return result;
    }
    if (request.path.starts_with("/v1/3d/generations/obj_1")) {
      const std::lock_guard<std::mutex> lock(mutex);
      objekt_job_headers.push_back(request.header(jobs::kJobHeader));
    }
    if (request.method == "DELETE" && request.path == "/v1/3d/generations/obj_1") {
      const std::lock_guard<std::mutex> lock(mutex);
      objekt_deleted = true;
      result.body = R"({"id":"obj_1","deleted":true})";
      return result;
    }
    if (request.method == "GET" && request.path == "/v1/3d/generations/obj_1") {
      const std::lock_guard<std::mutex> lock(mutex);
      if (objekt_flaky > 0) {
        --objekt_flaky;
        result.status = 503;
        result.body = R"({"error":{"message":"loading","code":"model_loading"}})";
        return result;
      }
      ++objekt_polls;
      result.body =
          release.load() || objekt_polls < 3
              ? R"({"id":"obj_1","status":"in_progress","progress":55,"phase":"decode"})"
              : R"({"id":"obj_1","status":"completed","progress":100,"phase":"done","data":[{"format":"glb","vertices":4,"faces":4,"closed":true}]})";
      return result;
    }
    if (request.path == "/v1/3d/generations/obj_1/content") {
      result.content_type = "model/gltf-binary";
      result.body = "glTFfake";
      return result;
    }
    if (request.path == "/v1/3d/generations/obj_1/preview") {
      result.content_type = "image/png";
      result.body = "PNGfake";
      return result;
    }
    result.status = 404;
    result.body = R"({"error":{"message":"no route"}})";
    return result;
  }
};

struct Delivery {
  std::uint32_t address;
  std::string target;
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
};

/// Fake webhook receiver: answers `statuses` in turn, then 200.
struct FakeReceiver {
  std::mutex mutex;
  std::vector<Delivery> deliveries;
  std::vector<int> statuses;

  int operator()(std::uint32_t address, const jobs::WebhookUrl& url,
                 const HttpRequest& request, std::chrono::milliseconds,
                 std::string* error) {
    const std::lock_guard<std::mutex> lock(mutex);
    deliveries.push_back({address, url.target, request.body, request.headers});
    const std::size_t n = deliveries.size();
    if (n <= statuses.size()) {
      if (statuses[n - 1] == 0)
        *error = "connect failed";
      return statuses[n - 1];
    }
    return 200;
  }
};

std::string Header(const Delivery& delivery, const std::string& name) {
  for (const auto& [key, value] : delivery.headers)
    if (key == name)
      return value;
  return "";
}

struct Harness {
  std::filesystem::path dir;
  std::shared_ptr<FakeApi> api = std::make_shared<FakeApi>();
  std::shared_ptr<FakeReceiver> receiver = std::make_shared<FakeReceiver>();
  std::unique_ptr<jobs::JobQueue> queue;
  std::chrono::seconds expire_after{std::chrono::hours(12)};
  /// Set before Open(true) to give the queue a (fake) pool partner.
  jobs::Options::Relay relay;
  std::string partner_name;
  /// Set before Open(true): capabilities (shared queue, LoRA routing).
  std::function<json::Value()> capabilities;
  std::function<std::optional<json::Value>()> partner_capabilities;
  std::string self_name;
  std::chrono::milliseconds pull_hold{std::chrono::seconds(60)};
  /// Set before Open(true): the router's GPU gate for pulls (empty = any).
  std::function<bool(const std::string&, std::string*)> may_pull;
  /// Set before Open(true): the 3D poll limit (zero keeps the default).
  std::chrono::milliseconds objekt_deadline{0};

  explicit Harness(const std::string& name, bool start = true,
                   bool keep_dir = false) {
    dir = std::filesystem::temp_directory_path() / ("gufo-jobs-test-" + name);
    if (!keep_dir)
      std::filesystem::remove_all(dir);
    Open(start);
  }

  void Open(bool start) {
    jobs::Options options;
    options.root = dir;
    auto api_ref = api;
    options.fetch = [api_ref](const HttpRequest& request) {
      return (*api_ref)(request);
    };
    options.models = {"qwen-image", "minimax-h3", "qwen3-tts", "ace-step",
                      "triposg"};
    options.expire_after = expire_after;
    std::string error;
    jobs::ParseAllowList("192.168.1.0/24", &options.webhook_allow, &error);
    options.webhook_secret = "geheim";
    options.webhook_retry = {std::chrono::milliseconds(20),
                             std::chrono::milliseconds(20)};
    options.resolve = FakeResolve;
    auto receiver_ref = receiver;
    options.deliver = [receiver_ref](std::uint32_t address,
                                     const jobs::WebhookUrl& url,
                                     const HttpRequest& request,
                                     std::chrono::milliseconds timeout,
                                     std::string* error) {
      return (*receiver_ref)(address, url, request, timeout, error);
    };
    options.video_poll = std::chrono::milliseconds(10);
    options.progress_poll = std::chrono::milliseconds(10);
    options.relay = relay;
    options.partner_name = partner_name;
    options.pool_states = [api_ref] { return api_ref->PoolStates(); };
    options.redirect_waiting = [api_ref](const std::string& id) {
      return api_ref->Redirect(id);
    };
    options.capabilities = capabilities;
    options.partner_capabilities = partner_capabilities;
    options.self_name = self_name;
    options.pull_poll = std::chrono::milliseconds(20);
    options.pull_hold = pull_hold;
    options.may_pull = may_pull;
    if (objekt_deadline.count() > 0)
      options.objekt_deadline = objekt_deadline;
    queue = std::make_unique<jobs::JobQueue>(std::move(options));
    if (start)
      queue->Start();
  }

  HttpResponse Call(const std::string& method, const std::string& path,
                    const std::string& body = "",
                    const std::string& idempotency = "") {
    HttpRequest request;
    request.method = method;
    const auto question = path.find('?');
    request.path = path.substr(0, question);
    if (question != std::string::npos)
      request.query = path.substr(question + 1);
    request.body = body;
    if (!idempotency.empty())
      request.headers.emplace_back("Idempotency-Key", idempotency);
    auto response = queue->Handle(request);
    if (!response) {
      HttpResponse none;
      none.status = 0;
      return none;
    }
    return *response;
  }

  json::Value Submit(const std::string& body, int expect = 202,
                     const std::string& idempotency = "") {
    const HttpResponse response = Call("POST", "/v1/jobs", body, idempotency);
    if (response.status != expect)
      std::cerr << "submit: HTTP " << response.status << " " << response.body
                << "\n";
    CHECK(response.status == expect);
    return json::parse(response.body);
  }

  json::Value WaitFor(const std::string& id, const std::string& status) {
    json::Value job;
    for (int i = 0; i < 400; ++i) {
      job = json::parse(Call("GET", "/v1/jobs/" + id).body);
      if (job.member_str("status") == status)
        return job;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cerr << "job " << id << " never reached " << status << ": "
              << job.dump() << "\n";
    ++failures;
    return job;
  }

  std::size_t Deliveries() {
    const std::lock_guard<std::mutex> lock(receiver->mutex);
    return receiver->deliveries.size();
  }

  void WaitDeliveries(std::size_t count) {
    for (int i = 0; i < 400 && Deliveries() < count; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
};

std::string ImageJob(const std::string& prompt,
                     const std::string& extra = "") {
  return R"({"request":{"path":"/v1/images/generations","body":{"model":"qwen-image","prompt":")" +
         prompt + R"(","size":"64x64"}})" + extra + "}";
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void TestImageLifecycle() {
  Harness h("image");
  const json::Value queued =
      h.Submit(ImageJob("a cat", R"(,"metadata":{"shot":"A1"})"));
  const std::string id = queued.member_str("id");
  CHECK(id.starts_with("job_"));
  CHECK(queued.member_str("status") == "queued" ||
        queued.member_str("status") == "in_progress");
  const json::Value done = h.WaitFor(id, "completed");
  CHECK(done.member_double("progress") == 100);
  CHECK(done.find("metadata")->member_str("shot") == "A1");
  const json::Value* result = done.find("result");
  CHECK(result != nullptr && result->is_object());
  const auto& files = result->find("files")->items();
  CHECK(files.size() == 3);  // two images + response.json
  CHECK(files[0].member_str("name") == "image-1.png");
  CHECK(files[0].member_str("url") ==
        "/v1/jobs/" + id + "/files/image-1.png");
  // The inline response carries urls instead of base64.
  const auto& data = result->find("response")->find("data")->items();
  CHECK(data.size() == 2);
  CHECK(!data[0].contains("b64_json"));
  CHECK(data[0].member_str("url") == "/v1/jobs/" + id + "/files/image-1.png");
  CHECK(data[0].member_str("revised_prompt") == "a cat");
  const HttpResponse file = h.Call("GET", "/v1/jobs/" + id + "/files/image-2.png");
  CHECK(file.status == 200 && file.body == kPng);
  CHECK(h.Call("GET", "/v1/jobs/" + id + "/files/request.body").status == 404);
  CHECK(h.Call("GET", "/v1/jobs/" + id + "/files/..%5Cjobs.json").status ==
        404);
  // The worker got a live preview id for the progress.
  {
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    bool preview_sent = false;
    for (const auto& call : h.api->calls)
      preview_sent = preview_sent || call == "POST /v1/images/generations";
    CHECK(preview_sent);
  }
  // Listing, newest first, with a status filter.
  const json::Value second = h.Submit(ImageJob("a dog"));
  h.WaitFor(second.member_str("id"), "completed");
  const json::Value list = json::parse(h.Call("GET", "/v1/jobs?limit=5").body);
  CHECK(list.find("data")->items().size() == 2);
  CHECK(list.find("data")->items()[0].member_str("id") ==
        second.member_str("id"));
  CHECK(json::parse(h.Call("GET", "/v1/jobs?status=failed").body)
            .find("data")
            ->items()
            .empty());
  CHECK(h.Call("GET", "/v1/jobs/job_0123456789abcdef").status == 404);
  CHECK(h.Call("GET", "/v1/jobs/../etc").status == 404);
  CHECK(h.Call("PUT", "/v1/jobs").status == 405);
  CHECK(h.Call("GET", "/v1/other").status == 0);
}

void TestFailedJob() {
  Harness h("failed");
  const json::Value job = h.Submit(ImageJob("FAIL please"));
  const json::Value done = h.WaitFor(job.member_str("id"), "failed");
  const json::Value* error = done.find("error");
  CHECK(error != nullptr && error->member_str("message") == "fake failure");
  CHECK(error->member_str("code") == "boom");
  CHECK(error->member_double("status_code") == 500);
}

void TestRequeueOnceWhenRemoteGone() {
  {
    Harness h("requeue-once");
    const json::Value job = h.Submit(ImageJob("UNREACHABLE ONCE"));
    h.WaitFor(job.member_str("id"), "completed");
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    CHECK(h.api->unreachable_calls == 2);
  }
  {
    Harness h("requeue-gives-up");
    const json::Value job = h.Submit(ImageJob("UNREACHABLE"));
    const json::Value done = h.WaitFor(job.member_str("id"), "failed");
    const json::Value* error = done.find("error");
    CHECK(error != nullptr &&
          error->member_str("code") == "remote_unavailable");
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    CHECK(h.api->unreachable_calls == 2);  // the first run and one requeue
  }
}

void TestBinaryResult() {
  Harness h("speech");
  const json::Value job = h.Submit(
      R"({"request":{"path":"/v1/audio/speech","body":{"model":"qwen3-tts","input":"Hallo"}}})");
  const json::Value done = h.WaitFor(job.member_str("id"), "completed");
  const auto& files = done.find("result")->find("files")->items();
  CHECK(files.size() == 1 && files[0].member_str("name") == "output.wav");
  CHECK(h.Call("GET", files[0].member_str("url")).body == "RIFFfake-wav");
}

void TestSubmitRejections() {
  Harness h("reject");
  h.Submit(R"({"request":{"path":"/v1/models","body":{"model":"qwen-image"}}})",
           400);
  h.Submit(R"({"request":{"path":"/v1/chat/completions","body":{"model":"qwen-image","stream":true}}})",
           400);
  h.Submit(R"({"request":{"path":"/v1/images/generations","body":{"prompt":"x"}}})",
           400);
  h.Submit(R"({"request":{"path":"/v1/images/generations","body":{"model":"nope"}}})",
           404);
  h.Submit(R"({"request":{"method":"GET","path":"/v1/videos","body":{"model":"minimax-h3"}}})",
           400);
  h.Submit("not json", 400);
  // Webhook targets outside the allow list never reach the queue.
  for (const char* url :
       {"http://127.0.0.1:8081/v1/models", "http://169.254.169.254/",
        "https://192.168.1.56/h", "ftp://192.168.1.56/h",
        "http://user@192.168.1.56/h", "http://rebind.evil/h"}) {
    const json::Value answer =
        h.Submit(ImageJob("x", std::string(R"(,"webhook_url":")") + url + "\""),
                 400);
    CHECK(answer.find("error")->member_str("code") == "webhook_not_allowed");
  }
  CHECK(json::parse(h.Call("GET", "/v1/jobs").body).find("data")->empty());
}

void TestCancel() {
  Harness h("cancel");
  const std::string running =
      h.Submit(ImageJob("SLOW one")).member_str("id");
  h.WaitFor(running, "in_progress");
  const std::string queued = h.Submit(ImageJob("SLOW two")).member_str("id");
  // A queued job is cancelled at once and never runs.
  const HttpResponse cancel_queued = h.Call("DELETE", "/v1/jobs/" + queued);
  CHECK(cancel_queued.status == 200);
  CHECK(json::parse(cancel_queued.body).member_str("status") == "cancelled");
  // A running job: the request to the worker is cancelled.
  const HttpResponse cancel_running =
      h.Call("POST", "/v1/jobs/" + running + "/cancel");
  CHECK(cancel_running.status == 200);
  h.WaitFor(running, "cancelled");
  // DELETE on a finished job removes it (no longer a 409).
  CHECK(h.Call("DELETE", "/v1/jobs/" + running).status == 200);
  CHECK(h.Call("GET", "/v1/jobs/" + running).status == 404);
  // The runner is free again.
  const std::string next = h.Submit(ImageJob("after")).member_str("id");
  h.WaitFor(next, "completed");
  const std::lock_guard<std::mutex> lock(h.api->mutex);
  int image_posts = 0;
  for (const auto& call : h.api->calls)
    image_posts += call == "POST /v1/images/generations";
  CHECK(image_posts == 2);  // "SLOW two" never ran
}

void TestVideo() {
  Harness h("video");
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/videos","body":{"model":"minimax-h3","prompt":"x"}}})")
          .member_str("id");
  const json::Value done = h.WaitFor(id, "completed");
  CHECK(done.member_str("video_id") == "video_1");
  CHECK(done.member_double("progress") == 100);
  const auto& files = done.find("result")->find("files")->items();
  CHECK(files[0].member_str("name") == "video.mp4");
  CHECK(h.Call("GET", files[0].member_str("url")).body == "FAKEMP4");
  CHECK(done.find("result")->find("response")->member_str("status") ==
        "completed");
}

// /v1/3d/generations (gufo serve objekt): 202 + polling like videos, result
// files model.glb (model/gltf-binary), preview.png and response.json.
void TestObjekt() {
  Harness h("objekt");
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/3d/generations","body":{"model":"triposg","image":"data:image/png;base64,AAAA","steps":50}}})")
          .member_str("id");
  const json::Value done = h.WaitFor(id, "completed");
  CHECK(done.member_double("progress") == 100);
  const auto& files = done.find("result")->find("files")->items();
  CHECK(files.size() == 3);
  CHECK(files[0].member_str("name") == "model.glb");
  CHECK(files[0].member_str("content_type") == "model/gltf-binary");
  CHECK(h.Call("GET", files[0].member_str("url")).body == "glTFfake");
  CHECK(files[1].member_str("name") == "preview.png");
  CHECK(h.Call("GET", files[1].member_str("url")).body == "PNGfake");
  CHECK(files[2].member_str("name") == "response.json");
  CHECK(done.find("result")->member_str("content_type") == "model/gltf-binary");
  const json::Value* response = done.find("result")->find("response");
  CHECK(response->member_str("status") == "completed");
  CHECK(response->find("data")->items()[0].member_double("faces") == 4);
}

void TestObjektCancel() {
  Harness h("objekt-cancel");
  h.api->release.store(true);  // the fake 3D job never completes
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/3d/generations","body":{"model":"triposg","image":"AAAA"}}})")
          .member_str("id");
  for (int i = 0; i < 400; ++i) {
    const json::Value job = json::parse(h.Call("GET", "/v1/jobs/" + id).body);
    if (job.member_double("progress", -1) == 55)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  h.Call("DELETE", "/v1/jobs/" + id);
  h.WaitFor(id, "cancelled");
  const std::lock_guard<std::mutex> lock(h.api->mutex);
  CHECK(h.api->objekt_deleted);
  // Polls and the cancel carry the job header like Call() (1.4.24, M2).
  CHECK(!h.api->objekt_job_headers.empty());
  for (const std::string& header : h.api->objekt_job_headers)
    CHECK(header == id);
}

// 1.4.24 (M1): a few 503 status answers (worker reloading) are tolerated.
void TestObjektTransient() {
  Harness h("objekt-transient", false);
  h.api->objekt_flaky = 3;
  h.queue->Start();
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/3d/generations","body":{"model":"triposg","image":"AAAA"}}})")
          .member_str("id");
  h.WaitFor(id, "completed");
}

// More 503 answers in a row than tolerated fail the job.
void TestObjektTransientGivesUp() {
  Harness h("objekt-giveup", false);
  h.api->objekt_flaky = 100;
  h.queue->Start();
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/3d/generations","body":{"model":"triposg","image":"AAAA"}}})")
          .member_str("id");
  const json::Value failed = h.WaitFor(id, "failed");
  CHECK(failed.find("error")->member_str("code") == "model_loading");
}

// 1.4.24 (M1): a worker stuck on in_progress fails the job at the deadline
// and gets the cancel.
void TestObjektDeadline() {
  Harness h("objekt-deadline", false);
  h.objekt_deadline = std::chrono::milliseconds(200);
  h.Open(true);
  h.api->release.store(true);  // never completes
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/3d/generations","body":{"model":"triposg","image":"AAAA"}}})")
          .member_str("id");
  const json::Value failed = h.WaitFor(id, "failed");
  CHECK(failed.find("error")->member_str("code") == "objekt_timeout");
  const std::lock_guard<std::mutex> lock(h.api->mutex);
  CHECK(h.api->objekt_deleted);
}

void TestVideoCancel() {
  Harness h("video-cancel");
  h.api->release.store(true);  // the fake video never completes
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/videos","body":{"model":"minimax-h3","prompt":"x"}}})")
          .member_str("id");
  for (int i = 0; i < 400; ++i) {
    const json::Value job = json::parse(h.Call("GET", "/v1/jobs/" + id).body);
    if (job.member_double("progress", -1) == 40)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const json::Value running = json::parse(h.Call("GET", "/v1/jobs/" + id).body);
  CHECK(running.member_str("preview_url") == "/v1/previews/video_1");
  h.Call("DELETE", "/v1/jobs/" + id);
  h.WaitFor(id, "cancelled");
  const std::lock_guard<std::mutex> lock(h.api->mutex);
  CHECK(h.api->video_deleted);
}

void TestWebhook() {
  Harness h("webhook");
  h.receiver->statuses = {500, 0};  // two failures, then 200
  const std::string id =
      h.Submit(ImageJob("a cat",
                        R"(,"webhook_url":"http://studio.lan:8090/hook?k=1")"))
          .member_str("id");
  h.WaitFor(id, "completed");
  h.WaitDeliveries(3);
  json::Value job;
  for (int i = 0; i < 200; ++i) {
    job = json::parse(h.Call("GET", "/v1/jobs/" + id).body);
    if (job.find("webhook")->member_str("state") == "delivered")
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(job.find("webhook")->member_str("state") == "delivered");
  CHECK(job.find("webhook")->member_double("attempts") == 3);
  const std::lock_guard<std::mutex> lock(h.receiver->mutex);
  CHECK(h.receiver->deliveries.size() == 3);
  const Delivery& last = h.receiver->deliveries.back();
  CHECK(last.address == Ip("192.168.1.56"));
  CHECK(last.target == "/hook?k=1");
  const json::Value payload = json::parse(last.body);
  CHECK(payload.member_str("type") == "job.completed");
  CHECK(payload.find("data")->member_str("id") == id);
  // Same message id on every attempt; signature over id.timestamp.body.
  const std::string message_id = Header(last, "webhook-id");
  CHECK(message_id == "msg_" + id + "_completed");
  CHECK(Header(h.receiver->deliveries.front(), "webhook-id") == message_id);
  const long long timestamp =
      std::strtoll(Header(last, "webhook-timestamp").c_str(), nullptr, 10);
  CHECK(Header(last, "webhook-signature") ==
        jobs::SignWebhook("geheim", message_id, timestamp, last.body));
}

void TestWebhookGivesUp() {
  Harness h("webhook-fail");
  h.receiver->statuses = {503, 503, 503, 503};
  const std::string id =
      h.Submit(ImageJob("FAIL", R"(,"webhook_url":"http://192.168.1.9/h")"))
          .member_str("id");
  h.WaitFor(id, "failed");
  h.WaitDeliveries(3);
  json::Value job;
  for (int i = 0; i < 200; ++i) {
    job = json::parse(h.Call("GET", "/v1/jobs/" + id).body);
    if (job.find("webhook")->member_str("state") == "failed")
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(job.find("webhook")->member_str("state") == "failed");
  CHECK(job.find("webhook")->member_double("attempts") == 3);  // 1 + 2 retries
  CHECK(job.find("webhook")->member_str("last_error") == "HTTP 503");
  const std::lock_guard<std::mutex> lock(h.receiver->mutex);
  CHECK(json::parse(h.receiver->deliveries[0].body).member_str("type") ==
        "job.failed");
}

void TestIdempotency() {
  Harness h("idem");
  const json::Value first = h.Submit(ImageJob("a cat"), 202, "key-1");
  const HttpResponse again =
      h.Call("POST", "/v1/jobs", ImageJob("a cat"), "key-1");
  CHECK(again.status == 200);
  CHECK(json::parse(again.body).member_str("id") == first.member_str("id"));
  CHECK(h.Call("POST", "/v1/jobs", ImageJob("a dog"), "key-1").status == 409);
  h.WaitFor(first.member_str("id"), "completed");
  CHECK(json::parse(h.Call("GET", "/v1/jobs").body).find("data")->size() == 1);
}

void TestRestart() {
  std::string queued_id;
  {
    // Not started: the job stays queued and is saved.
    Harness h("restart", /*start=*/false);
    queued_id = h.Submit(ImageJob("later"), 202, "key-r").member_str("id");
  }
  {
    Harness h("restart", /*start=*/true, /*keep_dir=*/true);
    h.WaitFor(queued_id, "completed");
    // The idempotency key survived the restart.
    CHECK(h.Call("POST", "/v1/jobs", ImageJob("later"), "key-r").status == 200);
  }
  // A job saved as running fails after the restart (and reports it).
  const auto dir =
      std::filesystem::temp_directory_path() / "gufo-jobs-test-restart";
  json::Value file = json::parse(ReadFile(dir / "jobs.json"));
  json::Value records = json::Value::array();
  for (const json::Value& record : file.find("jobs")->items()) {
    json::Value copy = record;
    copy["status"] = "in_progress";
    copy["webhook_url"] = "http://192.168.1.56/h";
    records.push_back(std::move(copy));
  }
  file["jobs"] = std::move(records);
  {
    std::ofstream out(dir / "jobs.json", std::ios::binary | std::ios::trunc);
    out << file.dump();
  }
  Harness h("restart", /*start=*/true, /*keep_dir=*/true);
  const json::Value job =
      json::parse(h.Call("GET", "/v1/jobs/" + queued_id).body);
  CHECK(job.member_str("status") == "failed");
  CHECK(job.find("error")->member_str("code") == "router_restarted");
  h.WaitDeliveries(1);
  CHECK(h.Deliveries() == 1);
}

long long Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string SpeechJob(const std::string& input, const std::string& extra = "") {
  return R"({"request":{"path":"/v1/audio/speech","body":{"model":"qwen3-tts","input":")" +
         input + R"("}})" + extra + "}";
}

std::size_t Count(Harness& h, const std::string& query) {
  const HttpResponse response = h.Call("GET", "/v1/jobs" + query);
  if (response.status != 200) {
    std::cerr << "GET /v1/jobs" << query << ": HTTP " << response.status
              << " " << response.body << "\n";
    ++failures;
    return 0;
  }
  return json::parse(response.body).find("data")->size();
}

void TestParseCutoff() {
  CHECK(jobs::ParseCutoff("3600", 10000) == 6400);
  CHECK(jobs::ParseCutoff(" 0 ", 10000) == 10000);
  CHECK(jobs::ParseCutoff("1794297600", 0) == 1794297600);
  CHECK(jobs::ParseCutoff("2026-11-10T08:00:00Z", 0) == 1794297600);
  CHECK(jobs::ParseCutoff("2026-11-10T08:00", 0) == 1794297600);
  CHECK(jobs::ParseCutoff("2026-11-10 08:00:00", 0) == 1794297600);
  CHECK(jobs::ParseCutoff("2026-11-10", 0) == 1794268800);
  CHECK(!jobs::ParseCutoff("", 0));
  CHECK(!jobs::ParseCutoff("gestern", 0));
  CHECK(!jobs::ParseCutoff("2026-13-01", 0));
  CHECK(!jobs::ParseCutoff("2026-11-10T08", 0));
  CHECK(!jobs::ParseCutoff("2026-11-10T08:00:00+01:00", 0));
  CHECK(!jobs::ParseCutoff("-5", 0));
  CHECK(jobs::KindOf("/v1/images/generations") == "image");
  CHECK(jobs::KindOf("/v1/images/edits") == "image");
  CHECK(jobs::KindOf("/v1/audio/music") == "music");
  CHECK(jobs::KindOf("/v1/audio/speech") == "speech");
  CHECK(jobs::KindOf("/v1/videos") == "video");
  CHECK(jobs::KindOf("/v1/chat/completions") == "other");
}

void TestKeepAndExpiry() {
  const long long later = Now() + 13 * 3600;
  std::string old_id;
  std::string kept_id;
  {
    Harness h("expiry");
    old_id = h.Submit(ImageJob("old one",
                               R"(,"metadata":{"titel":"Katze","prompt":"eine Katze","seite":"bild","geraet":"g1"})"))
                 .member_str("id");
    kept_id = h.Submit(ImageJob("kept one")).member_str("id");
    h.WaitFor(old_id, "completed");
    const json::Value fresh = h.WaitFor(kept_id, "completed");
    CHECK(fresh.find("keep") != nullptr && !fresh.find("keep")->as_bool());
    CHECK(fresh.find("expired_at")->is_null());
    CHECK(!fresh.find("progress_estimated")->as_bool());
    // PATCH keep, with the POST fallback for clients without PATCH.
    const HttpResponse patched =
        h.Call("PATCH", "/v1/jobs/" + kept_id, R"({"keep":true})");
    CHECK(patched.status == 200);
    CHECK(json::parse(patched.body).find("keep")->as_bool());
    CHECK(h.Call("PATCH", "/v1/jobs/" + kept_id, R"({"keep":"yes"})").status ==
          400);
    CHECK(h.Call("PATCH", "/v1/jobs/" + kept_id, "nope").status == 400);
    CHECK(h.Call("PATCH", "/v1/jobs/job_0123456789abcdef", R"({"keep":true})")
              .status == 404);
    CHECK(h.Call("POST", "/v1/jobs/" + kept_id + "/keep", R"({"keep":false})")
              .status == 200);
    CHECK(!json::parse(h.Call("GET", "/v1/jobs/" + kept_id).body)
               .find("keep")
               ->as_bool());
    // An empty body means keep.
    CHECK(json::parse(h.Call("POST", "/v1/jobs/" + kept_id + "/keep").body)
              .find("keep")
              ->as_bool());
    CHECK(h.Call("GET", "/v1/jobs/" + kept_id + "/keep").status == 405);
    // Too young: nothing expires.
    CHECK(h.queue->ExpireFiles(Now()) == 0);
    CHECK(h.Call("GET", "/v1/jobs/" + old_id + "/files/image-1.png").status ==
          200);
    // 13 hours later the old job's media go; the kept job stays whole.
    CHECK(h.queue->ExpireFiles(later) == 1);
    CHECK(h.queue->ExpireFiles(later) == 0);  // marked, not scanned again
    const json::Value expired =
        json::parse(h.Call("GET", "/v1/jobs/" + old_id).body);
    CHECK(expired.member_str("status") == "completed");
    CHECK(expired.member_double("expired_at") == static_cast<double>(later));
    CHECK(expired.find("metadata")->member_str("titel") == "Katze");
    CHECK(expired.find("metadata")->member_str("prompt") == "eine Katze");
    CHECK(expired.find("request")->member_str("model") == "qwen-image");
    const auto& files = expired.find("result")->find("files")->items();
    CHECK(files.size() == 1 && files[0].member_str("name") == "response.json");
    CHECK(h.Call("GET", "/v1/jobs/" + old_id + "/files/image-1.png").status ==
          404);
    CHECK(h.Call("GET", "/v1/jobs/" + old_id + "/files/response.json").status ==
          200);
    CHECK(std::filesystem::exists(h.dir / old_id / "request.body"));
    CHECK(!std::filesystem::exists(h.dir / old_id / "image-2.png"));
    const json::Value kept =
        json::parse(h.Call("GET", "/v1/jobs/" + kept_id).body);
    CHECK(kept.find("expired_at")->is_null());
    CHECK(kept.find("result")->find("files")->size() == 3);
    CHECK(h.Call("GET", "/v1/jobs/" + kept_id + "/files/image-1.png").status ==
          200);
    // Releasing the keep lets the next pass expire it.
    h.Call("PATCH", "/v1/jobs/" + kept_id, R"({"keep":false})");
    CHECK(h.queue->ExpireFiles(later) == 1);
    h.Call("PATCH", "/v1/jobs/" + kept_id, R"({"keep":true})");
  }
  {
    // keep and expired_at survive a restart.
    Harness h("expiry", /*start=*/true, /*keep_dir=*/true);
    const json::Value old_job =
        json::parse(h.Call("GET", "/v1/jobs/" + old_id).body);
    CHECK(old_job.member_double("expired_at") == static_cast<double>(later));
    CHECK(old_job.find("result")->find("files")->size() == 1);
    const json::Value kept =
        json::parse(h.Call("GET", "/v1/jobs/" + kept_id).body);
    CHECK(kept.find("keep")->as_bool());
    CHECK(h.queue->ExpireFiles(later + 3600) == 0);
  }
  {
    // Expiry off (--jobs-expire-hours 0): nothing ever goes.
    Harness h("expiry-never");
    const std::string id = h.Submit(ImageJob("stays")).member_str("id");
    h.WaitFor(id, "completed");
    h.queue.reset();
    h.expire_after = std::chrono::seconds(0);
    h.Open(/*start=*/false);
    CHECK(h.queue->ExpireFiles(later) == 0);
    CHECK(std::filesystem::exists(h.dir / id / "image-1.png"));
  }
}

void TestDeleteFinished() {
  Harness h("delete");
  const std::string id =
      h.Submit(ImageJob("gone soon"), 202, "key-d").member_str("id");
  h.WaitFor(id, "completed");
  CHECK(std::filesystem::exists(h.dir / id / "image-1.png"));
  const HttpResponse deleted = h.Call("DELETE", "/v1/jobs/" + id);
  CHECK(deleted.status == 200);
  const json::Value answer = json::parse(deleted.body);
  CHECK(answer.member_str("id") == id && answer.find("deleted")->as_bool());
  CHECK(h.Call("GET", "/v1/jobs/" + id).status == 404);
  CHECK(h.Call("DELETE", "/v1/jobs/" + id).status == 404);
  CHECK(!std::filesystem::exists(h.dir / id));
  CHECK(Count(h, "") == 0);
  // The idempotency key is free again: the same submit makes a new job.
  const HttpResponse again = h.Call("POST", "/v1/jobs", ImageJob("gone soon"), "key-d");
  CHECK(again.status == 202);
  CHECK(json::parse(again.body).member_str("id") != id);
  // A failed job can be deleted too; a kept one as well (explicit action).
  const std::string failed = h.Submit(ImageJob("FAIL")).member_str("id");
  h.WaitFor(failed, "failed");
  h.Call("PATCH", "/v1/jobs/" + failed, R"({"keep":true})");
  CHECK(h.Call("DELETE", "/v1/jobs/" + failed).status == 200);
  CHECK(h.Call("GET", "/v1/jobs/" + failed).status == 404);
}

void TestBulkDelete() {
  Harness h("bulk");
  const std::string a =
      h.Submit(ImageJob("a", R"(,"metadata":{"titel":"Katze","seite":"bild","geraet":"g1"})"))
          .member_str("id");
  const std::string b =
      h.Submit(SpeechJob("b", R"(,"metadata":{"titel":"Hund","seite":"stimme","geraet":"g2"})"))
          .member_str("id");
  const std::string c =
      h.Submit(ImageJob("c", R"(,"metadata":{"titel":"Maus","seite":"bild","geraet":"g1"})"))
          .member_str("id");
  h.WaitFor(a, "completed");
  h.WaitFor(b, "completed");
  h.WaitFor(c, "completed");
  h.Call("PATCH", "/v1/jobs/" + c, R"({"keep":true})");
  // No filter: refused. dry_run and include_kept are not filters.
  CHECK(h.Call("DELETE", "/v1/jobs").status == 400);
  CHECK(json::parse(h.Call("DELETE", "/v1/jobs?dry_run=1").body)
            .find("error")
            ->member_str("code") == "filter_required");
  CHECK(h.Call("DELETE", "/v1/jobs?include_kept=1").status == 400);
  CHECK(h.Call("DELETE", "/v1/jobs?older_than=gestern").status == 400);
  CHECK(h.Call("DELETE", "/v1/jobs?kind=nope").status == 400);
  // Dry run: counts and ids, nothing is deleted; kept jobs are skipped.
  json::Value dry = json::parse(
      h.Call("DELETE", "/v1/jobs?seite=bild&dry_run=1").body);
  CHECK(dry.find("dry_run")->as_bool());
  CHECK(dry.member_double("matched") == 1 && dry.member_double("deleted") == 0);
  CHECK(dry.find("ids")->size() == 1 &&
        dry.find("ids")->items()[0].str() == a);
  CHECK(Count(h, "") == 3);
  dry = json::parse(
      h.Call("DELETE", "/v1/jobs?seite=bild&dry_run=1&include_kept=1").body);
  CHECK(dry.member_double("matched") == 2);
  // Filters combine with AND.
  dry = json::parse(
      h.Call("DELETE", "/v1/jobs?seite=bild&geraet=g2&dry_run=1").body);
  CHECK(dry.member_double("matched") == 0);
  // q over the title: deletes a.
  json::Value real = json::parse(h.Call("DELETE", "/v1/jobs?q=KATZE").body);
  CHECK(!real.find("dry_run")->as_bool());
  CHECK(real.member_double("deleted") == 1);
  CHECK(real.find("ids")->items()[0].str() == a);
  CHECK(h.Call("GET", "/v1/jobs/" + a).status == 404);
  CHECK(!std::filesystem::exists(h.dir / a));
  // older_than in the future matches every finished job; c is kept.
  const std::string future = std::to_string(Now() + 1000);
  real = json::parse(h.Call("DELETE", "/v1/jobs?older_than=" + future).body);
  CHECK(real.member_double("deleted") == 1);
  CHECK(real.find("ids")->items()[0].str() == b);
  CHECK(Count(h, "") == 1);
  // An age in seconds: nothing finished more than an hour ago.
  real = json::parse(
      h.Call("DELETE", "/v1/jobs?older_than=3600&include_kept=1").body);
  CHECK(real.member_double("deleted") == 0);
  real = json::parse(
      h.Call("DELETE", "/v1/jobs?older_than=" + future + "&include_kept=1")
          .body);
  CHECK(real.member_double("deleted") == 1);
  CHECK(Count(h, "") == 0);
  // A running job is never bulk deleted.
  const std::string running = h.Submit(ImageJob("SLOW x")).member_str("id");
  h.WaitFor(running, "in_progress");
  real = json::parse(h.Call("DELETE", "/v1/jobs?older_than=" + future).body);
  CHECK(real.member_double("deleted") == 0);
  h.api->release.store(true);
  h.WaitFor(running, "completed");
}

void TestListFilters() {
  Harness h("filters");
  const long long before_all = Now();
  const std::string a =
      h.Submit(ImageJob("a", R"(,"metadata":{"titel":"Katze im Garten","prompt":"eine KATZE","seite":"bild","geraet":"g1"})"))
          .member_str("id");
  h.WaitFor(a, "completed");
  const std::string b =
      h.Submit(SpeechJob("Hallo", R"(,"metadata":{"titel":"Hallo","seite":"stimme","geraet":"g2"})"))
          .member_str("id");
  h.WaitFor(b, "completed");
  // Default order: newest first.
  const json::Value all = json::parse(h.Call("GET", "/v1/jobs").body);
  CHECK(all.find("data")->size() == 2);
  CHECK(all.find("data")->items()[0].member_str("id") == b);
  CHECK(Count(h, "?q=katze") == 1);
  CHECK(Count(h, "?q=GARTEN") == 1);
  CHECK(Count(h, "?q=qwen3") == 1);  // the model
  CHECK(Count(h, "?q=" + a.substr(0, 12)) == 1);  // the id
  CHECK(Count(h, "?q=nirgends") == 0);
  CHECK(Count(h, "?seite=stimme") == 1);
  CHECK(Count(h, "?seite=video") == 0);
  CHECK(Count(h, "?geraet=g1") == 1);
  CHECK(Count(h, "?kind=speech") == 1);
  CHECK(Count(h, "?kind=stimme") == 1);
  CHECK(Count(h, "?kind=image") == 1);
  CHECK(Count(h, "?kind=music") == 0);
  CHECK(h.Call("GET", "/v1/jobs?kind=nope").status == 400);
  CHECK(Count(h, "?since=" + std::to_string(before_all)) == 2);
  CHECK(Count(h, "?since=" + std::to_string(Now() + 100)) == 0);
  CHECK(Count(h, "?before=" + std::to_string(Now() + 100)) == 2);
  CHECK(Count(h, "?before=" + std::to_string(before_all)) == 0);
  CHECK(h.Call("GET", "/v1/jobs?since=heute").status == 400);
  // Filters combine with AND, with the old limit and status.
  CHECK(Count(h, "?status=completed&seite=bild") == 1);
  CHECK(Count(h, "?status=failed&seite=bild") == 0);
  CHECK(Count(h, "?kind=image&geraet=g2") == 0);
  CHECK(Count(h, "?limit=1&q=a") == 1);
  const json::Value one =
      json::parse(h.Call("GET", "/v1/jobs?geraet=g1").body);
  CHECK(one.find("data")->items()[0].member_str("id") == a);
}

void TestEstimatedProgress() {
  Harness h("estimate");
  // Nothing learned yet: a running speech job has no progress.
  h.api->release.store(false);
  const std::string first = h.Submit(SpeechJob("SLOW eins")).member_str("id");
  json::Value running = h.WaitFor(first, "in_progress");
  CHECK(running.find("progress")->is_null());
  CHECK(!running.find("progress_estimated")->as_bool());
  h.api->release.store(true);
  const json::Value done = h.WaitFor(first, "completed");
  CHECK(done.member_double("progress") == 100);
  CHECK(!done.find("progress_estimated")->as_bool());
  // The duration was learned and saved.
  const json::Value file = json::parse(ReadFile(h.dir / "jobs.json"));
  const json::Value* durations = file.find("durations");
  CHECK(durations != nullptr &&
        durations->member_double("/v1/audio/speech qwen3-tts") >= 1);
  // The next job of the same path and model reports an estimate.
  h.api->release.store(false);
  const std::string second = h.Submit(SpeechJob("SLOW zwei")).member_str("id");
  running = h.WaitFor(second, "in_progress");
  CHECK(running.find("progress_estimated")->as_bool());
  CHECK(running.member_double("progress", -1) >= 0);
  CHECK(running.member_double("progress", -1) <= 95);
  h.api->release.store(true);
  const json::Value second_done = h.WaitFor(second, "completed");
  CHECK(second_done.member_double("progress") == 100);
  CHECK(!second_done.find("progress_estimated")->as_bool());
  // Another model has no estimate.
  h.api->release.store(false);
  const std::string other =
      h.Submit(R"({"request":{"path":"/v1/audio/speech","body":{"model":"minimax-h3","input":"SLOW drei"}}})")
          .member_str("id");
  running = h.WaitFor(other, "in_progress");
  CHECK(!running.find("progress_estimated")->as_bool());
  h.api->release.store(true);
  h.WaitFor(other, "completed");
  // Image jobs read the live preview instead: never estimated.
  const std::string image = h.Submit(ImageJob("a cat")).member_str("id");
  const json::Value image_done = h.WaitFor(image, "completed");
  CHECK(!image_done.find("progress_estimated")->as_bool());
}

void TestMusic() {
  Harness h("music");
  h.api->music_hold.store(true);
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/audio/music","body":{"model":"ace-step","prompt":"x","duration":30}}})")
          .member_str("id");
  // Live progress from the worker's async music job.
  json::Value running;
  for (int i = 0; i < 400; ++i) {
    running = json::parse(h.Call("GET", "/v1/jobs/" + id).body);
    if (running.member_double("progress", -1) == 40)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(running.member_double("progress", -1) == 40);
  CHECK(!running.find("progress_estimated")->as_bool());
  CHECK(running.member_str("music_id") == "music_1");
  CHECK(!running.contains("video_id"));
  h.api->music_hold.store(false);
  const json::Value done = h.WaitFor(id, "completed");
  CHECK(done.member_double("progress") == 100);
  const auto& files = done.find("result")->find("files")->items();
  CHECK(files.size() == 2);
  CHECK(files[0].member_str("name") == "output.wav");
  CHECK(files[0].member_str("content_type") == "audio/wav");
  CHECK(h.Call("GET", files[0].member_str("url")).body == "RIFFmusic");
  CHECK(files[1].member_str("name") == "response.json");
  CHECK(done.find("result")->find("response")->member_str("status") ==
        "completed");
  const std::lock_guard<std::mutex> lock(h.api->mutex);
  CHECK(h.api->music_async);
  bool polled = false;
  bool content = false;
  for (const auto& call : h.api->calls) {
    polled = polled || call == "GET /v1/audio/music/music_1?model=ace-step";
    content = content ||
              call == "GET /v1/audio/music/music_1/content?model=ace-step";
  }
  CHECK(polled && content);
}

void TestConcatHelpers() {
  // URL -> job file: relative, absolute (any host), query/fragment dropped.
  auto ref = jobs::ParseJobFileUrl(
      "/v1/jobs/job_0123456789abcdef01234567/files/video.mp4");
  CHECK(ref && ref->id == "job_0123456789abcdef01234567" &&
        ref->name == "video.mp4");
  ref = jobs::ParseJobFileUrl(
      "http://192.168.178.198:8081/v1/jobs/job_abc/files/video.mp4?x=1");
  CHECK(ref && ref->id == "job_abc" && ref->name == "video.mp4");
  ref = jobs::ParseJobFileUrl("https://host/v1/jobs/job_abc/files/clip-2.mp4#t");
  CHECK(ref && ref->name == "clip-2.mp4");
  for (const char* bad :
       {"/v1/jobs/job_abc", "/v1/jobs/job_abc/files/",
        "/v1/jobs/job_abc/files/request.body",
        "/v1/jobs/job_abc/files/response.json",
        "/v1/jobs/job_abc/files/../x", "/v1/jobs/job_abc/files/Video.MP4",
        "/v1/jobs/nope/files/video.mp4", "/v1/jobs//files/video.mp4",
        "ftp://h/v1/jobs/job_abc/files/video.mp4", "/v1/videos/x",
        "http://host", "", "data:video/mp4;base64,AAAA"}) {
    if (jobs::ParseJobFileUrl(bad)) {
      std::cerr << "accepted bad job file url: " << bad << "\n";
      ++failures;
    }
  }
  // Data URLs: base64 only, media type without parameters.
  std::string type;
  auto data = jobs::ParseDataUrl("data:video/mp4;base64,RkFLRQ==", &type);
  CHECK(data && *data == "FAKE" && type == "video/mp4");
  data = jobs::ParseDataUrl("data:Video/MP4;charset=x;base64,RkFLRQ==", &type);
  CHECK(data && type == "video/mp4");
  CHECK(!jobs::ParseDataUrl("data:video/mp4,plain", &type));
  CHECK(!jobs::ParseDataUrl("data:video/mp4;base64,***", &type));
  CHECK(!jobs::ParseDataUrl("/v1/jobs/job_abc/files/video.mp4", &type));
  // Frame rates.
  CHECK(std::fabs(jobs::FrameRateValue("30000/1001") - 29.97) < 0.001);
  CHECK(jobs::FrameRateValue("25") == 25);
  CHECK(jobs::FrameRateValue("0/0") == 0);
  CHECK(jobs::FrameRateValue("abc") == 0);
  CHECK(jobs::FrameRateValue("24/") == 0);
  // ffprobe JSON (numbers as strings, as ffprobe prints them).
  const char* probe =
      R"({"streams":[{"codec_type":"video","codec_name":"h264","width":832,)"
      R"("height":480,"r_frame_rate":"24/1","avg_frame_rate":"24/1",)"
      R"("pix_fmt":"yuv420p","duration":"5.000000"},{"codec_type":"audio",)"
      R"("codec_name":"aac","sample_rate":"48000","channels":2}],)"
      R"("format":{"duration":"5.040000"}})";
  jobs::ProbeInfo a;
  std::string error;
  CHECK(jobs::ParseProbe(probe, &a, &error));
  CHECK(a.video_codec == "h264" && a.width == 832 && a.height == 480 &&
        a.frame_rate == "24/1" && a.pix_fmt == "yuv420p" && a.has_audio &&
        a.audio_codec == "aac" && a.sample_rate == 48000 && a.channels == 2 &&
        std::fabs(a.duration - 5.04) < 1e-6 &&
        std::fabs(a.video_duration - 5.0) < 1e-6);
  jobs::ProbeInfo silent;
  CHECK(jobs::ParseProbe(
      R"({"streams":[{"codec_type":"video","codec_name":"h264","width":832,)"
      R"("height":480,"r_frame_rate":"24/1","pix_fmt":"yuv420p",)"
      R"("duration":"3.5"}]})",
      &silent, &error));
  CHECK(!silent.has_audio && std::fabs(silent.duration - 3.5) < 1e-6);
  jobs::ProbeInfo none;
  CHECK(!jobs::ParseProbe(R"({"streams":[{"codec_type":"audio"}]})", &none,
                          &error) &&
        error == "no video stream");
  CHECK(!jobs::ParseProbe("not json", &none, &error));
  CHECK(!jobs::ParseProbe(
      R"({"streams":[{"codec_type":"video","width":0,"height":0}]})", &none,
      &error));
  // Same parameters -> copy; anything else, or reencode=true -> reencode.
  jobs::ProbeInfo b = a;
  CHECK(jobs::ConcatMode({a, b}, false) == "copy");
  CHECK(jobs::ConcatMode({a, b}, true) == "reencode");
  b.frame_rate = "48/2";
  CHECK(jobs::ConcatMode({a, b}, false) == "copy");
  b = a;
  b.width = 1280;
  CHECK(jobs::ConcatMode({a, b}, false) == "reencode");
  b = a;
  b.frame_rate = "25/1";
  CHECK(jobs::ConcatMode({a, b}, false) == "reencode");
  b = a;
  b.video_codec = "hevc";
  CHECK(jobs::ConcatMode({a, b}, false) == "reencode");
  b = a;
  b.pix_fmt = "yuv444p";
  CHECK(jobs::ConcatMode({a, b}, false) == "reencode");
  b = a;
  b.has_audio = false;
  CHECK(jobs::ConcatMode({a, b}, false) == "reencode");
  b = a;
  b.sample_rate = 44100;
  CHECK(jobs::ConcatMode({a, b}, false) == "reencode");
  b = a;
  b.channels = 1;
  CHECK(jobs::ConcatMode({a, a, b}, false) == "reencode");
  CHECK(jobs::ConcatMode({silent, silent}, false) == "copy");
  CHECK(jobs::ConcatMode({}, false) == "reencode");
}

void TestConcatEndpoint() {
  Harness h("concat");
  // The input list is checked before any ffmpeg call (none is configured).
  HttpResponse answer = h.Call("GET", "/v1/videos/concat");
  CHECK(answer.status == 405);
  answer = h.Call("POST", "/v1/videos/concat", "nope");
  CHECK(answer.status == 400);
  answer = h.Call("POST", "/v1/videos/concat",
                  R"({"videos":["/v1/jobs/job_abc/files/video.mp4"]})");
  CHECK(answer.status == 400 &&
        json::parse(answer.body).find("error")->member_str("code") ==
            "invalid_input");
  answer = h.Call("POST", "/v1/videos/concat",
                  R"({"videos":["/v1/jobs/job_abc/files/video.mp4",)"
                  R"("/v1/jobs/job_def/files/video.mp4"]})");
  CHECK(answer.status == 400 &&
        answer.body.find("not on this router") != std::string::npos);
  answer = h.Call("POST", "/v1/videos/concat",
                  R"({"videos":["/v1/jobs/job_abc/files/video.mp4",)"
                  R"("/v1/jobs/job_def/files/video.mp4"],"output_format":"webm"})");
  CHECK(answer.status == 400 &&
        answer.body.find("output_format") != std::string::npos);
  // A finished video job of this router resolves; then ffmpeg is missing.
  const std::string video =
      h.Submit(R"({"request":{"path":"/v1/videos","body":{"model":"minimax-h3","prompt":"x"}}})")
          .member_str("id");
  h.WaitFor(video, "completed");
  const std::string file = "/v1/jobs/" + video + "/files/video.mp4";
  answer = h.Call("POST", "/v1/videos/concat",
                  R"({"videos":[")" + file + R"(","http://127.0.0.1:8081)" +
                      file + R"("]})");
  CHECK(answer.status == 500 &&
        json::parse(answer.body).find("error")->member_str("code") ==
            "concat_failed" &&
        answer.body.find("ffmpeg") != std::string::npos);
  answer = h.Call("POST", "/v1/videos/concat",
                  R"({"videos":[")" + file + R"(","/v1/jobs/)" + video +
                      R"(/files/missing.mp4"]})");
  CHECK(answer.status == 400 &&
        answer.body.find("has no file") != std::string::npos);
  // As a job: no model, lands in the store as kind video, fails like the
  // direct call.
  const std::string id =
      h.Submit(R"({"request":{"path":"/v1/videos/concat","body":{"videos":[")" +
               file + R"(","/v1/jobs/job_abc/files/video.mp4"]}}})")
          .member_str("id");
  const json::Value done = h.WaitFor(id, "failed");
  const json::Value* error = done.find("error");
  CHECK(error != nullptr && error->member_str("code") == "invalid_input");
  CHECK(error->member_double("status_code") == 400);
  CHECK(done.find("request")->member_str("path") == "/v1/videos/concat");
  const json::Value listed =
      json::parse(h.Call("GET", "/v1/jobs?kind=video").body);
  bool found = false;
  for (const json::Value& job : listed.find("data")->items())
    found = found || job.member_str("id") == id;
  CHECK(found);
}

void TestThumb() {
  Harness h("thumb");
  const std::string id = h.Submit(ImageJob("a cat")).member_str("id");
  const json::Value done = h.WaitFor(id, "completed");
  CHECK(done.member_str("thumb_url") == "/v1/jobs/" + id + "/thumb");
  // The route exists; the harness has no ffmpeg, so a fresh thumb is 501.
  HttpResponse answer = h.Call("GET", "/v1/jobs/" + id + "/thumb?w=128");
  CHECK(answer.status == 501 &&
        json::parse(answer.body).find("error")->member_str("code") ==
            "thumb_unavailable");
  answer = h.Call("GET", "/v1/jobs/" + id + "/thumb?file=image-2.png");
  CHECK(answer.status == 501);
  answer = h.Call("GET", "/v1/jobs/" + id + "/thumb?file=missing.png");
  CHECK(answer.status == 404 &&
        json::parse(answer.body).find("error")->member_str("code") ==
            "file_not_found");
  CHECK(h.Call("GET", "/v1/jobs/" + id + "/thumb?file=response.json").status ==
        404);
  CHECK(h.Call("POST", "/v1/jobs/" + id + "/thumb").status == 405);
  answer = h.Call("GET", "/v1/jobs/job_0123456789abcdef01234567/thumb");
  CHECK(answer.status == 404 &&
        json::parse(answer.body).find("error")->member_str("code") ==
            "job_not_found");
  // A cached thumb next to the job's files is served without ffmpeg.
  const std::string bytes = "\xFF\xD8\xFF\xE0 fake jpeg";
  {
    std::ofstream out(h.dir / id / "thumb-128.jpg", std::ios::binary);
    out << bytes;
  }
  answer = h.Call("GET", "/v1/jobs/" + id + "/thumb?w=128");
  CHECK(answer.status == 200 && answer.body == bytes);
  bool jpeg = false;
  bool cacheable = false;
  for (const auto& [name, value] : answer.headers) {
    jpeg = jpeg || (name == "Content-Type" && value == "image/jpeg");
    cacheable = cacheable || (name == "Cache-Control" &&
                              value == "public, max-age=86400");
  }
  CHECK(jpeg && cacheable);
  // Another width is another cache file: still 501 here.
  CHECK(h.Call("GET", "/v1/jobs/" + id + "/thumb").status == 501);
  // A job without media has no thumb_url.
  const std::string chat =
      h.Submit(R"({"request":{"path":"/v1/chat/completions","body":{"model":"qwen-image","messages":[]}}})")
          .member_str("id");
  const json::Value chat_done = json::parse(h.Call("GET", "/v1/jobs/" + chat).body);
  CHECK(chat_done.find("thumb_url") != nullptr &&
        chat_done.find("thumb_url")->is_null());
}

void TestPoolRelay() {
  // Without a partner: scope=pool is the local list with partner "none",
  // and an unknown id stays 404.
  {
    Harness h("pool-none");
    const std::string id = h.Submit(ImageJob("a cat")).member_str("id");
    h.WaitFor(id, "completed");
    const json::Value listed =
        json::parse(h.Call("GET", "/v1/jobs?scope=pool").body);
    CHECK(listed.member_str("partner") == "none");
    CHECK(listed.find("data")->items().size() == 1);
    CHECK(listed.find("data")->items()[0].member_str("owner") == "local");
    CHECK(listed.find("data")->items()[0].member_str("id") == id);
    const std::string other = "job_0123456789abcdef01234567";
    HttpResponse answer = h.Call("GET", "/v1/jobs/" + other);
    CHECK(answer.status == 404 &&
          json::parse(answer.body).find("error")->member_str("code") ==
              "job_not_found");
    CHECK(h.Call("GET", "/v1/jobs/" + other + "/files/video.mp4").status == 404);
    CHECK(h.Call("GET", "/v1/jobs/" + other + "/thumb").status == 404);
    CHECK(h.Call("DELETE", "/v1/jobs/" + other).status == 404);
  }
  // With a (fake) partner: its jobs merge in newest first with its address
  // as owner, unknown ids go to it, relayed requests never go on.
  {
    const std::string partner_id = "job_aaaaaaaaaaaaaaaaaaaaaaaa";
    const std::string gone_id = "job_bbbbbbbbbbbbbbbbbbbbbbbb";
    auto seen = std::make_shared<std::vector<HttpRequest>>();
    Harness h("pool-partner", false);
    h.partner_name = "http://10.0.0.2:8081";
    h.relay = [seen, partner_id, gone_id](
                  const HttpRequest& request, std::chrono::milliseconds,
                  std::chrono::milliseconds,
                  std::string* error) -> std::optional<HttpResponse> {
      seen->push_back(request);
      const auto json_answer = [](int status, const std::string& body) {
        HttpResponse response;
        response.status = status;
        response.body = body;
        // What the partner's HTTP layer adds; must not come through twice.
        response.headers = {{"Content-Type", "application/json"},
                            {"Access-Control-Allow-Origin", "*"},
                            {"X-Request-ID", "r99"}};
        return response;
      };
      if (request.path == "/v1/jobs")
        return json_answer(
            200, R"({"object":"list","data":[{"id":")" + partner_id +
                     R"(","status":"completed","created_at":4102444800,)"
                     R"("thumb_url":"/v1/jobs/)" + partner_id + R"(/thumb"}]})");
      if (request.path == "/v1/jobs/" + partner_id ||
          request.path == "/v1/jobs/" + partner_id + "/keep")
        return json_answer(200, R"({"id":")" + partner_id + R"(","keep":true})");
      if (request.path == "/v1/jobs/" + partner_id + "/thumb") {
        HttpResponse response;
        response.status = 200;
        response.headers = {{"Content-Type", "image/jpeg"}};
        // Streamed like the proxy does it.
        response.streaming_body = [](const HttpResponse::BodyWriter& write) {
          write("\xFF\xD8");
          write("partner jpeg");
        };
        return response;
      }
      if (request.path.starts_with("/v1/jobs/" + gone_id)) {
        *error = "connect refused";
        return std::nullopt;
      }
      return json_answer(404, R"({"error":{"code":"job_not_found"}})");
    };
    h.Open(true);
    const std::string id = h.Submit(ImageJob("a dog")).member_str("id");
    h.WaitFor(id, "completed");
    const json::Value listed =
        json::parse(h.Call("GET", "/v1/jobs?scope=pool&limit=10").body);
    CHECK(listed.member_str("partner") == "http://10.0.0.2:8081");
    CHECK(listed.member_str("scope") == "pool");
    const auto& data = listed.find("data")->items();
    CHECK(data.size() == 2);
    CHECK(data[0].member_str("id") == partner_id);
    CHECK(data[0].member_str("owner") == "http://10.0.0.2:8081");
    CHECK(data[1].member_str("id") == id);
    CHECK(data[1].member_str("owner") == "local");
    CHECK(seen->size() == 1 && seen->back().path == "/v1/jobs" &&
          seen->back().query == "scope=pool&limit=10" &&
          seen->back().header(jobs::kRelayHeader) == "1");
    // The plain list stays local and unmarked.
    const json::Value plain = json::parse(h.Call("GET", "/v1/jobs").body);
    CHECK(plain.find("partner") == nullptr &&
          plain.find("data")->items().size() == 1 &&
          plain.find("data")->items()[0].find("owner") == nullptr);
    CHECK(seen->size() == 1);
    // A relayed pool request (from the partner) is answered locally.
    {
      HttpRequest request;
      request.method = "GET";
      request.path = "/v1/jobs";
      request.query = "scope=pool";
      request.headers.emplace_back(jobs::kRelayHeader, "1");
      const auto answer = h.queue->Handle(request);
      CHECK(answer && answer->status == 200);
      CHECK(json::parse(answer->body).find("partner") == nullptr);
      CHECK(seen->size() == 1);
    }
    // Unknown ids go to the partner; a local id never does.
    HttpResponse answer = h.Call("GET", "/v1/jobs/" + partner_id);
    CHECK(answer.status == 200 &&
          json::parse(answer.body).member_str("id") == partner_id);
    CHECK(seen->size() == 2 && seen->back().method == "GET");
    for (const auto& [name, value] : answer.headers)
      CHECK(name != "Access-Control-Allow-Origin" && name != "X-Request-ID");
    // Local jobs stay local with a partner configured: the list, the job,
    // its files and its thumb never go to the partner.
    CHECK(h.Call("GET", "/v1/jobs/" + id + "/files/image-1.png").status == 200);
    answer = h.Call("GET", "/v1/jobs/" + id + "/thumb");
    CHECK(answer.status == 501 &&
          json::parse(answer.body).find("error")->member_str("code") ==
              "thumb_unavailable");
    {
      std::ofstream out(h.dir / id / "thumb-256.jpg", std::ios::binary);
      out << "local jpeg";
    }
    answer = h.Call("GET", "/v1/jobs/" + id + "/thumb");
    CHECK(answer.status == 200 && answer.body == "local jpeg");
    CHECK(seen->size() == 2);
    CHECK(h.Call("POST", "/v1/jobs/" + partner_id + "/keep").status == 200);
    CHECK(seen->size() == 3 && seen->back().method == "POST" &&
          seen->back().path == "/v1/jobs/" + partner_id + "/keep");
    CHECK(h.Call("GET", "/v1/jobs/" + id).status == 200);
    CHECK(seen->size() == 3);
    CHECK(h.Call("GET", "/v1/jobs/" + partner_id + "/nothing").status == 404);
    CHECK(seen->size() == 3);
    answer = h.Call("GET", "/v1/jobs/" + gone_id);
    CHECK(answer.status == 502 &&
          json::parse(answer.body).find("error")->member_str("code") ==
              "partner_unreachable");
    // A relayed request for an unknown id is not relayed on: 404.
    {
      HttpRequest request;
      request.method = "GET";
      request.path = "/v1/jobs/" + partner_id;
      request.headers.emplace_back("X-Gufo-Forwarded", "other-router");
      const auto loop = h.queue->Handle(request);
      CHECK(loop && loop->status == 404);
    }
    // A partner thumb is streamed through once, then served from the
    // relay cache.
    const std::size_t before = seen->size();
    answer = h.Call("GET", "/v1/jobs/" + partner_id + "/thumb");
    CHECK(answer.status == 200 && answer.body == "\xFF\xD8partner jpeg");
    CHECK(seen->size() == before + 1);
    const std::filesystem::path cached =
        h.dir / "relay-cache" / partner_id / "thumb-256.jpg";
    CHECK(std::filesystem::is_regular_file(cached));
    answer = h.Call("GET", "/v1/jobs/" + partner_id + "/thumb");
    CHECK(answer.status == 200 && answer.body == "\xFF\xD8partner jpeg");
    CHECK(seen->size() == before + 1);
    bool relay_cached = false;
    for (const auto& [name, value] : answer.headers)
      relay_cached = relay_cached ||
                     (name == "X-Gufo-Thumb" && value == "relay-cached");
    CHECK(relay_cached);
    // Untouched for a day, the cache folder goes with the expiry pass.
    std::filesystem::last_write_time(
        cached, std::filesystem::file_time_type::clock::now() -
                    std::chrono::hours(25));
    h.queue->ExpireRelayCache();
    CHECK(!std::filesystem::exists(cached.parent_path()));
  }
}

// ---------------------------------------------------------- shared queue

/// An image job with one LoRA.
std::string LoraImageJob(const std::string& prompt, const std::string& lora) {
  return R"({"request":{"path":"/v1/images/generations","body":{"model":"qwen-image","prompt":")" +
         prompt + R"(","size":"64x64","loras":[{"name":")" + lora +
         R"(","scale":0.8}]}}})";
}

/// Capabilities of router A (here): qwen-image with the anime LoRA in the
/// pool, minimax-h3 in the pool, qwen3-tts local, ace-step placed remote.
json::Value CapsA() {
  return json::parse(
      R"({"router_id":"ra","partner":"http://10.0.0.2:8081","models":{)"
      R"("qwen-image":{"placement":"pool","loras":["anime"]},)"
      R"("minimax-h3":{"placement":"pool","loras":[]},)"
      R"("qwen3-tts":{"placement":"local","loras":[]},)"
      R"("ace-step":{"placement":"remote","loras":[]}}})");
}

/// Capabilities of router B (the partner): qwen-image with the winkel
/// LoRA, qwen3-tts placed remote (= here), ace-step local there only.
json::Value CapsB() {
  return json::parse(
      R"({"router_id":"rb","partner":"http://10.0.0.1:8081","models":{)"
      R"("qwen-image":{"placement":"pool","loras":["winkel"]},)"
      R"("minimax-h3":{"placement":"pool","loras":[]},)"
      R"("qwen3-tts":{"placement":"remote","loras":[]},)"
      R"("ace-step":{"placement":"local","loras":[]}}})");
}

HttpResponse JsonAnswer(int status, const std::string& body) {
  HttpResponse response;
  response.status = status;
  response.body = body;
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

std::string ErrorCodeOf(const HttpResponse& response) {
  try {
    return json::parse(response.body).find("error")->member_str("code");
  } catch (const std::exception&) {
    return "";
  }
}

std::string ErrorMessageOf(const HttpResponse& response) {
  try {
    return json::parse(response.body).find("error")->member_str("message");
  } catch (const std::exception&) {
    return "";
  }
}

void TestLoraNamesAndCanRun() {
  json::Value body = json::parse(
      R"({"loras":[{"name":"winkel","scale":1},{"name":"anime"},"plain",{"name":"winkel"}],)"
      R"("gufo":{"lora":"turbo"}})");
  const auto names = jobs::LoraNames(body);
  CHECK(names.size() == 4);
  CHECK(names[0] == "winkel" && names[1] == "anime" && names[2] == "plain" &&
        names[3] == "turbo");
  CHECK(jobs::LoraNames(json::parse(R"({"prompt":"x"})")).empty());
  std::string why;
  CHECK(jobs::CanRun(CapsA(), "qwen-image", {"anime"}, &why));
  CHECK(jobs::CanRun(CapsA(), "qwen-image", {}, &why));
  CHECK(!jobs::CanRun(CapsA(), "qwen-image", {"winkel"}, &why));
  CHECK(why.find("no LoRA 'winkel'") != std::string::npos &&
        why.find("registered: anime") != std::string::npos);
  CHECK(!jobs::CanRun(CapsA(), "ace-step", {}, &why));
  CHECK(why.find("placed remote") != std::string::npos);
  CHECK(!jobs::CanRun(CapsA(), "nope", {}, &why));
  CHECK(why.find("does not serve model 'nope'") != std::string::npos);
  CHECK(!jobs::CanRun(CapsB(), "minimax-h3", {"turbo"}, &why));
  CHECK(why.find("none registered") != std::string::npos);
  // The "models" object alone works too.
  CHECK(jobs::CanRun(*CapsB().find("models"), "qwen-image", {"winkel"}));
  // "available": false (a load failure holds): capable, but not now.
  const json::Value failed = json::parse(
      R"({"models":{"qwen-image":{"placement":"pool","loras":["anime"],)"
      R"("available":false,"unavailable":"local load failed"},)"
      R"("minimax-h3":{"placement":"pool","loras":[],"available":true}}})");
  CHECK(jobs::CanRun(failed, "qwen-image", {"anime"}, &why));
  CHECK(!jobs::CanRun(failed, "qwen-image", {"anime"}, &why, /*now=*/true));
  CHECK(why.find("cannot load model 'qwen-image' now (local load failed)") !=
        std::string::npos);
  CHECK(jobs::CanRun(failed, "minimax-h3", {}, &why, /*now=*/true));
  // Older routers send no field: may run.
  CHECK(jobs::CanRun(CapsA(), "qwen-image", {"anime"}, &why, /*now=*/true));
}

void TestCapabilityRouting() {
  // A job needing a LoRA only the partner has goes there (X-Gufo-Pool:
  // remote), one only this router has stays here (local), one both can
  // run carries no side; a job neither can run is refused at submit.
  Harness h("caps", false);
  h.partner_name = "http://10.0.0.2:8081";
  h.self_name = "http://10.0.0.1:8081";
  h.capabilities = CapsA;
  h.partner_capabilities = [] { return std::optional<json::Value>(CapsB()); };
  h.relay = [](const HttpRequest&, std::chrono::milliseconds,
               std::chrono::milliseconds,
               std::string*) -> std::optional<HttpResponse> {
    // An old partner: no queue to pull from, no pull route.
    return JsonAnswer(404, R"({"error":{"code":"not_found"}})");
  };
  h.Open(true);
  const std::string remote =
      h.Submit(LoraImageJob("needs winkel", "winkel")).member_str("id");
  const json::Value done = h.WaitFor(remote, "completed");
  CHECK(h.api->SideOf("needs winkel") == "remote");
  CHECK(done.find("request")->find("loras")->items().size() == 1 &&
        done.find("request")->find("loras")->items()[0].str() == "winkel");
  CHECK(done.find("running_on") != nullptr &&
        done.find("running_on")->is_null());
  const std::string local =
      h.Submit(LoraImageJob("needs anime", "anime")).member_str("id");
  h.WaitFor(local, "completed");
  CHECK(h.api->SideOf("needs anime") == "local");
  const std::string any = h.Submit(ImageJob("plain")).member_str("id");
  h.WaitFor(any, "completed");
  CHECK(h.api->SideOf("plain").empty());
  // qwen3-tts is placed remote on the partner: only here.
  const std::string speech =
      h.Submit(R"({"request":{"path":"/v1/audio/speech","body":{"model":"qwen3-tts","input":"hallo"}}})")
          .member_str("id");
  h.WaitFor(speech, "completed");
  CHECK(h.api->SideOf("hallo") == "local");
  // ace-step is remote here and local there: only the partner.
  const std::string music =
      h.Submit(R"({"request":{"path":"/v1/audio/music","body":{"model":"ace-step","prompt":"lied"}}})")
          .member_str("id");
  h.WaitFor(music, "completed");
  CHECK(h.api->SideOf("lied") == "remote");
  // Neither: 400 naming what each router has.
  const HttpResponse refused =
      h.Call("POST", "/v1/jobs", LoraImageJob("needs nix", "nix"));
  CHECK(refused.status == 400);
  CHECK(ErrorCodeOf(refused) == "no_capable_router");
  const std::string message = ErrorMessageOf(refused);
  CHECK(message.find("this router has no LoRA 'nix'") != std::string::npos);
  CHECK(message.find("registered: anime") != std::string::npos);
  CHECK(message.find("partner http://10.0.0.2:8081 has no LoRA 'nix'") !=
        std::string::npos);
  CHECK(message.find("registered: winkel") != std::string::npos);
  CHECK(h.Call("GET", "/v1/jobs").status == 200);
  // Partner capabilities unknown (unreachable): the partner may have it.
  h.queue->Stop();
  h.partner_capabilities = [] { return std::optional<json::Value>(); };
  h.Open(true);
  const std::string maybe =
      h.Submit(LoraImageJob("maybe there", "nix")).member_str("id");
  h.WaitFor(maybe, "completed");
  CHECK(h.api->SideOf("maybe there") == "remote");
  // Without any partner a LoRA this router lacks is refused too.
  h.queue->Stop();
  h.relay = nullptr;
  h.partner_name.clear();
  h.Open(true);
  CHECK(h.Call("POST", "/v1/jobs", LoraImageJob("alone", "nix")).status ==
        400);
  const std::string alone =
      h.Submit(LoraImageJob("alone anime", "anime")).member_str("id");
  h.WaitFor(alone, "completed");
  CHECK(h.api->SideOf("alone anime").empty());
}

/// The fake partner of the pull tests (router B as seen from A).
struct FakePartner {
  std::mutex mutex;
  /// Its queued list (newest first, as /v1/jobs answers).
  std::string queued;
  /// Its in_progress list (jobs waiting in its pool carry pool_wait).
  std::string in_progress = R"({"object":"list","data":[]})";
  /// Answer to POST /router/pool/pull.
  std::string pull_answer = R"({"job":"job_bbbbbbbbbbbbbbbbbbbbbbb2"})";
  int pull_status{200};
  int pulls{0};
  /// Reads of the queued list (the runner's view of this partner).
  int queued_reads{0};
  std::vector<HttpRequest> pull_requests;
  /// Polls of the pulled job before it completes.
  int polls_until_done{2};
  int polls{0};
  bool pulled{false};
  std::shared_ptr<FakeApi> api;

  std::optional<HttpResponse> operator()(const HttpRequest& request,
                                         std::chrono::milliseconds,
                                         std::chrono::milliseconds,
                                         std::string* error) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (request.header(jobs::kRelayHeader) != "1") {
      *error = "no relay header";
      return std::nullopt;
    }
    if (request.method == "GET" && request.path == "/v1/jobs" &&
        request.query.find("status=queued") != std::string::npos) {
      ++queued_reads;
      return JsonAnswer(200, pulled ? R"({"object":"list","data":[]})"
                                    : queued);
    }
    if (request.method == "GET" && request.path == "/v1/jobs" &&
        request.query.find("status=in_progress") != std::string::npos)
      return JsonAnswer(200, pulled ? R"({"object":"list","data":[]})"
                                    : in_progress);
    if (request.method == "GET" && request.path == "/v1/jobs")
      return JsonAnswer(
          200,
          R"({"object":"list","data":[)"
          R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb1","status":"queued","created_at":1000,"running_on":null},)"
          R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb3","status":"in_progress","created_at":1002,"running_on":"local"},)"
          R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb4","status":"in_progress","created_at":1003,"running_on":"http://10.0.0.1:8081"}]})");
    if (request.method == "POST" && request.path == jobs::kPullPath) {
      ++pulls;
      pull_requests.push_back(request);
      if (pull_status == 200) {
        pulled = true;
        if (api != nullptr) {
          const std::lock_guard<std::mutex> api_lock(api->mutex);
          api->calls.push_back("PULL");
        }
      }
      return JsonAnswer(pull_status,
                        pull_status == 200
                            ? pull_answer
                            : R"({"error":{"code":"not_found"}})");
    }
    if (request.method == "GET" &&
        request.path == "/v1/jobs/job_bbbbbbbbbbbbbbbbbbbbbbb2") {
      ++polls;
      if (polls < polls_until_done)
        return JsonAnswer(
            200,
            R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"in_progress","running_on":"http://10.0.0.1:8081"})");
      if (api != nullptr) {
        const std::lock_guard<std::mutex> api_lock(api->mutex);
        api->calls.push_back("PULLED DONE");
      }
      return JsonAnswer(
          200,
          R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"completed","running_on":null})");
    }
    return JsonAnswer(404, R"({"error":{"code":"not_found"}})");
  }
};

void TestPoolPullRequester() {
  // Router A: a free runner pulls the partner's oldest queued job it can
  // run (not the one needing the winkel LoRA) before its own newer job,
  // waits for it, and pulls nothing twice.
  auto partner = std::make_shared<FakePartner>();
  partner->queued =
      R"({"object":"list","data":[)"
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"queued","created_at":1001,"request":{"model":"qwen-image","loras":[]}},)"
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb1","status":"queued","created_at":1000,"request":{"model":"qwen-image","loras":["winkel"]}}]})";
  Harness h("pull-requester", false);
  partner->api = h.api;
  h.partner_name = "http://10.0.0.2:8081";
  h.self_name = "http://10.0.0.1:8081";
  h.capabilities = CapsA;
  h.partner_capabilities = [] { return std::optional<json::Value>(CapsB()); };
  h.relay = [partner](const HttpRequest& request, std::chrono::milliseconds a,
                      std::chrono::milliseconds b, std::string* error) {
    return (*partner)(request, a, b, error);
  };
  h.Open(true);
  const std::string mine = h.Submit(ImageJob("after the pull")).member_str("id");
  h.WaitFor(mine, "completed");
  {
    const std::lock_guard<std::mutex> lock(partner->mutex);
    CHECK(partner->pulls == 1);
    CHECK(partner->polls >= partner->polls_until_done);
    if (!partner->pull_requests.empty()) {
      const json::Value body = json::parse(partner->pull_requests[0].body);
      CHECK(body.member_str("from") == "http://10.0.0.1:8081");
      CHECK(body.member_str("router_id") == "ra");
      CHECK(body.member_str("job") == "job_bbbbbbbbbbbbbbbbbbbbbbb2");
      CHECK(body.find("capabilities") != nullptr &&
            body.find("capabilities")->find("models") != nullptr);
      CHECK(partner->pull_requests[0].header("Content-Type") ==
            "application/json");
    }
  }
  // Order: the pull and its completion came before our own job ran.
  {
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    const auto& calls = h.api->calls;
    const auto at = [&calls](const std::string& what) {
      return std::find(calls.begin(), calls.end(), what) - calls.begin();
    };
    CHECK(at("PULL") < at("PULLED DONE"));
    CHECK(at("PULLED DONE") < at("POST /v1/images/generations"));
    CHECK(at("POST /v1/images/generations") <
          static_cast<std::ptrdiff_t>(calls.size()));
  }
  const json::Value status = h.queue->PoolQueueStatus();
  CHECK(status.member_double("pulled") == 1);
  CHECK(status.member_double("pushed") == 0);
  CHECK(status.member_double("local") == 0);
  CHECK(status.member_str("partner_name") == "http://10.0.0.2:8081");
  CHECK(status.find("enabled")->as_bool());
  // The pool view: positions over both routers' queued jobs, running_on
  // from this router's point of view.
  const json::Value listed =
      json::parse(h.Call("GET", "/v1/jobs?scope=pool").body);
  const auto& data = listed.find("data")->items();
  CHECK(data.size() == 4);
  for (const json::Value& job : data) {
    const std::string id = job.member_str("id");
    if (id == "job_bbbbbbbbbbbbbbbbbbbbbbb1") {
      CHECK(job.member_double("position") == 1);
      CHECK(job.find("running_on")->is_null());
    } else if (id == "job_bbbbbbbbbbbbbbbbbbbbbbb3") {
      CHECK(job.find("position")->is_null());
      CHECK(job.member_str("running_on") == "http://10.0.0.2:8081");
    } else if (id == "job_bbbbbbbbbbbbbbbbbbbbbbb4") {
      CHECK(job.member_str("running_on") == "local");
    } else {
      CHECK(id == mine);
      CHECK(job.find("position")->is_null());
      CHECK(job.find("running_on")->is_null());
    }
  }
}

void TestPoolPullGpuBusy() {
  // A free runner whose GPU is busy (ComfyUI queue, another heavy model,
  // a pool request waiting here) pulls nothing: 12.11. 17:34 and 17:50
  // the .56 pulled while ComfyUI and the film ran, and the pulled request
  // waited 900 s. Once the GPU is free, the pull follows.
  auto partner = std::make_shared<FakePartner>();
  partner->queued =
      R"({"object":"list","data":[)"
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"queued","created_at":1001,"request":{"model":"qwen-image","loras":[]}}]})";
  Harness h("pull-gpu-busy", false);
  partner->api = h.api;
  h.partner_name = "http://10.0.0.2:8081";
  h.self_name = "http://10.0.0.1:8081";
  h.capabilities = CapsA;
  auto gpu_busy = std::make_shared<std::atomic<bool>>(true);
  auto asked = std::make_shared<std::atomic<int>>(0);
  h.may_pull = [gpu_busy, asked](const std::string& model, std::string* why) {
    ++*asked;
    if (gpu_busy->load()) {
      *why = "comfy queue 1";
      return false;
    }
    return model == "qwen-image";
  };
  h.relay = [partner](const HttpRequest& request, std::chrono::milliseconds a,
                      std::chrono::milliseconds b, std::string* error) {
    return (*partner)(request, a, b, error);
  };
  h.Open(true);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  {
    const std::lock_guard<std::mutex> lock(partner->mutex);
    CHECK(partner->pulls == 0);
  }
  CHECK(asked->load() > 0);
  CHECK(h.queue->PoolQueueStatus().member_double("pulled") == 0);
  gpu_busy->store(false);
  for (int i = 0; i < 200; ++i) {
    {
      const std::lock_guard<std::mutex> lock(partner->mutex);
      if (partner->pulls > 0)
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const std::lock_guard<std::mutex> lock(partner->mutex);
  CHECK(partner->pulls == 1);
}

void TestPoolPullOldPartner() {
  // The partner answers 404 on the pull route (older version): pulling
  // pauses for pull_hold, the local job runs as before, and a later
  // attempt follows after the hold.
  auto partner = std::make_shared<FakePartner>();
  partner->pull_status = 404;
  partner->queued =
      R"({"object":"list","data":[)"
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"queued","created_at":1001,"request":{"model":"qwen-image","loras":[]}}]})";
  Harness h("pull-old", false);
  h.partner_name = "http://10.0.0.2:8081";
  h.self_name = "http://10.0.0.1:8081";
  h.capabilities = CapsA;
  h.pull_hold = std::chrono::milliseconds(600);
  h.relay = [partner](const HttpRequest& request, std::chrono::milliseconds a,
                      std::chrono::milliseconds b, std::string* error) {
    return (*partner)(request, a, b, error);
  };
  h.Open(true);
  const std::string mine = h.Submit(ImageJob("as before")).member_str("id");
  h.WaitFor(mine, "completed");
  CHECK(h.api->SideOf("as before").empty());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {
    const std::lock_guard<std::mutex> lock(partner->mutex);
    CHECK(partner->pulls == 1);
  }
  CHECK(h.queue->PoolQueueStatus().member_double("pulled") == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(900));
  {
    const std::lock_guard<std::mutex> lock(partner->mutex);
    CHECK(partner->pulls >= 2);
  }
  // An unreachable partner: no pull, the job runs here.
  h.queue->Stop();
  h.relay = [](const HttpRequest&, std::chrono::milliseconds,
               std::chrono::milliseconds,
               std::string* error) -> std::optional<HttpResponse> {
    *error = "connect refused";
    return std::nullopt;
  };
  h.Open(true);
  const std::string again = h.Submit(ImageJob("partner gone")).member_str("id");
  h.WaitFor(again, "completed");
  CHECK(h.queue->PoolQueueStatus().find("partner")->is_null());
}

void TestPoolPinLocal() {
  // A job submitted with X-Gufo-Pool: local (the access gate sets it for
  // a user without the pool right) stays on this router: never handed
  // out on /router/pool/pull, runs with side local, keeps the pin over a
  // requeue and a restart; the owner comes from X-Gufo-User.
  Harness h("pin-local", false);
  h.capabilities = CapsB;
  h.self_name = "http://10.0.0.2:8081";
  h.Open(true);
  const auto submit = [&h](const std::string& body, bool pin,
                           const std::string& user) {
    HttpRequest request;
    request.method = "POST";
    request.path = "/v1/jobs";
    request.body = body;
    request.headers.emplace_back("Content-Type", "application/json");
    if (pin)
      request.headers.emplace_back(jobs::kPoolSideHeader, "local");
    if (!user.empty())
      request.headers.emplace_back("X-Gufo-User", user);
    const auto answer = h.queue->Handle(request);
    CHECK(answer.has_value());
    if (answer && answer->status != 202)
      std::cerr << "submit: HTTP " << answer->status << " " << answer->body
                << "\n";
    return answer ? json::parse(answer->body) : json::Value();
  };
  const std::string first =
      submit(ImageJob("SLOW first"), false, "").member_str("id");
  h.WaitFor(first, "in_progress");
  json::Value pinned = submit(
      ImageJob("pinned second", R"(,"metadata":{"nutzer":"fremd","titel":"t"})"),
      true, "tim");
  const std::string second = pinned.member_str("id");
  CHECK(pinned.member_str("pinned") == "local");
  CHECK(pinned.find("metadata")->member_str("nutzer") == "tim");  // header wins
  CHECK(pinned.find("metadata")->member_str("titel") == "t");
  const std::string third = submit(ImageJob("free third"), false, "").member_str("id");
  CHECK(json::parse(h.Call("GET", "/v1/jobs/" + third).body).find("pinned") == nullptr);
  const auto pull = [&h] {
    HttpRequest request;
    request.method = "POST";
    request.path = jobs::kPullPath;
    request.body = R"({"from":"http://10.0.0.1:8081","router_id":"ra","capabilities":)"
                   R"({"models":{"qwen-image":{"placement":"pool","loras":[]}}}})";
    request.headers.emplace_back("Content-Type", "application/json");
    request.headers.emplace_back(jobs::kRelayHeader, "1");
    const auto answer = h.queue->Handle(request);
    CHECK(answer && answer->status == 200);
    return answer ? json::parse(answer->body) : json::Value();
  };
  // The partner gets the free job, never the pinned one.
  CHECK(pull().member_str("job") == third);
  CHECK(pull().find("job")->is_null());
  h.api->release = true;  // the SLOW first job ends, the runner takes second
  h.WaitFor(second, "completed");
  CHECK(h.api->SideOf("pinned second") == "local");
  // The pin survives a restart of the queue (jobs.json).
  h.queue->Stop();
  h.Open(false);
  json::Value view = json::parse(h.Call("GET", "/v1/jobs/" + second).body);
  CHECK(view.member_str("pinned") == "local");
  CHECK(view.find("metadata")->member_str("nutzer") == "tim");
  // Only the partner could run it (a LoRA this router lacks): refused.
  Harness g("pin-local-refused", false);
  g.capabilities = CapsA;
  g.partner_capabilities = [] { return std::optional<json::Value>(CapsB()); };
  g.partner_name = "http://10.0.0.2:8081";
  g.relay = [](const HttpRequest&, std::chrono::milliseconds,
               std::chrono::milliseconds,
               std::string*) -> std::optional<HttpResponse> {
    return JsonAnswer(404, R"({"error":{"code":"not_found"}})");
  };
  g.Open(true);
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/jobs";
  request.body = LoraImageJob("needs winkel", "winkel");
  request.headers.emplace_back("Content-Type", "application/json");
  request.headers.emplace_back(jobs::kPoolSideHeader, "local");
  const auto refused = g.queue->Handle(request);
  CHECK(refused && refused->status == 400);
  CHECK(refused && json::parse(refused->body).find("error")->member_str("code") ==
                       "pool_not_allowed");
  request.headers.pop_back();
  const auto allowed = g.queue->Handle(request);
  CHECK(allowed && allowed->status == 202);
}

void TestPoolPullPinned() {
  // Requester: a partner job pinned to the partner (its user may not use
  // the pool) is no pull candidate: it neither orders ahead of ours nor
  // causes empty pull round trips.
  const auto run = [](const char* name, const std::string& partner_queue,
                      bool expect_pull) {
    auto partner = std::make_shared<FakePartner>();
    // The partner's list is there from the start: every decision of the
    // runner (before the first job, before ours, after ours) sees it.
    partner->queued = partner_queue;
    Harness h(name, false);
    partner->api = h.api;
    h.partner_name = "http://10.0.0.2:8081";
    h.self_name = "http://10.0.0.1:8081";
    h.capabilities = CapsA;
    h.partner_capabilities = [] {
      return std::optional<json::Value>(CapsB());
    };
    h.relay = [partner](const HttpRequest& request,
                        std::chrono::milliseconds a,
                        std::chrono::milliseconds b, std::string* error) {
      return (*partner)(request, a, b, error);
    };
    h.Open(true);
    const std::string first =
        h.Submit(ImageJob("SLOW first")).member_str("id");
    h.WaitFor(first, "in_progress");
    const std::string mine = h.Submit(ImageJob("mine")).member_str("id");
    h.WaitFor(mine, "queued");
    int reads_before = 0;
    {
      const std::lock_guard<std::mutex> lock(partner->mutex);
      reads_before = partner->queued_reads;
    }
    h.api->release.store(true);
    h.WaitFor(mine, "completed");
    // The runner looks at the partner's list once more after ours ran
    // (and pulls then, if anything is there): wait for that read.
    for (int i = 0; i < 400; ++i) {
      {
        const std::lock_guard<std::mutex> lock(partner->mutex);
        // Expected pull: wait for it; none expected: two more looks at
        // the list (the runner reads it again once ours ran) settle it.
        if (expect_pull ? partner->pulls >= 1
                        : partner->queued_reads >= reads_before + 2)
          break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    const std::lock_guard<std::mutex> lock(h.api->mutex);
    const auto& calls = h.api->calls;
    std::ptrdiff_t pulled = -1;
    std::ptrdiff_t ours = -1;
    int posts = 0;
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(calls.size()); ++i) {
      if (calls[i] == "PULL" && pulled < 0)
        pulled = i;
      if (calls[i] == "POST /v1/images/generations" && ++posts == 2)
        ours = i;
    }
    CHECK(ours >= 0);
    if (expect_pull) {
      CHECK(pulled >= 0);
      CHECK(ours < pulled);  // the unpinned one is newer and normal
    } else {
      CHECK(pulled < 0);  // only a pinned job there: no round trip
    }
  };
  const std::string pinned_head =
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"queued","created_at":4102444800,"priority":9,"pinned":"local",)"
      R"("request":{"model":"qwen-image","loras":[]}})";
  const std::string unpinned =
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb3","status":"queued","created_at":4102444801,"priority":0,)"
      R"("request":{"model":"qwen-image","loras":[]}})";
  run("pull-pinned-head", R"({"object":"list","data":[)" + pinned_head + "," +
                              unpinned + "]}",
      true);
  run("pull-pinned-only", R"({"object":"list","data":[)" + pinned_head + "]}",
      false);
}

void TestIdempotencyPerOwner() {
  // Idempotency keys are per owner (X-Gufo-User): another user with the
  // same key gets its own job, never a replay or a 409 of the first.
  Harness h("idempotency-owner", false);
  h.Open(false);
  const auto submit = [&h](const std::string& body, const std::string& user) {
    HttpRequest request;
    request.method = "POST";
    request.path = "/v1/jobs";
    request.body = body;
    request.headers.emplace_back("Content-Type", "application/json");
    request.headers.emplace_back("Idempotency-Key", "k1");
    if (!user.empty())
      request.headers.emplace_back("X-Gufo-User", user);
    const auto answer = h.queue->Handle(request);
    CHECK(answer.has_value());
    return answer ? *answer : HttpResponse{};
  };
  HttpResponse a = submit(ImageJob("a"), "anna");
  CHECK(a.status == 202);
  const std::string a_id = json::parse(a.body).member_str("id");
  HttpResponse b = submit(ImageJob("a"), "ben");
  CHECK(b.status == 202);
  CHECK(json::parse(b.body).member_str("id") != a_id);
  CHECK(json::parse(b.body).find("metadata")->member_str("nutzer") == "ben");
  HttpResponse again = submit(ImageJob("a"), "anna");
  CHECK(again.status == 200);
  CHECK(json::parse(again.body).member_str("id") == a_id);
  HttpResponse other = submit(ImageJob("b"), "ben");
  CHECK(other.status == 409);
  // Without a user (auth off): one shared key space, as before.
  HttpResponse none = submit(ImageJob("a"), "");
  CHECK(none.status == 202);
  CHECK(submit(ImageJob("a"), "").status == 200);
  // The split survives a restart.
  h.queue->Stop();
  h.Open(false);
  CHECK(submit(ImageJob("a"), "anna").status == 200);
  CHECK(json::parse(submit(ImageJob("a"), "anna").body).member_str("id") == a_id);
}

void TestPoolPullOwner() {
  // Router B: POST /router/pool/pull hands out the oldest queued job the
  // requester can run, never one a runner took, never twice, not one
  // needing a LoRA the requester lacks; the job runs with X-Gufo-Pool:
  // remote and shows the requester as running_on.
  Harness h("pull-owner", false);
  h.capabilities = CapsB;
  h.self_name = "http://10.0.0.2:8081";
  h.Open(true);
  const std::string first = h.Submit(ImageJob("SLOW first")).member_str("id");
  h.WaitFor(first, "in_progress");
  const std::string second =
      h.Submit(ImageJob("SLOW second")).member_str("id");
  const std::string third =
      h.Submit(LoraImageJob("third winkel", "winkel")).member_str("id");
  const std::string fourth = h.Submit(ImageJob("fourth")).member_str("id");
  const auto pull = [&h](const std::string& capabilities) {
    HttpRequest request;
    request.method = "POST";
    request.path = jobs::kPullPath;
    request.body = R"({"from":"http://10.0.0.1:8081","router_id":"ra","capabilities":)" +
                   capabilities + "}";
    request.headers.emplace_back("Content-Type", "application/json");
    request.headers.emplace_back(jobs::kRelayHeader, "1");
    const auto answer = h.queue->Handle(request);
    CHECK(answer && answer->status == 200);
    return answer ? json::parse(answer->body) : json::Value();
  };
  // A requester without the model gets nothing.
  CHECK(pull(R"({"models":{"minimax-h3":{"placement":"pool","loras":[]}}})")
            .find("job")
            ->is_null());
  // Nor one whose own load of the model just failed (12.11. 17:51: the
  // .198 pulled a qwen-image job 72 s after its own load failed with 502).
  CHECK(pull(R"({"models":{"qwen-image":{"placement":"pool","loras":[],)"
             R"("available":false,"unavailable":"local load failed"}}})")
            .find("job")
            ->is_null());
  // Without the winkel LoRA: the second job (the first is running).
  json::Value handed =
      pull(R"({"models":{"qwen-image":{"placement":"pool","loras":[]}}})");
  CHECK(handed.member_str("job") == second);
  CHECK(handed.member_str("model") == "qwen-image");
  CHECK(handed.member_double("queued") == 2);
  json::Value view = json::parse(h.Call("GET", "/v1/jobs/" + second).body);
  CHECK(view.member_str("status") == "in_progress");
  CHECK(view.member_str("running_on") == "http://10.0.0.1:8081");
  view = json::parse(h.Call("GET", "/v1/jobs/" + first).body);
  CHECK(view.member_str("running_on") == "local");
  // Pool view without a partner: positions of the queued jobs.
  const json::Value listed =
      json::parse(h.Call("GET", "/v1/jobs?scope=pool").body);
  CHECK(listed.member_str("partner") == "none");
  for (const json::Value& job : listed.find("data")->items()) {
    const std::string id = job.member_str("id");
    if (id == third)
      CHECK(job.member_double("position") == 1);
    else if (id == fourth)
      CHECK(job.member_double("position") == 2);
    else
      CHECK(job.find("position")->is_null());
  }
  json::Value status = h.queue->PoolQueueStatus();
  CHECK(status.member_double("pushed") == 1);
  CHECK(status.member_double("local") == 2);
  CHECK(status.find("pushed_active")->items().size() == 1 &&
        status.find("pushed_active")->items()[0].str() == second);
  // The same requester again: the fourth (third needs winkel).
  handed = pull(R"({"models":{"qwen-image":{"placement":"pool","loras":[]}}})");
  CHECK(handed.member_str("job") == fourth);
  h.WaitFor(fourth, "completed");
  CHECK(h.api->SideOf("fourth") == "remote");
  // With winkel: the third; then nothing is left.
  handed = pull(R"({"models":{"qwen-image":{"placement":"pool","loras":["winkel"]}}})");
  CHECK(handed.member_str("job") == third);
  h.WaitFor(third, "completed");
  CHECK(h.api->SideOf("third winkel") == "remote");
  handed = pull(R"({"models":{"qwen-image":{"placement":"pool","loras":["winkel"]}}})");
  CHECK(handed.find("job")->is_null());
  CHECK(handed.member_double("queued") == 0);
  // A model this router has not in the pool (ace-step: local only) is
  // never handed out, whatever the requester can run.
  const std::string music =
      h.Submit(R"({"request":{"path":"/v1/audio/music","body":{"model":"ace-step","prompt":"lied2"}}})")
          .member_str("id");
  CHECK(json::parse(h.Call("GET", "/v1/jobs/" + music).body)
            .member_str("status") == "queued");
  handed = pull(R"({"models":{"ace-step":{"placement":"pool","loras":[]}}})");
  CHECK(handed.find("job")->is_null());
  h.api->release.store(true);
  h.WaitFor(first, "completed");
  h.WaitFor(second, "completed");
  h.WaitFor(music, "completed");
  CHECK(h.api->SideOf("SLOW first").empty());
  CHECK(h.api->SideOf("SLOW second") == "remote");
  CHECK(h.api->SideOf("lied2").empty());
  status = h.queue->PoolQueueStatus();
  CHECK(status.member_double("pushed") == 3);
  CHECK(status.find("pushed_active")->items().empty());
  // Bad requests.
  HttpRequest bad;
  bad.method = "POST";
  bad.path = jobs::kPullPath;
  bad.body = R"({"from":"x"})";
  CHECK(h.queue->Handle(bad)->status == 400);
  bad.method = "GET";
  CHECK(h.queue->Handle(bad)->status == 405);
  // Without capabilities the route does not exist (as before 1.4.20).
  h.queue->Stop();
  h.capabilities = nullptr;
  h.Open(true);
  bad.method = "POST";
  bad.body = R"({"from":"http://10.0.0.1:8081","capabilities":{"models":{}}})";
  CHECK(h.queue->Handle(bad)->status == 404);
}

void TestReachableUrl() {
  // A router bound to 0.0.0.0 must not name itself so (12.11. 06:59: the
  // .198 pulled an ACE job and logged to=http://0.0.0.0:8081).
  CHECK(jobs::IsUnreachableUrl(""));
  CHECK(jobs::IsUnreachableUrl("http://0.0.0.0:8081"));
  CHECK(jobs::IsUnreachableUrl("http://[::]:8081"));
  CHECK(jobs::IsUnreachableUrl("http://::"));
  CHECK(jobs::IsUnreachableUrl("http://:8081"));
  CHECK(!jobs::IsUnreachableUrl("http://192.168.100.56:8081"));
  CHECK(!jobs::IsUnreachableUrl("http://evo56:8081/"));
  CHECK(jobs::ReachableUrl("http://0.0.0.0:8081", "http://192.168.100.56:8081") ==
        "http://192.168.100.56:8081");
  CHECK(jobs::ReachableUrl("http://10.0.0.1:8081", "x") ==
        "http://10.0.0.1:8081");
  // The route toward a loopback partner leaves from 127.0.0.1.
  CHECK(jobs::LocalUrlToward("http://127.0.0.1:8081", 8082) ==
        "http://127.0.0.1:8082");
  // Without a partner: the computer's name, never a bind-all address.
  const std::string own = jobs::LocalUrlToward("", 8081);
  CHECK(own.starts_with("http://") && own.ends_with(":8081"));
  CHECK(!jobs::IsUnreachableUrl(own));
}

void TestPoolPullBindAddress() {
  // A requester bound to 0.0.0.0 sends "from": "http://0.0.0.0:8081"; the
  // owner shows the partner URL it knows as running_on instead (1.4.22).
  Harness h("pull-bind", false);
  h.capabilities = CapsB;
  h.self_name = "http://10.0.0.2:8081";
  h.partner_name = "http://10.0.0.1:8081";
  h.Open(true);
  const std::string first = h.Submit(ImageJob("SLOW first")).member_str("id");
  h.WaitFor(first, "in_progress");
  const std::string second =
      h.Submit(ImageJob("SLOW second")).member_str("id");
  HttpRequest request;
  request.method = "POST";
  request.path = jobs::kPullPath;
  request.body =
      R"({"from":"http://0.0.0.0:8081","router_id":"ra","capabilities":{"models":{"qwen-image":{"placement":"pool","loras":[]}}}})";
  request.headers.emplace_back("Content-Type", "application/json");
  request.headers.emplace_back(jobs::kRelayHeader, "1");
  const auto answer = h.queue->Handle(request);
  CHECK(answer && answer->status == 200);
  CHECK(answer && json::parse(answer->body).member_str("job") == second);
  const json::Value view =
      json::parse(h.Call("GET", "/v1/jobs/" + second).body);
  CHECK(view.member_str("running_on") == "http://10.0.0.1:8081");
}

void TestPoolWaitTakeover() {
  // Router B with two runners busy: a job in_progress whose request waits
  // in the pool ("both busy") counts as queued for the shared queue,
  // shows running_on null / pool_wait true, has a position, and a pull
  // takes it over: the router answers the waiting request 503
  // pool_redirected and the runner sends it again with X-Gufo-Pool:
  // remote; nothing ran twice.
  Harness h("pool-wait", false);
  h.capabilities = CapsB;
  h.self_name = "http://10.0.0.2:8081";
  h.Open(true);
  const std::string first = h.Submit(ImageJob("WAIT first")).member_str("id");
  for (int i = 0; i < 200 && !h.api->Waiting(first); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  CHECK(h.api->Waiting(first));
  const std::string second = h.Submit(ImageJob("second")).member_str("id");
  json::Value view = json::parse(h.Call("GET", "/v1/jobs/" + first).body);
  CHECK(view.member_str("status") == "in_progress");
  CHECK(view.find("running_on")->is_null());
  CHECK(view.find("pool_wait")->as_bool());
  view = json::parse(h.Call("GET", "/v1/jobs/" + second).body);
  CHECK(view.member_str("status") == "queued");
  CHECK(!view.find("pool_wait")->as_bool());
  const json::Value listed =
      json::parse(h.Call("GET", "/v1/jobs?scope=pool").body);
  for (const json::Value& job : listed.find("data")->items()) {
    if (job.member_str("id") == first)
      CHECK(job.member_double("position") == 1);
    else
      CHECK(job.member_double("position") == 2);
  }
  json::Value status = h.queue->PoolQueueStatus();
  CHECK(status.member_double("local") == 2);
  CHECK(status.member_double("local_waiting") == 1);
  // The requester lacks nothing: the waiting job (oldest) is handed out.
  HttpRequest request;
  request.method = "POST";
  request.path = jobs::kPullPath;
  request.body =
      R"({"from":"http://10.0.0.1:8081","router_id":"ra","capabilities":{"models":{"qwen-image":{"placement":"pool","loras":[]}}}})";
  request.headers.emplace_back("Content-Type", "application/json");
  request.headers.emplace_back(jobs::kRelayHeader, "1");
  auto answer = h.queue->Handle(request);
  CHECK(answer && answer->status == 200);
  json::Value handed = json::parse(answer->body);
  CHECK(handed.member_str("job") == first);
  CHECK(handed.find("was_waiting")->as_bool());
  CHECK(handed.member_double("queued") == 1);
  h.WaitFor(first, "completed");
  CHECK(h.api->SideOf("WAIT first") == "remote");
  CHECK(!h.api->Waiting(first));
  h.WaitFor(second, "completed");
  CHECK(h.api->SideOf("second").empty());
  {
    // first (waited, then again as remote) + second: three POSTs, no
    // request ran twice on a worker.
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    CHECK(h.api->image_posts == 3);
  }
  status = h.queue->PoolQueueStatus();
  CHECK(status.member_double("pushed") == 1);
  CHECK(status.member_double("local") == 0);
  CHECK(status.find("pushed_active")->items().empty());
  // A job that waits no longer (released meanwhile) is not handed out.
  const std::string third = h.Submit(ImageJob("WAIT third")).member_str("id");
  for (int i = 0; i < 200 && !h.api->Waiting(third); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  h.api->release.store(true);
  for (int i = 0; i < 200 && h.api->Waiting(third); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  answer = h.queue->Handle(request);
  CHECK(answer && json::parse(answer->body).find("job")->is_null());
  h.WaitFor(third, "completed");
  CHECK(h.api->SideOf("WAIT third").empty());
}

void TestPoolPullWaitingRequester() {
  // Router A pulls the partner's job that is in_progress but waiting in
  // the partner's pool (pool_wait) when nothing is queued there.
  auto partner = std::make_shared<FakePartner>();
  partner->queued = R"({"object":"list","data":[]})";
  partner->in_progress =
      R"({"object":"list","data":[)"
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"in_progress","pool_wait":true,"running_on":null,"created_at":1001,"request":{"model":"qwen-image","loras":[]}},)"
      R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb3","status":"in_progress","pool_wait":false,"running_on":"local","created_at":1000,"request":{"model":"qwen-image","loras":[]}}]})";
  Harness h("pull-waiting", false);
  partner->api = h.api;
  h.partner_name = "http://10.0.0.2:8081";
  h.self_name = "http://10.0.0.1:8081";
  h.capabilities = CapsA;
  h.partner_capabilities = [] { return std::optional<json::Value>(CapsB()); };
  h.relay = [partner](const HttpRequest& request, std::chrono::milliseconds a,
                      std::chrono::milliseconds b, std::string* error) {
    return (*partner)(request, a, b, error);
  };
  h.Open(true);
  const std::string mine = h.Submit(ImageJob("after")).member_str("id");
  h.WaitFor(mine, "completed");
  {
    const std::lock_guard<std::mutex> lock(partner->mutex);
    CHECK(partner->pulls == 1);
    if (!partner->pull_requests.empty())
      CHECK(json::parse(partner->pull_requests[0].body).member_str("job") ==
            "job_bbbbbbbbbbbbbbbbbbbbbbb2");
  }
  CHECK(h.queue->PoolQueueStatus().member_double("pulled") == 1);
  // The partner's running job keeps its running_on; the pulled one shows
  // none while the partner lists it as waiting.
  partner->pulled = false;
  const json::Value listed =
      json::parse(h.Call("GET", "/v1/jobs?scope=pool").body);
  for (const json::Value& job : listed.find("data")->items()) {
    const std::string id = job.member_str("id");
    if (id == "job_bbbbbbbbbbbbbbbbbbbbbbb3")
      CHECK(job.member_str("running_on") == "http://10.0.0.2:8081");
  }
}

}  // namespace

/// GET /v1/jobs?scope=pool positions by job id.
std::map<std::string, double> Positions(Harness& h) {
  std::map<std::string, double> positions;
  const json::Value listed =
      json::parse(h.Call("GET", "/v1/jobs?scope=pool").body);
  for (const json::Value& job : listed.find("data")->items()) {
    const json::Value* position = job.find("position");
    positions[job.member_str("id")] =
        position == nullptr || position->is_null() ? 0
                                                   : position->as_double();
  }
  return positions;
}

void TestPriorityOrder() {
  // 2.0.3: the queue runs the highest priority first, then first in first
  // out; the priority comes from the header (wins), the submit body or
  // the wrapped body; every request of the job carries the header.
  Harness h("priority-order");
  const std::string first = h.Submit(ImageJob("SLOW first")).member_str("id");
  h.WaitFor(first, "in_progress");
  const std::string a = h.Submit(ImageJob("a")).member_str("id");
  const std::string b =
      h.Submit(ImageJob("b", R"(,"priority":5)")).member_str("id");
  const std::string c = h.Submit(ImageJob("c")).member_str("id");
  std::string d;
  {
    HttpRequest request;
    request.method = "POST";
    request.path = "/v1/jobs";
    request.body = ImageJob("d", R"(,"priority":2)");
    request.headers.emplace_back(gufo::router::kPriorityHeader, "9");
    const auto answer = h.queue->Handle(request);
    CHECK(answer && answer->status == 202);
    const json::Value view = json::parse(answer->body);
    d = view.member_str("id");
    CHECK(view.member_double("priority") == 9);  // the header wins
  }
  // The wrapped body's member counts too (a page that sends it inside).
  const std::string e =
      h.Submit(R"({"request":{"path":"/v1/images/generations","body":{"model":"qwen-image","prompt":"e","size":"64x64","priority":"5"}}})")
          .member_str("id");
  CHECK(json::parse(h.Call("GET", "/v1/jobs/" + a).body)
            .member_double("priority") == 0);
  CHECK(json::parse(h.Call("GET", "/v1/jobs/" + b).body)
            .member_double("priority") == 5);
  CHECK(json::parse(h.Call("GET", "/v1/jobs/" + e).body)
            .member_double("priority") == 5);
  // Positions: d (9), b (5), e (5, newer), a, c.
  std::map<std::string, double> positions = Positions(h);
  CHECK(positions[d] == 1);
  CHECK(positions[b] == 2);
  CHECK(positions[e] == 3);
  CHECK(positions[a] == 4);
  CHECK(positions[c] == 5);
  CHECK(positions[first] == 0);
  // Invalid values: 400 invalid_priority, nothing queued.
  json::Value refused = h.Submit(ImageJob("bad", R"(,"priority":12)"), 400);
  CHECK(refused.find("error")->member_str("code") == "invalid_priority");
  refused = h.Submit(ImageJob("bad", R"(,"priority":"hoch")"), 400);
  CHECK(refused.find("error")->member_str("code") == "invalid_priority");
  refused = h.Submit(ImageJob("bad", R"(,"priority":1.5)"), 400);
  CHECK(refused.find("error")->member_str("code") == "invalid_priority");
  {
    HttpRequest request;
    request.method = "POST";
    request.path = "/v1/jobs";
    request.body = ImageJob("bad");
    request.headers.emplace_back(gufo::router::kPriorityHeader, "10");
    const auto answer = h.queue->Handle(request);
    CHECK(answer && answer->status == 400);
    CHECK(answer && ErrorCodeOf(*answer) == "invalid_priority");
  }
  CHECK(Positions(h).size() == 6);
  // Run order after the first job ends: d, b, e, a, c.
  h.api->release.store(true);
  for (const std::string& id : {first, d, b, e, a, c})
    h.WaitFor(id, "completed");
  {
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    const std::vector<std::string> want = {"SLOW first", "d", "b",
                                           "e",          "a", "c"};
    CHECK(h.api->prompts == want);
    CHECK(h.api->priorities["d"] == "9");
    CHECK(h.api->priorities["b"] == "5");
    CHECK(h.api->priorities["a"].empty());  // normal: no header
  }
}

void TestPriorityFile() {
  // jobs.json keeps the priority; a file written before 2.0.3 has none and
  // reads as 0; queued jobs come back in run order.
  const std::string name = "priority-file";
  std::filesystem::path dir;
  std::string normal;
  std::string high;
  {
    Harness h(name, false);
    dir = h.dir;
    normal = h.Submit(ImageJob("normal")).member_str("id");
    high = h.Submit(ImageJob("high", R"(,"priority":7)")).member_str("id");
    const std::map<std::string, double> positions = Positions(h);
    CHECK(positions.at(high) == 1);
    CHECK(positions.at(normal) == 2);
  }
  const std::filesystem::path file = dir / "jobs.json";
  std::string text = ReadFile(file);
  CHECK(text.find("\"priority\":7") != std::string::npos);
  {
    Harness h(name, false, /*keep_dir=*/true);
    CHECK(json::parse(h.Call("GET", "/v1/jobs/" + high).body)
              .member_double("priority") == 7);
    CHECK(Positions(h).at(high) == 1);
  }
  // Drop the member as an older router wrote the file.
  {
    const json::Value parsed = json::parse(text);
    json::Value records = json::Value::array();
    for (const json::Value& record : parsed.find("jobs")->items()) {
      json::Value copy = json::Value::object();
      for (const auto& [key, value] : record.members())
        if (key != "priority")
          copy[key] = value;
      records.push_back(copy);
    }
    json::Value rewritten = json::Value::object();
    rewritten["jobs"] = std::move(records);
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << rewritten.dump();
  }
  {
    Harness h(name, false, /*keep_dir=*/true);
    const json::Value view = json::parse(h.Call("GET", "/v1/jobs/" + high).body);
    CHECK(view.member_str("status") == "queued");
    CHECK(view.member_double("priority") == 0);
    CHECK(Positions(h).at(normal) == 1);
    CHECK(Positions(h).at(high) == 2);
  }
}

void TestPoolPullPriority() {
  // Requester: the partner's newer job with a higher priority is pulled
  // before our own older job; a newer job of the same priority is not.
  const auto run = [](const char* name, int partner_priority,
                      bool expect_pull_first) {
    auto partner = std::make_shared<FakePartner>();
    partner->queued = R"({"object":"list","data":[]})";
    Harness h(name, false);
    partner->api = h.api;
    h.partner_name = "http://10.0.0.2:8081";
    h.self_name = "http://10.0.0.1:8081";
    h.capabilities = CapsA;
    h.partner_capabilities = [] {
      return std::optional<json::Value>(CapsB());
    };
    h.relay = [partner](const HttpRequest& request,
                        std::chrono::milliseconds a,
                        std::chrono::milliseconds b, std::string* error) {
      return (*partner)(request, a, b, error);
    };
    h.Open(true);
    // The runner is busy while the partner's job and ours queue up.
    const std::string first =
        h.Submit(ImageJob("SLOW first")).member_str("id");
    h.WaitFor(first, "in_progress");
    {
      // created_at in the year 2100: newer than anything submitted here.
      const std::lock_guard<std::mutex> lock(partner->mutex);
      partner->queued =
          R"({"object":"list","data":[)"
          R"({"id":"job_bbbbbbbbbbbbbbbbbbbbbbb2","status":"queued","created_at":4102444800,"priority":)" +
          std::to_string(partner_priority) +
          R"(,"request":{"model":"qwen-image","loras":[]}}]})";
    }
    const std::string mine = h.Submit(ImageJob("mine")).member_str("id");
    // The runner's view of the partner's queue is cached pull_poll (20 ms
    // here): let it age so the runner reads the new list.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    h.api->release.store(true);
    h.WaitFor(mine, "completed");
    // Once ours ran and nothing is queued here, the partner's job is
    // pulled anyway (as before).
    for (int i = 0; i < 200; ++i) {
      {
        const std::lock_guard<std::mutex> lock(partner->mutex);
        if (partner->pulls >= 1)
          break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const std::lock_guard<std::mutex> lock(h.api->mutex);
    const auto& calls = h.api->calls;
    const auto at = [&calls](const std::string& what, int nth) {
      std::ptrdiff_t index = 0;
      for (; index < static_cast<std::ptrdiff_t>(calls.size()); ++index)
        if (calls[index] == what && --nth == 0)
          break;
      return index;
    };
    const std::ptrdiff_t pulled = at("PULL", 1);
    const std::ptrdiff_t ours = at("POST /v1/images/generations", 2);
    CHECK(pulled < static_cast<std::ptrdiff_t>(calls.size()));
    CHECK(ours < static_cast<std::ptrdiff_t>(calls.size()));
    if (expect_pull_first)
      CHECK(pulled < ours);
    else
      CHECK(ours < pulled);
  };
  run("pull-priority-high", 5, true);
  run("pull-priority-same", 0, false);

  // Owner: POST /router/pool/pull hands out the queued job that runs
  // first (priority, then age), and names its priority.
  Harness h("pull-priority-owner", false);
  h.capabilities = CapsB;
  h.self_name = "http://10.0.0.2:8081";
  h.Open(true);
  const std::string first = h.Submit(ImageJob("SLOW first")).member_str("id");
  h.WaitFor(first, "in_progress");
  const std::string second = h.Submit(ImageJob("second")).member_str("id");
  const std::string third =
      h.Submit(ImageJob("third", R"(,"priority":9)")).member_str("id");
  const std::string fourth =
      h.Submit(ImageJob("fourth", R"(,"priority":9)")).member_str("id");
  const auto pull = [&h] {
    HttpRequest request;
    request.method = "POST";
    request.path = jobs::kPullPath;
    request.body =
        R"({"from":"http://10.0.0.1:8081","router_id":"ra","capabilities":{"models":{"qwen-image":{"placement":"pool","loras":[]}}}})";
    request.headers.emplace_back("Content-Type", "application/json");
    request.headers.emplace_back(jobs::kRelayHeader, "1");
    const auto answer = h.queue->Handle(request);
    CHECK(answer && answer->status == 200);
    return answer ? json::parse(answer->body) : json::Value();
  };
  json::Value handed = pull();
  CHECK(handed.member_str("job") == third);
  CHECK(handed.member_double("priority") == 9);
  handed = pull();
  CHECK(handed.member_str("job") == fourth);
  handed = pull();
  CHECK(handed.member_str("job") == second);
  CHECK(handed.member_double("priority") == 0);
  h.api->release.store(true);
  for (const std::string& id : {first, second, third, fourth})
    h.WaitFor(id, "completed");
  // The pushed jobs' requests carried the priority to the pool.
  CHECK(h.api->priorities["third"] == "9");
  CHECK(h.api->priorities["second"].empty());
}

int main() {
  TestParseWebhookUrl();
  TestAllowList();
  TestCheckTarget();
  TestSignature();
  TestImageLifecycle();
  TestFailedJob();
  TestRequeueOnceWhenRemoteGone();
  TestBinaryResult();
  TestSubmitRejections();
  TestCancel();
  TestVideo();
  TestVideoCancel();
  TestObjekt();
  TestObjektCancel();
  TestObjektTransient();
  TestObjektTransientGivesUp();
  TestObjektDeadline();
  TestWebhook();
  TestWebhookGivesUp();
  TestIdempotency();
  TestRestart();
  TestParseCutoff();
  TestKeepAndExpiry();
  TestDeleteFinished();
  TestBulkDelete();
  TestListFilters();
  TestEstimatedProgress();
  TestMusic();
  TestConcatHelpers();
  TestConcatEndpoint();
  TestThumb();
  TestPoolRelay();
  TestLoraNamesAndCanRun();
  TestCapabilityRouting();
  TestPoolPullRequester();
  TestPoolPullOldPartner();
  TestPoolPullGpuBusy();
  TestPoolPullOwner();
  TestReachableUrl();
  TestPoolPullBindAddress();
  TestPoolWaitTakeover();
  TestPoolPullWaitingRequester();
  TestPoolPinLocal();
  TestPoolPullPinned();
  TestIdempotencyPerOwner();
  TestPriorityOrder();
  TestPriorityFile();
  TestPoolPullPriority();
  if (failures != 0) {
    std::cerr << failures << " router job checks failed\n";
    return 1;
  }
  std::cout << "All router job tests passed.\n";
  return 0;
}
