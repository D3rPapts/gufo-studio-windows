// CPU tests for the router job API (src/cli/serve/router/jobs.*): webhook
// target checks (SSRF), Standard Webhooks signatures, the job life cycle
// against a fake router API and a fake webhook receiver, and the result
// store: keep flag, expiry, delete (single and bulk), list filters, music
// progress, the estimated progress and the video concat helpers (URL to
// job file, ffprobe parsing, copy vs re-encode; no ffmpeg runs here).

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/jobs.hpp"
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
  int unreachable_calls{0};
  /// Music: the async job stays in_progress while set.
  std::atomic<bool> music_hold{false};
  bool music_async{false};
  int music_polls{0};

  jobs::FetchResult operator()(const HttpRequest& request) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      calls.push_back(request.method + " " + request.path +
                      (request.query.empty() ? "" : "?" + request.query));
    }
    jobs::FetchResult result;
    result.status = 200;
    result.content_type = "application/json";
    if (request.path == "/v1/images/generations") {
      const json::Value body = json::parse(request.body);
      const std::string prompt = body.member_str("prompt");
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
    options.models = {"qwen-image", "minimax-h3", "qwen3-tts", "ace-step"};
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

}  // namespace

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
  if (failures != 0) {
    std::cerr << failures << " router job checks failed\n";
    return 1;
  }
  std::cout << "All router job tests passed.\n";
  return 0;
}
