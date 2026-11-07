// CPU tests for the router job API (src/cli/serve/router/jobs.*): webhook
// target checks (SSRF), Standard Webhooks signatures, and the job life
// cycle against a fake router API and a fake webhook receiver.

#include <atomic>
#include <chrono>
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
      result.content_type = "audio/wav";
      result.body = "RIFFfake-wav";
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
    options.models = {"qwen-image", "minimax-h3", "qwen3-tts"};
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
  CHECK(h.Call("DELETE", "/v1/jobs/" + running).status == 409);
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

}  // namespace

int main() {
  TestParseWebhookUrl();
  TestAllowList();
  TestCheckTarget();
  TestSignature();
  TestImageLifecycle();
  TestFailedJob();
  TestBinaryResult();
  TestSubmitRejections();
  TestCancel();
  TestVideo();
  TestVideoCancel();
  TestWebhook();
  TestWebhookGivesUp();
  TestIdempotency();
  TestRestart();
  if (failures != 0) {
    std::cerr << failures << " router job checks failed\n";
    return 1;
  }
  std::cout << "All router job tests passed.\n";
  return 0;
}
