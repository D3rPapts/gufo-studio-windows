// placement = pool over real sockets: `gufo router` (argv[1]) runs as a
// child process against a fake partner router on loopback, and the local
// twin is this test binary itself (--fake-worker).
//
// Regression 14.11. 10:33 (2.0.5, .198): the partner answered a pool chat
// with 503 and a body. The router dropped that answer to decide again
// locally, and destroying it under the state lock released the remote
// hold, which re-locked the same std::mutex inside a destructor: the
// process ended with 0xc0000409. Here the partner always answers 503 with
// a JSON body; the request must be served by the local worker and the
// router must stay up.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/cli/serve/router/workers.hpp"
#include "src/core/platform/net.hpp"

namespace {

namespace net = gufo::net;
using gufo::router::ProxyToWorker;
using gufo::router::UpstreamTarget;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

/// One loopback listener; each connection gets the raw request text (head
/// + Content-Length body) and is closed after `serve` answered it.
class Listener {
public:
  using Serve = std::function<std::string(const std::string& request)>;

  Listener(int port, Serve serve) : serve_(std::move(serve)) {
    listener_ = net::CreateStreamSocket(AF_INET);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (listener_ == net::kInvalidSocket ||
        ::bind(listener_, reinterpret_cast<const sockaddr*>(&address),
               sizeof(address)) != 0 ||
        ::listen(listener_, 16) != 0) {
      std::cerr << "cannot listen on 127.0.0.1:" << port << "\n";
      std::_Exit(3);
    }
    net::SocklenT length = sizeof(address);
    (void)::getsockname(listener_, reinterpret_cast<sockaddr*>(&address),
                        &length);
    port_ = ntohs(address.sin_port);
  }

  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;
  ~Listener() { net::CloseSocket(listener_); }

  int port() const { return port_; }

  /// Serves connections one after the other until the process ends; with
  /// `idle_exit`, ends the process after that long without a connection
  /// (the fake worker, should its router be gone without a job object).
  void Run(std::chrono::seconds idle_exit = std::chrono::seconds(0)) {
    auto last = std::chrono::steady_clock::now();
    for (;;) {
      if (net::PollReadable(listener_, 100) <= 0) {
        if (idle_exit.count() > 0 &&
            std::chrono::steady_clock::now() - last > idle_exit)
          std::_Exit(0);
        continue;
      }
      last = std::chrono::steady_clock::now();
      const net::Socket client = net::AcceptClientSocket(listener_);
      if (client == net::kInvalidSocket)
        continue;
      const std::string request = ReadRequest(client);
      const std::string text = serve_(request);
      std::string_view answer = text;
      while (!answer.empty()) {
        const auto sent = net::SendNoSignal(client, answer.data(), answer.size());
        if (sent <= 0)
          break;
        answer.remove_prefix(static_cast<std::size_t>(sent));
      }
      net::CloseSocket(client);
    }
  }

  void Start() {
    std::thread([this] { Run(); }).detach();
  }

private:
  static std::string ReadRequest(net::Socket client) {
    std::string data;
    char buffer[4096];
    std::size_t want = std::string::npos;
    while (want == std::string::npos || data.size() < want) {
      if (net::PollReadable(client, 10000) <= 0)
        break;
      const auto count = net::Receive(client, buffer, sizeof(buffer));
      if (count <= 0)
        break;
      data.append(buffer, static_cast<std::size_t>(count));
      const auto end = data.find("\r\n\r\n");
      if (want == std::string::npos && end != std::string::npos) {
        std::size_t body = 0;
        for (const char* name : {"Content-Length: ", "content-length: "}) {
          const auto at = data.find(name);
          if (at != std::string::npos && at < end)
            body = std::stoul(data.substr(at + 16));
        }
        want = end + 4 + body;
      }
    }
    return data;
  }

  Serve serve_;
  net::Socket listener_{net::kInvalidSocket};
  int port_{0};
};

std::string Answer(int status, std::string_view reason, std::string_view body) {
  return "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) +
         "\r\nContent-Type: application/json\r\nContent-Length: " +
         std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" +
         std::string(body);
}

bool StartsWith(const std::string& text, std::string_view prefix) {
  return std::string_view(text).starts_with(prefix);
}

/// The local twin: /health and every GET 200, every POST a chat answer
/// marked "where": "local".
int ServeFakeWorker(std::span<char*> args) {
  int port = 0;
  for (std::size_t i = 0; i + 1 < args.size(); ++i)
    if (std::string_view(args[i]) == "--port")
      port = std::atoi(args[i + 1]);
  if (port <= 0)
    return 2;
  Listener worker(port, [](const std::string& request) {
    if (StartsWith(request, "POST "))
      return Answer(200, "OK",
                    "{\"object\":\"chat.completion\",\"where\":\"local\"}");
    return Answer(200, "OK", "{\"status\":\"ok\"}");
  });
  worker.Run(std::chrono::seconds(60));
  return 0;
}

HttpResponse Ask(int port, const std::string& method, const std::string& path,
                 const std::string& body = "") {
  UpstreamTarget target;
  target.host = "127.0.0.1";
  target.port = port;
  target.connect_timeout = std::chrono::seconds(2);
  target.head_timeout = std::chrono::seconds(60);
  HttpRequest request;
  request.method = method;
  request.path = path;
  request.body = body;
  if (!body.empty())
    request.headers.emplace_back("Content-Type", "application/json");
  return ProxyToWorker(target, request);
}

std::string Drain(HttpResponse& response) {
  std::string body = response.body;
  if (response.streaming_body) {
    response.streaming_body([&body](std::string_view chunk) {
      body.append(chunk);
      return true;
    });
  }
  return body;
}

}  // namespace

int main(int argc, char** argv) {
  net::EnsureSocketRuntimeStarted();
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  for (char* arg : args)
    if (std::string_view(arg) == "--fake-worker")
      return ServeFakeWorker(args);
  if (argc < 2) {
    std::cerr << "usage: router_pool_relay_test <path to gufo>\n";
    return 2;
  }
  const std::string gufo_exe = argv[1];

  // The partner: lists "twin" as loaded and free, and answers every chat
  // with 503 and a JSON body (as the .56 did on 14.11. after its load
  // timed out).
  std::atomic<int> listings{0};
  std::atomic<int> chats{0};
  Listener partner(0, [&](const std::string& request) {
    if (StartsWith(request, "GET /v1/models")) {
      ++listings;
      return Answer(200, "OK",
                    "{\"object\":\"list\",\"data\":[{\"id\":\"twin\","
                    "\"object\":\"model\",\"loaded\":true,\"status\":"
                    "{\"value\":\"loaded\",\"args\":[]}}]}");
    }
    if (StartsWith(request, "POST /v1/chat/completions")) {
      ++chats;
      return Answer(503, "Service Unavailable",
                    "{\"error\":{\"message\":\"loading the model timed out\","
                    "\"type\":\"server_error\",\"code\":\"unavailable\"}}");
    }
    return Answer(404, "Not Found", "{\"error\":{\"code\":\"not_found\"}}");
  });
  partner.Start();

  const std::filesystem::path self =
      std::filesystem::absolute(std::filesystem::path(argv[0]));
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("gufo_pool_relay_" + std::to_string(partner.port()));
  std::filesystem::create_directories(dir);
  const std::filesystem::path preset = dir / "pool.preset";
  {
    std::ofstream out(preset, std::ios::binary);
    out << "[external/twin]\ncommand = " << self.string()
        << "\nargs = --fake-worker\n\n"
        << "[remote/twin]\nurl = http://127.0.0.1:" << partner.port()
        << "\nqueue-timeout-seconds = 20\nplacement = pool\n";
  }

  // The router under test: `gufo router` as a child process (its own
  // worker spawner starts it; the job object / PDEATHSIG ends it with us).
  std::string error;
  const int port = gufo::router::ReserveLoopbackPort(&error);
  if (port <= 0) {
    std::cerr << "no loopback port: " << error << "\n";
    return 1;
  }
  gufo::router::WorkerSpec spec;
  spec.model_id = "router";
  spec.port = port;
  spec.command = gufo_exe;
  spec.args = {"router",       "--models-preset", preset.string(),
               "--models-max", "0",               "--remote-allow",
               "127.0.0.1"};
  std::unique_ptr<gufo::router::Worker> router =
      gufo::router::Worker::Spawn(spec, gufo_exe, &error);
  if (router == nullptr) {
    std::cerr << "cannot start " << gufo_exe << ": " << error << "\n";
    return 1;
  }
  const auto router_gone = [&router] {
    int status = 0;
    if (!router->PollExit(&status))
      return false;
    std::cerr << "router exited with status " << status << " (0x" << std::hex
              << static_cast<unsigned>(status) << std::dec << ")\n";
    return true;
  };

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  bool healthy = false;
  while (!healthy && std::chrono::steady_clock::now() < deadline &&
         !router_gone()) {
    healthy = router->Healthy();
    if (!healthy)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  CHECK(healthy);
  if (!healthy) {
    std::cerr << "router did not come up\n";
    std::_Exit(1);
  }
  // Two listings: the probe has stored the partner's view at least once.
  while (listings.load() < 2 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(listings.load() >= 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Local cold, partner warm and free: the pool picks the partner, which
  // answers 503 with a body. The router drops it and serves the request
  // here (the crash happened while dropping it).
  HttpResponse answer = Ask(
      port, "POST", "/v1/chat/completions",
      "{\"model\":\"twin\",\"messages\":[{\"role\":\"user\",\"content\":"
      "\"x\"}]}");
  const std::string body = Drain(answer);
  // Give a dying router a moment to be reaped.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  CHECK(!router_gone());
  CHECK(chats.load() == 1);
  CHECK(answer.status == 200);
  CHECK(body.find("\"where\":\"local\"") != std::string::npos);
  if (answer.status != 200)
    std::cerr << "answer " << answer.status << ": " << body << "\n";

  // Still up, and the partner is held off: the next request stays local.
  HttpResponse again = Ask(
      port, "POST", "/v1/chat/completions",
      "{\"model\":\"twin\",\"messages\":[{\"role\":\"user\",\"content\":"
      "\"y\"}]}");
  const std::string again_body = Drain(again);
  CHECK(again.status == 200);
  CHECK(again_body.find("\"where\":\"local\"") != std::string::npos);
  CHECK(chats.load() == 1);
  CHECK(!router_gone());

  // Stopping the router ends its worker (job object / PDEATHSIG).
  router->Stop(std::chrono::seconds(5));
  router.reset();
  std::error_code ignored;
  std::filesystem::remove_all(dir, ignored);
  if (failures != 0) {
    std::cerr << failures << " router pool relay check(s) failed\n";
    std::cerr.flush();
    std::_Exit(1);
  }
  std::cout << "All router pool relay tests passed.\n";
  std::cout.flush();
  // The partner's listener thread never returns.
  std::_Exit(0);
}
