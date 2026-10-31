// Proxy options for remote routers (UpstreamTarget connect_timeout,
// head_timeout, extra_headers, keepalive). Portable: Winsock and POSIX
// through src/core/platform/net.hpp, threads instead of fork.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/core/platform/net.hpp"

namespace {

namespace net = gufo::net;
using gufo::router::ProxyToWorker;
using gufo::router::UpstreamTarget;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using Clock = std::chrono::steady_clock;

/// One listener on 127.0.0.1:<ephemeral>. Each accepted connection runs
/// `serve` with the raw request text (head + Content-Length body).
class FakeUpstream {
public:
  using Serve = std::function<void(net::Socket, const std::string&)>;

  explicit FakeUpstream(Serve serve, int backlog = 4, bool accept = true)
      : serve_(std::move(serve)) {
    listener_ = net::CreateStreamSocket(AF_INET);
    assert(listener_ != net::kInvalidSocket);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(::bind(listener_, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) == 0);
    assert(::listen(listener_, backlog) == 0);
    net::SocklenT length = sizeof(address);
    assert(::getsockname(listener_, reinterpret_cast<sockaddr*>(&address),
                         &length) == 0);
    port_ = ntohs(address.sin_port);
    if (accept) {
      thread_ = std::thread([this] { Run(); });
    }
  }
  FakeUpstream(const FakeUpstream&) = delete;
  FakeUpstream& operator=(const FakeUpstream&) = delete;
  ~FakeUpstream() {
    stop_ = true;
    net::CloseSocket(listener_);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  int port() const { return port_; }
  net::Socket listener() const { return listener_; }

  std::string Captured() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return captured_;
  }

private:
  void Run() {
    while (!stop_) {
      if (net::PollReadable(listener_, 50) <= 0) {
        continue;
      }
      const net::Socket client = net::AcceptClientSocket(listener_);
      if (client == net::kInvalidSocket) {
        continue;
      }
      std::string request = ReadRequest(client);
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        captured_ = request;
      }
      serve_(client, request);
      net::CloseSocket(client);
    }
  }

  static std::string ReadRequest(net::Socket client) {
    std::string data;
    char buffer[4096];
    std::size_t want = std::string::npos;
    while (want == std::string::npos || data.size() < want) {
      const auto count = net::Receive(client, buffer, sizeof(buffer));
      if (count <= 0) {
        break;
      }
      data.append(buffer, static_cast<std::size_t>(count));
      const auto end = data.find("\r\n\r\n");
      if (want == std::string::npos && end != std::string::npos) {
        std::size_t body = 0;
        const auto at = data.find("Content-Length: ");
        if (at != std::string::npos && at < end) {
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
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::mutex mutex_;
  std::string captured_;
};

void SendText(net::Socket socket, std::string_view text) {
  while (!text.empty()) {
    const auto count = net::SendNoSignal(socket, text.data(), text.size());
    assert(count > 0);
    text.remove_prefix(static_cast<std::size_t>(count));
  }
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

HttpRequest PostJson(std::string body) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/images/generations";
  request.query = "trace=1";
  request.body = std::move(body);
  request.headers = {{"Content-Type", "application/json"},
                     {"Authorization", "Bearer client-secret"},
                     {"X-Trace", "client"}};
  return request;
}

void TestExtraHeadersReplaceClientAuthorization() {
  FakeUpstream upstream([](net::Socket client, const std::string&) {
    SendText(client,
             "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
             "Content-Length: 11\r\n\r\n{\"ok\":true}");
  });
  UpstreamTarget target;
  target.port = upstream.port();
  target.connect_timeout = std::chrono::milliseconds(2000);
  target.keepalive = true;
  target.extra_headers = {{"Authorization", "Bearer remote-key"},
                          {"x-trace", "router"}};
  const std::string body = "{\"model\":\"m\",\"prompt\":\"\xC3\xA4\"}";
  HttpResponse response = ProxyToWorker(target, PostJson(body));
  assert(response.status == 200);
  assert(Drain(response) == "{\"ok\":true}");
  const std::string seen = upstream.Captured();
  assert(seen.starts_with("POST /v1/images/generations?trace=1 HTTP/1.1\r\n"));
  assert(seen.find("Bearer client-secret") == std::string::npos);
  assert(seen.find("Authorization: Bearer remote-key\r\n") !=
         std::string::npos);
  assert(seen.find("X-Trace: client") == std::string::npos);
  assert(seen.find("x-trace: router\r\n") != std::string::npos);
  // The body reaches the remote byte for byte.
  assert(seen.ends_with("\r\n\r\n" + body));
}

void TestDefaultTargetSendsNoExtras() {
  FakeUpstream upstream([](net::Socket client, const std::string&) {
    SendText(client, "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n");
  });
  UpstreamTarget target;
  target.port = upstream.port();
  HttpResponse response = ProxyToWorker(target, PostJson("{}"));
  assert(response.status == 204);
  Drain(response);
  const std::string seen = upstream.Captured();
  assert(seen.find("Authorization") == std::string::npos);
  assert(seen.find("X-Trace: client\r\n") != std::string::npos);
}

void TestConnectRefusedWithTimeout() {
  // A port nobody listens on: bind, read the port, close.
  int port = 0;
  {
    FakeUpstream closed([](net::Socket, const std::string&) {}, 1, false);
    port = closed.port();
  }
  UpstreamTarget target;
  target.port = port;
  target.connect_timeout = std::chrono::milliseconds(1500);
  const auto began = Clock::now();
  HttpResponse response = ProxyToWorker(target, PostJson("{}"));
  const auto took = Clock::now() - began;
  assert(response.status == 502 && !response.streaming_body);
  assert(response.body.find("upstream_unavailable") != std::string::npos);
  assert(took < std::chrono::milliseconds(3000));
}

void TestConnectTimeoutIsBounded() {
  // A listener that never accepts, with its backlog filled: Linux drops the
  // next SYN (connect hangs until the bound), Windows refuses it. Either
  // way the proxy answers 502 within the connect bound plus slack.
  FakeUpstream silent([](net::Socket, const std::string&) {}, 1, false);
  std::vector<net::Socket> fillers;
  for (int i = 0; i < 8; ++i) {
    const net::Socket filler = net::CreateStreamSocket(AF_INET);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(silent.port()));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    net::SetNonBlocking(filler);
    (void)::connect(filler, reinterpret_cast<const sockaddr*>(&address),
                    sizeof(address));
    fillers.push_back(filler);
  }
  UpstreamTarget target;
  target.port = silent.port();
  target.connect_timeout = std::chrono::milliseconds(700);
  // A kernel that still completes the handshake leaves the request waiting
  // for a head; the head bound ends that case the same way.
  target.head_timeout = std::chrono::milliseconds(700);
  const auto began = Clock::now();
  HttpResponse response = ProxyToWorker(target, PostJson("{}"));
  const auto took = Clock::now() - began;
  for (const net::Socket filler : fillers) {
    net::CloseSocket(filler);
  }
  assert(response.status == 502 && !response.streaming_body);
  assert(took < std::chrono::milliseconds(700 + 700 + 1500));
}

void TestHeadTimeout() {
  std::atomic<bool> release{false};
  FakeUpstream upstream([&release](net::Socket, const std::string&) {
    // Reads the request, then never answers until the test is done.
    while (!release) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
  UpstreamTarget target;
  target.port = upstream.port();
  target.connect_timeout = std::chrono::milliseconds(1000);
  target.head_timeout = std::chrono::milliseconds(400);
  const auto began = Clock::now();
  HttpResponse response = ProxyToWorker(target, PostJson("{}"));
  const auto took = Clock::now() - began;
  release = true;
  assert(response.status == 502 && !response.streaming_body);
  assert(response.log_details.find("no response head") != std::string::npos);
  assert(took >= std::chrono::milliseconds(350));
  assert(took < std::chrono::milliseconds(2500));
}

void TestHeadTimeoutDoesNotCutTheBody() {
  // The head deadline ends once the head arrived: a slow body still streams.
  FakeUpstream upstream([](net::Socket client, const std::string&) {
    SendText(client,
             "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
             "Transfer-Encoding: chunked\r\n\r\n");
    for (int i = 0; i < 3; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      SendText(client, "7\r\ndata: x\r\n");
    }
    SendText(client, "0\r\n\r\n");
  });
  UpstreamTarget target;
  target.port = upstream.port();
  target.head_timeout = std::chrono::milliseconds(200);
  HttpResponse response = ProxyToWorker(target, PostJson("{}"));
  assert(response.status == 200 && response.streaming_body);
  assert(Drain(response) == "data: xdata: xdata: x");
}

}  // namespace

int main() {
  net::EnsureSocketRuntimeStarted();  // WSAStartup on Windows
  TestExtraHeadersReplaceClientAuthorization();
  TestDefaultTargetSendsNoExtras();
  TestConnectRefusedWithTimeout();
  TestConnectTimeoutIsBounded();
  TestHeadTimeout();
  TestHeadTimeoutDoesNotCutTheBody();
  std::cout << "All router remote proxy tests passed.\n";
  return 0;
}
