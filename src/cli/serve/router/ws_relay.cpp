#include "src/cli/serve/router/ws_relay.hpp"

#include "src/core/platform/net.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/audio_stream.hpp"

namespace gufo::router {
namespace {

using gufo::server::EncodeAudioBase64;
using gufo::server::WebSocket;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

/// Upper bound for the worker to answer the handshake with 101, mirroring the
/// HTTP relay's header budget.
constexpr Milliseconds kHandshakeBudget{std::chrono::seconds(30)};
constexpr std::size_t kMaximumHandshakeHead = 1024 * 1024;
/// Cancellation granularity while pumping; data still wakes `poll` at once.
constexpr int kPumpSliceMs = 50;
constexpr std::size_t kPumpChunk = 16384;
constexpr std::string_view kGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

bool IEquals(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
        std::tolower(static_cast<unsigned char>(rhs[i]))) {
      return false;
    }
  }
  return true;
}

std::string Trim(std::string value) {
  const auto not_space = [](unsigned char c) { return c != ' ' && c != '\t'; };
  while (!value.empty() &&
         !not_space(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  std::size_t begin = 0;
  while (begin < value.size() &&
         !not_space(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  return value.substr(begin);
}

int ConnectTo(const UpstreamTarget& target, std::string* error) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  const std::string service = std::to_string(target.port);
  addrinfo* results = nullptr;
  const int rc =
      ::getaddrinfo(target.host.c_str(), service.c_str(), &hints, &results);
  if (rc != 0 || results == nullptr) {
    *error = "resolve failed";
    return -1;
  }
  int fd = -1;
  for (addrinfo* info = results; info != nullptr; info = info->ai_next) {
    const net::Socket socket = net::CreateStreamSocket(info->ai_family);
    if (socket == net::kInvalidSocket) {
      continue;
    }
    if (::connect(socket, info->ai_addr,
                  static_cast<net::SocklenT>(info->ai_addrlen)) == 0) {
      fd = static_cast<int>(socket);
      break;
    }
    net::CloseSocket(socket);
  }
  const int connect_error = net::LastSocketError();
  ::freeaddrinfo(results);
  if (fd < 0) {
    *error = "connect failed (socket error " + std::to_string(connect_error) +
             ")";
  }
  return fd;
}

/// A fresh 16-byte nonce, base64 encoded as the client handshake key.
std::string FreshKey() {
  std::array<unsigned char, 16> bytes{};
  std::random_device rng;
  for (unsigned char& byte : bytes) {
    byte = static_cast<unsigned char>(rng() & 255U);
  }
  return EncodeAudioBase64(std::string_view(
      reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

std::string AcceptFor(std::string_view key) {
  const std::string input = std::string(key) + std::string(kGuid);
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned size = 0;
  if (EVP_Digest(input.data(), input.size(), digest.data(), &size, EVP_sha1(),
                 nullptr) != 1) {
    return {};
  }
  return EncodeAudioBase64(
      std::string_view(reinterpret_cast<const char*>(digest.data()), size));
}

bool SendAll(int fd, std::string_view payload) {
  while (!payload.empty()) {
    const auto count = net::SendNoSignal(fd, payload.data(), payload.size());
    if (count <= 0) {
      return false;
    }
    payload.remove_prefix(static_cast<std::size_t>(count));
  }
  return true;
}

/// The pinned interface carries no client request, so the upgrade targets the
/// worker root; the worker owns routing from the protocol onward.
std::string BuildHandshake(const UpstreamTarget& target, std::string_view key) {
  return "GET / HTTP/1.1\r\n"
         "Host: " +
         target.host + ":" + std::to_string(target.port) +
         "\r\nUpgrade: websocket\r\n"
         "Connection: Upgrade\r\n"
         "Sec-WebSocket-Version: 13\r\n"
         "Sec-WebSocket-Key: " +
         std::string(key) + "\r\n\r\n";
}

/// Status line plus headers, bounded in time and size like the HTTP relay.
/// Bytes a worker pipelined after the terminating blank line are handed back
/// in `remainder` instead of discarded, mirroring the HTTP relay.
bool ReadHead(int fd, std::string* head, std::string* remainder,
              std::string* error) {
  const auto deadline = Clock::now() + kHandshakeBudget;
  std::size_t found = std::string::npos;
  while (found == std::string::npos) {
    const auto remaining =
        std::chrono::duration_cast<Milliseconds>(deadline - Clock::now())
            .count();
    if (remaining <= 0) {
      *error = "timed out waiting for the handshake response";
      return false;
    }
    pollfd slot{static_cast<decltype(pollfd::fd)>(fd), POLLIN, 0};
    const int ready = net::PollSocket(&slot, 1, static_cast<int>(remaining));
    if (ready < 0) {
      if (net::IsSocketInterrupted()) {
        continue;
      }
      *error = "poll failed (socket error " +
               std::to_string(net::LastSocketError()) + ")";
      return false;
    }
    if (ready == 0) {
      *error = "timed out waiting for the handshake response";
      return false;
    }
    char buffer[4096];
    const auto count = net::Receive(fd, buffer, sizeof(buffer), 0);
    if (count == 0) {
      *error = "worker closed before the handshake response";
      return false;
    }
    if (count < 0) {
      if (net::IsSocketInterrupted() || net::IsSocketWouldBlockOrTimeout()) {
        continue;
      }
      *error = "receive failed (socket error " +
               std::to_string(net::LastSocketError()) + ")";
      return false;
    }
    head->append(buffer, static_cast<std::size_t>(count));
    if (head->size() > kMaximumHandshakeHead) {
      *error = "handshake response headers too large";
      return false;
    }
    found = head->find("\r\n\r\n");
  }
  *remainder = head->substr(found + 4);
  head->resize(found + 4);
  return true;
}

/// Case-insensitive header value from a raw response head.
std::string HeadValue(std::string_view head, std::string_view name) {
  std::size_t position = head.find("\r\n");
  if (position == std::string_view::npos) {
    return {};
  }
  position += 2;
  while (position < head.size()) {
    auto end = head.find("\r\n", position);
    if (end == std::string_view::npos) {
      end = head.size();
    }
    const std::string_view line = head.substr(position, end - position);
    if (line.empty()) {
      break;
    }
    const auto colon = line.find(':');
    if (colon != std::string_view::npos &&
        IEquals(line.substr(0, colon), name)) {
      return Trim(std::string(line.substr(colon + 1)));
    }
    position = end + 2;
  }
  return {};
}

int HandshakeStatus(std::string_view head) {
  if (head.compare(0, 5, "HTTP/") != 0) {
    return -1;
  }
  const auto space = head.find(' ');
  if (space == std::string_view::npos) {
    return -1;
  }
  const auto code = head.substr(space + 1, 3);
  for (const char c : code) {
    if (std::isdigit(static_cast<unsigned char>(c)) == 0) {
      return -1;
    }
  }
  return std::atoi(std::string(code).c_str());
}

#if defined(_WIN32)
// Winsock has no per-call MSG_DONTWAIT. Drain and Flush run only after poll
// reported the socket readable or writable, so a plain call returns at once
// for reads; a write may block up to the socket's send timeout.
constexpr int kDontWait = 0;
#else
constexpr int kDontWait = MSG_DONTWAIT;
#endif

/// One non-blocking send attempt of `buffer` without SIGPIPE.
std::intptr_t SendSome(int fd, const std::string& buffer) {
#if defined(_WIN32)
  return ::send(static_cast<net::Socket>(fd), buffer.data(),
                static_cast<int>(std::min<std::size_t>(buffer.size(), 1U << 30)),
                0);
#else
  return ::send(fd, buffer.data(), buffer.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
#endif
}

/// Move everything the socket offers into `buffer`. False once the peer is
/// gone; the sockets are non-blocking, so a quiet socket returns true.
bool Drain(int fd, std::string* buffer) {
  char raw[kPumpChunk];
  const auto count = net::Receive(fd, raw, sizeof(raw), kDontWait);
  if (count > 0) {
    buffer->append(raw, static_cast<std::size_t>(count));
    return true;
  }
  if (count == 0) {
    return false;
  }
  return net::IsSocketInterrupted() || net::IsSocketWouldBlockOrTimeout();
}

/// Push `buffer` out, keeping whatever a busy socket did not accept for the
/// next POLLOUT. False only when the socket is dead.
bool Flush(int fd, std::string* buffer) {
  while (!buffer->empty()) {
    const auto count = SendSome(fd, *buffer);
    if (count > 0) {
      buffer->erase(0, static_cast<std::size_t>(count));
      continue;
    }
    if (net::IsSocketInterrupted()) {
      continue;
    }
    return net::IsSocketWouldBlockOrTimeout();
  }
  return true;
}

/// One raw direction of the relay: bytes read from `from` wait in `buffer`
/// until the peer takes them.
struct Direction {
  int from;
  int to;
  std::string buffer;
};

/// Push worker bytes that arrived with the 101 head to the client before the
/// pump starts, honouring cancellation with the same poll slice as the pump.
/// False once the front is gone.
bool WriteToFront(int front_fd, WebSocket& front, std::string_view bytes) {
  std::string buffer(bytes);
  while (!buffer.empty()) {
    if (front.cancelled()) {
      return false;
    }
    pollfd slot{static_cast<decltype(pollfd::fd)>(front_fd), POLLOUT, 0};
    const int ready = net::PollSocket(&slot, 1, kPumpSliceMs);
    if (ready < 0) {
      if (net::IsSocketInterrupted()) {
        continue;
      }
      return false;
    }
    if (ready == 0) {
      continue;
    }
    if (!Flush(front_fd, &buffer)) {
      return false;
    }
  }
  return true;
}

/// Raw bidirectional pump. Returns when either side closes, either socket
/// errors, or the front reports cancellation.
void Pump(int front_fd, int worker_fd, WebSocket& front, std::string pending) {
  Direction to_worker{front_fd, worker_fd, std::move(pending)};
  Direction to_front{worker_fd, front_fd, {}};
  for (;;) {
    if (front.cancelled()) {
      break;
    }
    // A direction reads no further chunk until its own buffer has drained:
    // that is the backpressure bound instead of an unbounded queue.
    std::array<pollfd, 2> slots{pollfd{static_cast<decltype(pollfd::fd)>(worker_fd), 0, 0},
                                pollfd{static_cast<decltype(pollfd::fd)>(front_fd), 0, 0}};
    if (to_front.buffer.empty()) {
      slots[0].events |= POLLIN;  // worker -> front
    }
    if (!to_worker.buffer.empty()) {
      slots[0].events |= POLLOUT;  // front -> worker, waiting for the worker
    }
    if (to_worker.buffer.empty()) {
      slots[1].events |= POLLIN;  // front -> worker
    }
    if (!to_front.buffer.empty()) {
      slots[1].events |= POLLOUT;  // worker -> front, waiting for the client
    }
    const int ready = net::PollSocket(
        slots.data(), static_cast<std::uint32_t>(slots.size()), kPumpSliceMs);
    if (ready < 0) {
      if (net::IsSocketInterrupted()) {
        continue;
      }
      break;
    }
    bool finished = (slots[0].revents & (POLLERR | POLLNVAL)) != 0 ||
                    (slots[1].revents & (POLLERR | POLLNVAL)) != 0;
    if (!finished && (slots[0].revents & POLLIN) != 0) {
      finished = !Drain(worker_fd, &to_front.buffer);
    }
    if (!finished && (slots[0].revents & POLLOUT) != 0) {
      finished = !Flush(worker_fd, &to_worker.buffer);
    }
    if (!finished && (slots[1].revents & POLLIN) != 0) {
      finished = !Drain(front_fd, &to_worker.buffer);
    }
    if (!finished && (slots[1].revents & POLLOUT) != 0) {
      finished = !Flush(front_fd, &to_front.buffer);
    }
    if (finished) {
      break;
    }
  }
  // Whatever was already read still belongs to the other side.
  (void)Flush(to_front.to, &to_front.buffer);
  (void)Flush(to_worker.to, &to_worker.buffer);
}

}  // namespace

bool RelayWebSocket(const UpstreamTarget& target,
                    gufo::server::WebSocket& front, std::string* error) {
  std::string scratch;
  std::string& failure = error != nullptr ? *error : scratch;
  failure.clear();

  // Stop the front frame reader before anything else so no byte is consumed
  // by the wrong side, whatever the handshake costs.
  int front_fd = -1;
  std::string pending;
  if (!front.DetachForRelay(&front_fd, &pending)) {
    failure = "front connection is already closed";
    return false;
  }

  const int worker_fd = ConnectTo(target, &failure);
  if (worker_fd < 0) {
    return false;
  }
  if (!net::SetSendTimeout(worker_fd,
                           static_cast<int>(kHandshakeBudget.count()))) {
    failure = "cannot set the worker send timeout";
    net::CloseSocket(worker_fd);
    return false;
  }
  const std::string key = FreshKey();
  if (!SendAll(worker_fd, BuildHandshake(target, key))) {
    failure = "failed to send the websocket handshake";
    net::CloseSocket(worker_fd);
    return false;
  }
  std::string head;
  std::string remainder;
  if (!ReadHead(worker_fd, &head, &remainder, &failure)) {
    net::CloseSocket(worker_fd);
    return false;
  }
  if (HandshakeStatus(head) != 101) {
    failure = "worker answered the upgrade with status " +
              std::to_string(HandshakeStatus(head));
    net::CloseSocket(worker_fd);
    return false;
  }
  const std::string expected = AcceptFor(key);
  const std::string offered = HeadValue(head, "Sec-WebSocket-Accept");
  if (expected.empty() || offered != expected) {
    failure = "worker sent a wrong Sec-WebSocket-Accept";
    net::CloseSocket(worker_fd);
    return false;
  }

  // Bytes a worker pipelined with its 101 head belong to the client and must
  // reach the front before the raw pump, exactly like the HTTP remainder.
  if (!WriteToFront(front_fd, front, remainder)) {
    net::CloseSocket(worker_fd);
    front.MarkClosed();
    return true;
  }

  // Bytes that arrived with the upgrade request belong to the worker stream.
  Pump(front_fd, worker_fd, front, std::move(pending));
  net::CloseSocket(worker_fd);
  front.MarkClosed();
  return true;
}

}  // namespace gufo::router
