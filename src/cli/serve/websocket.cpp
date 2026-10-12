#include "src/cli/serve/websocket.hpp"

#include <openssl/evp.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>

#include "src/cli/serve/audio_stream.hpp"
#include "src/core/platform/net.hpp"
#include "src/core/utf8.hpp"

#if defined(_WIN32)
constexpr int kShutdownBoth = SD_BOTH;
using PollDescriptor = WSAPOLLFD;
constexpr short kPollIn = POLLRDNORM;
#else
constexpr int kShutdownBoth = SHUT_RDWR;
using PollDescriptor = struct pollfd;
constexpr short kPollIn = POLLIN;
#endif

namespace gufo::server {
namespace {
constexpr std::size_t kMaximumMessage = 4U << 20;
constexpr std::size_t kMaximumQueue = 8U << 20;

bool ValidUtf8(std::string_view text) {
  core::Utf8Decoder decoder;
  return decoder.Push(text, true) == text;
}

bool HasToken(std::string value, std::string_view token) {
  for (char& c : value)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (std::size_t begin = 0; begin < value.size();) {
    auto end = value.find(',', begin);
    if (end == std::string::npos)
      end = value.size();
    auto item = std::string_view(value).substr(begin, end - begin);
    while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
      item.remove_prefix(1);
    while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
      item.remove_suffix(1);
    if (item == token)
      return true;
    begin = end + 1;
  }
  return false;
}
}  // namespace

bool IsWebSocketUpgrade(const HttpRequest& request) {
  return HasToken(request.header("Upgrade"), "websocket");
}

HttpResponse UpgradeWebSocket(const HttpRequest& request,
                              std::function<void(WebSocket&)> handler) {
  std::string decoded;
  const auto key = request.header("Sec-WebSocket-Key");
  if (request.method != "GET" || !IsWebSocketUpgrade(request) ||
      !HasToken(request.header("Connection"), "upgrade") ||
      request.header("Sec-WebSocket-Version") != "13" ||
      (!request.header("Content-Length").empty() &&
       request.header("Content-Length") != "0") ||
      !DecodeAudioBase64(key, 16, &decoded) || decoded.size() != 16) {
    return {
        .status = 400,
        .reason = "Bad Request",
        .body =
            R"({"error":{"message":"invalid WebSocket handshake","code":"invalid_websocket"}})"};
  }
  const auto input = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned size = 0;
  if (EVP_Digest(input.data(), input.size(), digest.data(), &size, EVP_sha1(),
                 nullptr) != 1)
    throw std::runtime_error("WebSocket handshake digest failed");
  HttpResponse response;
  response.status = 101;
  response.reason = "Switching Protocols";
  response.headers = {
      {"Upgrade", "websocket"},
      {"Connection", "Upgrade"},
      {"Sec-WebSocket-Accept",
       EncodeAudioBase64(std::string_view(
           reinterpret_cast<const char*>(digest.data()), size))}};
  response.websocket = std::move(handler);
  return response;
}

WebSocket::WebSocket(int fd, std::string buffered)
    : fd_(fd), buffered_(std::move(buffered)) {
  // Frame headers and audio payloads are separate writes; do not wait for a
  // delayed TCP acknowledgement before sending the payload.
  (void)net::SetNoDelay(fd_);
  // Bound backpressure from a client that stops consuming audio. Read timeout
  // stays at the server's idle timeout.
  (void)net::SetSendTimeout(fd_, 5'000);
  // A client that only listens to streamed output is not idle while the
  // stream keeps sending; keep it past the receive timeout.
  const auto recv_timeout = net::SocketRecvTimeoutMs(fd_);
  if (recv_timeout >= 0) {
    idle_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::milliseconds(recv_timeout));
  }
  last_send_ = std::chrono::steady_clock::now().time_since_epoch().count();
  reader_ = std::jthread([this] { ReadLoop(); });
}

WebSocket::~WebSocket() {
  Close();
  // Every path that marks the socket closed also shuts the socket down
  // (Close's drain or the send-failure path), which unblocks a reader parked
  // in recv. An extra read-side shutdown here must not be added: on Windows
  // SD_RECEIVE with unread data aborts the connection with RST and destroys
  // the close frame this destructor's Close just queued.
  reader_.join();
  (void)::shutdown(fd_, kShutdownBoth);
}

bool WebSocket::Read(char* data, std::size_t size) {
  while (size != 0 && !closed_.load()) {
    if (offset_ < buffered_.size()) {
      const auto count = std::min(size, buffered_.size() - offset_);
      std::memcpy(data, buffered_.data() + offset_, count);
      offset_ += count;
      data += count;
      size -= count;
      continue;
    }
    const auto count = ::recv(fd_, data, static_cast<int>(size), 0);
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0 &&
#if defined(_WIN32)
        (WSAGetLastError() == WSAETIMEDOUT ||
         WSAGetLastError() == WSAEWOULDBLOCK)) {
#else
        (errno == EAGAIN || errno == EWOULDBLOCK)) {
#endif
      // A client that only listens to streamed output is not idle.
      const std::chrono::steady_clock::duration since_send(
          std::chrono::steady_clock::now().time_since_epoch().count() -
          last_send_.load());
      if (since_send < idle_)
        continue;
      Close(1001);
      return false;
    }
    if (count <= 0)
      return false;
    data += count;
    size -= static_cast<std::size_t>(count);
  }
  return size == 0;
}

bool WebSocket::Send(std::uint8_t opcode, std::string_view bytes) {
  const std::lock_guard lock(send_mutex_);
  if (closed_.load() && opcode != 8)
    return false;
  std::string header(1, static_cast<char>(0x80 | opcode));
  if (bytes.size() < 126) {
    header.push_back(static_cast<char>(bytes.size()));
  } else {
    const auto count = bytes.size() <= 65535 ? 2U : 8U;
    header.push_back(static_cast<char>(count == 2 ? 126 : 127));
    const auto length = static_cast<std::uint64_t>(bytes.size());
    for (unsigned i = count; i != 0; --i)
      header.push_back(static_cast<char>((length >> ((i - 1) * 8)) & 255));
  }
  for (auto part : {std::string_view(header), bytes}) {
    while (!part.empty()) {
      const auto count = net::SendNoSignal(fd_, part.data(), part.size());
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0) {
        MarkClosed();
        (void)::shutdown(fd_, kShutdownBoth);
        return false;
      }
      part.remove_prefix(static_cast<std::size_t>(count));
      last_send_ = std::chrono::steady_clock::now().time_since_epoch().count();
    }
  }
  return true;
}

bool WebSocket::SendText(std::string_view text) {
  return Send(1, text);
}
bool WebSocket::SendBinary(std::string_view bytes) {
  return Send(2, bytes);
}

void WebSocket::Close(std::uint16_t code) {
  if (!MarkClosed())
    return;
  const std::array<char, 2> payload{static_cast<char>(code >> 8),
                                    static_cast<char>(code & 255)};
  (void)Send(8, std::string_view(payload.data(), payload.size()));
  // A frame rejected mid-parse leaves its unread remainder queued, and
  // closing a socket with unread data makes Windows abort with RST, which
  // discards the just-sent close frame on the peer before it is read. Drain
  // briefly so the peer sees the close frame followed by a clean FIN.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  char sink[512];
  while (std::chrono::steady_clock::now() < deadline) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count();
    if (remaining <= 0)
      break;
    PollDescriptor descriptor{};
    descriptor.fd = static_cast<decltype(descriptor.fd)>(fd_);
    descriptor.events = kPollIn;
    if (net::PollSocket(&descriptor, 1, static_cast<int>(remaining)) <= 0)
      break;
    if (::recv(fd_, sink, sizeof(sink), 0) <= 0)
      break;
  }
  (void)::shutdown(fd_, kShutdownBoth);
}

bool WebSocket::MarkClosed() {
  bool changed;
  {
    // Receive checks this predicate under queue_mutex_. Changing it under the
    // same lock prevents a close notification racing between its check and
    // wait.
    const std::lock_guard lock(queue_mutex_);
    changed = !closed_.exchange(true);
  }
  ready_.notify_all();
  return changed;
}

bool WebSocket::Receive(std::string* text) {
  std::unique_lock lock(queue_mutex_);
  ready_.wait(lock, [&] { return closed_.load() || !messages_.empty(); });
  if (closed_.load())
    return false;
  *text = std::move(messages_.front());
  messages_.pop_front();
  queued_bytes_ -= text->size();
  return true;
}

void WebSocket::ReadLoop() {
  try {
    std::string message;
    bool fragmented = false;
    while (!closed_.load()) {
      std::array<unsigned char, 2> header{};
      if (!Read(reinterpret_cast<char*>(header.data()), 2))
        break;
      const bool final = (header[0] & 128) != 0;
      const auto opcode = header[0] & 15;
      const bool control = opcode >= 8;
      if ((header[0] & 112) != 0 || (header[1] & 128) == 0 ||
          (opcode != 0 && opcode != 1 && opcode != 8 && opcode != 9 &&
           opcode != 10) ||
          (control && !final)) {
        Close(1002);
        return;
      }
      std::uint64_t length = header[1] & 127;
      if (length >= 126) {
        if (control) {
          Close(1002);
          return;
        }
        const unsigned count = length == 126 ? 2 : 8;
        std::array<unsigned char, 8> extended{};
        if (!Read(reinterpret_cast<char*>(extended.data()), count))
          break;
        length = 0;
        for (unsigned i = 0; i < count; ++i)
          length = (length << 8) | extended[i];
        if ((count == 8 && (extended[0] & 128) != 0) ||
            (count == 2 && length < 126) || (count == 8 && length <= 65535)) {
          Close(1002);
          return;
        }
      }
      if (length > kMaximumMessage ||
          (!control && length > kMaximumMessage - message.size())) {
        Close(1009);
        return;
      }
      std::array<unsigned char, 4> mask{};
      if (!Read(reinterpret_cast<char*>(mask.data()), mask.size()))
        break;
      std::string payload(static_cast<std::size_t>(length), '\0');
      if (!Read(payload.data(), payload.size()))
        break;
      for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<char>(static_cast<unsigned char>(payload[i]) ^
                                       mask[i % 4]);
      if (opcode == 8) {
        if (payload.size() == 1) {
          Close(1002);
          return;
        }
        if (payload.size() >= 2) {
          const unsigned code = (static_cast<unsigned char>(payload[0]) << 8) |
                                static_cast<unsigned char>(payload[1]);
          if (code < 1000 || code >= 5000 || code == 1004 || code == 1005 ||
              code == 1006 || code == 1015 || (code >= 1016 && code < 3000)) {
            Close(1002);
            return;
          }
          if (!ValidUtf8(std::string_view(payload).substr(2))) {
            Close(1007);
            return;
          }
        }
        if (MarkClosed())
          (void)Send(8, payload);
        break;
      }
      if (opcode == 9) {
        if (!Send(10, payload))
          break;
        continue;
      }
      if (opcode == 10)
        continue;
      if ((opcode == 0) != fragmented) {
        Close(1002);
        return;
      }
      message.append(payload);
      fragmented = !final;
      if (!final)
        continue;
      if (!ValidUtf8(message)) {
        Close(1007);
        return;
      }
      {
        std::unique_lock lock(queue_mutex_);
        if (messages_.size() >= 128 ||
            message.size() > kMaximumQueue - queued_bytes_) {
          lock.unlock();
          Close(1009);
          return;
        }
        queued_bytes_ += message.size();
        messages_.push_back(std::move(message));
      }
      message.clear();
      ready_.notify_one();
    }
  } catch (const std::exception&) {
    Close(1011);
  }
  MarkClosed();
}

}  // namespace gufo::server
