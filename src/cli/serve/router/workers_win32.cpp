// Windows workers: CreateProcessW children in one kill-on-close job, so they
// die with the router (PR_SET_PDEATHSIG on Linux), stdout/stderr pumped
// through anonymous pipes, and the same loopback /health probe.
//
// Stop() terminates without a grace period: Windows has no SIGTERM for a
// windowless child, the router stops only idle workers, and process exit
// returns the worker's GPU memory to the driver.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/workers.hpp"

namespace gufo::router {
namespace {

constexpr std::chrono::milliseconds kProbeBudget{300};

using Clock = std::chrono::steady_clock;

std::string LastErrorText(DWORD code) {
  char* text = nullptr;
  const DWORD length = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, 0, reinterpret_cast<char*>(&text), 0, nullptr);
  std::string message = length != 0 && text != nullptr
                            ? std::string(text, length)
                            : "error " + std::to_string(code);
  if (text != nullptr) {
    LocalFree(text);
  }
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r' ||
                              message.back() == ' ' || message.back() == '.')) {
    message.pop_back();
  }
  return message + " (" + std::to_string(code) + ")";
}

std::wstring Widen(std::string_view utf8) {
  if (utf8.empty()) {
    return {};
  }
  const int count = MultiByteToWideChar(CP_UTF8, 0, utf8.data(),
                                        static_cast<int>(utf8.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(count), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                      wide.data(), count);
  return wide;
}

/// Quotes one argument for CommandLineToArgvW / the CRT's parser.
void AppendArgument(std::wstring& command, const std::wstring& argument) {
  if (!command.empty()) {
    command.push_back(L' ');
  }
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    command += argument;
    return;
  }
  command.push_back(L'"');
  std::size_t backslashes = 0;
  for (const wchar_t c : argument) {
    if (c == L'\\') {
      ++backslashes;
      continue;
    }
    if (c == L'"') {
      command.append(backslashes * 2 + 1, L'\\');
    } else {
      command.append(backslashes, L'\\');
    }
    backslashes = 0;
    command.push_back(c);
  }
  command.append(backslashes * 2, L'\\');
  command.push_back(L'"');
}

/// The router's environment without HOST/PORT/GUFO_HOST/GUFO_PORT (serve
/// prefers HOST/PORT, so an inherited one would override the assignment),
/// plus the worker's loopback address. Sorted by name as CreateProcessW
/// expects.
std::wstring WorkerEnvironment(int port) {
  std::vector<std::wstring> entries;
  if (wchar_t* block = GetEnvironmentStringsW(); block != nullptr) {
    for (const wchar_t* entry = block; *entry != L'\0';
         entry += std::wcslen(entry) + 1) {
      const std::wstring_view text(entry);
      const auto equals = text.find(L'=', 1);
      std::wstring name(text.substr(0, equals));
      for (wchar_t& c : name) {
        c = static_cast<wchar_t>(std::towupper(c));
      }
      if (name == L"HOST" || name == L"PORT" || name == L"GUFO_HOST" ||
          name == L"GUFO_PORT") {
        continue;
      }
      entries.emplace_back(text);
    }
    FreeEnvironmentStringsW(block);
  }
  entries.push_back(L"GUFO_HOST=127.0.0.1");
  entries.push_back(L"GUFO_PORT=" + std::to_wstring(port));
  std::ranges::sort(entries, [](const std::wstring& a, const std::wstring& b) {
    return _wcsicmp(a.c_str(), b.c_str()) < 0;
  });
  std::wstring block;
  for (const std::wstring& entry : entries) {
    block += entry;
    block.push_back(L'\0');
  }
  block.push_back(L'\0');
  return block;
}

/// One job for every worker: closing its last handle (the router exiting,
/// however it exits) kills them all.
HANDLE WorkerJob() {
  static const HANDLE job = [] {
    const HANDLE handle = CreateJobObjectW(nullptr, nullptr);
    if (handle != nullptr) {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
      limits.BasicLimitInformation.LimitFlags =
          JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      (void)SetInformationJobObject(handle, JobObjectExtendedLimitInformation,
                                    &limits, sizeof(limits));
    }
    return handle;
  }();
  return job;
}

bool EnsureWinsock() {
  static const bool ready = [] {
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
  }();
  return ready;
}

/// Drains one pipe, logging each complete line; a trailing partial line is
/// still logged. Exits when the child closes its end.
void PumpPipe(HANDLE pipe, const std::string& model_id) {
  std::string pending;
  std::array<char, 4096> buffer{};
  DWORD count = 0;
  while (ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size()),
                  &count, nullptr) &&
         count != 0) {
    pending.append(buffer.data(), count);
    std::size_t newline = 0;
    while ((newline = pending.find('\n')) != std::string::npos) {
      std::string line = pending.substr(0, newline);
      pending.erase(0, newline + 1);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      gufo::server::Logger::Info("router", "model=" + model_id + " " + line);
    }
  }
  if (!pending.empty() && pending.back() == '\r') {
    pending.pop_back();
  }
  if (!pending.empty()) {
    gufo::server::Logger::Info("router", "model=" + model_id + " " + pending);
  }
  CloseHandle(pipe);
}

/// GET /health on 127.0.0.1:port under one deadline. Returns the HTTP status
/// code, or -1 on any failure.
int ProbeHealth(int port, Clock::time_point deadline) {
  if (!EnsureWinsock()) {
    return -1;
  }
  const SOCKET fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (fd == INVALID_SOCKET) {
    return -1;
  }
  const auto remaining = [&] {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - Clock::now());
    return static_cast<DWORD>(std::max<long long>(left.count(), 1));
  };
  u_long non_blocking = 1;
  (void)ioctlsocket(fd, FIONBIO, &non_blocking);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<u_short>(port));
  (void)inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
  int status = -1;
  if (::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) == 0 ||
      WSAGetLastError() == WSAEWOULDBLOCK) {
    fd_set writable;
    FD_ZERO(&writable);
    FD_SET(fd, &writable);
    fd_set failed = writable;
    const DWORD wait = remaining();
    timeval timeout{static_cast<long>(wait / 1000),
                    static_cast<long>((wait % 1000) * 1000)};
    if (::select(0, nullptr, &writable, &failed, &timeout) == 1 &&
        FD_ISSET(fd, &writable)) {
      non_blocking = 0;
      (void)ioctlsocket(fd, FIONBIO, &non_blocking);
      const DWORD io_wait = remaining();
      (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&io_wait), sizeof(io_wait));
      (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                       reinterpret_cast<const char*>(&io_wait), sizeof(io_wait));
      constexpr std::string_view kRequest =
          "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
      if (::send(fd, kRequest.data(), static_cast<int>(kRequest.size()), 0) ==
          static_cast<int>(kRequest.size())) {
        std::string response;
        std::array<char, 256> buffer{};
        while (response.find("\r\n") == std::string::npos &&
               Clock::now() < deadline) {
          const int count =
              ::recv(fd, buffer.data(), static_cast<int>(buffer.size()), 0);
          if (count <= 0) {
            break;
          }
          response.append(buffer.data(), static_cast<std::size_t>(count));
        }
        // "HTTP/1.1 200 OK"
        if (response.starts_with("HTTP/1.") && response.size() >= 12) {
          status = std::atoi(response.substr(9, 3).c_str());
        }
      }
    }
  }
  ::closesocket(fd);
  return status;
}

}  // namespace

struct Worker::Impl {
  HANDLE process{nullptr};
  int port{0};
  std::string model_id;
  std::thread out_pump;
  std::thread err_pump;
  std::mutex mutex;  // guards the process handle and the flags below
  bool reaped = false;
  bool stopped = false;
  int exit_status = -1;

  /// Caches the exit code once the process has ended.
  bool Reap(DWORD wait_ms) {
    if (reaped) {
      return true;
    }
    if (WaitForSingleObject(process, wait_ms) != WAIT_OBJECT_0) {
      return false;
    }
    DWORD code = 0;
    exit_status = GetExitCodeProcess(process, &code) != 0
                      ? static_cast<int>(code)
                      : -1;
    reaped = true;
    return true;
  }
};

Worker::Worker() : impl_(std::make_unique<Impl>()) {}

std::unique_ptr<Worker> Worker::Spawn(const WorkerSpec& spec,
                                      const std::filesystem::path& exe_path,
                                      std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return nullptr;
  };

  // Test-only seam (see header comment): replaces the executable and drops
  // the "serve" + modality prefix so a stand-in can play a worker.
  const char* seam = std::getenv("GUFO_ROUTER_WORKER_EXE");
  const bool skip_mode =
      seam != nullptr && *seam != '\0' && std::filesystem::exists(seam);
  const bool external = !spec.command.empty();
  const std::filesystem::path exe =
      skip_mode ? std::filesystem::path(seam)
      : external ? std::filesystem::path(Widen(spec.command))
                 : exe_path;

  std::wstring command;
  AppendArgument(command, exe.wstring());
  if (!skip_mode && !external) {
    AppendArgument(command, L"serve");
    AppendArgument(command, Widen(spec.modality));
  }
  for (const std::string& arg : spec.args) {
    AppendArgument(command, Widen(arg));
  }
  if (external) {
    for (const std::wstring& arg :
         {std::wstring(L"--host"), std::wstring(L"127.0.0.1"),
          std::wstring(L"--port"), std::to_wstring(spec.port)}) {
      AppendArgument(command, arg);
    }
  }
  std::wstring environment = WorkerEnvironment(spec.port);

  SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE out_read = nullptr;
  HANDLE out_write = nullptr;
  HANDLE err_read = nullptr;
  HANDLE err_write = nullptr;
  const auto close_all = [&] {
    for (HANDLE* handle : {&out_read, &out_write, &err_read, &err_write}) {
      if (*handle != nullptr) {
        CloseHandle(*handle);
        *handle = nullptr;
      }
    }
  };
  if (!CreatePipe(&out_read, &out_write, &inherit, 0) ||
      !CreatePipe(&err_read, &err_write, &inherit, 0) ||
      !SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0) ||
      !SetHandleInformation(err_read, HANDLE_FLAG_INHERIT, 0)) {
    const DWORD code = GetLastError();
    close_all();
    return fail("CreatePipe failed: " + LastErrorText(code));
  }

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = INVALID_HANDLE_VALUE;
  startup.hStdOutput = out_write;
  startup.hStdError = err_write;
  PROCESS_INFORMATION process{};
  // Suspended until it is in the job, so not even its first instructions
  // can outlive a router that dies right here.
  const BOOL created = CreateProcessW(
      exe.wstring().c_str(), command.data(), nullptr, nullptr, TRUE,
      CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
      environment.data(), nullptr, &startup, &process);
  const DWORD create_error = created ? 0 : GetLastError();
  CloseHandle(out_write);
  out_write = nullptr;
  CloseHandle(err_write);
  err_write = nullptr;
  if (!created) {
    close_all();
    return fail("CreateProcess " + exe.string() +
                " failed: " + LastErrorText(create_error));
  }
  if (const HANDLE job = WorkerJob();
      job == nullptr || !AssignProcessToJobObject(job, process.hProcess)) {
    const DWORD code = GetLastError();
    TerminateProcess(process.hProcess, 1);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    close_all();
    return fail("cannot attach worker to the router job: " +
                LastErrorText(code));
  }
  ResumeThread(process.hThread);
  CloseHandle(process.hThread);

  std::unique_ptr<Worker> worker(new Worker());
  Impl& impl = *worker->impl_;
  impl.process = process.hProcess;
  impl.port = spec.port;
  impl.model_id = spec.model_id;
  impl.out_pump = std::thread(PumpPipe, out_read, std::string(impl.model_id));
  impl.err_pump = std::thread(PumpPipe, err_read, std::string(impl.model_id));
  if (error != nullptr) {
    error->clear();
  }
  return worker;
}

int Worker::port() const {
  return impl_->port;
}

const std::string& Worker::model_id() const {
  return impl_->model_id;
}

bool Worker::Healthy() const {
  return ProbeHealth(impl_->port, Clock::now() + kProbeBudget) == 200;
}

bool Worker::PollExit(int* exit_status) {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->Reap(0)) {
    return false;
  }
  if (exit_status != nullptr) {
    *exit_status = impl_->exit_status;
  }
  return true;
}

void Worker::Stop(std::chrono::seconds grace) {
  (void)grace;
  std::lock_guard lock(impl_->mutex);
  Impl& impl = *impl_;
  if (impl.stopped) {
    return;
  }
  impl.stopped = true;
  if (!impl.Reap(0)) {
    TerminateProcess(impl.process, 1);
    (void)impl.Reap(INFINITE);
  }
  // The pumps end when the child's pipe ends close with the process.
  if (impl.out_pump.joinable()) {
    impl.out_pump.join();
  }
  if (impl.err_pump.joinable()) {
    impl.err_pump.join();
  }
  CloseHandle(impl.process);
  impl.process = nullptr;
}

Worker::~Worker() {
  Stop(std::chrono::seconds(0));
}

std::filesystem::path ExecutablePath() {
  std::wstring path(MAX_PATH, L'\0');
  for (;;) {
    const DWORD length = GetModuleFileNameW(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length < path.size()) {
      path.resize(length);
      return path;
    }
    path.resize(path.size() * 2);
  }
}

int ReserveLoopbackPort(std::string* error) {
  const auto fail = [&](const std::string& what) {
    if (error != nullptr) {
      *error = what + " failed: " + LastErrorText(WSAGetLastError());
    }
    return -1;
  };
  if (!EnsureWinsock()) {
    return fail("WSAStartup");
  }
  const SOCKET fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (fd == INVALID_SOCKET) {
    return fail("socket");
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  (void)inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
  // Exclusive, so a worker's later bind cannot share the port with another
  // listener.
  BOOL exclusive = TRUE;
  (void)setsockopt(fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                   reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
  if (::bind(fd, reinterpret_cast<const sockaddr*>(&address),
             sizeof(address)) != 0) {
    const int result = fail("bind");
    ::closesocket(fd);
    return result;
  }
  int length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    const int result = fail("getsockname");
    ::closesocket(fd);
    return result;
  }
  ::closesocket(fd);
  if (error != nullptr) {
    error->clear();
  }
  return ntohs(address.sin_port);
}

}  // namespace gufo::router
