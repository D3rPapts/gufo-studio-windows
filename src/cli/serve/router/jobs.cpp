#include "src/cli/serve/router/jobs.hpp"

#include "src/core/platform/net.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#endif

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iterator>
#include <random>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::router::jobs {

namespace fs = std::filesystem;
using server::HttpRequest;
using server::HttpResponse;
using server::Logger;

struct JobQueue::Job {
  std::string id;
  /// queued, in_progress, completed, failed, cancelled
  std::string status{"queued"};
  std::string path;
  std::string content_type;
  std::string model;
  json::Value metadata;
  std::string idempotency_key;
  std::string request_hash;
  long long created_at{0};
  long long started_at{0};
  long long completed_at{0};
  /// 0..100, negative = unknown.
  double progress{-1};
  /// Live preview of the running job (/v1/previews/{id}).
  std::string preview_id;
  /// The worker's /v1/videos job id.
  std::string upstream_id;
  json::Value result;
  json::Value error;
  std::string webhook_url;
  /// "", pending, delivered, failed
  std::string webhook_state;
  int webhook_attempts{0};
  int webhook_last_status{0};
  std::string webhook_last_error;
  std::chrono::steady_clock::time_point webhook_due{};
  /// Ran once already and was put back after the router found no machine
  /// for it (remote_unavailable, pool_busy) before any result.
  bool requeued{false};
  /// PATCH {"keep": true}: the result files never expire.
  bool keep{false};
  /// When the result media were deleted (expiry); 0 = still there.
  long long expired_at{0};
  std::atomic<bool> cancel{false};
  /// The cancel came from Stop(), not from a client.
  std::atomic<bool> shutdown{false};
};

namespace {

/// A job stopped by DELETE /v1/jobs/{id} or the router's shutdown.
struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("cancelled") {}
};

/// A failed job with an HTTP status from the router's API.
struct JobFailure : std::runtime_error {
  JobFailure(std::string message, std::string code, int status)
      : std::runtime_error(std::move(message)),
        code(std::move(code)),
        status(status) {}
  std::string code;
  int status;
};

constexpr std::size_t kMaxInlineResponse = 256 * 1024;
constexpr std::size_t kMaxMetadata = 16 * 1024;
constexpr std::size_t kMaxIdempotencyKey = 255;
constexpr std::size_t kMaxWebhookUrl = 2048;

/// The router API paths a job may wrap (all POST).
constexpr std::string_view kJobPaths[] = {
    "/v1/images/generations", "/v1/images/edits",
    "/v1/videos",             "/v1/audio/speech",
    "/v1/audio/music",        "/v1/audio/transcriptions",
    "/v1/chat/completions",   "/v1/completions",
    "/v1/responses",          "/v1/messages",
    "/v1/audio/foley",        "/v1/images/upscale",
    "/v1/videos/upscale",     "/v1/audio/convert",
    "/v1/audio/separate",     "/v1/videos/concat"};

/// Answered by the job store itself (ffmpeg), not by a worker.
constexpr std::string_view kConcatPath = "/v1/videos/concat";
constexpr std::size_t kConcatMinInputs = 2;
constexpr std::size_t kConcatMaxInputs = 20;
constexpr std::size_t kConcatMaxDataUrl = 64 * 1024 * 1024;

long long NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string IsoTime(long long seconds) {
  const std::time_t time = static_cast<std::time_t>(seconds);
  std::tm parts{};
#if defined(_WIN32)
  gmtime_s(&parts, &time);
#else
  gmtime_r(&time, &parts);
#endif
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &parts);
  return text;
}

std::string NewJobId() {
  static std::mutex mutex;
  static std::mt19937_64 engine{
      std::random_device{}() ^
      static_cast<std::uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count())};
  std::uint64_t hi = 0;
  std::uint64_t lo = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    hi = engine();
    lo = engine();
  }
  char text[40];
  std::snprintf(text, sizeof(text), "job_%016llx%08llx",
                static_cast<unsigned long long>(hi),
                static_cast<unsigned long long>(lo >> 32));
  return text;
}

/// Job ids as NewJobId makes them; anything else in a path is unknown.
bool ValidJobId(std::string_view id) {
  if (id.size() < 5 || id.size() > 64 || !id.starts_with("job_"))
    return false;
  for (const char c : id.substr(4))
    if (!std::isxdigit(static_cast<unsigned char>(c)))
      return false;
  return true;
}

std::string Lower(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

bool ValidHostName(std::string_view host) {
  if (host.empty() || host.size() > 253)
    return false;
  for (const char c : host) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
                    c == '-' || c == '_';
    if (!ok)
      return false;
  }
  return true;
}

bool ParsePort(std::string_view text, int* port) {
  if (text.empty() || text.size() > 5)
    return false;
  int value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9')
      return false;
    value = value * 10 + (c - '0');
  }
  if (value < 1 || value > 65535)
    return false;
  *port = value;
  return true;
}

/// 0.0.0.0/8, 169.254.0.0/16 and 224.0.0.0/3: never reached through "*".
bool BlockedForAny(std::uint32_t address) {
  return (address >> 24) == 0 || (address >> 16) == 0xA9FE ||
         (address >> 29) == 0x7;
}

HttpResponse JsonResponse(int status, std::string_view reason,
                          const json::Value& body) {
  return HttpResponse{.status = status,
                      .reason = std::string(reason),
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"},
                                  {"Cache-Control", "no-store"}}};
}

/// The router's OpenAI error shape.
HttpResponse Err(int status, std::string_view reason, std::string_view message,
                 std::string_view code) {
  json::Value detail = json::Value::object();
  detail["message"] = std::string(message);
  detail["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  detail["code"] = std::string(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return JsonResponse(status, reason, body);
}

std::string ReadAll(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + path.string());
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void WriteAll(const fs::path& path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
}

/// The message of an API error body, or the start of the body.
std::string ErrorMessage(const FetchResult& result) {
  try {
    const json::Value parsed = json::parse(result.body);
    if (const json::Value* error = parsed.find("error")) {
      if (error->is_string())
        return error->str();
      const std::string message = error->member_str("message");
      if (!message.empty())
        return message;
    }
  } catch (const std::exception&) {
  }
  return result.body.substr(0, 500);
}

std::string ErrorCode(const FetchResult& result) {
  try {
    const json::Value parsed = json::parse(result.body);
    if (const json::Value* error = parsed.find("error");
        error != nullptr && error->is_object()) {
      const std::string code = error->member_str("code");
      if (!code.empty())
        return code;
    }
  } catch (const std::exception&) {
  }
  return "upstream_error";
}

std::string MediaType(std::string_view content_type) {
  const auto semicolon = content_type.find(';');
  std::string type = Lower(content_type.substr(0, semicolon));
  while (!type.empty() && type.back() == ' ')
    type.pop_back();
  return type;
}

std::string ExtensionFor(std::string_view content_type) {
  const std::string type = MediaType(content_type);
  if (type == "image/png")
    return "png";
  if (type == "image/jpeg")
    return "jpg";
  if (type == "image/webp")
    return "webp";
  if (type == "audio/wav" || type == "audio/x-wav" || type == "audio/wave")
    return "wav";
  if (type == "audio/mpeg")
    return "mp3";
  if (type == "audio/mp4" || type == "audio/aac")
    return "m4a";
  if (type == "audio/flac")
    return "flac";
  if (type == "audio/ogg")
    return "ogg";
  if (type == "audio/opus")
    return "opus";
  if (type == "video/mp4")
    return "mp4";
  if (type == "text/vtt")
    return "vtt";
  if (type == "application/x-subrip")
    return "srt";
  if (type == "application/json")
    return "json";
  if (type.starts_with("text/"))
    return "txt";
  return "bin";
}

std::string ContentTypeFor(std::string_view name) {
  const auto dot = name.rfind('.');
  const std::string ext =
      Lower(dot == std::string_view::npos ? "" : name.substr(dot + 1));
  if (ext == "png")
    return "image/png";
  if (ext == "jpg")
    return "image/jpeg";
  if (ext == "webp")
    return "image/webp";
  if (ext == "wav")
    return "audio/wav";
  if (ext == "mp3")
    return "audio/mpeg";
  if (ext == "m4a")
    return "audio/mp4";
  if (ext == "flac")
    return "audio/flac";
  if (ext == "ogg")
    return "audio/ogg";
  if (ext == "opus")
    return "audio/opus";
  if (ext == "mp4")
    return "video/mp4";
  if (ext == "vtt")
    return "text/vtt; charset=utf-8";
  if (ext == "srt")
    return "application/x-subrip; charset=utf-8";
  if (ext == "json")
    return "application/json";
  if (ext == "txt")
    return "text/plain; charset=utf-8";
  return "application/octet-stream";
}

std::string SniffImageExtension(std::string_view bytes) {
  if (bytes.starts_with("\x89PNG"))
    return "png";
  if (bytes.starts_with("\xFF\xD8"))
    return "jpg";
  if (bytes.size() >= 12 && bytes.starts_with("RIFF") &&
      bytes.substr(8, 4) == "WEBP")
    return "webp";
  return "bin";
}

bool IsJsonType(std::string_view content_type) {
  const std::string type = MediaType(content_type);
  return type == "application/json" || type.ends_with("+json");
}

bool Finished(std::string_view status) {
  return status == "completed" || status == "failed" ||
         status == "cancelled" || status == "expired";
}

/// The small text files the expiry keeps; everything else in a job folder
/// is result media.
bool IsKeptFile(std::string_view name) {
  return name == "request.body" || name == "response.json";
}

/// Result names are ours ([a-z0-9._-]); request.body is never served.
bool ValidFileName(std::string_view name) {
  return !name.empty() && name.size() <= 64 && name != "request.body" &&
         name.front() != '.' &&
         std::all_of(name.begin(), name.end(), [](char c) {
           return std::islower(static_cast<unsigned char>(c)) ||
                  std::isdigit(static_cast<unsigned char>(c)) || c == '.' ||
                  c == '-' || c == '_';
         });
}

/// The last `limit` bytes of `text`, cut at a line start, for error texts.
std::string TailOf(std::string_view text, std::size_t limit) {
  while (!text.empty() &&
         (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
    text.remove_suffix(1);
  if (text.size() > limit) {
    text.remove_prefix(text.size() - limit);
    if (const auto nl = text.find('\n'); nl != std::string_view::npos)
      text.remove_prefix(nl + 1);
  }
  return std::string(text);
}

#if defined(_WIN32)
/// Appends one argument to a CreateProcess command line (MS quoting rules).
void AppendArgumentW(std::wstring& command, const std::wstring& arg) {
  if (!command.empty())
    command.push_back(L' ');
  const bool plain =
      !arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos;
  if (plain) {
    command += arg;
    return;
  }
  command.push_back(L'"');
  std::size_t backslashes = 0;
  for (const wchar_t c : arg) {
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

HANDLE OpenInheritable(const fs::path& path, bool write) {
  SECURITY_ATTRIBUTES attributes{};
  attributes.nLength = sizeof(attributes);
  attributes.bInheritHandle = TRUE;
  return CreateFileW(path.wstring().c_str(),
                     write ? GENERIC_WRITE : GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes,
                     write ? CREATE_ALWAYS : OPEN_EXISTING,
                     FILE_ATTRIBUTE_NORMAL, nullptr);
}

/// Runs `exe` with `args` (never through a shell), stdout and stderr into
/// the two files, stdin from NUL. Exit code, or -1 with *error when the
/// process could not start, timed out or was cancelled.
int RunCaptured(const fs::path& exe, const std::vector<fs::path>& args,
                std::chrono::minutes timeout, const fs::path& stdout_file,
                const fs::path& stderr_file, std::atomic<bool>* cancel,
                std::string* error) {
  std::wstring command;
  AppendArgumentW(command, exe.wstring());
  for (const fs::path& arg : args)
    AppendArgumentW(command, arg.wstring());
  HANDLE in = OpenInheritable("NUL", false);
  HANDLE out = OpenInheritable(stdout_file, true);
  HANDLE err = OpenInheritable(stderr_file, true);
  const auto close_all = [&] {
    for (HANDLE handle : {in, out, err})
      if (handle != INVALID_HANDLE_VALUE)
        CloseHandle(handle);
  };
  if (out == INVALID_HANDLE_VALUE || err == INVALID_HANDLE_VALUE) {
    close_all();
    *error = "cannot open the output files for " + exe.filename().string();
    return -1;
  }
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = in;
  startup.hStdOutput = out;
  startup.hStdError = err;
  PROCESS_INFORMATION process{};
  const BOOL started =
      CreateProcessW(exe.wstring().c_str(), command.data(), nullptr, nullptr,
                     TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                     &process);
  const DWORD start_error = GetLastError();
  close_all();
  if (!started) {
    *error = "cannot start " + exe.string() + " (error " +
             std::to_string(start_error) + ")";
    return -1;
  }
  CloseHandle(process.hThread);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const DWORD waited = WaitForSingleObject(process.hProcess, 200);
    if (waited == WAIT_OBJECT_0)
      break;
    const bool cancelled = cancel != nullptr && cancel->load();
    if (cancelled || std::chrono::steady_clock::now() > deadline) {
      TerminateProcess(process.hProcess, 1);
      WaitForSingleObject(process.hProcess, 5000);
      CloseHandle(process.hProcess);
      *error = cancelled ? "cancelled" : exe.filename().string() + " timed out";
      return -1;
    }
  }
  DWORD code = 1;
  GetExitCodeProcess(process.hProcess, &code);
  CloseHandle(process.hProcess);
  return static_cast<int>(code);
}
#else
int RunCaptured(const fs::path& exe, const std::vector<fs::path>& args,
                std::chrono::minutes timeout, const fs::path& stdout_file,
                const fs::path& stderr_file, std::atomic<bool>* cancel,
                std::string* error) {
  std::vector<std::string> storage{exe.string()};
  for (const fs::path& arg : args)
    storage.push_back(arg.string());
  std::vector<char*> argv;
  for (std::string& item : storage)
    argv.push_back(item.data());
  argv.push_back(nullptr);
  const int out = open(stdout_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  const int err = open(stderr_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  const int in = open("/dev/null", O_RDONLY);
  if (out < 0 || err < 0 || in < 0) {
    for (const int fd : {out, err, in})
      if (fd >= 0)
        close(fd);
    *error = "cannot open the output files for " + exe.filename().string();
    return -1;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    close(out);
    close(err);
    close(in);
    *error = "fork failed";
    return -1;
  }
  if (pid == 0) {
    dup2(in, STDIN_FILENO);
    dup2(out, STDOUT_FILENO);
    dup2(err, STDERR_FILENO);
    execv(argv[0], argv.data());
    _exit(127);
  }
  close(out);
  close(err);
  close(in);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    int status = 0;
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      if (WIFEXITED(status))
        return WEXITSTATUS(status);
      *error = exe.filename().string() + " was killed";
      return -1;
    }
    const bool cancelled = cancel != nullptr && cancel->load();
    if (cancelled || std::chrono::steady_clock::now() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      *error = cancelled ? "cancelled" : exe.filename().string() + " timed out";
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}
#endif

std::string ReadIfThere(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return "";
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

/// Removes the concat work folder however the work ends.
struct TempDir {
  fs::path path;
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

/// ffprobe prints numbers as strings ("48000") or as numbers; both count.
double NumberOf(const json::Value& object, const std::string& key) {
  const json::Value* value = object.find(key);
  if (value == nullptr)
    return 0;
  if (value->is_number())
    return value->as_double();
  if (value->is_string()) {
    const std::string& text = value->str();
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    return end != text.c_str() && std::isfinite(parsed) ? parsed : 0;
  }
  return 0;
}

std::string DurationKey(std::string_view path, std::string_view model) {
  return std::string(path) + " " + std::string(model);
}

bool AllDigits(std::string_view text) {
  return !text.empty() && std::all_of(text.begin(), text.end(), [](char c) {
    return c >= '0' && c <= '9';
  });
}

/// GET /v1/jobs and DELETE /v1/jobs filters from the query; false with
/// *error for a value that cannot be read.
bool ParseFilter(const HttpRequest& request, Filter* filter,
                 std::string* error) {
  filter->status = request.query_param("status");
  filter->q = Lower(request.query_param("q"));
  filter->seite = request.query_param("seite");
  filter->geraet = request.query_param("geraet");
  filter->kind = Lower(request.query_param("kind"));
  if (filter->kind == "bild")
    filter->kind = "image";
  else if (filter->kind == "musik")
    filter->kind = "music";
  else if (filter->kind == "stimme")
    filter->kind = "speech";
  else if (filter->kind == "sonstiges")
    filter->kind = "other";
  if (!filter->kind.empty() && filter->kind != "image" &&
      filter->kind != "music" && filter->kind != "speech" &&
      filter->kind != "video" && filter->kind != "other") {
    *error = "kind must be one of image, music, speech, video, other";
    return false;
  }
  const long long now = NowSeconds();
  for (const std::string_view name : {"since", "before", "older_than"}) {
    const std::string text = request.query_param(std::string(name));
    if (text.empty())
      continue;
    const auto value = ParseCutoff(text, now);
    if (!value || *value <= 0) {
      *error = std::string(name) +
               " must be unix seconds, an age in seconds or ISO 8601";
      return false;
    }
    if (name == "since")
      filter->since = *value;
    else if (name == "before")
      filter->before = *value;
    else
      filter->older_than = *value;
  }
  return true;
}

}  // namespace

std::string KindOf(std::string_view path) {
  if (path.starts_with("/v1/images"))
    return "image";
  if (path.starts_with("/v1/audio/music"))
    return "music";
  if (path.starts_with("/v1/audio/speech"))
    return "speech";
  if (path.starts_with("/v1/videos"))
    return "video";
  return "other";
}

std::optional<JobFileRef> ParseJobFileUrl(std::string_view url) {
  std::string_view path = url;
  if (const auto scheme = path.find("://"); scheme != std::string_view::npos) {
    // Only http(s) URLs carry a path we read; the host is not checked (the
    // job must exist here anyway).
    const std::string name = Lower(path.substr(0, scheme));
    if (name != "http" && name != "https")
      return std::nullopt;
    const auto slash = path.find('/', scheme + 3);
    if (slash == std::string_view::npos)
      return std::nullopt;
    path = path.substr(slash);
  }
  if (const auto cut = path.find_first_of("?#"); cut != std::string_view::npos)
    path = path.substr(0, cut);
  constexpr std::string_view kPrefix = "/v1/jobs/";
  if (!path.starts_with(kPrefix))
    return std::nullopt;
  path.remove_prefix(kPrefix.size());
  const auto slash = path.find('/');
  if (slash == std::string_view::npos)
    return std::nullopt;
  JobFileRef ref;
  ref.id = std::string(path.substr(0, slash));
  const std::string_view rest = path.substr(slash);
  constexpr std::string_view kFiles = "/files/";
  if (!rest.starts_with(kFiles))
    return std::nullopt;
  ref.name = std::string(rest.substr(kFiles.size()));
  if (!ValidJobId(ref.id) || !ValidFileName(ref.name) ||
      IsKeptFile(ref.name))
    return std::nullopt;
  return ref;
}

std::optional<std::string> ParseDataUrl(std::string_view url,
                                        std::string* media_type) {
  if (!url.starts_with("data:"))
    return std::nullopt;
  const auto comma = url.find(',');
  if (comma == std::string_view::npos)
    return std::nullopt;
  const std::string_view header = url.substr(5, comma - 5);
  bool base64 = false;
  std::string type;
  std::size_t start = 0;
  while (start <= header.size()) {
    const auto end = std::min(header.find(';', start), header.size());
    const std::string_view part = header.substr(start, end - start);
    if (start == 0)
      type = Lower(part);
    else if (part == "base64")
      base64 = true;
    start = end + 1;
  }
  if (!base64)
    return std::nullopt;
  auto bytes = comfy::Base64Decode(url.substr(comma + 1));
  if (!bytes)
    return std::nullopt;
  if (media_type != nullptr)
    *media_type = type;
  return bytes;
}

double FrameRateValue(std::string_view text) {
  const std::string copy(text);
  char* end = nullptr;
  const double numerator = std::strtod(copy.c_str(), &end);
  if (end == copy.c_str() || !std::isfinite(numerator) || numerator <= 0)
    return 0;
  if (*end == '\0')
    return numerator;
  if (*end != '/')
    return 0;
  char* end2 = nullptr;
  const double denominator = std::strtod(end + 1, &end2);
  if (end2 == end + 1 || *end2 != '\0' || !std::isfinite(denominator) ||
      denominator <= 0)
    return 0;
  return numerator / denominator;
}

bool ParseProbe(std::string_view text, ProbeInfo* out, std::string* error) {
  json::Value parsed;
  try {
    parsed = json::parse(std::string(text));
  } catch (const std::exception& failure) {
    *error = std::string("ffprobe output is not JSON: ") + failure.what();
    return false;
  }
  ProbeInfo info;
  bool video = false;
  double stream_duration = 0;
  if (const json::Value* streams = parsed.find("streams");
      streams != nullptr && streams->is_array()) {
    for (const json::Value& stream : streams->items()) {
      if (!stream.is_object())
        continue;
      const std::string type = stream.member_str("codec_type");
      if (type == "video" && !video) {
        video = true;
        info.video_codec = stream.member_str("codec_name");
        info.width = static_cast<int>(NumberOf(stream, "width"));
        info.height = static_cast<int>(NumberOf(stream, "height"));
        info.frame_rate = stream.member_str("r_frame_rate");
        if (FrameRateValue(info.frame_rate) <= 0)
          info.frame_rate = stream.member_str("avg_frame_rate");
        info.pix_fmt = stream.member_str("pix_fmt");
        stream_duration = NumberOf(stream, "duration");
      } else if (type == "audio" && !info.has_audio) {
        info.has_audio = true;
        info.audio_codec = stream.member_str("codec_name");
        info.sample_rate = static_cast<int>(NumberOf(stream, "sample_rate"));
        info.channels = static_cast<int>(NumberOf(stream, "channels"));
      }
    }
  }
  if (!video) {
    *error = "no video stream";
    return false;
  }
  if (info.width <= 0 || info.height <= 0 ||
      FrameRateValue(info.frame_rate) <= 0) {
    *error = "video stream without size or frame rate";
    return false;
  }
  if (const json::Value* format = parsed.find("format");
      format != nullptr && format->is_object())
    info.duration = NumberOf(*format, "duration");
  if (info.duration <= 0)
    info.duration = stream_duration;
  *out = std::move(info);
  return true;
}

std::string ConcatMode(const std::vector<ProbeInfo>& inputs, bool reencode) {
  if (reencode || inputs.empty())
    return "reencode";
  const ProbeInfo& first = inputs.front();
  const double fps = FrameRateValue(first.frame_rate);
  for (const ProbeInfo& input : inputs) {
    const bool same_video =
        input.video_codec == first.video_codec && input.width == first.width &&
        input.height == first.height && input.pix_fmt == first.pix_fmt &&
        std::fabs(FrameRateValue(input.frame_rate) - fps) < 1e-3;
    const bool same_audio =
        input.has_audio == first.has_audio &&
        (!first.has_audio || (input.audio_codec == first.audio_codec &&
                              input.sample_rate == first.sample_rate &&
                              input.channels == first.channels));
    if (!same_video || !same_audio)
      return "reencode";
  }
  return "copy";
}

std::optional<long long> ParseCutoff(std::string_view text, long long now) {
  while (!text.empty() && text.front() == ' ')
    text.remove_prefix(1);
  while (!text.empty() && text.back() == ' ')
    text.remove_suffix(1);
  if (text.empty() || text.size() > 32)
    return std::nullopt;
  if (AllDigits(text)) {
    const long long value =
        std::strtoll(std::string(text).c_str(), nullptr, 10);
    // Ages are far below any unix time of this century (10^8 s = 3 years).
    return value < 100000000LL ? now - value : value;
  }
  // ISO 8601: YYYY-MM-DD[(T| )HH:MM[:SS]][Z]
  std::tm parts{};
  int consumed = 0;
  if (std::sscanf(std::string(text).c_str(), "%4d-%2d-%2d%n", &parts.tm_year,
                  &parts.tm_mon, &parts.tm_mday, &consumed) != 3 ||
      consumed != 10)
    return std::nullopt;
  std::string_view rest = text.substr(10);
  if (!rest.empty() && (rest.front() == 'T' || rest.front() == ' ')) {
    rest.remove_prefix(1);
    const int fields =
        std::sscanf(std::string(rest).c_str(), "%2d:%2d:%2d%n",
                    &parts.tm_hour, &parts.tm_min, &parts.tm_sec, &consumed);
    if (fields < 2)
      return std::nullopt;
    if (fields == 2) {
      parts.tm_sec = 0;
      consumed = 5;
    }
    if (rest.size() < static_cast<std::size_t>(consumed))
      return std::nullopt;
    rest.remove_prefix(static_cast<std::size_t>(consumed));
  }
  if (rest == "Z")
    rest = {};
  if (!rest.empty())
    return std::nullopt;
  if (parts.tm_year < 1970 || parts.tm_mon < 1 || parts.tm_mon > 12 ||
      parts.tm_mday < 1 || parts.tm_mday > 31 || parts.tm_hour > 23 ||
      parts.tm_min > 59 || parts.tm_sec > 60)
    return std::nullopt;
  parts.tm_year -= 1900;
  parts.tm_mon -= 1;
#if defined(_WIN32)
  const std::time_t seconds = _mkgmtime(&parts);
#else
  const std::time_t seconds = timegm(&parts);
#endif
  if (seconds < 0)
    return std::nullopt;
  return static_cast<long long>(seconds);
}

// ---------------------------------------------------------------- targets

bool ParseIpv4(std::string_view text, std::uint32_t* out) {
  std::uint32_t address = 0;
  int parts = 0;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = std::min(text.find('.', start), text.size());
    const std::string_view part = text.substr(start, end - start);
    if (part.empty() || part.size() > 3 || ++parts > 4)
      return false;
    unsigned value = 0;
    for (const char c : part) {
      if (c < '0' || c > '9')
        return false;
      value = value * 10 + static_cast<unsigned>(c - '0');
    }
    if (value > 255)
      return false;
    address = (address << 8) | value;
    start = end + 1;
    if (end == text.size())
      break;
  }
  if (parts != 4)
    return false;
  *out = address;
  return true;
}

std::string FormatIpv4(std::uint32_t address) {
  return std::to_string(address >> 24) + "." +
         std::to_string((address >> 16) & 0xFF) + "." +
         std::to_string((address >> 8) & 0xFF) + "." +
         std::to_string(address & 0xFF);
}

bool ParseAllowList(std::string_view text, std::vector<AllowEntry>* out,
                    std::string* error) {
  out->clear();
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = std::min(text.find(',', start), text.size());
    std::string_view item = text.substr(start, end - start);
    start = end + 1;
    while (!item.empty() && item.front() == ' ')
      item.remove_prefix(1);
    while (!item.empty() && item.back() == ' ')
      item.remove_suffix(1);
    if (item.empty()) {
      if (end == text.size())
        break;
      continue;
    }
    AllowEntry entry;
    std::string_view base = item;
    if (const auto colon = item.rfind(':'); colon != std::string_view::npos) {
      if (!ParsePort(item.substr(colon + 1), &entry.port)) {
        *error = "bad port in webhook allow entry '" + std::string(item) + "'";
        return false;
      }
      base = item.substr(0, colon);
    }
    if (base == "*") {
      entry.kind = AllowEntry::Kind::kAny;
    } else if (const auto slash = base.find('/');
               slash != std::string_view::npos) {
      std::uint32_t network = 0;
      int prefix = -1;
      const std::string_view bits = base.substr(slash + 1);
      if (!bits.empty() && bits.size() <= 2 &&
          std::all_of(bits.begin(), bits.end(),
                      [](char c) { return c >= '0' && c <= '9'; }))
        prefix = std::stoi(std::string(bits));
      if (!ParseIpv4(base.substr(0, slash), &network) || prefix < 0 ||
          prefix > 32) {
        *error = "bad CIDR range in webhook allow entry '" + std::string(item) +
                 "' (want a.b.c.d/n)";
        return false;
      }
      entry.kind = AllowEntry::Kind::kNetwork;
      entry.mask = prefix == 0 ? 0 : ~std::uint32_t{0} << (32 - prefix);
      entry.network = network & entry.mask;
    } else if (std::uint32_t address = 0; ParseIpv4(base, &address)) {
      entry.kind = AllowEntry::Kind::kNetwork;
      entry.mask = ~std::uint32_t{0};
      entry.network = address;
    } else if (ValidHostName(base)) {
      entry.kind = AllowEntry::Kind::kHost;
      entry.host = Lower(base);
    } else {
      *error = "bad webhook allow entry '" + std::string(item) +
               "' (want a host name, an IPv4 address, a.b.c.d/n or *)";
      return false;
    }
    out->push_back(std::move(entry));
    if (end == text.size())
      break;
  }
  return true;
}

bool ParseWebhookUrl(std::string_view url, WebhookUrl* out,
                     std::string* error) {
  if (url.size() > kMaxWebhookUrl) {
    *error = "webhook_url is too long";
    return false;
  }
  for (const char c : url) {
    if (static_cast<unsigned char>(c) <= 0x20 || c == 0x7F) {
      *error = "webhook_url must not contain spaces or control characters";
      return false;
    }
  }
  const std::string lower = Lower(url.substr(0, 8));
  if (lower.starts_with("https://")) {
    *error =
        "https webhooks are not supported yet (gufo has no TLS client); use "
        "an http address in the LAN";
    return false;
  }
  if (!lower.starts_with("http://")) {
    *error = "webhook_url must be an http:// address";
    return false;
  }
  std::string_view rest = url.substr(7);
  if (const auto hash = rest.find('#'); hash != std::string_view::npos)
    rest = rest.substr(0, hash);
  const auto authority_end = rest.find_first_of("/?");
  const std::string_view authority = rest.substr(0, authority_end);
  std::string target(authority_end == std::string_view::npos
                         ? std::string_view("/")
                         : rest.substr(authority_end));
  if (target.starts_with("?"))
    target = "/" + target;
  if (authority.find('@') != std::string_view::npos) {
    *error = "webhook_url must not contain user info";
    return false;
  }
  if (authority.starts_with("[")) {
    *error = "IPv6 webhook targets are not supported";
    return false;
  }
  WebhookUrl parsed;
  std::string_view host = authority;
  if (const auto colon = authority.rfind(':');
      colon != std::string_view::npos) {
    if (!ParsePort(authority.substr(colon + 1), &parsed.port)) {
      *error = "bad port in webhook_url";
      return false;
    }
    host = authority.substr(0, colon);
  }
  if (!ValidHostName(host)) {
    *error = "bad host in webhook_url";
    return false;
  }
  parsed.host = Lower(host);
  parsed.target = std::move(target);
  *out = std::move(parsed);
  return true;
}

std::vector<std::uint32_t> ResolveIpv4(const std::string& host) {
  std::vector<std::uint32_t> addresses;
  if (std::uint32_t literal = 0; ParseIpv4(host, &literal)) {
    addresses.push_back(literal);
    return addresses;
  }
  net::EnsureSocketRuntimeStarted();
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* results = nullptr;
  if (::getaddrinfo(host.c_str(), nullptr, &hints, &results) != 0 ||
      results == nullptr)
    return addresses;
  for (addrinfo* info = results; info != nullptr; info = info->ai_next) {
    if (info->ai_family != AF_INET || info->ai_addr == nullptr)
      continue;
    const auto* in = reinterpret_cast<const sockaddr_in*>(info->ai_addr);
    const std::uint32_t address = ntohl(in->sin_addr.s_addr);
    if (std::find(addresses.begin(), addresses.end(), address) ==
        addresses.end())
      addresses.push_back(address);
  }
  ::freeaddrinfo(results);
  return addresses;
}

std::optional<std::uint32_t> CheckWebhookTarget(
    const WebhookUrl& url, const std::vector<AllowEntry>& allow,
    const Resolver& resolve, std::string* error) {
  if (allow.empty()) {
    *error =
        "webhooks are off on this router (start it with "
        "--jobs-webhook-allow)";
    return std::nullopt;
  }
  const auto port_ok = [&](const AllowEntry& entry) {
    return entry.port == 0 || entry.port == url.port;
  };
  std::vector<std::uint32_t> addresses;
  bool resolved = false;
  const auto addresses_of = [&]() -> const std::vector<std::uint32_t>& {
    if (!resolved) {
      addresses = resolve ? resolve(url.host) : ResolveIpv4(url.host);
      resolved = true;
    }
    return addresses;
  };
  const std::string where = url.host + ":" + std::to_string(url.port);
  // A named host is trusted as named, whatever it resolves to.
  for (const AllowEntry& entry : allow) {
    if (entry.kind == AllowEntry::Kind::kHost && entry.host == url.host &&
        port_ok(entry)) {
      if (addresses_of().empty()) {
        *error = "cannot resolve webhook host " + url.host;
        return std::nullopt;
      }
      return addresses_of().front();
    }
  }
  if (addresses_of().empty()) {
    *error = "cannot resolve webhook host " + url.host;
    return std::nullopt;
  }
  for (const std::uint32_t address : addresses_of())
    for (const AllowEntry& entry : allow)
      if (entry.kind == AllowEntry::Kind::kNetwork && port_ok(entry) &&
          (address & entry.mask) == entry.network)
        return address;
  for (const std::uint32_t address : addresses_of())
    for (const AllowEntry& entry : allow)
      if (entry.kind == AllowEntry::Kind::kAny && port_ok(entry) &&
          !BlockedForAny(address))
        return address;
  *error = "webhook target " + where +
           " is not allowed on this router (--jobs-webhook-allow)";
  return std::nullopt;
}

std::string SignWebhook(std::string_view secret, std::string_view id,
                        long long timestamp, std::string_view body) {
  std::string key(secret);
  if (secret.starts_with("whsec_")) {
    if (auto decoded = comfy::Base64Decode(secret.substr(6)))
      key = std::move(*decoded);
  }
  const std::string message = std::string(id) + "." +
                              std::to_string(timestamp) + "." +
                              std::string(body);
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
       reinterpret_cast<const unsigned char*>(message.data()), message.size(),
       digest, &length);
  return "v1," + comfy::Base64Encode(std::string_view(
                     reinterpret_cast<const char*>(digest), length));
}

int DeliverHttp(std::uint32_t address, const WebhookUrl& url,
                const HttpRequest& request, std::chrono::milliseconds timeout,
                std::string* error) {
  UpstreamTarget target;
  target.host = FormatIpv4(address);
  target.port = url.port;
  HttpRequest post = request;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  post.is_cancelled = [deadline] {
    return std::chrono::steady_clock::now() >= deadline;
  };
  HttpResponse response = ProxyToWorker(target, post);
  constexpr std::string_view kUnavailable = "upstream_unavailable detail=";
  if (response.log_details.starts_with("upstream_unavailable")) {
    *error = std::chrono::steady_clock::now() >= deadline
                 ? "no answer within " +
                       std::to_string(timeout.count() / 1000) + " s"
             : response.log_details.starts_with(kUnavailable)
                 ? response.log_details.substr(kUnavailable.size())
                 : response.log_details;
    return 0;
  }
  if (response.streaming_body) {
    // The answer body is not needed; read it so the connection ends cleanly.
    response.streaming_body([](std::string_view) { return true; });
  }
  return response.status;
}

// ---------------------------------------------------------------- queue

JobQueue::JobQueue(Options options) : options_(std::move(options)) {
  if (options_.parallel == 0)
    options_.parallel = 1;
  if (!options_.resolve)
    options_.resolve = ResolveIpv4;
  if (!options_.deliver)
    options_.deliver = DeliverHttp;
  fs::create_directories(options_.root);
  next_expiry_ = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  Load();
}

JobQueue::~JobQueue() {
  Stop();
}

void JobQueue::Start() {
  for (std::size_t i = 0; i < options_.parallel; ++i)
    runners_.emplace_back([this] { RunnerMain(); });
  webhook_thread_ = std::thread([this] { WebhookMain(); });
}

void JobQueue::Stop() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ && runners_.empty() && !webhook_thread_.joinable())
      return;
    stopping_ = true;
    for (const auto& job : order_) {
      if (job->status == "in_progress") {
        job->shutdown.store(true);
        job->cancel.store(true);
      }
    }
  }
  wake_.notify_all();
  webhook_wake_.notify_all();
  for (auto& runner : runners_)
    if (runner.joinable())
      runner.join();
  runners_.clear();
  if (webhook_thread_.joinable())
    webhook_thread_.join();
  const std::lock_guard<std::mutex> lock(mutex_);
  PersistLocked();
}

fs::path JobQueue::JobDir(const std::string& id) const {
  return options_.root / id;
}

std::optional<HttpResponse> JobQueue::Handle(const HttpRequest& request) {
  const std::string_view path = request.path;
  const std::string& method = request.method;
  if (path == kConcatPath) {
    if (method != "POST")
      return Err(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
    try {
      return Concat(request);
    } catch (const std::exception& error) {
      Logger::Warn("jobs", std::string("event=request_failed path=") +
                               request.path + " error=" + error.what());
      return Err(500, "Internal Server Error", error.what(), "concat_failed");
    }
  }
  if (path != "/v1/jobs" && !path.starts_with("/v1/jobs/"))
    return std::nullopt;
  try {
    if (path == "/v1/jobs") {
      if (method == "POST")
        return Submit(request);
      if (method == "GET")
        return List(request);
      if (method == "DELETE")
        return BulkDelete(request);
      return Err(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
    }
    std::string_view rest = path.substr(std::string_view("/v1/jobs/").size());
    const auto slash = rest.find('/');
    const std::string id(rest.substr(0, slash));
    const std::string_view tail =
        slash == std::string_view::npos ? "" : rest.substr(slash);
    if (!ValidJobId(id))
      return Err(404, "Not Found", "job not found", "job_not_found");
    if (tail.empty()) {
      if (method == "GET")
        return Get(id);
      if (method == "DELETE")
        return Delete(id);
      if (method == "PATCH")
        return Keep(id, request.body);
      return Err(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
    }
    if (tail == "/keep") {
      if (method != "POST")
        return Err(405, "Method Not Allowed", "method is not allowed",
                   "method_not_allowed");
      return Keep(id, request.body);
    }
    if (tail == "/cancel") {
      if (method != "POST")
        return Err(405, "Method Not Allowed", "method is not allowed",
                   "method_not_allowed");
      return Cancel(id);
    }
    if (tail.starts_with("/files/")) {
      if (method != "GET")
        return Err(405, "Method Not Allowed", "method is not allowed",
                   "method_not_allowed");
      return File(id, std::string(tail.substr(7)));
    }
    return Err(404, "Not Found", "no route for this path", "not_found");
  } catch (const std::exception& error) {
    Logger::Warn("jobs", std::string("event=request_failed path=") +
                             request.path + " error=" + error.what());
    return Err(500, "Internal Server Error", error.what(), "server_error");
  }
}

HttpResponse JobQueue::Submit(const HttpRequest& request) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& error) {
    return Err(400, "Bad Request",
               std::string("request body must be JSON: ") + error.what(),
               "invalid_json");
  }
  const json::Value* inner = body.find("request");
  if (inner == nullptr || !inner->is_object())
    return Err(400, "Bad Request",
               "'request' must be an object {\"path\": ..., \"body\": {...}}",
               "invalid_request");
  const std::string method = inner->member_str("method", "POST");
  const std::string path = inner->member_str("path");
  if (method != "POST")
    return Err(400, "Bad Request", "jobs wrap POST requests only",
               "invalid_request");
  if (std::find(std::begin(kJobPaths), std::end(kJobPaths), path) ==
      std::end(kJobPaths)) {
    std::string paths;
    for (const std::string_view allowed : kJobPaths)
      paths += (paths.empty() ? "" : ", ") + std::string(allowed);
    return Err(400, "Bad Request",
               "request.path '" + path + "' cannot run as a job; one of: " +
                   paths,
               "invalid_path");
  }
  auto job = std::make_shared<Job>();
  job->path = path;
  std::string payload;
  const json::Value* json_body = inner->find("body");
  const json::Value* raw_body = inner->find("body_base64");
  if ((json_body != nullptr) == (raw_body != nullptr))
    return Err(400, "Bad Request",
               "request needs exactly one of 'body' (JSON object) or "
               "'body_base64' with 'content_type' (e.g. multipart)",
               "invalid_request");
  if (json_body != nullptr) {
    if (!json_body->is_object())
      return Err(400, "Bad Request", "request.body must be a JSON object",
                 "invalid_request");
    if (const json::Value* stream = json_body->find("stream");
        stream != nullptr && stream->as_bool())
      return Err(400, "Bad Request",
                 "streaming responses cannot run as a job; drop 'stream'",
                 "stream_unsupported");
    job->content_type = "application/json";
    job->model = json_body->member_str("model");
    payload = json_body->dump();
  } else {
    job->content_type = inner->member_str("content_type");
    auto decoded = raw_body->is_string()
                       ? comfy::Base64Decode(raw_body->str())
                       : std::nullopt;
    if (!decoded || job->content_type.empty())
      return Err(400, "Bad Request",
                 "request.body_base64 must be base64 and needs "
                 "request.content_type",
                 "invalid_request");
    payload = std::move(*decoded);
    if (MediaType(job->content_type) == "multipart/form-data") {
      std::vector<comfy::MultipartPart> parts;
      std::string error;
      if (!comfy::ParseMultipart(job->content_type, payload, &parts, &error))
        return Err(400, "Bad Request", "bad multipart body: " + error,
                   "invalid_request");
      for (const auto& part : parts) {
        if (part.name == "model")
          job->model = part.data;
        if (part.name == "stream" && (part.data == "true" || part.data == "1"))
          return Err(400, "Bad Request",
                     "streaming responses cannot run as a job; drop 'stream'",
                     "stream_unsupported");
      }
    } else if (IsJsonType(job->content_type)) {
      try {
        job->model = json::parse(payload).member_str("model");
      } catch (const std::exception&) {
        return Err(400, "Bad Request", "request body is not valid JSON",
                   "invalid_json");
      }
    }
  }
  // The video concat runs on the router itself (ffmpeg), without a model.
  const bool needs_model = job->path != kConcatPath;
  if (needs_model && job->model.empty())
    return Err(400, "Bad Request", "missing 'model' in the wrapped request",
               "missing_model");
  if (needs_model && !options_.models.empty() &&
      std::find(options_.models.begin(), options_.models.end(), job->model) ==
          options_.models.end())
    return Err(404, "Not Found", "unknown model '" + job->model + "'",
               "model_not_found");
  if (const json::Value* metadata = body.find("metadata");
      metadata != nullptr && !metadata->is_null()) {
    if (!metadata->is_object() || metadata->dump().size() > kMaxMetadata)
      return Err(400, "Bad Request",
                 "metadata must be a JSON object of at most 16 KiB",
                 "invalid_metadata");
    job->metadata = *metadata;
  }
  if (const json::Value* webhook = body.find("webhook_url");
      webhook != nullptr && !webhook->is_null()) {
    WebhookUrl url;
    std::string error;
    if (!webhook->is_string() ||
        !ParseWebhookUrl(webhook->str(), &url, &error) ||
        !CheckWebhookTarget(url, options_.webhook_allow, options_.resolve,
                            &error))
      return Err(400, "Bad Request",
                 error.empty() ? "webhook_url must be a string" : error,
                 "webhook_not_allowed");
    job->webhook_url = webhook->str();
  }
  job->idempotency_key = request.header("Idempotency-Key");
  if (job->idempotency_key.empty())
    job->idempotency_key = body.member_str("idempotency_key");
  if (job->idempotency_key.size() > kMaxIdempotencyKey)
    return Err(400, "Bad Request", "Idempotency-Key is longer than 255",
               "invalid_idempotency_key");
  {
    json::Value fingerprint = json::Value::object();
    fingerprint["path"] = job->path;
    fingerprint["content_type"] = job->content_type;
    fingerprint["body"] = crypto::Sha256Hex(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(payload.data()),
        payload.size()));
    fingerprint["webhook_url"] = job->webhook_url;
    const std::string text = fingerprint.dump();
    job->request_hash = crypto::Sha256Hex(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (stopping_)
    return Err(503, "Service Unavailable", "router is shutting down",
               "shutting_down");
  if (!job->idempotency_key.empty()) {
    if (const auto known = idempotency_.find(job->idempotency_key);
        known != idempotency_.end()) {
      const auto& first = jobs_.at(known->second);
      if (first->request_hash != job->request_hash)
        return Err(409, "Conflict",
                   "Idempotency-Key was used for a different request",
                   "idempotency_key_reused");
      HttpResponse replay = JsonResponse(200, "OK", ViewLocked(*first));
      replay.headers.emplace_back("Idempotent-Replayed", "true");
      return replay;
    }
  }
  do {
    job->id = NewJobId();
  } while (jobs_.contains(job->id));
  job->created_at = NowSeconds();
  WriteAll(JobDir(job->id) / "request.body", payload);
  order_.push_back(job);
  jobs_[job->id] = job;
  if (!job->idempotency_key.empty())
    idempotency_[job->idempotency_key] = job->id;
  pending_.push_back(job);
  PersistLocked();
  const json::Value view = ViewLocked(*job);
  lock.unlock();
  wake_.notify_all();
  Logger::Info("jobs", "event=job_queued id=" + job->id + " path=" + job->path +
                           " model=" + job->model +
                           (job->webhook_url.empty() ? "" : " webhook=yes"));
  HttpResponse accepted = JsonResponse(202, "Accepted", view);
  accepted.headers.emplace_back("Location", "/v1/jobs/" + job->id);
  return accepted;
}

HttpResponse JobQueue::List(const HttpRequest& request) {
  std::size_t limit = 100;
  if (const std::string text = request.query_param("limit"); !text.empty())
    limit = std::clamp<std::size_t>(std::strtoull(text.c_str(), nullptr, 10),
                                    1, 1000);
  Filter filter;
  std::string error;
  if (!ParseFilter(request, &filter, &error))
    return Err(400, "Bad Request", error, "invalid_filter");
  json::Value data = json::Value::array();
  const std::lock_guard<std::mutex> lock(mutex_);
  // Newest first: order_ is the submit order.
  for (auto it = order_.rbegin(); it != order_.rend() && data.size() < limit;
       ++it) {
    if (!MatchesLocked(**it, filter))
      continue;
    data.push_back(ViewLocked(**it));
  }
  json::Value body = json::Value::object();
  body["object"] = "list";
  body["data"] = std::move(data);
  return JsonResponse(200, "OK", body);
}

bool JobQueue::MatchesLocked(const Job& job, const Filter& filter) const {
  if (!filter.status.empty() && job.status != filter.status)
    return false;
  if (!filter.kind.empty() && KindOf(job.path) != filter.kind)
    return false;
  if (!filter.seite.empty() &&
      job.metadata.member_str("seite") != filter.seite)
    return false;
  if (!filter.geraet.empty() &&
      job.metadata.member_str("geraet") != filter.geraet)
    return false;
  if (filter.since > 0 && job.created_at < filter.since)
    return false;
  if (filter.before > 0 && job.created_at >= filter.before)
    return false;
  if (filter.older_than > 0) {
    const long long at =
        job.completed_at > 0 ? job.completed_at : job.created_at;
    if (at >= filter.older_than)
      return false;
  }
  if (!filter.q.empty()) {
    const std::string text =
        Lower(job.metadata.member_str("titel") + "\n" +
              job.metadata.member_str("prompt") + "\n" + job.model + "\n" +
              job.id);
    if (text.find(filter.q) == std::string::npos)
      return false;
  }
  return true;
}

HttpResponse JobQueue::Get(const std::string& id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = jobs_.find(id);
  if (found == jobs_.end())
    return Err(404, "Not Found", "job not found", "job_not_found");
  return JsonResponse(200, "OK", ViewLocked(*found->second));
}

HttpResponse JobQueue::Cancel(const std::string& id) {
  std::shared_ptr<Job> job;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end())
      return Err(404, "Not Found", "job not found", "job_not_found");
    job = found->second;
    if (job->status == "queued") {
      std::erase(pending_, job);
      lock.unlock();
      Finish(job, "cancelled", json::Value(), json::Value());
      return Get(id);
    }
    if (job->status != "in_progress")
      return Err(409, "Conflict", "job already " + job->status,
                 "job_finished");
    job->cancel.store(true);
  }
  Logger::Info("jobs", "event=job_cancel id=" + id);
  // The runner notices within a poll or two; answer with the end state when
  // it comes quickly, else with the job as it is now (still in_progress).
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < until) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (job->status != "in_progress")
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return Get(id);
}

HttpResponse JobQueue::Keep(const std::string& id, const std::string& body) {
  bool keep = true;
  if (!body.empty()) {
    json::Value parsed;
    try {
      parsed = json::parse(body);
    } catch (const std::exception& error) {
      return Err(400, "Bad Request",
                 std::string("request body must be JSON: ") + error.what(),
                 "invalid_json");
    }
    const json::Value* flag = parsed.find("keep");
    if (!parsed.is_object() || flag == nullptr || !flag->is_bool())
      return Err(400, "Bad Request", "body must be {\"keep\": true|false}",
                 "invalid_request");
    keep = flag->as_bool();
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = jobs_.find(id);
  if (found == jobs_.end())
    return Err(404, "Not Found", "job not found", "job_not_found");
  if (found->second->keep != keep) {
    found->second->keep = keep;
    PersistLocked();
    Logger::Info("jobs", "event=job_keep id=" + id +
                             " keep=" + (keep ? "true" : "false"));
  }
  return JsonResponse(200, "OK", ViewLocked(*found->second));
}

void JobQueue::ForgetLocked(const std::shared_ptr<Job>& job) {
  jobs_.erase(job->id);
  std::erase(order_, job);
  std::erase(pending_, job);
  if (!job->idempotency_key.empty()) {
    const auto key = idempotency_.find(job->idempotency_key);
    if (key != idempotency_.end() && key->second == job->id)
      idempotency_.erase(key);
  }
}

HttpResponse JobQueue::Delete(const std::string& id) {
  std::shared_ptr<Job> job;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end())
      return Err(404, "Not Found", "job not found", "job_not_found");
    job = found->second;
    if (!Finished(job->status)) {
      job = nullptr;
    } else {
      ForgetLocked(job);
      PersistLocked();
    }
  }
  if (job == nullptr)
    return Cancel(id);  // queued or running: DELETE cancels, as before
  std::error_code ec;
  fs::remove_all(JobDir(id), ec);
  if (ec)
    Logger::Warn("jobs", "event=job_delete_files_failed id=" + id +
                             " error=" + ec.message());
  Logger::Info("jobs",
               "event=job_deleted id=" + id + " status=" + job->status);
  json::Value body = json::Value::object();
  body["id"] = id;
  body["object"] = "job";
  body["deleted"] = true;
  return JsonResponse(200, "OK", body);
}

HttpResponse JobQueue::BulkDelete(const HttpRequest& request) {
  Filter filter;
  std::string error;
  if (!ParseFilter(request, &filter, &error))
    return Err(400, "Bad Request", error, "invalid_filter");
  if (!filter.any())
    return Err(400, "Bad Request",
               "bulk delete needs at least one filter: older_than, q, seite, "
               "geraet, status, kind, since or before",
               "filter_required");
  const auto flag = [&](const char* name) {
    const std::string value = Lower(request.query_param(name));
    return value == "1" || value == "true" || value == "yes";
  };
  const bool dry_run = flag("dry_run");
  const bool include_kept = flag("include_kept");
  std::vector<std::shared_ptr<Job>> gone;
  json::Value ids = json::Value::array();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& job : order_) {
      if (!Finished(job->status) || (job->keep && !include_kept) ||
          !MatchesLocked(*job, filter))
        continue;
      gone.push_back(job);
      ids.push_back(job->id);
    }
    if (!dry_run && !gone.empty()) {
      for (const auto& job : gone)
        ForgetLocked(job);
      PersistLocked();
    }
  }
  if (!dry_run) {
    for (const auto& job : gone) {
      std::error_code ec;
      fs::remove_all(JobDir(job->id), ec);
      if (ec)
        Logger::Warn("jobs", "event=job_delete_files_failed id=" + job->id +
                                 " error=" + ec.message());
    }
    Logger::Info("jobs", "event=jobs_deleted count=" +
                             std::to_string(gone.size()) +
                             " query=" + request.query);
  }
  json::Value body = json::Value::object();
  body["object"] = "list";
  body["dry_run"] = dry_run;
  body["matched"] = static_cast<long long>(gone.size());
  body["deleted"] = static_cast<long long>(dry_run ? 0 : gone.size());
  body["ids"] = std::move(ids);
  return JsonResponse(200, "OK", body);
}

HttpResponse JobQueue::File(const std::string& id, const std::string& name) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!jobs_.contains(id))
      return Err(404, "Not Found", "job not found", "job_not_found");
  }
  const fs::path file = JobDir(id) / name;
  std::error_code ec;
  if (!ValidFileName(name) || !fs::is_regular_file(file, ec))
    return Err(404, "Not Found", "file not found", "file_not_found");
  return HttpResponse{.status = 200,
                      .reason = "OK",
                      .body = ReadAll(file),
                      .headers = {{"Content-Type", ContentTypeFor(name)},
                                  {"Cache-Control", "no-cache"}}};
}

json::Value JobQueue::ViewLocked(const Job& job) const {
  json::Value view = json::Value::object();
  view["id"] = job.id;
  view["object"] = "job";
  view["status"] = job.status;
  json::Value request = json::Value::object();
  request["method"] = "POST";
  request["path"] = job.path;
  request["model"] = job.model;
  view["request"] = std::move(request);
  view["created_at"] = job.created_at;
  view["started_at"] =
      job.started_at > 0 ? json::Value(job.started_at) : json::Value();
  view["completed_at"] =
      job.completed_at > 0 ? json::Value(job.completed_at) : json::Value();
  // Without a live source (preview step, video or music job) a running
  // job's progress is estimated from the learned duration of its path and
  // model: elapsed / expected, at most 95 until the result is stored.
  double progress = job.progress;
  bool estimated = false;
  // Video and music jobs report real progress once the worker accepted them;
  // before that they are usually waiting in the pool for a free GPU, and an
  // estimate would show "62 %" for a job that has not started (11.11.).
  const bool live_source = job.path.starts_with("/v1/videos") ||
                           job.path == "/v1/audio/music";
  if (job.status == "in_progress" && progress < 0 && job.started_at > 0 &&
      !live_source && job.preview_id.empty() && job.upstream_id.empty()) {
    const auto learned = durations_.find(DurationKey(job.path, job.model));
    if (learned != durations_.end() && learned->second > 0) {
      const double elapsed =
          static_cast<double>(std::max(0LL, NowSeconds() - job.started_at));
      progress =
          std::min(95.0, std::floor(100.0 * elapsed / learned->second));
      estimated = true;
    }
  }
  view["progress"] = progress >= 0 ? json::Value(progress) : json::Value();
  view["progress_estimated"] = estimated;
  view["keep"] = job.keep;
  view["expired_at"] =
      job.expired_at > 0 ? json::Value(job.expired_at) : json::Value();
  if (job.status == "in_progress" && !job.preview_id.empty())
    view["preview_url"] =
        "/v1/previews/" + job.preview_id +
        (job.path == "/v1/videos" ? std::string() : "?model=" + job.model);
  else if (job.status == "in_progress" && job.path == "/v1/audio/music" &&
           !job.upstream_id.empty())
    // Music previews (D3) live on the worker's own endpoint; the router
    // relays it by the music id (MusicJobId handles "/preview").
    view["preview_url"] = "/v1/audio/music/" + job.upstream_id + "/preview";
  else
    view["preview_url"] = nullptr;
  if (!job.upstream_id.empty())
    view[job.path == "/v1/audio/music" ? "music_id" : "video_id"] =
        job.upstream_id;
  view["metadata"] = job.metadata;
  view["result"] = job.result;
  view["error"] = job.error;
  if (!job.webhook_url.empty()) {
    json::Value webhook = json::Value::object();
    webhook["url"] = job.webhook_url;
    webhook["state"] = job.webhook_state.empty() ? std::string("waiting")
                                                 : job.webhook_state;
    webhook["attempts"] = job.webhook_attempts;
    webhook["last_status"] = job.webhook_last_status > 0
                                 ? json::Value(job.webhook_last_status)
                                 : json::Value();
    webhook["last_error"] = job.webhook_last_error.empty()
                                ? json::Value()
                                : json::Value(job.webhook_last_error);
    view["webhook"] = std::move(webhook);
  } else {
    view["webhook"] = nullptr;
  }
  return view;
}

json::Value JobQueue::RecordLocked(const Job& job) const {
  json::Value record = json::Value::object();
  record["id"] = job.id;
  record["status"] = job.status;
  record["path"] = job.path;
  record["content_type"] = job.content_type;
  record["model"] = job.model;
  record["metadata"] = job.metadata;
  record["idempotency_key"] = job.idempotency_key;
  record["request_hash"] = job.request_hash;
  record["created_at"] = job.created_at;
  record["started_at"] = job.started_at;
  record["completed_at"] = job.completed_at;
  record["progress"] = job.progress;
  record["upstream_id"] = job.upstream_id;
  record["result"] = job.result;
  record["error"] = job.error;
  record["webhook_url"] = job.webhook_url;
  record["webhook_state"] = job.webhook_state;
  record["webhook_attempts"] = job.webhook_attempts;
  record["webhook_last_status"] = job.webhook_last_status;
  record["webhook_last_error"] = job.webhook_last_error;
  record["keep"] = job.keep;
  record["expired_at"] = job.expired_at;
  return record;
}

void JobQueue::PersistLocked() {
  json::Value all = json::Value::array();
  for (const auto& job : order_)
    all.push_back(RecordLocked(*job));
  json::Value file = json::Value::object();
  file["jobs"] = std::move(all);
  json::Value durations = json::Value::object();
  for (const auto& [key, seconds] : durations_)
    durations[key] = seconds;
  file["durations"] = std::move(durations);
  const fs::path path = options_.root / "jobs.json";
  const fs::path temp = options_.root / "jobs.json.tmp";
  try {
    WriteAll(temp, file.dump());
    std::error_code ec;
    fs::rename(temp, path, ec);
    if (ec)
      Logger::Warn("jobs", "event=jobs_save_failed error=" + ec.message());
  } catch (const std::exception& error) {
    Logger::Warn("jobs",
                 std::string("event=jobs_save_failed error=") + error.what());
  }
}

void JobQueue::Load() {
  const fs::path path = options_.root / "jobs.json";
  std::error_code ec;
  if (!fs::is_regular_file(path, ec))
    return;
  json::Value file;
  try {
    file = json::parse(ReadAll(path));
  } catch (const std::exception& error) {
    // Keep the unreadable file for a look by hand; start with no jobs.
    Logger::Warn("jobs",
                 std::string("event=jobs_load_failed error=") + error.what());
    fs::rename(path, options_.root / "jobs.json.broken", ec);
    return;
  }
  if (const json::Value* durations = file.find("durations");
      durations != nullptr && durations->is_object()) {
    for (const auto& [key, seconds] : durations->members())
      if (seconds.is_number() && seconds.as_double() > 0)
        durations_[key] = seconds.as_double();
  }
  const json::Value* all = file.find("jobs");
  if (all == nullptr || !all->is_array())
    return;
  int requeued = 0;
  int failed = 0;
  for (const json::Value& record : all->items()) {
    auto job = std::make_shared<Job>();
    job->id = record.member_str("id");
    if (!ValidJobId(job->id) || jobs_.contains(job->id))
      continue;
    job->status = record.member_str("status", "failed");
    job->path = record.member_str("path");
    job->content_type = record.member_str("content_type");
    job->model = record.member_str("model");
    if (const json::Value* metadata = record.find("metadata"))
      job->metadata = *metadata;
    job->idempotency_key = record.member_str("idempotency_key");
    job->request_hash = record.member_str("request_hash");
    job->created_at =
        static_cast<long long>(record.member_double("created_at", 0));
    job->started_at =
        static_cast<long long>(record.member_double("started_at", 0));
    job->completed_at =
        static_cast<long long>(record.member_double("completed_at", 0));
    job->progress = record.member_double("progress", -1);
    job->upstream_id = record.member_str("upstream_id");
    if (const json::Value* result = record.find("result"))
      job->result = *result;
    if (const json::Value* error = record.find("error"))
      job->error = *error;
    job->webhook_url = record.member_str("webhook_url");
    job->webhook_state = record.member_str("webhook_state");
    job->webhook_attempts =
        static_cast<int>(record.member_double("webhook_attempts", 0));
    job->webhook_last_status =
        static_cast<int>(record.member_double("webhook_last_status", 0));
    job->webhook_last_error = record.member_str("webhook_last_error");
    if (const json::Value* keep = record.find("keep"))
      job->keep = keep->as_bool();
    job->expired_at =
        static_cast<long long>(record.member_double("expired_at", 0));
    if (job->status == "queued") {
      pending_.push_back(job);
      ++requeued;
    } else if (job->status == "in_progress") {
      // The worker request died with the old router: report it, never
      // re-run it silently (it may have side effects or cost an hour).
      job->status = "failed";
      job->completed_at = NowSeconds();
      json::Value error = json::Value::object();
      error["message"] = "the router stopped while the job ran";
      error["code"] = "router_restarted";
      job->error = std::move(error);
      if (!job->webhook_url.empty())
        job->webhook_state = "pending";
      ++failed;
    }
    order_.push_back(job);
    jobs_[job->id] = job;
    if (!job->idempotency_key.empty())
      idempotency_[job->idempotency_key] = job->id;
  }
  if (failed > 0)
    PersistLocked();
  Logger::Info("jobs", "event=jobs_loaded jobs=" +
                           std::to_string(order_.size()) +
                           " requeued=" + std::to_string(requeued) +
                           " failed_by_restart=" + std::to_string(failed));
}

// ---------------------------------------------------------------- running

void JobQueue::RunnerMain() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
      if (stopping_)
        return;
      job = pending_.front();
      pending_.pop_front();
      job->status = "in_progress";
      job->started_at = NowSeconds();
      PersistLocked();
    }
    Execute(job);
  }
}

FetchResult JobQueue::Call(const std::shared_ptr<Job>& job, std::string method,
                           std::string path, std::string query,
                           std::string body, std::string content_type) {
  HttpRequest request;
  request.method = std::move(method);
  request.path = std::move(path);
  request.query = std::move(query);
  request.body = std::move(body);
  if (!content_type.empty())
    request.headers.emplace_back("Content-Type", std::move(content_type));
  request.request_id = job->id;
  request.is_cancelled = [job] { return job->cancel.load(); };
  FetchResult result = options_.fetch(request);
  if (job->cancel.load())
    throw Cancelled();
  return result;
}

void JobQueue::PollPreview(const std::shared_ptr<Job>& job,
                           const std::string& model, std::stop_token stop) {
  double seen = -1;
  while (!stop.stop_requested()) {
    const auto until =
        std::chrono::steady_clock::now() + options_.progress_poll;
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (stop.stop_requested())
      break;
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/previews/" + job->preview_id;
    poll.query = "format=json&model=" + model;
    poll.request_id = job->id + "-progress";
    FetchResult result;
    try {
      result = options_.fetch(poll);
    } catch (const std::exception&) {
      continue;
    }
    if (result.status != 200)
      continue;
    try {
      const json::Value parsed = json::parse(result.body);
      const double step = parsed.member_double("step", 0);
      const double steps = parsed.member_double("steps", 0);
      if (steps <= 0 || step == seen)
        continue;
      seen = step;
      // 100 only once the result is stored.
      const double progress = std::min(99.0, std::floor(100.0 * step / steps));
      const std::lock_guard<std::mutex> lock(mutex_);
      if (job->status == "in_progress")
        job->progress = progress;
    } catch (const std::exception&) {
      // A malformed preview is skipped; the next poll tries again.
    }
  }
}

void JobQueue::Execute(const std::shared_ptr<Job>& job) {
  const auto started = std::chrono::steady_clock::now();
  Logger::Info("jobs", "event=job_start id=" + job->id + " path=" + job->path +
                           " model=" + job->model);
  struct DoneLog {
    const std::shared_ptr<Job>& job;
    std::mutex& mutex;
    std::chrono::steady_clock::time_point started;
    ~DoneLog() {
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      std::string status;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        status = job->status;
      }
      if (status == "queued")
        return;  // requeued: job_start comes again
      Logger::Info("jobs", "event=job_done id=" + job->id +
                               " status=" + status +
                               " seconds=" + std::to_string(seconds));
    }
  } done_log{job, mutex_, started};
  try {
    std::string payload = ReadAll(JobDir(job->id) / "request.body");
    if (job->path == kConcatPath) {
      RunConcat(job, payload);
      return;
    }
    // Image progress: the live preview's step, polled under the job id. A
    // request that brings its own preview id keeps it (and no progress).
    std::string with_preview;
    if (job->path == "/v1/images/generations" &&
        IsJsonType(job->content_type)) {
      try {
        json::Value body = json::parse(payload);
        const json::Value* gufo = body.find("gufo");
        if (gufo == nullptr ||
            (gufo->is_object() && !gufo->contains("preview"))) {
          json::Value preview = json::Value::object();
          preview["id"] = job->id;
          body["gufo"]["preview"] = std::move(preview);
          with_preview = body.dump();
        }
      } catch (const std::exception&) {
        // The worker answers the malformed body itself.
      }
    }
    // Music progress: the worker's async music job (202 + id), polled like
    // a video job. A worker that answers the audio at once is fine too.
    std::string as_async;
    if (job->path == "/v1/audio/music" && IsJsonType(job->content_type)) {
      try {
        json::Value body = json::parse(payload);
        body["async"] = true;
        as_async = body.dump();
      } catch (const std::exception&) {
        // The worker answers the malformed body itself.
      }
    }
    FetchResult result;
    if (!as_async.empty()) {
      result = Call(job, "POST", job->path, "", as_async, job->content_type);
      if (result.status == 202 && IsJsonType(result.content_type)) {
        RunMusic(job, result);
        return;
      }
    } else if (!with_preview.empty()) {
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        job->preview_id = job->id;
      }
      std::jthread progress([&](std::stop_token stop) {
        PollPreview(job, job->model, stop);
      });
      result = Call(job, "POST", job->path, "", with_preview,
                    job->content_type);
      progress.request_stop();
      progress.join();
      if (result.status == 400 &&
          ErrorMessage(result).find("preview") != std::string::npos) {
        // A worker without live previews renders as before.
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          job->preview_id.clear();
        }
        result = Call(job, "POST", job->path, "", payload, job->content_type);
      }
    } else {
      result = Call(job, "POST", job->path, "", payload, job->content_type);
    }
    if (result.status < 200 || result.status >= 300)
      throw JobFailure(ErrorMessage(result), ErrorCode(result), result.status);
    if (job->path == "/v1/videos" || job->path == "/v1/videos/upscale")
      RunVideo(job, result);  // both answer 202 with a worker video job
    else
      Finish(job, "completed", StoreResult(*job, result), json::Value());
  } catch (const Cancelled&) {
    if (job->shutdown.load()) {
      json::Value error = json::Value::object();
      error["message"] = "the router shut down while the job ran";
      error["code"] = "router_shutdown";
      Finish(job, "failed", json::Value(), std::move(error));
    } else {
      Finish(job, "cancelled", json::Value(), json::Value());
    }
  } catch (const JobFailure& failure) {
    // The router had no machine for the request (the remote router went
    // away before it answered, or the pool found none free): no work was
    // started, so the job goes back to the front of the queue once.
    if (failure.status == 503 && job->upstream_id.empty() &&
        (failure.code == "remote_unavailable" ||
         failure.code == "pool_busy")) {
      bool requeued = false;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!job->requeued && !stopping_ && !job->cancel.load()) {
          job->requeued = true;
          job->status = "queued";
          job->started_at = 0;
          job->progress = -1;
          job->preview_id.clear();
          pending_.push_front(job);
          PersistLocked();
          requeued = true;
        }
      }
      if (requeued) {
        Logger::Warn("jobs", "event=job_requeue id=" + job->id + " code=" +
                                 failure.code);
        wake_.notify_one();
        return;
      }
    }
    json::Value error = json::Value::object();
    error["message"] = std::string(failure.what());
    error["code"] = failure.code;
    error["status_code"] = failure.status;
    Finish(job, "failed", json::Value(), std::move(error));
  } catch (const std::exception& failure) {
    json::Value error = json::Value::object();
    error["message"] = std::string(failure.what());
    error["code"] = "job_error";
    Finish(job, "failed", json::Value(), std::move(error));
  }
}

void JobQueue::RunVideo(const std::shared_ptr<Job>& job,
                        const FetchResult& created) {
  std::string video_id;
  try {
    video_id = json::parse(created.body).member_str("id");
  } catch (const std::exception&) {
  }
  if (video_id.empty())
    throw JobFailure("the video response has no id", "upstream_error",
                     created.status);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    job->upstream_id = video_id;
    // The worker's video preview lives under the video job id.
    job->preview_id = video_id;
    job->progress = 0;
    PersistLocked();
  }
  json::Value final_status;
  for (;;) {
    const auto until = std::chrono::steady_clock::now() + options_.video_poll;
    while (!job->cancel.load() && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (job->cancel.load()) {
      HttpRequest cancel;
      cancel.method = "DELETE";
      cancel.path = "/v1/videos/" + video_id;
      cancel.request_id = job->id;
      try {
        (void)options_.fetch(cancel);
      } catch (const std::exception&) {
      }
      throw Cancelled();
    }
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/videos/" + video_id;
    poll.request_id = job->id;
    const FetchResult state = options_.fetch(poll);
    if (state.status != 200)
      throw JobFailure("video status failed: " + ErrorMessage(state),
                       ErrorCode(state), state.status);
    json::Value parsed = json::parse(state.body);
    const std::string status = parsed.member_str("status");
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      job->progress =
          std::min(99.0, std::max(0.0, parsed.member_double("progress", 0)));
    }
    if (status == "completed") {
      final_status = std::move(parsed);
      break;
    }
    if (status == "failed" || status == "cancelled" || status == "expired") {
      const json::Value* error = parsed.find("error");
      throw JobFailure(
          "video " + status +
              (error != nullptr && error->is_object()
                   ? ": " + error->member_str("message")
                   : std::string()),
          "video_" + status, 200);
    }
  }
  const FetchResult content =
      Call(job, "GET", "/v1/videos/" + video_id + "/content", "", "", "");
  if (content.status != 200 || content.body.empty())
    throw JobFailure("video content failed: " + ErrorMessage(content),
                     ErrorCode(content), content.status);
  const std::string name =
      "video." + (content.content_type.empty()
                      ? std::string("mp4")
                      : ExtensionFor(content.content_type));
  WriteAll(JobDir(job->id) / name, content.body);
  WriteAll(JobDir(job->id) / "response.json", final_status.dump());
  const std::string base = "/v1/jobs/" + job->id + "/files/";
  json::Value files = json::Value::array();
  json::Value video = json::Value::object();
  video["name"] = name;
  video["url"] = base + name;
  video["content_type"] = ContentTypeFor(name);
  video["bytes"] = static_cast<long long>(content.body.size());
  files.push_back(std::move(video));
  json::Value status_file = json::Value::object();
  status_file["name"] = "response.json";
  status_file["url"] = base + "response.json";
  status_file["content_type"] = "application/json";
  status_file["bytes"] = static_cast<long long>(final_status.dump().size());
  files.push_back(std::move(status_file));
  json::Value result = json::Value::object();
  result["status_code"] = created.status;
  result["content_type"] = ContentTypeFor(name);
  result["files"] = std::move(files);
  result["response"] = std::move(final_status);
  Finish(job, "completed", std::move(result), json::Value());
}

void JobQueue::RunMusic(const std::shared_ptr<Job>& job,
                        const FetchResult& created) {
  std::string music_id;
  try {
    music_id = json::parse(created.body).member_str("id");
  } catch (const std::exception&) {
  }
  if (music_id.empty())
    throw JobFailure("the music response has no id", "upstream_error",
                     created.status);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    job->upstream_id = music_id;
    job->progress = 0;
    PersistLocked();
  }
  // The worker has no cancel for music jobs: a cancelled job stops polling
  // and the worker finishes the piece on its own.
  const std::string query = "model=" + job->model;
  json::Value final_status;
  for (;;) {
    const auto until = std::chrono::steady_clock::now() + options_.video_poll;
    while (!job->cancel.load() && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (job->cancel.load())
      throw Cancelled();
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/audio/music/" + music_id;
    poll.query = query;
    poll.request_id = job->id;
    const FetchResult state = options_.fetch(poll);
    if (state.status != 200 && state.status != 202)
      throw JobFailure("music status failed: " + ErrorMessage(state),
                       ErrorCode(state), state.status);
    json::Value parsed = json::parse(state.body);
    const std::string status = parsed.member_str("status");
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      job->progress =
          std::min(99.0, std::max(0.0, parsed.member_double("progress", 0)));
    }
    if (status == "completed") {
      final_status = std::move(parsed);
      break;
    }
    if (status == "failed" || status == "cancelled") {
      const std::string message = parsed.member_str("error");
      throw JobFailure("music " + status +
                           (message.empty() ? std::string() : ": " + message),
                       "music_" + status, 200);
    }
  }
  const FetchResult content = Call(
      job, "GET", "/v1/audio/music/" + music_id + "/content", query, "", "");
  if (content.status != 200 || content.body.empty())
    throw JobFailure("music content failed: " + ErrorMessage(content),
                     ErrorCode(content), content.status);
  const std::string name =
      "output." + (content.content_type.empty()
                       ? std::string("wav")
                       : ExtensionFor(content.content_type));
  WriteAll(JobDir(job->id) / name, content.body);
  const std::string status_text = final_status.dump();
  WriteAll(JobDir(job->id) / "response.json", status_text);
  const std::string base = "/v1/jobs/" + job->id + "/files/";
  json::Value files = json::Value::array();
  json::Value audio = json::Value::object();
  audio["name"] = name;
  audio["url"] = base + name;
  audio["content_type"] = ContentTypeFor(name);
  audio["bytes"] = static_cast<long long>(content.body.size());
  files.push_back(std::move(audio));
  json::Value status_file = json::Value::object();
  status_file["name"] = "response.json";
  status_file["url"] = base + "response.json";
  status_file["content_type"] = "application/json";
  status_file["bytes"] = static_cast<long long>(status_text.size());
  files.push_back(std::move(status_file));
  json::Value result = json::Value::object();
  result["status_code"] = created.status;
  result["content_type"] = ContentTypeFor(name);
  result["files"] = std::move(files);
  result["response"] = std::move(final_status);
  Finish(job, "completed", std::move(result), json::Value());
}

// ----------------------------------------------------------- video concat

void JobQueue::ConcatVideos(const json::Value& body, const fs::path& output,
                            std::atomic<bool>* cancel, json::Value* summary) {
  const auto invalid = [](const std::string& message) {
    return JobFailure(message, "invalid_input", 400);
  };
  const auto failed = [](const std::string& message) {
    return JobFailure(message, "concat_failed", 500);
  };
  const auto check_cancel = [&] {
    if (cancel != nullptr && cancel->load())
      throw Cancelled();
  };
  if (!body.is_object())
    throw invalid("request body must be a JSON object");
  const json::Value* videos = body.find("videos");
  if (videos == nullptr || !videos->is_array())
    throw invalid("'videos' must be an array of job file URLs or data URLs");
  if (videos->size() < kConcatMinInputs || videos->size() > kConcatMaxInputs)
    throw invalid("'videos' needs " + std::to_string(kConcatMinInputs) +
                  " to " + std::to_string(kConcatMaxInputs) + " entries");
  if (const json::Value* format = body.find("output_format");
      format != nullptr && !format->is_null() &&
      (!format->is_string() || Lower(format->str()) != "mp4"))
    throw invalid("'output_format' must be \"mp4\"");
  bool reencode = false;
  if (const json::Value* flag = body.find("reencode");
      flag != nullptr && !flag->is_null()) {
    if (!flag->is_bool())
      throw invalid("'reencode' must be true or false");
    reencode = flag->as_bool();
  }

  // Inputs: a result file of a finished job here, or a data URL. Each is
  // resolved first so a bad list fails before any ffmpeg call.
  struct Input {
    fs::path source;  // empty for data URLs
    std::string bytes;
    std::string label;
  };
  std::vector<Input> inputs;
  int index = 0;
  for (const json::Value& item : videos->items()) {
    ++index;
    const std::string where = "videos[" + std::to_string(index - 1) + "]";
    if (!item.is_string())
      throw invalid(where + " must be a string");
    const std::string& text = item.str();
    Input input;
    if (text.starts_with("data:")) {
      std::string type;
      auto bytes = ParseDataUrl(text, &type);
      if (!bytes)
        throw invalid(where + " is not a base64 data URL");
      if (bytes->size() > kConcatMaxDataUrl)
        throw invalid(where + " exceeds " +
                      std::to_string(kConcatMaxDataUrl / (1024 * 1024)) +
                      " MiB");
      if (!type.empty() && !type.starts_with("video/") &&
          type != "application/octet-stream")
        throw invalid(where + " has media type " + type + ", not video/*");
      input.bytes = std::move(*bytes);
      input.label = "data URL " + std::to_string(input.bytes.size()) + " bytes";
    } else {
      const auto ref = ParseJobFileUrl(text);
      if (!ref)
        throw invalid(where + " must be /v1/jobs/{id}/files/{name} of this "
                              "router or a data URL");
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto it = jobs_.find(ref->id);
        if (it == jobs_.end())
          throw invalid(where + ": job " + ref->id + " is not on this router");
        if (it->second->status != "completed")
          throw invalid(where + ": job " + ref->id + " is " +
                        it->second->status + ", not completed");
        if (it->second->expired_at != 0)
          throw invalid(where + ": the files of job " + ref->id +
                        " have expired");
      }
      input.source = JobDir(ref->id) / ref->name;
      std::error_code ec;
      if (!fs::is_regular_file(input.source, ec))
        throw invalid(where + ": job " + ref->id + " has no file " +
                      ref->name);
      input.label = "/v1/jobs/" + ref->id + "/files/" + ref->name;
    }
    inputs.push_back(std::move(input));
  }
  check_cancel();

  if (options_.ffmpeg.empty() || options_.ffprobe.empty())
    throw failed("video concat needs ffmpeg and ffprobe (--comfy-ffmpeg)");

  // Work folder: the clips under short names (the concat list needs no
  // quoting then), probes, the list and ffmpeg's output.
  TempDir work{options_.root / "temp" /
               ("concat-" + NewJobId().substr(4, 12))};
  std::error_code ec;
  fs::create_directories(work.path, ec);
  if (ec)
    throw failed("cannot create " + work.path.string() + ": " + ec.message());
  std::vector<fs::path> clips;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const fs::path clip = work.path / ("in-" + std::to_string(i + 1) + ".mp4");
    if (inputs[i].source.empty()) {
      WriteAll(clip, inputs[i].bytes);
      inputs[i].bytes.clear();
    } else {
      fs::create_hard_link(inputs[i].source, clip, ec);
      if (ec) {
        ec.clear();
        fs::copy_file(inputs[i].source, clip, ec);
        if (ec)
          throw failed("cannot copy " + inputs[i].label + ": " + ec.message());
      }
    }
    clips.push_back(clip);
  }

  // Probe every clip.
  std::vector<ProbeInfo> infos;
  for (std::size_t i = 0; i < clips.size(); ++i) {
    check_cancel();
    const fs::path out =
        work.path / ("probe-" + std::to_string(i + 1) + ".json");
    const fs::path err =
        work.path / ("probe-" + std::to_string(i + 1) + ".err");
    std::string error;
    const int code = RunCaptured(
        options_.ffprobe,
        {"-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", clips[i]},
        std::chrono::minutes(2), out, err, cancel, &error);
    if (error == "cancelled")
      throw Cancelled();
    if (code != 0)
      throw invalid("videos[" + std::to_string(i) + "] (" + inputs[i].label +
                    ") is not a readable video: " +
                    (error.empty() ? TailOf(ReadIfThere(err), 600) : error));
    ProbeInfo info;
    if (!ParseProbe(ReadIfThere(out), &info, &error))
      throw invalid("videos[" + std::to_string(i) + "] (" + inputs[i].label +
                    "): " + error);
    infos.push_back(std::move(info));
  }
  const std::string mode = ConcatMode(infos, reencode);
  check_cancel();

  std::vector<fs::path> args = {"-hide_banner", "-loglevel", "error",
                                "-nostdin", "-y"};
  if (mode == "copy") {
    std::string list;
    for (const fs::path& clip : clips)
      list += "file '" + clip.filename().string() + "'\n";
    const fs::path list_file = work.path / "list.txt";
    WriteAll(list_file, list);
    for (const char* arg : {"-f", "concat", "-safe", "0", "-i"})
      args.emplace_back(arg);
    args.push_back(list_file);
    for (const char* arg : {"-c", "copy", "-movflags", "+faststart"})
      args.emplace_back(arg);
  } else {
    // Every clip is scaled and padded to the first one's size, brought to
    // its frame rate and yuv420p; audio is resampled to 48 kHz stereo, a
    // clip without audio gets silence of its length so the concat filter
    // sees the same streams everywhere.
    const ProbeInfo& first = infos.front();
    const bool any_audio =
        std::any_of(infos.begin(), infos.end(),
                    [](const ProbeInfo& info) { return info.has_audio; });
    const std::string size =
        std::to_string(first.width) + ":" + std::to_string(first.height);
    std::string filter;
    std::string joined;
    for (std::size_t i = 0; i < clips.size(); ++i) {
      args.emplace_back("-i");
      args.push_back(clips[i]);
      const std::string n = std::to_string(i);
      filter += "[" + n + ":v:0]scale=" + size +
                ":force_original_aspect_ratio=decrease,pad=" + size +
                ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" + first.frame_rate +
                ",format=yuv420p[v" + n + "];";
      joined += "[v" + n + "]";
      if (any_audio) {
        if (infos[i].has_audio) {
          filter += "[" + n +
                    ":a:0]aresample=48000,aformat=sample_fmts=fltp:"
                    "channel_layouts=stereo[a" +
                    n + "];";
        } else {
          if (infos[i].duration <= 0)
            throw invalid("videos[" + n + "] (" + inputs[i].label +
                          ") has no audio and no known duration");
          char seconds[32];
          std::snprintf(seconds, sizeof(seconds), "%.3f", infos[i].duration);
          filter += "anullsrc=channel_layout=stereo:sample_rate=48000,"
                    "atrim=0:" +
                    std::string(seconds) + "[a" + n + "];";
        }
        joined += "[a" + n + "]";
      }
    }
    filter += joined + "concat=n=" + std::to_string(clips.size()) +
              ":v=1:a=" + (any_audio ? "1" : "0") + "[v]" +
              (any_audio ? "[a]" : "");
    args.emplace_back("-filter_complex");
    args.emplace_back(filter);
    for (const char* arg : {"-map", "[v]"})
      args.emplace_back(arg);
    if (any_audio)
      for (const char* arg : {"-map", "[a]"})
        args.emplace_back(arg);
    for (const char* arg :
         {"-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt",
          "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"})
      args.emplace_back(arg);
  }
  const fs::path stitched = work.path / "video.mp4";
  args.push_back(stitched);
  const fs::path out = work.path / "ffmpeg.out";
  const fs::path err = work.path / "ffmpeg.err";
  std::string error;
  const int code = RunCaptured(options_.ffmpeg, args, std::chrono::minutes(30),
                               out, err, cancel, &error);
  if (error == "cancelled")
    throw Cancelled();
  if (code != 0)
    throw failed("ffmpeg (" + mode + ") failed: " +
                 (error.empty() ? TailOf(ReadIfThere(err), 800) : error));
  if (!fs::is_regular_file(stitched, ec) || fs::file_size(stitched, ec) == 0)
    throw failed("ffmpeg (" + mode + ") wrote no output");
  fs::create_directories(output.parent_path(), ec);
  fs::remove(output, ec);
  ec.clear();
  fs::rename(stitched, output, ec);
  if (ec) {
    ec.clear();
    fs::copy_file(stitched, output, fs::copy_options::overwrite_existing, ec);
    if (ec)
      throw failed("cannot store the result: " + ec.message());
  }
  if (summary != nullptr) {
    json::Value result = json::Value::object();
    result["mode"] = mode;
    result["reencode"] = mode == "reencode";
    json::Value list = json::Value::array();
    double total = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      json::Value entry = json::Value::object();
      entry["input"] = inputs[i].label;
      entry["codec"] = infos[i].video_codec;
      entry["width"] = static_cast<double>(infos[i].width);
      entry["height"] = static_cast<double>(infos[i].height);
      entry["frame_rate"] = infos[i].frame_rate;
      entry["audio"] = infos[i].has_audio;
      entry["duration"] = infos[i].duration;
      total += infos[i].duration;
      list.push_back(std::move(entry));
    }
    result["inputs"] = std::move(list);
    result["duration"] = total;
    result["bytes"] = static_cast<double>(fs::file_size(output, ec));
    *summary = std::move(result);
  }
}

HttpResponse JobQueue::Concat(const HttpRequest& request) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& error) {
    return Err(400, "Bad Request",
               std::string("request body must be JSON: ") + error.what(),
               "invalid_input");
  }
  TempDir out{options_.root / "temp" /
              ("concat-out-" + NewJobId().substr(4, 12))};
  const fs::path file = out.path / "video.mp4";
  std::atomic<bool> cancel{false};
  try {
    ConcatVideos(body, file, &cancel, nullptr);
  } catch (const JobFailure& failure) {
    Logger::Warn("jobs", "event=concat_failed code=" + failure.code +
                             " error=" + failure.what());
    return Err(failure.status,
               failure.status >= 500 ? "Internal Server Error" : "Bad Request",
               failure.what(), failure.code);
  }
  std::string bytes = ReadAll(file);
  Logger::Info("jobs",
               "event=concat_done bytes=" + std::to_string(bytes.size()));
  return HttpResponse{
      .status = 200,
      .reason = "OK",
      .body = std::move(bytes),
      .headers = {{"Content-Type", "video/mp4"},
                  {"Content-Disposition", "attachment; filename=\"video.mp4\""},
                  {"Cache-Control", "no-store"}}};
}

void JobQueue::RunConcat(const std::shared_ptr<Job>& job,
                         const std::string& payload) {
  json::Value body;
  try {
    body = json::parse(payload);
  } catch (const std::exception& error) {
    throw JobFailure(std::string("request body must be JSON: ") + error.what(),
                     "invalid_input", 400);
  }
  const fs::path dir = JobDir(job->id);
  json::Value summary;
  ConcatVideos(body, dir / "video.mp4", &job->cancel, &summary);
  const std::string summary_text = summary.dump();
  WriteAll(dir / "response.json", summary_text);
  const std::string base = "/v1/jobs/" + job->id + "/files/";
  std::error_code ec;
  json::Value files = json::Value::array();
  json::Value video = json::Value::object();
  video["name"] = "video.mp4";
  video["url"] = base + "video.mp4";
  video["content_type"] = "video/mp4";
  video["bytes"] = static_cast<double>(fs::file_size(dir / "video.mp4", ec));
  files.push_back(std::move(video));
  json::Value status_file = json::Value::object();
  status_file["name"] = "response.json";
  status_file["url"] = base + "response.json";
  status_file["content_type"] = "application/json";
  status_file["bytes"] = static_cast<double>(summary_text.size());
  files.push_back(std::move(status_file));
  json::Value result = json::Value::object();
  result["status_code"] = 200;
  result["content_type"] = "video/mp4";
  result["files"] = std::move(files);
  result["response"] = std::move(summary);
  Finish(job, "completed", std::move(result), json::Value());
}

json::Value JobQueue::StoreResult(Job& job, const FetchResult& result) {
  const fs::path dir = JobDir(job.id);
  const std::string base = "/v1/jobs/" + job.id + "/files/";
  json::Value files = json::Value::array();
  const auto add_file = [&](const std::string& name, std::string_view bytes) {
    WriteAll(dir / name, bytes);
    json::Value file = json::Value::object();
    file["name"] = name;
    file["url"] = base + name;
    file["content_type"] = ContentTypeFor(name);
    file["bytes"] = static_cast<long long>(bytes.size());
    files.push_back(std::move(file));
  };
  json::Value stored = json::Value::object();
  stored["status_code"] = result.status;
  stored["content_type"] = result.content_type;
  bool is_json = IsJsonType(result.content_type);
  json::Value parsed;
  if (is_json || result.content_type.empty()) {
    try {
      parsed = json::parse(result.body);
      is_json = true;
    } catch (const std::exception&) {
      is_json = false;
    }
  }
  if (is_json) {
    // Images: every data[].b64_json becomes a file and a url.
    if (const json::Value* data = parsed.find("data");
        data != nullptr && data->is_array()) {
      json::Value items = json::Value::array();
      int index = 0;
      for (const json::Value& item : data->items()) {
        const json::Value* b64 = item.find("b64_json");
        if (b64 == nullptr || !b64->is_string()) {
          items.push_back(item);
          continue;
        }
        const auto bytes = comfy::Base64Decode(b64->str());
        if (!bytes) {
          items.push_back(item);
          continue;
        }
        const std::string name = "image-" + std::to_string(++index) + "." +
                                 SniffImageExtension(*bytes);
        add_file(name, *bytes);
        json::Value entry = json::Value::object();
        for (const auto& [key, value] : item.members())
          if (key != "b64_json")
            entry[key] = value;
        entry["url"] = base + name;
        items.push_back(std::move(entry));
      }
      parsed["data"] = std::move(items);
    }
    const std::string text = parsed.dump();
    add_file("response.json", text);
    if (text.size() <= kMaxInlineResponse)
      stored["response"] = std::move(parsed);
  } else {
    add_file("output." + ExtensionFor(result.content_type), result.body);
  }
  stored["files"] = std::move(files);
  return stored;
}

void JobQueue::Finish(const std::shared_ptr<Job>& job, std::string status,
                      json::Value result, json::Value error) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    job->status = std::move(status);
    job->completed_at = NowSeconds();
    job->result = std::move(result);
    job->error = std::move(error);
    job->preview_id.clear();
    if (job->status == "completed") {
      job->progress = 100;
      if (job->started_at > 0) {
        // Learned duration for the estimated progress: half old, half new,
        // at least one second (the timestamps are whole seconds).
        const double seconds = static_cast<double>(
            std::max(1LL, job->completed_at - job->started_at));
        double& learned = durations_[DurationKey(job->path, job->model)];
        learned = learned > 0 ? 0.5 * learned + 0.5 * seconds : seconds;
      }
    }
    if (!job->webhook_url.empty()) {
      job->webhook_state = "pending";
      job->webhook_due = std::chrono::steady_clock::now();
    }
    PersistLocked();
  }
  webhook_wake_.notify_all();
}

// ---------------------------------------------------------------- expiry

void JobQueue::Tick() {
  const auto now = std::chrono::steady_clock::now();
  if (now < next_expiry_)
    return;
  next_expiry_ = now + std::chrono::seconds(60);
  try {
    ExpireFiles(NowSeconds());
  } catch (const std::exception& error) {
    Logger::Warn("jobs", std::string("event=jobs_expire_failed error=") +
                             error.what());
  }
}

std::size_t JobQueue::ExpireFiles(long long now) {
  const long long after = options_.expire_after.count();
  if (after <= 0)
    return 0;
  std::size_t expired = 0;
  // The files go under the lock, so a PATCH keep that arrives meanwhile
  // either comes first (nothing is deleted) or finds the job expired.
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& job : order_) {
    if (!Finished(job->status) || job->keep || job->expired_at != 0 ||
        job->completed_at <= 0 || now - job->completed_at < after)
      continue;
    std::size_t files = 0;
    unsigned long long bytes = 0;
    std::error_code ec;
    std::vector<fs::path> media;
    for (const auto& entry : fs::directory_iterator(JobDir(job->id), ec)) {
      if (!IsKeptFile(entry.path().filename().string()) &&
          entry.is_regular_file(ec))
        media.push_back(entry.path());
    }
    for (const fs::path& file : media) {
      const auto size = fs::file_size(file, ec);
      if (!ec)
        bytes += size;
      if (fs::remove(file, ec))
        ++files;
      else if (ec)
        Logger::Warn("jobs", "event=job_expire_file_failed id=" + job->id +
                                 " file=" + file.filename().string() +
                                 " error=" + ec.message());
    }
    if (job->result.is_object()) {
      if (const json::Value* listed = job->result.find("files");
          listed != nullptr && listed->is_array()) {
        json::Value left = json::Value::array();
        for (const json::Value& file : listed->items())
          if (IsKeptFile(file.member_str("name")))
            left.push_back(file);
        job->result["files"] = std::move(left);
      }
    }
    job->expired_at = now;
    ++expired;
    Logger::Info("jobs", "event=job_expired id=" + job->id +
                             " status=" + job->status +
                             " files=" + std::to_string(files) +
                             " bytes=" + std::to_string(bytes) +
                             " age_hours=" +
                             std::to_string((now - job->completed_at) / 3600));
  }
  if (expired > 0)
    PersistLocked();
  return expired;
}

// ---------------------------------------------------------------- webhooks

void JobQueue::WebhookMain() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopping_) {
    std::shared_ptr<Job> due;
    auto next = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    const auto now = std::chrono::steady_clock::now();
    for (const auto& job : order_) {
      if (job->webhook_state != "pending")
        continue;
      if (job->webhook_due <= now) {
        due = job;
        break;
      }
      next = std::min(next, job->webhook_due);
    }
    if (due == nullptr) {
      webhook_wake_.wait_until(lock, next);
      continue;
    }
    lock.unlock();
    DeliverOne(due);
    lock.lock();
  }
}

void JobQueue::DeliverOne(const std::shared_ptr<Job>& job) {
  std::string body;
  std::string event;
  std::string url_text;
  int attempt = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    event = "job." + job->status;
    json::Value payload = json::Value::object();
    payload["type"] = event;
    payload["timestamp"] = IsoTime(job->completed_at);
    payload["data"] = ViewLocked(*job);
    body = payload.dump();
    url_text = job->webhook_url;
    attempt = ++job->webhook_attempts;
  }
  int status = 0;
  std::string error;
  WebhookUrl url;
  // The target is checked again at every attempt: the allow list or DNS may
  // have changed since the submit.
  if (ParseWebhookUrl(url_text, &url, &error)) {
    if (const auto address = CheckWebhookTarget(url, options_.webhook_allow,
                                                options_.resolve, &error)) {
      HttpRequest post;
      post.method = "POST";
      post.path = url.target;
      post.body = body;
      // One id per job and event: the same across retries, so receivers can
      // drop duplicates.
      const std::string message_id = "msg_" + job->id + "_" + event.substr(4);
      const long long timestamp = NowSeconds();
      post.headers = {{"Content-Type", "application/json"},
                      {"User-Agent", "gufo-router"},
                      {"webhook-id", message_id},
                      {"webhook-timestamp", std::to_string(timestamp)}};
      if (!options_.webhook_secret.empty())
        post.headers.emplace_back(
            "webhook-signature",
            SignWebhook(options_.webhook_secret, message_id, timestamp, body));
      try {
        status = options_.deliver(*address, url, post,
                                  options_.webhook_timeout, &error);
      } catch (const std::exception& failure) {
        status = 0;
        error = failure.what();
      }
    }
  }
  const bool delivered = status >= 200 && status < 300;
  // 4xx other than 408/429 will not get better by retrying.
  const bool permanent = status >= 400 && status < 500 && status != 408 &&
                         status != 429;
  const std::lock_guard<std::mutex> lock(mutex_);
  job->webhook_last_status = status;
  job->webhook_last_error =
      delivered ? std::string()
                : (status > 0 ? "HTTP " + std::to_string(status) : error);
  const std::size_t retries = options_.webhook_retry.size();
  if (delivered) {
    job->webhook_state = "delivered";
  } else if (permanent || static_cast<std::size_t>(attempt) > retries) {
    job->webhook_state = "failed";
  } else {
    job->webhook_due = std::chrono::steady_clock::now() +
                       options_.webhook_retry[static_cast<std::size_t>(
                           attempt - 1)];
  }
  Logger::Info("jobs", "event=webhook id=" + job->id + " attempt=" +
                           std::to_string(attempt) + " status=" +
                           std::to_string(status) + " state=" +
                           job->webhook_state +
                           (job->webhook_last_error.empty()
                                ? ""
                                : " error=\"" + job->webhook_last_error +
                                      "\""));
  PersistLocked();
}

}  // namespace gufo::router::jobs
