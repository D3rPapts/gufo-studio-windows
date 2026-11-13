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
#include "src/cli/serve/router/pool.hpp"
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
  /// LoRA names the wrapped request asks for (capability check).
  std::vector<std::string> loras;
  /// Shared queue: the partner's url while it runs the job on its worker
  /// (POST /router/pool/pull), else empty.
  std::string pulled_by;
  /// X-Gufo-Pool for the job's requests ("local", "remote", "" = any).
  std::string side;
  /// 0..kPriorityMax, higher runs first (with created_at the run order).
  int priority{0};
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
    "/v1/audio/separate",     "/v1/videos/concat",
    "/v1/3d/generations"};

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
  if (type == "model/gltf-binary")
    return "glb";
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
  if (ext == "glb")
    return "model/gltf-binary";
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

/// image, video or audio by the file name's type; "" for anything else.
std::string MediaKindOf(std::string_view name) {
  const std::string type = ContentTypeFor(name);
  if (type.starts_with("image/"))
    return "image";
  if (type.starts_with("video/"))
    return "video";
  if (type.starts_with("audio/"))
    return "audio";
  return "";
}

/// The first image, video or audio file in a result's "files" list, in
/// the order the page shows them; "" without one.
std::string FirstMediaFile(const json::Value& result) {
  if (!result.is_object())
    return "";
  const json::Value* files = result.find("files");
  if (files == nullptr || !files->is_array())
    return "";
  for (const json::Value& file : files->items()) {
    const std::string name = file.member_str("name");
    if (!IsKeptFile(name) && !MediaKindOf(name).empty())
      return name;
  }
  return "";
}

std::string ThumbContentType(const std::string& ext) {
  return ext == "png" ? "image/png" : "image/jpeg";
}

/// ?w=: 64..1024, default 256.
int ThumbWidth(const HttpRequest& request) {
  int width = 256;
  if (const std::string text = request.query_param("w"); !text.empty())
    width = std::clamp(
        static_cast<int>(std::strtol(text.c_str(), nullptr, 10)), 64, 1024);
  return width;
}

/// thumb-<w>.<ext> for the job's first media file, thumb-<w>-<stem>.<ext>
/// for any other file named with ?file=.
std::string ThumbName(int width, const std::string& file,
                      const std::string& first, const std::string& ext) {
  std::string name = "thumb-" + std::to_string(width);
  if (!file.empty() && file != first) {
    std::string stem = file.substr(0, file.rfind('.'));
    std::replace(stem.begin(), stem.end(), '.', '_');
    name += "-" + stem;
  }
  return name + "." + ext;
}

/// The request came from the partner (or another router): never relay it.
bool IsRelayed(const HttpRequest& request) {
  return !request.header(kRelayHeader).empty() ||
         !request.header("X-Gufo-Forwarded").empty();
}

bool WantsPool(const HttpRequest& request) {
  const std::string pool = Lower(request.query_param("pool"));
  return Lower(request.query_param("scope")) == "pool" || pool == "1" ||
         pool == "true" || pool == "yes";
}

/// Reads a streamed answer into its body (the relay streams through).
std::string CollectBody(HttpResponse& response) {
  std::string body;
  if (response.streaming_body) {
    response.streaming_body([&body](std::string_view chunk) {
      body.append(chunk);
      return true;
    });
    response.streaming_body = nullptr;
  } else {
    body = std::move(response.body);
  }
  return body;
}

/// Drops what the partner's HTTP layer added and ours adds again (CORS,
/// request id), so a relayed answer does not carry them twice.
void StripTransportHeaders(HttpResponse& response) {
  std::erase_if(response.headers, [](const auto& header) {
    const std::string name = Lower(header.first);
    return name.starts_with("access-control-") || name == "x-request-id";
  });
}

std::string ResponseContentType(const HttpResponse& response) {
  for (const auto& [name, value] : response.headers)
    if (name.size() == 12 && Lower(name) == "content-type")
      return value;
  return "";
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

/// "http://host:port/..." -> host (without brackets) and port text; false
/// for a URL whose host cannot be told apart.
bool SplitUrlHost(std::string_view url, std::string* host, std::string* port) {
  if (const auto scheme = url.find("://"); scheme != std::string_view::npos)
    url.remove_prefix(scheme + 3);
  if (const auto slash = url.find('/'); slash != std::string_view::npos)
    url = url.substr(0, slash);
  port->clear();
  if (url.starts_with('[')) {
    const auto close = url.find(']');
    if (close == std::string_view::npos)
      return false;
    *host = std::string(url.substr(1, close - 1));
    const std::string_view rest = url.substr(close + 1);
    if (rest.starts_with(':'))
      *port = std::string(rest.substr(1));
    return true;
  }
  const auto colon = url.rfind(':');
  if (colon != std::string_view::npos && url.find(':') == colon) {
    *host = std::string(url.substr(0, colon));
    *port = std::string(url.substr(colon + 1));
  } else {
    *host = std::string(url);  // no port, or a bare IPv6 address
  }
  return true;
}

}  // namespace

bool IsUnreachableUrl(std::string_view url) {
  std::string host, port;
  if (!SplitUrlHost(url, &host, &port))
    return true;
  return host.empty() || host == "0.0.0.0" || host == "::";
}

std::string ReachableUrl(std::string_view url, std::string_view fallback) {
  return std::string(IsUnreachableUrl(url) ? fallback : url);
}

std::string LocalUrlToward(std::string_view toward, int port) {
  try {
    net::EnsureSocketRuntimeStarted();
  } catch (const std::exception&) {
    return "";
  }
  std::string address;
  std::string host, remote_port;
  if (!toward.empty() && SplitUrlHost(toward, &host, &remote_port) &&
      !host.empty() && host != "0.0.0.0" && host != "::") {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* found = nullptr;
    if (::getaddrinfo(host.c_str(),
                      remote_port.empty() ? "8081" : remote_port.c_str(),
                      &hints, &found) == 0 &&
        found != nullptr) {
      for (addrinfo* it = found; it != nullptr && address.empty();
           it = it->ai_next) {
        const net::Socket sock =
            ::socket(it->ai_family, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == net::kInvalidSocket)
          continue;
        // A connected UDP socket picks the route; nothing is sent.
        if (::connect(sock, it->ai_addr,
                      static_cast<net::SocklenT>(it->ai_addrlen)) == 0) {
          sockaddr_storage local{};
          net::SocklenT length = sizeof(local);
          char text[NI_MAXHOST]{};
          if (::getsockname(sock, reinterpret_cast<sockaddr*>(&local),
                            &length) == 0 &&
              ::getnameinfo(reinterpret_cast<sockaddr*>(&local), length, text,
                            sizeof(text), nullptr, 0, NI_NUMERICHOST) == 0)
            address = text;
        }
        net::CloseSocket(sock);
      }
      ::freeaddrinfo(found);
    }
  }
  if (address.empty()) {
    char name[256]{};
    if (::gethostname(name, sizeof(name) - 1) == 0)
      address = name;
  }
  if (address.empty() || address == "0.0.0.0" || address == "::")
    return "";
  if (address.find(':') != std::string::npos)
    address = "[" + address + "]";
  return "http://" + address + ":" + std::to_string(port);
}

double ProbeFrameRate(const fs::path& ffprobe, const fs::path& video,
                      std::string* error) {
  if (ffprobe.empty()) {
    *error = "no ffprobe configured (--comfy-ffmpeg)";
    return 0;
  }
  static std::atomic<unsigned> counter{0};
  std::error_code ec;
  const fs::path base = fs::temp_directory_path(ec);
  if (ec) {
    *error = "no temp directory: " + ec.message();
    return 0;
  }
  TempDir work{base / ("gufo-probe-" +
                       std::to_string(std::chrono::steady_clock::now()
                                          .time_since_epoch()
                                          .count()) +
                       "-" + std::to_string(counter++))};
  fs::create_directories(work.path, ec);
  if (!fs::is_directory(work.path, ec)) {
    *error = "cannot create " + work.path.string();
    return 0;
  }
  const fs::path out = work.path / "probe.json";
  const fs::path err = work.path / "probe.err";
  const int code = RunCaptured(ffprobe,
                               {"-v", "error", "-print_format", "json",
                                "-show_streams", "-show_format", video},
                               std::chrono::minutes(1), out, err, nullptr,
                               error);
  if (code != 0) {
    if (error->empty())
      *error = "ffprobe exit " + std::to_string(code) + ": " +
               TailOf(ReadIfThere(err), 300);
    return 0;
  }
  ProbeInfo info;
  if (!ParseProbe(ReadIfThere(out), &info, error))
    return 0;
  return FrameRateValue(info.frame_rate);
}

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
  info.video_duration = stream_duration;
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

std::vector<std::string> LoraNames(const json::Value& body) {
  std::vector<std::string> names;
  const auto add = [&names](const std::string& name) {
    if (!name.empty() &&
        std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(name);
  };
  if (const json::Value* loras = body.find("loras");
      loras != nullptr && loras->is_array()) {
    for (const json::Value& entry : loras->items()) {
      if (entry.is_object())
        add(entry.member_str("name"));
      else if (entry.is_string())
        add(entry.str());
    }
  }
  if (const json::Value* gufo = body.find("gufo");
      gufo != nullptr && gufo->is_object()) {
    if (const json::Value* lora = gufo->find("lora");
        lora != nullptr && lora->is_string())
      add(lora->str());
  }
  return names;
}

bool CanRun(const json::Value& capabilities, const std::string& model,
            const std::vector<std::string>& loras, std::string* why,
            bool now) {
  const json::Value* models = capabilities.find("models");
  if (models == nullptr || !models->is_object())
    models = &capabilities;
  const json::Value* entry = models->find(model);
  if (entry == nullptr || !entry->is_object()) {
    if (why != nullptr)
      *why = "does not serve model '" + model + "'";
    return false;
  }
  if (entry->member_str("placement") == "remote") {
    if (why != nullptr)
      *why = "has model '" + model + "' placed remote";
    return false;
  }
  if (now) {
    // A load failure holds there (older routers send no field: may).
    const json::Value* available = entry->find("available");
    if (available != nullptr && available->is_bool() &&
        !available->as_bool()) {
      if (why != nullptr) {
        const std::string reason = entry->member_str("unavailable");
        *why = "cannot load model '" + model + "' now" +
               (reason.empty() ? "" : " (" + reason + ")");
      }
      return false;
    }
  }
  const json::Value* have = entry->find("loras");
  for (const std::string& need : loras) {
    bool found = false;
    std::string registered;
    if (have != nullptr && have->is_array()) {
      for (const json::Value& name : have->items()) {
        if (!name.is_string())
          continue;
        found = found || name.str() == need;
        registered += (registered.empty() ? "" : ", ") + name.str();
      }
    }
    if (!found) {
      if (why != nullptr)
        *why = "has no LoRA '" + need + "' for model '" + model + "'" +
               (registered.empty() ? " (none registered)"
                                   : " (registered: " + registered + ")");
      return false;
    }
  }
  return true;
}

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
  std::vector<PulledThread> pulled;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    pulled.swap(pulled_threads_);
  }
  for (auto& entry : pulled)
    if (entry.thread.joinable())
      entry.thread.join();
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
  if (path == kPullPath) {
    if (method != "POST")
      return Err(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
    try {
      return Pull(request);
    } catch (const std::exception& error) {
      Logger::Warn("jobs", std::string("event=request_failed path=") +
                               request.path + " error=" + error.what());
      return Err(500, "Internal Server Error", error.what(), "server_error");
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
    bool known = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      known = jobs_.contains(id);
    }
    if (!known && options_.relay && !IsRelayed(request))
      return RelayJob(id, tail, request);
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
    if (tail == "/thumb") {
      if (method != "GET")
        return Err(405, "Method Not Allowed", "method is not allowed",
                   "method_not_allowed");
      return Thumb(id, request);
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
  // A multipart form field "priority" (edits, upscale).
  std::string priority_part;
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
    job->loras = LoraNames(*json_body);
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
        if (part.name == "priority")
          priority_part = part.data;
        if (part.name == "loras") {
          // The ComfyUI adapter sends the LoRA list as a JSON part.
          try {
            json::Value wrapped = json::Value::object();
            wrapped["loras"] = json::parse(part.data);
            job->loras = LoraNames(wrapped);
          } catch (const std::exception&) {
            // The worker answers the malformed part itself.
          }
        }
        if (part.name == "stream" && (part.data == "true" || part.data == "1"))
          return Err(400, "Bad Request",
                     "streaming responses cannot run as a job; drop 'stream'",
                     "stream_unsupported");
      }
    } else if (IsJsonType(job->content_type)) {
      try {
        const json::Value parsed = json::parse(payload);
        job->model = parsed.member_str("model");
        job->loras = LoraNames(parsed);
      } catch (const std::exception&) {
        return Err(400, "Bad Request", "request body is not valid JSON",
                   "invalid_json");
      }
    }
  }
  // Priority (2.0.3): the header wins, then the submit body's "priority",
  // then the wrapped body's (JSON member or multipart field).
  {
    std::string header = request.header(kPriorityHeader);
    const json::Value* member = nullptr;
    if (header.empty()) {
      member = body.find("priority");
      if (member == nullptr && json_body != nullptr)
        member = json_body->find("priority");
      if (member == nullptr && !priority_part.empty())
        header = priority_part;
    }
    std::string why;
    if (!ParsePriority(header, member, &job->priority, &why))
      return Err(400, "Bad Request", why, "invalid_priority");
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
  if (needs_model) {
    // Neither router can run it (model placed remote on both, a LoRA no
    // machine has): say so now instead of failing the job later.
    json::Value refusal;
    SideFor(*job, &refusal);
    if (!refusal.is_null())
      return Err(400, "Bad Request", refusal.member_str("message"),
                 refusal.member_str("code"));
  }
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
  InsertPendingLocked(job, /*front=*/false);
  PersistLocked();
  const json::Value view = ViewLocked(*job);
  lock.unlock();
  wake_.notify_all();
  Logger::Info("jobs", "event=job_queued id=" + job->id + " path=" + job->path +
                           " model=" + job->model +
                           (job->priority > 0
                                ? " priority=" + std::to_string(job->priority)
                                : "") +
                           (job->webhook_url.empty() ? "" : " webhook=yes"));
  HttpResponse accepted = JsonResponse(202, "Accepted", view);
  accepted.headers.emplace_back("Location", "/v1/jobs/" + job->id);
  return accepted;
}

HttpResponse JobQueue::List(const HttpRequest& request) {
  if (WantsPool(request) && !IsRelayed(request))
    return PoolList(request);
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

void JobQueue::InsertPendingLocked(const std::shared_ptr<Job>& job,
                                   bool front) {
  // Run order: the highest priority first, then the oldest (first in, first
  // out). A requeued job goes back to the front of its priority.
  auto it = pending_.begin();
  while (it != pending_.end() &&
         (front ? (*it)->priority > job->priority
                : (*it)->priority >= job->priority))
    ++it;
  pending_.insert(it, job);
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

HttpResponse JobQueue::Thumb(const std::string& id,
                             const HttpRequest& request) {
  const int width = ThumbWidth(request);
  const std::string requested = request.query_param("file");
  std::string first;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end())
      return Err(404, "Not Found", "job not found", "job_not_found");
    first = FirstMediaFile(found->second->result);
  }
  const std::string name = requested.empty() ? first : requested;
  if (name.empty())
    return Err(404, "Not Found", "job has no image, video or audio file",
               "file_not_found");
  if (!ValidFileName(name) || IsKeptFile(name))
    return Err(404, "Not Found", "file not found", "file_not_found");
  const fs::path source = JobDir(id) / name;
  std::error_code ec;
  if (!fs::is_regular_file(source, ec))
    return Err(404, "Not Found", "file not found", "file_not_found");
  const std::string kind = MediaKindOf(name);
  if (kind.empty())
    return Err(400, "Bad Request",
               name + " is not an image, video or audio file",
               "thumb_unsupported");
  const std::string ext = kind == "audio" ? "png" : "jpg";
  const fs::path cache = JobDir(id) / ThumbName(width, requested, first, ext);
  const auto serve = [&](const char* how) {
    return HttpResponse{.status = 200,
                        .reason = "OK",
                        .body = ReadAll(cache),
                        .headers = {{"Content-Type", ThumbContentType(ext)},
                                    {"Cache-Control", "public, max-age=86400"},
                                    {"X-Gufo-Thumb", how}}};
  };
  if (fs::is_regular_file(cache, ec))
    return serve("cached");
  if (options_.ffmpeg.empty())
    return Err(501, "Not Implemented",
               "thumbnails need ffmpeg (--comfy-ffmpeg)", "thumb_unavailable");
  const std::lock_guard<std::mutex> thumb_lock(thumb_mutex_);
  if (fs::is_regular_file(cache, ec))
    return serve("cached");  // made while waiting for the lock
  const auto started = std::chrono::steady_clock::now();
  TempDir work{options_.root / "temp" / ("thumb-" + id)};
  fs::create_directories(work.path, ec);
  if (ec)
    return Err(500, "Internal Server Error",
               "cannot create " + work.path.string() + ": " + ec.message(),
               "thumb_failed");
  const fs::path out = work.path / ("thumb." + ext);
  const fs::path log_out = work.path / "ffmpeg.out";
  const fs::path log_err = work.path / "ffmpeg.err";
  const std::string scale =
      "scale=" + std::to_string(width) + ":-2,format=yuvj420p";
  double at = 0;
  if (kind == "video") {
    // A frame at 10 % of the length, 0.5 s when the length is unknown.
    at = 0.5;
    if (!options_.ffprobe.empty()) {
      const fs::path probe_out = work.path / "probe.json";
      const fs::path probe_err = work.path / "probe.err";
      std::string error;
      const int code = RunCaptured(
          options_.ffprobe,
          {"-v", "error", "-print_format", "json", "-show_streams",
           "-show_format", source},
          std::chrono::minutes(2), probe_out, probe_err, nullptr, &error);
      ProbeInfo info;
      if (code == 0 && ParseProbe(ReadIfThere(probe_out), &info, &error) &&
          info.duration > 0)
        at = info.duration * 0.1;
    }
  }
  const auto run = [&](double seek) {
    std::vector<fs::path> args = {"-hide_banner", "-loglevel", "error",
                                  "-nostdin", "-y"};
    if (kind == "video") {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.3f", seek);
      args.emplace_back("-ss");
      args.emplace_back(buf);
    }
    args.emplace_back("-i");
    args.push_back(source);
    if (kind == "audio") {
      args.emplace_back("-filter_complex");
      args.emplace_back("showwavespic=s=" + std::to_string(width) + "x" +
                        std::to_string(std::max(1, width / 3)) +
                        ":colors=#5b4fc4");
      args.emplace_back("-frames:v");
      args.emplace_back("1");
    } else {
      args.emplace_back("-frames:v");
      args.emplace_back("1");
      args.emplace_back("-vf");
      args.emplace_back(scale);
      args.emplace_back("-q:v");
      args.emplace_back("3");
    }
    args.push_back(out);
    std::string error;
    const int code = RunCaptured(options_.ffmpeg, args, std::chrono::minutes(2),
                                 log_out, log_err, nullptr, &error);
    std::error_code exists_ec;
    if (code == 0 && fs::is_regular_file(out, exists_ec))
      return std::string();
    if (!error.empty())
      return error;
    return "ffmpeg exit " + std::to_string(code) + ": " +
           TailOf(ReadIfThere(log_err), 600);
  };
  std::string failure = run(at);
  if (!failure.empty() && kind == "video" && at > 0)
    failure = run(0);  // the seek point lay past the last frame
  if (!failure.empty()) {
    Logger::Warn("jobs", "event=thumb_failed id=" + id + " file=" + name +
                             " error=" + failure);
    return Err(500, "Internal Server Error",
               "thumbnail of " + name + " failed: " + failure, "thumb_failed");
  }
  fs::rename(out, cache, ec);
  if (ec) {
    ec.clear();
    fs::copy_file(out, cache, fs::copy_options::overwrite_existing, ec);
    if (ec)
      return Err(500, "Internal Server Error",
                 "cannot store " + cache.string() + ": " + ec.message(),
                 "thumb_failed");
  }
  const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  Logger::Info("jobs", "event=thumb id=" + id + " file=" + name +
                           " kind=" + kind + " w=" + std::to_string(width) +
                           " ms=" + std::to_string(took.count()));
  return serve("generated");
}

HttpResponse JobQueue::PoolList(const HttpRequest& request) {
  std::size_t limit = 100;
  if (const std::string text = request.query_param("limit"); !text.empty())
    limit = std::clamp<std::size_t>(std::strtoull(text.c_str(), nullptr, 10),
                                    1, 1000);
  Filter filter;
  std::string error;
  if (!ParseFilter(request, &filter, &error))
    return Err(400, "Bad Request", error, "invalid_filter");
  std::vector<json::Value> merged;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = order_.rbegin(); it != order_.rend() && merged.size() < limit;
         ++it) {
      if (!MatchesLocked(**it, filter))
        continue;
      json::Value view = ViewLocked(**it);
      view["owner"] = "local";
      merged.push_back(std::move(view));
    }
  }
  std::string partner = "none";
  if (options_.relay) {
    // The partner answers its own list (the relay header keeps it from
    // asking us back); its jobs get its address as owner.
    HttpRequest copy;
    copy.method = "GET";
    copy.path = "/v1/jobs";
    copy.query = request.query;
    copy.request_id = request.request_id;
    copy.headers.emplace_back(kRelayHeader, "1");
    bool ok = false;
    std::string why;
    auto answer = options_.relay(copy, std::chrono::seconds(3),
                                 std::chrono::seconds(3), &why);
    if (answer) {
      const std::string body = CollectBody(*answer);
      if (answer->status == 200) {
        try {
          const json::Value parsed = json::parse(body);
          if (const json::Value* data = parsed.find("data");
              data != nullptr && data->is_array()) {
            for (const json::Value& item : data->items()) {
              if (!item.is_object())
                continue;
              json::Value job = item;
              job["owner"] = options_.partner_name;
              merged.push_back(std::move(job));
            }
            ok = true;
          } else {
            why = "no job list in the answer";
          }
        } catch (const std::exception& failure) {
          why = std::string("answer is not JSON: ") + failure.what();
        }
      } else {
        why = "HTTP " + std::to_string(answer->status) + " " +
              TailOf(body, 200);
      }
    }
    if (ok) {
      partner = options_.partner_name;
    } else {
      partner = "unreachable";
      Logger::Warn("jobs", "event=pool_list_partner_failed partner=" +
                               options_.partner_name + " error=" + why);
    }
  }
  // "running_on" from this router's point of view: the partner's "local"
  // is the partner, and a job the partner runs here (pulled) is local.
  std::string me;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    me = self_name_.empty() ? options_.self_name : self_name_;
  }
  for (json::Value& job : merged) {
    if (job.member_str("owner") == "local")
      continue;
    const std::string running_on = job.member_str("running_on");
    if (running_on == "local")
      job["running_on"] = options_.partner_name;
    else if (!running_on.empty() &&
             (running_on == me || running_on == options_.self_name))
      job["running_on"] = "local";
  }
  std::stable_sort(merged.begin(), merged.end(),
                   [](const json::Value& a, const json::Value& b) {
                     return a.member_double("created_at") >
                            b.member_double("created_at");
                   });
  // "position": rank among the queued (or waiting in the pool) jobs of
  // both routers, oldest = 1 (the order the shared queue runs them in);
  // null for the others.
  {
    // Within one router the submit order holds for jobs of the same second
    // (the lists come newest first, so walk them backwards).
    std::vector<std::size_t> queued;
    for (std::size_t i = merged.size(); i-- > 0;) {
      const json::Value* wait = merged[i].find("pool_wait");
      if (merged[i].member_str("status") == "queued" ||
          (merged[i].member_str("status") == "in_progress" &&
           wait != nullptr && wait->as_bool()))
        queued.push_back(i);
    }
    std::stable_sort(queued.begin(), queued.end(),
                     [&merged](std::size_t a, std::size_t b) {
                       return RunsBefore(
                           PriorityOfView(merged[a]),
                           static_cast<long long>(
                               merged[a].member_double("created_at")),
                           PriorityOfView(merged[b]),
                           static_cast<long long>(
                               merged[b].member_double("created_at")));
                     });
    for (json::Value& job : merged)
      job["position"] = nullptr;
    long long position = 0;
    for (const std::size_t i : queued)
      merged[i]["position"] = ++position;
  }
  if (merged.size() > limit)
    merged.resize(limit);
  json::Value data = json::Value::array();
  for (json::Value& job : merged)
    data.push_back(std::move(job));
  json::Value body = json::Value::object();
  body["object"] = "list";
  body["scope"] = "pool";
  body["partner"] = partner;
  body["data"] = std::move(data);
  return JsonResponse(200, "OK", body);
}

HttpResponse JobQueue::RelayJob(const std::string& id, std::string_view tail,
                                const HttpRequest& request) {
  const bool is_file = tail.starts_with("/files/");
  const bool is_thumb = tail == "/thumb";
  if (!(tail.empty() || tail == "/keep" || tail == "/cancel" || is_file ||
        is_thumb))
    return Err(404, "Not Found", "no route for this path", "not_found");
  HttpRequest copy = request;
  copy.headers.emplace_back(kRelayHeader, "1");
  const auto unreachable = [&](const std::string& why) {
    Logger::Warn("jobs", "event=relay_failed id=" + id + " partner=" +
                             options_.partner_name + " error=" + why);
    return Err(502, "Bad Gateway",
               "partner router " + options_.partner_name +
                   " not reachable: " + why,
               "partner_unreachable");
  };
  std::string why;
  if (is_thumb) {
    // Cached under relay-cache/<id>/, removed a day after the last use.
    const std::string file = request.query_param("file");
    if (!file.empty() && (!ValidFileName(file) || IsKeptFile(file)))
      return Err(404, "Not Found", "file not found", "file_not_found");
    const int width = ThumbWidth(request);
    const fs::path dir = options_.root / "relay-cache" / id;
    for (const char* ext : {"jpg", "png"}) {
      const fs::path cached = dir / ThumbName(width, file, "", ext);
      std::error_code ec;
      if (fs::is_regular_file(cached, ec)) {
        fs::last_write_time(cached, fs::file_time_type::clock::now(), ec);
        return HttpResponse{
            .status = 200,
            .reason = "OK",
            .body = ReadAll(cached),
            .headers = {{"Content-Type", ThumbContentType(ext)},
                        {"Cache-Control", "public, max-age=86400"},
                        {"X-Gufo-Thumb", "relay-cached"}}};
      }
    }
    auto answer = options_.relay(copy, std::chrono::seconds(60),
                                 std::chrono::minutes(10), &why);
    if (!answer)
      return unreachable(why);
    if (answer->status != 200) {
      StripTransportHeaders(*answer);
      return std::move(*answer);
    }
    const std::string type = MediaType(ResponseContentType(*answer));
    const std::string body = CollectBody(*answer);
    const std::string ext = type == "image/png" ? "png" : "jpg";
    try {
      WriteAll(dir / ThumbName(width, file, "", ext), body);
    } catch (const std::exception& failure) {
      Logger::Warn("jobs", std::string("event=relay_cache_write_failed id=") +
                               id + " error=" + failure.what());
    }
    return HttpResponse{.status = 200,
                        .reason = "OK",
                        .body = body,
                        .headers = {{"Content-Type", ThumbContentType(ext)},
                                    {"Cache-Control", "public, max-age=86400"},
                                    {"X-Gufo-Thumb", "relayed"}}};
  }
  auto answer = is_file ? options_.relay(copy, std::chrono::seconds(60),
                                         std::chrono::minutes(10), &why)
                        : options_.relay(copy, std::chrono::seconds(10),
                                         std::chrono::seconds(10), &why);
  if (!answer)
    return unreachable(why);
  StripTransportHeaders(*answer);
  return std::move(*answer);
}

void JobQueue::ExpireRelayCache() {
  const fs::path root = options_.root / "relay-cache";
  std::error_code ec;
  if (!fs::is_directory(root, ec))
    return;
  const auto cutoff =
      fs::file_time_type::clock::now() - std::chrono::hours(24);
  std::vector<fs::path> stale;
  for (const auto& entry : fs::directory_iterator(root, ec)) {
    if (!entry.is_directory(ec))
      continue;
    bool fresh = false;
    for (const auto& file : fs::directory_iterator(entry.path(), ec)) {
      const auto written = fs::last_write_time(file.path(), ec);
      if (!ec && written > cutoff) {
        fresh = true;
        break;
      }
    }
    if (!fresh)
      stale.push_back(entry.path());
  }
  for (const fs::path& dir : stale) {
    fs::remove_all(dir, ec);
    if (!ec)
      Logger::Info("jobs", "event=relay_cache_expired id=" +
                               dir.filename().string());
  }
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
  json::Value loras = json::Value::array();
  for (const std::string& name : job.loras)
    loras.push_back(name);
  request["loras"] = std::move(loras);
  view["request"] = std::move(request);
  // Where the work runs: here, on the partner's worker (pulled, or the
  // pool sent it there), or nowhere yet (waiting in the pool).
  const std::string pool_state =
      job.status == "in_progress" ? PoolStateOf(job) : std::string();
  if (job.status != "in_progress" || pool_state == "wait")
    view["running_on"] = nullptr;
  else if (!job.pulled_by.empty())
    view["running_on"] = job.pulled_by;
  else if (pool_state == "remote" && !options_.partner_name.empty())
    view["running_on"] = options_.partner_name;
  else
    view["running_on"] = "local";
  view["pool_wait"] = pool_state == "wait";
  view["priority"] = static_cast<long long>(job.priority);
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
  const std::string media = job.expired_at == 0 ? FirstMediaFile(job.result)
                                                : std::string();
  view["thumb_url"] = media.empty()
                          ? json::Value()
                          : json::Value("/v1/jobs/" + job.id + "/thumb");
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
  record["priority"] = static_cast<long long>(job.priority);
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
  json::Value loras = json::Value::array();
  for (const std::string& name : job.loras)
    loras.push_back(name);
  record["loras"] = std::move(loras);
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
    // A file written before 2.0.3 has no priority: normal.
    job->priority = PriorityOfView(record);
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
    if (const json::Value* loras = record.find("loras");
        loras != nullptr && loras->is_array())
      for (const json::Value& name : loras->items())
        if (name.is_string() && !name.str().empty())
          job->loras.push_back(name.str());
    if (job->status == "queued") {
      InsertPendingLocked(job, /*front=*/false);
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
  // With a partner and capabilities a free runner also looks at the
  // partner's queue (shared queue): every pull_poll while idle.
  const bool may_pull = options_.relay && options_.capabilities;
  for (;;) {
    std::shared_ptr<Job> job;
    long long mine = 0;
    int mine_priority = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      const auto ready = [this] { return stopping_ || !pending_.empty(); };
      if (may_pull)
        wake_.wait_for(lock, options_.pull_poll, ready);
      else
        wake_.wait(lock, ready);
      if (stopping_)
        return;
      ReapPulledLocked();
      if (!pending_.empty()) {
        mine = pending_.front()->created_at;
        mine_priority = pending_.front()->priority;
      }
    }
    // Global order: the partner's first queued job (priority, then age)
    // goes first when it runs before ours and this machine can run it.
    if (may_pull && TryPull(mine_priority, mine))
      continue;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_)
        return;
      if (pending_.empty())
        continue;
      job = pending_.front();
      pending_.pop_front();
      job->status = "in_progress";
      job->started_at = NowSeconds();
      PersistLocked();
    }
    Execute(job);
  }
}

// ----------------------------------------------------------- shared queue

std::optional<std::vector<JobQueue::PartnerJob>> JobQueue::PartnerQueued() {
  const auto now = std::chrono::steady_clock::now();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (partner_queued_at_ != std::chrono::steady_clock::time_point{} &&
        now - partner_queued_at_ < options_.pull_poll)
      return partner_queued_;
  }
  // Queued jobs, and in_progress jobs whose request waits in the
  // partner's pool ("pool_wait": nothing started there yet).
  std::vector<PartnerJob> list;
  bool ok = true;
  for (const char* status : {"queued", "in_progress"}) {
    HttpRequest copy;
    copy.method = "GET";
    copy.path = "/v1/jobs";
    copy.query = std::string("status=") + status + "&limit=1000";
    copy.headers.emplace_back(kRelayHeader, "1");
    std::string why;
    auto answer = options_.relay(copy, std::chrono::seconds(3),
                                 std::chrono::seconds(3), &why);
    if (!answer || answer->status != 200) {
      if (answer)
        CollectBody(*answer);
      ok = false;
      break;
    }
    const std::string body = CollectBody(*answer);
    try {
      const json::Value parsed = json::parse(body);
      const json::Value* data = parsed.find("data");
      if (data == nullptr || !data->is_array()) {
        ok = false;
        break;
      }
      for (const json::Value& item : data->items()) {
        if (!item.is_object())
          continue;
        const bool waiting = item.member_str("status") == "in_progress";
        if (waiting) {
          const json::Value* wait = item.find("pool_wait");
          if (wait == nullptr || !wait->as_bool())
            continue;
        } else if (item.member_str("status") != "queued") {
          continue;
        }
        PartnerJob job;
        job.id = item.member_str("id");
        job.waiting = waiting;
        job.created_at =
            static_cast<long long>(item.member_double("created_at", 0));
        job.priority = PriorityOfView(item);
        if (const json::Value* request = item.find("request");
            request != nullptr && request->is_object()) {
          job.model = request->member_str("model");
          if (const json::Value* loras = request->find("loras");
              loras != nullptr && loras->is_array())
            for (const json::Value& name : loras->items())
              if (name.is_string())
                job.loras.push_back(name.str());
        }
        if (ValidJobId(job.id) && !job.model.empty())
          list.push_back(std::move(job));
      }
    } catch (const std::exception&) {
      ok = false;  // not a job list: unknown
      break;
    }
  }
  std::optional<std::vector<PartnerJob>> result;
  if (ok) {
    // The lists come newest first; the shared queue wants the run order
    // (highest priority first, then the oldest).
    std::reverse(list.begin(), list.end());
    std::stable_sort(list.begin(), list.end(),
                     [](const PartnerJob& a, const PartnerJob& b) {
                       return RunsBefore(a.priority, a.created_at, b.priority,
                                         b.created_at);
                     });
    result = std::move(list);
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  partner_queued_at_ = now;
  partner_queued_ = result;
  return result;
}

bool JobQueue::TryPull(int mine_priority, long long mine_created_at) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (std::chrono::steady_clock::now() < pull_hold_until_)
      return false;
  }
  const auto queued = PartnerQueued();
  if (!queued || queued->empty())
    return false;
  const json::Value mine = options_.capabilities();
  const PartnerJob* pick = nullptr;
  for (const PartnerJob& candidate : *queued) {
    if (mine_created_at > 0 &&
        !RunsBefore(candidate.priority, candidate.created_at, mine_priority,
                    mine_created_at))
      break;  // ours goes first (higher priority, or as old and ours)
    if (!CanRun(mine, candidate.model, candidate.loras, nullptr,
                /*now=*/true))
      continue;
    // The runner is free, but the GPU may not be (12.11.: the .56 pulled
    // while ComfyUI and the film ran there).
    std::string busy_why;
    if (options_.may_pull && !options_.may_pull(candidate.model, &busy_why))
      continue;
    pick = &candidate;
    break;
  }
  if (pick == nullptr)
    return false;
  json::Value body = json::Value::object();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    body["from"] = self_name_.empty() ? options_.self_name : self_name_;
  }
  body["router_id"] = mine.member_str("router_id");
  body["job"] = pick->id;
  body["capabilities"] = mine;
  HttpRequest copy;
  copy.method = "POST";
  copy.path = kPullPath;
  copy.body = body.dump();
  copy.headers.emplace_back("Content-Type", "application/json");
  copy.headers.emplace_back(kRelayHeader, "1");
  std::string why;
  auto answer = options_.relay(copy, std::chrono::seconds(5),
                               std::chrono::seconds(5), &why);
  if (!answer) {
    Logger::Warn("jobs", "event=pool_pull_failed partner=" +
                             options_.partner_name + " error=" + why);
    return false;
  }
  const std::string text = CollectBody(*answer);
  if (answer->status == 404) {
    // The partner runs an older version without the route: as before.
    const std::lock_guard<std::mutex> lock(mutex_);
    pull_hold_until_ = std::chrono::steady_clock::now() + options_.pull_hold;
    if (!pull_hold_logged_) {
      pull_hold_logged_ = true;
      Logger::Warn("jobs", "event=pool_pull_unsupported partner=" +
                               options_.partner_name +
                               " detail=\"no /router/pool/pull (older "
                               "version); pulling paused\"");
    }
    return false;
  }
  if (answer->status != 200) {
    Logger::Warn("jobs", "event=pool_pull_failed partner=" +
                             options_.partner_name + " error=HTTP " +
                             std::to_string(answer->status) + " " +
                             TailOf(text, 200));
    return false;
  }
  std::string id;
  try {
    id = json::parse(text).member_str("job");
  } catch (const std::exception&) {
    return false;
  }
  if (id.empty())
    return false;  // the partner kept it (a runner took it meanwhile)
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    ++pulled_;
    pulling_.push_back(id);
    partner_queued_.reset();
    partner_queued_at_ = {};
  }
  Logger::Info("jobs", "event=pool_pulled id=" + id + " partner=" +
                           options_.partner_name + " wanted=" + pick->id);
  WaitForPulled(id);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::erase(pulling_, id);
  }
  return true;
}

void JobQueue::WaitForPulled(const std::string& id) {
  int failures = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait_for(lock, options_.pull_poll, [this] { return stopping_; });
      if (stopping_)
        return;
    }
    HttpRequest copy;
    copy.method = "GET";
    copy.path = "/v1/jobs/" + id;
    copy.headers.emplace_back(kRelayHeader, "1");
    std::string why;
    auto answer = options_.relay(copy, std::chrono::seconds(5),
                                 std::chrono::seconds(5), &why);
    if (!answer) {
      if (++failures >= 5) {
        Logger::Warn("jobs", "event=pool_pull_lost id=" + id +
                                 " error=" + why);
        return;
      }
      continue;
    }
    failures = 0;
    const std::string text = CollectBody(*answer);
    if (answer->status != 200)
      return;  // gone (deleted) or unknown: nothing to wait for
    std::string status;
    std::string running_on;
    try {
      const json::Value parsed = json::parse(text);
      status = parsed.member_str("status");
      running_on = parsed.member_str("running_on");
    } catch (const std::exception&) {
      return;
    }
    if (status != "in_progress")
      return;
    std::string me;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      me = self_name_.empty() ? options_.self_name : self_name_;
    }
    // Requeued on the partner, or running there now: no longer ours.
    if (!running_on.empty() && running_on != me &&
        running_on != options_.self_name)
      return;
  }
}

HttpResponse JobQueue::Pull(const HttpRequest& request) {
  if (!options_.capabilities)
    return Err(404, "Not Found", "this router has no shared queue",
               "not_found");
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& error) {
    return Err(400, "Bad Request",
               std::string("request body must be JSON: ") + error.what(),
               "invalid_json");
  }
  const std::string from = body.member_str("from");
  const json::Value* caps = body.find("capabilities");
  if (from.empty() || caps == nullptr || !caps->is_object())
    return Err(400, "Bad Request",
               "send {\"from\": \"http://host:port\", \"capabilities\": "
               "{...}} (GET /router/capabilities of the requester)",
               "invalid_request");
  // A requester bound to 0.0.0.0 names itself so; the job then shows the
  // partner's known URL (1.4.22: "running_on": "http://0.0.0.0:8081").
  const std::string requester = ReachableUrl(from, options_.partner_name);
  const json::Value mine = options_.capabilities();
  const json::Value* models = mine.find("models");
  if (models == nullptr || !models->is_object())
    models = &mine;
  std::shared_ptr<Job> picked;
  bool picked_waiting = false;
  std::size_t queued = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
      return Err(503, "Service Unavailable", "router is shutting down",
                 "shutting_down");
    ReapPulledLocked();
    // Candidates, oldest first: queued jobs (a job a runner already took
    // is not in pending_) and in_progress jobs whose request still waits
    // in the pool for a machine (nothing started yet).
    // order_ is the submit order (created_at has whole seconds only), and
    // a requeued job keeps its place in it.
    std::vector<std::pair<std::shared_ptr<Job>, bool>> candidates;
    std::size_t waiting_count = 0;
    for (const auto& job : order_) {
      if (job->status == "queued") {
        candidates.emplace_back(job, false);
      } else if (job->status == "in_progress" && job->pulled_by.empty() &&
                 PoolStateOf(*job) == "wait") {
        candidates.emplace_back(job, true);
        ++waiting_count;
      }
    }
    // Run order: the highest priority first, within it the submit order.
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) {
                       return a.first->priority > b.first->priority;
                     });
    for (const auto& [job, waiting] : candidates) {
      if (job->path == kConcatPath || job->cancel.load())
        continue;  // runs on this router's ffmpeg
      // Only a model this router has in the pool can be moved: the
      // request runs through the pool decision with X-Gufo-Pool: remote.
      const json::Value* entry = models->find(job->model);
      if (entry == nullptr || !entry->is_object() ||
          entry->member_str("placement") != "pool")
        continue;
      if (!CanRun(*caps, job->model, job->loras, nullptr, /*now=*/true))
        continue;
      if (waiting) {
        // The router redirects the waiting request (503 pool_redirected)
        // only while nothing has started; otherwise the job stays.
        if (!options_.redirect_waiting || !options_.redirect_waiting(job->id))
          continue;
        picked = job;
        picked_waiting = true;
        --waiting_count;
      } else {
        picked = job;
        std::erase(pending_, job);
      }
      break;
    }
    queued = pending_.size() + waiting_count;
    if (picked != nullptr) {
      picked->pulled_by = requester;
      picked->side = "remote";
      ++pushed_;
      if (!picked_waiting) {
        picked->status = "in_progress";
        picked->started_at = NowSeconds();
        auto done = std::make_shared<std::atomic<bool>>(false);
        std::thread thread([this, picked, done] {
          Execute(picked);
          done->store(true);
        });
        pulled_threads_.push_back({std::move(thread), done});
      }
      PersistLocked();
    }
  }
  json::Value out = json::Value::object();
  if (picked != nullptr) {
    out["job"] = picked->id;
    out["model"] = picked->model;
    out["created_at"] = picked->created_at;
    out["priority"] = static_cast<long long>(picked->priority);
    out["was_waiting"] = picked_waiting;
    Logger::Info("jobs", "event=pool_pushed id=" + picked->id + " model=" +
                             picked->model + " to=" + requester +
                             (picked_waiting ? " was=waiting" : " was=queued"));
  } else {
    out["job"] = nullptr;
  }
  out["queued"] = static_cast<long long>(queued);
  return JsonResponse(200, "OK", out);
}

std::string JobQueue::SideFor(const Job& job, json::Value* refusal) {
  if (!options_.capabilities || job.path == kConcatPath)
    return "";
  const json::Value mine = options_.capabilities();
  std::string why_local;
  std::string why_partner;
  const bool local_ok = CanRun(mine, job.model, job.loras, &why_local);
  const bool has_partner = static_cast<bool>(options_.relay);
  std::optional<json::Value> partner;
  if (has_partner && options_.partner_capabilities)
    partner = options_.partner_capabilities();
  // Unknown partner capabilities (unreachable, older version): it may.
  bool partner_ok = has_partner;
  if (partner.has_value()) {
    partner_ok = CanRun(*partner, job.model, job.loras, &why_partner);
    const std::string me = partner->member_str("partner");
    if (!me.empty()) {
      const std::lock_guard<std::mutex> lock(mutex_);
      self_name_ = me;
    }
  }
  if (local_ok && (partner_ok || !has_partner))
    return "";  // any side (or the only one)
  if (local_ok)
    return "local";
  if (partner_ok)
    return "remote";
  if (refusal != nullptr) {
    std::string message = "no router can run this job: this router " +
                          why_local;
    if (has_partner)
      message += "; partner " + options_.partner_name + " " + why_partner;
    json::Value error = json::Value::object();
    error["message"] = message;
    error["code"] = "no_capable_router";
    *refusal = std::move(error);
  }
  return "";
}

void JobQueue::ReapPulledLocked() {
  for (auto it = pulled_threads_.begin(); it != pulled_threads_.end();) {
    if (it->done->load()) {
      if (it->thread.joinable())
        it->thread.join();
      it = pulled_threads_.erase(it);
    } else {
      ++it;
    }
  }
}

std::string JobQueue::PoolStateOf(const Job& job) const {
  if (!options_.pool_states || job.status != "in_progress" ||
      !job.upstream_id.empty())
    return "";
  const std::map<std::string, std::string> states = options_.pool_states();
  const auto it = states.find(job.id);
  return it == states.end() ? std::string() : it->second;
}

FetchResult JobQueue::Post(const std::shared_ptr<Job>& job,
                           const std::string& body) {
  for (int attempt = 0;; ++attempt) {
    FetchResult result =
        Call(job, "POST", job->path, "", body, job->content_type);
    if (result.status == 503 && ErrorCode(result) == kPoolRedirected &&
        attempt < 3) {
      // Handed to the partner while it waited: the same request again,
      // now with X-Gufo-Pool: remote (Call reads the side).
      std::string to;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        to = job->pulled_by;
      }
      Logger::Info("jobs", "event=job_redirected id=" + job->id + " to=" + to);
      continue;
    }
    return result;
  }
}

json::Value JobQueue::PoolQueueStatus() {
  json::Value out = json::Value::object();
  const std::lock_guard<std::mutex> lock(mutex_);
  long long waiting = 0;
  for (const auto& job : order_)
    if (PoolStateOf(*job) == "wait")
      ++waiting;
  out["local"] = static_cast<long long>(pending_.size()) + waiting;
  out["local_waiting"] = waiting;
  out["partner"] = partner_queued_.has_value()
                       ? json::Value(static_cast<long long>(
                             partner_queued_->size()))
                       : json::Value();
  out["pulled"] = pulled_;
  out["pushed"] = pushed_;
  json::Value pulling = json::Value::array();
  for (const std::string& id : pulling_)
    pulling.push_back(id);
  out["pulling"] = std::move(pulling);
  json::Value pushed = json::Value::array();
  for (const auto& job : order_)
    if (job->status == "in_progress" && !job->pulled_by.empty())
      pushed.push_back(job->id);
  out["pushed_active"] = std::move(pushed);
  const auto now = std::chrono::steady_clock::now();
  if (now < pull_hold_until_)
    out["pull_hold_s"] = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::seconds>(pull_hold_until_ -
                                                         now)
            .count());
  out["partner_name"] = options_.partner_name.empty()
                            ? json::Value()
                            : json::Value(options_.partner_name);
  out["enabled"] = static_cast<bool>(options_.relay) &&
                   static_cast<bool>(options_.capabilities);
  return out;
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
  request.headers.emplace_back(kJobHeader, job->id);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!job->side.empty())
      request.headers.emplace_back(kPoolSideHeader, job->side);
    // The pool decision (here and on the partner) sees the priority.
    if (job->priority > 0)
      request.headers.emplace_back(kPriorityHeader,
                                   std::to_string(job->priority));
  }
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
  {
    // Which side of the pool must run it (a pulled job is already set).
    bool decide = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      decide = job->side.empty();
    }
    if (decide) {
      json::Value refusal;
      const std::string side = SideFor(*job, &refusal);
      if (!refusal.is_null()) {
        Finish(job, "failed", json::Value(), std::move(refusal));
        return;
      }
      const std::lock_guard<std::mutex> lock(mutex_);
      job->side = side;
    }
  }
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
      result = Post(job, as_async);
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
      result = Post(job, with_preview);
      progress.request_stop();
      progress.join();
      if (result.status == 400 &&
          ErrorMessage(result).find("preview") != std::string::npos) {
        // A worker without live previews renders as before.
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          job->preview_id.clear();
        }
        result = Post(job, payload);
      }
    } else {
      result = Post(job, payload);
    }
    if (result.status < 200 || result.status >= 300)
      throw JobFailure(ErrorMessage(result), ErrorCode(result), result.status);
    if (job->path == "/v1/videos" || job->path == "/v1/videos/upscale")
      RunVideo(job, result);  // both answer 202 with a worker video job
    else if (job->path == "/v1/3d/generations")
      RunObjekt(job, result);  // 202 with a worker 3D job (gufo serve objekt)
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
          job->pulled_by.clear();
          job->side.clear();
          InsertPendingLocked(job, /*front=*/true);
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

// gufo serve objekt: poll /v1/3d/generations/<id> until completed, then fetch
// the GLB (model.glb) and the preview (preview.png) into the job directory.
void JobQueue::RunObjekt(const std::shared_ptr<Job>& job,
                         const FetchResult& created) {
  std::string objekt_id;
  try {
    objekt_id = json::parse(created.body).member_str("id");
  } catch (const std::exception&) {
  }
  if (objekt_id.empty())
    throw JobFailure("the 3D response has no id", "upstream_error",
                     created.status);
  // The id goes into the request path: word characters only (the worker's
  // ids are obj_<24 hex>).
  if (!std::all_of(objekt_id.begin(), objekt_id.end(), [](unsigned char c) {
        return std::isalnum(c) != 0 || c == '_' || c == '-';
      }))
    throw JobFailure("the 3D response has an invalid id", "upstream_error",
                     created.status);
  const std::string base_path = "/v1/3d/generations/" + objekt_id;
  // Status, content and cancel carry no model in the body: the router routes
  // them to the worker by the `model` query (like the music jobs), otherwise
  // it answers "no route for this path" (12.11., 1.4.23b).
  const std::string query = "model=" + job->model;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    job->upstream_id = objekt_id;
    job->progress = 0;
    PersistLocked();
  }
  // The cancel carries the job and pool-side headers like Call() (1.4.24);
  // it is built here because Call() aborts at once for a cancelled job.
  const auto cancel_worker = [&] {
    HttpRequest cancel;
    cancel.method = "DELETE";
    cancel.path = base_path;
    cancel.query = query;
    cancel.request_id = job->id;
    cancel.headers.emplace_back(kJobHeader, job->id);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!job->side.empty())
        cancel.headers.emplace_back(kPoolSideHeader, job->side);
    }
    try {
      (void)options_.fetch(cancel);
    } catch (const std::exception&) {
    }
  };
  const auto started = std::chrono::steady_clock::now();
  // Transient status answers in a row (5xx while the worker reloads or a
  // relay hiccups, a connection error, a body that is no JSON).
  int transient = 0;
  json::Value final_status;
  for (;;) {
    const auto until = std::chrono::steady_clock::now() + options_.video_poll;
    while (!job->cancel.load() && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (job->cancel.load()) {
      cancel_worker();
      throw Cancelled();
    }
    if (options_.objekt_deadline.count() > 0 &&
        std::chrono::steady_clock::now() - started >=
            options_.objekt_deadline) {
      // A worker stuck on in_progress must not hold the job forever.
      cancel_worker();
      throw JobFailure(
          "the 3D job did not finish within " +
              std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                 options_.objekt_deadline)
                                 .count()) +
              " s",
          "objekt_timeout", 504);
    }
    // Over Call(): job and pool-side headers, and a cancel interrupts a
    // poll that hangs.
    FetchResult state;
    std::string problem;
    json::Value parsed;
    try {
      state = Call(job, "GET", base_path, query, "", "");
      if (state.status >= 500) {
        problem = "status " + std::to_string(state.status) + ": " +
                  ErrorMessage(state);
      } else if (state.status != 200) {
        throw JobFailure("3D status failed: " + ErrorMessage(state),
                         ErrorCode(state), state.status);
      } else {
        try {
          parsed = json::parse(state.body);
        } catch (const std::exception& error) {
          problem = std::string("no JSON status: ") + error.what();
        }
      }
    } catch (const Cancelled&) {
      cancel_worker();
      throw;
    } catch (const JobFailure&) {
      throw;
    } catch (const std::exception& error) {
      problem = error.what();
    }
    if (!problem.empty()) {
      if (++transient > options_.objekt_poll_retries)
        throw JobFailure("3D status failed " + std::to_string(transient) +
                             " times in a row: " + problem,
                         state.status >= 500 ? ErrorCode(state)
                                             : std::string("upstream_error"),
                         state.status >= 500 ? state.status : 502);
      Logger::Warn("jobs", "event=objekt_poll_retry job=" + job->id +
                               " attempt=" + std::to_string(transient) +
                               " detail=\"" + problem + "\"");
      continue;
    }
    transient = 0;
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
          "3D " + status +
              (error != nullptr && error->is_object()
                   ? ": " + error->member_str("message")
                   : std::string()),
          "objekt_" + status, 200);
    }
  }
  const FetchResult content =
      Call(job, "GET", base_path + "/content", query, "", "");
  if (content.status != 200 || content.body.empty())
    throw JobFailure("3D content failed: " + ErrorMessage(content),
                     ErrorCode(content), content.status);
  const std::string base = "/v1/jobs/" + job->id + "/files/";
  json::Value files = json::Value::array();
  const auto add_file = [&](const std::string& name, const std::string& body,
                            const std::string& type) {
    WriteAll(JobDir(job->id) / name, body);
    json::Value file = json::Value::object();
    file["name"] = name;
    file["url"] = base + name;
    file["content_type"] = type;
    file["bytes"] = static_cast<long long>(body.size());
    files.push_back(std::move(file));
  };
  add_file("model.glb", content.body, "model/gltf-binary");
  try {
    const FetchResult preview =
        Call(job, "GET", base_path + "/preview", query, "", "");
    if (preview.status == 200 && !preview.body.empty())
      add_file("preview.png", preview.body, "image/png");
  } catch (const std::exception&) {
  }
  add_file("response.json", final_status.dump(), "application/json");
  json::Value result = json::Value::object();
  result["status_code"] = created.status;
  result["content_type"] = "model/gltf-binary";
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
    for (std::size_t i = 0; i < clips.size(); ++i) {
      list += "file '" + clips[i].filename().string() + "'\n";
      // Offset the next clip by this one's video length, not by its
      // (slightly longer) audio: keeps the frame timing regular across
      // the seam (11.11.: r_frame_rate 120/1 from a 30 ms gap).
      if (infos[i].video_duration > 0) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.6f", infos[i].video_duration);
        list += std::string("duration ") + buf + "\n";
      }
    }
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
    ExpireRelayCache();
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
