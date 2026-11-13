#include "src/cli/serve/music_encode.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif

#if !defined(_WIN32)
extern char** environ;  // NOLINT
#endif

namespace gufo::server {
namespace {

namespace fs = std::filesystem;

// FFmpeg gets this long for a transcode before it is killed; a 10 minute song
// encodes in seconds.
constexpr auto kEncodeTimeout = std::chrono::minutes(10);

const char* FfmpegProgram() {
  const char* override = std::getenv("GUFO_FFMPEG");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFMPEG_EXECUTABLE;
}

std::string Lower(std::string text) {
  for (char& c : text)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

bool WriteFile(const fs::path& path, const std::string& bytes,
               std::string* error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    *error = "cannot write " + path.string();
    return false;
  }
  return true;
}

bool ReadFile(const fs::path& path, std::string* bytes, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot read " + path.string();
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *bytes = buffer.str();
  return true;
}

// The first lines FFmpeg wrote to stderr (`-loglevel error`), for the error.
std::string ErrorTail(const fs::path& log) {
  std::string text;
  std::string ignored;
  if (!ReadFile(log, &text, &ignored))
    return {};
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' ||
                           text.back() == ' '))
    text.pop_back();
  std::replace(text.begin(), text.end(), '\n', ' ');
  std::replace(text.begin(), text.end(), '\r', ' ');
  if (text.size() > 300)
    text = text.substr(0, 300) + "...";
  return text;
}

fs::path TemporaryBase() {
  static std::atomic<std::uint64_t> sequence{0};
#if defined(_WIN32)
  const unsigned long pid = GetCurrentProcessId();
#else
  const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
  std::error_code ignored;
  fs::path dir = fs::temp_directory_path(ignored);
  if (dir.empty())
    dir = fs::current_path(ignored);
  return dir / ("gufo-music-" + std::to_string(pid) + "-" +
                std::to_string(sequence.fetch_add(1)));
}

#if defined(_WIN32)

std::wstring Widen(std::string_view text) {
  if (text.empty())
    return {};
  const int length = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(std::max(length, 0)), L'\0');
  if (length > 0)
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        wide.data(), length);
  return wide;
}

std::string WindowsMessage(DWORD code) {
  return std::system_category().message(static_cast<int>(code));
}

class Handle {
public:
  Handle() = default;
  explicit Handle(HANDLE handle)
      : handle_(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
  ~Handle() { Reset(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  [[nodiscard]] HANDLE get() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }
  void Reset() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
      handle_ = nullptr;
    }
  }

private:
  HANDLE handle_{nullptr};
};

// CommandLineToArgvW quoting: the arguments are passed as a list and quoted
// here, never through a shell.
void AppendArgument(std::wstring* line, std::wstring_view argument) {
  if (!line->empty())
    line->push_back(L' ');
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring_view::npos) {
    line->append(argument);
    return;
  }
  line->push_back(L'"');
  for (auto it = argument.begin();; ++it) {
    std::size_t backslashes = 0;
    while (it != argument.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == argument.end()) {
      line->append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"')
      line->append(backslashes * 2 + 1, L'\\');
    else
      line->append(backslashes, L'\\');
    line->push_back(*it);
  }
  line->push_back(L'"');
}

Handle InheritableFile(const wchar_t* name, DWORD access, DWORD disposition) {
  SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
  return Handle(CreateFileW(name, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &attributes, disposition, FILE_ATTRIBUTE_NORMAL,
                            nullptr));
}

// Runs FFmpeg to completion with stdin/stdout on NUL and stderr in
// `stderr_path`; only those three handles are inherited.
bool RunFfmpeg(const std::vector<std::string>& arguments,
               const fs::path& stderr_path, std::string* error) {
  const char* program = FfmpegProgram();
  std::wstring line;
  AppendArgument(&line, Widen(program));
  for (const std::string& argument : arguments)
    AppendArgument(&line, Widen(argument));
  Handle input = InheritableFile(L"NUL", GENERIC_READ, OPEN_EXISTING);
  Handle output = InheritableFile(L"NUL", GENERIC_WRITE, OPEN_EXISTING);
  Handle errors =
      InheritableFile(stderr_path.c_str(), GENERIC_WRITE, CREATE_ALWAYS);
  if (!input || !output || !errors) {
    *error = std::string("cannot start ") + program + ": " +
             WindowsMessage(GetLastError());
    return false;
  }
  std::array<HANDLE, 3> inherited = {input.get(), output.get(), errors.get()};
  SIZE_T attribute_bytes = 0;
  (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
  std::vector<unsigned char> storage(attribute_bytes);
  auto* attributes =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
  if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes)) {
    *error = std::string("cannot start ") + program + ": " +
             WindowsMessage(GetLastError());
    return false;
  }
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.get();
  startup.StartupInfo.hStdOutput = output.get();
  startup.StartupInfo.hStdError = errors.get();
  startup.lpAttributeList = attributes;
  PROCESS_INFORMATION info{};
  const bool created =
      UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
          sizeof(inherited), nullptr, nullptr) &&
      CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                     EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                     nullptr, &startup.StartupInfo, &info);
  const DWORD failure = created ? ERROR_SUCCESS : GetLastError();
  DeleteProcThreadAttributeList(attributes);
  if (!created) {
    *error = std::string("cannot start ") + program + ": " +
             WindowsMessage(failure);
    return false;
  }
  CloseHandle(info.hThread);
  Handle process(info.hProcess);
  errors.Reset();
  const DWORD waited = WaitForSingleObject(
      process.get(),
      static_cast<DWORD>(
          std::chrono::duration_cast<std::chrono::milliseconds>(kEncodeTimeout)
              .count()));
  if (waited != WAIT_OBJECT_0) {
    TerminateProcess(process.get(), 1);
    WaitForSingleObject(process.get(), 5000);
    *error = std::string(program) + " did not finish in time";
    return false;
  }
  DWORD code = 1;
  if (!GetExitCodeProcess(process.get(), &code))
    code = 1;
  if (code != 0) {
    *error = std::string(program) + " exited with " + std::to_string(code);
    return false;
  }
  return true;
}

#else

bool RunFfmpeg(const std::vector<std::string>& arguments,
               const fs::path& stderr_path, std::string* error) {
  const char* program = FfmpegProgram();
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                   O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                   O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO,
                                   stderr_path.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(program));
  for (const std::string& argument : arguments)
    argv.push_back(const_cast<char*>(argument.c_str()));
  argv.push_back(nullptr);
  pid_t pid = -1;
  const int status =
      posix_spawnp(&pid, program, &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (status != 0) {
    *error = std::string("cannot start ") + program + ": " +
             std::strerror(status);
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + kEncodeTimeout;
  int wait_status = 0;
  for (;;) {
    const pid_t waited = waitpid(pid, &wait_status, WNOHANG);
    if (waited == pid)
      break;
    if (waited < 0 && errno != EINTR) {
      *error = std::string("cannot wait for ") + program + ": " +
               std::strerror(errno);
      return false;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      (void)kill(pid, SIGKILL);
      (void)waitpid(pid, &wait_status, 0);
      *error = std::string(program) + " did not finish in time";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const int code = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1;
  if (code != 0) {
    *error = std::string(program) + " exited with " + std::to_string(code);
    return false;
  }
  return true;
}

#endif

}  // namespace

bool ParseMusicOutputFormat(const json::Value& body, MusicOutputFormat* format,
                            std::string* message, std::string* code) {
  MusicOutputFormat parsed;
  std::string name = "wav";
  if (const json::Value* value = body.find("response_format");
      value != nullptr) {
    if (!value->is_string()) {
      *message = "response_format must be wav, mp3 or aac";
      *code = "unsupported_format";
      return false;
    }
    name = Lower(value->str());
  }
  if (name == "wav") {
    parsed.name = "wav";
    parsed.content_type = "audio/wav";
    parsed.filename = "output.wav";
  } else if (name == "mp3") {
    parsed.name = "mp3";
    parsed.content_type = "audio/mpeg";
    parsed.filename = "output.mp3";
  } else if (name == "aac" || name == "m4a") {
    parsed.name = "m4a";
    parsed.content_type = "audio/mp4";
    parsed.filename = "output.m4a";
  } else {
    *message = "response_format must be wav, mp3 or aac";
    *code = "unsupported_format";
    return false;
  }
  if (const json::Value* value = body.find("bitrate_kbps"); value != nullptr) {
    const double kbps = value->is_number() ? value->as_double() : -1.0;
    if (!value->is_number() || std::floor(kbps) != kbps || kbps < 96 ||
        kbps > 320) {
      *message = "bitrate_kbps must be an integer 96..320";
      *code = "invalid_bitrate";
      return false;
    }
    parsed.bitrate_kbps = static_cast<int>(kbps);
  }
  *format = std::move(parsed);
  return true;
}

std::vector<std::string> MusicEncodeArguments(const MusicOutputFormat& format,
                                              const fs::path& wav,
                                              const fs::path& output) {
  if (!format.needs_ffmpeg())
    return {};
  std::vector<std::string> arguments = {
      "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i",
      wav.string(), "-vn", "-codec:a",
      format.name == "mp3" ? "libmp3lame" : "aac", "-b:a",
      std::to_string(format.bitrate_kbps) + "k"};
  if (format.name == "m4a") {
    arguments.emplace_back("-movflags");
    arguments.emplace_back("+faststart");
  }
  arguments.push_back(output.string());
  return arguments;
}

bool EncodeMusic(const std::string& wav, const MusicOutputFormat& format,
                 std::string* encoded, std::string* error) {
  if (!format.needs_ffmpeg()) {
    *encoded = wav;
    return true;
  }
  const std::string prefix = format.name + " encoding needs ffmpeg: ";
  const fs::path base = TemporaryBase();
  const fs::path wav_path = base.string() + ".wav";
  const fs::path out_path = base.string() + "." + format.name;
  const fs::path log_path = base.string() + ".log";
  std::string detail;
  if (!WriteFile(wav_path, wav, &detail)) {
    *error = prefix + detail;
    return false;
  }
  std::error_code ignored;
  const bool ran =
      RunFfmpeg(MusicEncodeArguments(format, wav_path, out_path), log_path,
                &detail);
  const std::string tail = ErrorTail(log_path);
  if (!ran || !fs::exists(out_path, ignored) ||
      fs::file_size(out_path, ignored) == 0) {
    if (ran)
      detail = "no output written";
    if (!tail.empty())
      detail += " (" + tail + ")";
    // The generated audio is not lost: the WAV stays where FFmpeg found it.
    *error = prefix + detail + "; WAV kept at " + wav_path.string();
    fs::remove(out_path, ignored);
    fs::remove(log_path, ignored);
    return false;
  }
  const bool read = ReadFile(out_path, encoded, &detail);
  fs::remove(out_path, ignored);
  fs::remove(log_path, ignored);
  if (!read) {
    *error = prefix + detail + "; WAV kept at " + wav_path.string();
    return false;
  }
  fs::remove(wav_path, ignored);
  return true;
}

bool DecodeAudioWithFfmpeg(const std::string& bytes, std::string* wav,
                           std::string* error) {
  const std::string prefix = "audio decoding needs ffmpeg: ";
  const fs::path base = TemporaryBase();
  const fs::path in_path = base.string() + ".in";
  const fs::path out_path = base.string() + ".wav";
  const fs::path log_path = base.string() + ".log";
  std::string detail;
  if (!WriteFile(in_path, bytes, &detail)) {
    *error = prefix + detail;
    return false;
  }
  const std::vector<std::string> arguments = {
      "-y",  "-hide_banner", "-nostdin", "-loglevel",  "error", "-i",
      in_path.string(), "-vn", "-f", "wav", "-codec:a", "pcm_s16le",
      "-ar", "48000", "-ac", "2", out_path.string()};
  std::error_code ignored;
  const bool ran = RunFfmpeg(arguments, log_path, &detail);
  const std::string tail = ErrorTail(log_path);
  bool ok = ran && fs::exists(out_path, ignored) &&
            fs::file_size(out_path, ignored) > 44;
  if (ok)
    ok = ReadFile(out_path, wav, &detail);
  else if (ran)
    detail = "no output written";
  if (!ok && !tail.empty())
    detail += " (" + tail + ")";
  fs::remove(in_path, ignored);
  fs::remove(out_path, ignored);
  fs::remove(log_path, ignored);
  if (!ok)
    *error = prefix + detail;
  return ok;
}

}  // namespace gufo::server
