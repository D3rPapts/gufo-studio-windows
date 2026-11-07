#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/models/rife/video_io.hpp"

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif
#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

// FFmpeg runs as a child with anonymous pipes on stdin (encoder) or stdout
// (decoder); the program is looked up like CreateProcess does, so the copy
// beside gufo.exe wins. GUFO_FFMPEG / GUFO_FFPROBE override the names.

namespace gufo::models::rife {
namespace {

constexpr DWORD kPipeBytes = 4U << 20U;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

std::string WindowsMessage(DWORD code) {
  return std::system_category().message(static_cast<int>(code));
}

const char* Program(const char* variable, const char* fallback) {
  const char* value = std::getenv(variable);
  return value != nullptr && *value != '\0' ? value : fallback;
}

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

class Handle {
public:
  Handle() = default;
  explicit Handle(HANDLE handle)
      : handle_(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
  ~Handle() { Reset(); }
  Handle(Handle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      Reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
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

// CommandLineToArgvW quoting.
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

Handle InheritableNul(DWORD access) {
  SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
  return Handle(CreateFileW(L"NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &attributes, OPEN_EXISTING, 0, nullptr));
}

Handle InheritableStandard(DWORD which) {
  const HANDLE source = GetStdHandle(which);
  HANDLE copy = nullptr;
  if (source != nullptr && source != INVALID_HANDLE_VALUE &&
      DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &copy,
                      0, TRUE, DUPLICATE_SAME_ACCESS))
    return Handle(copy);
  return InheritableNul(GENERIC_WRITE);
}

// Starts `program` with stdin `input` and stdout `output` (NUL when null) and
// the parent's stderr; only these three handles are inherited.
bool Spawn(const char* program, const std::vector<std::wstring>& arguments,
           HANDLE input, HANDLE output, Handle* process, std::string* error) {
  std::wstring line;
  AppendArgument(&line, Widen(program));
  for (const std::wstring& argument : arguments)
    AppendArgument(&line, argument);
  Handle nul_in, nul_out;
  if (input == nullptr) {
    nul_in = InheritableNul(GENERIC_READ);
    input = nul_in.get();
  }
  if (output == nullptr) {
    nul_out = InheritableNul(GENERIC_WRITE);
    output = nul_out.get();
  }
  Handle errors = InheritableStandard(STD_ERROR_HANDLE);
  if (input == nullptr || output == nullptr || !errors) {
    SetError(error, std::string("cannot start ") + program + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  std::array<HANDLE, 3> inherited = {input, output, errors.get()};
  SIZE_T attribute_bytes = 0;
  (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
  std::vector<unsigned char> storage(attribute_bytes);
  auto* attributes =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
  if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes)) {
    SetError(error, std::string("cannot start ") + program + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input;
  startup.StartupInfo.hStdOutput = output;
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
    SetError(error, std::string("cannot start ") + program + ": " +
                        WindowsMessage(failure));
    return false;
  }
  CloseHandle(info.hThread);
  *process = Handle(info.hProcess);
  return true;
}

// A pipe whose `child_end` is inheritable and whose parent end is not.
bool CreateChildPipe(bool child_reads, Handle* parent_end, Handle* child_end,
                     std::string* error) {
  SECURITY_ATTRIBUTES inheritable{sizeof(inheritable), nullptr, TRUE};
  HANDLE read_end = nullptr, write_end = nullptr;
  if (!CreatePipe(&read_end, &write_end, &inheritable, kPipeBytes)) {
    SetError(error,
             "cannot create FFmpeg pipe: " + WindowsMessage(GetLastError()));
    return false;
  }
  *parent_end = Handle(child_reads ? write_end : read_end);
  *child_end = Handle(child_reads ? read_end : write_end);
  SetHandleInformation(parent_end->get(), HANDLE_FLAG_INHERIT, 0);
  return true;
}

int WaitExit(HANDLE process) {
  DWORD code = 1;
  if (WaitForSingleObject(process, INFINITE) != WAIT_OBJECT_0 ||
      !GetExitCodeProcess(process, &code))
    return -1;
  return static_cast<int>(code);
}

bool ParseInt(std::string_view text, int* out) {
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return ec == std::errc{} && end == text.data() + text.size();
}

// Single-image output ("*.png", any case) instead of an H.264 MP4.
bool IsPngPath(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  for (char& c : extension)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return extension == ".png";
}

}  // namespace

bool ProbeVideo(const std::filesystem::path& path, VideoInfo* info,
                std::string* error) {
  *info = {};
  Handle reader, writer;
  if (!CreateChildPipe(false, &reader, &writer, error))
    return false;
  Handle child;
  const char* program = Program("GUFO_FFPROBE", GUFO_FFPROBE_EXECUTABLE);
  if (!Spawn(program,
             {L"-v", L"error", L"-show_entries",
              L"stream=codec_type,width,height,r_frame_rate,nb_frames", L"-of",
              L"default=noprint_wrappers=1", path.wstring()},
             nullptr, writer.get(), &child, error))
    return false;
  writer.Reset();
  std::string output;
  std::array<char, 4096> buffer{};
  DWORD amount = 0;
  while (ReadFile(reader.get(), buffer.data(),
                  static_cast<DWORD>(buffer.size()), &amount, nullptr) &&
         amount != 0) {
    if (output.size() + amount > (1U << 20U)) {
      TerminateProcess(child.get(), 1);
      break;
    }
    output.append(buffer.data(), amount);
  }
  if (WaitExit(child.get()) != 0) {
    SetError(error, "ffprobe cannot read " + path.string());
    return false;
  }
  // Streams print codec_type first; the first video stream wins.
  std::string_view text = output;
  std::string type;
  bool video_done = false, in_video = false;
  while (!text.empty()) {
    const std::size_t newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    text = newline == std::string_view::npos ? std::string_view{}
                                             : text.substr(newline + 1);
    const std::size_t equals = line.find('=');
    if (equals == std::string_view::npos)
      continue;
    const std::string_view key = line.substr(0, equals);
    const std::string_view value = line.substr(equals + 1);
    if (key == "codec_type") {
      if (in_video)
        video_done = true;
      in_video = value == "video" && !video_done;
      if (value == "audio")
        info->has_audio = true;
      continue;
    }
    if (!in_video)
      continue;
    if (key == "width")
      (void)ParseInt(value, &info->width);
    else if (key == "height")
      (void)ParseInt(value, &info->height);
    else if (key == "nb_frames")
      (void)ParseInt(value, &info->frames);
    else if (key == "r_frame_rate") {
      const std::size_t slash = value.find('/');
      if (slash == std::string_view::npos) {
        (void)ParseInt(value, &info->fps_num);
        info->fps_den = 1;
      } else {
        (void)ParseInt(value.substr(0, slash), &info->fps_num);
        (void)ParseInt(value.substr(slash + 1), &info->fps_den);
      }
    }
  }
  if (info->width < 2 || info->height < 2 || info->fps_num < 1 ||
      info->fps_den < 1) {
    SetError(error, "no usable video stream in " + path.string());
    return false;
  }
  return true;
}

struct FrameReader::Impl {
  Handle pipe;
  Handle child;
  std::size_t frame_bytes{0};
};

FrameReader::FrameReader() : impl_(std::make_unique<Impl>()) {}
FrameReader::~FrameReader() {
  Close();
}

bool FrameReader::Open(const std::filesystem::path& path, int width, int height,
                       std::string* error) {
  Close();
  Handle child_end;
  if (!CreateChildPipe(false, &impl_->pipe, &child_end, error))
    return false;
  if (!Spawn(
          Program("GUFO_FFMPEG", GUFO_FFMPEG_EXECUTABLE),
          {L"-nostdin", L"-loglevel", L"error", L"-i", path.wstring(), L"-map",
           L"0:v:0", L"-f", L"rawvideo", L"-pix_fmt", L"rgb24", L"pipe:1"},
          nullptr, child_end.get(), &impl_->child, error))
    return false;
  impl_->frame_bytes = static_cast<std::size_t>(width) * height * 3;
  return true;
}

bool FrameReader::Read(std::uint8_t* rgb24, bool* end, std::string* error) {
  *end = false;
  std::size_t done = 0;
  while (done < impl_->frame_bytes) {
    DWORD amount = 0;
    const DWORD request = static_cast<DWORD>(
        std::min<std::size_t>(impl_->frame_bytes - done, kPipeBytes));
    if (!ReadFile(impl_->pipe.get(), rgb24 + done, request, &amount, nullptr) ||
        amount == 0) {
      const DWORD code = GetLastError();
      if (done == 0 && (amount == 0 || code == ERROR_BROKEN_PIPE)) {
        *end = true;
        return false;
      }
      SetError(error, "FFmpeg decoder stopped inside a frame");
      return false;
    }
    done += amount;
  }
  return true;
}

void FrameReader::Close() {
  // A decoder stopped early is terminated before its pipe closes, so it does
  // not report the broken pipe.
  if (impl_->child) {
    if (WaitForSingleObject(impl_->child.get(), 0) != WAIT_OBJECT_0)
      TerminateProcess(impl_->child.get(), 1);
    (void)WaitExit(impl_->child.get());
    impl_->child.Reset();
  }
  impl_->pipe.Reset();
}

struct FrameWriter::Impl {
  Handle pipe;
  Handle child;
  std::filesystem::path path;
  std::filesystem::path partial;
  std::size_t frame_bytes{0};
  int crf{18};
};

FrameWriter::FrameWriter() : impl_(std::make_unique<Impl>()) {}

void FrameWriter::SetCrf(int crf) {
  impl_->crf = crf < 0 ? 0 : (crf > 51 ? 51 : crf);
}
FrameWriter::~FrameWriter() {
  Abort();
}

bool FrameWriter::Open(const std::filesystem::path& path, int width, int height,
                       int fps_num, int fps_den,
                       const std::filesystem::path& audio_source,
                       std::string* error) {
  Abort();
  const bool png = IsPngPath(path);
  if (!png && ((width & 1) != 0 || (height & 1) != 0)) {
    SetError(error, "H.264 yuv420p output needs an even width and height");
    return false;
  }
  static std::atomic<std::uint64_t> sequence{0};
  impl_->path = path;
  impl_->partial =
      path.parent_path() / (path.filename().wstring() + L".gufo-partial-" +
                            std::to_wstring(GetCurrentProcessId()) + L"-" +
                            std::to_wstring(sequence.fetch_add(1)) +
                            (png ? L".png" : L".mp4"));
  std::error_code ignored;
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path(), ignored);
  std::vector<std::wstring> arguments = {
      L"-y",
      L"-nostdin",
      L"-loglevel",
      L"error",
      L"-f",
      L"rawvideo",
      L"-pixel_format",
      L"rgb24",
      L"-video_size",
      std::to_wstring(width) + L"x" + std::to_wstring(height),
      L"-framerate",
      std::to_wstring(fps_num) + L"/" + std::to_wstring(fps_den),
      L"-i",
      L"pipe:0"};
  if (!audio_source.empty()) {
    arguments.emplace_back(L"-i");
    arguments.push_back(audio_source.wstring());
    for (const wchar_t* a :
         {L"-map", L"0:v:0", L"-map", L"1:a:0?", L"-c:a", L"copy"})
      arguments.emplace_back(a);
  }
  if (png) {
    for (const wchar_t* a : {L"-frames:v", L"1", L"-c:v", L"png"})
      arguments.emplace_back(a);
  } else {
    for (const wchar_t* a : {L"-c:v", L"libx264", L"-preset", L"fast", L"-crf"})
      arguments.emplace_back(a);
    arguments.push_back(std::to_wstring(impl_->crf));
    for (const wchar_t* a :
         {L"-pix_fmt", L"yuv420p", L"-movflags", L"+faststart"})
      arguments.emplace_back(a);
  }
  arguments.push_back(impl_->partial.wstring());
  Handle child_end;
  if (!CreateChildPipe(true, &impl_->pipe, &child_end, error))
    return false;
  if (!Spawn(Program("GUFO_FFMPEG", GUFO_FFMPEG_EXECUTABLE), arguments,
             child_end.get(), nullptr, &impl_->child, error)) {
    impl_->pipe.Reset();
    return false;
  }
  impl_->frame_bytes = static_cast<std::size_t>(width) * height * 3;
  return true;
}

bool FrameWriter::Write(const std::uint8_t* rgb24, std::string* error) {
  std::size_t done = 0;
  while (done < impl_->frame_bytes) {
    DWORD amount = 0;
    const DWORD request = static_cast<DWORD>(
        std::min<std::size_t>(impl_->frame_bytes - done, kPipeBytes));
    if (!WriteFile(impl_->pipe.get(), rgb24 + done, request, &amount,
                   nullptr) ||
        amount == 0) {
      SetError(error,
               "FFmpeg encoder stopped: " + WindowsMessage(GetLastError()));
      return false;
    }
    done += amount;
  }
  return true;
}

bool FrameWriter::Finish(std::string* error) {
  impl_->pipe.Reset();
  if (!impl_->child) {
    SetError(error, "FFmpeg encoder is not running");
    return false;
  }
  const int status = WaitExit(impl_->child.get());
  impl_->child.Reset();
  std::error_code fs;
  if (status != 0) {
    std::filesystem::remove(impl_->partial, fs);
    SetError(error,
             "FFmpeg encoder exited with status " + std::to_string(status));
    return false;
  }
  std::filesystem::rename(impl_->partial, impl_->path, fs);
  if (fs) {
    std::error_code ignored;
    std::filesystem::remove(impl_->partial, ignored);
    SetError(error,
             "cannot publish " + impl_->path.string() + ": " + fs.message());
    return false;
  }
  return true;
}

void FrameWriter::Abort() {
  impl_->pipe.Reset();
  if (impl_->child) {
    TerminateProcess(impl_->child.get(), 1);
    (void)WaitExit(impl_->child.get());
    impl_->child.Reset();
    std::error_code ignored;
    std::filesystem::remove(impl_->partial, ignored);
  }
}

}  // namespace gufo::models::rife
