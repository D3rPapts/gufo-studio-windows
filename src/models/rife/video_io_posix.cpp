#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <system_error>
#include <vector>

#include "src/models/rife/video_io.hpp"

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif
#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

extern char** environ;  // NOLINT

namespace gufo::models::rife {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

const char* Program(const char* variable, const char* fallback) {
  const char* value = std::getenv(variable);
  return value != nullptr && *value != '\0' ? value : fallback;
}

// Starts `program` with `arguments`; `child_stdin` / `child_stdout` replace
// the standard descriptors when >= 0 (else /dev/null). The parent ends given
// in `close_in_child` are closed in the child.
bool Spawn(const char* program, const std::vector<std::string>& arguments,
           int child_stdin, int child_stdout, int close_in_child, pid_t* pid,
           std::string* error) {
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  if (child_stdin >= 0)
    posix_spawn_file_actions_adddup2(&actions, child_stdin, STDIN_FILENO);
  else
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                     O_RDONLY, 0);
  if (child_stdout >= 0)
    posix_spawn_file_actions_adddup2(&actions, child_stdout, STDOUT_FILENO);
  else
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                     O_WRONLY, 0);
  if (close_in_child >= 0)
    posix_spawn_file_actions_addclose(&actions, close_in_child);
  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(program));
  for (const std::string& a : arguments)
    argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  const int status =
      posix_spawnp(pid, program, &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (status != 0) {
    SetError(error, std::string("cannot start ") + program + ": " +
                        std::strerror(status));
    return false;
  }
  return true;
}

int WaitExit(pid_t pid) {
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR)
      return -1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

bool ParseInt(std::string_view text, int* out) {
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return ec == std::errc{} && end == text.data() + text.size();
}

bool MakePipe(int fds[2], std::string* error) {
  if (pipe(fds) != 0) {
    SetError(error,
             std::string("cannot create FFmpeg pipe: ") + std::strerror(errno));
    return false;
  }
  fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  fcntl(fds[1], F_SETFD, FD_CLOEXEC);
  return true;
}

}  // namespace

bool ProbeVideo(const std::filesystem::path& path, VideoInfo* info,
                std::string* error) {
  *info = {};
  int fds[2];
  if (!MakePipe(fds, error))
    return false;
  pid_t pid = 0;
  const bool started =
      Spawn(Program("GUFO_FFPROBE", GUFO_FFPROBE_EXECUTABLE),
            {"-v", "error", "-show_entries",
             "stream=codec_type,width,height,r_frame_rate,nb_frames", "-of",
             "default=noprint_wrappers=1", path.string()},
            -1, fds[1], fds[0], &pid, error);
  close(fds[1]);
  if (!started) {
    close(fds[0]);
    return false;
  }
  std::string output;
  char buffer[4096];
  ssize_t amount = 0;
  while ((amount = read(fds[0], buffer, sizeof(buffer))) > 0 &&
         output.size() < (1U << 20U))
    output.append(buffer, static_cast<std::size_t>(amount));
  close(fds[0]);
  if (WaitExit(pid) != 0) {
    SetError(error, "ffprobe cannot read " + path.string());
    return false;
  }
  std::string_view text = output;
  bool video_done = false, in_video = false;
  while (!text.empty()) {
    const std::size_t newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
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
  int fd{-1};
  pid_t pid{0};
  std::size_t frame_bytes{0};
};

FrameReader::FrameReader() : impl_(std::make_unique<Impl>()) {}
FrameReader::~FrameReader() {
  Close();
}

bool FrameReader::Open(const std::filesystem::path& path, int width, int height,
                       std::string* error) {
  Close();
  int fds[2];
  if (!MakePipe(fds, error))
    return false;
  const bool started =
      Spawn(Program("GUFO_FFMPEG", GUFO_FFMPEG_EXECUTABLE),
            {"-nostdin", "-loglevel", "error", "-i", path.string(), "-map",
             "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"},
            -1, fds[1], fds[0], &impl_->pid, error);
  close(fds[1]);
  if (!started) {
    close(fds[0]);
    return false;
  }
  impl_->fd = fds[0];
  impl_->frame_bytes = static_cast<std::size_t>(width) * height * 3;
  return true;
}

bool FrameReader::Read(std::uint8_t* rgb24, bool* end, std::string* error) {
  *end = false;
  std::size_t done = 0;
  while (done < impl_->frame_bytes) {
    const ssize_t amount =
        read(impl_->fd, rgb24 + done, impl_->frame_bytes - done);
    if (amount < 0 && errno == EINTR)
      continue;
    if (amount <= 0) {
      if (done == 0 && amount == 0) {
        *end = true;
        return false;
      }
      SetError(error, "FFmpeg decoder stopped inside a frame");
      return false;
    }
    done += static_cast<std::size_t>(amount);
  }
  return true;
}

void FrameReader::Close() {
  if (impl_->fd >= 0) {
    close(impl_->fd);
    impl_->fd = -1;
  }
  if (impl_->pid > 0) {
    kill(impl_->pid, SIGTERM);
    (void)WaitExit(impl_->pid);
    impl_->pid = 0;
  }
}

struct FrameWriter::Impl {
  int fd{-1};
  pid_t pid{0};
  std::filesystem::path path;
  std::filesystem::path partial;
  std::size_t frame_bytes{0};
};

FrameWriter::FrameWriter() : impl_(std::make_unique<Impl>()) {}
FrameWriter::~FrameWriter() {
  Abort();
}

bool FrameWriter::Open(const std::filesystem::path& path, int width, int height,
                       int fps_num, int fps_den,
                       const std::filesystem::path& audio_source,
                       std::string* error) {
  Abort();
  if ((width & 1) != 0 || (height & 1) != 0) {
    SetError(error, "H.264 yuv420p output needs an even width and height");
    return false;
  }
  static std::atomic<std::uint64_t> sequence{0};
  impl_->path = path;
  impl_->partial =
      path.parent_path() /
      (path.filename().string() + ".gufo-partial-" + std::to_string(getpid()) +
       "-" + std::to_string(sequence.fetch_add(1)) + ".mp4");
  std::error_code ignored;
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path(), ignored);
  std::vector<std::string> arguments = {
      "-y",
      "-nostdin",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pixel_format",
      "rgb24",
      "-video_size",
      std::to_string(width) + "x" + std::to_string(height),
      "-framerate",
      std::to_string(fps_num) + "/" + std::to_string(fps_den),
      "-i",
      "pipe:0"};
  if (!audio_source.empty()) {
    arguments.emplace_back("-i");
    arguments.push_back(audio_source.string());
    for (const char* a : {"-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"})
      arguments.emplace_back(a);
  }
  for (const char* a : {"-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"})
    arguments.emplace_back(a);
  arguments.push_back(impl_->partial.string());
  int fds[2];
  if (!MakePipe(fds, error))
    return false;
  const bool started = Spawn(Program("GUFO_FFMPEG", GUFO_FFMPEG_EXECUTABLE),
                             arguments, fds[0], -1, fds[1], &impl_->pid, error);
  close(fds[0]);
  if (!started) {
    close(fds[1]);
    return false;
  }
  impl_->fd = fds[1];
  impl_->frame_bytes = static_cast<std::size_t>(width) * height * 3;
  return true;
}

bool FrameWriter::Write(const std::uint8_t* rgb24, std::string* error) {
  std::size_t done = 0;
  while (done < impl_->frame_bytes) {
    const ssize_t amount =
        write(impl_->fd, rgb24 + done, impl_->frame_bytes - done);
    if (amount < 0 && errno == EINTR)
      continue;
    if (amount <= 0) {
      SetError(error,
               std::string("FFmpeg encoder stopped: ") + std::strerror(errno));
      return false;
    }
    done += static_cast<std::size_t>(amount);
  }
  return true;
}

bool FrameWriter::Finish(std::string* error) {
  if (impl_->fd >= 0) {
    close(impl_->fd);
    impl_->fd = -1;
  }
  if (impl_->pid <= 0) {
    SetError(error, "FFmpeg encoder is not running");
    return false;
  }
  const int status = WaitExit(impl_->pid);
  impl_->pid = 0;
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
  if (impl_->fd >= 0) {
    close(impl_->fd);
    impl_->fd = -1;
  }
  if (impl_->pid > 0) {
    kill(impl_->pid, SIGTERM);
    (void)WaitExit(impl_->pid);
    impl_->pid = 0;
    std::error_code ignored;
    std::filesystem::remove(impl_->partial, ignored);
  }
}

}  // namespace gufo::models::rife
