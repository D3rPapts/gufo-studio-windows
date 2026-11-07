#ifndef GUFO_MODELS_RIFE_VIDEO_IO_HPP_
#define GUFO_MODELS_RIFE_VIDEO_IO_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace gufo::models::rife {

struct VideoInfo {
  int width{0};
  int height{0};
  int fps_num{0};  // r_frame_rate as a fraction
  int fps_den{1};
  int frames{-1};  // nb_frames when the container states it
  bool has_audio{false};
};

/// FFprobe of the first video stream and whether an audio stream exists.
[[nodiscard]] bool ProbeVideo(const std::filesystem::path& path,
                              VideoInfo* info, std::string* error);

/// Streams decoded RGB24 frames from FFmpeg (`-f rawvideo -pix_fmt rgb24`).
class FrameReader {
public:
  FrameReader();
  ~FrameReader();
  FrameReader(const FrameReader&) = delete;
  FrameReader& operator=(const FrameReader&) = delete;

  bool Open(const std::filesystem::path& path, int width, int height,
            std::string* error);
  /// Reads one frame; returns false at the end (with *end = true) or on an
  /// error (with *end = false).
  bool Read(std::uint8_t* rgb24, bool* end, std::string* error);
  void Close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Encodes RGB24 frames to H.264 MP4 (CRF 18 unless SetCrf, yuv420p) through
/// FFmpeg's stdin. When `audio_source` is not empty its first audio stream is
/// copied unchanged. A `path` ending in ".png" writes the first frame as a
/// PNG instead. The file appears at `path` only after FFmpeg succeeded.
class FrameWriter {
public:
  FrameWriter();
  ~FrameWriter();
  FrameWriter(const FrameWriter&) = delete;
  FrameWriter& operator=(const FrameWriter&) = delete;

  bool Open(const std::filesystem::path& path, int width, int height,
            int fps_num, int fps_den, const std::filesystem::path& audio_source,
            std::string* error);
  bool Write(const std::uint8_t* rgb24, std::string* error);
  /// H.264 constant rate factor (0..51) for the next Open.
  void SetCrf(int crf);
  /// Closes FFmpeg's input, waits and publishes the file.
  bool Finish(std::string* error);
  /// Kills FFmpeg and removes the partial file.
  void Abort();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::rife

#endif  // GUFO_MODELS_RIFE_VIDEO_IO_HPP_
