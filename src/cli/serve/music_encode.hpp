#ifndef GUFO_SERVER_MUSIC_ENCODE_HPP_
#define GUFO_SERVER_MUSIC_ENCODE_HPP_

#include <filesystem>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {

/// Container of a finished /v1/audio/music result: the generated WAV as it
/// is, or an FFmpeg transcode of it.
///   response_format  "wav" (default) | "mp3" | "aac" (alias "m4a")
///   bitrate_kbps     constant bitrate for mp3 / aac, 96..320, default 192
///                    (ignored for wav)
struct MusicOutputFormat {
  std::string name{"wav"};  // "wav" | "mp3" | "m4a"
  std::string content_type{"audio/wav"};
  std::string filename{"output.wav"};
  int bitrate_kbps{192};

  [[nodiscard]] bool needs_ffmpeg() const { return name != "wav"; }
};

/// Reads `response_format` and `bitrate_kbps` from a request body. On a bad
/// value returns false with a client message and an error code for a 400.
[[nodiscard]] bool ParseMusicOutputFormat(const json::Value& body,
                                          MusicOutputFormat* format,
                                          std::string* message,
                                          std::string* code);

/// The FFmpeg argument list (without the program) that transcodes `wav` to
/// `output` in `format`: libmp3lame or aac at a constant bitrate.
[[nodiscard]] std::vector<std::string> MusicEncodeArguments(
    const MusicOutputFormat& format, const std::filesystem::path& wav,
    const std::filesystem::path& output);

/// Transcodes in-memory WAV bytes with FFmpeg (GUFO_FFMPEG overrides the
/// program name). The WAV is written to a temporary file, FFmpeg is started
/// directly (no shell) and the encoded file is read back. On success all
/// temporary files are removed; on failure the temporary WAV is kept and
/// its path is part of `*error`, which starts with
/// "<format> encoding needs ffmpeg".
[[nodiscard]] bool EncodeMusic(const std::string& wav,
                               const MusicOutputFormat& format,
                               std::string* encoded, std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MUSIC_ENCODE_HPP_
