// /v1/audio/music output formats (src/cli/serve/music_encode): the request
// parser (response_format, bitrate_kbps), the FFmpeg argument list and, when
// an ffmpeg is on the PATH (or GUFO_FFMPEG names one), a real mp3 transcode
// of a tiny WAV. Without ffmpeg the transcode part is skipped (exit 77).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/cli/serve/music_encode.hpp"
#include "src/core/json.hpp"

namespace json = gufo::json;
using gufo::server::EncodeMusic;
using gufo::server::MusicEncodeArguments;
using gufo::server::MusicOutputFormat;
using gufo::server::ParseMusicOutputFormat;

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL music_encode_test: " << what << '\n';
    ++failures;
  }
}

bool Contains(const std::vector<std::string>& list, const std::string& item) {
  for (const std::string& entry : list)
    if (entry == item)
      return true;
  return false;
}

// Parses `text` as a request body; returns whether the format was accepted.
bool Parse(const std::string& text, MusicOutputFormat* format,
           std::string* code) {
  std::string message;
  const json::Value body = json::parse(text);
  const bool ok = ParseMusicOutputFormat(body, format, &message, code);
  if (!ok)
    Check(!message.empty(), "rejection carries a message: " + text);
  return ok;
}

// A 0.25 s stereo 16 kHz PCM16 WAV (a 440 Hz tone), enough for FFmpeg.
std::string TinyWav() {
  constexpr std::uint32_t kRate = 16000;
  constexpr std::uint32_t kFrames = 4000;
  constexpr std::uint16_t kChannels = 2;
  std::string wav;
  const auto u32 = [&](std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
      wav.push_back(static_cast<char>((value >> shift) & 255U));
  };
  const auto u16 = [&](std::uint16_t value) {
    wav.push_back(static_cast<char>(value & 255U));
    wav.push_back(static_cast<char>(value >> 8U));
  };
  const std::uint32_t data_bytes = kFrames * kChannels * 2;
  wav += "RIFF";
  u32(36 + data_bytes);
  wav += "WAVEfmt ";
  u32(16);
  u16(1);
  u16(kChannels);
  u32(kRate);
  u32(kRate * kChannels * 2);
  u16(kChannels * 2);
  u16(16);
  wav += "data";
  u32(data_bytes);
  for (std::uint32_t frame = 0; frame < kFrames; ++frame) {
    const double phase = 2.0 * 3.14159265358979 * 440.0 * frame / kRate;
    const auto sample = static_cast<std::uint16_t>(
        static_cast<std::int16_t>(std::sin(phase) * 8000.0));
    u16(sample);
    u16(sample);
  }
  return wav;
}

void TestParser() {
  MusicOutputFormat format;
  std::string code;

  Check(Parse(R"({"prompt":"x"})", &format, &code), "default accepted");
  Check(format.name == "wav" && format.content_type == "audio/wav" &&
            format.filename == "output.wav" && !format.needs_ffmpeg(),
        "default is wav");

  Check(Parse(R"({"response_format":"wav"})", &format, &code),
        "wav accepted");
  Check(format.name == "wav", "explicit wav");

  Check(Parse(R"({"response_format":"mp3"})", &format, &code),
        "mp3 accepted");
  Check(format.name == "mp3" && format.content_type == "audio/mpeg" &&
            format.filename == "output.mp3" && format.bitrate_kbps == 192 &&
            format.needs_ffmpeg(),
        "mp3 defaults: audio/mpeg, output.mp3, 192 kbps");

  Check(Parse(R"({"response_format":"MP3","bitrate_kbps":320})", &format,
              &code),
        "mp3 case-insensitive with bitrate");
  Check(format.name == "mp3" && format.bitrate_kbps == 320, "bitrate 320");

  Check(Parse(R"({"response_format":"aac"})", &format, &code),
        "aac accepted");
  Check(format.name == "m4a" && format.content_type == "audio/mp4" &&
            format.filename == "output.m4a",
        "aac maps to m4a / audio/mp4");
  Check(Parse(R"({"response_format":"m4a","bitrate_kbps":96})", &format,
              &code),
        "m4a alias accepted");
  Check(format.name == "m4a" && format.bitrate_kbps == 96, "bitrate 96");

  Check(!Parse(R"({"response_format":"ogg"})", &format, &code),
        "ogg rejected");
  Check(code == "unsupported_format", "ogg code is unsupported_format");
  Check(!Parse(R"({"response_format":"pcm"})", &format, &code),
        "pcm rejected");
  Check(!Parse(R"({"response_format":3})", &format, &code),
        "numeric response_format rejected");
  Check(code == "unsupported_format", "number code is unsupported_format");

  Check(!Parse(R"({"response_format":"mp3","bitrate_kbps":64})", &format,
               &code),
        "bitrate below 96 rejected");
  Check(code == "invalid_bitrate", "low bitrate code");
  Check(!Parse(R"({"response_format":"mp3","bitrate_kbps":321})", &format,
               &code),
        "bitrate above 320 rejected");
  Check(!Parse(R"({"response_format":"mp3","bitrate_kbps":192.5})", &format,
               &code),
        "fractional bitrate rejected");
  Check(!Parse(R"({"response_format":"mp3","bitrate_kbps":"192"})", &format,
               &code),
        "string bitrate rejected");
  Check(!Parse(R"({"response_format":"wav","bitrate_kbps":12})", &format,
               &code),
        "bad bitrate rejected even for wav");
}

void TestArguments() {
  MusicOutputFormat mp3;
  mp3.name = "mp3";
  mp3.bitrate_kbps = 256;
  const std::vector<std::string> args =
      MusicEncodeArguments(mp3, "in dir/a.wav", "out dir/a.mp3");
  Check(!args.empty() && args.back() == "out dir/a.mp3", "output last");
  Check(Contains(args, "libmp3lame"), "mp3 uses libmp3lame");
  Check(Contains(args, "256k"), "bitrate as <kbps>k");
  Check(Contains(args, "-y") && Contains(args, "-nostdin"),
        "overwrite, never prompt");
  Check(Contains(args, "in dir/a.wav"), "input path unquoted (no shell)");
  Check(!Contains(args, "+faststart"), "mp3 has no movflags");

  MusicOutputFormat m4a;
  m4a.name = "m4a";
  const std::vector<std::string> aac =
      MusicEncodeArguments(m4a, "a.wav", "a.m4a");
  Check(Contains(aac, "aac") && Contains(aac, "+faststart") &&
            Contains(aac, "192k"),
        "m4a uses aac, faststart, default bitrate");

  MusicOutputFormat wav;
  Check(MusicEncodeArguments(wav, "a.wav", "a.wav").empty(),
        "wav needs no ffmpeg");
}

// Returns false when ffmpeg is unavailable (the caller skips).
bool TestTranscode() {
  const std::string wav = TinyWav();
  MusicOutputFormat passthrough;
  std::string out, error;
  Check(EncodeMusic(wav, passthrough, &out, &error) && out == wav,
        "wav passes through unchanged");

  MusicOutputFormat mp3;
  mp3.name = "mp3";
  mp3.content_type = "audio/mpeg";
  mp3.filename = "output.mp3";
  mp3.bitrate_kbps = 128;
  out.clear();
  if (!EncodeMusic(wav, mp3, &out, &error)) {
    if (error.find("cannot start") != std::string::npos) {
      std::cerr << "SKIP music_encode_test: " << error << '\n';
      // The WAV is kept on failure; tidy it for the skip.
      const auto kept = error.find("WAV kept at ");
      if (kept != std::string::npos) {
        std::error_code ignored;
        std::filesystem::remove(error.substr(kept + 12), ignored);
      }
      return false;
    }
    Check(false, "mp3 transcode: " + error);
    return true;
  }
  Check(out.size() > 100 && out.size() < wav.size(),
        "mp3 is smaller than the WAV and not empty");
  const bool id3 = out.substr(0, 3) == "ID3";
  const bool sync = out.size() > 1 &&
                    static_cast<unsigned char>(out[0]) == 0xFF &&
                    (static_cast<unsigned char>(out[1]) & 0xE0) == 0xE0;
  Check(id3 || sync, "mp3 starts with ID3 or a frame sync");

  // A failing encode reports clearly and keeps the WAV.
  MusicOutputFormat broken;
  broken.name = "mp3";
  broken.bitrate_kbps = 192;
  std::string junk = "not a wav at all";
  out.clear();
  const bool failed = !EncodeMusic(junk, broken, &out, &error);
  Check(failed, "garbage input fails");
  Check(error.rfind("mp3 encoding needs ffmpeg", 0) == 0,
        "failure names the format and ffmpeg: " + error);
  const auto kept = error.find("WAV kept at ");
  Check(kept != std::string::npos, "failure names the kept WAV");
  if (kept != std::string::npos) {
    const std::filesystem::path path = error.substr(kept + 12);
    std::error_code ignored;
    Check(std::filesystem::exists(path, ignored), "kept WAV exists");
    std::filesystem::remove(path, ignored);
  }
  return true;
}

}  // namespace

int main() {
  TestParser();
  TestArguments();
  const bool ran = TestTranscode();
  if (failures != 0)
    return 1;
  if (!ran) {
    std::cout << "music_encode_test: parser ok, transcode skipped\n";
    return 77;
  }
  std::cout << "music_encode_test: ok\n";
  return 0;
}
