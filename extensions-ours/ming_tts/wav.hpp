// Minimal WAV reader for the Ming-3.0-tts and VoxCPM2 extensions (tests, speak)
// (PCM 16/24/32-bit and IEEE float 32-bit, any channel count). Replaces
// src/models/qwen3_tts/audio.hpp, which is private to the gufo core and not
// reachable from an out-of-tree extension. Written with AI assistance
// (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_WAV_HPP_
#define GUFO_MODELS_MING_TTS_WAV_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ming_tts {

struct WavAudio {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  std::vector<float> samples;  // interleaved
};

inline std::uint32_t WavU32(const std::byte* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8U) |
         (static_cast<std::uint32_t>(p[2]) << 16U) |
         (static_cast<std::uint32_t>(p[3]) << 24U);
}

inline std::uint16_t WavU16(const std::byte* p) {
  return static_cast<std::uint16_t>(static_cast<std::uint32_t>(p[0]) |
                                    (static_cast<std::uint32_t>(p[1]) << 8U));
}

[[nodiscard]] inline bool DecodeWav(std::span<const std::byte> bytes,
                                    WavAudio* output,
                                    std::string* error = nullptr) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    return fail("not a RIFF/WAVE file");
  }
  std::uint16_t format = 0;
  std::uint16_t channels = 0;
  std::uint32_t rate = 0;
  std::uint16_t bits = 0;
  const std::byte* data = nullptr;
  std::size_t data_size = 0;
  std::size_t pos = 12;
  while (pos + 8 <= bytes.size()) {
    const std::byte* chunk = bytes.data() + pos;
    const std::uint32_t size = WavU32(chunk + 4);
    const std::size_t body = pos + 8;
    const std::size_t available = bytes.size() - body;
    if (std::memcmp(chunk, "fmt ", 4) == 0 && size >= 16 && available >= 16) {
      format = WavU16(chunk + 8);
      channels = WavU16(chunk + 10);
      rate = WavU32(chunk + 12);
      bits = WavU16(chunk + 22);
      if (format == 0xFFFE && size >= 40 && available >= 40) {
        format = WavU16(chunk + 32);  // WAVE_FORMAT_EXTENSIBLE sub-format
      }
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data = chunk + 8;
      data_size = size <= available ? size : available;
    }
    pos = body + size + (size & 1U);
  }
  if (data == nullptr || channels == 0 || rate == 0) {
    return fail("WAV file has no fmt or data chunk");
  }
  const std::size_t width = bits / 8U;
  if (!((format == 1 && (bits == 16 || bits == 24 || bits == 32)) ||
        (format == 3 && bits == 32))) {
    return fail("unsupported WAV sample format");
  }
  const std::size_t count = data_size / width;
  output->sample_rate = rate;
  output->channels = channels;
  output->samples.resize(count - count % channels);
  for (std::size_t i = 0; i < output->samples.size(); ++i) {
    const std::byte* p = data + i * width;
    float value = 0.0F;
    if (format == 3) {
      std::memcpy(&value, p, 4);
    } else if (bits == 16) {
      value = static_cast<float>(static_cast<std::int16_t>(WavU16(p))) / 32768.0F;
    } else if (bits == 24) {
      std::int32_t v = static_cast<std::int32_t>(
          (static_cast<std::uint32_t>(p[0]) << 8U) |
          (static_cast<std::uint32_t>(p[1]) << 16U) |
          (static_cast<std::uint32_t>(p[2]) << 24U));
      value = static_cast<float>(v >> 8) / 8388608.0F;
    } else {
      value = static_cast<float>(static_cast<std::int32_t>(WavU32(p))) / 2147483648.0F;
    }
    output->samples[i] = value;
  }
  return true;
}

// Mono PCM16 WAV file bytes (non-finite samples become 0, others clipped).
[[nodiscard]] inline std::string EncodeWavPcm16(std::span<const float> samples,
                                                std::uint32_t sample_rate) {
  const auto put16 = [](std::string* out, std::uint32_t v) {
    out->push_back(static_cast<char>(v & 0xFFU));
    out->push_back(static_cast<char>((v >> 8U) & 0xFFU));
  };
  const auto put32 = [&](std::string* out, std::uint32_t v) {
    put16(out, v & 0xFFFFU);
    put16(out, v >> 16U);
  };
  const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2U);
  std::string out;
  out.reserve(44U + data_bytes);
  out.append("RIFF", 4);
  put32(&out, 36U + data_bytes);
  out.append("WAVEfmt ", 8);
  put32(&out, 16U);
  put16(&out, 1U);
  put16(&out, 1U);
  put32(&out, sample_rate);
  put32(&out, sample_rate * 2U);
  put16(&out, 2U);
  put16(&out, 16U);
  out.append("data", 4);
  put32(&out, data_bytes);
  for (const float s : samples) {
    float v = s == s && s - s == 0.0F ? s : 0.0F;
    v = v < -1.0F ? -1.0F : (v > 1.0F ? 1.0F : v);
    put16(&out, static_cast<std::uint16_t>(static_cast<std::int16_t>(v * 32767.0F)));
  }
  return out;
}

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_WAV_HPP_
