#ifndef GUFO_MODELS_QWEN3_TTS_AUDIO_HPP_
#define GUFO_MODELS_QWEN3_TTS_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen3_tts {

struct AudioBuffer {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  // Interleaved float samples.
  std::vector<float> samples;
};

[[nodiscard]] bool DecodeBase64Wav(std::string_view encoded,
                                   AudioBuffer* output,
                                   std::string* error = nullptr);

[[nodiscard]] bool DecodeWav(std::span<const std::byte> bytes,
                             AudioBuffer* output, std::string* error = nullptr);

[[nodiscard]] std::vector<float> ResampleMono(const AudioBuffer& audio,
                                              std::uint32_t output_sample_rate);

inline constexpr double kMinimumSpeechSpeed = 0.25;
inline constexpr double kMaximumSpeechSpeed = 4.0;

/// Changes the speaking rate of mono audio without changing its pitch
/// (WSOLA: 30 ms Hann frames, 50 % overlap, +-10 ms similarity search).
/// speed > 1 is faster/shorter: the result has ceil(size / speed) samples.
/// speed == 1 returns the input unchanged. Throws std::invalid_argument for
/// a speed outside [kMinimumSpeechSpeed, kMaximumSpeechSpeed] or a zero rate.
[[nodiscard]] std::vector<float> TimeStretchMono(std::span<const float> mono,
                                                 std::uint32_t sample_rate,
                                                 double speed);

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_AUDIO_HPP_
