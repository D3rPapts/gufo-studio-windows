#include "src/models/qwen3_tts/audio.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace qwen3_tts = gufo::models::qwen3_tts;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_audio_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

double RootMeanSquare(std::span<const float> samples) {
  double sum = 0.0;
  for (const float sample : samples) {
    sum += static_cast<double>(sample) * sample;
  }
  return std::sqrt(sum / static_cast<double>(samples.size()));
}

qwen3_tts::AudioBuffer MakeSine(std::uint32_t sample_rate, double frequency,
                                std::size_t frames) {
  qwen3_tts::AudioBuffer audio{
      .sample_rate = sample_rate,
      .channels = 1,
      .samples = {},
  };
  audio.samples.resize(frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    audio.samples[frame] =
        static_cast<float>(std::sin(2.0 * std::numbers::pi * frequency *
                                    static_cast<double>(frame) / sample_rate));
  }
  return audio;
}

void TestIdentityAndMonoMix() {
  const qwen3_tts::AudioBuffer audio{
      .sample_rate = 24000,
      .channels = 2,
      .samples = {1.0F, -1.0F, 0.5F, 0.25F},
  };
  const std::vector<float> mono = qwen3_tts::ResampleMono(audio, 24000);
  Check(mono == std::vector<float>({0.0F, 0.375F}),
        "same-rate stereo input is mixed without resampling drift");
}

void TestDownsampleAntiAlias() {
  constexpr std::uint32_t kInputRate = 48000;
  constexpr std::uint32_t kOutputRate = 24000;
  constexpr std::size_t kFrames = kInputRate;
  constexpr std::size_t kTrim = 64;

  const std::vector<float> passband = qwen3_tts::ResampleMono(
      MakeSine(kInputRate, 3000.0, kFrames), kOutputRate);
  const std::vector<float> stopband = qwen3_tts::ResampleMono(
      MakeSine(kInputRate, 18000.0, kFrames), kOutputRate);
  const std::span<const float> passband_middle(passband.data() + kTrim,
                                               passband.size() - 2 * kTrim);
  const std::span<const float> stopband_middle(stopband.data() + kTrim,
                                               stopband.size() - 2 * kTrim);
  const double passband_rms = RootMeanSquare(passband_middle);
  const double stopband_rms = RootMeanSquare(stopband_middle);

  Check(passband.size() == kOutputRate, "downsample frame count is exact");
  Check(passband_rms > 0.68 && passband_rms < 0.73,
        "speech-band energy is preserved");
  Check(stopband_rms < 0.01,
        "above-Nyquist energy is filtered instead of aliased");
}

double ZeroCrossingHz(std::span<const float> samples, std::uint32_t rate) {
  std::size_t crossings = 0;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    crossings += (samples[i - 1] < 0.0F) != (samples[i] < 0.0F) ? 1 : 0;
  }
  return 0.5 * static_cast<double>(crossings) * rate /
         static_cast<double>(samples.size());
}

void TestTimeStretchKeepsPitch() {
  constexpr std::uint32_t kRate = 24000;
  const qwen3_tts::AudioBuffer sine = MakeSine(kRate, 220.0, kRate);
  Check(qwen3_tts::TimeStretchMono(sine.samples, kRate, 1.0) == sine.samples,
        "speed 1.0 leaves audio bit-identical");
  for (const double speed : {0.8, 1.25, 1.5, 2.0}) {
    const std::vector<float> out =
        qwen3_tts::TimeStretchMono(sine.samples, kRate, speed);
    const auto expected =
        static_cast<std::size_t>(std::ceil(kRate / speed));
    Check(out.size() == expected, "stretched length is ceil(size / speed)");
    const std::span<const float> middle(out.data() + 1200, out.size() - 2400);
    const double hz = ZeroCrossingHz(middle, kRate);
    Check(hz > 214.0 && hz < 226.0, "time stretch keeps the pitch");
    const double rms = RootMeanSquare(middle);
    Check(rms > 0.66 && rms < 0.75, "time stretch keeps the loudness");
  }
}

void TestTimeStretchScalesSpeechTiming() {
  // 0.5 s tone then 0.5 s silence: the tone must end at 0.5 s / speed.
  constexpr std::uint32_t kRate = 24000;
  qwen3_tts::AudioBuffer audio = MakeSine(kRate, 300.0, kRate);
  for (std::size_t i = kRate / 2; i < audio.samples.size(); ++i) {
    audio.samples[i] = 0.0F;
  }
  for (const double speed : {0.75, 1.4}) {
    const std::vector<float> out =
        qwen3_tts::TimeStretchMono(audio.samples, kRate, speed);
    std::size_t last = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (std::abs(out[i]) > 0.05F) {
        last = i;
      }
    }
    const double end_seconds = static_cast<double>(last) / kRate;
    const double expected = 0.5 / speed;
    // Within one 30 ms analysis frame.
    Check(std::abs(end_seconds - expected) < 0.03,
          "speech end moves with the speed factor");
  }
}

void TestTimeStretchRejectsInvalidSpeed() {
  const std::vector<float> samples(100, 0.0F);
  for (const double speed : {0.0, 0.2, 4.5, std::nan("")}) {
    bool threw = false;
    try {
      (void)qwen3_tts::TimeStretchMono(samples, 24000, speed);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "out-of-range speed is rejected");
  }
}

}  // namespace

int main() {
  TestIdentityAndMonoMix();
  TestDownsampleAntiAlias();
  TestTimeStretchKeepsPitch();
  TestTimeStretchScalesSpeechTiming();
  TestTimeStretchRejectsInvalidSpeed();
  std::cout << "PASS qwen3_tts_audio_test\n";
  return 0;
}
