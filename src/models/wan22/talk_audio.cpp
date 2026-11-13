// Written with AI assistance (Claude Code).
// Ported from the 1.x fork (src/cli/serve/video_talk.cpp and
// src/models/minimax_h3/audio_input.cpp) without behaviour changes.
#include "src/models/wan22/talk_audio.hpp"

#include <sndfile.h>
#include <soxr.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <utility>

namespace gufo::models::wan22 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

constexpr double kPi = 3.14159265358979323846;

struct Reader {
  std::span<const std::uint8_t> bytes;
  sf_count_t position{0};
};

sf_count_t Length(void* context) {
  return static_cast<sf_count_t>(static_cast<Reader*>(context)->bytes.size());
}

sf_count_t Seek(sf_count_t offset, int whence, void* context) {
  auto& reader = *static_cast<Reader*>(context);
  const sf_count_t size = static_cast<sf_count_t>(reader.bytes.size());
  const sf_count_t base = whence == SEEK_SET   ? 0
                          : whence == SEEK_CUR ? reader.position
                          : whence == SEEK_END ? size
                                               : -1;
  if (base < 0 || offset < -base || offset > size - base)
    return -1;
  reader.position = base + offset;
  return reader.position;
}

sf_count_t Read(void* target, sf_count_t count, void* context) {
  auto& reader = *static_cast<Reader*>(context);
  if (count < 0)
    return 0;
  count = std::min(
      count, static_cast<sf_count_t>(reader.bytes.size()) - reader.position);
  std::memcpy(target, reader.bytes.data() + reader.position,
              static_cast<std::size_t>(count));
  reader.position += count;
  return count;
}

sf_count_t Tell(void* context) {
  return static_cast<Reader*>(context)->position;
}

// The wav2vec2 convolution stack: encoder frames for 16 kHz samples.
int EncoderFrames(std::size_t samples) {
  static constexpr int kKernels[7] = {10, 3, 3, 3, 3, 2, 2};
  static constexpr int kStrides[7] = {5, 2, 2, 2, 2, 2, 2};
  long long length = static_cast<long long>(samples);
  for (int i = 0; i < 7; ++i) {
    if (length < kKernels[i])
      return 0;
    length = (length - kKernels[i]) / kStrides[i] + 1;
  }
  return static_cast<int>(length);
}

std::size_t Samples16k(const AudioWaveform& audio) {
  if (audio.sample_rate <= 0)
    return 0;
  return static_cast<std::size_t>(
      (static_cast<std::uint64_t>(audio.samples) * 16000U +
       static_cast<std::uint64_t>(audio.sample_rate) - 1U) /
      static_cast<std::uint64_t>(audio.sample_rate));
}

}  // namespace

bool DecodeSpeech(std::span<const std::uint8_t> bytes, AudioWaveform* waveform,
                  std::string* error) {
  if (waveform == nullptr) {
    SetError(error, "missing audio output");
    return false;
  }
  if (bytes.empty() || bytes.size() > kMaximumSpeechBytes) {
    SetError(error, "is empty or exceeds 32 MiB");
    return false;
  }
  Reader reader{bytes, 0};
  SF_VIRTUAL_IO io{Length, Seek, Read, nullptr, Tell};
  SF_INFO info{};
  std::unique_ptr<SNDFILE, decltype(&sf_close)> file(
      sf_open_virtual(&io, SFM_READ, &info, &reader), sf_close);
  if (!file) {
    SetError(error, "could not be decoded (WAV, FLAC, Ogg or MP3 expected)");
    return false;
  }
  if (info.channels < 1 || info.channels > 16 || info.samplerate < 8000 ||
      info.samplerate > 192000) {
    SetError(error, "has an unsupported channel count or rate");
    return false;
  }
  const int max_frames = info.samplerate * kMaximumSpeechSeconds;
  // Interleaved stereo at the source rate.
  std::vector<float> stereo;
  std::vector<float> block(static_cast<std::size_t>(4096) * info.channels);
  for (;;) {
    const sf_count_t frames = sf_readf_float(file.get(), block.data(), 4096);
    if (frames <= 0)
      break;
    if (stereo.size() / 2U + static_cast<std::size_t>(frames) >
        static_cast<std::size_t>(max_frames)) {
      SetError(error, "exceeds " + std::to_string(kMaximumSpeechSeconds) +
                          " seconds");
      return false;
    }
    for (sf_count_t t = 0; t < frames; ++t) {
      const float* sample = block.data() + t * info.channels;
      const float left = sample[0];
      const float right = info.channels > 1 ? sample[1] : sample[0];
      if (!std::isfinite(left) || !std::isfinite(right)) {
        SetError(error, "has non-finite samples");
        return false;
      }
      stereo.push_back(left);
      stereo.push_back(right);
    }
  }
  if (stereo.empty() || sf_error(file.get()) != SF_ERR_NO_ERROR) {
    SetError(error, "is empty or truncated");
    return false;
  }
  const std::size_t source_frames = stereo.size() / 2U;
  std::vector<float> resampled;
  if (info.samplerate == kSpeechSampleRate) {
    resampled = std::move(stereo);
  } else {
    const std::size_t length =
        (source_frames * kSpeechSampleRate + info.samplerate - 1) /
        static_cast<std::size_t>(info.samplerate);
    resampled.assign((length + 1U) * 2U, 0.0F);
    const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    const soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    std::size_t consumed = 0;
    std::size_t produced = 0;
    const soxr_error_t failure =
        soxr_oneshot(info.samplerate, kSpeechSampleRate, 2, stereo.data(),
                     source_frames, &consumed, resampled.data(), length + 1U,
                     &produced, &io_spec, &quality, nullptr);
    if (failure != nullptr || consumed != source_frames || produced == 0) {
      SetError(error, "resampling to 32 kHz failed");
      return false;
    }
    resampled.resize(std::min(produced, length) * 2U);
  }
  const std::size_t samples = resampled.size() / 2U;
  waveform->channels = 2;
  waveform->sample_rate = kSpeechSampleRate;
  waveform->samples = static_cast<int>(samples);
  waveform->pcm.assign(samples * 2U, 0.0F);
  for (std::size_t t = 0; t < samples; ++t) {
    waveform->pcm[t] = resampled[t * 2U];
    waveform->pcm[samples + t] = resampled[t * 2U + 1U];
  }
  return true;
}

int TalkFramesForAudio(const AudioWaveform& audio) {
  const int frames50 = EncoderFrames(Samples16k(audio));
  return static_cast<int>(static_cast<double>(frames50) / 50.0 * 25.0);
}

std::vector<float> TalkMono16k(const AudioWaveform& audio) {
  const std::size_t n = static_cast<std::size_t>(std::max(audio.samples, 0));
  const int channels = std::max(audio.channels, 1);
  std::vector<float> mono(n, 0.0F);
  for (int c = 0; c < channels; ++c)
    for (std::size_t i = 0; i < n && c * n + i < audio.pcm.size(); ++i)
      mono[i] += audio.pcm[c * n + i] / static_cast<float>(channels);
  if (audio.sample_rate == 16000 || n == 0)
    return mono;
  // Hann-windowed sinc, 32 zero crossings each side, cutoff at the lower
  // Nyquist rate.
  const std::size_t out = Samples16k(audio);
  const double ratio = static_cast<double>(audio.sample_rate) / 16000.0;
  const double cutoff = std::min(1.0, 1.0 / ratio);
  constexpr int kZeros = 32;
  const double half = kZeros / cutoff;
  std::vector<float> result(out, 0.0F);
  // Taps depend only on the fractional source position: one table per phase
  // (a single one for 32 kHz).
  struct Taps {
    long long lead{0};
    std::vector<double> weights;
  };
  std::map<double, Taps> phases;
  for (std::size_t j = 0; j < out; ++j) {
    const double centre = static_cast<double>(j) * ratio;
    const double base = std::floor(centre);
    const double frac = centre - base;
    auto found = phases.find(frac);
    if (found == phases.end()) {
      Taps taps;
      taps.lead = static_cast<long long>(std::ceil(frac - half));
      const long long tail = static_cast<long long>(std::floor(frac + half));
      for (long long k = taps.lead; k <= tail; ++k) {
        const double d = static_cast<double>(k) - frac;
        const double x = d * cutoff;
        const double sinc = x == 0.0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
        const double window = 0.5 + 0.5 * std::cos(kPi * d / half);
        taps.weights.push_back(sinc * window * cutoff);
      }
      found = phases.emplace(frac, std::move(taps)).first;
    }
    const Taps& taps = found->second;
    double sum = 0, weights = 0;
    for (std::size_t k = 0; k < taps.weights.size(); ++k) {
      const long long i = static_cast<long long>(base) + taps.lead +
                          static_cast<long long>(k);
      if (i < 0 || i >= static_cast<long long>(n))
        continue;
      sum += taps.weights[k] * mono[static_cast<std::size_t>(i)];
      weights += taps.weights[k];
    }
    // Normalized by the taps' sum (unity gain, also where the kernel is cut
    // at the edges).
    result[j] = weights != 0.0 ? static_cast<float>(sum / weights) : 0.0F;
  }
  return result;
}

bool ParseTalkOptions(const json::Value& talk, std::string* normalized,
                      std::string* error) {
  if (!talk.is_object()) {
    SetError(error, "'gufo.talk' must be an object");
    return false;
  }
  for (const auto& [key, value] : talk.members()) {
    (void)value;
    if (key != "audio_scale" && key != "audio_mode" &&
        key != "motion_frames" && key != "denoise") {
      SetError(error, "'gufo.talk' may contain only 'audio', "
                      "'speaker_masks', 'video', 'audio_scale', "
                      "'audio_mode', 'motion_frames' and 'denoise'");
      return false;
    }
  }
  json::Value out = json::Value::object();
  if (const json::Value* scale = talk.find("audio_scale")) {
    const double v = scale->is_number() ? scale->as_double(NAN) : NAN;
    if (!std::isfinite(v) || v < -10.0 || v > 10.0) {
      SetError(error, "'gufo.talk.audio_scale' must be -10..10");
      return false;
    }
    out["audio_scale"] = v;
  }
  if (const json::Value* mode = talk.find("audio_mode")) {
    if (!mode->is_string() || (mode->str() != "add" && mode->str() != "para" &&
                               mode->str() != "auto")) {
      SetError(error,
               "'gufo.talk.audio_mode' must be \"add\", \"para\" or \"auto\"");
      return false;
    }
    out["audio_mode"] = mode->str();
  }
  if (const json::Value* motion = talk.find("motion_frames")) {
    const double v = motion->is_number() ? motion->as_double(NAN) : NAN;
    if (!std::isfinite(v) || std::floor(v) != v || v < 1 || v > 33 ||
        (static_cast<int>(v) - 1) % 4 != 0) {
      SetError(error, "'gufo.talk.motion_frames' must be 4k + 1 in 1..33");
      return false;
    }
    out["motion_frames"] = static_cast<int>(v);
  }
  if (const json::Value* denoise = talk.find("denoise")) {
    const double v = denoise->is_number() ? denoise->as_double(NAN) : NAN;
    if (!std::isfinite(v) || v <= 0.0 || v > 1.0) {
      SetError(error, "'gufo.talk.denoise' must be > 0 and <= 1");
      return false;
    }
    out["denoise"] = v;
  }
  *normalized = out.dump();
  return true;
}

bool WriteWav(const std::filesystem::path& path, int channels, int rate,
              const std::vector<float>& pcm, std::string* error) {
  const std::size_t frames = channels > 0 ? pcm.size() / channels : 0;
  std::string data(frames * channels * 2, '\0');
  for (std::size_t i = 0; i < frames; ++i)
    for (int c = 0; c < channels; ++c) {
      const float v = std::clamp(pcm[c * frames + i], -1.0F, 1.0F);
      const auto s = static_cast<std::int16_t>(std::lround(v * 32767.0F));
      std::memcpy(&data[(i * channels + c) * 2], &s, 2);
    }
  const auto u32 = [](std::uint32_t v) {
    std::string out(4, '\0');
    std::memcpy(out.data(), &v, 4);
    return out;
  };
  const auto u16 = [](std::uint16_t v) {
    std::string out(2, '\0');
    std::memcpy(out.data(), &v, 2);
    return out;
  };
  std::ofstream file(path, std::ios::binary);
  file << "RIFF" << u32(static_cast<std::uint32_t>(36 + data.size()))
       << "WAVEfmt " << u32(16) << u16(1)
       << u16(static_cast<std::uint16_t>(channels))
       << u32(static_cast<std::uint32_t>(rate))
       << u32(static_cast<std::uint32_t>(rate * channels * 2))
       << u16(static_cast<std::uint16_t>(channels * 2)) << u16(16) << "data"
       << u32(static_cast<std::uint32_t>(data.size())) << data;
  if (!file) {
    SetError(error, "cannot write " + path.string());
    return false;
  }
  return true;
}

}  // namespace gufo::models::wan22
