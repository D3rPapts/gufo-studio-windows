#include "src/models/minimax_h3/audio_input.hpp"

#include <sndfile.h>
#include <soxr.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

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
  if (base < 0 || offset < -base || offset > size - base) {
    return -1;
  }
  reader.position = base + offset;
  return reader.position;
}

sf_count_t Read(void* target, sf_count_t count, void* context) {
  auto& reader = *static_cast<Reader*>(context);
  if (count < 0) {
    return 0;
  }
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

}  // namespace

bool DecodeGuideAudio(std::span<const std::uint8_t> bytes,
                      AudioWaveform* waveform, std::string* error) {
  if (waveform == nullptr) {
    SetError(error, "missing audio guide output");
    return false;
  }
  if (bytes.empty() || bytes.size() > kH3MaximumGuideAudioBytes) {
    SetError(error, "audio guide is empty or exceeds 32 MiB");
    return false;
  }
  Reader reader{bytes, 0};
  SF_VIRTUAL_IO io{Length, Seek, Read, nullptr, Tell};
  SF_INFO info{};
  std::unique_ptr<SNDFILE, decltype(&sf_close)> file(
      sf_open_virtual(&io, SFM_READ, &info, &reader), sf_close);
  if (!file) {
    SetError(error,
             "audio guide could not be decoded (WAV, FLAC, Ogg or MP3 "
             "expected)");
    return false;
  }
  const int max_frames = info.samplerate * kH3MaximumGuideAudioSeconds;
  if (info.channels < 1 || info.channels > 16 || info.samplerate < 8000 ||
      info.samplerate > 192000) {
    SetError(error, "audio guide has an unsupported channel count or rate");
    return false;
  }
  // Interleaved stereo at the source rate.
  std::vector<float> stereo;
  std::vector<float> block(static_cast<std::size_t>(4096) * info.channels);
  for (;;) {
    const sf_count_t frames = sf_readf_float(file.get(), block.data(), 4096);
    if (frames <= 0) {
      break;
    }
    if (stereo.size() / 2U + static_cast<std::size_t>(frames) >
        static_cast<std::size_t>(max_frames)) {
      SetError(error, "audio guide exceeds " +
                          std::to_string(kH3MaximumGuideAudioSeconds) +
                          " seconds");
      return false;
    }
    for (sf_count_t t = 0; t < frames; ++t) {
      const float* sample = block.data() + t * info.channels;
      const float left = sample[0];
      const float right = info.channels > 1 ? sample[1] : sample[0];
      if (!std::isfinite(left) || !std::isfinite(right)) {
        SetError(error, "audio guide has non-finite samples");
        return false;
      }
      stereo.push_back(left);
      stereo.push_back(right);
    }
  }
  if (stereo.empty() || sf_error(file.get()) != SF_ERR_NO_ERROR) {
    SetError(error, "audio guide is empty or truncated");
    return false;
  }
  const std::size_t source_frames = stereo.size() / 2U;
  std::vector<float> resampled;
  if (info.samplerate == kH3AudioVaeSampleRate) {
    resampled = std::move(stereo);
  } else {
    const std::size_t length =
        (source_frames * kH3AudioVaeSampleRate + info.samplerate - 1) /
        static_cast<std::size_t>(info.samplerate);
    resampled.assign((length + 1U) * 2U, 0.0F);
    const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    const soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    std::size_t consumed = 0;
    std::size_t produced = 0;
    const soxr_error_t failure =
        soxr_oneshot(info.samplerate, kH3AudioVaeSampleRate, 2, stereo.data(),
                     source_frames, &consumed, resampled.data(), length + 1U,
                     &produced, &io_spec, &quality, nullptr);
    if (failure != nullptr || consumed != source_frames || produced == 0) {
      SetError(error, "audio guide resampling to 32 kHz failed");
      return false;
    }
    resampled.resize(std::min(produced, length) * 2U);
  }
  const std::size_t samples = resampled.size() / 2U;
  waveform->channels = kH3AudioVaeStereoChannels;
  waveform->sample_rate = kH3AudioVaeSampleRate;
  waveform->samples = static_cast<int>(samples);
  waveform->pcm.assign(samples * 2U, 0.0F);
  for (std::size_t t = 0; t < samples; ++t) {
    waveform->pcm[t] = resampled[t * 2U];
    waveform->pcm[samples + t] = resampled[t * 2U + 1U];
  }
  return true;
}

}  // namespace gufo::minimax_h3
