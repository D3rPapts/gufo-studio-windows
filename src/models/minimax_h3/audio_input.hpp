#ifndef GUFO_MODELS_MINIMAX_H3_AUDIO_INPUT_HPP_
#define GUFO_MODELS_MINIMAX_H3_AUDIO_INPUT_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "src/models/minimax_h3/audio_vae.hpp"

namespace gufo::minimax_h3 {

// Encoded audio guides are bounded to 32 MiB and 60 seconds.
inline constexpr std::size_t kH3MaximumGuideAudioBytes = 32U << 20U;
inline constexpr int kH3MaximumGuideAudioSeconds = 60;

/// Decodes an audio guide (WAV, FLAC, Ogg/Vorbis/Opus or MP3) to the
/// AudioVAE's input: 32 kHz stereo, channel-major F32 PCM. Mono is copied to
/// both tracks (the reference's replicate padding), more than two channels
/// keep the first two, other rates are resampled with SoX (HQ).
[[nodiscard]] bool DecodeGuideAudio(std::span<const std::uint8_t> bytes,
                                    AudioWaveform* waveform,
                                    std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_AUDIO_INPUT_HPP_
