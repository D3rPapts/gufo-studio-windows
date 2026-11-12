// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_TALK_AUDIO_HPP_
#define GUFO_MODELS_WAN22_TALK_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

namespace gufo::models::wan22 {

// Speech for InfiniteTalk: at most 32 MiB encoded and 60 seconds.
inline constexpr std::size_t kMaximumSpeechBytes = 32U << 20U;
inline constexpr int kMaximumSpeechSeconds = 60;
inline constexpr int kSpeechSampleRate = 32000;

/// Decoded audio: channel-major F32 PCM [channels][samples].
struct AudioWaveform {
  int channels{0};
  int samples{0};
  int sample_rate{0};
  std::vector<float> pcm;
};

/// Decodes WAV, FLAC, Ogg/Vorbis/Opus or MP3 to 32 kHz stereo (mono is
/// copied to both channels, more than two keep the first two, other rates are
/// resampled with SoX HQ). The same conversion the 1.x MiniMax H3 audio
/// guides used, so InfiniteTalk sees the same samples as before.
[[nodiscard]] bool DecodeSpeech(std::span<const std::uint8_t> bytes,
                                AudioWaveform* waveform,
                                std::string* error = nullptr);

/// Video frames at 25 fps one speaker's audio drives (wav2vec2 at 50 Hz,
/// halved as WanInfiniteTalkToVideo does).
[[nodiscard]] int TalkFramesForAudio(const AudioWaveform& audio);

/// Mono 16 kHz from channel-major PCM at any rate (channel mean,
/// windowed-sinc resampling).
[[nodiscard]] std::vector<float> TalkMono16k(const AudioWaveform& audio);

/// "gufo.talk" without the media fields: {"audio_scale": -10..10 (1),
/// "audio_mode": "add" | "para" | "auto", "motion_frames": 4k + 1 in 1..33,
/// "denoise": 0 < d <= 1}. Writes the normalized object to `normalized`.
[[nodiscard]] bool ParseTalkOptions(const json::Value& talk,
                                    std::string* normalized,
                                    std::string* error);

/// 16-bit PCM WAV of channel-major float PCM.
[[nodiscard]] bool WriteWav(const std::filesystem::path& path, int channels,
                            int rate, const std::vector<float>& pcm,
                            std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TALK_AUDIO_HPP_
