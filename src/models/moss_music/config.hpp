#ifndef GUFO_MODELS_MOSS_MUSIC_CONFIG_HPP_
#define GUFO_MODELS_MOSS_MUSIC_CONFIG_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

/// MOSS-Music-8B-Instruct (OpenMOSS): music captioning, lyrics transcription
/// and musical question answering. One audio clip plus a text question in,
/// text out. The model is a Qwen3-8B dense LLM behind a Whisper-style audio
/// encoder with a Conv2d stem (the AuT stem of Qwen3-ASR/Qwen3-Omni, 480
/// channels, three 3x3/stride-2 convolutions), a gated-MLP adapter and three
/// DeepStack injections (encoder layers 8/16/24 -> LLM layers 0/1/2).
///
/// Dimensions follow config.json of OpenMOSS-Team/MOSS-Music-8B-Instruct and
/// src/configuration_moss_music.py of github.com/OpenMOSS/MOSS-Music (the
/// chunked encoder that infer.py runs; the trust_remote_code copy on the Hub
/// has no chunking and is not the reference).
namespace gufo::models::moss_music {

inline constexpr std::uint32_t kSampleRate = 16000;
inline constexpr std::uint32_t kFftSize = 400;
inline constexpr std::uint32_t kHopLength = 160;
inline constexpr std::uint32_t kMelBins = 128;
/// One encoder token covers 8 log-mel frames of 10 ms: 12.5 tokens/s.
inline constexpr std::uint32_t kDownsampleRate = 8;
inline constexpr std::uint32_t kTokensPerTwoSeconds = 25;
/// A time marker ("2", "4", ..., "120") is written after every 2 seconds of
/// audio tokens: digits of the elapsed seconds, one vocabulary token each.
inline constexpr std::uint32_t kMarkerEverySeconds = 2;

struct AudioEncoderConfig {
  std::uint32_t mel_bins{kMelBins};
  std::uint32_t d_model{1280};
  std::uint32_t layers{32};
  std::uint32_t heads{20};
  std::uint32_t ffn{5120};
  /// downsample_hidden_size: channels of the three stem convolutions.
  std::uint32_t conv_channels{480};
  /// Chunks of 2 * n_window log-mel frames (400 frames = 4 s) go through the
  /// stem and all 32 layers independently, positions restarting at 0.
  std::uint32_t n_window{200};
  /// Chunks per encoder batch in the reference; no numerical effect.
  std::uint32_t conv_chunksize{64};
  std::uint32_t max_source_positions{1500};
  float eps{1.0e-5F};
  /// Zero-based encoder layers whose outputs (before the final LayerNorm)
  /// feed the DeepStack mergers, in injection order.
  std::array<std::uint32_t, 3> deepstack_layers{8, 16, 24};

  [[nodiscard]] constexpr std::uint32_t HeadDim() const {
    return d_model / heads;
  }
  [[nodiscard]] constexpr std::uint32_t ChunkFrames() const {
    return 2 * n_window;
  }
  /// Encoder tokens of one full chunk: 400 -> 200 -> 100 -> 50.
  [[nodiscard]] constexpr std::uint32_t ChunkTokens() const {
    return ChunkFrames() / kDownsampleRate;
  }
  /// Mel bins after the three stride-2 convolutions: 128 -> 64 -> 32 -> 16.
  [[nodiscard]] constexpr std::uint32_t StemFrequency() const {
    return ((((mel_bins - 1) / 2 + 1 - 1) / 2 + 1 - 1) / 2) + 1;
  }
  /// Width of the stem_proj input: channels * frequency = 480 * 16 = 7680.
  [[nodiscard]] constexpr std::uint32_t StemFlattened() const {
    return conv_channels * StemFrequency();
  }
};

/// audio_adapter and each deepstack_audio_merger: GatedMLP
/// down(silu(gate(x)) * up(x)), no biases.
struct AdapterConfig {
  std::uint32_t input{1280};
  std::uint32_t hidden{8192};
  std::uint32_t output{4096};
};

/// Qwen3-8B dense (language_config).
struct TextConfig {
  std::uint32_t layers{36};
  std::uint32_t hidden{4096};
  std::uint32_t heads{32};
  std::uint32_t kv_heads{8};
  std::uint32_t head_dim{128};
  std::uint32_t intermediate{12288};
  std::uint32_t vocab{151936};
  std::uint32_t max_positions{40960};
  float rope_theta{1.0e6F};
  float eps{1.0e-6F};
  /// LLM layers 0..k-1 receive a DeepStack residual after their output.
  std::uint32_t deepstack_inject_layers{3};

  [[nodiscard]] constexpr std::uint32_t QDim() const {
    return heads * head_dim;
  }
  [[nodiscard]] constexpr std::uint32_t KvDim() const {
    return kv_heads * head_dim;
  }
};

/// Qwen3 vocabulary ids the prompt uses. The processor aliases the audio
/// tokens onto existing vision ids (<|vision_pad|>, <|system|>, <|user|>).
struct TokenIds {
  std::int32_t im_start{151644};
  std::int32_t im_end{151645};
  std::int32_t endoftext{151643};
  std::int32_t audio_pad{151654};    // <|AUDIO|>
  std::int32_t audio_start{151669};  // <|audio_bos|>
  std::int32_t audio_end{151670};    // <|audio_eos|>
  /// "0".."9" are the single tokens 15..24.
  std::int32_t digit_zero{15};
};

struct ModelConfig {
  AudioEncoderConfig audio;
  AdapterConfig adapter;
  TextConfig text;
  TokenIds tokens;
};

// ---------------------------------------------------------------------------
// Shape math of the Conv2d stem (kernel 3, stride 2, padding 1 on both axes).

/// Output length of one stride-2 convolution: (L - 1) / 2 + 1 = ceil(L / 2).
[[nodiscard]] constexpr std::uint32_t ConvOutLength(std::uint32_t length) {
  return length == 0 ? 0 : (length - 1) / 2 + 1;
}

/// Length after the three convolutions of the stem.
[[nodiscard]] constexpr std::uint32_t StemOutLength(std::uint32_t length) {
  return ConvOutLength(ConvOutLength(ConvOutLength(length)));
}

/// [channels, frequency, time] after conv3 for `frames` log-mel frames of
/// one chunk; the stem flattens it time-major to [time, channels * freq].
struct StemShape {
  std::uint32_t channels;
  std::uint32_t frequency;
  std::uint32_t time;

  [[nodiscard]] constexpr std::uint32_t Flattened() const {
    return channels * frequency;
  }
  [[nodiscard]] constexpr std::size_t Elements() const {
    return static_cast<std::size_t>(channels) * frequency * time;
  }
};

[[nodiscard]] constexpr StemShape StemOutput(
    std::uint32_t frames, const AudioEncoderConfig& config = {}) {
  return StemShape{config.conv_channels, StemOutLength(config.mel_bins),
                   StemOutLength(frames)};
}

/// Index into the flattened stem row of (channel c, frequency f):
/// x.permute(0, 3, 1, 2).flatten(2) -> c * frequency + f.
[[nodiscard]] constexpr std::uint32_t StemFlatIndex(
    std::uint32_t channel, std::uint32_t frequency,
    const AudioEncoderConfig& config = {}) {
  return channel * config.StemFrequency() + frequency;
}

/// Frame counts of the chunks `frames` log-mel frames split into: full
/// chunks of ChunkFrames() and one shorter tail (never empty).
[[nodiscard]] inline std::vector<std::uint32_t> ChunkFrameCounts(
    std::uint32_t frames, const AudioEncoderConfig& config = {}) {
  std::vector<std::uint32_t> chunks;
  const std::uint32_t chunk = config.ChunkFrames();
  while (frames > 0) {
    const std::uint32_t take = frames < chunk ? frames : chunk;
    chunks.push_back(take);
    frames -= take;
  }
  return chunks;
}

/// Encoder tokens for `frames` log-mel frames: ChunkTokens() per full chunk
/// plus the convolved length of the tail. Equals StemOutLength(frames)
/// because the chunk length is a multiple of the downsample rate; the
/// processor relies on exactly that when it counts <|AUDIO|> placeholders
/// without looking at the chunking.
[[nodiscard]] inline std::size_t AudioTokenCount(
    std::uint32_t frames, const AudioEncoderConfig& config = {}) {
  std::size_t tokens = 0;
  for (const std::uint32_t chunk : ChunkFrameCounts(frames, config)) {
    tokens += StemOutLength(chunk);
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Time markers (MossMusicProcessor._build_audio_tokens_with_time_markers).

/// The prompt span standing for `audio_tokens` encoder outputs: <|AUDIO|>
/// placeholders with the digits of the elapsed seconds inserted after every
/// 25 placeholders (2 s), for every even second up to the last full second.
/// The span excludes <|audio_bos|>/<|audio_eos|>.
[[nodiscard]] inline std::vector<std::int32_t> AudioSpanWithMarkers(
    std::size_t audio_tokens, const TokenIds& ids = {}) {
  std::vector<std::int32_t> span;
  // int(audio_tokens / 12.5) without floating point: tokens * 2 / 25.
  const std::size_t full_seconds = audio_tokens * 2 / kTokensPerTwoSeconds;
  std::size_t consumed = 0;
  for (std::size_t second = kMarkerEverySeconds; second <= full_seconds;
       second += kMarkerEverySeconds) {
    const std::size_t marker_position =
        (second / kMarkerEverySeconds) * kTokensPerTwoSeconds;
    if (marker_position > consumed) {
      span.insert(span.end(), marker_position - consumed, ids.audio_pad);
      consumed = marker_position;
    }
    // Decimal digits of `second`, most significant first.
    std::array<std::int32_t, 20> digits{};
    std::size_t count = 0;
    std::size_t value = second;
    do {
      digits[count++] = ids.digit_zero + static_cast<std::int32_t>(value % 10);
      value /= 10;
    } while (value != 0);
    while (count > 0) {
      span.push_back(digits[--count]);
    }
  }
  if (audio_tokens > consumed) {
    span.insert(span.end(), audio_tokens - consumed, ids.audio_pad);
  }
  return span;
}

/// Number of marker digit tokens the span for `audio_tokens` carries.
[[nodiscard]] inline std::size_t MarkerTokenCount(std::size_t audio_tokens) {
  return AudioSpanWithMarkers(audio_tokens).size() - audio_tokens;
}

/// The audio span of a prompt: with markers (the processor's default and
/// infer.py's setting) or plain placeholders.
[[nodiscard]] inline std::vector<std::int32_t> AudioSpan(
    std::size_t audio_tokens, bool time_markers, const TokenIds& ids = {}) {
  if (time_markers) {
    return AudioSpanWithMarkers(audio_tokens, ids);
  }
  return std::vector<std::int32_t>(audio_tokens, ids.audio_pad);
}

/// Prompt positions holding <|AUDIO|>, in order: the rows that receive the
/// adapter output and the DeepStack additions (markers interrupt them).
[[nodiscard]] inline std::vector<std::uint32_t> AudioRowIndices(
    const std::vector<std::int32_t>& prompt, const TokenIds& ids = {}) {
  std::vector<std::uint32_t> rows;
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    if (prompt[index] == ids.audio_pad) {
      rows.push_back(static_cast<std::uint32_t>(index));
    }
  }
  return rows;
}

}  // namespace gufo::models::moss_music

#endif  // GUFO_MODELS_MOSS_MUSIC_CONFIG_HPP_
