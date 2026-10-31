#ifndef GUFO_MODELS_QWEN3_ASR_SUBTITLES_HPP_
#define GUFO_MODELS_QWEN3_ASR_SUBTITLES_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_asr/audio.hpp"

namespace gufo::models::qwen3_asr {

/// Qwen3-ASR emits text only, without timestamps. Subtitle timing therefore
/// comes from the audio itself: an energy detector finds speech between
/// measured pauses, and each segment is transcribed on its own. Cue times are
/// the measured sample boundaries of those segments; nothing is interpolated.
struct SpeechSegmentOptions {
  /// Analysis frame (20 ms at 16 kHz).
  std::size_t frame_samples{kAudioSampleRate / 50};
  /// A frame is speech when it is this far above the estimated noise floor
  /// (10th percentile of frame levels) ...
  double noise_margin_db{12.0};
  /// ... but the threshold never rises above the loud level (95th
  /// percentile) minus this margin, so a recording without pauses still
  /// counts as speech ...
  double loud_margin_db{10.0};
  /// ... and never falls below this absolute floor, so digital silence and
  /// faint hiss never become speech.
  double floor_dbfs{-70.0};
  /// Speech with a pause at least this long becomes two runs.
  std::size_t min_pause_samples{kAudioSampleRate * 3 / 10};
  /// Isolated speech shorter than this is treated as a click and dropped.
  std::size_t min_speech_samples{kAudioSampleRate * 12 / 100};
  /// Context kept before and after detected speech (never overlapping).
  std::size_t padding_samples{kAudioSampleRate / 10};
  /// Neighbouring runs are joined while the result stays this short and the
  /// pause between them is at most `merge_gap_samples`.
  std::size_t merge_target_samples{6 * kAudioSampleRate};
  std::size_t merge_gap_samples{kAudioSampleRate};
  /// Longer speech without a pause is cut at the quietest frame inside
  /// [max/3, max] of the run.
  std::size_t max_segment_samples{10 * kAudioSampleRate};
};

/// Ordered, non-overlapping speech segments of a 16-kHz mono waveform.
/// Silence between segments is excluded, so the segments need not cover the
/// whole input. Returns no segments for silent input.
[[nodiscard]] std::vector<AudioChunk> SegmentSpeech(
    std::span<const float> waveform, const SpeechSegmentOptions& options = {});

struct SubtitleCue {
  std::uint64_t start_ms{0};
  std::uint64_t end_ms{0};
  std::string text;
};

/// "HH:MM:SS,mmm" for SRT (separator ',') or "HH:MM:SS.mmm" for WebVTT
/// (separator '.'). Hours grow beyond two digits when needed.
[[nodiscard]] std::string FormatSubtitleTimestamp(std::uint64_t milliseconds,
                                                  char separator);

/// Collapses every whitespace or control-character run to one space and trims
/// the ends, so the text cannot contain the blank line that ends a cue.
[[nodiscard]] std::string NormalizeSubtitleText(std::string_view text);

/// Greedy word wrap counting Unicode scalars, not bytes. A word longer than
/// the limit (for example unspaced CJK text) is split at scalar boundaries.
[[nodiscard]] std::vector<std::string> WrapSubtitleText(
    std::string_view text, std::size_t max_columns = 42);

/// SubRip text. Cues with empty text are skipped, numbering stays dense from
/// 1, and an end time not after the start becomes start + 1 ms.
[[nodiscard]] std::string FormatSrt(std::span<const SubtitleCue> cues);

/// WebVTT text ("WEBVTT" header). Same cue rules as SRT; '&', '<' and '>'
/// are escaped so transcript text cannot form tags or a "-->" arrow.
[[nodiscard]] std::string FormatWebVtt(std::span<const SubtitleCue> cues);

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_SUBTITLES_HPP_
