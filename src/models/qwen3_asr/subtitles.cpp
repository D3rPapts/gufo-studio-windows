#include "src/models/qwen3_asr/subtitles.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::qwen3_asr {
namespace {

struct FrameRun {
  std::size_t begin;  // first frame
  std::size_t end;    // one past the last frame
};

bool IsUtf8Continuation(char byte) {
  return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

std::size_t ScalarCount(std::string_view text) {
  return static_cast<std::size_t>(
      std::ranges::count_if(text, [](char c) { return !IsUtf8Continuation(c); }));
}

/// Byte length of the longest prefix holding at most `scalars` scalars.
std::size_t PrefixBytes(std::string_view text, std::size_t scalars) {
  std::size_t seen = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (!IsUtf8Continuation(text[i])) {
      if (seen == scalars) {
        return i;
      }
      ++seen;
    }
  }
  return text.size();
}

std::string FormatCues(std::span<const SubtitleCue> cues, char separator,
                       bool escape_html, std::string header) {
  std::string output = std::move(header);
  std::size_t index = 0;
  for (const SubtitleCue& cue : cues) {
    const std::string text = NormalizeSubtitleText(cue.text);
    if (text.empty()) {
      continue;
    }
    ++index;
    const std::uint64_t end =
        cue.end_ms > cue.start_ms ? cue.end_ms : cue.start_ms + 1U;
    output += std::to_string(index);
    output += '\n';
    output += FormatSubtitleTimestamp(cue.start_ms, separator);
    output += " --> ";
    output += FormatSubtitleTimestamp(end, separator);
    output += '\n';
    for (const std::string& line : WrapSubtitleText(text)) {
      if (!escape_html) {
        output += line;
      } else {
        for (const char c : line) {
          switch (c) {
            case '&':
              output += "&amp;";
              break;
            case '<':
              output += "&lt;";
              break;
            case '>':
              output += "&gt;";
              break;
            default:
              output += c;
          }
        }
      }
      output += '\n';
    }
    output += '\n';
  }
  return output;
}

}  // namespace

std::vector<AudioChunk> SegmentSpeech(std::span<const float> waveform,
                                      const SpeechSegmentOptions& options) {
  const std::size_t frame = std::max<std::size_t>(1, options.frame_samples);
  if (waveform.empty()) {
    return {};
  }
  const std::size_t frames = (waveform.size() + frame - 1U) / frame;
  std::vector<double> level(frames);
  for (std::size_t f = 0; f < frames; ++f) {
    const std::size_t begin = f * frame;
    const std::size_t end = std::min(waveform.size(), begin + frame);
    double energy = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      energy += static_cast<double>(waveform[i]) * waveform[i];
    }
    level[f] = 10.0 * std::log10(energy / static_cast<double>(end - begin) +
                                 1e-20);
  }
  std::vector<double> sorted = level;
  std::ranges::sort(sorted);
  const double noise = sorted[(frames - 1U) * 10U / 100U];
  const double loud = sorted[(frames - 1U) * 95U / 100U];
  double threshold = noise + options.noise_margin_db;
  threshold = std::min(threshold, loud - options.loud_margin_db);
  threshold = std::max(threshold, options.floor_dbfs);

  // Speech runs in frames, bridging pauses shorter than the minimum.
  const auto to_frames = [&](std::size_t samples) {
    return (samples + frame - 1U) / frame;
  };
  const std::size_t min_pause = to_frames(options.min_pause_samples);
  const std::size_t min_speech = to_frames(options.min_speech_samples);
  std::vector<FrameRun> runs;
  for (std::size_t f = 0; f < frames;) {
    if (level[f] <= threshold) {
      ++f;
      continue;
    }
    std::size_t end = f + 1U;
    while (end < frames && level[end] > threshold) {
      ++end;
    }
    if (!runs.empty() && f - runs.back().end < min_pause) {
      runs.back().end = end;
    } else {
      runs.push_back({f, end});
    }
    f = end;
  }
  std::erase_if(runs, [&](const FrameRun& run) {
    return run.end - run.begin < min_speech;
  });

  // Cut speech longer than the maximum at its quietest frame.
  const std::size_t max_frames =
      std::max<std::size_t>(3, options.max_segment_samples / frame);
  std::vector<FrameRun> pieces;
  for (FrameRun run : runs) {
    while (run.end - run.begin > max_frames) {
      std::size_t best = run.begin + max_frames / 3U;
      for (std::size_t f = best + 1U; f < run.begin + max_frames; ++f) {
        if (level[f] < level[best]) {
          best = f;
        }
      }
      pieces.push_back({run.begin, best});
      run.begin = best;
    }
    pieces.push_back(run);
  }

  // Frames to samples, padding outward without overlapping a neighbour.
  std::vector<AudioChunk> segments;
  std::size_t previous_end = 0;
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    const std::size_t begin = pieces[i].begin * frame;
    const std::size_t end = std::min(waveform.size(), pieces[i].end * frame);
    const std::size_t next_begin =
        i + 1U < pieces.size() ? pieces[i + 1U].begin * frame : waveform.size();
    const std::size_t padded_begin =
        std::max(previous_end, begin > options.padding_samples
                                   ? begin - options.padding_samples
                                   : std::size_t{0});
    const std::size_t padded_end =
        std::min({waveform.size(), next_begin, end + options.padding_samples});
    segments.push_back({padded_begin, padded_end - padded_begin});
    previous_end = padded_end;
  }

  // Join short neighbours into readable cues.
  std::vector<AudioChunk> merged;
  for (const AudioChunk& segment : segments) {
    if (!merged.empty()) {
      AudioChunk& last = merged.back();
      const std::size_t last_end = last.offset + last.samples;
      const std::size_t gap = segment.offset - last_end;
      const std::size_t combined = segment.offset + segment.samples - last.offset;
      if (gap <= options.merge_gap_samples &&
          combined <= options.merge_target_samples) {
        last.samples = combined;
        continue;
      }
    }
    merged.push_back(segment);
  }
  return merged;
}

std::string FormatSubtitleTimestamp(std::uint64_t milliseconds,
                                    char separator) {
  const std::uint64_t hours = milliseconds / 3'600'000U;
  const std::uint64_t minutes = milliseconds / 60'000U % 60U;
  const std::uint64_t seconds = milliseconds / 1'000U % 60U;
  const std::uint64_t millis = milliseconds % 1'000U;
  char buffer[48];
  std::snprintf(buffer, sizeof(buffer), "%02llu:%02llu:%02llu%c%03llu",
                static_cast<unsigned long long>(hours),
                static_cast<unsigned long long>(minutes),
                static_cast<unsigned long long>(seconds), separator,
                static_cast<unsigned long long>(millis));
  return buffer;
}

std::string NormalizeSubtitleText(std::string_view text) {
  std::string output;
  output.reserve(text.size());
  bool pending_space = false;
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (byte < 0x20U || byte == 0x7FU || c == ' ') {
      pending_space = !output.empty();
      continue;
    }
    if (pending_space) {
      output += ' ';
      pending_space = false;
    }
    output += c;
  }
  return output;
}

std::vector<std::string> WrapSubtitleText(std::string_view text,
                                          std::size_t max_columns) {
  max_columns = std::max<std::size_t>(1, max_columns);
  std::vector<std::string> lines;
  std::string line;
  std::size_t line_columns = 0;
  const auto flush = [&] {
    if (!line.empty()) {
      lines.push_back(std::move(line));
      line.clear();
      line_columns = 0;
    }
  };
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    if (text[cursor] == ' ') {
      ++cursor;
      continue;
    }
    const std::size_t space = text.find(' ', cursor);
    std::string_view word = text.substr(
        cursor, space == std::string_view::npos ? std::string_view::npos
                                                : space - cursor);
    cursor += word.size();
    std::size_t columns = ScalarCount(word);
    if (line_columns > 0 && line_columns + 1U + columns <= max_columns) {
      line += ' ';
      line += word;
      line_columns += 1U + columns;
      continue;
    }
    flush();
    while (columns > max_columns) {
      const std::size_t bytes = PrefixBytes(word, max_columns);
      lines.emplace_back(word.substr(0, bytes));
      word.remove_prefix(bytes);
      columns -= max_columns;
    }
    line = std::string(word);
    line_columns = columns;
  }
  flush();
  return lines;
}

std::string FormatSrt(std::span<const SubtitleCue> cues) {
  return FormatCues(cues, ',', false, {});
}

std::string FormatWebVtt(std::span<const SubtitleCue> cues) {
  return FormatCues(cues, '.', true, "WEBVTT\n\n");
}

}  // namespace gufo::models::qwen3_asr
