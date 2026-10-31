#include "src/models/qwen3_asr/subtitles.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_asr/audio.hpp"

namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

constexpr std::size_t kRate = qwen3_asr::kAudioSampleRate;

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "FAIL qwen3_asr_subtitles_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

void CheckEqual(const std::string& actual, const std::string& expected,
                std::string_view message) {
  if (actual != expected) {
    std::cerr << "expected:\n[" << expected << "]\nactual:\n[" << actual
              << "]\n";
    Fail(message);
  }
}

/// Appends `seconds` of a 220-Hz tone (amplitude 0.3) or of silence.
void Append(std::vector<float>* waveform, double seconds, bool tone,
            float amplitude = 0.3F) {
  const auto samples = static_cast<std::size_t>(seconds * kRate);
  for (std::size_t i = 0; i < samples; ++i) {
    const double t = static_cast<double>(waveform->size()) / kRate;
    waveform->push_back(
        tone ? amplitude * static_cast<float>(
                               std::sin(2.0 * std::numbers::pi * 220.0 * t))
             : 0.0F);
  }
}

double Seconds(std::size_t samples) {
  return static_cast<double>(samples) / kRate;
}

void TestTimestamps() {
  CheckEqual(qwen3_asr::FormatSubtitleTimestamp(0, ','), "00:00:00,000",
             "zero SRT timestamp");
  CheckEqual(qwen3_asr::FormatSubtitleTimestamp(3'723'004, ','),
             "01:02:03,004", "SRT timestamp fields");
  CheckEqual(qwen3_asr::FormatSubtitleTimestamp(3'723'004, '.'),
             "01:02:03.004", "WebVTT uses a full stop");
  CheckEqual(qwen3_asr::FormatSubtitleTimestamp(59'999, ','), "00:00:59,999",
             "no rounding into the next second");
  CheckEqual(qwen3_asr::FormatSubtitleTimestamp(360'000'000, '.'),
             "100:00:00.000", "hours grow beyond two digits");
}

void TestTextNormalizationAndWrap() {
  CheckEqual(qwen3_asr::NormalizeSubtitleText("  Hallo\r\n\r\nWelt\t !  "),
             "Hallo Welt !", "whitespace runs collapse and ends trim");
  CheckEqual(qwen3_asr::NormalizeSubtitleText(std::string("a\0b\x7f" "c", 5)),
             "a b c", "control characters become spaces");
  CheckEqual(qwen3_asr::NormalizeSubtitleText(" \n\t "), "",
             "blank text stays empty");

  Check(qwen3_asr::WrapSubtitleText("").empty(), "empty text has no lines");
  const auto short_lines = qwen3_asr::WrapSubtitleText("Guten Morgen.");
  Check(short_lines.size() == 1 && short_lines[0] == "Guten Morgen.",
        "short text stays on one line");
  const auto wrapped = qwen3_asr::WrapSubtitleText(
      "Das ist ein ziemlich langer Satz, der auf zwei Zeilen umbrechen muss.");
  Check(wrapped.size() == 2 &&
            wrapped[0] == "Das ist ein ziemlich langer Satz, der auf" &&
            wrapped[1] == "zwei Zeilen umbrechen muss.",
        "greedy wrap at 42 columns");
  // Two words of 20 umlauts are 81 bytes but 41 columns: one line.
  std::string umlauts;
  for (int i = 0; i < 20; ++i) umlauts += "\xC3\xA4";  // U+00E4
  const auto utf8 = qwen3_asr::WrapSubtitleText(umlauts + " " + umlauts);
  Check(utf8.size() == 1, "columns count Unicode scalars, not bytes");
  // 50 CJK scalars without spaces split at a scalar boundary.
  std::string cjk;
  for (int i = 0; i < 50; ++i) cjk += "\xE4\xBD\xA0";  // U+4F60
  const auto cjk_lines = qwen3_asr::WrapSubtitleText(cjk);
  Check(cjk_lines.size() == 2 && cjk_lines[0].size() == 42U * 3U &&
            cjk_lines[1].size() == 8U * 3U,
        "unspaced text splits at scalar boundaries");
  const auto exact = qwen3_asr::WrapSubtitleText(std::string(84, 'x'), 42);
  Check(exact.size() == 2 && exact[0].size() == 42 && exact[1].size() == 42,
        "a word of exactly two lines leaves no empty line");
}

void TestSrt() {
  const std::vector<qwen3_asr::SubtitleCue> cues = {
      {.start_ms = 1'000, .end_ms = 2'500, .text = "Hallo Welt."},
      {.start_ms = 3'000, .end_ms = 4'000, .text = "  \n "},
      {.start_ms = 61'000,
       .end_ms = 63'250,
       .text = "Erste Zeile\n\nzweite <i>Zeile</i> & mehr --> Ende."},
      {.start_ms = 70'000, .end_ms = 70'000, .text = "Kurz."},
  };
  CheckEqual(qwen3_asr::FormatSrt(cues),
             "1\n00:00:01,000 --> 00:00:02,500\nHallo Welt.\n\n"
             "2\n00:01:01,000 --> 00:01:03,250\n"
             "Erste Zeile zweite <i>Zeile</i> & mehr -->\nEnde.\n\n"
             "3\n00:01:10,000 --> 00:01:10,001\nKurz.\n\n",
             "SRT numbering, times, wrap, empty cue skipped, end > start");
  CheckEqual(qwen3_asr::FormatSrt({}), "", "no cues give an empty SRT");
}

void TestWebVtt() {
  const std::vector<qwen3_asr::SubtitleCue> cues = {
      {.start_ms = 500, .end_ms = 1'750, .text = "a < b && c > d --> e"},
      {.start_ms = 3'600'000,
       .end_ms = 3'601'000,
       .text = "Gr\xC3\xBC\xC3\x9F" "e, \xE4\xBD\xA0\xE5\xA5\xBD!"},
  };
  CheckEqual(qwen3_asr::FormatWebVtt(cues),
             "WEBVTT\n\n"
             "1\n00:00:00.500 --> 00:00:01.750\n"
             "a &lt; b &amp;&amp; c &gt; d --&gt; e\n\n"
             "2\n01:00:00.000 --> 01:00:01.000\nGr\xC3\xBC\xC3\x9F" "e, \xE4\xBD\xA0\xE5\xA5\xBD!\n\n",
             "WebVTT header, full-stop times and escaping");
  CheckEqual(qwen3_asr::FormatWebVtt({}), "WEBVTT\n\n",
             "no cues give a header-only WebVTT");
}

void TestSegmentation() {
  Check(qwen3_asr::SegmentSpeech({}).empty(), "empty input has no segments");
  const std::vector<float> silence(3 * kRate, 0.0F);
  Check(qwen3_asr::SegmentSpeech(silence).empty(),
        "digital silence has no segments");

  // 0.5 s silence, 1 s speech, 0.8 s pause, 1 s speech, 2 s silence.
  std::vector<float> two;
  Append(&two, 0.5, false);
  Append(&two, 1.0, true);
  Append(&two, 0.8, false);
  Append(&two, 1.0, true);
  Append(&two, 2.0, false);
  qwen3_asr::SpeechSegmentOptions separate;
  separate.merge_target_samples = 0;
  const auto apart = qwen3_asr::SegmentSpeech(two, separate);
  Check(apart.size() == 2, "a 0.8 s pause separates two segments");
  Check(std::abs(Seconds(apart[0].offset) - 0.4) < 0.025 &&
            std::abs(Seconds(apart[0].offset + apart[0].samples) - 1.6) <
                0.025 &&
            std::abs(Seconds(apart[1].offset) - 2.2) < 0.025 &&
            std::abs(Seconds(apart[1].offset + apart[1].samples) - 3.4) <
                0.025,
        "segment boundaries are the measured speech plus 100 ms padding");
  const auto joined = qwen3_asr::SegmentSpeech(two);
  Check(joined.size() == 1 && joined[0].offset == apart[0].offset &&
            joined[0].offset + joined[0].samples ==
                apart[1].offset + apart[1].samples,
        "short neighbours merge into one readable cue");

  // A 0.2 s pause is shorter than the minimum pause: one run.
  std::vector<float> bridged;
  Append(&bridged, 0.5, false);
  Append(&bridged, 1.0, true);
  Append(&bridged, 0.2, false);
  Append(&bridged, 1.0, true);
  Append(&bridged, 0.5, false);
  Check(qwen3_asr::SegmentSpeech(bridged, separate).size() == 1,
        "pauses shorter than 300 ms do not split");

  // A 40 ms click is dropped, real speech is kept.
  std::vector<float> click;
  Append(&click, 1.0, false);
  Append(&click, 0.04, true);
  Append(&click, 1.0, false);
  Append(&click, 1.0, true);
  Append(&click, 1.0, false);
  const auto no_click = qwen3_asr::SegmentSpeech(click, separate);
  Check(no_click.size() == 1 && Seconds(no_click[0].offset) > 1.5,
        "isolated clicks shorter than 120 ms are dropped");

  // 25 s of speech without pause: cut at the quiet dips, never above 10 s.
  std::vector<float> long_speech;
  Append(&long_speech, 7.0, true);
  Append(&long_speech, 0.1, true, 0.01F);  // a quiet dip, below 300 ms
  Append(&long_speech, 17.9, true);
  const auto pieces = qwen3_asr::SegmentSpeech(long_speech);
  std::size_t covered = 0;
  for (const auto& piece : pieces) {
    Check(piece.samples <= 10 * kRate, "no segment exceeds 10 s");
    Check(piece.offset == covered, "cut pieces of one run touch exactly");
    covered = piece.offset + piece.samples;
  }
  Check(pieces.size() >= 3 && covered == long_speech.size(),
        "continuous speech is fully covered");
  Check(std::abs(Seconds(pieces[0].samples) - 7.0) < 0.1,
        "the cut lands on the measured quiet dip");

  // Constant background noise at about -50 dBFS does not become speech.
  std::vector<float> noisy = two;
  std::uint32_t state = 1;
  for (float& sample : noisy) {
    state = state * 1664525U + 1013904223U;
    sample += 0.003F * (static_cast<float>(state >> 8U) / 8388608.0F - 1.0F);
  }
  const auto noisy_segments = qwen3_asr::SegmentSpeech(noisy, separate);
  Check(noisy_segments.size() == 2 &&
            std::abs(Seconds(noisy_segments[0].offset) - 0.4) < 0.05,
        "the threshold follows the noise floor");

  // Segments are ordered and never overlap.
  for (std::size_t i = 1; i < apart.size(); ++i) {
    Check(apart[i].offset >= apart[i - 1].offset + apart[i - 1].samples,
          "segments are ordered without overlap");
  }
}

/// Diagnostic: `qwen3_asr_subtitles_test file.wav` prints the segments.
int PrintSegments(const char* path) {
  std::ifstream input(path, std::ios::binary);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>()};
  qwen3_asr::AudioBuffer audio;
  std::string error;
  if (!qwen3_asr::DecodeWav(std::as_bytes(std::span(bytes)), &audio, &error)) {
    std::cerr << "cannot decode " << path << ": " << error << '\n';
    return 2;
  }
  const auto waveform = qwen3_asr::ResampleMono16k(audio);
  const auto segments = qwen3_asr::SegmentSpeech(waveform);
  std::cout << "duration " << Seconds(waveform.size()) << " s, "
            << segments.size() << " segments\n";
  for (const auto& segment : segments) {
    std::cout << qwen3_asr::FormatSubtitleTimestamp(
                     (segment.offset * 1000U + kRate / 2U) / kRate, '.')
              << " --> "
              << qwen3_asr::FormatSubtitleTimestamp(
                     ((segment.offset + segment.samples) * 1000U + kRate / 2U) /
                         kRate,
                     '.')
              << "  (" << Seconds(segment.samples) << " s)\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    return PrintSegments(argv[1]);
  }
  TestTimestamps();
  TestTextNormalizationAndWrap();
  TestSrt();
  TestWebVtt();
  TestSegmentation();
  std::cout << "PASS qwen3_asr_subtitles_test\n";
  return 0;
}
