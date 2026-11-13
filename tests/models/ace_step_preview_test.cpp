// Live music preview helpers (src/models/ace_step/preview): the clean
// estimate x0 = x_t - t * v, the preview window around the middle of the
// song (with decode margin), the step rule and the mono 24 kHz WAV
// encoder. CPU only, no model.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/ace_step/preview.hpp"

using gufo::models::ace_step::EncodePreviewWav;
using gufo::models::ace_step::kLatentFramesPerSecond;
using gufo::models::ace_step::kPreviewMarginFrames;
using gufo::models::ace_step::kPreviewSampleRate;
using gufo::models::ace_step::kSamplesPerFrame;
using gufo::models::ace_step::MusicCleanEstimate;
using gufo::models::ace_step::MusicPreviewStep;
using gufo::models::ace_step::MusicPreviewWindow;
using gufo::models::ace_step::PreviewWindow;

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL ace_step_preview_test: " << what << '\n';
    ++failures;
  }
}

std::uint32_t U32(const std::string& bytes, std::size_t at) {
  std::uint32_t value = 0;
  for (int i = 3; i >= 0; --i)
    value = (value << 8) | static_cast<std::uint8_t>(bytes[at + i]);
  return value;
}

std::uint16_t U16(const std::string& bytes, std::size_t at) {
  return static_cast<std::uint16_t>(
      static_cast<std::uint8_t>(bytes[at]) |
      (static_cast<std::uint8_t>(bytes[at + 1]) << 8));
}

std::int16_t Sample(const std::string& wav, std::size_t index) {
  return static_cast<std::int16_t>(U16(wav, 44 + 2 * index));
}

// x0 = x_t - t * v, and the Euler step of the pipeline reaches the same
// point on the last step (dt = t).
void TestCleanEstimate() {
  const std::vector<float> x0_true{0.5F, -1.0F, 2.0F, 0.0F};
  const std::vector<float> noise{1.0F, 1.0F, -1.0F, 0.25F};
  const float t = 0.6F;
  std::vector<float> xt(4), v(4);
  for (int i = 0; i < 4; ++i) {
    xt[i] = (1.0F - t) * x0_true[i] + t * noise[i];
    v[i] = noise[i] - x0_true[i];
  }
  std::vector<float> x0;
  MusicCleanEstimate(xt, v, t, &x0);
  Check(x0.size() == 4, "estimate has the input length");
  for (int i = 0; i < 4; ++i)
    Check(std::fabs(x0[i] - x0_true[i]) < 1e-6F,
          "x0 recovers the clean latent at index " + std::to_string(i));
  // The pipeline's last Euler step: x -= v * t.
  for (int i = 0; i < 4; ++i)
    Check(std::fabs((xt[i] - v[i] * t) - x0[i]) < 1e-6F,
          "last Euler step equals the estimate at index " +
              std::to_string(i));
  // At t = 0 the estimate is x itself; mismatched lengths use the shorter.
  MusicCleanEstimate(xt, v, 0.0F, &x0);
  Check(x0 == xt, "t = 0 leaves x unchanged");
  std::vector<float> short_v{1.0F, 1.0F};
  MusicCleanEstimate(xt, short_v, 1.0F, &x0);
  Check(x0.size() == 2 && x0[0] == xt[0] - 1.0F, "shorter v wins");
}

void TestWindow() {
  // 30 s song = 750 frames: 6 s = 150 frames centred, margin 8 per side.
  PreviewWindow w = MusicPreviewWindow(750, 6.0, 8);
  Check(w.count == 150, "6 s window is 150 frames");
  Check(w.first == 300, "window starts at frame 300 (middle)");
  Check(w.decode_first == 292 && w.decode_count == 166,
        "decode slice adds 8 frames per side");
  Check(w.skip_samples() == 8 * kSamplesPerFrame, "skip the front margin");
  Check(w.samples() == 150 * kSamplesPerFrame, "audible samples");
  Check(std::fabs(w.start_seconds() - 12.0) < 1e-9, "window starts at 12 s");
  Check(std::fabs(w.seconds() - 6.0) < 1e-9, "window is 6 s");
  Check(kLatentFramesPerSecond == 25, "25 latent frames per second");

  // Shorter than the window: the whole song, no margin possible.
  w = MusicPreviewWindow(100, 6.0, 8);
  Check(w.first == 0 && w.count == 100, "short song: whole song");
  Check(w.decode_first == 0 && w.decode_count == 100,
        "short song: no margin beyond the song");
  Check(w.skip_samples() == 0, "short song: nothing to skip");

  // Window at the start when the margin would go below zero.
  w = MusicPreviewWindow(160, 6.0, 8);
  Check(w.first == 5 && w.count == 150, "160 frames: window at 5");
  Check(w.decode_first == 0 && w.decode_count == 160,
        "margin clamped to the song on both sides");
  Check(w.skip_samples() == 5 * kSamplesPerFrame, "skip 5 frames");

  // Odd remainder rounds the start down; default arguments; empty song.
  w = MusicPreviewWindow(151, 6.0, 0);
  Check(w.first == 0 && w.count == 150 && w.decode_count == 150,
        "no margin when asked");
  w = MusicPreviewWindow(15000);
  Check(w.count == 150 && w.first == 7425 &&
            w.decode_first == 7425 - kPreviewMarginFrames,
        "defaults: 6 s, margin " + std::to_string(kPreviewMarginFrames));
  w = MusicPreviewWindow(0);
  Check(w.count == 0 && w.decode_count == 0, "empty song: empty window");
  w = MusicPreviewWindow(10, 0.0, 0);
  Check(w.count == 1, "a zero-length window still has one frame");
}

void TestStepRule() {
  Check(!MusicPreviewStep(1, 8, 2), "step 1 of 8 is skipped with every 2");
  Check(MusicPreviewStep(2, 8, 2), "step 2 of 8 previews");
  Check(!MusicPreviewStep(3, 8, 2), "step 3 of 8 is skipped");
  Check(MusicPreviewStep(8, 8, 2), "the last step previews");
  Check(MusicPreviewStep(7, 7, 2), "an odd last step previews");
  Check(MusicPreviewStep(1, 1, 2), "a single step previews");
  Check(MusicPreviewStep(3, 8, 1), "every 1: every step");
  Check(MusicPreviewStep(3, 8, 0), "every 0 counts as 1");
  Check(!MusicPreviewStep(0, 8, 2), "step 0 never");
  Check(!MusicPreviewStep(4, 0, 2), "no steps never");
}

void TestWav() {
  // Planar stereo: left = 0.5 constant, right = -0.5 constant -> mono 0.
  const std::size_t n = 8;
  std::vector<float> planar(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    planar[i] = 0.5F;
    planar[n + i] = -0.5F;
  }
  std::string wav = EncodePreviewWav(planar);
  Check(wav.size() == 44 + 2 * (n / 2), "mono 16-bit at half the samples");
  Check(wav.substr(0, 4) == "RIFF" && wav.substr(8, 8) == "WAVEfmt ",
        "RIFF/WAVE header");
  Check(U32(wav, 4) == 36 + 2 * (n / 2), "RIFF size");
  Check(U16(wav, 20) == 1, "PCM");
  Check(U16(wav, 22) == 1, "mono");
  Check(U32(wav, 24) == static_cast<std::uint32_t>(kPreviewSampleRate) &&
            kPreviewSampleRate == 24000,
        "24 kHz");
  Check(U32(wav, 28) == 24000U * 2U, "byte rate");
  Check(U16(wav, 32) == 2 && U16(wav, 34) == 16, "block align / bits");
  Check(wav.substr(36, 4) == "data" && U32(wav, 40) == 2 * (n / 2),
        "data chunk");
  for (std::size_t i = 0; i < n / 2; ++i)
    Check(Sample(wav, i) == 0, "opposite channels cancel to 0");

  // Both channels 0.5 -> 0.5 * 32767; pairs are averaged (box filter).
  planar.assign(2 * n, 0.5F);
  wav = EncodePreviewWav(planar);
  Check(Sample(wav, 0) == std::lround(0.5F * 32767.0F), "mono 0.5");
  planar[0] = 1.0F;
  planar[1] = 0.0F;
  planar[n + 0] = 1.0F;
  planar[n + 1] = 0.0F;
  wav = EncodePreviewWav(planar);
  Check(Sample(wav, 0) == std::lround(0.5F * 32767.0F),
        "a pair (1, 0) averages to 0.5");

  // A peak above 1 is normalized: 2.0 everywhere becomes full scale.
  planar.assign(2 * n, 2.0F);
  wav = EncodePreviewWav(planar);
  Check(Sample(wav, 1) == 32767, "peak 2 normalized to full scale");

  // Odd sample count drops the last sample; empty input is a valid header.
  planar.assign(2 * 5, 0.25F);
  wav = EncodePreviewWav(planar);
  Check(wav.size() == 44 + 2 * 2, "5 samples -> 2 mono samples");
  wav = EncodePreviewWav(std::vector<float>{});
  Check(wav.size() == 44 && U32(wav, 40) == 0, "empty input: header only");
}

}  // namespace

int main() {
  TestCleanEstimate();
  TestWindow();
  TestStepRule();
  TestWav();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ace_step_preview_test: all checks passed\n";
  return 0;
}
