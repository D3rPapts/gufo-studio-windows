// CPU tests for MMAudio's host side: frame selection (MMAudio read_frames and
// ComfyUI-MMAudio's consecutive frames), resizing, noise, WAV and the CLIP
// tokenizer (when GUFO_MMAUDIO_TOKENIZER points at a directory with
// vocab.json / merges.txt; the ids were produced by transformers'
// CLIPTokenizer for laion/CLIP-ViT-H-14-laion2B-s32B-b79K).
// Written with AI assistance (Claude Code).
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/mmaudio/preprocess.hpp"
#include "src/models/mmaudio/tokenizer.hpp"

namespace mm = gufo::models::mmaudio;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::vector<std::uint8_t> Solid(int width, int height, std::uint8_t r,
                                std::uint8_t g, std::uint8_t b) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  for (std::size_t i = 0; i < rgb.size(); i += 3) {
    rgb[i] = r;
    rgb[i + 1] = g;
    rgb[i + 2] = b;
  }
  return rgb;
}

void TestResize() {
  // A solid image stays solid through the bicubic filter.
  const auto rgb = Solid(640, 360, 200, 100, 50);
  const auto out = mm::ResizeBicubic(rgb, 640, 360, 384, 384);
  Expect(out.size() == 384u * 384u * 3u, "resize size");
  Expect(out[0] == 200 && out[1] == 100 && out[2] == 50, "solid resize");
  const auto clip = mm::ClipPixels(rgb, 640, 360);
  Expect(clip.size() == 3u * 384u * 384u, "clip pixels size");
  Expect(std::fabs(clip[0] - (200.0F / 255.0F - 0.48145466F) / 0.26862954F) <
             1e-5F,
         "clip normalization");
  const auto sync = mm::SyncPixels(rgb, 640, 360);
  Expect(sync.size() == 3u * 224u * 224u, "sync pixels size");
  Expect(std::fabs(sync[0] - (200.0F / 255.0F - 0.5F) / 0.5F) < 1e-5F,
         "sync normalization");
}

int CountAdded(mm::VideoSampler& sampler, int frames) {
  const auto rgb = Solid(32, 32, 10, 20, 30);
  int fed = 0;
  for (; fed < frames; ++fed)
    if (!sampler.Add(rgb, 32, 32)) {
      ++fed;
      break;
    }
  return fed;
}

void TestSampler() {
  {
    // 25 fps source, 8 s: 64 CLIP and 200 Synchformer frames.
    mm::VideoSampler sampler(8.0, 25.0, mm::VideoSampler::Mode::kResample);
    CountAdded(sampler, 400);
    const auto result = sampler.Finish();
    Expect(result.clip_frames == 64, "resample clip 64, got " +
                                         std::to_string(result.clip_frames));
    Expect(result.sync_frames == 200, "resample sync 200, got " +
                                          std::to_string(result.sync_frames));
    Expect(result.duration == 8.0, "resample duration");
  }
  {
    // A 3 s video (72 frames at 24 fps) asked for 8 s: the last frame is at
    // 2.958 s, so 24 CLIP / 74 sync frames and duration 74 / 25 = 2.96 s
    // (computed with MMAudio's read_frames / load_video logic in Python).
    mm::VideoSampler sampler(8.0, 24.0, mm::VideoSampler::Mode::kResample);
    CountAdded(sampler, 72);
    const auto result = sampler.Finish();
    Expect(std::fabs(result.duration - 2.96) < 1e-9,
           "short video duration " + std::to_string(result.duration));
    Expect(result.clip_frames == 23 && result.sync_frames == 74,
           "short video frames");
  }
  {
    // ComfyUI-MMAudio: consecutive frames, 100 frames for 8 s -> 4 s.
    mm::VideoSampler sampler(8.0, 30.0, mm::VideoSampler::Mode::kConsecutive);
    CountAdded(sampler, 100);
    const auto result = sampler.Finish();
    Expect(std::fabs(result.duration - 4.0) < 1e-9,
           "consecutive duration " + std::to_string(result.duration));
    Expect(result.clip_frames == 32 && result.sync_frames == 100,
           "consecutive frames");
  }
}

void TestNoiseAndWav() {
  const auto a = mm::GaussianNoise(42, 10001);
  const auto b = mm::GaussianNoise(42, 10001);
  Expect(a == b, "noise is deterministic");
  double mean = 0, var = 0;
  for (const float v : a)
    mean += v;
  mean /= a.size();
  for (const float v : a)
    var += (v - mean) * (v - mean);
  var /= a.size();
  Expect(std::fabs(mean) < 0.05 && std::fabs(var - 1.0) < 0.05,
         "noise is standard normal");
  const std::vector<float> samples = {0.0F, 1.0F, -1.0F};
  const std::string wav = mm::EncodeMonoWav(samples, 44100);
  Expect(wav.size() == 44 + 6 && wav.substr(0, 4) == "RIFF" &&
             wav.substr(8, 4) == "WAVE",
         "wav header");
}

void TestTokenizer() {
  const char* dir = std::getenv("GUFO_MMAUDIO_TOKENIZER");
  if (dir == nullptr)
    return;
  mm::ClipTokenizer tokenizer;
  std::string error;
  if (!tokenizer.Load(dir, &error)) {
    Expect(false, "tokenizer load: " + error);
    return;
  }
  const auto ids = tokenizer.Encode("dog barking");
  Expect(ids.size() == 77u, "77 tokens");
  Expect(ids[0] == mm::ClipTokenizer::kStartOfText, "start token");
  // "dog</w>" = 1929, "barking</w>" = 32676 in CLIP's vocabulary.
  Expect(ids[1] == 1929 && ids[2] == 32676 &&
             ids[3] == mm::ClipTokenizer::kEndOfText && ids[4] == 0,
         "dog barking ids " + std::to_string(ids[1]) + " " +
             std::to_string(ids[2]));
  const auto empty = tokenizer.Encode("");
  Expect(empty[0] == mm::ClipTokenizer::kStartOfText &&
             empty[1] == mm::ClipTokenizer::kEndOfText && empty[2] == 0,
         "empty prompt");
}

}  // namespace

int main() {
  TestResize();
  TestSampler();
  TestNoiseAndWav();
  TestTokenizer();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "mmaudio preprocess tests passed\n";
  return 0;
}
