// ACE-Step 1.5 host side: model file layout, the text / lyric / LM prompts,
// the seeded noise and the WAV writer. CPU only (split from pipeline.cpp so
// `gufo serve music` request handling and the token oracle test build
// without HIP).
// Written with AI assistance (Claude Code).
#include "src/models/ace_step/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <system_error>

namespace gufo::models::ace_step {
namespace {

std::string OrNa(const std::string& value) {
  return value.empty() ? "N/A" : value;
}

void AppendU16(std::string* out, std::uint16_t value) {
  out->push_back(static_cast<char>(value & 0xFFU));
  out->push_back(static_cast<char>(value >> 8U));
}

void AppendU32(std::string* out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out->push_back(static_cast<char>((value >> shift) & 0xFFU));
}

bool MissingFile(std::string* error, const std::filesystem::path& path) {
  if (error != nullptr)
    *error = "ACE-Step model file missing: " + path.string();
  return false;
}

}  // namespace

bool ModelPaths::FromRoot(const std::filesystem::path& root, ModelPaths* paths,
                          std::string* error) {
  paths->dit = root / "dit.safetensors";
  paths->text = root / "text_encoder.safetensors";
  paths->vae = root / "vae.safetensors";
  paths->tokenizer = root / "tokenizer.json";
  paths->silence = root / "silence_latent.safetensors";
  for (const auto* path : {&paths->dit, &paths->text, &paths->vae,
                           &paths->tokenizer, &paths->silence}) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(*path, ec))
      return MissingFile(error, *path);
  }
  std::error_code ec;
  if (std::filesystem::is_regular_file(root / "lm.safetensors", ec))
    paths->lm = root / "lm.safetensors";
  return true;
}

std::string MusicTextPrompt(const MusicRequest& r) {
  const std::string metas =
      "- bpm: " + OrNa(r.bpm) + "\n- timesignature: " + OrNa(r.timesignature) +
      "\n- keyscale: " + OrNa(r.keyscale) +
      "\n- duration: " + std::to_string(static_cast<int>(r.duration)) +
      " seconds\n";
  return "# Instruction\nFill the audio semantic mask based on the given "
         "conditions:\n\n# Caption\n" +
         r.caption + "\n\n# Metas\n" + metas + "<|endoftext|>\n";
}

std::string MusicLyricPrompt(const MusicRequest& r) {
  const std::string lyrics = r.lyrics.empty() ? "[Instrumental]" : r.lyrics;
  return "# Languages\n" +
         (r.language.empty() ? std::string("en") : r.language) +
         "\n\n# Lyric\n" + lyrics + "<|endoftext|>";
}

namespace {

constexpr const char* kLmSystem =
    "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based "
    "on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n";

std::string Instrumental(const std::string& lyrics) {
  return lyrics.empty() ? "[Instrumental]" : lyrics;
}

}  // namespace

std::string MusicLmPrompt(const MusicRequest& r) {
  // Sorted YAML of the user metadata, as ComfyUI writes it.
  std::string metas;
  if (!r.bpm.empty())
    metas += "bpm: " + r.bpm + "\n";
  metas +=
      "duration: " + std::to_string(static_cast<int>(std::ceil(r.duration))) +
      "\n";
  if (!r.keyscale.empty())
    metas += "keyscale: " + r.keyscale + "\n";
  if (!r.timesignature.empty())
    metas += "timesignature: " + r.timesignature + "\n";
  metas.pop_back();
  return std::string(kLmSystem) + "# Caption\n" + r.caption + "\n\n# Lyric\n" +
         Instrumental(r.lyrics) +
         "\n<|im_end|>\n<|im_start|>assistant\n<think>\n" + metas +
         "\n</think>\n\n";
}

std::string MusicLmUnconditionalPrompt() {
  return std::string(kLmSystem) +
         "NO USER INPUT<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
         "</think>\n\n";
}

std::vector<float> MusicNoise(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> noise(count);
  for (float& value : noise)
    value = normal(engine);
  return noise;
}

std::string EncodeStereoWav(std::span<const float> planar, int sample_rate) {
  const std::size_t samples = planar.size() / 2;
  const auto data_bytes = static_cast<std::uint32_t>(samples * 4);
  std::string out;
  out.reserve(44U + data_bytes);
  out.append("RIFF", 4);
  AppendU32(&out, 36U + data_bytes);
  out.append("WAVEfmt ", 8);
  AppendU32(&out, 16);
  AppendU16(&out, 1);  // PCM
  AppendU16(&out, 2);
  AppendU32(&out, static_cast<std::uint32_t>(sample_rate));
  AppendU32(&out, static_cast<std::uint32_t>(sample_rate) * 4U);
  AppendU16(&out, 4);
  AppendU16(&out, 16);
  out.append("data", 4);
  AppendU32(&out, data_bytes);
  for (std::size_t i = 0; i < samples; ++i)
    for (int ch = 0; ch < 2; ++ch) {
      const float v = std::clamp(planar[ch * samples + i], -1.0F, 1.0F);
      AppendU16(&out, static_cast<std::uint16_t>(static_cast<std::int16_t>(
                          std::lround(v * 32767.0F))));
    }
  return out;
}

}  // namespace gufo::models::ace_step
