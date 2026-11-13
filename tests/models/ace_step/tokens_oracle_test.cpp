// ACE-Step 1.5 host side against the torch oracle (ft/ace-step/tools/
// ace_oracle.py, manifest.json + raw INT32 files): the text, lyric and LM
// prompts gufo builds from a request, and their Qwen3 token ids. CPU only;
// the device stages are checked by gufo-ace-check on the GPU.
//   GUFO_ACE_ORACLE    oracle directory (manifest.json, c<N>_*_ids.i32, ...)
//   GUFO_ACE_TOKENIZER Qwen3-Embedding-0.6B tokenizer.json
// Exits 77 (skipped) when either is missing.
// Written with AI assistance (Claude Code).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/ace_step/pipeline.hpp"
#include "src/models/ace_step/tokenizer.hpp"
#include "src/models/ace_step/vae.hpp"

namespace ace = gufo::models::ace_step;
namespace json = gufo::json;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

std::vector<std::uint32_t> ReadIds(const std::filesystem::path& path) {
  const std::string bytes = ReadText(path);
  std::vector<std::int32_t> raw(bytes.size() / sizeof(std::int32_t));
  std::memcpy(raw.data(), bytes.data(), raw.size() * sizeof(std::int32_t));
  return {raw.begin(), raw.end()};
}

// Reads "bpm": 120 or "timesignature": "4" as text.
std::string Text(const json::Value& value, const char* key) {
  const json::Value* member = value.find(key);
  if (member == nullptr)
    return {};
  if (member->is_string())
    return member->str();
  const double number = member->as_double();
  return std::to_string(static_cast<long long>(std::llround(number)));
}

ace::MusicRequest RequestOf(const json::Value& c) {
  ace::MusicRequest request;
  request.caption = c.member_str("caption");
  request.lyrics = c.member_str("lyrics");
  request.language = c.member_str("language", "en");
  request.bpm = Text(c, "bpm");
  request.keyscale = c.member_str("keyscale");
  request.timesignature = Text(c, "timesignature");
  request.duration = c.member_double("duration", 30.0);
  return request;
}

std::string FirstDifference(const std::string& got, const std::string& want) {
  std::size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i])
    ++i;
  return "differs at byte " + std::to_string(i) + " (gufo " +
         std::to_string(got.size()) + ", oracle " +
         std::to_string(want.size()) + " bytes)";
}

}  // namespace

int main() {
  const char* oracle_env = std::getenv("GUFO_ACE_ORACLE");
  const char* tokenizer_env = std::getenv("GUFO_ACE_TOKENIZER");
  if (oracle_env == nullptr || tokenizer_env == nullptr ||
      !std::filesystem::is_regular_file(std::filesystem::path(oracle_env) /
                                        "manifest.json") ||
      !std::filesystem::is_regular_file(tokenizer_env)) {
    std::cout << "SKIP: set GUFO_ACE_ORACLE and GUFO_ACE_TOKENIZER\n";
    return 77;
  }
  const std::filesystem::path oracle = oracle_env;
  const json::Value manifest = json::parse(ReadText(oracle / "manifest.json"));
  const json::Value* report = manifest.find("report");
  const json::Value* cases = report != nullptr ? report->find("cases") : nullptr;
  if (cases == nullptr || cases->items().empty()) {
    std::cerr << "FAIL: manifest.json has no report.cases\n";
    return 1;
  }
  const ace::Tokenizer tokenizer(tokenizer_env);
  constexpr std::uint32_t kEndOfText = 151643;  // appended by the encoder

  std::size_t index = 0;
  for (const json::Value& c : cases->items()) {
    const std::string name = "c" + std::to_string(index++);
    const ace::MusicRequest request = RequestOf(c);
    const std::string text = ace::MusicTextPrompt(request);
    const std::string lyric = ace::MusicLyricPrompt(request);
    Expect(text == c.member_str("text_prompt"),
           name + " text prompt " +
               FirstDifference(text, c.member_str("text_prompt")));
    Expect(lyric == c.member_str("lyric_prompt"),
           name + " lyric prompt " +
               FirstDifference(lyric, c.member_str("lyric_prompt")));
    auto text_ids = tokenizer.Encode(text);
    text_ids.push_back(kEndOfText);
    auto lyric_ids = tokenizer.Encode(lyric);
    lyric_ids.push_back(kEndOfText);
    const auto want_text = ReadIds(oracle / (name + "_text_ids.i32"));
    const auto want_lyric = ReadIds(oracle / (name + "_lyric_ids.i32"));
    Expect(!want_text.empty() && text_ids == want_text,
           name + " text tokens " + std::to_string(text_ids.size()) + "/" +
               std::to_string(want_text.size()));
    Expect(!want_lyric.empty() && lyric_ids == want_lyric,
           name + " lyric tokens " + std::to_string(lyric_ids.size()) + "/" +
               std::to_string(want_lyric.size()));
    std::cout << name << ": prompts and " << text_ids.size() << " + "
              << lyric_ids.size() << " tokens checked\n";
  }

  // The 5 Hz audio-code LM prompt of case 0 and its unconditional twin.
  if (const json::Value* lm = report->find("lm");
      lm != nullptr && std::filesystem::is_regular_file(oracle / "lm_ids.i32")) {
    const std::string prompt =
        ace::MusicLmPrompt(RequestOf(cases->items().front()));
    Expect(prompt == lm->member_str("lm_prompt"),
           "LM prompt " + FirstDifference(prompt, lm->member_str("lm_prompt")));
    Expect(tokenizer.Encode(prompt) == ReadIds(oracle / "lm_ids.i32"),
           "LM prompt tokens");
    if (std::filesystem::is_regular_file(oracle / "lm_uncond_ids.i32"))
      Expect(tokenizer.Encode(ace::MusicLmUnconditionalPrompt()) ==
                 ReadIds(oracle / "lm_uncond_ids.i32"),
             "unconditional LM prompt tokens");
    std::cout << "lm: prompt and tokens checked\n";
  }

  // The WAV writer: 16-bit PCM stereo, interleaved from planar samples.
  const std::vector<float> planar = {0.5F, -1.0F, 0.0F, 2.0F};
  const std::string wav = ace::EncodeStereoWav(planar, ace::kSampleRate);
  Expect(wav.size() == 44 + 8 && wav.compare(0, 4, "RIFF") == 0 &&
             wav.compare(8, 8, "WAVEfmt ") == 0,
         "stereo WAV header");
  const auto sample = [&](std::size_t i) {
    return static_cast<std::int16_t>(
        static_cast<std::uint8_t>(wav[44 + i * 2]) |
        (static_cast<std::uint8_t>(wav[45 + i * 2]) << 8));
  };
  Expect(sample(0) == 16384 && sample(1) == 0 && sample(2) == -32767 &&
             sample(3) == 32767,
         "stereo WAV interleaves and clamps");

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "ACE-Step token oracle test passed.\n";
  return 0;
}
