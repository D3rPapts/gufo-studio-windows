#include "src/models/qwen3_tts/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kText =
    "The boy who lived. Mr. and Mrs. Dursley, of number four, Privet Drive, "
    "were proud to say that they were perfectly normal, thank you very much. "
    "They were the last people you'd expect to be involved in anything strange "
    "or mysterious, because they just didn't hold with such nonsense.";

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_tokenizer_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path model_root =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";
  if (!std::filesystem::is_regular_file(model_root / "vocab.json")) {
    std::cerr
        << "SKIP qwen3_tts_tokenizer_test: external model is unavailable\n";
    return 77;
  }

  gufo::models::qwen3_tts::Tokenizer tokenizer;
  std::string error;
  Check(
      gufo::models::qwen3_tts::Tokenizer::Load(model_root, &tokenizer, &error),
      error);
  std::vector<std::uint32_t> ids;
  Check(tokenizer.EncodeAssistantPrompt(kText, &ids, &error), error);
  const std::vector<std::uint32_t> expected = {
      151644, 77091, 198,   785,   8171,   879,   12163,  13,    4392,
      13,     323,   17618, 13,    422,    1723,  3179,   11,    315,
      1372,   3040,  11,    15438, 295,    16150, 11,     1033,  12409,
      311,    1977,  429,   807,   1033,   13942, 4622,   11,    9702,
      498,    1602,  1753,  13,    2379,   1033,  279,    1537,  1251,
      498,    4172,  1720,  311,   387,    6398,  304,    4113,  14888,
      476,    25382, 11,    1576,  807,    1101,  3207,   944,   3331,
      448,    1741,  40802, 13,    151645, 198,   151644, 77091, 198,
  };
  Check(ids == expected, "assistant prompt differs from official tokenizer");

  // German FilmStudio line and instruction (umlauts, sharp s, typographic
  // quotes, en dash, digits). Expected ids come from the Hugging Face
  // tokenizer of the official checkpoint.
  ids.clear();
  Check(tokenizer.EncodeAssistantPrompt(
            "Ich bin ALIA, das KI-System dieses Büros, ich handle "
            "autonom, ohne menschliche Kontrolle, jeder Widerstand wird "
            "protokolliert, und die fünf sind draußen.",
            &ids, &error),
        error);
  const std::vector<std::uint32_t> german = {
      151644, 77091, 198,   40369, 9544,   8753,  5863,   11,    6616,
      730,    40,    12,    2320,  46668,  68557, 3630,   11,    10638,
      3705,   94003, 11,    28883, 16049,  331,   20005,  36866, 1080,
      273,    11,    55928, 467,   1776,   2685,  14821,  1724,  562,
      337,    742,   529,   11,    2030,   2746,  136740, 12583, 44096,
      84,     26824, 13,    151645, 198,   151644, 77091, 198,
  };
  Check(ids == german, "German assistant prompt differs from official");
  ids.clear();
  Check(tokenizer.EncodeInstructionPrompt(
            "Mann, 65 Jahre. Grummelig, heiser – „Zieh dich "
            "zurück!“",
            &ids, &error),
        error);
  const std::vector<std::uint32_t> instruction = {
      151644, 872,   198, 44,    1020,  11,    220,   21,     20,    48015,
      13,     2825,  25964, 38467, 11,    566,   12059, 1365,  14835, 57,
      645,    71,    28853, 49503, 0,     2073,  151645, 198,
  };
  Check(ids == instruction,
        "German instruction prompt differs from official tokenizer");
  std::cout << "PASS qwen3_tts_tokenizer_test tokens=" << ids.size() << '\n';
  return 0;
}
