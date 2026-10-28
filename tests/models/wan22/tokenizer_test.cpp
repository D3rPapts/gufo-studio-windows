#include "src/models/wan22/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace wan22 = gufo::models::wan22;

namespace {

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL wan22_tokenizer_test: " << message << '\n';
    ++failures;
  }
}

void ExpectIds(const wan22::Tokenizer& tokenizer, std::string_view text,
               const std::vector<std::int32_t>& expected) {
  const auto ids = tokenizer.Encode(wan22::CleanPrompt(text));
  if (ids == expected)
    return;
  std::cerr << "got:";
  for (const auto id : ids)
    std::cerr << ' ' << id;
  std::cerr << '\n';
  Expect(false, std::string(text));
}

}  // namespace

int main() {
  Expect(wan22::CleanPrompt("  a \t\n b\xC2\xA0 c  ") == "a b c",
         "whitespace clean");
  Expect(wan22::CleanPrompt("") == "", "empty clean");

  // WAN22_MODEL_ROOT names the Diffusers checkpoint directory.
  const char* configured = std::getenv("WAN22_MODEL_ROOT");
  const std::filesystem::path root =
      configured != nullptr
          ? configured
          : "D:/llm-modelle/Wan-AI/Wan2.2-TI2V-5B-Diffusers";
  const auto path = root / "tokenizer" / "tokenizer.json";
  if (!std::filesystem::exists(path)) {
    std::cerr << "SKIP wan22_tokenizer_test: no " << path.string() << '\n';
    return 77;
  }
  std::string error;
  const auto tokenizer = wan22::Tokenizer::Load(path, &error);
  if (!tokenizer) {
    std::cerr << "FAIL wan22_tokenizer_test: " << error << '\n';
    return 1;
  }
  Expect(tokenizer->vocab_size() == 256300, "vocabulary size");

  // Ids of the Diffusers WanPipeline (max_length 512, real tokens only).
  ExpectIds(*tokenizer, "A red fox walking through fresh snow at golden hour",
            {320, 4062, 273, 56209, 53049, 3311, 22961, 45540, 367, 54091,
             27239, 1});
  ExpectIds(*tokenizer,
            "Ein alter Leuchtturm bei Nacht, Wellen schlagen gegen die Felsen",
            {2216, 33581, 85091, 83610, 1008, 18852, 275, 162143, 186938, 5594,
             366, 554, 21111, 1});
  // Cleaning collapses the spaces before tokenizing.
  ExpectIds(*tokenizer,
            "  A red fox walking   through fresh snow at golden hour\n",
            {320, 4062, 273, 56209, 53049, 3311, 22961, 45540, 367, 54091,
             27239, 1});
  Expect(tokenizer->Encode("").size() == 1, "empty prompt is </s>");
  Expect(tokenizer->Encode("</s>") == std::vector<std::int32_t>{1, 1},
         "added token");

  // Truncation keeps max_tokens - 1 content tokens and the final </s>.
  std::string long_text;
  for (int i = 0; i < 400; ++i)
    long_text += "red fox ";
  const auto truncated = tokenizer->Encode(long_text, 512);
  Expect(truncated.size() == 512 && truncated.back() == 1, "truncation");

  if (failures == 0)
    std::cout << "wan22_tokenizer_test passed\n";
  return failures == 0 ? 0 : 1;
}
