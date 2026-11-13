// VoxCPM2 extension API 0.2 request mapping (gufo::ext::SynthesisRequest ->
// engine request): instructions, reference vs. "ultimate" cloning, the
// 60-second reference limit, mixdown and caps. CPU only, no checkpoint.
// Written with AI assistance (Claude Code).
#include <gufo/ext/tts.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "voxcpm2/ext_request.hpp"

namespace vox = gufo::models::voxcpm2;

namespace {

int g_failures = 0;

void Expect(const std::string& name, bool ok, const std::string& detail = {}) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-44s %s\n", ok ? "OK" : "FAIL", name.c_str(), ok ? "" : detail.c_str());
}

}  // namespace

int main() {
  std::string error;
  vox::SynthesisRequest out;

  gufo::ext::SynthesisRequest zero;
  zero.text = "Hallo, ich bin eine neue Stimme.";
  zero.instruct = "fröhlich und begeistert";
  zero.sampling.seed = 9;
  Expect("zero-shot", vox::BuildEngineRequest(zero, &out, &error) &&
                          out.text == zero.text && out.instruction == zero.instruct &&
                          out.reference_audio.empty() && out.prompt_audio.empty() &&
                          out.seed == 9 && out.max_len == 3000,
         error);

  gufo::ext::SynthesisRequest clone;
  clone.text = "Psst.";
  clone.reference_audio = {.sample_rate = 16000, .channels = 2,
                           .samples = {0.5F, -0.5F, 1.0F, 0.0F}};
  clone.max_new_tokens = 8192;
  Expect("reference mode (no transcript)",
         vox::BuildEngineRequest(clone, &out, &error) && out.reference_audio.size() == 2 &&
             out.reference_audio[0] == 0.0F && out.reference_audio[1] == 0.5F &&
             out.reference_sample_rate == 16000 && out.prompt_audio.empty() &&
             out.prompt_text.empty() && out.max_len == 4096,
         error);

  clone.reference_text = "Transkript.";
  Expect("ultimate cloning (transcript)",
         vox::BuildEngineRequest(clone, &out, &error) &&
             out.prompt_audio == out.reference_audio && out.prompt_sample_rate == 16000 &&
             out.prompt_text == "Transkript.",
         error);

  clone.speaker_embedding_only = true;
  Expect("speaker_embedding_only keeps reference mode",
         vox::BuildEngineRequest(clone, &out, &error) && out.prompt_audio.empty() &&
             out.prompt_text.empty() && !out.reference_audio.empty(),
         error);

  gufo::ext::SynthesisRequest too_long;
  too_long.text = "Hallo.";
  too_long.reference_audio = {.sample_rate = 16000, .channels = 1,
                              .samples = std::vector<float>(16000U * 61U, 0.0F)};
  Expect("reference longer than 60 s rejected",
         !vox::BuildEngineRequest(too_long, &out, &error) &&
             error.find("60 seconds") != std::string::npos,
         error);
  too_long.reference_audio.samples.resize(16000U * 60U);
  Expect("reference of 60 s accepted", vox::BuildEngineRequest(too_long, &out, &error), error);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
