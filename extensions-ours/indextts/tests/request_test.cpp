// IndexTTS-2.5 extension API 0.2 request mapping (gufo::ext::SynthesisRequest
// -> engine request): emotion in `instructions` (key=value, JSON, 1.x string
// forms), language, sampling defaults, reference limits, mixdown, the int16
// round trip and, with GUFO_INDEXTTS_MODEL, the bundle probe (file checks
// only, nothing is loaded). CPU only. Written with AI assistance (Claude Code).
#include <gufo/ext/tts.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "indextts/bundle.hpp"
#include "indextts/ext_request.hpp"

namespace itts = gufo::models::indextts;

namespace {

int g_failures = 0;

void Expect(const std::string& name, bool ok, const std::string& detail = {}) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-52s %s\n", ok ? "OK" : "FAIL", name.c_str(), ok ? "" : detail.c_str());
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-5F; }

bool VectorIs(const std::optional<std::array<float, 8>>& v, const std::array<float, 8>& want) {
  if (!v.has_value()) {
    return false;
  }
  for (std::size_t i = 0; i < 8; ++i) {
    if (!Near((*v)[i], want[i])) {
      return false;
    }
  }
  return true;
}

// Parses `text` and checks the emotion vector and emo_alpha.
void ExpectControls(const std::string& name, const std::string& text,
                    const std::array<float, 8>& want, float alpha = 1.0F) {
  itts::InstructControls c;
  std::string error;
  const bool ok = itts::ParseInstructControls(text, &c, &error);
  Expect(name, ok && VectorIs(c.emotion, want) && Near(c.emo_alpha, alpha),
         ok ? "unexpected vector/alpha" : error);
}

void ExpectReject(const std::string& name, const std::string& text, const std::string& needle) {
  itts::InstructControls c;
  std::string error;
  const bool ok = itts::ParseInstructControls(text, &c, &error);
  Expect(name, !ok && error.find(needle) != std::string::npos,
         ok ? "accepted" : "error: " + error);
}

gufo::ext::SynthesisRequest Clone(std::uint32_t rate, std::uint32_t channels, double seconds) {
  gufo::ext::SynthesisRequest r;
  r.text = "Ich kann nicht mehr. Es tut so weh.";
  r.speaker = "voice-clone";
  r.reference_audio.sample_rate = rate;
  r.reference_audio.channels = channels;
  r.reference_audio.samples.assign(
      static_cast<std::size_t>(seconds * rate) * channels, 0.0F);
  return r;
}

}  // namespace

int main() {
  // --- emotion in instructions -------------------------------------------
  {
    itts::InstructControls c;
    std::string error;
    Expect("empty instructions: no emotion",
           itts::ParseInstructControls("  ", &c, &error) && !c.emotion.has_value() &&
               Near(c.emo_alpha, 1.0F),
           error);
  }
  const std::array<float, 8> sad_mel{0, 0, 0.7F, 0, 0, 0.3F, 0, 0};
  ExpectControls("emotion=[8 numbers]", "emotion=[0,0,0.7,0,0,0.3,0,0]", sad_mel);
  ExpectControls("emotion = [ ... ] ; emo_alpha", " emotion = [0, 0, 0.7, 0, 0, 0.3, 0, 0] ; emo_alpha=0.8",
                 sad_mel, 0.8F);
  ExpectControls("emo_alpha first, then emotion vector", "emo_alpha=0.5, emotion=[0,0,0.7,0,0,0.3,0,0]",
                 sad_mel, 0.5F);
  ExpectControls("emotion=names with weights, emo_alpha",
                 "emotion=Wut:0.6, Angst:0.2; emo_alpha=0.8", {0, 0.6F, 0, 0.2F, 0, 0, 0, 0}, 0.8F);
  ExpectControls("emotion=name, alpha=", "emotion=traurig, alpha=0.25", {0, 0, 1, 0, 0, 0, 0, 0}, 0.25F);
  ExpectControls("emotion=8 numbers without brackets", "emotion=0,0,0.7,0,0,0.3,0,0", sad_mel);
  ExpectControls("JSON {emotion: vector}", R"({"emotion":[0,0,0.7,0,0,0.3,0,0]})", sad_mel);
  ExpectControls("JSON {emotion: name, emo_alpha}", R"({"emotion":"traurig","emo_alpha":0.8})",
                 {0, 0, 1, 0, 0, 0, 0, 0}, 0.8F);
  ExpectControls("JSON {emotion: {name: weight}}", R"({"emotion":{"Wut":0.6,"Angst":0.2}})",
                 {0, 0.6F, 0, 0.2F, 0, 0, 0, 0});
  ExpectControls("JSON {name: weight} (1.x object form)", R"({"Wut":0.6,"Angst":0.2})",
                 {0, 0.6F, 0, 0.2F, 0, 0, 0, 0});
  ExpectControls("JSON array of 8", "[0,0,0.7,0,0,0.3,0,0]", sad_mel);
  ExpectControls("1.x string: one German name", "Wut", {0, 1, 0, 0, 0, 0, 0, 0});
  ExpectControls("1.x string: umlaut capital", "\xC3\x9C" "berraschung", {0, 0, 0, 0, 0, 0, 1, 0});
  ExpectControls("1.x string: name:weight list", "sad:0.6, calm:0.2", {0, 0, 0.6F, 0, 0, 0, 0, 0.2F});
  ExpectControls("1.x string: 8 numbers", "0,0,0.6,0,0,0.2,0,0", {0, 0, 0.6F, 0, 0, 0.2F, 0, 0});

  ExpectReject("free text rejected", "sprich bitte ganz traurig", "unknown emotion");
  ExpectReject("vector of 3 rejected", "emotion=[0,0,1]", "unknown emotion");
  ExpectReject("JSON vector of 7 rejected", R"({"emotion":[0,0,0.7,0,0,0.3,0]})", "8 numbers");
  ExpectReject("value above 1.2 rejected", "emotion=[0,0,1.5,0,0,0,0,0]", "between 0 and 1.2");
  ExpectReject("unknown key rejected", "tempo=slow", "unknown key 'tempo'");
  ExpectReject("unknown JSON key rejected", R"({"pitch":"low"})", "unknown key 'pitch'");
  ExpectReject("emo_alpha above 1 rejected", "emotion=sad; emo_alpha=2", "emo_alpha");
  ExpectReject("emotion twice rejected", "emotion=sad; emotion=calm", "twice");
  ExpectReject("emotion plus name keys rejected", R"({"emotion":"sad","Wut":0.5})", "either");
  ExpectReject("invalid JSON rejected", R"({"emotion":)", "invalid JSON");
  ExpectReject("missing ] rejected", "emotion=[0,0,1", "missing ']'");

  // --- language -----------------------------------------------------------
  {
    std::string lang;
    std::string error;
    Expect("language: core default 'english' -> de",
           itts::ResolveLanguage("english", &lang, &error) && lang == "de", error);
    Expect("language: 'Deutsch' -> de", itts::ResolveLanguage("Deutsch", &lang, &error) && lang == "de",
           error);
    Expect("language: empty -> de", itts::ResolveLanguage("", &lang, &error) && lang == "de", error);
    Expect("language: 'fr' rejected",
           !itts::ResolveLanguage("fr", &lang, &error) && error.find("'de' only") != std::string::npos,
           error);
  }

  // --- sampling defaults --------------------------------------------------
  {
    const gufo::ext::TtsSamplingOptions core;
    const itts::SamplingOptions s = itts::ResolveSampling(core, 3000);
    Expect("sampling: core defaults -> indextts_de.py defaults",
           Near(s.temperature, 0.7F) && s.top_k == 30 && Near(s.top_p, 0.8F) &&
               Near(s.repetition_penalty, 10.0F) && s.max_tokens == 1500);
    gufo::ext::TtsSamplingOptions custom;
    custom.temperature = 0.5F;
    custom.top_k = 10;
    const itts::SamplingOptions c = itts::ResolveSampling(custom, 200);
    Expect("sampling: client values kept, rest defaulted",
           Near(c.temperature, 0.5F) && c.top_k == 10 && Near(c.top_p, 0.8F) &&
               Near(c.repetition_penalty, 10.0F) && c.max_tokens == 200);
  }

  // --- engine request -----------------------------------------------------
  {
    std::string error;
    itts::SynthesisRequest out;
    gufo::ext::SynthesisRequest none;
    none.text = "Hallo.";
    none.speaker = "voice-clone";
    Expect("no reference rejected",
           !itts::BuildEngineRequest(none, &out, &error) &&
               error.find("needs a reference voice") != std::string::npos,
           error);

    gufo::ext::SynthesisRequest stereo = Clone(16000, 2, 0.0);
    stereo.reference_audio.samples = {0.5F, -0.5F, 1.0F, 0.0F};
    stereo.instruct = "emotion=[0,0,0.7,0,0,0.3,0,0]; emo_alpha=0.9";
    stereo.reference_text = "ignored";
    stereo.sampling.seed = 7;
    stereo.max_new_tokens = 8192;
    Expect("stereo mixdown, emotion, seed, caps",
           itts::BuildEngineRequest(stereo, &out, &error) && out.reference.size() == 2 &&
               out.reference[0] == 0.0F && out.reference[1] == 0.5F &&
               out.reference_rate == 16000 && out.language == "de" &&
               VectorIs(out.emotion, sad_mel) && Near(out.emo_alpha, 0.9F) && out.seed == 7 &&
               out.sampling.max_tokens == 1500 && out.text == stereo.text,
           error);

    stereo.instruct = "sprich traurig";
    Expect("bad instructions fail the request",
           !itts::BuildEngineRequest(stereo, &out, &error) &&
               error.find("IndexTTS instructions") != std::string::npos,
           error);

    gufo::ext::SynthesisRequest ten = Clone(24000, 1, 10.0);
    Expect("inline reference of 10 s accepted", itts::BuildEngineRequest(ten, &out, &error), error);
    gufo::ext::SynthesisRequest eleven = Clone(24000, 1, 10.5);
    Expect("inline reference above 10 s rejected",
           !itts::BuildEngineRequest(eleven, &out, &error) &&
               error.find("10 seconds") != std::string::npos,
           error);
    eleven.speaker = "erzaehler";  // a --voice preset: the engine cuts at 15 s
    Expect("preset reference above 10 s accepted", itts::BuildEngineRequest(eleven, &out, &error),
           error);

    gufo::ext::SynthesisRequest french = Clone(16000, 1, 1.0);
    french.language = "fr";
    Expect("language fr fails the request", !itts::BuildEngineRequest(french, &out, &error), error);
  }

  // --- int16 round trip through the core's truncating PCM16 encoder -------
  {
    bool exact = true;
    int first_bad = 0;
    for (int s = -32767; s <= 32767; ++s) {
      const float f = itts::Int16ToFloat(static_cast<std::int16_t>(s));
      const float bounded = f < -1.0F ? -1.0F : (f > 1.0F ? 1.0F : f);
      if (static_cast<std::int16_t>(bounded * 32767.0F) != s) {
        exact = false;
        first_bad = s;
        break;
      }
    }
    Expect("int16 -> float -> PCM16 is exact", exact, "first mismatch at " + std::to_string(first_bad));
  }

  // --- bundle probe (file checks only) ------------------------------------
  {
    Expect("probe rejects an empty directory",
           !itts::LooksLikeIndexTts25(std::filesystem::temp_directory_path() / "gufo-no-indextts"));
    if (const char* model = std::getenv("GUFO_INDEXTTS_MODEL"); model != nullptr && *model != '\0') {
      Expect("probe accepts GUFO_INDEXTTS_MODEL", itts::LooksLikeIndexTts25(model), model);
      Expect("German GPT path",
             itts::GptPath(model) == std::filesystem::path(model) / "german" / "gpt.pth");
    } else {
      std::printf("SKIP probe of a real bundle (GUFO_INDEXTTS_MODEL not set)\n");
    }
  }

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
