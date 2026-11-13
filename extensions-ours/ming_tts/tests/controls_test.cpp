// Ming-3.0-tts BASIC (tempo/pitch/volume) and DIALECT controls: value
// mapping, captions against the reference tts_tasks.build_task_instruction
// output, task resolution, and (extension API 0.2) the controls carried in
// `instructions` plus the gufo::ext::SynthesisRequest -> engine request
// mapping. CPU only, no checkpoint.
// Written with AI assistance (Claude Code).
#include <gufo/ext/tts.hpp>

#include <cstdio>
#include <optional>
#include <string>

#include "ming_tts/ext_request.hpp"
#include "ming_tts/prompt.hpp"
#include "ming_tts/synthesis.hpp"

namespace ming = gufo::models::ming_tts;

namespace {

int g_failures = 0;

void Expect(const std::string& name, bool ok, const std::string& detail = {}) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-44s %s\n", ok ? "OK" : "FAIL", name.c_str(),
              ok ? "" : detail.c_str());
}

// python -X utf8: tts_tasks.build_task_instruction("BASIC", speed="慢速", pitch="低")
constexpr const char* kBasicSlowLow =
    "{\"audio_sequence\": [{\"序号\": 1, \"说话人\": \"speaker_1\", \"方言\": null, "
    "\"风格\": null, \"语速\": \"慢速\", \"基频\": \"低\", \"音量\": null, \"情感\": null, "
    "\"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, "
    "\"ENV\": null, \"SNR\": null}, \"IP\": null}]}";
// tts_tasks.build_task_instruction("DIALECT", dialect="cantonese")
constexpr const char* kDialectCantonese =
    "{\"audio_sequence\": [{\"id\": 1, \"speaker\": \"speaker_1\", \"dialect\": "
    "\"cantonese\", \"style\": null, \"speed\": null, \"pitch\": null, \"volume\": null, "
    "\"emotion\": null, \"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, "
    "\"Theme\": null, \"ENV\": null, \"SNR\": null}, \"ip\": null}]}";

void TestLabels() {
  Expect("tempo slow", ming::SpeedLabel("slow").value_or("") == "慢速");
  Expect("tempo Schnell", ming::SpeedLabel(" Schnell ").value_or("") == "快速");
  Expect("tempo normal", ming::SpeedLabel("normal").value_or("") == "中速");
  Expect("tempo Chinese passthrough", ming::SpeedLabel("中速").value_or("") == "中速");
  Expect("tempo unknown", !ming::SpeedLabel("1.5").has_value());
  Expect("pitch hoch", ming::PitchLabel("hoch").value_or("") == "高");
  Expect("pitch tief", ming::PitchLabel("tief").value_or("") == "低");
  Expect("pitch loud rejected", !ming::PitchLabel("loud").has_value());
  Expect("volume leise", ming::VolumeLabel("leise").value_or("") == "低");
  Expect("volume LOUD", ming::VolumeLabel("LOUD").value_or("") == "高");
  Expect("dialect chuanyu", ming::DialectLabel("chuanyu").value_or("") == "chuanyu dialect");
  Expect("dialect Kantonesisch", ming::DialectLabel("Kantonesisch").value_or("") == "cantonese");
  Expect("dialect bairisch rejected", !ming::DialectLabel("bairisch").has_value());
}

void TestCaptions() {
  ming::StructuredControls basic;
  basic.speed = "慢速";
  basic.pitch = "低";
  Expect("BASIC caption == reference", ming::BuildStructuredInstruction(basic) == kBasicSlowLow,
         ming::BuildStructuredInstruction(basic));
  ming::StructuredControls dialect;
  dialect.dialect = "cantonese";
  const std::string en =
      ming::BuildStructuredInstruction(dialect, ming::InstructionSchema::kEnglish);
  Expect("DIALECT caption == reference (en schema)", en == kDialectCantonese, en);
}

void TestResolve() {
  std::optional<std::string> instruction;
  std::string error;
  ming::SynthesisRequest request;
  Expect("no controls -> no caption",
         ming::ResolveTaskInstruction(request, &instruction, &error) &&
             !instruction.has_value());
  request.speed = "slow";
  request.pitch = "low";
  Expect("BASIC resolve",
         ming::ResolveTaskInstruction(request, &instruction, &error) &&
             instruction.value_or("") == kBasicSlowLow,
         error);
  request = {};
  request.dialect = "cantonese";
  Expect("DIALECT resolve",
         ming::ResolveTaskInstruction(request, &instruction, &error) &&
             instruction.value_or("") == kDialectCantonese,
         error);
  request.volume = "high";
  Expect("BASIC + DIALECT rejected",
         !ming::ResolveTaskInstruction(request, &instruction, &error));
  request = {};
  request.speed = "fast";
  request.emotion = "angry";
  Expect("BASIC + emotion rejected",
         !ming::ResolveTaskInstruction(request, &instruction, &error));
  request = {};
  request.volume = "whisper";
  Expect("unknown volume rejected",
         !ming::ResolveTaskInstruction(request, &instruction, &error) &&
             error.find("volume") != std::string::npos,
         error);
}

std::string Join(const ming::InstructControls& c) {
  return "emotion=" + c.emotion + "|instruction=" + c.instruction + "|speed=" + c.speed +
         "|pitch=" + c.pitch + "|volume=" + c.volume + "|dialect=" + c.dialect;
}

// API 0.2 carries the controls in `instructions` (SynthesisRequest::instruct).
void TestInstructParsing() {
  ming::InstructControls c;
  std::string error;
  Expect("empty instructions", ming::ParseInstructControls("  ", &c, &error) &&
                                   Join(c) == Join(ming::InstructControls{}));
  Expect("JSON BASIC",
         ming::ParseInstructControls(R"({"tempo":"langsam","pitch":"low","volume":"laut"})",
                                     &c, &error) &&
             c.speed == "langsam" && c.pitch == "low" && c.volume == "laut" &&
             c.dialect.empty() && c.emotion.empty() && c.instruction.empty(),
         Join(c) + " " + error);
  Expect("JSON DIALECT", ming::ParseInstructControls(R"( {"dialect":"cantonese"} )", &c, &error) &&
                             c.dialect == "cantonese",
         Join(c) + " " + error);
  Expect("JSON speed alias + caption",
         ming::ParseInstructControls(R"({"speed":"fast","instruction":"leise"})", &c, &error) &&
             c.speed == "fast" && c.instruction == "leise",
         Join(c) + " " + error);
  Expect("JSON unknown key rejected",
         !ming::ParseInstructControls(R"({"tempo":"fast","speed2":"x"})", &c, &error) &&
             error.find("speed2") != std::string::npos,
         error);
  Expect("JSON non-string rejected",
         !ming::ParseInstructControls(R"({"tempo":1.5})", &c, &error), error);
  Expect("JSON malformed rejected", !ming::ParseInstructControls(R"({"tempo":)", &c, &error),
         error);
  Expect("key=value list",
         ming::ParseInstructControls("Tempo = slow; pitch=low", &c, &error) &&
             c.speed == "slow" && c.pitch == "low",
         Join(c) + " " + error);
  Expect("key=value dialect with space",
         ming::ParseInstructControls("dialect=chuanyu dialect", &c, &error) &&
             c.dialect == "chuanyu dialect",
         Join(c) + " " + error);
  Expect("list mixed with free text rejected",
         !ming::ParseInstructControls("tempo=slow, whisper please", &c, &error), error);
  Expect("emotion name", ming::ParseInstructControls(" Wut ", &c, &error) &&
                             c.emotion == "Wut" && c.instruction.empty(),
         Join(c));
  const std::string caption =
      "Whisper very softly and tensely, almost breathless, close to the microphone.";
  Expect("free-text caption", ming::ParseInstructControls(caption, &c, &error) &&
                                  c.instruction == caption && c.emotion.empty(),
         Join(c));
  Expect("free text with '=' but no known key",
         ming::ParseInstructControls("say 2+2=4 slowly", &c, &error) &&
             c.instruction == "say 2+2=4 slowly",
         Join(c));
}

void TestEngineRequest() {
  std::string error;
  gufo::ext::SynthesisRequest in;
  in.text = "Hallo.";
  in.language = "de";
  in.instruct = R"({"tempo":"langsam","pitch":"low","volume":"laut"})";
  in.max_new_tokens = 3000;
  in.sampling.seed = 7;
  ming::SynthesisRequest out;
  Expect("BASIC request", ming::BuildEngineRequest(in, &out, &error) &&
                              out.speed == "langsam" && out.pitch == "low" &&
                              out.volume == "laut" && out.language == "de" &&
                              out.seed == 7 && out.max_decode_steps == 1000,
         error);
  std::optional<std::string> instruction;
  Expect("BASIC request caption",
         ming::ResolveTaskInstruction(out, &instruction, &error) && instruction.has_value());

  const auto expect_error = [&](const std::string& name, const std::string& instruct,
                                const std::string& needle) {
    gufo::ext::SynthesisRequest bad = in;
    bad.instruct = instruct;
    std::string message;
    ming::SynthesisRequest ignored;
    Expect(name, !ming::BuildEngineRequest(bad, &ignored, &message) &&
                     message.find(needle) != std::string::npos,
           message);
  };
  expect_error("bad tempo", R"({"tempo":"1.5"})", "speed");
  expect_error("bad pitch", R"({"pitch":"loud"})", "pitch");
  expect_error("bad volume", R"({"volume":"x"})", "volume");
  expect_error("bad dialect", R"({"dialect":"bairisch"})", "dialect");
  expect_error("tempo + emotion", R"({"tempo":"fast","emotion":"angry"})", "one task");
  expect_error("pitch + caption", R"({"pitch":"high","instruction":"whisper"})", "one task");
  expect_error("volume + dialect", R"({"volume":"high","dialect":"cantonese"})", "one task");
  expect_error("bad emotion (JSON)", R"({"emotion":"langweilig"})", "emotion");

  // Reference audio: stereo is mixed down; the transcript is dropped for
  // speaker_embedding_only.
  gufo::ext::SynthesisRequest clone;
  clone.text = "Hallo.";
  clone.reference_audio = {.sample_rate = 16000, .channels = 2,
                           .samples = {0.5F, -0.5F, 1.0F, 0.0F}};
  clone.reference_text = "Transkript.";
  Expect("clone mixdown",
         ming::BuildEngineRequest(clone, &out, &error) && out.reference_audio.size() == 2 &&
             out.reference_audio[0] == 0.0F && out.reference_audio[1] == 0.5F &&
             out.reference_sample_rate == 16000 && out.reference_text == "Transkript.",
         error);
  clone.speaker_embedding_only = true;
  Expect("speaker_embedding_only drops transcript",
         ming::BuildEngineRequest(clone, &out, &error) && out.reference_text.empty(), error);
  clone.max_new_tokens = 200;
  Expect("max_new_tokens below the cap",
         ming::BuildEngineRequest(clone, &out, &error) && out.max_decode_steps == 200, error);
}

}  // namespace

int main() {
  TestLabels();
  TestCaptions();
  TestResolve();
  TestInstructParsing();
  TestEngineRequest();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
