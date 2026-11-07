// Ming-3.0-tts BASIC (tempo/pitch/volume) and DIALECT controls: value
// mapping, captions against the reference tts_tasks.build_task_instruction
// output, task resolution and the /v1/audio/speech fields. CPU only, no
// checkpoint: the API part uses a stub model directory and a fake runner.
// Written with AI assistance (Claude Code).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>

#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/models/ming_tts/prompt.hpp"
#include "src/models/ming_tts/synthesis.hpp"

namespace ming = gufo::models::ming_tts;
namespace qwen3_tts = gufo::models::qwen3_tts;
using gufo::server::HandleAudioTtsApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::TtsService;
using gufo::server::TtsServiceOptions;

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

// Smallest config.json that LoadModelConfigFromPath accepts.
constexpr const char* kStubConfig = R"({
  "architectures": ["BailingMMNativeForConditionalGeneration"],
  "model_type": "dense",
  "llm_config": {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                 "num_attention_heads": 2, "num_key_value_heads": 1,
                 "rms_norm_eps": 1e-6, "rope_theta": 10000},
  "ditar_config": {"hidden_size": 8, "depth": 1, "num_heads": 2, "mlp_ratio": 2},
  "aggregator_config": {"hidden_size": 8, "depth": 1, "num_heads": 2, "mlp_ratio": 2},
  "audio_tokenizer_config": {
    "patch_size": 4, "sample_rate": 44100,
    "dec_kwargs": {"latent_dim": 4, "output_dim": 16,
                   "backbone": {"hidden_size": 8, "intermediate_size": 16,
                                "num_hidden_layers": 1, "num_attention_heads": 2,
                                "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
                                "rope_theta": 10000}},
    "enc_kwargs": {"input_dim": 16,
                   "backbone": {"hidden_size": 8, "intermediate_size": 16,
                                "num_hidden_layers": 1, "num_attention_heads": 2,
                                "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
                                "rope_theta": 10000}}
  }
})";

struct FakeRunner {
  int calls{0};
  qwen3_tts::SynthesisRequest latest;
};

HttpResponse Post(TtsService& service, const std::string& body) {
  return HandleAudioTtsApiRequest(HttpRequest{.method = "POST",
                                              .path = "/v1/audio/speech",
                                              .query = {},
                                              .body = body,
                                              .headers = {},
                                              .is_cancelled = {}},
                                  service);
}

std::string ErrorCode(const HttpResponse& response) {
  const auto at = response.body.find("\"code\"");
  if (at == std::string::npos) {
    return response.body;
  }
  const auto open = response.body.find('"', response.body.find(':', at) + 1);
  const auto close = response.body.find('"', open + 1);
  return response.body.substr(open + 1, close - open - 1);
}

TtsServiceOptions FakeOptions(const std::filesystem::path& root, FakeRunner* runner) {
  return TtsServiceOptions{
      .model_root = root,
      .native_context_tokens = 4096,
      .validate_model = false,
      .model_id = {},
      .voices = {},
      .voice_presets = {},
      .runner = [runner](const qwen3_tts::SynthesisRequest& request,
                         const qwen3_tts::CancellationCheck&,
                         qwen3_tts::SynthesisResult* result, std::string*) {
        ++runner->calls;
        runner->latest = request;
        result->sample_rate = 44100;
        result->code_groups = 1;
        result->samples = {0.0F, 0.25F, -0.25F, 0.0F};
        result->sample_count = result->samples.size();
        return true;
      },
  };
}

void TestApi() {
  std::error_code ec;
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "gufo_ming_controls_test";
  std::filesystem::create_directories(root, ec);
  std::ofstream(root / "config.json", std::ios::binary) << kStubConfig;
  std::ofstream(root / "model.safetensors", std::ios::binary) << "";
  std::ofstream(root / "campplus.onnx", std::ios::binary) << "";

  FakeRunner runner;
  TtsService service(FakeOptions(root, &runner));
  Expect("stub detected as Ming", service.ready() && service.is_ming_tts(),
         service.initialization_error());
  if (!service.ready()) {
    std::filesystem::remove_all(root, ec);
    return;
  }

  HttpResponse ok = Post(service,
                         R"({"model":"ming-3.0-tts","input":"Hallo.","tempo":"langsam",)"
                         R"("pitch":"low","volume":"laut"})");
  Expect("BASIC request 200", ok.status == 200 && runner.calls == 1, ok.body);
  Expect("BASIC fields forwarded", runner.latest.tempo == "langsam" &&
                                       runner.latest.pitch == "low" &&
                                       runner.latest.volume == "laut" &&
                                       runner.latest.dialect.empty());

  ok = Post(service, R"({"model":"ming-3.0-tts","input":"你好。","dialect":"cantonese"})");
  Expect("DIALECT request 200", ok.status == 200 && runner.latest.dialect == "cantonese",
         ok.body);

  ok = Post(service, R"({"model":"ming-3.0-tts","input":"Hallo.","tempo":"fast","speed":1.2})");
  Expect("tempo + OpenAI speed allowed", ok.status == 200, ok.body);

  const auto expect_error = [&](const std::string& name, const std::string& body,
                                const std::string& code) {
    const int before = runner.calls;
    const HttpResponse response = Post(service, body);
    Expect(name, response.status == 400 && ErrorCode(response) == code &&
                     runner.calls == before,
           std::to_string(response.status) + " " + response.body);
  };
  expect_error("bad tempo", R"({"model":"ming-3.0-tts","input":"Hallo.","tempo":"1.5"})",
               "invalid_tempo");
  expect_error("bad pitch", R"({"model":"ming-3.0-tts","input":"Hallo.","pitch":"loud"})",
               "invalid_pitch");
  expect_error("bad volume", R"({"model":"ming-3.0-tts","input":"Hallo.","volume":"x"})",
               "invalid_volume");
  expect_error("bad dialect", R"({"model":"ming-3.0-tts","input":"Hallo.","dialect":"bairisch"})",
               "invalid_dialect");
  expect_error("tempo not a string", R"({"model":"ming-3.0-tts","input":"Hallo.","tempo":1})",
               "invalid_parameter_type");
  expect_error("tempo + emotion",
               R"({"model":"ming-3.0-tts","input":"Hallo.","tempo":"fast","emotion":"angry"})",
               "conflicting_controls");
  expect_error("pitch + instructions",
               R"({"model":"ming-3.0-tts","input":"Hallo.","pitch":"high","instructions":"whisper"})",
               "conflicting_controls");
  expect_error("volume + dialect",
               R"({"model":"ming-3.0-tts","input":"Hallo.","volume":"high","dialect":"cantonese"})",
               "conflicting_controls");

  // Non-Ming services reject the fields.
  FakeRunner other_runner;
  TtsServiceOptions other = FakeOptions({}, &other_runner);
  other.model_id = "qwen3-tts-12hz-1.7b-customvoice";
  other.voices = {"vivian"};
  TtsService qwen(std::move(other));
  const HttpResponse rejected =
      Post(qwen, R"({"model":"qwen3-tts","voice":"vivian","input":"Hi.","pitch":"high"})");
  Expect("Qwen3-TTS rejects pitch",
         rejected.status == 400 && ErrorCode(rejected) == "unsupported_field" &&
             other_runner.calls == 0,
         rejected.body);
  std::filesystem::remove_all(root, ec);
}

}  // namespace

int main() {
  TestLabels();
  TestCaptions();
  TestResolve();
  TestApi();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
