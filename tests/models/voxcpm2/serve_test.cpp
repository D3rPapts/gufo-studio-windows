// /v1/audio/speech with a real VoxCPM2 checkpoint through TtsService and the
// audio API handler (in process, no sockets): detection, voice preset and
// inline reference cloning, emotion / free-text instructions, language field,
// zero-shot, PCM streaming and request validation (400s). CPU runtime unless
// a HIP build finds a GPU.
//
//   GUFO_VOXCPM_MODEL       checkpoint directory
//   GUFO_VOXCPM_VOICE       reference WAV for the preset / inline clone (<= 10 s)
//   GUFO_VOXCPM_OUT         optional directory for the produced WAV/PCM files
//   GUFO_VOXCPM_VOICE_TEXT  optional UTF-8 transcript file: adds an
//                           "ultimate cloning" request (reference + transcript)
//
// Exits 77 (skip) when the model or voice is not configured.
// Written with AI assistance (Claude Code).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>

#include "src/cli/serve/audio_stream.hpp"
#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen3_tts/audio.hpp"

using gufo::server::HandleAudioTtsApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::TtsService;

namespace {

int g_failures = 0;

void Expect(const std::string& name, bool ok, const std::string& detail = {}) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-40s %s\n", ok ? "OK" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

HttpResponse Send(TtsService& service, const std::string& method, const std::string& path,
                  std::string body = {}) {
  return HandleAudioTtsApiRequest(HttpRequest{.method = method,
                                              .path = path,
                                              .query = {},
                                              .body = std::move(body),
                                              .headers = {},
                                              .is_cancelled = {}},
                                  service);
}

std::string Header(const HttpResponse& response, const std::string& name) {
  for (const auto& [key, value] : response.headers) {
    if (key == name) {
      return value;
    }
  }
  return {};
}

// Silent 16-bit mono WAV of `seconds` at 16 kHz.
std::string SilentWav(int seconds) {
  const std::uint32_t rate = 16000;
  const std::uint32_t bytes = rate * 2U * static_cast<std::uint32_t>(seconds);
  std::string wav;
  auto u32 = [&](std::uint32_t v) { wav.append(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](std::uint16_t v) { wav.append(reinterpret_cast<const char*>(&v), 2); };
  wav += "RIFF";
  u32(36 + bytes);
  wav += "WAVEfmt ";
  u32(16);
  u16(1);
  u16(1);
  u32(rate);
  u32(rate * 2);
  u16(2);
  u16(16);
  wav += "data";
  u32(bytes);
  wav.append(bytes, '\0');
  return wav;
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_VOXCPM_MODEL");
  const char* voice = std::getenv("GUFO_VOXCPM_VOICE");
  if (model == nullptr || voice == nullptr) {
    std::printf("SKIP: set GUFO_VOXCPM_MODEL and GUFO_VOXCPM_VOICE\n");
    return 77;
  }
  const char* out_env = std::getenv("GUFO_VOXCPM_OUT");
  const std::filesystem::path out_dir = out_env != nullptr ? out_env : "";
  const std::string voice_bytes = ReadFile(voice);
  gufo::models::qwen3_tts::AudioBuffer reference;
  std::string error;
  if (!gufo::models::qwen3_tts::DecodeWav(
          std::span(reinterpret_cast<const std::byte*>(voice_bytes.data()), voice_bytes.size()),
          &reference, &error)) {
    std::printf("FAIL reference wav: %s\n", error.c_str());
    return 1;
  }
  std::map<std::string, gufo::server::TtsVoicePreset> presets;
  presets["erzaehler"] = {.reference_audio = reference,
                          .reference_text = {},
                          .speaker_embedding_only = true,
                          .language = "de"};
  const auto load_start = std::chrono::steady_clock::now();
  TtsService service(gufo::server::TtsServiceOptions{
      .model_root = model,
      .native_context_tokens = 4096,
      .validate_model = true,
      .model_id = {},
      .voices = {},
      .voice_presets = std::move(presets),
      .runner = {},
  });
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  Expect("service ready", service.ready(), service.initialization_error());
  if (!service.ready()) {
    return 1;
  }
  Expect("detected as VoxCPM2",
         service.is_voxcpm2() && !service.is_ming_tts() && !service.is_fish_drama3(),
         service.model_id() + ", load " + std::to_string(load_s) + " s");
  Expect("sample rate 48000", service.sample_rate() == 48000);
  const HttpResponse voices = Send(service, "GET", "/v1/audio/voices");
  Expect("voices list",
         voices.status == 200 && voices.body.find("erzaehler") != std::string::npos,
         voices.body.substr(0, 160));

  const auto speak = [&](const std::string& name, const std::string& body,
                         int want_status = 200) {
    const auto start = std::chrono::steady_clock::now();
    HttpResponse response = Send(service, "POST", "/v1/audio/speech", body);
    std::string audio = response.body;
    std::size_t chunks = 0;
    if (response.streaming_body) {
      response.streaming_body([&](std::string_view chunk) {
        audio.append(chunk);
        ++chunks;
        return true;
      });
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const bool wav = audio.size() > 44 && audio.compare(0, 4, "RIFF") == 0;
    const double audio_s = static_cast<double>(audio.size() - (wav ? 44 : 0)) / 2.0 / 48000.0;
    char detail[240];
    std::snprintf(detail, sizeof(detail),
                  "status %d, %.2f s audio in %.2f s, backend %s, chunks %zu",
                  response.status, response.status == 200 ? audio_s : 0.0, seconds,
                  Header(response, "X-Gufo-TTS-Backend").c_str(), chunks);
    const bool ok = response.status == want_status && (want_status != 200 || audio_s > 0.3);
    Expect(name, ok,
           ok || response.status == 200 ? detail : detail + std::string(" ") + response.body);
    if (!out_dir.empty() && response.status == 200) {
      std::ofstream(out_dir / name, std::ios::binary) << audio;
    }
  };

  speak("voxcpm_froehlich_preset.wav",
        R"({"model":"voxcpm2","voice":"erzaehler","emotion":"fröhlich und begeistert",)"
        R"("input":"Guten Morgen! Heute scheint die Sonne."})");
  speak("voxcpm_fluestern_inline.wav",
        std::string(R"({"model":"voxcpm2","voice":"voice-clone","language":"de",)"
                    R"("instructions":"flüsternd, leise und angespannt",)"
                    R"("input":"Psst, sei leise, sonst hört er uns.","reference_audio":")") +
            gufo::server::EncodeAudioBase64(voice_bytes) + "\"}");
  if (const char* voice_text = std::getenv("GUFO_VOXCPM_VOICE_TEXT"); voice_text != nullptr) {
    gufo::json::Value body = gufo::json::Value::object();
    body["model"] = "voxcpm2";
    body["voice"] = "voice-clone";
    body["input"] = "Wir sehen uns morgen am Bahnhof.";
    body["reference_audio"] = gufo::server::EncodeAudioBase64(voice_bytes);
    // A path to a UTF-8 transcript file (environment strings are not UTF-8 on Windows).
    body["reference_text"] = ReadFile(voice_text);
    speak("voxcpm_ultimate_inline.wav", body.dump());
  }
  speak("voxcpm_stream.pcm",
        R"({"model":"voxcpm2","voice":"erzaehler","emotion":"wütend","response_format":"pcm",)"
        R"("language":"de","input":"Das kann doch nicht dein Ernst sein!"})");
  speak("voxcpm_zeroshot.wav",
        R"({"model":"voxcpm2","voice":"default","input":"Hallo, ich bin eine neue Stimme."})");
  speak("bad_model", R"({"model":"ming-3.0-tts","voice":"erzaehler","input":"Hallo."})", 400);
  speak("bad_emotion_type", R"({"model":"voxcpm2","voice":"erzaehler","emotion":3,"input":"Hallo."})",
        400);
  speak("bad_preset_plus_reference",
        R"({"model":"voxcpm2","voice":"erzaehler","reference_audio":"UklGRg==","input":"Hallo."})",
        400);
  speak("bad_reference_too_long",
        std::string(R"({"model":"voxcpm2","voice":"voice-clone","input":"Hallo.","reference_audio":")") +
            gufo::server::EncodeAudioBase64(SilentWav(61)) + "\"}",
        400);
  speak("bad_empty_input", R"({"model":"voxcpm2","voice":"erzaehler","input":""})", 400);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
