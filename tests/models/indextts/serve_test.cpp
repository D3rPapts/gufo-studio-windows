// /v1/audio/speech with the real IndexTTS-2.5-German bundle through TtsService
// and the audio API handler (in process, no sockets): detection, voice preset
// and inline reference cloning (<= 10 s), emotion as the 8-value vector, as
// German/English names and as an object, PCM streaming and request
// validation (400s). CPU runtime unless a HIP build finds a GPU.
//
//   GUFO_INDEXTTS_MODEL  bundle directory
//   GUFO_INDEXTTS_VOICE  reference WAV (<= 10 s)
//   GUFO_INDEXTTS_OUT    optional directory for the produced WAV/PCM files
//   GUFO_INDEXTTS_QUICK  "1": only one synthesis request plus the 400 cases
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
  const char* model = std::getenv("GUFO_INDEXTTS_MODEL");
  const char* voice = std::getenv("GUFO_INDEXTTS_VOICE");
  if (model == nullptr || voice == nullptr) {
    std::printf("SKIP: set GUFO_INDEXTTS_MODEL and GUFO_INDEXTTS_VOICE\n");
    return 77;
  }
  const char* out_env = std::getenv("GUFO_INDEXTTS_OUT");
  const char* quick_env = std::getenv("GUFO_INDEXTTS_QUICK");
  const bool quick = quick_env != nullptr && std::string(quick_env) == "1";
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
  Expect("detected as IndexTTS-2.5",
         service.is_indextts() && !service.is_voxcpm2() && !service.is_ming_tts(),
         service.model_id() + ", load " + std::to_string(load_s) + " s");
  Expect("sample rate 22050", service.sample_rate() == 22050);
  const HttpResponse voices = Send(service, "GET", "/v1/audio/voices");
  Expect("voices list",
         voices.status == 200 && voices.body.find("erzaehler") != std::string::npos,
         voices.body.substr(0, 160));

  const auto speak = [&](const std::string& name, const std::string& body, int want_status = 200) {
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
    const double audio_s = static_cast<double>(audio.size() - (wav ? 44 : 0)) / 2.0 / 22050.0;
    char detail[240];
    std::snprintf(detail, sizeof(detail), "status %d, %.2f s audio in %.2f s, backend %s, chunks %zu",
                  response.status, response.status == 200 ? audio_s : 0.0, seconds,
                  Header(response, "X-Gufo-TTS-Backend").c_str(), chunks);
    const bool ok = response.status == want_status && (want_status != 200 || audio_s > 0.3);
    Expect(name, ok, ok || response.status == 200 ? detail : detail + std::string(" ") + response.body);
    if (!out_dir.empty() && response.status == 200) {
      std::ofstream(out_dir / name, std::ios::binary) << audio;
    }
  };

  speak("indextts_trauer_preset.wav",
        R"({"model":"indextts-2.5-german","voice":"erzaehler","emotion":"Trauer",)"
        R"("input":"Ich kann nicht mehr. Es tut so weh."})");
  if (!quick) {
    speak("indextts_vector_inline.wav",
          std::string(R"({"model":"indextts-2.5","voice":"voice-clone","language":"de",)"
                      R"("emotion":[0,0,0.6,0,0,0.2,0,0],)"
                      R"("input":"Warum hast du mich verlassen?","reference_audio":")") +
              gufo::server::EncodeAudioBase64(voice_bytes) + "\"}");
    speak("indextts_mix_object.wav",
          R"({"model":"indextts-2.5-german","voice":"erzaehler","emotion":{"Wut":0.6,"Angst":0.2},)"
          R"("input":"Das darf doch nicht wahr sein!"})");
    speak("indextts_stream.pcm",
          R"({"model":"indextts-2.5-german","voice":"erzaehler","emotion":"Überraschung:0.7",)"
          R"("response_format":"pcm","input":"Wir haben es geschafft! Endlich!"})");
  }
  speak("bad_model", R"({"model":"voxcpm2","voice":"erzaehler","input":"Hallo."})", 400);
  speak("bad_emotion_name",
        R"({"model":"indextts-2.5","voice":"erzaehler","emotion":"Langeweile","input":"Hallo."})", 400);
  speak("bad_emotion_vector",
        R"({"model":"indextts-2.5","voice":"erzaehler","emotion":[0,1,2],"input":"Hallo."})", 400);
  speak("bad_language",
        R"({"model":"indextts-2.5","voice":"erzaehler","language":"en","input":"Hello."})", 400);
  speak("bad_missing_reference", R"({"model":"indextts-2.5","voice":"voice-clone","input":"Hallo."})",
        400);
  speak("bad_preset_plus_reference",
        R"({"model":"indextts-2.5","voice":"erzaehler","reference_audio":"UklGRg==","input":"Hallo."})",
        400);
  speak("bad_reference_too_long",
        std::string(R"({"model":"indextts-2.5","voice":"voice-clone","input":"Hallo.","reference_audio":")") +
            gufo::server::EncodeAudioBase64(SilentWav(11)) + "\"}",
        400);
  speak("bad_empty_input", R"({"model":"indextts-2.5","voice":"erzaehler","input":""})", 400);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
