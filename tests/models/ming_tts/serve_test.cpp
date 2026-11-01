// /v1/audio/speech with a real Ming-3.0-tts checkpoint through TtsService and
// the audio API handler (in process, no sockets): detection, voice preset and
// inline reference cloning, emotion and free-text instructions, PCM streaming
// and request validation. CPU runtime.
//
//   GUFO_MING_MODEL   checkpoint directory
//   GUFO_MING_VOICE   reference WAV for the preset / inline clone
//   GUFO_MING_OUT     optional directory for the produced WAV/PCM files
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

HttpResponse Send(TtsService& service, const std::string& method,
                  const std::string& path, std::string body = {}) {
  return HandleAudioTtsApiRequest(
      HttpRequest{.method = method,
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

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_MING_MODEL");
  const char* voice = std::getenv("GUFO_MING_VOICE");
  if (model == nullptr || voice == nullptr) {
    std::printf("SKIP: set GUFO_MING_MODEL and GUFO_MING_VOICE\n");
    return 77;
  }
  const char* out_env = std::getenv("GUFO_MING_OUT");
  const std::filesystem::path out_dir = out_env != nullptr ? out_env : "";
  const std::string voice_bytes = ReadFile(voice);
  gufo::models::qwen3_tts::AudioBuffer reference;
  std::string error;
  if (!gufo::models::qwen3_tts::DecodeWav(
          std::span(reinterpret_cast<const std::byte*>(voice_bytes.data()),
                    voice_bytes.size()),
          &reference, &error)) {
    std::printf("FAIL reference wav: %s\n", error.c_str());
    return 1;
  }
  std::map<std::string, gufo::server::TtsVoicePreset> presets;
  presets["fabian"] = {.reference_audio = reference,
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
  const double load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - load_start)
                            .count();
  Expect("service ready", service.ready(), service.initialization_error());
  if (!service.ready()) {
    return 1;
  }
  Expect("detected as Ming", service.is_ming_tts() && !service.is_fish_drama3(),
         service.model_id() + ", load " + std::to_string(load_s) + " s");
  Expect("sample rate 44100", service.sample_rate() == 44100);
  const HttpResponse voices = Send(service, "GET", "/v1/audio/voices");
  Expect("voices list", voices.status == 200 &&
                            voices.body.find("fabian") != std::string::npos,
         voices.body.substr(0, 160));

  const auto speak = [&](const std::string& name, const std::string& body,
                         int want_status = 200) {
    const auto start = std::chrono::steady_clock::now();
    HttpResponse response = Send(service, "POST", "/v1/audio/speech", body);
    std::string audio = response.body;
    if (response.streaming_body) {
      response.streaming_body([&](std::string_view chunk) {
        audio.append(chunk);
        return true;
      });
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const bool wav = audio.size() > 44 && audio.compare(0, 4, "RIFF") == 0;
    const double audio_s =
        static_cast<double>(audio.size() - (wav ? 44 : 0)) / 2.0 / 44100.0;
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "status %d, %.2f s audio in %.2f s, backend %s, steps %s",
                  response.status, response.status == 200 ? audio_s : 0.0, seconds,
                  Header(response, "X-Gufo-TTS-Backend").c_str(),
                  Header(response, "X-Gufo-Codec-Steps").c_str());
    const bool ok = response.status == want_status &&
                    (want_status != 200 || audio_s > 0.3);
    Expect(name, ok, ok || response.status == 200 ? detail : detail + std::string(" ") + response.body);
    if (!out_dir.empty() && response.status == 200) {
      std::ofstream(out_dir / name, std::ios::binary) << audio;
    }
  };

  speak("ming_wut_preset.wav",
        R"({"model":"ming-3.0-tts","voice":"fabian","emotion":"Wut",)"
        R"("input":"Das kann doch nicht dein Ernst sein! Ich habe dir das schon dreimal gesagt."})");
  speak("ming_trauer_inline.wav",
        std::string(R"({"model":"ming-3.0-tts-0.5b","voice":"voice-clone","language":"de",)"
                    R"("emotion":"sad","input":"Es ist vorbei. Sie kommt nicht mehr zurück.",)"
                    R"("reference_audio":")") +
            gufo::server::EncodeAudioBase64(voice_bytes) + "\"}");
  speak("ming_fluestern.wav",
        R"({"model":"ming-3.0-tts","voice":"fabian","instructions":)"
        R"("Whisper very softly and tensely, almost breathless, close to the microphone.",)"
        R"("input":"Psst, sei leise, sonst hört er uns."})");
  speak("ming_stream.pcm",
        R"({"model":"ming-3.0-tts","voice":"fabian","emotion":"happy","response_format":"pcm",)"
        R"("input":"Guten Morgen! Heute scheint die Sonne, und wir fahren ans Meer."})");
  speak("bad_emotion", R"({"model":"ming-3.0-tts","voice":"fabian","emotion":"langweilig","input":"Hallo."})",
        400);
  speak("bad_model", R"({"model":"fish-drama-3","voice":"fabian","input":"Hallo."})", 400);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
