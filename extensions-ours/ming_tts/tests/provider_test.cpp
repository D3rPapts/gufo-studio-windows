// The Ming-3.0-tts speech_synthesis provider with a real checkpoint, driven
// through the extension API 0.2 TtsRunner (in process, no server): metadata,
// voice-preset style and inline cloning, emotion, BASIC controls and free-text
// captions in `instructions`, streaming, and request validation. Replaces the
// 1.x serve_test (the core TtsService / HTTP handler are not reachable from an
// out-of-tree extension).
//
//   GUFO_MING_MODEL      checkpoint directory
//   GUFO_MING_VOICE      reference WAV for the clone requests
//   GUFO_MING_OUT        optional directory for the produced WAV files
//   GUFO_MING_VOICE_TEXT optional UTF-8 transcript file: adds a ZERO_SHOT request
//
// Exits 77 (skip) when the model or voice is not configured.
// Written with AI assistance (Claude Code).
#include <gufo/ext/tts.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <utility>

#include "ming_tts/provider.hpp"
#include "ming_tts/wav.hpp"

namespace ext = gufo::ext;
namespace ming = gufo::models::ming_tts;

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
  ming::WavAudio wav;
  std::string error;
  if (!ming::DecodeWav(std::span(reinterpret_cast<const std::byte*>(voice_bytes.data()),
                                 voice_bytes.size()),
                       &wav, &error)) {
    std::printf("FAIL reference wav: %s\n", error.c_str());
    return 1;
  }
  const ext::TtsAudioBuffer reference{
      .sample_rate = wav.sample_rate, .channels = wav.channels, .samples = wav.samples};

  const auto load_start = std::chrono::steady_clock::now();
  auto runner = ming::CreateMingTtsRunner(model, 4096, &error);
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  Expect("runner created", runner != nullptr, error);
  if (runner == nullptr) {
    return 1;
  }
  Expect("model id", runner->ModelId() == "ming-3.0-tts-0.5b",
         runner->ModelId() + ", load " + std::to_string(load_s) + " s");
  Expect("sample rate 44100", runner->SampleRate() == 44100);
  const auto cond = runner->Conditioning();
  Expect("conditioning", cond.unconditioned && !cond.require_voice &&
                             cond.reference_audio == ext::TtsConditioningMode::kOptional &&
                             cond.reference_presets &&
                             cond.instructions == ext::TtsConditioningMode::kOptional);

  const auto speak = [&](const std::string& name, ext::SynthesisRequest request,
                         bool want_ok = true, bool stream = false) {
    std::size_t streamed = 0;
    if (stream) {
      request.on_audio = [&](std::span<const float> chunk) {
        streamed += chunk.size();
        return true;
      };
    }
    ext::SynthesisResult result;
    std::string message;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = runner->Synthesize(request, {}, &result, &message);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::size_t samples = stream ? streamed : result.samples.size();
    const double audio_s = static_cast<double>(samples) / 44100.0;
    char detail[200];
    std::snprintf(detail, sizeof(detail), "%.2f s audio in %.2f s, steps %zu", audio_s,
                  seconds, result.codes.size());
    if (want_ok) {
      Expect(name, ok && result.sample_rate == 44100 && audio_s > 0.3,
             ok ? detail : message);
    } else {
      Expect(name, !ok && !message.empty(), message);
    }
    if (ok && !out_dir.empty() && !stream) {
      std::ofstream(out_dir / (name + ".wav"), std::ios::binary)
          << ming::EncodeWavPcm16(result.samples, result.sample_rate);
    }
  };

  ext::SynthesisRequest wut;
  wut.text = "Das kann doch nicht dein Ernst sein! Ich habe dir das schon dreimal gesagt.";
  wut.language = "de";
  wut.instruct = "Wut";
  wut.reference_audio = reference;
  wut.speaker_embedding_only = true;
  speak("ming_wut_clone", wut);

  ext::SynthesisRequest basic;
  basic.text = "Ich spreche jetzt langsam und tief.";
  basic.language = "de";
  basic.instruct = R"({"tempo":"langsam","pitch":"tief"})";
  basic.reference_audio = reference;
  basic.speaker_embedding_only = true;
  speak("ming_basic_slow_low", basic);

  if (const char* voice_text = std::getenv("GUFO_MING_VOICE_TEXT"); voice_text != nullptr) {
    ext::SynthesisRequest zs;
    zs.text = "Wir sehen uns morgen am Bahnhof, ich bringe die Karten mit.";
    zs.language = "de";
    zs.reference_audio = reference;
    // A path to a UTF-8 transcript file (environment strings are not UTF-8 on Windows).
    zs.reference_text = ReadFile(voice_text);
    speak("ming_zeroshot", zs);
  }

  ext::SynthesisRequest whisper;
  whisper.text = "Psst, sei leise, sonst hört er uns.";
  whisper.language = "de";
  whisper.instruct =
      "Whisper very softly and tensely, almost breathless, close to the microphone.";
  speak("ming_fluestern_unconditioned", whisper);

  ext::SynthesisRequest stream;
  stream.text = "Guten Morgen! Heute scheint die Sonne, und wir fahren ans Meer.";
  stream.language = "de";
  stream.instruct = "happy";
  stream.reference_audio = reference;
  stream.speaker_embedding_only = true;
  speak("ming_stream", stream, true, true);

  ext::SynthesisRequest bad = whisper;
  bad.instruct = R"({"emotion":"langweilig"})";
  speak("bad_emotion", bad, false);
  bad.instruct = "tempo=fast, dialect=cantonese";
  speak("conflicting_controls", bad, false);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
