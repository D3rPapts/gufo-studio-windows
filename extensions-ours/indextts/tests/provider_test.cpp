// The IndexTTS-2.5 speech_synthesis provider with the real German bundle,
// driven through the extension API 0.2 TtsRunner (in process, no server):
// metadata, preset-style and inline reference cloning, emotion in
// `instructions` (vector, names, JSON object), PCM streaming and the request
// errors the 1.x serve test checked as 400s (here: Synthesize fails before
// the engine runs). Replaces the 1.x serve_test (the core TtsService / HTTP
// handler are not reachable from an out-of-tree extension). CPU runtime unless
// a HIP build finds a GPU.
//
//   GUFO_INDEXTTS_MODEL  bundle directory
//   GUFO_INDEXTTS_VOICE  reference WAV (<= 10 s)
//   GUFO_INDEXTTS_OUT    optional directory for the produced WAV files
//   GUFO_INDEXTTS_QUICK  "1": only one synthesis plus the error cases
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
#include <vector>

#include "indextts/provider.hpp"
#include "ming_tts/wav.hpp"

namespace ext = gufo::ext;
namespace itts = gufo::models::indextts;
namespace wav = gufo::models::ming_tts;

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
  wav::WavAudio reference;
  std::string error;
  if (!wav::DecodeWav(
          std::span(reinterpret_cast<const std::byte*>(voice_bytes.data()), voice_bytes.size()),
          &reference, &error)) {
    std::printf("FAIL reference wav: %s\n", error.c_str());
    return 1;
  }

  const auto load_start = std::chrono::steady_clock::now();
  auto runner = itts::CreateIndexTtsRunner(model, 4096, &error);
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  Expect("runner created", runner != nullptr, error + " load " + std::to_string(load_s) + " s");
  if (runner == nullptr) {
    return 1;
  }
  Expect("sample rate 22050", runner->SampleRate() == 22050);
  Expect("model id", runner->ModelId() == "indextts-2.5-german", runner->ModelId());
  const auto cond = runner->Conditioning();
  Expect("conditioning: reference required, presets, instructions",
         !cond.unconditioned && cond.reference_audio == ext::TtsConditioningMode::kRequired &&
             cond.reference_presets &&
             cond.instructions == ext::TtsConditioningMode::kOptional &&
             cond.default_voice == "voice-clone");

  const auto base = [&](const std::string& speaker) {
    ext::SynthesisRequest r;
    r.speaker = speaker;
    r.language = "english";  // what the core sends when the client names none
    r.reference_audio = {.sample_rate = reference.sample_rate,
                         .channels = reference.channels,
                         .samples = reference.samples};
    r.speaker_embedding_only = true;
    r.sampling.seed = 1234;
    return r;
  };

  const auto speak = [&](const std::string& name, ext::SynthesisRequest request, bool stream,
                         bool want_ok = true) {
    std::vector<float> streamed;
    std::size_t chunks = 0;
    if (stream) {
      request.on_audio = [&](std::span<const float> chunk) {
        streamed.insert(streamed.end(), chunk.begin(), chunk.end());
        ++chunks;
        return true;
      };
    }
    ext::SynthesisResult result;
    std::string synth_error;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = runner->Synthesize(request, {}, &result, &synth_error);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::vector<float>& audio = stream ? streamed : result.samples;
    const double audio_s = static_cast<double>(audio.size()) / 22050.0;
    char detail[240];
    std::snprintf(detail, sizeof(detail), "%.2f s audio in %.2f s, %zu steps, chunks %zu", audio_s,
                  seconds, result.codes.size(), chunks);
    if (!want_ok) {
      Expect(name, !ok, ok ? "accepted" : synth_error);
      return;
    }
    const bool good = ok && audio_s > 0.3 && result.sample_rate == 22050 &&
                      result.sample_count == audio.size() && (!stream || result.samples.empty());
    Expect(name, good, ok ? std::string(detail) : synth_error);
    if (!out_dir.empty() && ok) {
      std::ofstream(out_dir / name, std::ios::binary) << wav::EncodeWavPcm16(audio, 22050);
    }
  };

  {
    auto r = base("erzaehler");
    r.text = "Ich kann nicht mehr. Es tut so weh.";
    r.instruct = "Trauer";
    speak("indextts_trauer_preset.wav", std::move(r), false);
  }
  if (!quick) {
    auto r = base("voice-clone");
    r.text = "Warum hast du mich verlassen?";
    r.language = "de";
    r.instruct = "emotion=[0,0,0.6,0,0,0.2,0,0]";
    speak("indextts_vector_inline.wav", std::move(r), false);

    auto m = base("erzaehler");
    m.text = "Das darf doch nicht wahr sein!";
    m.instruct = R"({"emotion":{"Wut":0.6,"Angst":0.2}})";
    speak("indextts_mix_object.wav", std::move(m), false);

    auto s = base("erzaehler");
    s.text = "Wir haben es geschafft! Endlich!";
    s.instruct = "emotion=\xC3\x9C" "berraschung:0.7";
    speak("indextts_stream.wav", std::move(s), true);
  }

  auto bad = base("erzaehler");
  bad.text = "Hallo.";
  bad.instruct = "Langeweile";
  speak("bad_emotion_name", bad, false, false);
  bad.instruct = "emotion=[0,1,2]";
  speak("bad_emotion_vector", bad, false, false);
  bad.instruct.clear();
  bad.language = "en";
  speak("bad_language", bad, false, false);
  auto missing = base("voice-clone");
  missing.text = "Hallo.";
  missing.reference_audio = {};
  speak("bad_missing_reference", missing, false, false);
  auto too_long = base("voice-clone");
  too_long.text = "Hallo.";
  too_long.reference_audio = {.sample_rate = 16000, .channels = 1,
                              .samples = std::vector<float>(16000U * 11U, 0.0F)};
  speak("bad_reference_too_long", too_long, false, false);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
