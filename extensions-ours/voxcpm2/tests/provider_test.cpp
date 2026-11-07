// The VoxCPM2 speech_synthesis provider with a real checkpoint, driven
// through the extension API 0.2 TtsRunner (in process, no server): metadata,
// reference cloning with a style instruction, "ultimate" cloning, zero-shot,
// streaming and the reference length limit. Replaces the 1.x serve_test (the
// core TtsService / HTTP handler are not reachable from an out-of-tree
// extension). CPU runtime unless a HIP build finds a GPU.
//
//   GUFO_VOXCPM_MODEL       checkpoint directory
//   GUFO_VOXCPM_VOICE       reference WAV for the clone requests (<= 10 s)
//   GUFO_VOXCPM_OUT         optional directory for the produced WAV files
//   GUFO_VOXCPM_VOICE_TEXT  optional UTF-8 transcript file: adds an
//                           "ultimate cloning" request (reference + transcript)
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

#include "ming_tts/wav.hpp"
#include "voxcpm2/provider.hpp"

namespace ext = gufo::ext;
namespace vox = gufo::models::voxcpm2;
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
  const char* model = std::getenv("GUFO_VOXCPM_MODEL");
  const char* voice = std::getenv("GUFO_VOXCPM_VOICE");
  if (model == nullptr || voice == nullptr) {
    std::printf("SKIP: set GUFO_VOXCPM_MODEL and GUFO_VOXCPM_VOICE\n");
    return 77;
  }
  const char* out_env = std::getenv("GUFO_VOXCPM_OUT");
  const std::filesystem::path out_dir = out_env != nullptr ? out_env : "";
  const std::string voice_bytes = ReadFile(voice);
  wav::WavAudio decoded;
  std::string error;
  if (!wav::DecodeWav(std::span(reinterpret_cast<const std::byte*>(voice_bytes.data()),
                                voice_bytes.size()),
                      &decoded, &error)) {
    std::printf("FAIL reference wav: %s\n", error.c_str());
    return 1;
  }
  const ext::TtsAudioBuffer reference{.sample_rate = decoded.sample_rate,
                                      .channels = decoded.channels,
                                      .samples = decoded.samples};

  const auto load_start = std::chrono::steady_clock::now();
  auto runner = vox::CreateVoxCpm2TtsRunner(model, 4096, &error);
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  Expect("runner created", runner != nullptr, error);
  if (runner == nullptr) {
    return 1;
  }
  Expect("model id", runner->ModelId() == "voxcpm2",
         runner->ModelId() + ", load " + std::to_string(load_s) + " s");
  Expect("sample rate 48000", runner->SampleRate() == 48000);

  const auto speak = [&](const std::string& name, ext::SynthesisRequest request,
                         bool want_ok = true, bool stream = false) {
    std::size_t streamed = 0;
    std::size_t chunks = 0;
    if (stream) {
      request.on_audio = [&](std::span<const float> chunk) {
        streamed += chunk.size();
        ++chunks;
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
    const double audio_s = static_cast<double>(samples) / 48000.0;
    char detail[200];
    std::snprintf(detail, sizeof(detail), "%.2f s audio in %.2f s, steps %zu, chunks %zu",
                  audio_s, seconds, result.codes.size(), chunks);
    if (want_ok) {
      Expect(name, ok && result.sample_rate == 48000 && audio_s > 0.3,
             ok ? detail : message);
    } else {
      Expect(name, !ok && !message.empty(), message);
    }
    if (ok && !out_dir.empty() && !stream) {
      std::ofstream(out_dir / (name + ".wav"), std::ios::binary)
          << wav::EncodeWavPcm16(result.samples, result.sample_rate);
    }
  };

  ext::SynthesisRequest happy;
  happy.text = "Guten Morgen! Heute scheint die Sonne.";
  happy.instruct = "fröhlich und begeistert";
  happy.reference_audio = reference;
  happy.speaker_embedding_only = true;
  speak("voxcpm_froehlich_clone", happy);

  if (const char* voice_text = std::getenv("GUFO_VOXCPM_VOICE_TEXT"); voice_text != nullptr) {
    ext::SynthesisRequest ultimate;
    ultimate.text = "Wir sehen uns morgen am Bahnhof.";
    ultimate.reference_audio = reference;
    // A path to a UTF-8 transcript file (environment strings are not UTF-8 on Windows).
    ultimate.reference_text = ReadFile(voice_text);
    speak("voxcpm_ultimate", ultimate);
  }

  ext::SynthesisRequest angry;
  angry.text = "Das kann doch nicht dein Ernst sein!";
  angry.instruct = "wütend";
  angry.reference_audio = reference;
  angry.speaker_embedding_only = true;
  speak("voxcpm_stream", angry, true, true);

  ext::SynthesisRequest zero;
  zero.text = "Hallo, ich bin eine neue Stimme.";
  speak("voxcpm_zeroshot", zero);

  ext::SynthesisRequest too_long;
  too_long.text = "Hallo.";
  too_long.reference_audio = {.sample_rate = 16000, .channels = 1,
                              .samples = std::vector<float>(16000U * 61U, 0.0F)};
  speak("bad_reference_too_long", too_long, false);

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
