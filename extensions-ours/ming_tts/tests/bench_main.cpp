// ming_tts_bench: end-to-end Ming-3.0-tts synthesis of a sentence list for
// several reference voices, with timing. Development tool (not a server).
//
//   ming_tts_bench --model DIR --sentences tts_testsaetze.json
//                  --voices DIR --voice lars --voice harold ... --out DIR
//                  [--mode emotion|zeroshot|both] [--seed N] [--limit N]
//
// Each voice is <voices>/<name>_ref.wav (transcript <name>_ref.txt for
// zeroshot). Emotions map like the Python listening test (ming_test.py):
// Wut->angry, Angst->fear, Trauer->sad, Flüstern->fear, Verzweiflung->sad,
// Triumph->happy. Writes <out>/ming_<mode>_<voice>_<nr>.wav and
// <out>/ming_bench.json. The backend follows GUFO_MING_BACKEND (default: GPU
// when available). Written with AI assistance (Claude Code).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>
#include "ming_tts/synthesis.hpp"
#include "ming_tts/wav.hpp"

namespace json = gufo::json;
namespace ming = gufo::models::ming_tts;

namespace {

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteWav(const std::filesystem::path& path, const std::vector<float>& samples,
              std::uint32_t rate) {
  std::ofstream f(path, std::ios::binary);
  const auto n = static_cast<std::uint32_t>(samples.size());
  const std::uint32_t data_bytes = n * 2, riff = 36 + data_bytes, byte_rate = rate * 2, fmt = 16;
  const std::uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
  f.write("RIFF", 4).write(reinterpret_cast<const char*>(&riff), 4).write("WAVEfmt ", 8);
  f.write(reinterpret_cast<const char*>(&fmt), 4).write(reinterpret_cast<const char*>(&pcm), 2);
  f.write(reinterpret_cast<const char*>(&ch), 2).write(reinterpret_cast<const char*>(&rate), 4);
  f.write(reinterpret_cast<const char*>(&byte_rate), 4).write(reinterpret_cast<const char*>(&align), 2);
  f.write(reinterpret_cast<const char*>(&bits), 2).write("data", 4);
  f.write(reinterpret_cast<const char*>(&data_bytes), 4);
  for (const float v : samples) {
    const float c = v < -1.0F ? -1.0F : (v > 1.0F ? 1.0F : v);
    const auto s = static_cast<std::int16_t>(c * 32767.0F + (c >= 0 ? 0.5F : -0.5F));
    f.write(reinterpret_cast<const char*>(&s), 2);
  }
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, sentences, voices_dir, out_dir, mode = "emotion";
  std::vector<std::string> voices;
  std::uint64_t seed = 4242;
  int limit = 1000;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--model") model = value;
    else if (key == "--sentences") sentences = value;
    else if (key == "--voices") voices_dir = value;
    else if (key == "--voice") voices.push_back(value);
    else if (key == "--out") out_dir = value;
    else if (key == "--mode") mode = value;
    else if (key == "--seed") seed = std::stoull(value);
    else if (key == "--limit") limit = std::stoi(value);
    else {
      std::fprintf(stderr, "unknown option %s\n", key.c_str());
      return 2;
    }
  }
  if (model.empty() || sentences.empty() || voices_dir.empty() || out_dir.empty() ||
      voices.empty()) {
    std::fprintf(stderr, "usage: ming_tts_bench --model DIR --sentences JSON --voices DIR "
                         "--voice NAME [--voice NAME ...] --out DIR [--mode emotion|zeroshot|both]\n");
    return 2;
  }
  std::filesystem::create_directories(out_dir);
  const std::map<std::string, std::string> emotion = {
      {"Wut", "angry"},          {"Angst", "fear"}, {"Trauer", "sad"}, {"Flüstern", "fear"},
      {"Verzweiflung", "sad"},   {"Triumph", "happy"}};
  const json::Value lines = json::parse(ReadFile(sentences));
  auto t = std::chrono::steady_clock::now();
  std::string error;
  auto engine = ming::Engine::Create(model, &error);
  if (engine == nullptr) {
    std::fprintf(stderr, "engine: %s\n", error.c_str());
    return 1;
  }
  json::Value report = json::Value::object();
  report["load_s"] = Seconds(t);
  report["backend"] = std::string(engine->backend_name());
  std::printf("engine %s, load %.1f s\n", engine->backend_name(), Seconds(t));
  std::fflush(stdout);
  json::Value clips = json::Value::array();
  double total_s = 0, total_audio = 0;
  std::vector<std::string> modes;
  if (mode == "emotion" || mode == "both") modes.push_back("emotion");
  if (mode == "zeroshot" || mode == "both") modes.push_back("zeroshot");
  for (const std::string& voice : voices) {
    const std::filesystem::path wav_path = std::filesystem::path(voices_dir) / (voice + "_ref.wav");
    const std::string bytes = ReadFile(wav_path);
    gufo::models::ming_tts::WavAudio audio;
    if (!gufo::models::ming_tts::DecodeWav(
            std::span(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()), &audio,
            &error)) {
      std::fprintf(stderr, "%s: %s\n", wav_path.string().c_str(), error.c_str());
      return 1;
    }
    std::vector<float> mono(audio.samples.size() / audio.channels);
    for (std::size_t f = 0; f < mono.size(); ++f) {
      double sum = 0;
      for (std::uint32_t c = 0; c < audio.channels; ++c) sum += audio.samples[f * audio.channels + c];
      mono[f] = static_cast<float>(sum / audio.channels);
    }
    const std::string transcript =
        ReadFile(std::filesystem::path(voices_dir) / (voice + "_ref.txt"));
    for (const std::string& m : modes) {
      int count = 0;
      for (const auto& line : lines.find("saetze")->items()) {
        if (count++ >= limit) break;
        ming::SynthesisRequest request;
        request.text = line.member_str("text");
        request.language = "de";
        const auto found = emotion.find(line.member_str("emotion"));
        request.emotion = found != emotion.end() ? found->second : "neutral";
        request.reference_audio = mono;
        request.reference_sample_rate = audio.sample_rate;
        if (m == "zeroshot") {
          request.reference_text = transcript;
        }
        request.seed = seed;
        ming::SynthesisResult result;
        t = std::chrono::steady_clock::now();
        if (!engine->Generate(request, {}, &result, &error)) {
          std::fprintf(stderr, "generate failed: %s\n", error.c_str());
          return 1;
        }
        const double s = Seconds(t);
        const double audio_s = static_cast<double>(result.samples.size()) / result.sample_rate;
        const int nr = static_cast<int>(line.find("nr")->as_double());
        const std::string name = "ming_" + m + "_" + voice + "_" + std::to_string(nr) + ".wav";
        WriteWav(std::filesystem::path(out_dir) / name, result.samples, result.sample_rate);
        std::int64_t steps = 0;
        for (const auto st : result.fragment_steps) steps += st;
        json::Value clip = json::Value::object();
        clip["file"] = name;
        clip["mode"] = m;
        clip["voice"] = voice;
        clip["nr"] = static_cast<std::int64_t>(nr);
        clip["emotion"] = line.member_str("emotion");
        clip["label"] = request.emotion;
        clip["seconds"] = s;
        clip["audio_s"] = audio_s;
        clip["rtf"] = s / (audio_s > 0 ? audio_s : 1e-9);
        clip["steps"] = steps;
        clip["fragments"] = static_cast<std::int64_t>(result.fragments.size());
        clips.push_back(clip);
        total_s += s;
        total_audio += audio_s;
        std::printf("%s  %.2f s for %.2f s audio (RTF %.3f, %lld steps)\n", name.c_str(), s,
                    audio_s, s / (audio_s > 0 ? audio_s : 1e-9), static_cast<long long>(steps));
        std::fflush(stdout);
      }
    }
  }
  report["clips"] = clips;
  report["total_s"] = total_s;
  report["total_audio_s"] = total_audio;
  report["rtf"] = total_s / (total_audio > 0 ? total_audio : 1e-9);
  std::ofstream(std::filesystem::path(out_dir) / "ming_bench.json") << report.dump();
  std::printf("=== %zu clips, %.1f s for %.1f s audio, RTF %.3f\n", clips.size(), total_s,
              total_audio, total_s / (total_audio > 0 ? total_audio : 1e-9));
  return 0;
}
