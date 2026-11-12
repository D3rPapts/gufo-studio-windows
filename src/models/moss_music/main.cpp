// gufo-moss-music: driver for the MOSS-Music audio side (stage 2 tooling).
//
//   gufo-moss-music --audio-gguf moss-music-8b.audio.gguf --audio clip.wav
//                   --dump PREFIX [--stage frontend|encoder] [--threads N]
//
// Decodes the WAV, computes the Whisper log-mel features and runs the F32 CPU
// reference. Writes PREFIX.mel.f32 ([128, frames], mel-major),
// PREFIX.frontend.f32 ([tokens, 1280]) and, with --stage encoder,
// PREFIX.enc{8,16,24}.f32 and PREFIX.final.f32 ([tokens, 1280]) plus
// PREFIX.adapter.f32 ([tokens, 4096]); PREFIX.meta.json records the shapes.
// tools/moss_music/check_frontend.py replays the stem in numpy against these.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "src/models/moss_music/config.hpp"
#include "src/models/moss_music/model.hpp"
#include "src/models/moss_music/reference.hpp"
#include "src/models/qwen3_asr/audio.hpp"

namespace {

namespace moss = gufo::models::moss_music;
namespace asr = gufo::models::qwen3_asr;

struct Arguments {
  std::string audio_gguf;
  std::string audio;
  std::string dump;
  std::string stage{"frontend"};
  unsigned threads{0};
};

void Usage() {
  std::cerr << "usage: gufo-moss-music --audio-gguf FILE.gguf --audio FILE.wav "
               "--dump PREFIX [--stage frontend|encoder] [--threads N]\n";
}

bool Parse(int argc, char** argv, Arguments* arguments) {
  for (int index = 1; index < argc; ++index) {
    const std::string name = argv[index];
    if (index + 1 >= argc) {
      return false;
    }
    const std::string value = argv[++index];
    if (name == "--audio-gguf") {
      arguments->audio_gguf = value;
    } else if (name == "--audio") {
      arguments->audio = value;
    } else if (name == "--dump") {
      arguments->dump = value;
    } else if (name == "--stage") {
      arguments->stage = value;
    } else if (name == "--threads") {
      arguments->threads = static_cast<unsigned>(std::stoul(value));
    } else {
      return false;
    }
  }
  return !arguments->audio_gguf.empty() && !arguments->audio.empty() &&
         !arguments->dump.empty() &&
         (arguments->stage == "frontend" || arguments->stage == "encoder");
}

bool WriteFloats(const std::string& path, std::span<const float> values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!Parse(argc, argv, &arguments)) {
    Usage();
    return 2;
  }
  std::ifstream wav_file(arguments.audio, std::ios::binary);
  if (!wav_file) {
    std::cerr << "cannot read " << arguments.audio << '\n';
    return 1;
  }
  const std::vector<char> wav_bytes((std::istreambuf_iterator<char>(wav_file)),
                                    std::istreambuf_iterator<char>());
  asr::AudioBuffer buffer;
  std::string error;
  if (!asr::DecodeWav(std::as_bytes(std::span(wav_bytes)), &buffer, &error)) {
    std::cerr << "WAV: " << error << '\n';
    return 1;
  }
  const std::vector<float> waveform = asr::ResampleMono16k(buffer);
  const asr::LogMelFeatures features = asr::ComputeLogMelFeatures(waveform);
  std::cout << "audio: " << buffer.sample_rate << " Hz, " << buffer.channels
            << " ch -> " << waveform.size() << " samples at 16 kHz, "
            << features.frames << " mel frames\n";

  std::unique_ptr<moss::AudioModel> model =
      moss::AudioModel::Load(arguments.audio_gguf, &error);
  if (model == nullptr) {
    std::cerr << "load: " << error << '\n';
    return 1;
  }
  const moss::AudioEncoderConfig& config = model->config().audio;
  const std::size_t tokens = moss::AudioTokenCount(
      static_cast<std::uint32_t>(features.frames), config);
  const std::vector<std::uint32_t> chunks = moss::ChunkFrameCounts(
      static_cast<std::uint32_t>(features.frames), config);
  std::cout << "chunks: " << chunks.size() << " (" << config.ChunkFrames()
            << " frames each, tail " << chunks.back() << "), tokens: " << tokens
            << ", marker span: " << moss::AudioSpanWithMarkers(tokens).size()
            << '\n';

  const moss::CpuReference reference(*model, arguments.threads);
  const auto started = std::chrono::steady_clock::now();
  std::ofstream meta(arguments.dump + ".meta.json");
  meta << "{\n  \"sample_rate\": 16000,\n  \"samples\": " << waveform.size()
       << ",\n  \"mel_bins\": " << config.mel_bins
       << ",\n  \"frames\": " << features.frames
       << ",\n  \"chunk_frames\": " << config.ChunkFrames()
       << ",\n  \"tokens\": " << tokens << ",\n  \"d_model\": " << config.d_model
       << ",\n  \"stage\": \"" << arguments.stage << "\"";
  bool ok = WriteFloats(arguments.dump + ".mel.f32", features.values);
  if (arguments.stage == "frontend") {
    const std::vector<float> frontend =
        reference.EncodeFrontend(features.values, features.frames);
    ok = ok && WriteFloats(arguments.dump + ".frontend.f32", frontend);
  } else {
    const std::vector<float> frontend =
        reference.EncodeFrontend(features.values, features.frames);
    ok = ok && WriteFloats(arguments.dump + ".frontend.f32", frontend);
    const moss::EncoderOutput encoded =
        reference.EncodeAudio(features.values, features.frames);
    for (std::size_t tap = 0; tap < encoded.deepstack.size(); ++tap) {
      ok = ok && WriteFloats(arguments.dump + ".enc" +
                                 std::to_string(config.deepstack_layers[tap]) +
                                 ".f32",
                             encoded.deepstack[tap]);
    }
    ok = ok && WriteFloats(arguments.dump + ".final.f32", encoded.final);
    const std::vector<float> adapted =
        reference.GatedMlp(model->adapter(), encoded.final, encoded.tokens);
    ok = ok && WriteFloats(arguments.dump + ".adapter.f32", adapted);
    for (std::size_t tap = 0; tap < encoded.deepstack.size(); ++tap) {
      const std::vector<float> merged = reference.GatedMlp(
          model->mergers()[tap], encoded.deepstack[tap], encoded.tokens);
      ok = ok && WriteFloats(arguments.dump + ".merger" + std::to_string(tap) +
                                 ".f32",
                             merged);
    }
    meta << ",\n  \"adapter_out\": " << model->config().adapter.output;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  meta << ",\n  \"seconds\": " << seconds << "\n}\n";
  if (!ok || !meta) {
    std::cerr << "cannot write dumps at " << arguments.dump << '\n';
    return 1;
  }
  std::cout << "done in " << seconds << " s -> " << arguments.dump << ".*\n";
  return 0;
}
