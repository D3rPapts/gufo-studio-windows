// Encodes one audio guide with the native AudioVAE encoder (decode to 32 kHz
// stereo, DAC encoder, attention projection, normalization) and writes the
// normalized latent [32][2][T] as F32 for a comparison with the reference.
// GUFO_H3_MODEL_ROOT=<MiniMax-H3>
// minimax_h3_audio_guide_probe <audio file> <latent.f32> [<pcm32k.f32>]
// Exits 77 when the model root is unset.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/models/minimax_h3/audio_input.hpp"
#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/runtime.hpp"

namespace h3 = gufo::minimax_h3;

int main(int argc, char** argv) {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr || (argc != 3 && argc != 4)) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset or missing arguments\n";
    return 77;
  }
  std::string error;
  std::ifstream input(argv[1], std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                        std::istreambuf_iterator<char>());
  h3::AudioWaveform waveform;
  if (!h3::DecodeGuideAudio(bytes, &waveform, &error)) {
    std::cerr << "decode: " << error << "\n";
    return 1;
  }
  if (argc == 4) {
    std::ofstream pcm(argv[3], std::ios::binary);
    pcm.write(
        reinterpret_cast<const char*>(waveform.pcm.data()),
        static_cast<std::streamsize>(waveform.pcm.size() * sizeof(float)));
  }
  const auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  if (!inventory) {
    std::cerr << "inventory: " << error << "\n";
    return 1;
  }
  h3::AudioVaeEncoderTelemetry telemetry;
  auto encoder =
      h3::AudioVaeEncoder::Create(*inventory, nullptr, &telemetry, &error);
  if (encoder == nullptr) {
    std::cerr << "create: " << error << "\n";
    return 1;
  }
  std::vector<float> latent;
  int frames = 0;
  if (!encoder->Encode(waveform, nullptr, &latent, &frames, &telemetry,
                       &error)) {
    std::cerr << "encode: " << error << "\n";
    return 1;
  }
  std::ofstream output(argv[2], std::ios::binary);
  output.write(reinterpret_cast<const char*>(latent.data()),
               static_cast<std::streamsize>(latent.size() * sizeof(float)));
  std::printf(
      "H3 audio guide samples=%d latent_frames=%d load_ms=%.1f encode_ms=%.1f "
      "weight_mib=%.1f dispatches=%llu\n",
      waveform.samples, frames, telemetry.load_ms, telemetry.encode_ms,
      static_cast<double>(telemetry.weight_bytes) / (1024.0 * 1024.0),
      static_cast<unsigned long long>(telemetry.dispatches));
  return output ? 0 : 1;
}
