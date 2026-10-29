// Decodes one dumped H3 video latent (gufo video --latents-dir) with the
// native VisualVAE and reports load/decode timings; optionally writes the F32
// RGB frames for an A/B comparison between precision modes.
// GUFO_H3_MODEL_ROOT=<MiniMax-H3> [GUFO_H3_VAE_PRECISION=f32|bf16|f16]
// minimax_h3_video_vae_bench <video_final.f32> <width> <height> <frames>
//     [<frames.f32>]
// Exits 77 when the model root is unset.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae.hpp"

namespace h3 = gufo::minimax_h3;

int main(int argc, char** argv) {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr || (argc != 5 && argc != 6)) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset or missing arguments\n";
    return 77;
  }
  std::string error;
  const auto inventory =
      h3::ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  if (!inventory) {
    std::cerr << "inventory: " << error << "\n";
    return 1;
  }
  const auto geometry = h3::ResolveGenerationGeometry(
      std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), &error);
  if (!geometry) {
    std::cerr << "geometry: " << error << "\n";
    return 1;
  }
  const std::size_t elements =
      static_cast<std::size_t>(h3::kH3VideoVaeLatentChannels) *
      geometry->temporal.video_latent_frames * geometry->latent_height *
      geometry->latent_width;
  std::vector<float> latent(elements);
  {
    std::ifstream input(argv[1], std::ios::binary);
    input.read(reinterpret_cast<char*>(latent.data()),
               static_cast<std::streamsize>(elements * sizeof(float)));
    if (!input || input.peek() != EOF) {
      std::cerr << "latent size differs from " << elements << " floats\n";
      return 1;
    }
  }
  h3::VideoVaeTelemetry telemetry;
  auto decoder = h3::VideoVaeDecoder::Create(
      *inventory, {.geometry = *geometry}, nullptr, &telemetry, &error);
  if (decoder == nullptr) {
    std::cerr << "create: " << error << "\n";
    return 1;
  }
  h3::VideoFrames frames;
  if (!decoder->Decode(latent, nullptr, nullptr, nullptr, &frames, &telemetry,
                       &error)) {
    std::cerr << "decode: " << error << "\n";
    return 1;
  }
  const double tiles =
      std::accumulate(telemetry.tile_ms.begin(), telemetry.tile_ms.end(), 0.0);
  std::printf(
      "H3 VisualVAE precision=%s load_ms=%.1f decode_ms=%.1f tiles=%zu "
      "tile_sum_ms=%.1f tile_max_ms=%.1f weight_gib=%.2f peak_gib=%.2f "
      "frames=%zu\n",
      std::getenv("GUFO_H3_VAE_PRECISION") != nullptr
          ? std::getenv("GUFO_H3_VAE_PRECISION")
          : "default",
      telemetry.load_ms, telemetry.decode_ms, telemetry.tile_ms.size(), tiles,
      telemetry.tile_ms.empty() ? 0.0
                                : *std::ranges::max_element(telemetry.tile_ms),
      static_cast<double>(telemetry.weight_bytes) / (1U << 30U),
      static_cast<double>(telemetry.peak_live_bytes) / (1U << 30U),
      frames.frame_indices.size());
  if (argc == 6) {
    std::ofstream output(argv[5], std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(frames.rgb.data()),
        static_cast<std::streamsize>(frames.rgb.size() * sizeof(float)));
    if (!output) {
      std::cerr << "cannot write " << argv[5] << "\n";
      return 1;
    }
  }
  return 0;
}
