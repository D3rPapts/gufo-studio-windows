// Native H3 keyframe encoder against a reference posterior.
// GUFO_H3_MODEL_ROOT=<MiniMax-H3> minimax_h3_keyframe_hip_test
//     <canvas.png> <reference.moments.f32>
// The reference is tools-local/h3i2v/vae_encode_ref.py (diffusers encoder on
// the same weights, CPU fp32). Exits 77 when the model root is unset.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace h3 = gufo::minimax_h3;

int main(int argc, char** argv) {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr || argc != 3) {
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
  gufo::core::Image image;
  try {
    image = gufo::core::DecodeImage(gufo::core::ReadImageFile(argv[1]));
  } catch (const std::exception& exception) {
    std::cerr << "image: " << exception.what() << "\n";
    return 1;
  }
  const h3::KeyframeImage keyframe{.width = static_cast<int>(image.width),
                                   .height = static_cast<int>(image.height),
                                   .rgb = image.pixels};
  std::vector<h3::KeyframeMoments> moments;
  h3::VideoVaeEncoderTelemetry telemetry;
  if (!h3::EncodeKeyframeMoments(*inventory, {&keyframe, 1}, nullptr, &moments,
                                 &telemetry, &error)) {
    std::cerr << "encode: " << error << "\n";
    return 1;
  }
  std::ifstream input(argv[2], std::ios::binary);
  std::vector<float> reference(moments.front().moments.size());
  input.read(reinterpret_cast<char*>(reference.data()),
             static_cast<std::streamsize>(reference.size() * sizeof(float)));
  if (!input || input.peek() != EOF) {
    std::cerr << "reference size differs from " << reference.size()
              << " floats\n";
    return 1;
  }
  const std::vector<float>& actual = moments.front().moments;
  const std::size_t half = actual.size() / 2;
  double max_mean = 0.0;
  double max_logvar = 0.0;
  double squared = 0.0;
  double reference_squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double delta = std::fabs(static_cast<double>(actual[index]) -
                                   static_cast<double>(reference[index]));
    (index < half ? max_mean : max_logvar) =
        std::max(index < half ? max_mean : max_logvar, delta);
    if (index < half) {
      squared += delta * delta;
      reference_squared +=
          static_cast<double>(reference[index]) * reference[index];
    }
  }
  const double relative = std::sqrt(squared / reference_squared);
  std::cout << "keyframe " << image.width << "x" << image.height
            << " tiles=" << telemetry.tiles << " load_ms=" << telemetry.load_ms
            << " encode_ms=" << telemetry.encode_ms
            << " weight_bytes=" << telemetry.weight_bytes
            << " scratch_bytes=" << telemetry.scratch_bytes
            << "\nmax_abs_mean=" << max_mean << " max_abs_logvar=" << max_logvar
            << " relative_l2_mean=" << relative << "\n";
  const bool pass = relative < 1.0e-3 && max_mean < 2.0e-2;
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
