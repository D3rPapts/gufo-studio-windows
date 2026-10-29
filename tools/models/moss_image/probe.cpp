// Runs one MOSS-Image generation and writes every observed tensor as raw FP32
// (<name>.f32) next to image.png, for comparison with the PyTorch reference.
//
//   moss_image_probe MODEL_DIR TEXT_ENCODER.gguf OUTPUT_DIR [PROMPT] [SIZE]
//                    [STEPS] [--blocks]
//
// MOSS_IMAGE_NOISE replays the reference noise. Without --blocks only the
// stage outputs are written, not the 42 DiT block outputs per forward.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "src/models/moss_image/model.hpp"

int main(int argc, char** argv) {
  using namespace gufo::models::moss_image;
  if (argc < 4) {
    std::cerr << "usage: moss_image_probe MODEL_DIR TEXT_ENCODER.gguf "
                 "OUTPUT_DIR [PROMPT] [SIZE] [STEPS] [--blocks]\n";
    return 2;
  }
  try {
    const auto output = std::filesystem::path(argv[3]);
    std::filesystem::create_directories(output);
    Model model(argv[1], Options{.text_encoder = argv[2]});
    Request request;
    request.prompt =
        argc > 4 ? argv[4] : "A red fox sitting in fresh snow, golden hour light";
    request.width = request.height = argc > 5 ? std::stoi(argv[5]) : 256;
    request.steps = argc > 6 ? std::stoi(argv[6]) : 4;
    request.seed = 1;
    const bool blocks = argc > 7 && std::string_view(argv[7]) == "--blocks";
    Observer observer;
    observer.include = [blocks](std::string_view name) {
      return blocks || name.find(".block.") == std::string_view::npos;
    };
    observer.write = [&](std::string_view name, std::span<const float> values,
                         int rows, int cols) {
      std::ofstream stream(output / (std::string(name) + ".f32"),
                           std::ios::binary);
      stream.write(reinterpret_cast<const char*>(values.data()),
                   static_cast<std::streamsize>(values.size_bytes()));
      std::cout << name << ' ' << rows << 'x' << cols << '\n' << std::flush;
    };
    const auto result = model.Generate(request, {}, observer);
    const auto png = gufo::models::qwen_image_21::EncodePng(result.image);
    std::ofstream image(output / "image.png", std::ios::binary);
    image.write(reinterpret_cast<const char*>(png.data()),
                static_cast<std::streamsize>(png.size()));
    std::cout << "total_ms=" << result.metrics.total_ms
              << " prompt_ms=" << result.metrics.prompt_ms
              << " denoise_ms=" << result.metrics.denoise_ms
              << " vae_ms=" << result.metrics.vae_ms << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
