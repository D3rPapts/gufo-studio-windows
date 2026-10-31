#include "src/cli/video/video.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

void TestPresets() {
  const std::array<const char*, 7> fast = {"--model", "/models/h3", "--preset",
                                           "fast",    "--output",   "out.mp4",
                                           "A fox"};
  std::string error;
  auto options = gufo::cli::ParseVideoOptions(fast, &error);
  Expect(options.has_value(), "fast preset parses");
  Expect(options->request.parameters.internal_width == 384,
         "fast internal width");
  Expect(options->request.parameters.output_width == 512, "fast output width");
  Expect(options->request.parameters.evaluations == 19,
         "fast denoiser evaluations");
  Expect(options->request.parameters.active_blocks == 45, "fast blocks");
  Expect(options->request.parameters.reuse_interval == 2, "fast reuse");
  const std::string parameters = gufo::minimax_h3::GenerationParametersJson(
      options->request.parameters, 42,
      "df0ff96bcdb3a350f115cc39daaf0d7523814258890aa90c06fa784e3ad06169");
  Expect(parameters.find("gufo.minimax-h3-text-generation.v1") !=
             std::string::npos,
         "versioned parameter schema");
  Expect(parameters.find("\"token_reduction\": false") != std::string::npos,
         "token reduction remains disabled");

  const std::array<const char*, 7> aggressive = {
      "--model",  "/models/h3", "--preset", "aggressive",
      "--output", "out.mp4",    "A fox"};
  options = gufo::cli::ParseVideoOptions(aggressive, &error);
  Expect(options.has_value(), "aggressive preset parses");
  Expect(options->request.parameters.internal_width == 320,
         "aggressive internal width");
  Expect(options->request.parameters.active_blocks == 40, "aggressive blocks");
  Expect(options->request.parameters.reuse_interval == 3, "aggressive reuse");

  const std::array<const char*, 7> fullres = {
      "--model",  "/models/h3", "--preset",    "exact-1344x768",
      "--output", "out.mp4",    "A goalkeeper"};
  options = gufo::cli::ParseVideoOptions(fullres, &error);
  Expect(options.has_value(), "full-resolution exact preset parses");
  Expect(options->request.parameters.internal_width == 1344 &&
             options->request.parameters.internal_height == 768 &&
             options->request.parameters.frames == 124 &&
             options->request.parameters.evaluations == 49,
         "full-resolution CLI contract");
}

void TestDevelopmentAndOverrides() {
  const std::array<const char*, 9> args = {
      "--model", "/models/h3", "--preset", "dev",  "--frames-dir",
      "frames",  "--seed",     "123",      "A fox"};
  std::string error;
  auto options = gufo::cli::ParseVideoOptions(args, &error);
  Expect(options.has_value(), "development preset parses");
  Expect(!options->request.parameters.mux, "development skips mux");
  Expect(!options->request.parameters.decode_audio, "development skips audio");
  Expect(options->request.parameters.selected_frames ==
             std::vector<int>({0, 11, 21}),
         "development selected frames");
  Expect(options->request.seed == 123, "seed override");

  const std::array<const char*, 19> custom = {"--model",
                                              "/models/h3",
                                              "--preset",
                                              "dev",
                                              "--frames-dir",
                                              "frames",
                                              "--latents-dir",
                                              "latents",
                                              "--width",
                                              "320",
                                              "--height",
                                              "320",
                                              "--steps",
                                              "7",
                                              "--blocks",
                                              "48",
                                              "--attention-kernel",
                                              "scalar",
                                              "A fox"};
  options = gufo::cli::ParseVideoOptions(custom, &error);
  Expect(options.has_value(), "custom controls parse");
  Expect(options->request.parameters.internal_width == 320, "width override");
  Expect(options->request.parameters.evaluations == 7, "step override");
  Expect(options->request.parameters.active_blocks == 48, "block override");
  Expect(options->request.latents_directory == "latents",
         "latent diagnostic output override");
  Expect(!options->request.parameters.row_parallel_attention,
         "scalar attention fallback override");
}

void TestSelectedFramesAndFailures() {
  const std::array<const char*, 9> selected = {
      "--model",      "/models/h3", "--preset",          "dev",
      "--frames-dir", "frames",     "--selected-frames", "last,first,middle",
      "A fox"};
  std::string error;
  auto options = gufo::cli::ParseVideoOptions(selected, &error);
  Expect(options.has_value(), "symbolic selected frames parse");
  Expect(options->request.parameters.selected_frames ==
             std::vector<int>({0, 11, 21}),
         "selected frames normalize");

  // --first-frame/--last-frame are FL2VA keyframe inputs; an unreadable image
  // is reported as such instead of being an unknown option.
  const std::array<const char*, 5> missing = {
      "--model", "/models/h3", "--first-frame",
      "gufo-video-cli-test-missing.png", "A fox"};
  error.clear();
  Expect(!gufo::cli::ParseVideoOptions(missing, &error).has_value(),
         "missing first-frame image rejected");
  Expect(error.find("cannot read keyframe") != std::string::npos,
         "missing first-frame image names the keyframe");

  const std::array<const char*, 4> no_audio = {"--model", "/models/h3",
                                               "--no-audio", "A fox"};
  Expect(!gufo::cli::ParseVideoOptions(no_audio, &error).has_value(),
         "audio cannot be disabled while muxing");
}

void TestKeyframes() {
  // 2x1 RGB PNG: red, blue.
  constexpr std::array<unsigned char, 70> kPng = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
      0x08, 0x02, 0x00, 0x00, 0x00, 0x7b, 0x40, 0xe8, 0xdd, 0x00, 0x00, 0x00,
      0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0x00, 0x04,
      0xff, 0x01, 0x07, 0x00, 0x01, 0xff, 0xe2, 0x23, 0x9e, 0x59, 0x00, 0x00,
      0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
  const std::filesystem::path png =
      std::filesystem::temp_directory_path() / "gufo-video-cli-test-key.png";
  {
    std::ofstream output(png, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(kPng.data()), kPng.size());
    Expect(static_cast<bool>(output), "keyframe fixture written");
  }
  const std::string path = png.string();
  const std::array<const char*, 9> args = {
      "--model",      "/models/h3",   "--output", "out.mp4", "--first-frame",
      path.c_str(),   "--last-frame", path.c_str(), "A fox"};
  std::string error;
  const auto options = gufo::cli::ParseVideoOptions(args, &error);
  std::error_code ignored;
  std::filesystem::remove(png, ignored);
  Expect(options.has_value(), "first and last keyframes parse");
  const auto& keyframes = options->request.keyframes;
  Expect(keyframes.size() == 2, "two keyframes");
  Expect(keyframes[0].anchor == gufo::minimax_h3::KeyframeAnchor::kFirst &&
             keyframes[1].anchor == gufo::minimax_h3::KeyframeAnchor::kLast,
         "keyframe anchors");
  Expect(keyframes[0].width == 2 && keyframes[0].height == 1 &&
             keyframes[0].rgb ==
                 std::vector<std::uint8_t>({255, 0, 0, 0, 0, 255}),
         "keyframe pixels decoded");
}

void TestSharedOptionParsing() {
  std::string error;
  const auto options = gufo::cli::ParseVideoOptions(
      std::array{"--model", "--preset", "--steps=7", "--preset=dev",
                 "--frames-dir=frames", "--profile", "A fox"},
      &error);
  Expect(options.has_value(), "inline options and flag-like values parse");
  Expect(options->request.model_root == "--preset",
         "option values are not scanned as flags");
  Expect(options->request.parameters.evaluations == 7,
         "explicit overrides win regardless of preset argument order");
  Expect(options->profile, "documented profile option is wired");

  const auto help_value = gufo::cli::ParseVideoOptions(
      std::array{"--model", "--help", "--preset=dev", "--frames-dir=frames",
                 "A fox"},
      &error);
  Expect(help_value && help_value->request.model_root == "--help",
         "help used as an option value does not trigger help");
  Expect(
      !gufo::cli::ParseVideoOptions(std::array{"--selected-frames="}, &error) &&
          !error.empty(),
      "explicit empty frame selection is rejected");
  Expect(!gufo::cli::ParseVideoOptions(std::array{"--steps=7junk"}, &error) &&
             !error.empty(),
         "partial integers are rejected");
  Expect(!gufo::cli::ParseVideoOptions(std::array{"--help"}, &error) &&
             error.empty(),
         "actual help request succeeds without requiring a model");
}

}  // namespace

int main() {
  TestPresets();
  TestDevelopmentAndOverrides();
  TestSelectedFramesAndFailures();
  TestKeyframes();
  TestSharedOptionParsing();
  std::cout << "All MiniMax H3 video CLI tests passed.\n";
  return 0;
}
