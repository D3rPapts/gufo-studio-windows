// gufo-wan-dit: one forward of the Wan 2.2 TI2V-5B transformer.
//
//   gufo-wan-dit --model <Diffusers root> --oracle DIR [--runs N]
//   gufo-wan-dit --model <Diffusers root> --bench FRAMES HEIGHT WIDTH
//
// With `--oracle`, DIR holds the inputs and outputs of one Diffusers call
// (tools-local/wan_dit_oracle.py): latent, timestep and text are fed to the
// transformer; noise_pred and the intermediates are compared by cosine
// similarity and maximum absolute error (noise_pred also per frame).
// `--bench` times one forward of a latent of the given size (zeros, one
// timestep, 512 zero text rows).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/wan22/transformer.hpp"

namespace {

namespace wan22 = gufo::models::wan22;
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

// A little-endian, C-order FP32 .npy array.
bool ReadNpy(const std::filesystem::path& path, std::vector<float>* values,
             std::vector<std::size_t>* shape) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    return false;
  std::uint32_t header_size = 0;
  if (magic[6] == 1) {
    std::uint16_t size16 = 0;
    in.read(reinterpret_cast<char*>(&size16), 2);
    header_size = size16;
  } else {
    in.read(reinterpret_cast<char*>(&header_size), 4);
  }
  std::string header(header_size, '\0');
  if (!in.read(header.data(), header_size) ||
      header.find("'descr': '<f4'") == std::string::npos)
    return false;
  // np.save writes a transposed (non-contiguous) tensor in Fortran order.
  const bool fortran =
      header.find("'fortran_order': True") != std::string::npos;
  const auto open = header.find('(');
  const auto close = header.find(')');
  if (open == std::string::npos || close == std::string::npos)
    return false;
  shape->clear();
  std::size_t count = 1;
  for (std::size_t pos = open + 1; pos < close;) {
    const auto end = std::min(header.find(',', pos), close);
    const std::string item = header.substr(pos, end - pos);
    if (item.find_first_of("0123456789") != std::string::npos) {
      shape->push_back(std::stoull(item));
      count *= shape->back();
    }
    pos = end + 1;
  }
  values->resize(count);
  if (!in.read(reinterpret_cast<char*>(values->data()),
               static_cast<std::streamsize>(count * 4)))
    return false;
  if (!fortran)
    return true;
  if (shape->size() != 2)
    return false;
  const std::size_t rows = (*shape)[0], columns = (*shape)[1];
  std::vector<float> transposed(count);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < columns; ++c)
      transposed[r * columns + c] = (*values)[c * rows + r];
  values->swap(transposed);
  return true;
}

struct Comparison {
  double cosine{0};
  double max_error{0};
};

Comparison Compare(const float* a, const float* b, std::size_t count) {
  double dot = 0, a2 = 0, b2 = 0, error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    dot += double{a[i]} * b[i];
    a2 += double{a[i]} * a[i];
    b2 += double{b[i]} * b[i];
    error = std::max(error, std::abs(double{a[i]} - b[i]));
  }
  return {dot / std::sqrt(a2 * b2), error};
}

int Usage() {
  std::cerr << "usage: gufo-wan-dit --model DIFFUSERS_ROOT "
               "(--oracle DIR [--runs N] | --bench FRAMES HEIGHT WIDTH)\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string oracle;
  int runs = 2;
  int bench[3]{};
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--model" && i + 1 < argc)
      model = argv[++i];
    else if (name == "--oracle" && i + 1 < argc)
      oracle = argv[++i];
    else if (name == "--runs" && i + 1 < argc)
      runs = std::max(1, std::atoi(argv[++i]));
    else if (name == "--bench" && i + 3 < argc)
      for (int& value : bench)
        value = std::atoi(argv[++i]);
    else
      return Usage();
  }
  if (model.empty() || oracle.empty() == (bench[0] == 0))
    return Usage();

  wan22::TransformerInput input;
  std::vector<float> latent, timestep, text;
  std::vector<std::size_t> shape;
  const std::filesystem::path dir(oracle);
  if (!oracle.empty()) {
    std::vector<std::size_t> text_shape;
    if (!ReadNpy(dir / "latent.npy", &latent, &shape) || shape.size() != 4 ||
        shape[0] != wan22::Transformer::kChannels ||
        !ReadNpy(dir / "timestep.npy", &timestep, &text_shape) ||
        !ReadNpy(dir / "text.npy", &text, &text_shape) ||
        text_shape.size() != 2 ||
        text_shape[1] != wan22::Transformer::kTextWidth) {
      std::cerr << "cannot read oracle inputs in " << oracle << "\n";
      return 1;
    }
    input.frames = static_cast<int>(shape[1]);
    input.height = static_cast<int>(shape[2]);
    input.width = static_cast<int>(shape[3]);
    input.text_rows = static_cast<int>(text_shape[0]);
  } else {
    input.frames = bench[0];
    input.height = bench[1];
    input.width = bench[2];
    latent.assign(static_cast<std::size_t>(wan22::Transformer::kChannels) *
                      input.frames * input.height * input.width,
                  0.0F);
    timestep = {999.0F};
    input.text_rows = 512;
    text.assign(static_cast<std::size_t>(512) * wan22::Transformer::kTextWidth,
                0.0F);
  }
  input.latent = latent;
  input.timesteps = timestep;
  input.text = text;
  const int tokens = input.frames * (input.height / 2) * (input.width / 2);
  std::cout << "latent " << input.frames << " x " << input.height << " x "
            << input.width << ", " << tokens << " tokens, " << timestep.size()
            << " timesteps (first " << timestep[0] << ")\n";

  std::size_t free_before = 0, total = 0;
  (void)hipMemGetInfo(&free_before, &total);
  std::string error;
  auto start = Clock::now();
  auto transformer = wan22::Transformer::Load(
      std::filesystem::path(model) / "transformer", &error);
  if (!transformer) {
    std::cerr << "load: " << error << "\n";
    return 1;
  }
  std::cout << "load " << Seconds(start) << " s, weights "
            << transformer->weight_bytes() / 1048576.0 << " MiB\n";

  wan22::TransformerProbes probes;
  std::vector<float> noise;
  for (int run = 0; run < runs; ++run) {
    start = Clock::now();
    const bool probing = !oracle.empty() && run == 0;
    if (!transformer->Forward(input, &noise, probing ? &probes : nullptr,
                              &error)) {
      std::cerr << "forward: " << error << "\n";
      return 1;
    }
    std::cout << "forward run " << run << (probing ? " (probes)" : "") << ": "
              << Seconds(start) * 1000 << " ms (blocks "
              << transformer->last_blocks_ms() << " ms, self-attention "
              << transformer->last_attention_ms() << " ms)\n";
  }
  std::size_t free_after = 0;
  (void)hipMemGetInfo(&free_after, &total);
  std::cout << "activations " << transformer->activation_bytes() / 1048576.0
            << " MiB; device memory in use by the process ~"
            << (free_before - std::min(free_before, free_after)) / 1048576.0
            << " MiB\n";
  if (oracle.empty())
    return 0;

  bool pass = true;
  const auto check = [&](const char* name, const std::vector<float>& ours,
                         bool gate) {
    std::vector<float> reference;
    std::vector<std::size_t> reference_shape;
    if (!ReadNpy(dir / (std::string(name) + ".npy"), &reference,
                 &reference_shape) ||
        reference.size() != ours.size()) {
      std::cout << name << ": cannot compare (size " << ours.size() << " vs "
                << reference.size() << ")\n";
      pass = false;
      return;
    }
    const Comparison c = Compare(ours.data(), reference.data(), ours.size());
    std::cout << std::defaultfloat << std::setprecision(7) << name
              << ": cosine " << c.cosine << ", max abs " << c.max_error << "\n";
    if (gate && !(c.cosine >= 0.999))
      pass = false;
  };
  check("patch", probes.patch, false);
  check("temb", probes.temb, false);
  check("timestep_proj", probes.timestep_proj, false);
  check("text_states", probes.text_states, false);
  check("block0_attn1", probes.block0_attn1, false);
  check("block0_attn2", probes.block0_attn2, false);
  check("block0_ffn", probes.block0_ffn, false);
  check("block0", probes.block0, false);
  check("block_last", probes.block_last, false);
  check("noise_pred", noise, true);

  std::vector<float> reference;
  if (ReadNpy(dir / "noise_pred.npy", &reference, &shape) &&
      reference.size() == noise.size()) {
    const std::size_t plane =
        static_cast<std::size_t>(input.height) * input.width;
    double min_cosine = 1;
    for (int f = 0; f < input.frames; ++f) {
      std::vector<float> a, b;
      for (int c = 0; c < wan22::Transformer::kChannels; ++c) {
        const std::size_t offset =
            (static_cast<std::size_t>(c) * input.frames + f) * plane;
        a.insert(a.end(), noise.begin() + offset,
                 noise.begin() + offset + plane);
        b.insert(b.end(), reference.begin() + offset,
                 reference.begin() + offset + plane);
      }
      const Comparison c = Compare(a.data(), b.data(), a.size());
      std::cout << std::defaultfloat << std::setprecision(7)
                << "noise_pred frame " << f << ": cosine " << c.cosine
                << ", max abs " << c.max_error << "\n";
      min_cosine = std::min(min_cosine, c.cosine);
    }
    if (!(min_cosine >= 0.999))
      pass = false;
  }
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
