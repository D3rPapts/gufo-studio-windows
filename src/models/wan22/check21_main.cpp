// gufo-wan-check21: oracle checks of the Wan 2.1 image-to-video parts.
//
//   gufo-wan-check21 --model ROOT --clip-oracle DIR
//   gufo-wan-check21 --dit FILE [--lora FILE] --clip DIR --dit-oracle DIR
//   gufo-wan-check21 --dit FILE [--lora FILE] --clip DIR --dit-oracle DIR
//                    --dit-bench FRAMES H W
//
// `--dit-oracle` runs the 14B transformer on DIR/x.npy with DIR/text.npy and
// the CLIP states (--clip DIR)/clip_states.npy at timestep 875 and compares
// the embeddings, block 0 and the output with tools-local/wan21_dit_oracle.py.
// With `--dit-bench` it times Forward on random input of the given latent
// shape instead (21 60 104 = 81 frames of 832x480), the context from DIR.
// Both refuse to load with less than 8 GiB of free host RAM (the
// 14B path needs ~1.1 GiB of host memory).
//
//   gufo-wan-check21 --i2v OUT.mp4 --model ROOT --t5 ROOT --dit FILE
//                    [--lora FILE] --image FILE --prompt TEXT [--width W]
//                    [--height H] [--frames N] [--steps S] [--seed N]
//                    [--uni3c FILE --guide-raw FILE.rgb [--uni3c-strength S]]
//                    [--dump DIR]
//
//   gufo-wan-check21 --dit FILE --lora FILE --clip DIR --dit-oracle DIR
//                    --uni3c FILE --uni3c-oracle DIR
//
// `--uni3c-oracle` runs the DiT input of the DiT oracle with the Uni3C
// ControlNet on DIR/guide.npy and compares the residuals of control blocks 0
// and 19 with tools-local/wan21_uni3c_oracle.py; it also reports how much the
// control changes the output.
//
// `--i2v` runs the whole Wan 2.1 image-to-video pipeline (pipeline21.hpp);
// --guide-raw is rgb24 frames at the video size (ffmpeg -f rawvideo);
// --dump writes the pipeline intermediates as .npy for
// tools-local/wan21_pipe_oracle.py.
//
// `--clip-oracle` runs the CLIP vision encoder on DIR/clip_pixels.npy
// [3, 224, 224] and compares with DIR/clip_states.npy [257, 1280]; it also
// preprocesses DIR/clip_image.npy (RGB as FP32 0..255, [H, W, 3]) and
// compares the pixels with clip_pixels.npy.

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/wan22/clip_vision.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/pipeline21.hpp"
#include "src/models/wan22/transformer21.hpp"
#include "src/models/wan22/uni3c.hpp"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// psapi.h needs windows.h first.
#include <psapi.h>

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
      header.find("'descr': '<f4'") == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos)
    return false;
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
  return static_cast<bool>(in.read(reinterpret_cast<char*>(values->data()),
                                   static_cast<std::streamsize>(count * 4)));
}

// Writes a C-order FP32 .npy array.
bool WriteNpy(const std::filesystem::path& path,
              const std::vector<float>& values,
              const std::vector<std::size_t>& shape) {
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (std::size_t dim : shape)
    header += std::to_string(dim) + ", ";
  header += "), }";
  while ((10 + header.size() + 1) % 64 != 0)
    header += ' ';
  header += '\n';
  std::ofstream out(path, std::ios::binary);
  const std::uint16_t size = static_cast<std::uint16_t>(header.size());
  out.write("\x93NUMPY\x01\x00", 8);
  out.write(reinterpret_cast<const char*>(&size), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * 4));
  return static_cast<bool>(out);
}

struct Comparison {
  double cosine{0};
  double max_error{0};
  double mean_error{0};
};

Comparison Compare(const float* a, const float* b, std::size_t count) {
  double dot = 0, a2 = 0, b2 = 0, error = 0, sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    dot += double{a[i]} * b[i];
    a2 += double{a[i]} * a[i];
    b2 += double{b[i]} * b[i];
    const double e = std::abs(double{a[i]} - b[i]);
    error = std::max(error, e);
    sum += e;
  }
  return {dot / std::sqrt(a2 * b2), error,
          sum / std::max<std::size_t>(count, 1)};
}

void Print(const std::string& name, const Comparison& c) {
  std::cout << std::defaultfloat << std::setprecision(7) << name << ": cosine "
            << c.cosine << ", max abs " << c.max_error << ", mean abs "
            << c.mean_error << "\n";
}

int ClipOracle(const std::filesystem::path& model,
               const std::filesystem::path& dir) {
  std::vector<float> pixels, reference, image;
  std::vector<std::size_t> shape, reference_shape, image_shape;
  if (!ReadNpy(dir / "clip_pixels.npy", &pixels, &shape) ||
      !ReadNpy(dir / "clip_states.npy", &reference, &reference_shape)) {
    std::cerr << "cannot read CLIP oracle in " << dir << "\n";
    return 1;
  }
  std::string error;
  auto start = Clock::now();
  auto clip = wan22::ClipVision::Load(model / "image_encoder", &error);
  if (!clip) {
    std::cerr << "clip load: " << error << "\n";
    return 1;
  }
  std::cout << "clip load " << Seconds(start) << " s, weights "
            << clip->weight_bytes() / 1048576.0 << " MiB\n";
  std::vector<float> states;
  for (int run = 0; run < 2; ++run) {
    start = Clock::now();
    if (!clip->Encode(pixels, &states, &error)) {
      std::cerr << "clip encode: " << error << "\n";
      return 1;
    }
    std::cout << "clip encode run " << run << ": " << Seconds(start) * 1000
              << " ms\n";
  }
  if (states.size() != reference.size()) {
    std::cerr << "size mismatch\n";
    return 1;
  }
  const Comparison c = Compare(states.data(), reference.data(), states.size());
  Print("clip_states", c);
  WriteNpy(dir / "clip_states_gufo.npy", states,
           {static_cast<std::size_t>(wan22::ClipVision::kTokens),
            static_cast<std::size_t>(wan22::ClipVision::kWidth)});
  double worst = 1;
  for (int t = 0; t < wan22::ClipVision::kTokens; ++t) {
    const std::size_t offset =
        static_cast<std::size_t>(t) * wan22::ClipVision::kWidth;
    worst = std::min(worst,
                     Compare(states.data() + offset, reference.data() + offset,
                             wan22::ClipVision::kWidth)
                         .cosine);
  }
  std::cout << "worst token cosine " << worst << "\n";
  // torch BF16 reaches only 0.964 overall here (some tokens below 0), so
  // single tokens near 0.9 are within BF16 noise.
  bool pass = c.cosine >= 0.999 && worst >= 0.9;
  if (ReadNpy(dir / "clip_image.npy", &image, &image_shape) &&
      image_shape.size() == 3 && image_shape[2] == 3) {
    std::vector<std::uint8_t> rgb(image.size());
    for (std::size_t i = 0; i < image.size(); ++i)
      rgb[i] = static_cast<std::uint8_t>(std::lround(image[i]));
    const auto ours =
        wan22::ClipVision::Preprocess(rgb, static_cast<int>(image_shape[1]),
                                      static_cast<int>(image_shape[0]));
    Print("preprocess", Compare(ours.data(), pixels.data(), ours.size()));
    std::vector<float> end_to_end;
    if (clip->Encode(ours, &end_to_end, &error))
      Print("clip_states from our preprocessing",
            Compare(end_to_end.data(), reference.data(), end_to_end.size()));
  }
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

double AvailableGiB() {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return static_cast<double>(status.ullAvailPhys) / (1ULL << 30U);
}

double PeakWorkingSetGiB() {
  PROCESS_MEMORY_COUNTERS counters{};
  GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
  return static_cast<double>(counters.PeakWorkingSetSize) / (1ULL << 30U);
}

double DeviceUsedGiB() {
  std::size_t free_bytes = 0, total = 0;
  (void)hipMemGetInfo(&free_bytes, &total);
  return static_cast<double>(total - free_bytes) / (1ULL << 30U);
}

std::unique_ptr<wan22::Transformer21> LoadDit(
    const std::filesystem::path& dit, const std::filesystem::path& lora) {
  const double available = AvailableGiB();
  std::cout << "host RAM available " << available << " GiB, device used "
            << DeviceUsedGiB() << " GiB\n";
  if (available < 8.0) {
    std::cerr << "less than 8 GiB host RAM available, not loading\n";
    return nullptr;
  }
  std::string error;
  const auto start = Clock::now();
  auto model = wan22::Transformer21::Load(dit, lora, 1.0F, &error);
  if (!model) {
    std::cerr << "dit load: " << error << "\n";
    return nullptr;
  }
  std::cout << "dit load " << Seconds(start) << " s, weights "
            << model->weight_bytes() / 1073741824.0 << " GiB, device used "
            << DeviceUsedGiB() << " GiB, peak working set "
            << PeakWorkingSetGiB() << " GiB\n";
  return model;
}

bool PrepareContext(wan22::Transformer21& model,
                    const std::filesystem::path& dir,
                    const std::filesystem::path& clip_dir,
                    std::vector<float>* context_text,
                    std::vector<float>* context_image) {
  std::vector<float> text, image;
  std::vector<std::size_t> shape, image_shape;
  if (!ReadNpy(dir / "text.npy", &text, &shape) || shape.size() != 2 ||
      !ReadNpy(clip_dir / "clip_states.npy", &image, &image_shape)) {
    std::cerr << "cannot read text.npy / clip_states.npy\n";
    return false;
  }
  std::string error;
  const auto start = Clock::now();
  if (!model.PrepareContext(0, text, static_cast<int>(shape[0]), image,
                            context_text, context_image, &error)) {
    std::cerr << "context: " << error << "\n";
    return false;
  }
  std::cout << "context " << Seconds(start) * 1000 << " ms\n";
  return true;
}

int DitOracle(const std::filesystem::path& dit,
              const std::filesystem::path& lora,
              const std::filesystem::path& dir,
              const std::filesystem::path& clip_dir) {
  std::vector<float> x;
  std::vector<std::size_t> shape;
  if (!ReadNpy(dir / "x.npy", &x, &shape) || shape.size() != 4 ||
      shape[0] != 36) {
    std::cerr << "cannot read x.npy\n";
    return 1;
  }
  auto model = LoadDit(dit, lora);
  if (!model)
    return 1;
  std::vector<float> context_text, context_image;
  if (!PrepareContext(*model, dir, clip_dir, &context_text, &context_image))
    return 1;
  wan22::Transformer21Input input;
  input.latent = x;
  input.frames = static_cast<int>(shape[1]);
  input.height = static_cast<int>(shape[2]);
  input.width = static_cast<int>(shape[3]);
  input.timestep = 875.0F;
  std::vector<float> noise;
  wan22::Transformer21Probes probes;
  std::string error;
  const auto start = Clock::now();
  if (!model->Forward(input, &noise, &probes, &error)) {
    std::cerr << "forward: " << error << "\n";
    return 1;
  }
  std::cout << "forward " << Seconds(start) << " s\n";
  bool pass = true;
  const auto compare = [&](const char* name, const std::vector<float>& ours,
                           double minimum) {
    std::vector<float> reference;
    std::vector<std::size_t> reference_shape;
    if (!ReadNpy(dir / (std::string(name) + ".npy"), &reference,
                 &reference_shape) ||
        reference.size() != ours.size()) {
      std::cerr << "missing or misshaped " << name << "\n";
      pass = false;
      return;
    }
    const Comparison c = Compare(ours.data(), reference.data(), ours.size());
    Print(name, c);
    pass = pass && c.cosine >= minimum;
  };
  compare("temb", probes.temb, 0.9999);
  compare("context_text", context_text, 0.999);
  compare("context_img", context_image, 0.999);
  compare("block0", probes.block0, 0.999);
  compare("noise", noise, 0.995);
  std::cout << "device used " << DeviceUsedGiB() << " GiB, peak working set "
            << PeakWorkingSetGiB() << " GiB\n";
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

int DitBench(const std::filesystem::path& dit,
             const std::filesystem::path& lora,
             const std::filesystem::path& dir,
             const std::filesystem::path& clip_dir, int frames, int height,
             int width) {
  auto model = LoadDit(dit, lora);
  if (!model || !PrepareContext(*model, dir, clip_dir, nullptr, nullptr))
    return 1;
  std::vector<float> x(static_cast<std::size_t>(36) * frames * height * width);
  std::mt19937 random(7);
  std::normal_distribution<float> normal;
  for (float& value : x)
    value = normal(random);
  wan22::Transformer21Input input;
  input.latent = x;
  input.frames = frames;
  input.height = height;
  input.width = width;
  input.timestep = 875.0F;
  std::vector<float> noise;
  std::string error;
  for (int run = 0; run < 2; ++run) {
    const auto start = Clock::now();
    if (!model->Forward(input, &noise, nullptr, &error)) {
      std::cerr << "forward: " << error << "\n";
      return 1;
    }
    std::cout << "forward " << frames << "x" << height << "x" << width
              << " (tokens " << frames * (height / 2) * (width / 2) << ") run "
              << run << ": " << Seconds(start) << " s, blocks "
              << model->last_blocks_ms() / 1000 << " s, self-attention "
              << model->last_attention_ms() / 1000 << " s, activations "
              << model->activation_bytes() / 1073741824.0
              << " GiB, device used " << DeviceUsedGiB() << " GiB\n";
  }
  double sum = 0;
  for (float value : noise)
    sum += static_cast<double>(value) * value;
  std::cout << "output rms " << std::sqrt(sum / noise.size())
            << ", peak working set " << PeakWorkingSetGiB() << " GiB\n";
  return std::isfinite(sum) ? 0 : 1;
}

int Uni3COracle(const std::filesystem::path& dit,
                const std::filesystem::path& lora,
                const std::filesystem::path& dit_dir,
                const std::filesystem::path& clip_dir,
                const std::filesystem::path& uni3c_file,
                const std::filesystem::path& dir) {
  std::vector<float> x, guide, plain_ref;
  std::vector<std::size_t> shape, guide_shape, ref_shape;
  if (!ReadNpy(dit_dir / "x.npy", &x, &shape) || shape.size() != 4 ||
      !ReadNpy(dir / "guide.npy", &guide, &guide_shape) ||
      !ReadNpy(dit_dir / "noise.npy", &plain_ref, &ref_shape)) {
    std::cerr << "cannot read x.npy / guide.npy / noise.npy\n";
    return 1;
  }
  auto model = LoadDit(dit, lora);
  if (!model || !PrepareContext(*model, dit_dir, clip_dir, nullptr, nullptr))
    return 1;
  std::string error;
  auto uni3c = wan22::Uni3C::Load(uni3c_file, model->stream(), &error);
  const int frames = static_cast<int>(shape[1]);
  const int height = static_cast<int>(shape[2]);
  const int width = static_cast<int>(shape[3]);
  if (!uni3c || !uni3c->SetGuide(guide, frames, height, width, &error)) {
    std::cerr << "uni3c: " << error << "\n";
    return 1;
  }
  std::cout << "uni3c weights " << uni3c->weight_bytes() / 1073741824.0
            << " GiB\n";
  uni3c->set_probe(true);
  wan22::Transformer21Input input;
  input.latent = x;
  input.frames = frames;
  input.height = height;
  input.width = width;
  input.timestep = 875.0F;
  std::vector<float> plain, controlled;
  if (!model->Forward(input, &plain, nullptr, &error)) {
    std::cerr << "forward: " << error << "\n";
    return 1;
  }
  input.hook = uni3c.get();
  const auto start = Clock::now();
  if (!model->Forward(input, &controlled, nullptr, &error)) {
    std::cerr << "forward with uni3c: " << error << "\n";
    return 1;
  }
  std::cout << "forward with uni3c " << Seconds(start) << " s\n";
  bool pass = true;
  const auto compare = [&](const char* name, const std::vector<float>& ours) {
    std::vector<float> reference;
    std::vector<std::size_t> reference_shape;
    if (!ReadNpy(dir / (std::string(name) + ".npy"), &reference,
                 &reference_shape) ||
        reference.size() != ours.size()) {
      std::cerr << "missing or misshaped " << name << "\n";
      pass = false;
      return;
    }
    const Comparison c = Compare(ours.data(), reference.data(), ours.size());
    Print(name, c);
    pass = pass && c.cosine >= 0.999;
  };
  compare("residual0", uni3c->residual_first());
  compare("residual19", uni3c->residual_last());
  Print("noise without control vs DiT oracle",
        Compare(plain.data(), plain_ref.data(), plain.size()));
  Print("noise with vs without control",
        Compare(controlled.data(), plain.data(), plain.size()));
  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

int RunI2V(const wan22::I2V21Request& base, const std::filesystem::path& out,
           const std::filesystem::path& image_path,
           const std::filesystem::path& guide_path,
           const std::filesystem::path& dump_dir) {
  wan22::I2V21Request request = base;
  std::vector<std::uint8_t> first_frame, guide;
  try {
    const auto image =
        gufo::core::DecodeImage(gufo::core::ReadImageFile(image_path));
    first_frame = wan22::FitImage(image.pixels, static_cast<int>(image.width),
                                  static_cast<int>(image.height), request.width,
                                  request.height);
  } catch (const std::exception& e) {
    std::cerr << "cannot read " << image_path << ": " << e.what() << "\n";
    return 1;
  }
  request.first_frame = first_frame;
  if (!guide_path.empty()) {
    std::ifstream in(guide_path, std::ios::binary | std::ios::ate);
    const std::size_t frame =
        static_cast<std::size_t>(request.width) * request.height * 3;
    const auto size = static_cast<std::size_t>(in.tellg());
    if (!in || size == 0 || size % frame != 0) {
      std::cerr << "guide " << guide_path << " is not rgb24 frames of "
                << request.width << "x" << request.height << "\n";
      return 1;
    }
    guide.resize(size);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(guide.data()),
            static_cast<std::streamsize>(size));
    request.guide = guide;
    request.guide_frames = static_cast<int>(size / frame);
    std::cout << "guide " << guide_path << ": " << request.guide_frames
              << " frames\n";
  }
  if (AvailableGiB() < 8.0) {
    std::cerr << "less than 8 GiB host RAM available (" << AvailableGiB()
              << "), not running\n";
    return 1;
  }
  wan22::I2V21Callbacks callbacks;
  if (!dump_dir.empty()) {
    std::filesystem::create_directories(dump_dir);
    std::vector<float> frame(first_frame.begin(), first_frame.end());
    WriteNpy(dump_dir / "first_frame.npy", frame,
             {static_cast<std::size_t>(request.height),
              static_cast<std::size_t>(request.width), 3});
    callbacks.dump = [&](const std::string& name,
                         const std::vector<float>& values,
                         const std::vector<std::size_t>& shape) {
      WriteNpy(dump_dir / (name + ".npy"), values, shape);
    };
  }
  callbacks.progress = [](const std::string& phase, int done, int total) {
    std::cout << "  " << phase << " " << done << "/" << total << "\n"
              << std::flush;
  };
  wan22::DecodedVideo video;
  wan22::I2V21Telemetry tel;
  std::string error;
  if (!wan22::GenerateI2V21(request, callbacks, nullptr, &video, &tel,
                            &error)) {
    std::cerr << "i2v: " << error << "\n";
    return 1;
  }
  std::cout << "text " << tel.text_ms / 1000 << " s, clip "
            << tel.clip_ms / 1000 << " s, condition " << tel.condition_ms / 1000
            << " s, guide " << tel.guide_ms / 1000 << " s, dit load "
            << tel.dit_load_ms / 1000 << " s, context " << tel.context_ms / 1000
            << " s, decode " << tel.decode_ms / 1000 << " s, total "
            << tel.total_ms / 1000 << " s\n";
  for (std::size_t i = 0; i < tel.step_ms.size(); ++i)
    std::cout << "step " << i << ": " << tel.step_ms[i] / 1000 << " s\n";
  std::cout << "peak device " << tel.peak_device_bytes / 1073741824.0
            << " GiB, peak working set " << PeakWorkingSetGiB() << " GiB\n";
  if (!wan22::WriteMp4(out, video, 16, &error)) {
    std::cerr << "mp4: " << error << "\n";
    return 1;
  }
  std::cout << "wrote " << out << " (" << video.frames << " frames "
            << video.width << "x" << video.height << ")\n";
  return 0;
}

int Usage() {
  std::cerr << "usage: gufo-wan-check21 --model ROOT --clip-oracle DIR\n"
               "       gufo-wan-check21 --dit FILE [--lora FILE] --clip DIR "
               "--dit-oracle DIR [--dit-bench FRAMES H W]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model, clip_oracle, dit, lora, dit_oracle, clip_dir;
  int bench[3]{};
  wan22::I2V21Request i2v;
  std::filesystem::path i2v_out, image, guide_raw, dump_dir, uni3c_oracle;
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    const bool more = i + 1 < argc;
    if (name == "--model" && more)
      model = argv[++i];
    else if (name == "--clip-oracle" && more)
      clip_oracle = argv[++i];
    else if (name == "--dit" && more)
      dit = argv[++i];
    else if (name == "--lora" && more)
      lora = argv[++i];
    else if (name == "--dit-oracle" && more)
      dit_oracle = argv[++i];
    else if (name == "--clip" && more)
      clip_dir = argv[++i];
    else if (name == "--i2v" && more)
      i2v_out = argv[++i];
    else if (name == "--t5" && more)
      i2v.t5_root = argv[++i];
    else if (name == "--image" && more)
      image = argv[++i];
    else if (name == "--prompt" && more)
      i2v.prompt = argv[++i];
    else if (name == "--width" && more)
      i2v.width = std::atoi(argv[++i]);
    else if (name == "--height" && more)
      i2v.height = std::atoi(argv[++i]);
    else if (name == "--frames" && more)
      i2v.frames = std::atoi(argv[++i]);
    else if (name == "--steps" && more)
      i2v.steps = std::atoi(argv[++i]);
    else if (name == "--seed" && more)
      i2v.seed = std::strtoull(argv[++i], nullptr, 10);
    else if (name == "--uni3c" && more)
      i2v.uni3c_file = argv[++i];
    else if (name == "--guide-raw" && more)
      guide_raw = argv[++i];
    else if (name == "--uni3c-strength" && more)
      i2v.uni3c_strength = static_cast<float>(std::atof(argv[++i]));
    else if (name == "--uni3c-oracle" && more)
      uni3c_oracle = argv[++i];
    else if (name == "--dump" && more)
      dump_dir = argv[++i];
    else if (name == "--dit-bench" && i + 3 < argc)
      for (int& value : bench)
        value = std::atoi(argv[++i]);
    else
      return Usage();
  }
  if (!uni3c_oracle.empty())
    return Uni3COracle(dit, lora, dit_oracle, clip_dir, i2v.uni3c_file,
                       uni3c_oracle);
  if (!i2v_out.empty()) {
    i2v.wan21_root = model;
    i2v.dit_file = dit;
    i2v.lora_file = lora;
    return RunI2V(i2v, i2v_out, image, guide_raw, dump_dir);
  }
  if (!dit.empty() && !dit_oracle.empty() && bench[0] > 0)
    return DitBench(dit, lora, dit_oracle, clip_dir, bench[0], bench[1],
                    bench[2]);
  if (!dit.empty() && !dit_oracle.empty())
    return DitOracle(dit, lora, dit_oracle, clip_dir);
  if (model.empty())
    return Usage();
  if (!clip_oracle.empty())
    return ClipOracle(model, clip_oracle);
  return Usage();
}
