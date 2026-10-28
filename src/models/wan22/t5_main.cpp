// gufo-wan-t5: tokenizes and encodes one prompt with the umT5-XXL text
// encoder of Wan 2.2 TI2V-5B.
//
//   gufo-wan-t5 --model <Diffusers root> --prompt TEXT
//               [--oracle PREFIX [--index K]] [--dump FILE]
//
// `--dump` writes the states as raw little-endian FP32 [tokens, 4096].
// With `--oracle`, PREFIX_ids_K.npy, PREFIX_mask_K.npy and PREFIX_states_K.npy
// (the Diffusers pipeline's padded ids, attention mask and last hidden state
// of the real tokens) are compared: ids exactly, states per row by cosine
// similarity and maximum absolute error.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"

namespace {

namespace wan22 = gufo::models::wan22;
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

// A little-endian, C-order .npy array as doubles.
bool ReadNpy(const std::string& path, std::vector<double>* values,
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
      header.find("'fortran_order': False") == std::string::npos)
    return false;
  const auto descr = header.find("'descr': '");
  const auto open = header.find('(');
  const auto close = header.find(')');
  if (descr == std::string::npos || open == std::string::npos ||
      close == std::string::npos)
    return false;
  const std::string type = header.substr(descr + 10, 3);
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
  for (std::size_t i = 0; i < count; ++i) {
    if (type == "<f4") {
      float v;
      in.read(reinterpret_cast<char*>(&v), 4);
      (*values)[i] = v;
    } else if (type == "<i8") {
      std::int64_t v;
      in.read(reinterpret_cast<char*>(&v), 8);
      (*values)[i] = static_cast<double>(v);
    } else if (type == "<i4") {
      std::int32_t v;
      in.read(reinterpret_cast<char*>(&v), 4);
      (*values)[i] = v;
    } else if (type == "|b1" || type == "|u1") {
      std::uint8_t v;
      in.read(reinterpret_cast<char*>(&v), 1);
      (*values)[i] = v;
    } else {
      return false;
    }
  }
  return static_cast<bool>(in);
}

int Usage() {
  std::cerr << "usage: gufo-wan-t5 --model DIFFUSERS_ROOT --prompt TEXT "
               "[--oracle PREFIX [--index K]] [--dump FILE]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string prompt;
  std::string oracle;
  std::string index = "0";
  std::string dump;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string name = argv[i];
    if (name == "--model")
      model = argv[i + 1];
    else if (name == "--prompt")
      prompt = argv[i + 1];
    else if (name == "--oracle")
      oracle = argv[i + 1];
    else if (name == "--index")
      index = argv[i + 1];
    else if (name == "--dump")
      dump = argv[i + 1];
    else
      return Usage();
  }
  if (model.empty() || prompt.empty() || argc % 2 == 0)
    return Usage();
  const std::filesystem::path root(model);
  std::string error;
  auto start = Clock::now();
  const auto tokenizer =
      wan22::Tokenizer::Load(root / "tokenizer" / "tokenizer.json", &error);
  if (!tokenizer) {
    std::cerr << "tokenizer: " << error << "\n";
    return 1;
  }
  std::cout << "tokenizer load " << Seconds(start) << " s\n";
  const auto ids = tokenizer->Encode(wan22::CleanPrompt(prompt));
  std::cout << "ids (" << ids.size() << "):";
  for (const auto id : ids)
    std::cout << " " << id;
  std::cout << "\n";

  std::size_t free_before = 0;
  std::size_t total = 0;
  (void)hipMemGetInfo(&free_before, &total);
  start = Clock::now();
  auto encoder = wan22::TextEncoder::Load(root / "text_encoder", &error);
  if (!encoder) {
    std::cerr << "encoder: " << error << "\n";
    return 1;
  }
  std::cout << "encoder load " << Seconds(start) << " s, "
            << encoder->device_bytes() / 1048576.0 << " MiB device\n";
  std::vector<float> states;
  for (int run = 0; run < 3; ++run) {
    start = Clock::now();
    if (!encoder->Encode(ids, &states, &error)) {
      std::cerr << "encode: " << error << "\n";
      return 1;
    }
    std::cout << "encode run " << run << ": " << Seconds(start) * 1000
              << " ms\n";
  }
  std::size_t free_after = 0;
  (void)hipMemGetInfo(&free_after, &total);
  std::cout << "device memory in use by the process ~"
            << (free_before - std::min(free_before, free_after)) / 1048576.0
            << " MiB\n";
  constexpr std::size_t kWidth = wan22::TextEncoder::kModelWidth;
  std::cout << "row 0 [0..4]:";
  for (std::size_t c = 0; c < 4; ++c)
    std::cout << " " << states[c];
  std::cout << "\n";
  if (!dump.empty()) {
    std::ofstream out(dump, std::ios::binary);
    out.write(reinterpret_cast<const char*>(states.data()),
              static_cast<std::streamsize>(states.size() * sizeof(float)));
  }
  if (oracle.empty())
    return 0;

  std::vector<double> oracle_ids, mask, oracle_states;
  std::vector<std::size_t> shape;
  if (!ReadNpy(oracle + "_ids_" + index + ".npy", &oracle_ids, &shape) ||
      !ReadNpy(oracle + "_mask_" + index + ".npy", &mask, &shape) ||
      !ReadNpy(oracle + "_states_" + index + ".npy", &oracle_states, &shape) ||
      shape.size() != 2 || shape[1] != kWidth) {
    std::cerr << "cannot read oracle " << oracle << "\n";
    return 1;
  }
  std::size_t real = 0;
  for (const double m : mask)
    real += m > 0;
  bool ids_equal = real == ids.size() && shape[0] == ids.size();
  for (std::size_t t = 0; ids_equal && t < ids.size(); ++t)
    ids_equal = oracle_ids[t] == ids[t];
  std::cout << "oracle ids " << (ids_equal ? "EQUAL" : "DIFFERENT") << " ("
            << real << " real tokens)\n";
  if (!ids_equal)
    return 1;
  double min_cosine = 1;
  double max_error = 0;
  for (std::size_t t = 0; t < ids.size(); ++t) {
    double dot = 0, a2 = 0, b2 = 0, row_error = 0;
    for (std::size_t c = 0; c < kWidth; ++c) {
      const double a = states[t * kWidth + c];
      const double b = oracle_states[t * kWidth + c];
      dot += a * b;
      a2 += a * a;
      b2 += b * b;
      row_error = std::max(row_error, std::abs(a - b));
    }
    const double cosine = dot / std::sqrt(a2 * b2);
    std::cout << "row " << t << " cosine " << cosine << " max_abs "
              << row_error << "\n";
    min_cosine = std::min(min_cosine, cosine);
    max_error = std::max(max_error, row_error);
  }
  std::cout << "min cosine " << min_cosine << ", max abs error " << max_error
            << (min_cosine >= 0.999 ? " PASS" : " FAIL") << "\n";
  return min_cosine >= 0.999 ? 0 : 1;
}
