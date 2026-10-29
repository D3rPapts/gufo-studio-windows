// Times the MiniMax H3 DiT WMMA attention (dit_ops::LaunchWmmaAttention) on
// seeded random BF16 Q/K/V and writes the BF16 output for comparisons.
// GUFO_H3_ATTENTION_KEYS selects the kernel variant as in production.
// Usage: minimax_h3_attention_bench [rows=6157] [heads=56] [output.bf16]

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/minimax_h3/attention.hip.hpp"

namespace {

#define CHECK_HIP(expression)                              \
  do {                                                     \
    const hipError_t status_ = (expression);               \
    if (status_ != hipSuccess) {                           \
      std::fprintf(stderr, "%s failed: %s\n", #expression, \
                   hipGetErrorString(status_));            \
      std::exit(1);                                        \
    }                                                      \
  } while (false)

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                    16);
}

}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t rows = argc > 1 ? std::atoi(argv[1]) : 6157;
  const std::uint32_t heads = argc > 2 ? std::atoi(argv[2]) : 56;
  const char* output_path = argc > 3 ? argv[3] : nullptr;
  const std::size_t elements = std::size_t(rows) * heads * 128;
  const std::size_t padded = std::size_t((rows + 31) / 32 * 32) * heads * 128;
  std::vector<std::uint16_t> host(elements);
  std::mt19937 random(20261029);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uint16_t* device[3]{};
  for (auto& buffer : device) {
    // Peaky scores: scale Q/K so q.k/sqrt(128) has a standard deviation ~3.
    for (auto& value : host) {
      value = ToBf16(normal(random) * 1.3F);
    }
    CHECK_HIP(hipMalloc(&buffer, elements * 2U));
    CHECK_HIP(
        hipMemcpy(buffer, host.data(), elements * 2U, hipMemcpyHostToDevice));
  }
  std::uint16_t* output = nullptr;
  std::uint16_t* packed = nullptr;
  CHECK_HIP(hipMalloc(&output, elements * 2U));
  CHECK_HIP(hipMalloc(&packed, padded * 2U));
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  const auto run = [&] {
    gufo::minimax_h3::dit_ops::LaunchWmmaAttention(
        device[0], device[1], device[2], output, packed, rows, heads, nullptr);
  };
  run();
  CHECK_HIP(hipDeviceSynchronize());
  constexpr int kRepeats = 10;
  CHECK_HIP(hipEventRecord(start, nullptr));
  for (int index = 0; index < kRepeats; ++index) {
    run();
  }
  CHECK_HIP(hipEventRecord(stop, nullptr));
  CHECK_HIP(hipEventSynchronize(stop));
  float elapsed = 0.0F;
  CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
  const double ms = elapsed / kRepeats;
  const double flops = 4.0 * rows * double(rows) * 128.0 * heads;
  const char* keys = std::getenv("GUFO_H3_ATTENTION_KEYS");
  std::printf("attention rows=%u heads=%u keys=%s: %.2f ms (%.1f TFLOPS)\n",
              rows, heads, keys == nullptr ? "32" : keys, ms, flops / ms / 1e9);
  if (output_path != nullptr) {
    CHECK_HIP(
        hipMemcpy(host.data(), output, elements * 2U, hipMemcpyDeviceToHost));
    std::FILE* file = std::fopen(output_path, "wb");
    if (file == nullptr ||
        std::fwrite(host.data(), 2U, elements, file) != elements) {
      std::fprintf(stderr, "cannot write %s\n", output_path);
      return 1;
    }
    std::fclose(file);
  }
  return 0;
}
