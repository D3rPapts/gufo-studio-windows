#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

// The routed-expert formats of the unsloth UD-Q3_K_XL GGUF (IQ3_XXS gate/up,
// one IQ4_XS layer, IQ4_NL down) through every expert entry the executor
// uses: the vector projection (one matrix and a gate/up pair) and the tiled
// prefill GEMM. Each output is checked against the CPU dequantization times
// the activations, so a wrong block layout, codebook or dispatch shows up as
// an O(1) error, far above the activation quantization the kernels apply.
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

struct Format {
  int ggml_type;
  const char* name;
  std::size_t block_elements;
  std::size_t block_bytes;
  void (*dequantize)(const void*, float*, std::size_t);
};

constexpr int kExperts = 8;
constexpr int kUsed = 4;

/// Random blocks with a moderate FP16 scale in front of each one; every other
/// byte (grid indices, signs, sub-block scales, nibbles) is a valid code.
std::vector<std::uint8_t> MakeWeights(const Format& f, std::size_t rows,
                                      std::size_t cols, std::uint32_t seed) {
  const std::size_t blocks = rows * cols / f.block_elements;
  std::vector<std::uint8_t> w(blocks * f.block_bytes);
  for (auto& b : w) {
    b = static_cast<std::uint8_t>(NextRandom(&seed));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    const float scale = 0.002F + 0.001F * static_cast<float>(b % 5);
    const __half d = __float2half(scale);
    std::memcpy(w.data() + b * f.block_bytes, &d, sizeof(d));
  }
  return w;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T)), "allocation");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> host(count);
  CheckHip(hipMemcpy(host.data(), device, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  return host;
}

/// Worst error of `actual` against the FP64 oracle, relative to the sum of
/// |weight * activation| of the same output (the scale the int8 activations
/// are rounded against).
double Compare(const std::vector<float>& actual,
               const std::vector<float>& weights, const std::vector<float>& x,
               const std::vector<std::int32_t>& ids, int rows, int cols,
               int tokens, int used) {
  double worst = 0.0;
  for (int t = 0; t < tokens; ++t) {
    for (int u = 0; u < used; ++u) {
      const int slot = t * used + u;
      const int expert = ids[slot];
      const float* in = x.data() + static_cast<std::size_t>(t) * cols;
      for (int r = 0; r < rows; ++r) {
        const float* w = weights.data() +
                         (static_cast<std::size_t>(expert) * rows + r) * cols;
        double sum = 0.0, magnitude = 0.0;
        for (int k = 0; k < cols; ++k) {
          sum += double(w[k]) * in[k];
          magnitude += std::abs(double(w[k]) * in[k]);
        }
        const float got = actual[static_cast<std::size_t>(slot) * rows + r];
        if (!std::isfinite(got)) {
          throw std::runtime_error("nonfinite expert output");
        }
        worst = std::max(worst, std::abs(got - sum) / (magnitude + 1e-6));
      }
    }
  }
  return worst;
}

bool Check(const Format& f, int tokens, bool pair, std::uint32_t seed) {
  constexpr int rows = 96, cols = 512;
  const auto wa = MakeWeights(f, std::size_t(kExperts) * rows, cols, seed);
  const auto wb = MakeWeights(f, std::size_t(kExperts) * rows, cols, seed + 1);
  std::vector<float> da(wa.size() / f.block_bytes * f.block_elements);
  std::vector<float> db(da.size());
  f.dequantize(wa.data(), da.data(), da.size());
  f.dequantize(wb.data(), db.data(), db.size());
  std::vector<float> x(static_cast<std::size_t>(tokens) * cols);
  std::uint32_t state = seed ^ 0x9E3779B9U;
  for (float& v : x) {
    v = static_cast<float>(static_cast<int>(NextRandom(&state) % 2001) - 1000) /
        1000.0F;
  }
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * kUsed);
  for (int t = 0; t < tokens; ++t) {
    for (int u = 0; u < kUsed; ++u) {
      ids[t * kUsed + u] = (t + 2 * u + t / 3) % kExperts;
    }
  }
  void* d_wa = Upload(wa);
  void* d_wb = Upload(wb);
  float* d_x = Upload(x);
  std::int32_t* d_ids = Upload(ids);
  const std::size_t outputs = ids.size() * rows;
  float* d_a = Upload(std::vector<float>(outputs, 0.0F));
  float* d_b = Upload(std::vector<float>(outputs, 0.0F));
  const bool tiled = tokens > 32;
  int rc = 0;
  if (tiled) {
    qfn_mmq_set_routed_max_expert_rows(tokens);
    rc = qfn_mmq_iq_moe_raw(f.ggml_type, d_wa, d_x, d_ids, d_a, rows, cols,
                            tokens, kExperts, kUsed, nullptr);
    qfn_mmq_set_routed_max_expert_rows(0);
  } else {
    rc = qfn_mmq_moe_vec(f.ggml_type, d_wa, d_x, d_ids, d_a, rows, cols, tokens,
                         kExperts, kUsed, nullptr, pair ? d_wb : nullptr,
                         pair ? d_b : nullptr);
  }
  CheckHip(hipDeviceSynchronize(), "expert projection");
  double worst = 0.0;
  if (rc == 0) {
    worst =
        Compare(Download(d_a, outputs), da, x, ids, rows, cols, tokens, kUsed);
    if (pair) {
      worst = std::max(worst, Compare(Download(d_b, outputs), db, x, ids, rows,
                                      cols, tokens, kUsed));
    }
  }
  for (void* p :
       {d_wa, d_wb, static_cast<void*>(d_x), static_cast<void*>(d_ids),
        static_cast<void*>(d_a), static_cast<void*>(d_b)}) {
    CheckHip(hipFree(p), "free");
  }
  const char* route = tiled ? "tiled" : pair ? "vector pair" : "vector";
  // The kernels round activations to int8 per 32 values; that costs well
  // under 1 % of the summed magnitude, a wrong layout costs all of it.
  const bool ok = rc == 0 && worst < 1e-2;
  std::cout << f.name << ' ' << route << " tokens=" << tokens << ": rc=" << rc
            << " worst relative error " << worst << (ok ? "" : "  FAILED")
            << '\n';
  return ok;
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0) {
      throw std::runtime_error("prepare quantized projections");
    }
    namespace quant = gufo::quant;
    const Format formats[] = {
        {18, "IQ3_XXS", 256, sizeof(quant::block_iq3_xxs),
         quant::DequantizeIQ3_XXS},
        {23, "IQ4_XS", 256, sizeof(quant::block_iq4_xs),
         quant::DequantizeIQ4_XS},
        {20, "IQ4_NL", 32, sizeof(quant::block_iq4_nl),
         quant::DequantizeIQ4_NL},
    };
    bool ok = true;
    std::uint32_t seed = 0x1A3C5E7U;
    for (const Format& f : formats) {
      ok = Check(f, 1, false, seed++) && ok;
      ok = Check(f, 4, false, seed++) && ok;
      ok = Check(f, 4, true, seed++) && ok;
      ok = Check(f, 70, false, seed++) && ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
