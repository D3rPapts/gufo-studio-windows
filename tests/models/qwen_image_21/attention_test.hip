// Qwen-Image attention kernels against the frozen former kernel
// (attention_reference.hip): the exact kernels must match it byte for byte on
// random inputs, layouts, masks and segment splits. `--bench` also times a
// keyframe-sized step (4032 queries, 42168 segmented keys, 32 heads).
// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace gufo::models::qwen_image_21::hip::reference {
void FusedAttention128Segmented(const std::uint16_t* queries,
                                const KeySegments& segments,
                                std::uint16_t* output, int heads, int rows,
                                int query_start, int query_count,
                                hipStream_t stream, const int* limits);
void FusedAttention128(const std::uint16_t* queries, const std::uint16_t* keys,
                       const std::uint16_t* values, std::uint16_t* output,
                       int heads, int rows, int count, int query_start,
                       hipStream_t stream, int query_count,
                       const int* limits);
}  // namespace gufo::models::qwen_image_21::hip::reference

namespace {
namespace image = gufo::models::qwen_image_21::hip;

template<typename T>
struct Device {
  T* data{};
  std::size_t count{};
  explicit Device(std::size_t n) : count(n) {
    image::Check(hipMalloc(&data, n * sizeof(T)));
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void Put(const std::vector<T>& host) {
    assert(host.size() == count);
    image::Check(
        hipMemcpy(data, host.data(), count * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> Get() const {
    std::vector<T> host(count);
    image::Check(
        hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost));
    return host;
  }
};

std::uint16_t Bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  return (bits + 32767 + ((bits >> 16) & 1)) >> 16;
}

float Float(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t(value) << 16);
}

constexpr std::uint16_t kSentinel = 0x7fc1;
constexpr int kDim = 128;

std::vector<std::uint16_t> Random(std::mt19937& rng, std::size_t n,
                                  float scale) {
  std::vector<std::uint16_t> x(n);
  std::normal_distribution<float> normal(0.0F, scale);
  for (auto& v : x)
    v = Bf16(normal(rng));
  return x;
}

struct Case {
  int heads, rows, keys, start, count;
  float query_scale;
  bool masked;
};

int mismatches_reported = 0;

void ExpectSame(const std::vector<std::uint16_t>& expected,
                const std::vector<std::uint16_t>& actual,
                const std::string& what) {
  if (expected == actual)
    return;
  std::size_t first = 0, differing = 0;
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (expected[i] != actual[i]) {
      if (!differing)
        first = i;
      ++differing;
    }
  std::cerr << "MISMATCH " << what << ": " << differing << " of "
            << expected.size() << " values differ, first at " << first
            << " (" << Float(expected[first]) << " vs " << Float(actual[first])
            << ")\n";
  ++mismatches_reported;
}

/// Splits head-major [heads][keys][128] into three padded segments.
struct Segments {
  std::vector<std::unique_ptr<Device<std::uint16_t>>> parts;
  image::KeySegments keys;
  Segments(const std::vector<std::uint16_t>& k,
           const std::vector<std::uint16_t>& v, int heads, int count,
           std::array<int, 3> sizes) {
    const int strides[3] = {sizes[0] + 7, sizes[1], sizes[2] + 3};
    for (int s = 0, first = 0; s < 3; first += sizes[s], ++s) {
      std::vector<std::uint16_t> pk(std::size_t(heads) * strides[s] * kDim,
                                    kSentinel),
          pv(pk.size(), kSentinel);
      for (int head = 0; head < heads; ++head)
        for (int row = 0; row < sizes[s]; ++row)
          for (int d = 0; d < kDim; ++d) {
            const auto from = (std::size_t(head) * count + first + row) * kDim + d;
            const auto to = (std::size_t(head) * strides[s] + row) * kDim + d;
            pk[to] = k[from];
            pv[to] = v[from];
          }
      for (auto* host : {&pk, &pv}) {
        parts.push_back(std::make_unique<Device<std::uint16_t>>(host->size()));
        parts.back()->Put(*host);
      }
      keys.keys[s] = parts[parts.size() - 2]->data;
      keys.values[s] = parts.back()->data;
      keys.end[s] = first + sizes[s];
      keys.stride[s] = strides[s];
    }
  }
};

void Exact(std::mt19937& rng, const Case& c) {
  const auto queries = Random(rng, std::size_t(c.heads) * c.rows * kDim,
                              c.query_scale);
  const auto keys = Random(rng, std::size_t(c.heads) * c.keys * kDim, 1.0F);
  const auto values = Random(rng, keys.size(), 1.0F);
  std::vector<int> limits(c.rows);
  for (int q = 0; q < c.rows; ++q)
    limits[q] = std::max(1, int(rng() % c.keys) + 1);
  // Causal-looking prefix plus fully visible rows.
  for (int q = 0; q < c.rows; q += 3)
    limits[q] = c.keys;
  Device<std::uint16_t> q(queries.size()), k(keys.size()), v(values.size()),
      o(std::size_t(c.rows) * c.heads * kDim);
  Device<int> l(limits.size());
  q.Put(queries);
  k.Put(keys);
  v.Put(values);
  l.Put(limits);
  const int* mask = c.masked ? l.data : nullptr;
  const std::vector<std::uint16_t> blank(o.count, kSentinel);
  const std::string name = "heads=" + std::to_string(c.heads) +
                           " rows=" + std::to_string(c.rows) +
                           " keys=" + std::to_string(c.keys) +
                           " start=" + std::to_string(c.start) +
                           " count=" + std::to_string(c.count) +
                           " masked=" + std::to_string(c.masked);
  o.Put(blank);
  image::reference::FusedAttention128(q.data, k.data, v.data, o.data, c.heads,
                                      c.rows, c.keys, c.start, nullptr,
                                      c.count, mask);
  image::Check(hipGetLastError());
  const auto expected = o.Get();
  o.Put(blank);
  image::FusedAttention128(q.data, k.data, v.data, o.data, c.heads, c.rows,
                           c.keys, c.start, nullptr, c.count, mask);
  image::Check(hipGetLastError());
  ExpectSame(expected, o.Get(), "contiguous " + name);
  for (const auto& sizes :
       {std::array{c.keys / 3, c.keys / 2 - c.keys / 3, c.keys - c.keys / 2},
        std::array{c.keys - 40, 9, 31}, std::array{c.keys - 17, 17, 0},
        std::array{5, c.keys - 5, 0}}) {
    Segments segments(keys, values, c.heads, c.keys, sizes);
    o.Put(blank);
    image::reference::FusedAttention128Segmented(
        q.data, segments.keys, o.data, c.heads, c.rows, c.start, c.count,
        nullptr, mask);
    image::Check(hipGetLastError());
    const auto segmented_expected = o.Get();
    ExpectSame(expected, segmented_expected, "reference segmented " + name);
    o.Put(blank);
    image::FusedAttention128Segmented(q.data, segments.keys, o.data, c.heads,
                                      c.rows, c.start, c.count, nullptr, mask);
    image::Check(hipGetLastError());
    ExpectSame(segmented_expected, o.Get(),
               "segmented " + name + " split=" + std::to_string(sizes[0]) +
                   "/" + std::to_string(sizes[1]) + "/" +
                   std::to_string(sizes[2]));
  }
}

void Bench() {
  // Keyframe step: 4032 image rows against 38136 cached condition rows plus
  // the 4032 new rows, 32 heads (bericht.md, job r486).
  constexpr int heads = 32, rows = 4032, cached = 38136;
  constexpr int keys = cached + rows;
  std::mt19937 rng(7);
  const auto queries = Random(rng, std::size_t(heads) * rows * kDim, 1.0F);
  const auto kv = Random(rng, std::size_t(heads) * keys * kDim, 1.0F);
  const auto vv = Random(rng, kv.size(), 1.0F);
  Device<std::uint16_t> q(queries.size()), o(std::size_t(rows) * heads * kDim);
  q.Put(queries);
  Segments segments(kv, vv, heads, keys, {cached, rows, 0});
  hipEvent_t begin, end;
  image::Check(hipEventCreate(&begin));
  image::Check(hipEventCreate(&end));
  const double flop = 4.0 * heads * double(rows) * keys * kDim;
  const auto time = [&](const char* label, auto&& launch) {
    launch();  // warm-up
    image::Check(hipDeviceSynchronize());
    constexpr int kRuns = 3;
    image::Check(hipEventRecord(begin));
    for (int i = 0; i < kRuns; ++i)
      launch();
    image::Check(hipEventRecord(end));
    image::Check(hipEventSynchronize(end));
    float ms = 0;
    image::Check(hipEventElapsedTime(&ms, begin, end));
    ms /= kRuns;
    std::cout << label << ": " << ms << " ms per call, "
              << flop / (ms * 1e-3) / 1e12 << " TFLOP/s\n";
    return o.Get();
  };
  const auto expected = time("reference", [&] {
    image::reference::FusedAttention128Segmented(q.data, segments.keys, o.data,
                                                 heads, rows, 0, rows, nullptr,
                                                 nullptr);
  });
  const auto exact = time("exact", [&] {
    image::FusedAttention128Segmented(q.data, segments.keys, o.data, heads,
                                      rows, 0, rows, nullptr, nullptr);
  });
  ExpectSame(expected, exact, "bench exact");
  (void)hipEventDestroy(begin);
  (void)hipEventDestroy(end);
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
    return 77;
  std::mt19937 rng(20261101);
  for (const auto& c : {Case{2, 384, 512, 128, 128, 1.0F, false},
                        Case{2, 384, 512, 128, 128, 1.0F, true},
                        Case{2, 253, 513, 0, 253, 10.0F, false},
                        Case{2, 253, 513, 0, 253, 10.0F, true},
                        Case{3, 300, 1000, 37, 263, 1.0F, true},
                        Case{2, 384, 4132, 0, 384, 1.0F, false},
                        Case{2, 384, 4132, 0, 384, 3.0F, true},
                        Case{1, 640, 8230, 128, 500, 1.0F, false},
                        Case{1, 129, 777, 1, 128, 1.0F, false}})
    Exact(rng, c);
  if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    Bench();
  if (mismatches_reported) {
    std::cerr << mismatches_reported << " attention comparisons failed\n";
    return 1;
  }
  std::cout << "Qwen-Image attention kernels match the former kernel "
               "byte for byte\n";
}
