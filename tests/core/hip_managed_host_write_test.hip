// Managed placement uploads weights with host writes (std::memcpy into
// hipMallocManaged memory) because HIP's copy engine writing into managed
// memory faults on some Windows drivers. This pins the property that
// placement depends on: the CPU can write large managed allocations, the
// device reads those bytes in a kernel, and the copy engine can read the
// result back.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/managed_alloc.hpp"

namespace {

void Check(hipError_t status, const char* call) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(call) + ": " +
                             hipGetErrorString(status));
}

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

__global__ void Transform(const std::uint32_t* in, std::uint32_t* out,
                          std::size_t count) {
  const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = in[i] * 1664525U + 1013904223U;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t bytes = 256ull << 20;
    bool engine_upload = false;
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg = argv[i];
      if (arg == "--hip-copy") {
        engine_upload = true;
      } else {
        const unsigned long long parsed = std::stoull(std::string(arg));
        Require(parsed != 0, "size must be positive");
        bytes = static_cast<std::size_t>(parsed);
      }
    }
    Require(bytes % (256 * sizeof(std::uint32_t)) == 0,
            "size must be a multiple of 1 KiB");
    const std::size_t count = bytes / sizeof(std::uint32_t);
    std::vector<std::uint32_t> input(count), expected(count), actual(count);
    for (std::size_t i = 0; i < count; ++i) {
      input[i] = static_cast<std::uint32_t>(i * 2654435761ULL);
      expected[i] = input[i] * 1664525U + 1013904223U;
    }
    void* memory = nullptr;
    Check(gufo::hip::AllocateManaged(&memory, bytes), "AllocateManaged");
    // The upload path under test: plain host writes into managed memory.
    if (engine_upload) {
      // Comparison mode: the HIP copy engine writing into managed memory,
      // which faults on some Windows drivers once the destination is large.
      for (std::size_t done = 0; done < count;) {
        const std::size_t chunk =
            std::min<std::size_t>(count - done, 4ull << 20);
        Check(hipMemcpy(static_cast<std::uint8_t*>(memory) + done * 4,
                        input.data() + done, chunk * 4, hipMemcpyHostToDevice),
              "hipMemcpy H2D into managed");
        done += chunk;
      }
    } else {
      for (std::size_t done = 0; done < count;) {
        const std::size_t chunk =
            std::min<std::size_t>(count - done, 4ull << 20);
        std::memcpy(static_cast<std::uint8_t*>(memory) + done * 4,
                    input.data() + done, chunk * 4);
        done += chunk;
      }
    }
    Transform<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        static_cast<const std::uint32_t*>(memory),
        static_cast<std::uint32_t*>(memory), count);
    Check(hipDeviceSynchronize(), "hipDeviceSynchronize after host write");
    // The engine reads managed memory on every logits copy.
    Check(hipMemcpy(actual.data(), memory, bytes, hipMemcpyDeviceToHost),
          "hipMemcpy D2H from managed");
    Require(actual == expected, "engine readback of host-written managed");
    // And the CPU reads the kernel's result directly.
    const auto* cpu = static_cast<const std::uint32_t*>(memory);
    for (std::size_t i = 0; i < count; i += 4096)
      Require(cpu[i] == expected[i], "kernel writes not visible to CPU");
    Check(hipFree(memory), "hipFree managed");
    std::cout << "PASS: host writes into " << (bytes >> 20) << " MiB of"
              << " managed memory are device-visible and engine-readable\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
