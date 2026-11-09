// The vendored MMQ library asks its host for producer-quantized activations;
// DeepSeek's backend answers from a registry. gufo-omni has none, so every
// lookup misses and the kernels quantize their own inputs (DeepSeek's ROCm
// backend stub behaves the same). Linked only into the standalone driver;
// a binary that also links DeepSeek uses DeepSeek's definition.

#include <cstdint>

extern "C" int ds4_hip_q8_fold_take_q81(const void* src, std::uint64_t in_dim,
                                        const void** q81) {
  (void)src;
  (void)in_dim;
  if (q81 != nullptr) {
    *q81 = nullptr;
  }
  return 0;
}
