#ifndef GUFO_MODELS_WAN22_ATTENTION_HIP_HPP_
#define GUFO_MODELS_WAN22_ATTENTION_HIP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::wan22 {

/// Transposes head-major BF16 values [heads][rows][128] into the layout the
/// attention kernel reads, FP16 [heads][128][PaddedRows(rows)] (clamped to
/// +-65504), zeroing the pad.
void LaunchPackValues(const std::uint16_t* values, std::uint16_t* packed,
                      std::uint32_t rows, std::uint32_t heads,
                      hipStream_t stream);

/// Rows rounded up to the 32-key tile of the attention kernel.
constexpr std::uint32_t PaddedRows(std::uint32_t rows) {
  return (rows + 31U) & ~31U;
}

/// Dense, unmasked BF16 attention with head dimension 128 and scale
/// 1/sqrt(128). Queries [heads][query_rows][128] and keys
/// [heads][key_rows][128] are head-major, values packed by LaunchPackValues.
/// Writes row-major output [query_rows][heads * 128]. Scores, softmax and the
/// value accumulation are FP32; probabilities and values enter the PV product
/// as FP16. The query blocks are split across equal launches of at most 2^31
/// query-key pairs (Windows' 2 s GPU watchdog).
void LaunchAttention(const std::uint16_t* query, const std::uint16_t* key,
                     const std::uint16_t* packed_values, std::uint16_t* output,
                     std::uint32_t query_rows, std::uint32_t key_rows,
                     std::uint32_t heads, hipStream_t stream);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_ATTENTION_HIP_HPP_
