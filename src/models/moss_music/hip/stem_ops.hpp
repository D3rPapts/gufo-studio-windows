#ifndef GUFO_MODELS_MOSS_MUSIC_HIP_STEM_OPS_HPP_
#define GUFO_MODELS_MOSS_MUSIC_HIP_STEM_OPS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// HIP kernels of the MOSS-Music audio side that Qwen3-ASR's audio_ops do not
/// cover: the 400-frame chunk layout (ASR's are fixed to 100 frames / 13
/// tokens), the WMMA convolutions at MOSS's shapes (128x400 -> 480x64x200 ->
/// 480x32x100 -> 480x16x50), stem_proj's bias with the sinusoidal positions,
/// and the gated-MLP activation of the adapters. All BF16 rounding points
/// follow the bf16 reference model (Qwen3-ASR's conventions).
namespace gufo::models::moss_music::hip {

/// Mel-major log-mel [mel_bins, frames] F32 -> BF16 chunks
/// [chunks][mel_bins][chunk_frames], zero-padded past `frames`.
void LaunchChunkLogMel(const float* input, void* output_bfloat16,
                       std::size_t frames, std::size_t chunks,
                       std::size_t mel_bins, std::size_t chunk_frames,
                       hipStream_t stream);

/// conv1: BF16 [chunks][1][128][400] with weight [480][9] (unpacked) ->
/// BF16 [chunks][480][64][200], no bias.
void LaunchFirstConvolution(const void* input_bfloat16,
                            const void* weight_bfloat16, void* output_bfloat16,
                            std::size_t chunks, hipStream_t stream);

/// conv2/conv3: BF16 [chunks][480][height][width] with a weight packed by
/// qwen3_asr::hip::LaunchPackConvolutionWeights -> [chunks][480][h/2][w/2].
/// Supported (height, width): (64, 200) and (32, 100).
void LaunchConvolution(const void* input_bfloat16, const void* packed_weight,
                       void* output_bfloat16, std::size_t chunks,
                       std::size_t height, std::size_t width,
                       hipStream_t stream);

/// [chunks][480][16][chunk_time] -> [chunks * chunk_time][480 * 16], the
/// time-major flatten (index channel * 16 + frequency) of stem_proj's input.
void LaunchConvOutputLayout(const void* input_bfloat16, void* output_bfloat16,
                            std::size_t chunks, std::size_t chunk_time,
                            hipStream_t stream);

/// stem_proj bias and per-chunk sinusoidal positions (computed in BF16 like
/// the bf16 reference), dropping the padded rows of the tail chunk:
/// projected [chunks * chunk_time][hidden] F32 -> compact [tokens][hidden]
/// F32 (BF16-valued). `tokens` is the valid token count of all chunks.
void LaunchAddBiasPositionAndCompact(const float* projected,
                                     const void* bias_bfloat16,
                                     float* compact_output,
                                     std::size_t feature_frames,
                                     std::size_t chunks,
                                     std::size_t chunk_frames,
                                     std::size_t chunk_time, std::size_t tokens,
                                     std::size_t hidden, hipStream_t stream);

/// out = bf16(bf16(silu(bf16(gate))) * bf16(up)) for a gated MLP.
void LaunchSiluMulToBfloat16(const float* gate, const float* up,
                             void* output_bfloat16, std::size_t elements,
                             hipStream_t stream);

/// In place: F32 values rounded to the nearest BF16 value.
void LaunchRoundBfloat16(float* values, std::size_t elements,
                         hipStream_t stream);

/// Zeroes the time columns >= `valid` of a BF16 [rows][width] activation:
/// a clip shorter than one chunk is evaluated unpadded by the reference, so
/// after each stem convolution the padded region must read as zeros.
void LaunchZeroColumns(void* values_bfloat16, std::size_t rows,
                       std::size_t width, std::size_t valid,
                       hipStream_t stream);

void LaunchFloatToBfloat16(const float* input, void* output_bfloat16,
                           std::size_t elements, hipStream_t stream);

void LaunchBfloat16ToFloat(const void* input_bfloat16, float* output,
                           std::size_t elements, hipStream_t stream);

/// hidden[rows[k]] = bf16(source[k]) for k < count: audio embeddings take the
/// place of the <|AUDIO|> rows of the embedded prompt.
void LaunchScatterRowsRounded(float* hidden, const float* source,
                              const std::uint32_t* rows, std::size_t count,
                              std::size_t hidden_size, hipStream_t stream);

/// hidden[rows[k]] = bf16(hidden[rows[k]] + source[k]): the DeepStack merger
/// joins the audio rows after an LLM layer.
void LaunchAddRowsRounded(float* hidden, const float* source,
                          const std::uint32_t* rows, std::size_t count,
                          std::size_t hidden_size, hipStream_t stream);

}  // namespace gufo::models::moss_music::hip

#endif  // GUFO_MODELS_MOSS_MUSIC_HIP_STEM_OPS_HPP_
