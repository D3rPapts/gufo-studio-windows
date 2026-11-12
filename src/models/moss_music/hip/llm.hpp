#ifndef GUFO_MODELS_MOSS_MUSIC_HIP_LLM_HPP_
#define GUFO_MODELS_MOSS_MUSIC_HIP_LLM_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/moss_music/config.hpp"

namespace gufo::models::moss_music::hip {

/// Host copies of prefill states for the reference comparison (PLAN.md D7):
/// the embedded prompt (hs0), the outputs of LLM layers 0..2 after the
/// DeepStack addition (hs1..hs3) and the final normed state (hs_final), each
/// [prompt_tokens, hidden].
struct PrefillTrace {
  std::vector<float> hs0;
  std::array<std::vector<float>, 3> hs_layer;
  std::vector<float> hs_final;
};

/// The Qwen3-8B language model of MOSS-Music on the GPU, from the llama.cpp
/// `qwen3` GGUF the converter produces (BF16 matrices, F32 norms). It runs
/// Qwen3-ASR's text kernels (same architecture, wider): BF16 GEMMs with F32
/// accumulation, F32 KV cache holding BF16 values, the official eager
/// rounding boundaries. Audio rows of the prompt take the adapter output in
/// place of token embeddings; the DeepStack mergers are added to those rows
/// after layers 0, 1 and 2 during the prefill.
class Llm {
public:
  static std::unique_ptr<Llm> Create(const std::filesystem::path& gguf,
                                     std::size_t max_context,
                                     hipStream_t stream, std::string* error);
  ~Llm();

  Llm(const Llm&) = delete;
  Llm& operator=(const Llm&) = delete;

  [[nodiscard]] const TextConfig& config() const;
  [[nodiscard]] const core::GgufReader& reader() const;
  [[nodiscard]] std::size_t position() const;
  [[nodiscard]] std::size_t max_context() const;
  [[nodiscard]] std::size_t weight_bytes() const;

  void Reset();

  /// Runs the prompt at positions 0..n-1. `audio_rows[k]` is the prompt index
  /// of the k-th audio token; `audio_device` holds the adapter output
  /// [audio_tokens, hidden] on the device and `deepstack_device[j]` the j-th
  /// merger output, same shape (any may be null when audio_rows is empty).
  /// Writes the last row's BF16-valued logits [vocab] to `logits`.
  bool Prefill(std::span<const std::int32_t> prompt,
               std::span<const std::uint32_t> audio_rows,
               const float* audio_device,
               const std::array<const float*, 3>& deepstack_device,
               std::vector<float>* logits, std::string* error,
               PrefillTrace* trace = nullptr,
               const std::function<bool()>& is_cancelled = {});

  /// Runs one token at the next position and writes its logits.
  bool Decode(std::int32_t token, std::vector<float>* logits,
              std::string* error);

private:
  struct Impl;
  explicit Llm(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::moss_music::hip

#endif  // GUFO_MODELS_MOSS_MUSIC_HIP_LLM_HPP_
