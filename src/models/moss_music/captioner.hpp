#ifndef GUFO_MODELS_MOSS_MUSIC_CAPTIONER_HPP_
#define GUFO_MODELS_MOSS_MUSIC_CAPTIONER_HPP_

#include <cstddef>
#include <memory>
#include <string>

#include "src/models/qwen3_omni/captioner.hpp"

namespace gufo::models::moss_music {

/// MOSS-Music-8B-Instruct loaded once and reused across requests: one WAV
/// plus a question in, text out, through the same request/result shapes as
/// the Qwen3-Omni captioner so `gufo serve` and its clients only switch the
/// model. Calls are serialized: one caption runs at a time. HIP only.
class Captioner {
public:
  using Request = qwen3_omni::CaptionRequest;
  using Result = qwen3_omni::CaptionResult;
  using CancellationCheck = qwen3_omni::CancellationCheck;

  struct Options {
    /// `<stem>.audio.gguf` from tools/moss_music/convert_hf_to_gufo.py.
    std::string audio_gguf;
    /// llama.cpp `qwen3` GGUF of the LLM (BF16, Q8_0, Q4_K_M, Q6_K).
    std::string llm_gguf;
    /// KV capacity: audio span with markers plus prompt plus answer.
    std::size_t context{4096};
    /// Default question when a request carries none.
    std::string default_instruction{
        "Please give a detailed musical description of this clip."};
    std::string default_system{"You are a helpful assistant."};
  };

  static std::unique_ptr<Captioner> Create(const Options& options,
                                           std::string* error);
  ~Captioner();

  Captioner(const Captioner&) = delete;
  Captioner& operator=(const Captioner&) = delete;

  bool Caption(const Request& request, const CancellationCheck& is_cancelled,
               Result* result, std::string* error);

  [[nodiscard]] std::size_t max_context() const;
  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::string quantization() const;

private:
  struct Impl;
  explicit Captioner(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::moss_music

#endif  // GUFO_MODELS_MOSS_MUSIC_CAPTIONER_HPP_
