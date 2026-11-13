#include "src/models/moss_music/captioner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/utf8.hpp"
#include "src/models/moss_music/config.hpp"
#include "src/models/moss_music/hip/audio_encoder.hpp"
#include "src/models/moss_music/hip/llm.hpp"
#include "src/models/moss_music/model.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen3_asr/audio.hpp"

namespace gufo::models::moss_music {
namespace {

namespace asr = qwen3_asr;
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

/// Greedy at temperature 0; otherwise top-k, then top-p, then a draw
/// (infer.py: temperature 1.0, top_p 0.8, top_k 50).
std::int32_t Sample(const std::vector<float>& logits,
                    const Captioner::Request& request, std::mt19937_64* rng) {
  if (request.temperature <= 0.0F) {
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  std::vector<std::int32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const std::size_t k = std::clamp<std::size_t>(request.top_k, 1, order.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k),
                    order.end(), [&](std::int32_t a, std::int32_t b) {
                      return logits[a] > logits[b];
                    });
  std::vector<double> probabilities(k);
  double total = 0.0;
  for (std::size_t index = 0; index < k; ++index) {
    probabilities[index] = std::exp((logits[order[index]] - logits[order[0]]) /
                                    request.temperature);
    total += probabilities[index];
  }
  std::size_t kept = 0;
  double cumulative = 0.0;
  while (kept < k) {
    cumulative += probabilities[kept++] / total;
    if (cumulative >= request.top_p) {
      break;
    }
  }
  std::discrete_distribution<std::size_t> draw(
      probabilities.begin(), probabilities.begin() + static_cast<std::ptrdiff_t>(kept));
  return order[draw(*rng)];
}

}  // namespace

struct Captioner::Impl {
  Options options;
  std::unique_ptr<AudioModel> model;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  hipStream_t stream{nullptr};
  std::unique_ptr<hip::AudioEncoder> encoder;
  std::unique_ptr<hip::Llm> llm;
  std::mutex mutex;

  ~Impl() {
    encoder.reset();
    llm.reset();
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  void Run(const Request& request, const CancellationCheck& is_cancelled,
           Result* result) {
    std::string error;
    asr::AudioBuffer buffer;
    if (!asr::DecodeWav(request.wav, &buffer, &error)) {
      throw std::invalid_argument(error);
    }
    const std::vector<float> waveform = asr::ResampleMono16k(buffer);
    if (waveform.size() < 400U) {
      throw std::invalid_argument("audio is empty or shorter than 25 ms");
    }
    const asr::LogMelFeatures features = asr::ComputeLogMelFeatures(waveform);
    result->audio_seconds = static_cast<double>(waveform.size()) / 16000.0;

    auto begin = Clock::now();
    hip::EncodedAudio encoded;
    if (!encoder->Encode(features.values, features.frames, &encoded, &error,
                         is_cancelled)) {
      throw std::runtime_error(error);
    }
    result->audio_tokens = encoded.tokens;
    result->encoder_ms = Milliseconds(begin);

    // MossMusicProcessor._build_default_prompt + _build_input_from_prompt.
    const TokenIds& ids = model->config().tokens;
    std::vector<std::int32_t> prompt;
    const auto append = [&](const std::string& text) {
      for (const auto id : tokenizer->Encode(text)) {
        prompt.push_back(static_cast<std::int32_t>(id));
      }
    };
    const std::string& system =
        request.system.empty() ? options.default_system : request.system;
    const std::string& instruction = request.instruction.empty()
                                         ? options.default_instruction
                                         : request.instruction;
    append("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n");
    prompt.push_back(ids.audio_start);
    if (request.time_markers) {
      const std::vector<std::int32_t> span =
          AudioSpanWithMarkers(encoded.tokens, ids);
      prompt.insert(prompt.end(), span.begin(), span.end());
    } else {
      prompt.insert(prompt.end(), encoded.tokens, ids.audio_pad);
    }
    prompt.push_back(ids.audio_end);
    append("\n" + instruction + "<|im_end|>\n<|im_start|>assistant\n");
    std::vector<std::uint32_t> audio_rows;
    for (std::size_t index = 0; index < prompt.size(); ++index) {
      if (prompt[index] == ids.audio_pad) {
        audio_rows.push_back(static_cast<std::uint32_t>(index));
      }
    }
    if (audio_rows.size() != encoded.tokens || prompt.front() != ids.im_start) {
      throw std::runtime_error("prompt tokens do not match the audio span");
    }
    result->prompt_tokens = prompt.size();
    const std::size_t context = llm->max_context();
    if (prompt.size() + 1 > context) {
      result->invalid_request = true;
      throw std::invalid_argument(
          "audio needs " + std::to_string(prompt.size()) +
          " prompt tokens, more than the context of " +
          std::to_string(context));
    }
    const std::size_t max_tokens =
        std::min(request.max_tokens, context - prompt.size());

    begin = Clock::now();
    llm->Reset();
    std::vector<float> logits;
    if (!llm->Prefill(prompt, audio_rows, encoded.adapter, encoded.mergers,
                      &logits, &error, nullptr, is_cancelled)) {
      throw std::runtime_error(error);
    }
    result->prefill_ms = Milliseconds(begin);

    begin = Clock::now();
    std::mt19937_64 rng(request.seed);
    core::Utf8Decoder utf8;
    std::vector<std::int32_t> generated;
    bool listening = true;
    const auto publish = [&](const std::string& piece) {
      result->text += piece;
      if (listening && request.on_text && !piece.empty()) {
        listening = request.on_text(piece);
      }
    };
    result->finish_reason = "length";
    while (generated.size() < max_tokens) {
      if (!listening || (is_cancelled && is_cancelled())) {
        result->finish_reason = "cancelled";
        break;
      }
      const std::int32_t next = Sample(logits, request, &rng);
      if (next == ids.im_end || next == ids.endoftext) {
        result->finish_reason = "stop";
        break;
      }
      generated.push_back(next);
      publish(utf8.Push(tokenizer->DecodeTokenCopy(
          static_cast<tokenization::TokenId>(next))));
      if (generated.size() == max_tokens) {
        break;
      }
      if (!llm->Decode(next, &logits, &error)) {
        throw std::runtime_error(error);
      }
    }
    publish(utf8.Push("", true));
    result->completion_tokens = generated.size();
    result->decode_ms = Milliseconds(begin);
  }
};

Captioner::Captioner(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Captioner::~Captioner() = default;

std::unique_ptr<Captioner> Captioner::Create(const Options& options,
                                             std::string* error) {
  auto impl = std::make_unique<Impl>();
  impl->options = options;
  if (options.context < 64U) {
    *error = "context must be at least 64";
    return nullptr;
  }
  impl->model = AudioModel::Load(options.audio_gguf, error);
  if (impl->model == nullptr) {
    return nullptr;
  }
  if (hipStreamCreate(&impl->stream) != hipSuccess) {
    *error = "hipStreamCreate failed";
    return nullptr;
  }
  impl->encoder = hip::AudioEncoder::Create(*impl->model, impl->stream, error);
  if (impl->encoder == nullptr) {
    return nullptr;
  }
  impl->llm =
      hip::Llm::Create(options.llm_gguf, options.context, impl->stream, error);
  if (impl->llm == nullptr) {
    return nullptr;
  }
  impl->tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(impl->llm->reader(), error);
  if (impl->tokenizer == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<Captioner>(new Captioner(std::move(impl)));
}

bool Captioner::Caption(const Request& request,
                        const CancellationCheck& is_cancelled, Result* result,
                        std::string* error) {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  *result = {};
  try {
    impl_->Run(request, is_cancelled, result);
    return true;
  } catch (const std::invalid_argument& exception) {
    result->invalid_request = true;
    *error = exception.what();
    return false;
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
}

std::size_t Captioner::max_context() const { return impl_->llm->max_context(); }

std::size_t Captioner::weight_bytes() const {
  return impl_->encoder->weight_bytes() + impl_->llm->weight_bytes();
}

std::string Captioner::quantization() const {
  return impl_->llm->reader().GetQuantizationLabel();
}

}  // namespace gufo::models::moss_music
