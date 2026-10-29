#include "src/models/qwen3_omni/captioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/utf8.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_omni/hip/audio_encoder.hpp"
#include "src/models/qwen3_omni/hip/device_buffer.hpp"
#include "src/models/qwen3_omni/hip/thinker.hpp"
#include "src/models/qwen3_omni/model.hpp"
#include "src/models/qwen3_omni/reference.hpp"

namespace gufo::models::qwen3_omni {
namespace {

namespace asr = gufo::models::qwen3_asr;
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

void WriteFloats(const std::string& path, const std::vector<float>& values) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
}

/// Encoder and thinker behind one interface, so the caption loop is
/// backend-blind.
class Backend {
public:
  virtual ~Backend() = default;
  virtual std::vector<float> Encode(std::span<const float> log_mel,
                                    std::size_t frames) = 0;
  virtual void Reset() = 0;
  virtual void Forward(std::span<const float> embeddings, std::size_t count,
                       std::vector<float>* logits) = 0;
};

class CpuBackend final : public Backend {
public:
  CpuBackend(const Model& model, unsigned threads)
      : reference_(model, threads) {}
  std::vector<float> Encode(std::span<const float> log_mel,
                            std::size_t frames) override {
    return reference_.EncodeAudio(log_mel, frames);
  }
  void Reset() override { reference_.Reset(); }
  void Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* logits) override {
    reference_.Forward(embeddings, count, logits);
  }

private:
  CpuReference reference_;
};

class GpuBackend final : public Backend {
public:
  GpuBackend(const Model& model, std::size_t context)
      : width_(model.audio_config().output) {
    hip::RequireHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                    "hipStreamCreate Qwen3-Omni");
    std::string error;
    encoder_ = hip::AudioEncoder::Create(model, stream_, &error);
    if (encoder_ == nullptr) {
      throw std::runtime_error("audio encoder: " + error);
    }
    thinker_ = hip::Thinker::Create(model, context, stream_, &error);
    if (thinker_ == nullptr) {
      throw std::runtime_error("thinker: " + error);
    }
  }
  ~GpuBackend() override {
    thinker_.reset();
    encoder_.reset();
    (void)hipStreamDestroy(stream_);
  }
  std::vector<float> Encode(std::span<const float> log_mel,
                            std::size_t frames) override {
    const float* device = nullptr;
    std::size_t tokens = 0;
    std::string error;
    if (!encoder_->Encode(log_mel, frames, &device, &tokens, &error)) {
      throw std::runtime_error(error);
    }
    std::vector<float> host(tokens * width_);
    hip::RequireHip(hipMemcpy(host.data(), device, host.size() * sizeof(float),
                              hipMemcpyDeviceToHost),
                    "hipMemcpy Qwen3-Omni audio embeddings");
    return host;
  }
  void Reset() override { thinker_->Reset(); }
  void Forward(std::span<const float> embeddings, std::size_t count,
               std::vector<float>* logits) override {
    std::string error;
    if (!thinker_->Forward(embeddings, count, logits, &error)) {
      throw std::runtime_error(error);
    }
  }

private:
  std::size_t width_;
  hipStream_t stream_{nullptr};
  std::unique_ptr<hip::AudioEncoder> encoder_;
  std::unique_ptr<hip::Thinker> thinker_;
};

/// Greedy at temperature 0; otherwise top-k, then top-p, then a draw.
std::int32_t Sample(const std::vector<float>& logits,
                    const CaptionRequest& request, std::mt19937_64* rng) {
  if (request.temperature <= 0.0F) {
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  std::vector<std::int32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const std::size_t k = std::clamp<std::size_t>(request.top_k, 1, order.size());
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](std::int32_t a, std::int32_t b) {
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
  std::discrete_distribution<std::size_t> draw(probabilities.begin(),
                                               probabilities.begin() + kept);
  return order[draw(*rng)];
}

}  // namespace

struct Captioner::Impl {
  std::unique_ptr<Model> model;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  std::unique_ptr<Backend> backend;
  std::size_t context{0};
  std::mutex mutex;

  void Run(const CaptionRequest& request,
           const CancellationCheck& is_cancelled, CaptionResult* result);
};

void Captioner::Impl::Run(const CaptionRequest& request,
                          const CancellationCheck& is_cancelled,
                          CaptionResult* result) {
  std::string error;
  asr::AudioBuffer buffer;
  if (!asr::DecodeWav(request.wav, &buffer, &error)) {
    throw std::invalid_argument(error);
  }
  const std::vector<float> waveform = asr::ResampleMono16k(buffer);
  if (waveform.empty()) {
    throw std::invalid_argument("audio is empty");
  }
  const asr::LogMelFeatures features = asr::ComputeLogMelFeatures(waveform);
  result->audio_seconds = static_cast<double>(waveform.size()) / 16000.0;

  auto begin = Clock::now();
  const std::vector<float> audio =
      backend->Encode(features.values, features.frames);
  const std::size_t hidden = model->text_config().hidden;
  result->audio_tokens = audio.size() / hidden;
  result->encoder_ms = Milliseconds(begin);

  // Captioner prompt: one user turn holding only the audio.
  const TokenIds& ids = model->tokens();
  std::vector<std::int32_t> prompt;
  const auto append = [&](std::string_view text) {
    for (const auto id : tokenizer->Encode(text)) {
      prompt.push_back(static_cast<std::int32_t>(id));
    }
  };
  append("<|im_start|>user\n<|audio_start|>");
  prompt.insert(prompt.end(), result->audio_tokens, ids.audio_pad);
  append("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
  if (std::count(prompt.begin(), prompt.end(), ids.audio_pad) !=
          static_cast<std::ptrdiff_t>(result->audio_tokens) ||
      prompt.front() != ids.im_start) {
    throw std::runtime_error(
        "prompt special tokens do not match the vocabulary");
  }
  result->prompt_tokens = prompt.size();
  if (prompt.size() + 1 > context) {
    throw std::invalid_argument(
        "audio needs " + std::to_string(prompt.size()) +
        " prompt tokens, more than the context of " + std::to_string(context));
  }
  const std::size_t max_tokens =
      std::min(request.max_tokens, context - prompt.size());

  std::vector<float> embeddings(prompt.size() * hidden);
  std::size_t next_audio = 0;
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    float* row = embeddings.data() + index * hidden;
    if (prompt[index] == ids.audio_pad) {
      std::copy_n(audio.data() + next_audio++ * hidden, hidden, row);
    } else {
      EmbedToken(*model, prompt[index], row);
    }
  }

  backend->Reset();
  begin = Clock::now();
  std::vector<float> logits;
  backend->Forward(embeddings, prompt.size(), &logits);
  result->prefill_ms = Milliseconds(begin);
  if (!request.dump_prefix.empty()) {
    WriteFloats(request.dump_prefix + ".audio.f32", audio);
    WriteFloats(request.dump_prefix + ".logits.f32", logits);
  }

  begin = Clock::now();
  std::mt19937_64 rng(request.seed);
  core::Utf8Decoder utf8;
  std::vector<float> row(hidden);
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
    EmbedToken(*model, next, row.data());
    backend->Forward(row, 1, &logits);
  }
  publish(utf8.Push("", true));
  result->completion_tokens = generated.size();
  result->decode_ms = Milliseconds(begin);
  if (!request.dump_prefix.empty()) {
    std::ofstream tokens(request.dump_prefix + ".tokens.txt");
    for (const std::int32_t id : generated) {
      tokens << id << '\n';
    }
  }
}

Captioner::Captioner(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Captioner::~Captioner() = default;

std::unique_ptr<Captioner> Captioner::Create(const Options& options,
                                             std::string* error) {
  if (options.backend != "gpu" && options.backend != "cpu") {
    *error = "backend must be gpu or cpu";
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->context = options.context;
  try {
    impl->model = Model::Load(options.model, options.mmproj, error);
    if (impl->model == nullptr) {
      return nullptr;
    }
    impl->tokenizer = tokenization::QwenTokenizer::CreateFromGguf(
        impl->model->text_reader(), error);
    if (impl->tokenizer == nullptr) {
      *error = "tokenizer: " + *error;
      return nullptr;
    }
    if (options.backend == "cpu") {
      const unsigned threads =
          options.threads != 0U
              ? options.threads
              : std::max(1U, std::thread::hardware_concurrency() / 2U);
      impl->backend = std::make_unique<CpuBackend>(*impl->model, threads);
    } else {
      impl->backend = std::make_unique<GpuBackend>(*impl->model,
                                                   options.context);
    }
  } catch (const std::exception& exception) {
    *error = exception.what();
    return nullptr;
  }
  return std::unique_ptr<Captioner>(new Captioner(std::move(impl)));
}

bool Captioner::Caption(const CaptionRequest& request,
                        const CancellationCheck& is_cancelled,
                        CaptionResult* result, std::string* error) {
  std::lock_guard lock(impl_->mutex);
  *result = {};
  try {
    impl_->Run(request, is_cancelled, result);
  } catch (const std::invalid_argument& exception) {
    result->invalid_request = true;
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
  return true;
}

std::size_t Captioner::max_context() const {
  return impl_->context;
}

}  // namespace gufo::models::qwen3_omni
