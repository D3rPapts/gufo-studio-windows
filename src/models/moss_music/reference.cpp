#include "src/models/moss_music/reference.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace gufo::models::moss_music {

/// Fixed workers that run one indexed loop at a time; the caller joins in.
class ThreadPool {
public:
  explicit ThreadPool(unsigned threads) {
    for (unsigned index = 1; index < std::max(threads, 1U); ++index) {
      workers_.emplace_back([this] { Work(); });
    }
  }

  ~ThreadPool() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (std::thread& worker : workers_) {
      worker.join();
    }
  }

  /// Calls fn(index) for every index in [0, count).
  void For(std::size_t count, const std::function<void(std::size_t)>& fn) {
    if (count == 0U) {
      return;
    }
    if (workers_.empty() || count == 1U) {
      for (std::size_t index = 0; index < count; ++index) {
        fn(index);
      }
      return;
    }
    {
      const std::lock_guard lock(mutex_);
      task_ = &fn;
      count_ = count;
      next_.store(0);
      active_ = workers_.size();
      ++generation_;
    }
    start_.notify_all();
    Drain(fn, count);
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return active_ == 0U; });
    task_ = nullptr;
  }

private:
  void Drain(const std::function<void(std::size_t)>& fn, std::size_t count) {
    for (std::size_t index = next_.fetch_add(1); index < count;
         index = next_.fetch_add(1)) {
      fn(index);
    }
  }

  void Work() {
    std::uint64_t seen = 0;
    for (;;) {
      const std::function<void(std::size_t)>* task = nullptr;
      std::size_t count = 0;
      {
        std::unique_lock lock(mutex_);
        start_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
        task = task_;
        count = count_;
      }
      Drain(*task, count);
      {
        const std::lock_guard lock(mutex_);
        --active_;
      }
      done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  const std::function<void(std::size_t)>* task_{nullptr};
  std::size_t count_{0};
  std::atomic<std::size_t> next_{0};
  std::size_t active_{0};
  std::uint64_t generation_{0};
  bool stop_{false};
};

namespace {

constexpr std::size_t kRowBlock = 16;

float Dot(const float* a, const float* b, std::size_t n) {
  float sum[8] = {};
  std::size_t index = 0;
  for (; index + 8 <= n; index += 8) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      sum[lane] += a[index + lane] * b[index + lane];
    }
  }
  float total = ((sum[0] + sum[1]) + (sum[2] + sum[3])) +
                ((sum[4] + sum[5]) + (sum[6] + sum[7]));
  for (; index < n; ++index) {
    total += a[index] * b[index];
  }
  return total;
}

/// y[t][r] = W[r] . x[t] (+ bias[r]) for t < n; y rows are `rows` apart.
void MatMul(ThreadPool& pool, const Weight& weight, const float* x,
            std::size_t n, float* y, const float* bias = nullptr) {
  const std::size_t rows = static_cast<std::size_t>(weight.rows);
  const std::size_t cols = static_cast<std::size_t>(weight.cols);
  const std::size_t blocks = (rows + kRowBlock - 1) / kRowBlock;
  pool.For(blocks, [&](std::size_t block) {
    thread_local std::vector<float> decoded;
    const std::size_t first = block * kRowBlock;
    const std::size_t count = std::min(kRowBlock, rows - first);
    decoded.resize(kRowBlock * cols);
    for (std::size_t row = 0; row < count; ++row) {
      DecodeRow(weight, first + row, decoded.data() + row * cols);
    }
    for (std::size_t token = 0; token < n; ++token) {
      const float* input = x + token * cols;
      float* output = y + token * rows + first;
      for (std::size_t row = 0; row < count; ++row) {
        output[row] = Dot(decoded.data() + row * cols, input, cols) +
                      (bias != nullptr ? bias[first + row] : 0.0F);
      }
    }
  });
}

/// nn.GELU() default: the exact erf form.
float Gelu(float x) {
  return 0.5F * x * (1.0F + std::erf(x * 0.70710678118654752440F));
}

float Silu(float x) { return x / (1.0F + std::exp(-x)); }

void LayerNorm(const float* x, const float* gamma, const float* beta,
               std::size_t rows, std::size_t cols, float eps, float* out) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* in = x + row * cols;
    double mean = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      mean += in[col];
    }
    mean /= static_cast<double>(cols);
    double variance = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      const double centered = in[col] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(cols);
    const float scale = static_cast<float>(1.0 / std::sqrt(variance + eps));
    for (std::size_t col = 0; col < cols; ++col) {
      out[row * cols + col] =
          static_cast<float>(in[col] - mean) * scale * gamma[col] + beta[col];
    }
  }
}

/// 3x3, stride 2, padding 1 convolution with bias and GELU over one chunk.
/// in [cin][h][w] -> out [cout][(h-1)/2+1][(w-1)/2+1].
void ConvGelu(ThreadPool& pool, const Weight& weight, const Weight& bias,
              const std::vector<float>& in, std::size_t cin, std::size_t h,
              std::size_t w, std::vector<float>* out, std::size_t* out_h,
              std::size_t* out_w) {
  const std::size_t oh = ConvOutLength(static_cast<std::uint32_t>(h));
  const std::size_t ow = ConvOutLength(static_cast<std::uint32_t>(w));
  const std::size_t cols = cin * 9;
  const std::size_t positions = oh * ow;
  std::vector<float> patches(positions * cols, 0.0F);
  for (std::size_t y = 0; y < oh; ++y) {
    for (std::size_t x = 0; x < ow; ++x) {
      float* patch = patches.data() + (y * ow + x) * cols;
      for (std::size_t channel = 0; channel < cin; ++channel) {
        for (std::size_t ky = 0; ky < 3; ++ky) {
          const std::ptrdiff_t sy = static_cast<std::ptrdiff_t>(2 * y + ky) - 1;
          for (std::size_t kx = 0; kx < 3; ++kx) {
            const std::ptrdiff_t sx =
                static_cast<std::ptrdiff_t>(2 * x + kx) - 1;
            if (sy >= 0 && sx >= 0 && sy < static_cast<std::ptrdiff_t>(h) &&
                sx < static_cast<std::ptrdiff_t>(w)) {
              patch[channel * 9 + ky * 3 + kx] =
                  in[(channel * h + static_cast<std::size_t>(sy)) * w +
                     static_cast<std::size_t>(sx)];
            }
          }
        }
      }
    }
  }
  const std::size_t cout = static_cast<std::size_t>(weight.rows);
  std::vector<float> result(positions * cout);
  MatMul(pool, weight, patches.data(), positions, result.data(), bias.F32());
  out->assign(cout * positions, 0.0F);
  for (std::size_t position = 0; position < positions; ++position) {
    for (std::size_t channel = 0; channel < cout; ++channel) {
      (*out)[channel * positions + position] =
          Gelu(result[position * cout + channel]);
    }
  }
  *out_h = oh;
  *out_w = ow;
}

}  // namespace

std::vector<float> SinusoidPositions(std::size_t positions,
                                     std::size_t d_model) {
  const std::size_t half = d_model / 2;
  const double increment = std::log(10000.0) / static_cast<double>(half - 1);
  std::vector<float> inv(half);
  for (std::size_t index = 0; index < half; ++index) {
    // torch.exp(-increment * arange(half).float()) in float32.
    inv[index] = std::exp(static_cast<float>(-increment) *
                          static_cast<float>(index));
  }
  std::vector<float> out(positions * d_model);
  for (std::size_t t = 0; t < positions; ++t) {
    for (std::size_t index = 0; index < half; ++index) {
      const float scaled = static_cast<float>(t) * inv[index];
      out[t * d_model + index] = std::sin(scaled);
      out[t * d_model + half + index] = std::cos(scaled);
    }
  }
  return out;
}

CpuReference::CpuReference(const AudioModel& model, unsigned threads)
    : model_(model),
      pool_(std::make_unique<ThreadPool>(
          threads == 0U ? std::thread::hardware_concurrency() : threads)) {}

CpuReference::~CpuReference() = default;

std::vector<float> CpuReference::EncodeFrontend(std::span<const float> log_mel,
                                                std::size_t frames) const {
  const AudioEncoderConfig& config = model_.config().audio;
  const StemWeights& stem = model_.audio().stem;
  const std::size_t mel = config.mel_bins;
  if (frames == 0U || log_mel.size() != mel * frames) {
    throw std::invalid_argument("MOSS-Music log-mel must be [128, frames]");
  }
  const std::size_t hidden = config.d_model;
  const std::vector<float> positions =
      SinusoidPositions(config.ChunkTokens(), hidden);
  std::vector<float> tokens;
  tokens.reserve(AudioTokenCount(static_cast<std::uint32_t>(frames), config) *
                 hidden);
  // The reference batches conv_chunksize chunks and zero-pads every chunk of
  // a batch to the batch's longest chunk before the convolutions; the padded
  // region feeds gelu(bias) into the edge of the tail's valid outputs, so the
  // tail is computed at the padded length (400 when a full chunk shares its
  // batch) and only its StemOutLength(length) rows are kept.
  const std::vector<std::uint32_t> chunks =
      ChunkFrameCounts(static_cast<std::uint32_t>(frames), config);
  const std::size_t batch_size = std::max<std::size_t>(config.conv_chunksize, 1);
  std::size_t begin = 0;
  for (std::size_t first = 0; first < chunks.size(); first += batch_size) {
    const std::size_t last = std::min(first + batch_size, chunks.size());
    const std::size_t padded =
        *std::max_element(chunks.begin() + static_cast<std::ptrdiff_t>(first),
                          chunks.begin() + static_cast<std::ptrdiff_t>(last));
    for (std::size_t index = first; index < last; ++index) {
      const std::uint32_t length = chunks[index];
      std::vector<float> input(mel * padded, 0.0F);
      for (std::size_t bin = 0; bin < mel; ++bin) {
        for (std::size_t frame = 0; frame < length; ++frame) {
          input[bin * padded + frame] = log_mel[bin * frames + begin + frame];
        }
      }
      std::vector<float> first_out;
      std::vector<float> second;
      std::size_t h = mel;
      std::size_t w = padded;
      ConvGelu(*pool_, stem.conv1_weight, stem.conv1_bias, input, 1, h, w,
               &first_out, &h, &w);
      ConvGelu(*pool_, stem.conv2_weight, stem.conv2_bias, first_out,
               config.conv_channels, h, w, &second, &h, &w);
      ConvGelu(*pool_, stem.conv3_weight, stem.conv3_bias, second,
               config.conv_channels, h, w, &first_out, &h, &w);
      // [C, F, T] -> [T, C * F]
      const std::size_t flattened = config.conv_channels * h;
      std::vector<float> features(w * flattened);
      for (std::size_t time = 0; time < w; ++time) {
        for (std::size_t channel = 0; channel < config.conv_channels;
             ++channel) {
          for (std::size_t frequency = 0; frequency < h; ++frequency) {
            features[time * flattened + channel * h + frequency] =
                first_out[(channel * h + frequency) * w + time];
          }
        }
      }
      std::vector<float> projected(w * hidden);
      MatMul(*pool_, stem.proj_weight, features.data(), w, projected.data(),
             stem.proj_bias.F32());
      const std::size_t valid = StemOutLength(length);
      if (valid > w) {
        throw std::runtime_error("MOSS-Music stem length mismatch");
      }
      for (std::size_t time = 0; time < valid; ++time) {
        for (std::size_t column = 0; column < hidden; ++column) {
          tokens.push_back(projected[time * hidden + column] +
                           positions[time * hidden + column]);
        }
      }
      begin += length;
    }
  }
  if (tokens.size() / hidden !=
      AudioTokenCount(static_cast<std::uint32_t>(frames), config)) {
    throw std::runtime_error("MOSS-Music frontend token count mismatch");
  }
  return tokens;
}

EncoderOutput CpuReference::EncodeAudio(std::span<const float> log_mel,
                                        std::size_t frames) const {
  const AudioEncoderConfig& config = model_.config().audio;
  const AudioWeights& weights = model_.audio();
  const std::size_t hidden = config.d_model;
  std::vector<float> state = EncodeFrontend(log_mel, frames);
  const std::size_t count = state.size() / hidden;
  const std::size_t chunk_tokens = config.ChunkTokens();
  const std::size_t heads = config.heads;
  const std::size_t head_dim = config.HeadDim();
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  EncoderOutput output;
  output.tokens = count;
  std::vector<float> normalized(count * hidden);
  std::vector<float> q(count * hidden);
  std::vector<float> k(count * hidden);
  std::vector<float> v(count * hidden);
  std::vector<float> attention(count * hidden);
  std::vector<float> projection(count * hidden);
  std::vector<float> ffn(count * config.ffn);
  for (std::size_t index = 0; index < weights.layers.size(); ++index) {
    const AudioLayerWeights& layer = weights.layers[index];
    LayerNorm(state.data(), layer.ln1_weight.F32(), layer.ln1_bias.F32(), count,
              hidden, config.eps, normalized.data());
    MatMul(*pool_, layer.q_weight, normalized.data(), count, q.data(),
           layer.q_bias.F32());
    MatMul(*pool_, layer.k_weight, normalized.data(), count, k.data());
    MatMul(*pool_, layer.v_weight, normalized.data(), count, v.data(),
           layer.v_bias.F32());
    // Bidirectional attention inside the token's chunk (all full chunks
    // hold chunk_tokens rows; the tail holds the remainder).
    pool_->For(count * heads, [&](std::size_t item) {
      const std::size_t token = item / heads;
      const std::size_t head = item % heads;
      const std::size_t begin = token / chunk_tokens * chunk_tokens;
      const std::size_t end = std::min(begin + chunk_tokens, count);
      const float* query = q.data() + token * hidden + head * head_dim;
      std::vector<float> scores(end - begin);
      float maximum = -INFINITY;
      for (std::size_t key = begin; key < end; ++key) {
        scores[key - begin] =
            Dot(query, k.data() + key * hidden + head * head_dim, head_dim) *
            scale;
        maximum = std::max(maximum, scores[key - begin]);
      }
      float total = 0.0F;
      for (float& score : scores) {
        score = std::exp(score - maximum);
        total += score;
      }
      float* out = attention.data() + token * hidden + head * head_dim;
      std::fill(out, out + head_dim, 0.0F);
      for (std::size_t key = begin; key < end; ++key) {
        const float weight = scores[key - begin] / total;
        const float* value = v.data() + key * hidden + head * head_dim;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          out[dim] += weight * value[dim];
        }
      }
    });
    MatMul(*pool_, layer.out_weight, attention.data(), count, projection.data(),
           layer.out_bias.F32());
    for (std::size_t i = 0; i < count * hidden; ++i) {
      state[i] += projection[i];
    }
    LayerNorm(state.data(), layer.ln2_weight.F32(), layer.ln2_bias.F32(), count,
              hidden, config.eps, normalized.data());
    MatMul(*pool_, layer.fc1_weight, normalized.data(), count, ffn.data(),
           layer.fc1_bias.F32());
    for (float& value : ffn) {
      value = Gelu(value);
    }
    MatMul(*pool_, layer.fc2_weight, ffn.data(), count, projection.data(),
           layer.fc2_bias.F32());
    for (std::size_t i = 0; i < count * hidden; ++i) {
      state[i] += projection[i];
    }
    for (const std::uint32_t tap : config.deepstack_layers) {
      if (tap == index) {
        output.deepstack.push_back(state);
      }
    }
  }
  output.final.resize(count * hidden);
  LayerNorm(state.data(), weights.post_ln_weight.F32(),
            weights.post_ln_bias.F32(), count, hidden, config.eps,
            output.final.data());
  return output;
}

std::vector<float> CpuReference::GatedMlp(const GatedMlpWeights& mlp,
                                          std::span<const float> input,
                                          std::size_t rows) const {
  const std::size_t in = static_cast<std::size_t>(mlp.gate.cols);
  const std::size_t hidden = static_cast<std::size_t>(mlp.gate.rows);
  const std::size_t out = static_cast<std::size_t>(mlp.down.rows);
  if (input.size() != rows * in) {
    throw std::invalid_argument("MOSS-Music GatedMLP input shape");
  }
  std::vector<float> gate(rows * hidden);
  std::vector<float> up(rows * hidden);
  MatMul(*pool_, mlp.gate, input.data(), rows, gate.data());
  MatMul(*pool_, mlp.up, input.data(), rows, up.data());
  for (std::size_t i = 0; i < rows * hidden; ++i) {
    gate[i] = Silu(gate[i]) * up[i];
  }
  std::vector<float> result(rows * out);
  MatMul(*pool_, mlp.down, gate.data(), rows, result.data());
  return result;
}

}  // namespace gufo::models::moss_music
