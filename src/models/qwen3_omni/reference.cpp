#include "src/models/qwen3_omni/reference.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen3_omni {

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

using core::GgmlType;

constexpr std::size_t kRowBlock = 16;

void DequantizeRow(const Weight& weight, const void* row, float* out) {
  const std::size_t cols = static_cast<std::size_t>(weight.cols);
  switch (weight.type) {
    case GgmlType::kF32:
      std::memcpy(out, row, cols * sizeof(float));
      return;
    case GgmlType::kF16: {
      const auto* half = static_cast<const std::uint16_t*>(row);
      for (std::size_t index = 0; index < cols; ++index) {
        out[index] = quant::Fp16ToFloat(half[index]);
      }
      return;
    }
    case GgmlType::kQ4_K:
      quant::DequantizeQ4_K(row, out, cols);
      return;
    case GgmlType::kQ6_K:
      quant::DequantizeQ6_K(row, out, cols);
      return;
    default:
      throw std::runtime_error("Qwen3-Omni reference: unsupported weight type");
  }
}

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

/// y[t][r] = W[expert][r] . x[t] (+ bias[r]) for t < n; y rows are
/// `y_stride` floats apart (0: W rows).
void MatMul(ThreadPool& pool, const Weight& weight, std::size_t expert,
            const float* x, std::size_t n, float* y, const float* bias = nullptr,
            std::size_t y_stride = 0) {
  const std::size_t rows = static_cast<std::size_t>(weight.rows);
  const std::size_t cols = static_cast<std::size_t>(weight.cols);
  const std::size_t stride = y_stride == 0U ? rows : y_stride;
  const std::size_t blocks = (rows + kRowBlock - 1) / kRowBlock;
  pool.For(blocks, [&](std::size_t block) {
    thread_local std::vector<float> dequantized;
    const std::size_t first = block * kRowBlock;
    const std::size_t count = std::min(kRowBlock, rows - first);
    dequantized.resize(kRowBlock * cols);
    for (std::size_t row = 0; row < count; ++row) {
      DequantizeRow(weight, weight.Row(expert, first + row),
                    dequantized.data() + row * cols);
    }
    for (std::size_t token = 0; token < n; ++token) {
      const float* input = x + token * cols;
      float* output = y + token * stride + first;
      for (std::size_t row = 0; row < count; ++row) {
        output[row] = Dot(dequantized.data() + row * cols, input, cols) +
                      (bias != nullptr ? bias[first + row] : 0.0F);
      }
    }
  });
}

float Gelu(float x) {
  return 0.5F * x * (1.0F + std::erf(x * 0.70710678118654752440F));
}

float Silu(float x) {
  return x / (1.0F + std::exp(-x));
}

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

void RmsNorm(const float* x, const float* gamma, std::size_t rows,
             std::size_t cols, float eps, float* out) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* in = x + row * cols;
    double squares = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      squares += static_cast<double>(in[col]) * in[col];
    }
    const float scale = static_cast<float>(
        1.0 / std::sqrt(squares / static_cast<double>(cols) + eps));
    for (std::size_t col = 0; col < cols; ++col) {
      out[row * cols + col] = in[col] * scale * gamma[col];
    }
  }
}

/// Rotate-half RoPE over the full head width at integer `position`.
void Rope(float* head, std::size_t dim, std::size_t position, float theta) {
  const std::size_t half = dim / 2;
  for (std::size_t index = 0; index < half; ++index) {
    const double frequency =
        std::pow(static_cast<double>(theta),
                 -2.0 * static_cast<double>(index) / static_cast<double>(dim));
    const double angle = static_cast<double>(position) * frequency;
    const float cosine = static_cast<float>(std::cos(angle));
    const float sine = static_cast<float>(std::sin(angle));
    const float first = head[index];
    const float second = head[index + half];
    head[index] = first * cosine - second * sine;
    head[index + half] = second * cosine + first * sine;
  }
}

/// 3x3, stride 2, padding 1 convolution with bias and GELU over one chunk.
/// in [cin][h][w] -> out [cout][(h-1)/2+1][(w-1)/2+1].
void ConvGelu(ThreadPool& pool, const Weight& weight, const Weight& bias,
              const std::vector<float>& in, std::size_t cin, std::size_t h,
              std::size_t w, std::vector<float>* out, std::size_t* out_h,
              std::size_t* out_w) {
  const std::size_t oh = (h - 1) / 2 + 1;
  const std::size_t ow = (w - 1) / 2 + 1;
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
  MatMul(pool, weight, 0, patches.data(), positions, result.data(),
         bias.F32());
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

CpuReference::CpuReference(const Model& model, unsigned threads)
    : model_(model), pool_(std::make_unique<ThreadPool>(threads)) {
  Reset();
}

CpuReference::~CpuReference() = default;

void CpuReference::Reset() {
  position_ = 0;
  keys_.assign(model_.text_config().layers, {});
  values_.assign(model_.text_config().layers, {});
}

std::vector<float> CpuReference::EncodeAudio(std::span<const float> log_mel,
                                             std::size_t frames) {
  const AudioConfig& config = model_.audio_config();
  const AudioWeights& weights = model_.audio();
  const std::size_t mel = config.mel_bins;
  const std::size_t chunk_frames = config.ChunkFrames();
  if (frames == 0U || log_mel.size() != mel * frames) {
    throw std::invalid_argument("Qwen3-Omni log-mel must be [128, frames]");
  }
  const std::size_t hidden = config.hidden;
  const std::size_t chunks = (frames + chunk_frames - 1) / chunk_frames;
  const std::size_t padded = std::min(frames, chunk_frames);

  // Frontend: per chunk, three strided convolutions, the flattening
  // projection and the per-chunk sinusoidal positions; padding positions of
  // the last chunk are dropped afterwards.
  std::vector<float> tokens;
  std::size_t max_valid = 0;
  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    const std::size_t begin = chunk * chunk_frames;
    const std::size_t length = std::min(chunk_frames, frames - begin);
    std::vector<float> input(mel * padded, 0.0F);
    for (std::size_t bin = 0; bin < mel; ++bin) {
      for (std::size_t frame = 0; frame < length; ++frame) {
        input[bin * padded + frame] = log_mel[bin * frames + begin + frame];
      }
    }
    std::vector<float> first;
    std::vector<float> second;
    std::size_t h = mel;
    std::size_t w = padded;
    ConvGelu(*pool_, weights.conv1_weight, weights.conv1_bias, input, 1, h, w,
             &first, &h, &w);
    ConvGelu(*pool_, weights.conv2_weight, weights.conv2_bias, first,
             config.conv_channels, h, w, &second, &h, &w);
    ConvGelu(*pool_, weights.conv3_weight, weights.conv3_bias, second,
             config.conv_channels, h, w, &first, &h, &w);
    const std::size_t flattened = config.conv_channels * h;
    std::vector<float> features(w * flattened);
    for (std::size_t time = 0; time < w; ++time) {
      for (std::size_t channel = 0; channel < config.conv_channels; ++channel) {
        for (std::size_t frequency = 0; frequency < h; ++frequency) {
          features[time * flattened + channel * h + frequency] =
              first[(channel * h + frequency) * w + time];
        }
      }
    }
    std::vector<float> projected(w * hidden);
    MatMul(*pool_, weights.conv_out, 0, features.data(), w, projected.data());
    const std::size_t valid = AudioTokenCount(length, config.n_window);
    max_valid = std::max(max_valid, valid);
    const float* positions =
        static_cast<const float*>(model_.projector_reader()
                                      .FindTensor("a.position_embd.weight")
                                      ->data);
    for (std::size_t time = 0; time < valid; ++time) {
      for (std::size_t column = 0; column < hidden; ++column) {
        tokens.push_back(projected[time * hidden + column] +
                         positions[time * hidden + column]);
      }
    }
  }
  const std::size_t count = tokens.size() / hidden;
  if (count != AudioTokenCount(frames, config.n_window)) {
    throw std::runtime_error("Qwen3-Omni audio frontend token count mismatch");
  }

  // Encoder layers with bidirectional attention inside fixed windows.
  const std::size_t window =
      max_valid * (config.n_window_infer / config.ChunkFrames());
  const std::size_t heads = config.heads;
  const std::size_t head_dim = config.HeadDim();
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> normalized(count * hidden);
  std::vector<float> q(count * hidden);
  std::vector<float> k(count * hidden);
  std::vector<float> v(count * hidden);
  std::vector<float> attention(count * hidden);
  std::vector<float> projection(count * hidden);
  std::vector<float> ffn(count * config.ffn);
  for (const AudioLayer& layer : weights.layers) {
    LayerNorm(tokens.data(), layer.ln1_weight.F32(), layer.ln1_bias.F32(),
              count, hidden, config.eps, normalized.data());
    MatMul(*pool_, layer.q_weight, 0, normalized.data(), count, q.data(),
           layer.q_bias.F32());
    MatMul(*pool_, layer.k_weight, 0, normalized.data(), count, k.data(),
           layer.k_bias.F32());
    MatMul(*pool_, layer.v_weight, 0, normalized.data(), count, v.data(),
           layer.v_bias.F32());
    pool_->For(count * heads, [&](std::size_t item) {
      const std::size_t token = item / heads;
      const std::size_t head = item % heads;
      const std::size_t begin = token / window * window;
      const std::size_t end = std::min(begin + window, count);
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
      float* output = attention.data() + token * hidden + head * head_dim;
      std::fill(output, output + head_dim, 0.0F);
      for (std::size_t key = begin; key < end; ++key) {
        const float weight = scores[key - begin] / total;
        const float* value = v.data() + key * hidden + head * head_dim;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          output[dim] += weight * value[dim];
        }
      }
    });
    MatMul(*pool_, layer.out_weight, 0, attention.data(), count,
           projection.data(), layer.out_bias.F32());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      tokens[index] += projection[index];
    }
    LayerNorm(tokens.data(), layer.ln2_weight.F32(), layer.ln2_bias.F32(),
              count, hidden, config.eps, normalized.data());
    MatMul(*pool_, layer.fc1_weight, 0, normalized.data(), count, ffn.data(),
           layer.fc1_bias.F32());
    for (float& value : ffn) {
      value = Gelu(value);
    }
    MatMul(*pool_, layer.fc2_weight, 0, ffn.data(), count, projection.data(),
           layer.fc2_bias.F32());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      tokens[index] += projection[index];
    }
  }

  LayerNorm(tokens.data(), weights.post_ln_weight.F32(),
            weights.post_ln_bias.F32(), count, hidden, config.eps,
            normalized.data());
  MatMul(*pool_, weights.proj1_weight, 0, normalized.data(), count,
         projection.data(), weights.proj1_bias.F32());
  for (float& value : projection) {
    value = Gelu(value);
  }
  std::vector<float> output(count * config.output);
  MatMul(*pool_, weights.proj2_weight, 0, projection.data(), count,
         output.data(), weights.proj2_bias.F32());
  return output;
}

void CpuReference::Embed(std::int32_t token, float* out) const {
  EmbedToken(model_, token, out);
}

void EmbedToken(const Model& model, std::int32_t token, float* out) {
  const Weight& table = model.text().token_embd;
  if (token < 0 || static_cast<std::uint64_t>(token) >= table.rows) {
    throw std::out_of_range("Qwen3-Omni token id out of range");
  }
  DequantizeRow(table, table.Row(0, static_cast<std::uint64_t>(token)), out);
}

void CpuReference::Attention(std::size_t layer, const float* x,
                             std::size_t count, float* out) {
  const TextConfig& config = model_.text_config();
  const TextLayer& weights = model_.text().layers[layer];
  const std::size_t head_dim = config.head_dim;
  const std::size_t q_dim = config.QDim();
  const std::size_t kv_dim = config.KvDim();
  std::vector<float> q(count * q_dim);
  std::vector<float> k(count * kv_dim);
  std::vector<float> v(count * kv_dim);
  MatMul(*pool_, weights.q, 0, x, count, q.data());
  MatMul(*pool_, weights.k, 0, x, count, k.data());
  MatMul(*pool_, weights.v, 0, x, count, v.data());
  for (std::size_t token = 0; token < count; ++token) {
    for (std::size_t head = 0; head < config.heads; ++head) {
      float* row = q.data() + token * q_dim + head * head_dim;
      RmsNorm(row, weights.q_norm.F32(), 1, head_dim, config.eps, row);
      Rope(row, head_dim, position_ + token, config.rope_theta);
    }
    for (std::size_t head = 0; head < config.kv_heads; ++head) {
      float* row = k.data() + token * kv_dim + head * head_dim;
      RmsNorm(row, weights.k_norm.F32(), 1, head_dim, config.eps, row);
      Rope(row, head_dim, position_ + token, config.rope_theta);
    }
  }
  std::vector<float>& keys = keys_[layer];
  std::vector<float>& values = values_[layer];
  keys.insert(keys.end(), k.begin(), k.end());
  values.insert(values.end(), v.begin(), v.end());

  const std::size_t group = config.heads / config.kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> attended(count * q_dim);
  pool_->For(count * config.heads, [&](std::size_t item) {
    const std::size_t token = item / config.heads;
    const std::size_t head = item % config.heads;
    const std::size_t kv_head = head / group;
    const std::size_t visible = position_ + token + 1;
    const float* query = q.data() + token * q_dim + head * head_dim;
    std::vector<float> scores(visible);
    float maximum = -INFINITY;
    for (std::size_t key = 0; key < visible; ++key) {
      scores[key] =
          Dot(query, keys.data() + key * kv_dim + kv_head * head_dim,
              head_dim) *
          scale;
      maximum = std::max(maximum, scores[key]);
    }
    float total = 0.0F;
    for (float& score : scores) {
      score = std::exp(score - maximum);
      total += score;
    }
    float* output = attended.data() + token * q_dim + head * head_dim;
    std::fill(output, output + head_dim, 0.0F);
    for (std::size_t key = 0; key < visible; ++key) {
      const float weight = scores[key] / total;
      const float* value = values.data() + key * kv_dim + kv_head * head_dim;
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        output[dim] += weight * value[dim];
      }
    }
  });
  MatMul(*pool_, weights.o, 0, attended.data(), count, out);
}

void CpuReference::Moe(std::size_t layer, const float* x, std::size_t count,
                       float* out) {
  const TextConfig& config = model_.text_config();
  const TextLayer& weights = model_.text().layers[layer];
  const std::size_t hidden = config.hidden;
  const std::size_t experts = config.experts;
  const std::size_t used = config.experts_used;
  const std::size_t ff = config.expert_ff;
  std::vector<float> logits(count * experts);
  MatMul(*pool_, weights.router, 0, x, count, logits.data());

  // Softmax over all experts, top-k, renormalized (norm_topk_prob).
  std::vector<std::vector<std::pair<std::size_t, float>>> assigned(experts);
  for (std::size_t token = 0; token < count; ++token) {
    const float* row = logits.data() + token * experts;
    const float maximum = *std::max_element(row, row + experts);
    std::vector<float> probabilities(experts);
    float total = 0.0F;
    for (std::size_t expert = 0; expert < experts; ++expert) {
      probabilities[expert] = std::exp(row[expert] - maximum);
      total += probabilities[expert];
    }
    std::vector<std::size_t> order(experts);
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(order.begin(), order.begin() + used, order.end(),
                      [&](std::size_t a, std::size_t b) {
                        return probabilities[a] > probabilities[b] ||
                               (probabilities[a] == probabilities[b] && a < b);
                      });
    float selected = 0.0F;
    for (std::size_t slot = 0; slot < used; ++slot) {
      selected += probabilities[order[slot]] / total;
    }
    for (std::size_t slot = 0; slot < used; ++slot) {
      assigned[order[slot]].emplace_back(
          token, probabilities[order[slot]] / total / selected);
    }
  }

  std::fill(out, out + count * hidden, 0.0F);
  for (std::size_t expert = 0; expert < experts; ++expert) {
    const auto& rows = assigned[expert];
    if (rows.empty()) {
      continue;
    }
    std::vector<float> input(rows.size() * hidden);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      std::memcpy(input.data() + row * hidden, x + rows[row].first * hidden,
                  hidden * sizeof(float));
    }
    std::vector<float> gate(rows.size() * ff);
    std::vector<float> up(rows.size() * ff);
    MatMul(*pool_, weights.gate_exps, expert, input.data(), rows.size(),
           gate.data());
    MatMul(*pool_, weights.up_exps, expert, input.data(), rows.size(),
           up.data());
    for (std::size_t index = 0; index < gate.size(); ++index) {
      gate[index] = Silu(gate[index]) * up[index];
    }
    std::vector<float> down(rows.size() * hidden);
    MatMul(*pool_, weights.down_exps, expert, gate.data(), rows.size(),
           down.data());
    for (std::size_t row = 0; row < rows.size(); ++row) {
      float* target = out + rows[row].first * hidden;
      const float weight = rows[row].second;
      for (std::size_t column = 0; column < hidden; ++column) {
        target[column] += weight * down[row * hidden + column];
      }
    }
  }
}

void CpuReference::Forward(std::span<const float> embeddings,
                           std::size_t count, std::vector<float>* logits) {
  const TextConfig& config = model_.text_config();
  const std::size_t hidden = config.hidden;
  if (count == 0U || embeddings.size() != count * hidden) {
    throw std::invalid_argument("Qwen3-Omni embeddings must be [count, hidden]");
  }
  std::vector<float> residual(embeddings.begin(), embeddings.end());
  std::vector<float> normalized(count * hidden);
  std::vector<float> update(count * hidden);
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const TextLayer& weights = model_.text().layers[layer];
    RmsNorm(residual.data(), weights.attn_norm.F32(), count, hidden,
            config.eps, normalized.data());
    Attention(layer, normalized.data(), count, update.data());
    for (std::size_t index = 0; index < residual.size(); ++index) {
      residual[index] += update[index];
    }
    RmsNorm(residual.data(), weights.ffn_norm.F32(), count, hidden, config.eps,
            normalized.data());
    Moe(layer, normalized.data(), count, update.data());
    for (std::size_t index = 0; index < residual.size(); ++index) {
      residual[index] += update[index];
    }
  }
  position_ += count;
  std::vector<float> last(hidden);
  RmsNorm(residual.data() + (count - 1) * hidden,
          model_.text().output_norm.F32(), 1, hidden, config.eps, last.data());
  logits->resize(config.vocab);
  MatMul(*pool_, model_.text().output, 0, last.data(), 1, logits->data());
}

}  // namespace gufo::models::qwen3_omni
