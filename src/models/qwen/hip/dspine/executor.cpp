#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/ops/token.hpp"
#include "src/models/qwen/hip/dspine.hpp"
#include "src/models/qwen/hip/dspine/kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"

namespace gufo::hip {
namespace {

/// Target feature rows projected per injection pass.
constexpr std::uint32_t kInjectRows = 16;
/// Rows the shared exact BF16 GEMM takes at once.
constexpr std::size_t kBf16GemmRows = 16;
/// Rows the quantized FP32-activation GEMM takes at once (it groups inside).
constexpr std::size_t kQuantGemmRows = 64;

}  // namespace

QwenDSpineGpuExecutor::QwenDSpineGpuExecutor(
    std::shared_ptr<const QwenDSpineGpuModel> model, std::uint32_t max_context)
    : model_(std::move(model)), max_context_(max_context) {}

QwenDSpineGpuExecutor::~QwenDSpineGpuExecutor() { Free(); }

std::unique_ptr<QwenDSpineGpuExecutor> QwenDSpineGpuExecutor::Create(
    std::shared_ptr<const QwenDSpineGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0) {
    if (error_msg) *error_msg = "DSpine executor needs a model and a context";
    return nullptr;
  }
  std::unique_ptr<QwenDSpineGpuExecutor> executor(
      new QwenDSpineGpuExecutor(std::move(model), max_context));
  try {
    executor->Allocate();
  } catch (const std::exception& e) {
    if (error_msg) *error_msg = e.what();
    return nullptr;
  }
  return executor;
}

float* QwenDSpineGpuExecutor::Scratch(std::size_t floats) {
  void* p = detail::AllocateDevice(std::max<std::size_t>(floats, 1) * sizeof(float));
  buffers_.push_back(p);
  return static_cast<float*>(p);
}

void QwenDSpineGpuExecutor::Allocate() {
  const auto& c = model_->Config();
  const std::size_t T = c.block_size, H = c.hidden, F = c.ffn, C = c.code_size,
                    R = c.rank, V = c.vocab, K = c.head_topk;
  const std::size_t qw = std::size_t{c.heads} * c.head_dim;
  const std::size_t kvw = std::size_t{c.kv_heads} * c.head_dim;
  HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  for (std::uint32_t L = 0; L < c.layers; ++L) {
    for (auto* list : {&cache_k_, &cache_v_}) {
      void* p = detail::AllocateDevice(std::size_t{max_context_} * kvw * sizeof(std::uint16_t));
      buffers_.push_back(p);
      list->push_back(static_cast<std::uint16_t*>(p));
    }
  }
  d_residual_ = Scratch(T * H);
  d_x_in_ = Scratch(T * H);
  d_q_ = Scratch(T * qw);
  d_k_ = Scratch(T * kvw);
  d_v_ = Scratch(T * kvw);
  d_att_ = Scratch(T * qw);
  d_tmp_h_ = Scratch(T * H);
  d_h_ = Scratch(T * H);
  d_hn_ = Scratch(T * H);
  d_gate_ = Scratch(T * F);
  d_up_ = Scratch(T * F);
  d_out_ = Scratch(T * H);
  d_rms_ = Scratch(T);
  d_norm_rows_ = Scratch(T * H);
  d_proj_ = Scratch(T * C);
  d_codes_ = Scratch(T * C);
  d_packed_ = Scratch(T * 3 * R);
  d_message_ = Scratch(T * R);
  d_delta_ = Scratch(T * H);
  d_scores_ = Scratch(T * c.heads * (std::size_t{max_context_} + T));
  d_head_in_ = Scratch(2 * (T - 1) * H);
  d_logits_ = Scratch(2 * (T - 1) * V);
  d_cand_values_ = Scratch((T - 1) * K);
  d_features_ = Scratch(std::size_t{kInjectRows} * c.FeatureWidth());
  d_ctx_ = Scratch(std::size_t{kInjectRows} * H);
  d_ctx_k_ = Scratch(std::size_t{kInjectRows} * kvw);
  d_ctx_v_ = Scratch(std::size_t{kInjectRows} * kvw);
  d_tokens_ = reinterpret_cast<std::uint32_t*>(Scratch(T));
  d_cand_ = reinterpret_cast<std::int32_t*>(Scratch((T - 1) * K));
  d_idx_ = reinterpret_cast<std::int32_t*>(Scratch((T - 1) * K));
}

void QwenDSpineGpuExecutor::Free() noexcept {
  for (void* p : buffers_) (void)hipFree(p);
  buffers_.clear();
  cache_k_.clear();
  cache_v_.clear();
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenDSpineGpuExecutor::Gemm(const models::QwenTensorRef& weight,
                                 const float* x, float* y, std::size_t rows,
                                 std::size_t m, std::size_t k, bool round) {
  if (weight.data == nullptr || weight.num_elements != m * k)
    throw std::logic_error("DSpine GEMM weight shape mismatch");
  const bool bf16 = weight.type == core::GgmlType::kBF16;
  const std::size_t step = bf16 ? kBf16GemmRows : kQuantGemmRows;
  for (std::size_t first = 0; first < rows; first += step) {
    const std::size_t n = std::min(step, rows - first);
    if (bf16)
      LaunchExactBf16GEMMFp32SmallBatch(weight.data, x + first * k, y + first * m,
                                        n, m, k, stream_);
    else
      LaunchBatchedQuantGEMMFp32(weight.type, weight.data, x + first * k,
                                 y + first * m, n, m, k, stream_);
  }
  if (round) dspine::LaunchRoundBf16(y, rows * m, stream_);
}

bool QwenDSpineGpuExecutor::InjectTargetContext(std::span<const float> features,
                                                std::uint32_t position,
                                                std::uint32_t count) {
  const auto& c = model_->Config();
  const std::size_t FW = c.FeatureWidth(), H = c.hidden;
  const std::size_t kvw = std::size_t{c.kv_heads} * c.head_dim;
  if (features.size() != std::size_t{count} * FW || position > context_ ||
      std::size_t{position} + count > max_context_)
    return false;
  context_ = position;  // rows past `position` are superseded
  for (std::uint32_t done = 0; done < count;) {
    const std::uint32_t n = std::min(kInjectRows, count - done);
    const std::uint32_t pos = position + done;
    HIP_CHECK(hipMemcpyAsync(d_features_, features.data() + std::size_t{done} * FW,
                             std::size_t{n} * FW * sizeof(float),
                             hipMemcpyHostToDevice, stream_));
    // dspine_mlx receives the target hidden states in BF16.
    dspine::LaunchRoundBf16(d_features_, std::size_t{n} * FW, stream_);
    Gemm(model_->Fc(), d_features_, d_ctx_, n, H, FW);
    dspine::LaunchRmsNormBf(d_ctx_, model_->HiddenNorm(), d_ctx_, n,
                            static_cast<std::uint32_t>(H), c.rms_eps, stream_);
    for (std::uint32_t L = 0; L < c.layers; ++L) {
      const auto& layer = model_->Layers()[L];
      Gemm(layer.attn_k, d_ctx_, d_ctx_k_, n, kvw, H);
      Gemm(layer.attn_v, d_ctx_, d_ctx_v_, n, kvw, H);
      dspine::LaunchHeadNormRopeBf(d_ctx_k_, static_cast<const float*>(layer.k_norm.data),
                                   n, c.kv_heads, c.head_dim, pos, c.rope_theta,
                                   c.rms_eps, stream_);
      dspine::LaunchStoreBf16Rows(d_ctx_k_, cache_k_[L], n,
                                  static_cast<std::uint32_t>(kvw), pos, stream_);
      dspine::LaunchStoreBf16Rows(d_ctx_v_, cache_v_[L], n,
                                  static_cast<std::uint32_t>(kvw), pos, stream_);
    }
    // The next chunk overwrites d_features_; the caller may free `features`.
    HIP_CHECK(hipStreamSynchronize(stream_));
    done += n;
  }
  context_ = position + count;
  return true;
}

std::vector<tokenization::TokenId> QwenDSpineGpuExecutor::ForwardBlock(
    tokenization::TokenId anchor, std::uint32_t draft_count,
    std::vector<std::int32_t>* candidates) {
  const auto& c = model_->Config();
  if (draft_count == 0) return {};
  if (draft_count + 1 > c.block_size || context_ + draft_count + 1 > max_context_ ||
      static_cast<std::uint32_t>(anchor) >= c.vocab)
    throw std::invalid_argument("DSpine block exceeds its limits");
  const std::uint32_t T = draft_count + 1, n = draft_count, S = context_;
  const std::size_t H = c.hidden, F = c.ffn, C = c.code_size, R = c.rank,
                    V = c.vocab, K = c.head_topk;
  const std::size_t qw = std::size_t{c.heads} * c.head_dim;
  const std::size_t kvw = std::size_t{c.kv_heads} * c.head_dim;
  const auto& target = model_->Target().GetWeights();

  std::vector<std::uint32_t> tokens(T, c.mask_token);
  tokens[0] = static_cast<std::uint32_t>(anchor);
  HIP_CHECK(hipMemcpyAsync(d_tokens_, tokens.data(), T * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream_));
  LaunchBatchedEmbeddingLookup(target.token_embd.data, target.token_embd.type,
                               d_tokens_, d_residual_, T, H, stream_);
  dspine::LaunchRoundBf16(d_residual_, T * H, stream_);
  dspine::LaunchRmsNormBf(d_residual_,
                          static_cast<const float*>(model_->Layers()[0].attn_norm.data),
                          d_x_in_, T, static_cast<std::uint32_t>(H), c.rms_eps, stream_);

  for (std::uint32_t L = 0; L < c.layers; ++L) {
    const auto& layer = model_->Layers()[L];
    Gemm(layer.attn_q, d_x_in_, d_q_, T, qw, H);
    Gemm(layer.attn_k, d_x_in_, d_k_, T, kvw, H);
    Gemm(layer.attn_v, d_x_in_, d_v_, T, kvw, H);
    dspine::LaunchHeadNormRopeBf(d_q_, static_cast<const float*>(layer.q_norm.data), T,
                                 c.heads, c.head_dim, S, c.rope_theta, c.rms_eps, stream_);
    dspine::LaunchHeadNormRopeBf(d_k_, static_cast<const float*>(layer.k_norm.data), T,
                                 c.kv_heads, c.head_dim, S, c.rope_theta, c.rms_eps,
                                 stream_);
    dspine::LaunchBlockAttentionBf(d_q_, cache_k_[L], cache_v_[L], d_k_, d_v_, d_att_,
                                   d_scores_, S, T, c.sliding_window, c.heads,
                                   c.kv_heads, c.head_dim, stream_);
    Gemm(layer.attn_output, d_att_, d_tmp_h_, T, H, qw);
    dspine::LaunchAddBf(d_residual_, d_tmp_h_, d_h_, T * H, stream_);
    dspine::LaunchRmsNormBf(d_h_, static_cast<const float*>(layer.ffn_norm.data), d_hn_,
                            T, static_cast<std::uint32_t>(H), c.rms_eps, stream_);
    Gemm(layer.ffn_gate, d_hn_, d_gate_, T, F, H);
    Gemm(layer.ffn_up, d_hn_, d_up_, T, F, H);
    dspine::LaunchSwiGluBf(d_gate_, d_up_, T * F, stream_);
    Gemm(layer.ffn_down, d_gate_, d_tmp_h_, T, H, F);
    dspine::LaunchAddBf(d_h_, d_tmp_h_, d_out_, T * H, stream_);

    // Code bridge (bridge.py): read, codes, packed gates, message, write.
    dspine::LaunchBridgeNormalize(d_out_, d_rms_, d_norm_rows_, T,
                                  static_cast<std::uint32_t>(H), c.bridge_eps, stream_);
    Gemm(layer.bridge_read, d_norm_rows_ + H, d_proj_ + C, n, C, H);
    dspine::LaunchBridgeCodes(d_proj_ + C, d_codes_ + C, n,
                              static_cast<std::uint32_t>(C), c.bridge_eps, stream_);
    Gemm(model_->BridgePacked(), d_codes_ + C, d_packed_ + 3 * R, n, 3 * R, C);
    // Row 0 carries the anchor's code: its packed row is a TSP table row.
    dspine::LaunchCopyBf16Row(model_->TableTsp(), static_cast<std::uint32_t>(anchor),
                              static_cast<std::uint32_t>(3 * R), d_packed_, stream_);
    dspine::LaunchBridgeMessage(d_packed_, model_->BridgeBias(), d_message_, T,
                                static_cast<std::uint32_t>(R), stream_);
    Gemm(layer.feature_write, d_message_, d_delta_, T, H, R);
    const float* next_w = L + 1 < c.layers
        ? static_cast<const float*>(model_->Layers()[L + 1].attn_norm.data)
        : model_->OutputNorm();
    // The last layer leaves d_out_, d_rms_ and d_packed_ as the tail's stash.
    dspine::LaunchBridgeWriteNorm(d_out_, d_rms_, d_delta_, next_w, d_residual_,
                                  d_x_in_, T, static_cast<std::uint32_t>(H), c.rms_eps,
                                  stream_);
  }

  // Refinement tail (propose.py RefinementTail.greedy). One pass over the
  // target head scores both the block rows and the base terms raw * norm.
  HIP_CHECK(hipMemcpyAsync(d_head_in_, d_x_in_ + H, n * H * sizeof(float),
                           hipMemcpyDeviceToDevice, stream_));
  dspine::LaunchMulRows(d_out_ + H, model_->OutputNorm(), d_head_in_ + n * H, n,
                        static_cast<std::uint32_t>(H), stream_);
  Gemm(target.output, d_head_in_, d_logits_, 2 * std::size_t{n}, V, H, false);
  dspine::LaunchTopKAscending(d_logits_, n, static_cast<std::uint32_t>(V),
                              static_cast<std::uint32_t>(K), d_cand_, d_cand_values_,
                              stream_);
  dspine::LaunchRefineScores(d_packed_, d_rms_, d_logits_ + n * V, d_cand_,
                             model_->TableTg(), model_->TableTsp(), model_->BridgeBias(),
                             static_cast<std::uint32_t>(anchor), n,
                             static_cast<std::uint32_t>(K), static_cast<std::uint32_t>(R),
                             static_cast<std::uint32_t>(V), d_idx_, stream_);
  std::vector<std::int32_t> cand(n * K), idx(n * K);
  HIP_CHECK(hipMemcpyAsync(cand.data(), d_cand_, cand.size() * sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipMemcpyAsync(idx.data(), d_idx_, idx.size() * sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));

  std::vector<tokenization::TokenId> drafts(n);
  std::int32_t j = idx[0];
  for (std::uint32_t i = 0; i < n; ++i) {
    if (j < 0 || static_cast<std::size_t>(j) >= K)
      throw std::runtime_error("DSpine refinement index out of range");
    drafts[i] = static_cast<tokenization::TokenId>(cand[i * K + static_cast<std::size_t>(j)]);
    if (i + 1 < n) j = idx[(i + 1) * K + static_cast<std::size_t>(j)];
  }
  if (candidates != nullptr) *candidates = std::move(cand);
  return drafts;
}

std::size_t QwenDSpineGpuExecutor::SnapshotPayloadBytes() const noexcept {
  const auto& c = model_->Config();
  return std::size_t{2} * c.layers * context_ * c.kv_heads * c.head_dim *
         sizeof(std::uint16_t);
}

std::unique_ptr<QwenDSpineGpuSnapshot> QwenDSpineGpuExecutor::SaveSnapshot() const {
  const auto& c = model_->Config();
  const std::size_t bytes =
      std::size_t{context_} * c.kv_heads * c.head_dim * sizeof(std::uint16_t);
  auto snapshot = std::make_unique<QwenDSpineGpuSnapshot>();
  snapshot->context = context_;
  for (std::uint32_t L = 0; L < c.layers; ++L)
    for (const auto* source : {cache_k_[L], cache_v_[L]}) {
      void* p = detail::AllocateDevice(std::max<std::size_t>(bytes, 1));
      snapshot->buffers.push_back(p);
      if (bytes)
        HIP_CHECK(hipMemcpyAsync(p, source, bytes, hipMemcpyDeviceToDevice, stream_));
    }
  snapshot->bytes = bytes * 2 * c.layers;
  HIP_CHECK(hipStreamSynchronize(stream_));
  return snapshot;
}

void QwenDSpineGpuExecutor::RestoreSnapshot(const QwenDSpineGpuSnapshot& snapshot) {
  const auto& c = model_->Config();
  if (snapshot.buffers.size() != std::size_t{2} * c.layers ||
      snapshot.context > max_context_)
    throw std::invalid_argument("DSpine snapshot does not fit this executor");
  const std::size_t bytes =
      std::size_t{snapshot.context} * c.kv_heads * c.head_dim * sizeof(std::uint16_t);
  for (std::uint32_t L = 0; L < c.layers; ++L) {
    if (!bytes) break;
    HIP_CHECK(hipMemcpyAsync(cache_k_[L], snapshot.buffers[2 * L], bytes,
                             hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(cache_v_[L], snapshot.buffers[2 * L + 1], bytes,
                             hipMemcpyDeviceToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  context_ = snapshot.context;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
