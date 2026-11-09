#ifndef GUFO_MODELS_QWEN_HIP_DSPINE_HPP_
#define GUFO_MODELS_QWEN_HIP_DSPINE_HPP_

// GPU runner of the DSpine block draft (tencent/Qwen3.8-27B-DSpine-b16) for
// the Qwen3.8-27B target: --speculative dspine --dspine-model <gguf>.
//
// One proposal = one block of (anchor + masks) through five Qwen3 layers that
// attend to the committed target context (fc + hidden_norm over the target
// layer outputs named in the GGUF, cached as K/V per draft layer), the code
// bridge after every layer, the target head, and the greedy serial refinement
// tail. The refinement tables TG [V, R] and TSP [V, 3R] are computed once at
// load (BF16). Greedy proposals only; verification, hidden-state capture and
// the target's DeltaNet/attention rollback are the shared Qwen speculative
// verifier's (as for DFlash2). CPU oracle: src/models/qwen/dspine_reference.*.

#include <cstddef>
#include <cstdint>
#include <gufo/ext/draft_backend.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen/dspine_config.hpp"
#include "src/models/qwen/hip/executor.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::hip {

/// Immutable device weights and refinement tables of one DSpine draft.
class QwenDSpineGpuModel final {
public:
  struct Layer {
    models::QwenTensorRef attn_norm, ffn_norm, q_norm, k_norm;  // F32
    models::QwenTensorRef attn_q, attn_k, attn_v, attn_output;  // matrices
    models::QwenTensorRef ffn_gate, ffn_up, ffn_down;
    models::QwenTensorRef bridge_read;    // [C, H] (early_read[L] or read)
    models::QwenTensorRef feature_write;  // [H, R] BF16 slice of layer L
  };

  ~QwenDSpineGpuModel();
  QwenDSpineGpuModel(const QwenDSpineGpuModel&) = delete;
  QwenDSpineGpuModel& operator=(const QwenDSpineGpuModel&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenDSpineGpuModel> Create(
      std::shared_ptr<const core::GgufReader> reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const models::qwen::DSpineConfig& Config() const noexcept {
    return config_;
  }
  [[nodiscard]] const QwenGpuModel& Target() const noexcept {
    return *target_model_;
  }
  [[nodiscard]] std::span<const Layer> Layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] const models::QwenTensorRef& Fc() const noexcept { return fc_; }
  [[nodiscard]] const float* HiddenNorm() const noexcept {
    return static_cast<const float*>(hidden_norm_.data);
  }
  [[nodiscard]] const float* OutputNorm() const noexcept {
    return static_cast<const float*>(output_norm_.data);
  }
  /// [3R, C] BF16: gate, predecessor_gate and prev stacked.
  [[nodiscard]] const models::QwenTensorRef& BridgePacked() const noexcept {
    return bridge_packed_;
  }
  [[nodiscard]] const float* BridgeBias() const noexcept {
    return static_cast<const float*>(bridge_bias_.data);
  }
  /// TG [V, R]: bf16(bf16(head_v * norm) @ feature_write[last]).
  [[nodiscard]] const std::uint16_t* TableTg() const noexcept { return tg_; }
  /// TSP [V, 3R]: bf16(bf16(sqrt(C) * codebook_v) @ [gate; pred; prev]^T).
  [[nodiscard]] const std::uint16_t* TableTsp() const noexcept { return tsp_; }
  [[nodiscard]] std::size_t DeviceBytes() const noexcept { return bytes_; }
  [[nodiscard]] double TableSeconds() const noexcept { return table_seconds_; }

private:
  QwenDSpineGpuModel() = default;
  void Upload(const core::GgufReader& reader);
  void BuildTables(const core::GgufReader& reader);
  [[nodiscard]] models::QwenTensorRef Matrix(const core::GgufReader& reader,
                                             std::string_view name);
  [[nodiscard]] models::QwenTensorRef Vector(const core::GgufReader& reader,
                                             std::string_view name);
  void* Allocate(std::size_t bytes);

  std::shared_ptr<const core::GgufReader> reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  models::qwen::DSpineConfig config_;
  std::vector<Layer> layers_;
  models::QwenTensorRef fc_, hidden_norm_, output_norm_;
  models::QwenTensorRef bridge_packed_, bridge_bias_;
  std::uint16_t* tg_{nullptr};
  std::uint16_t* tsp_{nullptr};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  double table_seconds_{0.0};
};

/// Committed draft K/V of every layer at one context length (device copies).
class QwenDSpineGpuSnapshot final {
public:
  QwenDSpineGpuSnapshot() = default;
  ~QwenDSpineGpuSnapshot() {
    for (void* p : buffers) (void)hipFree(p);
  }
  QwenDSpineGpuSnapshot(const QwenDSpineGpuSnapshot&) = delete;
  QwenDSpineGpuSnapshot& operator=(const QwenDSpineGpuSnapshot&) = delete;

  std::uint32_t context{0};
  std::vector<void*> buffers;  // k0, v0, k1, v1, ...
  std::size_t bytes{0};
};

/// Per-session draft state: the context K/V of every draft layer plus scratch.
class QwenDSpineGpuExecutor final {
public:
  ~QwenDSpineGpuExecutor();
  QwenDSpineGpuExecutor(const QwenDSpineGpuExecutor&) = delete;
  QwenDSpineGpuExecutor& operator=(const QwenDSpineGpuExecutor&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenDSpineGpuExecutor> Create(
      std::shared_ptr<const QwenDSpineGpuModel> model,
      std::uint32_t max_context, std::string* error_msg = nullptr);

  void Reset() noexcept { context_ = 0; }

  /// Appends `count` committed rows of concatenated target features (width
  /// Config().FeatureWidth()) at absolute positions position.. .
  bool InjectTargetContext(std::span<const float> features,
                           std::uint32_t position, std::uint32_t count);

  /// Drafts `draft_count` (<= block_size - 1) tokens after `anchor`, which sits
  /// at position InjectedContext(). `candidates` (tests) receives the
  /// refinement candidates [draft_count, head_topk], ids ascending per row.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor, std::uint32_t draft_count,
      std::vector<std::int32_t>* candidates = nullptr);

  [[nodiscard]] std::uint32_t InjectedContext() const noexcept {
    return context_;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] const QwenDSpineGpuModel& Model() const noexcept {
    return *model_;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;
  [[nodiscard]] std::unique_ptr<QwenDSpineGpuSnapshot> SaveSnapshot() const;
  void RestoreSnapshot(const QwenDSpineGpuSnapshot& snapshot);

private:
  QwenDSpineGpuExecutor(std::shared_ptr<const QwenDSpineGpuModel> model,
                        std::uint32_t max_context);
  void Allocate();
  void Free() noexcept;
  /// y[rows, m] = bf(x[rows, k] @ W^T).
  void Gemm(const models::QwenTensorRef& weight, const float* x, float* y,
            std::size_t rows, std::size_t m, std::size_t k, bool round = true);
  float* Scratch(std::size_t floats);

  std::shared_ptr<const QwenDSpineGpuModel> model_;
  std::uint32_t max_context_{0};
  std::uint32_t context_{0};
  hipStream_t stream_{nullptr};
  std::vector<std::uint16_t*> cache_k_, cache_v_;
  std::vector<void*> buffers_;
  // Block activations [T, *].
  float *d_residual_{}, *d_x_in_{}, *d_q_{}, *d_k_{}, *d_v_{}, *d_att_{},
      *d_tmp_h_{}, *d_h_{}, *d_hn_{}, *d_gate_{}, *d_up_{}, *d_out_{},
      *d_rms_{}, *d_norm_rows_{}, *d_proj_{}, *d_codes_{}, *d_packed_{},
      *d_message_{}, *d_delta_{}, *d_scores_{}, *d_head_in_{}, *d_logits_{},
      *d_cand_values_{};
  // Injection chunk [kInjectRows, *].
  float *d_features_{}, *d_ctx_{}, *d_ctx_k_{}, *d_ctx_v_{};
  std::uint32_t* d_tokens_{nullptr};
  std::int32_t* d_cand_{nullptr};
  std::int32_t* d_idx_{nullptr};
};

struct QwenDSpineGpuDraftConfig {
  std::uint32_t max_context{4096};
  /// Draft tokens per block; clamped to block_size - 1 (15 for b16).
  std::uint32_t max_draft_tokens{15};
};

/// IDraftBackend over the executor: greedy block proposals, committed target
/// features injected lazily before the next proposal (as DFlash2 does).
class QwenDSpineGpuDraftBackend final : public speculative::IDraftBackend {
public:
  [[nodiscard]] static std::unique_ptr<QwenDSpineGpuDraftBackend> CreateFromGguf(
      std::string_view path, std::shared_ptr<const QwenGpuModel> target_model,
      QwenDSpineGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenDSpineGpuDraftBackend";
  }
  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }
  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return executor_->Model().Config().target_layers;
  }
  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;
  void UpdateTargetHidden(std::span<const float> hidden) override;
  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;
  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void Reset() noexcept override;

  /// Rounds proposed / drafted / accepted, for the measurement against DFlash2.
  struct Stats {
    std::uint64_t blocks{0}, drafted{0}, accepted{0};
  };
  [[nodiscard]] const Stats& GetStats() const noexcept { return stats_; }

private:
  QwenDSpineGpuDraftBackend(std::unique_ptr<QwenDSpineGpuExecutor> executor,
                            QwenDSpineGpuDraftConfig config)
      : executor_(std::move(executor)), config_(config) {}
  void InjectPending(std::uint32_t position);

  std::unique_ptr<QwenDSpineGpuExecutor> executor_;
  QwenDSpineGpuDraftConfig config_;
  std::vector<float> pending_;
  std::size_t proposed_{0};
  bool primed_{false};
  bool proposal_active_{false};
  Stats stats_;
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DSPINE_HPP_
