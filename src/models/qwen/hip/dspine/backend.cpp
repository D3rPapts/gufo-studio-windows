#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/dspine.hpp"

namespace gufo::hip {
namespace {

class QwenDSpineDraftSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  QwenDSpineDraftSnapshot(std::unique_ptr<QwenDSpineGpuSnapshot> gpu,
                          std::vector<float> pending, bool primed)
      : gpu(std::move(gpu)), pending(std::move(pending)), primed(primed) {}
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu ? gpu->bytes : 0) + pending.size() * sizeof(float);
  }
  std::unique_ptr<QwenDSpineGpuSnapshot> gpu;
  std::vector<float> pending;
  bool primed{false};
};

}  // namespace

std::unique_ptr<QwenDSpineGpuDraftBackend> QwenDSpineGpuDraftBackend::CreateFromGguf(
    std::string_view path, std::shared_ptr<const QwenGpuModel> target_model,
    QwenDSpineGpuDraftConfig config, std::string* error_msg) {
  if (path.empty() || target_model == nullptr) {
    if (error_msg) *error_msg = "DSpine GGUF path and target GPU model are required";
    return nullptr;
  }
  auto owner = core::GgufReader::OpenFile(std::string(path), error_msg);
  if (owner == nullptr) return nullptr;
  std::shared_ptr<const core::GgufReader> reader(std::move(owner));
  auto model = QwenDSpineGpuModel::Create(std::move(reader), std::move(target_model),
                                          error_msg);
  if (model == nullptr) return nullptr;
  config.max_draft_tokens =
      std::clamp(config.max_draft_tokens, 1U, model->Config().block_size - 1U);
  auto executor = QwenDSpineGpuExecutor::Create(std::move(model), config.max_context,
                                                error_msg);
  if (executor == nullptr) return nullptr;
  return std::unique_ptr<QwenDSpineGpuDraftBackend>(
      new QwenDSpineGpuDraftBackend(std::move(executor), config));
}

bool QwenDSpineGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  const std::size_t width = executor_->Model().Config().FeatureWidth();
  const std::size_t rows = context.prompt_tokens.size();
  if (rows == 0 || context.hidden_size != width ||
      context.prompt_hidden_states.size() != rows * width)
    return false;
  try {
    if (!executor_->InjectTargetContext(context.prompt_hidden_states, 0,
                                        static_cast<std::uint32_t>(rows)))
      return false;
  } catch (const std::exception&) {
    Reset();
    return false;
  }
  primed_ = true;
  return true;
}

bool QwenDSpineGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  const std::size_t width = executor_->Model().Config().FeatureWidth();
  if (!primed_ || proposal_active_ || context.hidden_size != width ||
      context.prompt_hidden_states.size() != context.prompt_tokens.size() * width)
    return false;
  InjectPending(position);
  return executor_->InjectTargetContext(
      context.prompt_hidden_states, position,
      static_cast<std::uint32_t>(context.prompt_tokens.size()));
}

void QwenDSpineGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != executor_->Model().Config().FeatureWidth())
    throw std::invalid_argument("DSpine committed target feature width is invalid");
  pending_.insert(pending_.end(), hidden.begin(), hidden.end());
}

void QwenDSpineGpuDraftBackend::InjectPending(std::uint32_t position) {
  const std::size_t width = executor_->Model().Config().FeatureWidth();
  const std::uint32_t injected = executor_->InjectedContext();
  if (position < injected)
    throw std::logic_error("DSpine committed position precedes injected history");
  const std::uint32_t count = position - injected;
  if (pending_.size() != std::size_t{count} * width)
    throw std::logic_error("DSpine committed target feature history is incomplete");
  if (count == 0) return;
  if (!executor_->InjectTargetContext(pending_, injected, count))
    throw std::runtime_error("DSpine committed target feature injection failed");
  pending_.clear();
}

speculative::DraftProposal QwenDSpineGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens, std::uint32_t current_pos,
    std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0)
    throw std::logic_error("DSpine GPU draft backend is not primed");
  if (proposal_active_)
    throw std::logic_error("DSpine GPU proposal feedback is pending");
  // Every committed row before the anchor reaches the draft K/V first; the
  // anchor (prompt_tokens.back()) sits at current_pos.
  InjectPending(current_pos);
  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t budget = current_pos + 1 < executor_->MaxContext()
                                   ? executor_->MaxContext() - current_pos - 1
                                   : 0;
  const std::uint32_t count =
      std::min({max_tokens, config_.max_draft_tokens, budget});
  if (count == 0) return proposal;
  proposal.tokens = executor_->ForwardBlock(prompt_tokens.back(), count);
  proposed_ = proposal.tokens.size();
  proposal_active_ = proposed_ != 0;
  ++stats_.blocks;
  stats_.drafted += proposed_;
  return proposal;
}

void QwenDSpineGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted, tokenization::TokenId) {
  if (!proposal_active_ || accepted.size() > proposed_)
    throw std::logic_error("DSpine GPU proposal feedback is invalid");
  stats_.accepted += accepted.size();
  proposal_active_ = false;
  proposed_ = 0;
}

std::size_t QwenDSpineGpuDraftBackend::SnapshotPayloadBytes() const {
  return executor_->SnapshotPayloadBytes() + pending_.size() * sizeof(float);
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenDSpineGpuDraftBackend::Snapshot() const {
  if (proposal_active_)
    throw std::logic_error("DSpine snapshot requires a committed proposal boundary");
  return std::make_unique<QwenDSpineDraftSnapshot>(executor_->SaveSnapshot(), pending_,
                                                   primed_);
}

void QwenDSpineGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* own = dynamic_cast<const QwenDSpineDraftSnapshot*>(&snapshot);
  const std::size_t width = executor_->Model().Config().FeatureWidth();
  if (own == nullptr || own->gpu == nullptr || own->pending.size() % width != 0)
    throw std::invalid_argument("DSpine draft snapshot is incompatible");
  executor_->RestoreSnapshot(*own->gpu);
  pending_ = own->pending;
  primed_ = own->primed;
  proposal_active_ = false;
  proposed_ = 0;
}

void QwenDSpineGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  pending_.clear();
  proposed_ = 0;
  primed_ = false;
  proposal_active_ = false;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
