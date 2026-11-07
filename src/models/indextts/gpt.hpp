// UnifiedVoice (IndexTTS-2.5 GPT, German fine-tune gpt.pth) on the CPU:
// emotion conditioning (Conformer + Perceiver + emovec/emo layers), the
// CAM++ speaker projection, the GPT-2 (24 x 1280, 20 heads) text->semantic
// AR decoder with KV cache, and the HF generate() logits pipeline
// (repetition penalty, temperature, top-k, top-p, inverse-CDF sampling).
// FP32, weights used in place from the mapped checkpoint.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_GPT_HPP_
#define GUFO_MODELS_INDEXTTS_GPT_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/indextts/pth.hpp"

namespace gufo::models::indextts {

struct GptConfig {
  std::int64_t dim{1280};
  std::int64_t layers{24};
  std::int64_t heads{20};
  std::int64_t mel_codes{8194};
  std::int32_t start_mel{8192};
  std::int32_t stop_mel{8193};
  std::int32_t start_text{0};
  std::int32_t stop_text{1};
  std::int64_t max_mel_positions{1818};
  std::int64_t max_text_positions{602};
};

struct SamplingOptions {
  float temperature{0.7F};
  std::int32_t top_k{30};
  float top_p{0.8F};
  float repetition_penalty{10.0F};
  std::int32_t max_tokens{1500};
};

using Tap = std::function<void(const std::string&, const std::vector<float>&)>;

// Penalised scores -> token, as HF generate(do_sample=True) with
// Temperature/TopK/TopP warpers and torch.multinomial replaced by an
// inverse CDF over the float32 probabilities with the uniform `u`.
[[nodiscard]] std::int32_t SampleToken(std::span<const float> penalized_scores,
                                       const SamplingOptions& options, double u,
                                       std::int32_t* kept = nullptr);
// RepetitionPenaltyLogitsProcessor over `history` (in place).
void ApplyRepetitionPenalty(std::span<float> scores, std::span<const std::int32_t> history,
                            float penalty);

class UnifiedVoice {
public:
  [[nodiscard]] static std::unique_ptr<UnifiedVoice> Load(const std::filesystem::path& gpt_pth,
                                                          std::string* error);
  [[nodiscard]] const GptConfig& config() const noexcept { return config_; }

  // get_emovec: w2v-BERT condition [rows, 1024] -> emotion vector [1280].
  [[nodiscard]] std::vector<float> EmoVec(const float* cond, std::int64_t rows,
                                          const Tap& tap = {}) const;
  // spk_emb_proj(style) -> [1280]
  [[nodiscard]] std::vector<float> SpeakerLatent(std::span<const float> style) const;
  // prepare_gpt_inputs: [cond (spk + emovec), 0, 0] + text embeddings -> [rows, 1280]
  [[nodiscard]] std::vector<float> BuildPrefix(std::span<const float> spk_latent,
                                               std::span<const float> emovec,
                                               std::span<const std::int32_t> text_ids,
                                               std::int32_t lang_token, std::int64_t* rows) const;

  struct State {
    std::vector<std::vector<float>> k;  // per layer [capacity, dim]
    std::vector<std::vector<float>> v;
    std::int64_t len{0};
    std::int64_t capacity{0};
    std::int64_t mel_tokens{0};  // tokens fed after the prefix (start token = 1)
  };
  // Feeds the prefix plus the start token; writes the logits [mel_codes].
  void Prefill(const std::vector<float>& prefix, std::int64_t rows, State* state,
               std::vector<float>* logits, std::int64_t max_new = 1500) const;
  // Feeds one generated token (mel position quirk of GPT2InferenceModel).
  void Step(std::int32_t token, State* state, std::vector<float>* logits) const;

  [[nodiscard]] const PthArchive& archive() const noexcept { return *sd_; }

private:
  void Forward(std::vector<float>& x, std::int64_t rows, State* state, float* last_logits) const;
  [[nodiscard]] const float* W(const std::string& name, std::int64_t numel = -1) const;

  std::unique_ptr<PthArchive> sd_;
  GptConfig config_;
};

// IndexTTS2.normalize_emo_vec (apply_bias=True).
[[nodiscard]] std::vector<float> NormalizeEmotionVector(std::span<const float> v);

// emovec_mat from feat1/feat2 for an 8-value emotion vector and the CAM++
// style; `index_out` receives the selected matrix rows.
class EmotionMatrix {
public:
  [[nodiscard]] static std::unique_ptr<EmotionMatrix> Load(const std::filesystem::path& feat1,
                                                           const std::filesystem::path& feat2,
                                                           std::string* error);
  [[nodiscard]] std::vector<float> Mix(std::span<const float> weights,
                                       std::span<const float> style,
                                       std::vector<std::int32_t>* index_out = nullptr) const;

private:
  std::vector<float> spk_;  // [73, 192]
  std::vector<float> emo_;  // [73, 1280]
};

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_GPT_HPP_
