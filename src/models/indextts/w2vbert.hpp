// w2v-BERT 2.0 (facebook/w2v-bert-2.0, Wav2Vec2BertModel) for IndexTTS-2.5:
// feature projection + the first 17 conformer layers (hidden_states[17]),
// normalised with wav2vec2bert_stats.pt (IndexTTS2.get_emb). FP32, weights
// used in place from the mapped model.safetensors.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_W2VBERT_HPP_
#define GUFO_MODELS_INDEXTTS_W2VBERT_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/indextts/audio.hpp"
#include "src/models/indextts/pth.hpp"
#include "src/models/ming_tts/safetensors.hpp"

namespace gufo::models::indextts {

class W2vBert {
public:
  static constexpr std::int64_t kHidden = 1024;
  static constexpr int kLayers = 17;  // hidden_states[17]

  [[nodiscard]] static std::unique_ptr<W2vBert> Load(const std::filesystem::path& model_dir,
                                                     const std::filesystem::path& stats_pt,
                                                     std::string* error);
  // features -> normalised hidden states [rows, 1024]. `tap(name, data)` (optional)
  // sees "proj", "layer0", "layer8", "hidden17".
  using Tap = std::function<void(const std::string&, const std::vector<float>&)>;
  [[nodiscard]] std::vector<float> Embed(const W2vFeatures& features, const Tap& tap = {}) const;

  [[nodiscard]] const float* F(const std::string& name, std::int64_t numel) const;

  // Drops this process's resident pages of the weight mapping (#829).
  void ReleaseHostPages() const noexcept { st_->ReleaseResidentPages(); }

private:
  std::unique_ptr<ming_tts::SafetensorsFile> st_;
  std::vector<float> mean_;
  std::vector<float> std_;
};

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_W2VBERT_HPP_
