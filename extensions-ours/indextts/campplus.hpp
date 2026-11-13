// CAM++ speaker encoder (3D-Speaker campplus_cn_common.bin, 192-dim) as used
// by IndexTTS-2.5 for the s2mel style vector and the GPT speaker condition.
// Native CPU port read from the torch state dict (BatchNorm in eval mode).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_CAMPPLUS_HPP_
#define GUFO_MODELS_INDEXTTS_CAMPPLUS_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "indextts/pth.hpp"

namespace gufo::models::indextts {

class CamPlus {
public:
  [[nodiscard]] static std::unique_ptr<CamPlus> Load(const std::filesystem::path& bin,
                                                     std::string* error);
  // fbank [frames, 80] (mean-normalised) -> 192-dim style. `head_out`
  // (optional) receives the FCM head output [320, T'] for tests.
  [[nodiscard]] std::vector<float> Embed(std::span<const float> fbank, std::int64_t frames,
                                         std::vector<float>* head_out = nullptr) const;

  // Drops this process's resident pages of the weight mapping (#829).
  void ReleaseHostPages() const noexcept { sd_->ReleaseResidentPages(); }

private:
  std::unique_ptr<PthArchive> sd_;
};

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_CAMPPLUS_HPP_
