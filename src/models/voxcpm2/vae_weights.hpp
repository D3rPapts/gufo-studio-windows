// Reader for the VoxCPM2 AudioVAE checkpoint: audiovae.pth (a torch.save zip
// archive with an uncompressed pickle, read without Python) or an
// audiovae.safetensors file. Old-style weight norm (weight_g / weight_v,
// dim 0) is folded into plain FP32 weights on load.
// The zip/pickle reader follows src/models/fish_drama3/loader.cpp.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VOXCPM2_VAE_WEIGHTS_HPP_
#define GUFO_MODELS_VOXCPM2_VAE_WEIGHTS_HPP_

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::voxcpm2 {

struct VaeTensor {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
};

// name -> FP32 tensor (state_dict names without the "audio_vae." prefix;
// "<conv>.weight" for folded weight-norm pairs).
using VaeWeights = std::map<std::string, VaeTensor>;

[[nodiscard]] bool LoadVaeWeights(const std::filesystem::path& model_dir,
                                  VaeWeights* out, std::string* error);

}  // namespace gufo::models::voxcpm2

#endif  // GUFO_MODELS_VOXCPM2_VAE_WEIGHTS_HPP_
