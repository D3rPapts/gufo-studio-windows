// MMAudio for gufo: video / text to 44.1 kHz audio (hkchengrex/MMAudio,
// large_44k_v2, as run by ComfyUI-MMAudio).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_PIPELINE_HPP_
#define GUFO_MODELS_MMAUDIO_PIPELINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/mmaudio/clip.hpp"
#include "src/models/mmaudio/network.hpp"
#include "src/models/mmaudio/synchformer.hpp"
#include "src/models/mmaudio/tokenizer.hpp"
#include "src/models/mmaudio/vocoder.hpp"

namespace gufo::models::mmaudio {

inline constexpr int kSampleRate = 44100;

/// Files of one model directory. FromRoot() looks for (fp16 or fp32):
///   mmaudio_large_44k_v2_*.safetensors          network
///   mmaudio_vae_44k_*.safetensors               VAE
///   mmaudio_synchformer_*.safetensors           Synchformer
///   apple_DFN5B-CLIP-ViT-H-14-384_*.safetensors CLIP
///   bigvgan_v2_44khz_128band_512x.safetensors   vocoder (convert_bigvgan.py)
///   tokenizer/vocab.json + tokenizer/merges.txt CLIP BPE
struct ModelPaths {
  std::filesystem::path network, vae, synchformer, clip, vocoder, tokenizer;
  static bool FromRoot(const std::filesystem::path& root, ModelPaths* paths,
                       std::string* error);
};

struct FoleyRequest {
  std::string prompt;
  std::string negative_prompt;
  double duration{8.0};
  int steps{25};
  float cfg_strength{4.5F};
  std::uint64_t seed{0};
  /// Preprocessed frames (VideoSampler); empty = text to audio.
  std::vector<float> clip_pixels;  // [clip_frames][3][384][384]
  std::vector<float> sync_pixels;  // [sync_frames][3][224][224]
  int clip_frames{0};
  int sync_frames{0};
  /// MMAudio's image_input: one CLIP frame repeated, no Synchformer input.
  bool image_input{false};
  /// ComfyUI-MMAudio mask_away_clip: CLIP features replaced by the empty
  /// ones (the Synchformer still sees the video).
  bool mask_away_clip{false};
};

struct FoleyTelemetry {
  double features_seconds{0};
  double sampling_seconds{0};
  double decode_seconds{0};
  double duration{0};
};

class FoleyPipeline {
public:
  [[nodiscard]] static std::unique_ptr<FoleyPipeline> Load(
      const ModelPaths& paths, std::string* error);

  /// Generates mono audio (44.1 kHz, latent * 1024 samples, as
  /// ComfyUI-MMAudio returns it). `progress(fraction, stage)` returns false
  /// to cancel.
  bool Generate(const FoleyRequest& request, std::vector<float>* audio,
                FoleyTelemetry* telemetry,
                const std::function<bool(double, std::string_view)>& progress,
                std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;

  ClipEncoder& clip() { return *clip_; }
  Synchformer& synchformer() { return *synchformer_; }
  FlowNetwork& network() { return *network_; }
  VaeDecoder& vae() { return *vae_; }
  BigVgan& vocoder() { return *vocoder_; }
  const ClipTokenizer& tokenizer() const { return tokenizer_; }

private:
  FoleyPipeline() = default;

  ClipTokenizer tokenizer_;
  std::unique_ptr<ClipEncoder> clip_;
  std::unique_ptr<Synchformer> synchformer_;
  std::unique_ptr<FlowNetwork> network_;
  std::unique_ptr<VaeDecoder> vae_;
  std::unique_ptr<BigVgan> vocoder_;
};

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_PIPELINE_HPP_
