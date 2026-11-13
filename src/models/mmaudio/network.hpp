// MMAudio for gufo: the flow-matching multimodal transformer (large_44k_v2).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_NETWORK_HPP_
#define GUFO_MODELS_MMAUDIO_NETWORK_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/models/mmaudio/common.hpp"

namespace gufo::models::mmaudio {

/// MMAudio's SequenceConfig for 44.1 kHz (spectrogram hop 512, latent
/// downsampling 2, CLIP at 8 fps, Synchformer at 25 fps in 16-frame
/// segments every 8 frames, 8 features per segment).
struct SequenceLengths {
  int latent{0};
  int clip{0};
  int sync{0};
  /// Audio samples the vocoder returns: latent * 1024.
  [[nodiscard]] int audio_samples() const { return latent * 1024; }
};
SequenceLengths LengthsFor(double duration);

/// The network of mmaudio_large_44k_v2 (hidden 896, 14 heads of 64, 7 joint
/// blocks of latent / CLIP / text streams, the last one pre-only for CLIP and
/// text, then 14 fused latent blocks). Activations FP32; weights in the given
/// precision. One sample at a time.
class FlowNetwork {
public:
  static constexpr int kLatentDim = 40;
  static constexpr int kHidden = 896;
  static constexpr int kHeads = 14;
  static constexpr int kHeadDim = 64;
  static constexpr int kClipDim = 1024;
  static constexpr int kSyncDim = 768;
  static constexpr int kTextDim = 1024;
  static constexpr int kTextTokens = 77;

  /// Cached per-condition tensors (MMAudio's PreprocessedConditions).
  struct Conditions {
    SequenceLengths lengths;
    DeviceBuffer clip;    // [clip, 896]
    DeviceBuffer sync;    // [latent, 896], upsampled
    DeviceBuffer text;    // [77, 896]
    DeviceBuffer global;  // [896]: global_cond_mlp(clip_f_c + text_f_c)
  };

  [[nodiscard]] static std::unique_ptr<FlowNetwork> Load(
      const std::filesystem::path& path, Precision precision,
      std::string* error);
  ~FlowNetwork();
  FlowNetwork(const FlowNetwork&) = delete;
  FlowNetwork& operator=(const FlowNetwork&) = delete;

  /// preprocess_conditions. `clip` [lengths.clip][1024] and `sync`
  /// [lengths.sync][768] may be null for the learned empty features; `text`
  /// [77][1024] (null = the learned empty string features).
  bool Prepare(const SequenceLengths& lengths, const std::vector<float>* clip,
               const std::vector<float>* sync, const std::vector<float>* text,
               Conditions* out, std::string* error);

  /// predict_flow for one latent [latent, 40] (device) at time t.
  bool PredictFlow(const float* latent, float t, const Conditions& conditions,
                   float* flow, std::string* error);

  /// Euler integration from `noise` (host [latent, 40]) over `steps` with
  /// classifier-free guidance; returns the unnormalized latent
  /// (x * latent_std + latent_mean). `progress(step, steps)` returns false
  /// to cancel.
  bool Sample(const std::vector<float>& noise, int steps, float cfg_strength,
              const Conditions& positive, const Conditions& negative,
              std::vector<float>* latent,
              const std::function<bool(int, int)>& progress,
              std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;
  /// Device stream all work runs on.
  [[nodiscard]] hipStream_t stream() const;

private:
  struct Impl;
  explicit FlowNetwork(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// The rotary table of compute_rope_rotations(length, 64, 10000,
/// freq_scaling): [length][32][2] (cos, sin).
std::vector<float> RopeTable(int length, int head_dim, float freq_scaling);

/// TimestepEmbedder.timestep_embedding of the v2 model: frequencies
/// 10000 / 10000^(2i / 896) for i < 448, [cos(t f), sin(t f)].
std::vector<float> TimestepFrequencies(float t);

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_NETWORK_HPP_
