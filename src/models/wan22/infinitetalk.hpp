// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_INFINITETALK_HPP_
#define GUFO_MODELS_WAN22_INFINITETALK_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/transformer21.hpp"

namespace gufo::models::wan22 {

/// InfiniteTalk (MeiGen-AI; MultiTalk architecture) for Wan 2.1 I2V 14B as
/// ComfyUI's WanInfiniteTalkToVideo + ModelPatchLoader run it (Comfy-Org's
/// `wan2.1_infiniteTalk_{single,multi}_fp16.safetensors`):
///  * audio projection: wav2vec2 layer features (12 x 768 per 25 fps video
///    frame, a window of 5 frames around each) -> proj1 (first frame) /
///    proj1_vf (8 windows of each later group of 4 frames) -> ReLU -> proj2
///    -> ReLU -> proj3 -> LayerNorm: 32 tokens of 768 per latent frame and
///    speaker;
///  * after every DiT block's text/image cross-attention: x += audio_scale *
///    proj(attention(q_linear(LayerNorm(x)), kv_linear(audio tokens of the
///    token's latent frame))), 40 heads of 128, no QK norm;
///  * two speakers: the tokens of speaker 2 follow those of speaker 1; each
///    video token is routed by its self-attention to the first frame's
///    speaker masks (MultiTalk's x_ref_attn_map) and a 1D RoPE gives queries
///    a position in [0, 4] (speaker 1) or [20, 24] (speaker 2) and audio
///    keys the bucket centres 2 and 22.
/// Adapter matrices BF16 (from F16), norms and the audio projection FP32.
class InfiniteTalk : public Transformer21Hook {
public:
  static constexpr int kContextTokens = 32;
  static constexpr int kAudioWidth = 768;
  static constexpr int kAudioLayers = 12;
  static constexpr int kAudioWindow = 5;
  static constexpr int kMaxSpeakers = 2;

  /// Uses `stream` (the transformer's) for all device work.
  [[nodiscard]] static std::unique_ptr<InfiniteTalk> Load(
      const std::filesystem::path& file, hipStream_t stream,
      std::string* error);
  ~InfiniteTalk() override;
  InfiniteTalk(const InfiniteTalk&) = delete;
  InfiniteTalk& operator=(const InfiniteTalk&) = delete;

  /// The checkpoint's metadata model_type ("infinitetalk-single" /
  /// "infinitetalk-multi"), "" when absent.
  [[nodiscard]] const std::string& variant() const;

  /// Audio tokens of one window (project_audio_features): `features` holds
  /// one array per speaker, frame-major [feature_frames][12][768] at 25 fps
  /// (all the same length; frames past the end repeat frame 0), the window
  /// covers video frames [start, start + length), length = 4k + 1.
  bool SetAudio(const std::vector<std::span<const float>>& features,
                int feature_frames, int start, int length, std::string* error);
  /// Two-speaker routing masks over the first latent frame's token grid
  /// ([2][token_height * token_width], non-zero = the speaker; TokenMask).
  /// Empty: no routing (one speaker).
  bool SetSpeakerMasks(const std::vector<std::vector<std::uint8_t>>& masks,
                       int token_height, int token_width, std::string* error);
  void set_audio_scale(float scale);
  /// The projected tokens of the last SetAudio, [latent frames][32 *
  /// speakers][768] (oracle checks).
  bool DownloadAudioTokens(std::vector<float>* tokens, std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;

  bool Begin(const float* input, int frames, int height, int width,
             const float* temb, hipStream_t stream,
             std::string* error) override;
  bool AfterBlock(int layer, float* hidden, hipStream_t stream,
                  std::string* error) override;
  bool AfterSelfAttention(int layer, const std::uint16_t* query,
                          const std::uint16_t* key, int tokens,
                          hipStream_t stream, std::string* error) override;
  bool AfterCrossAttention(int layer, float* hidden, hipStream_t stream,
                           std::string* error) override;

private:
  struct Impl;
  explicit InfiniteTalk(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// ComfyUI's mask downsampling for InfiniteTalk: torch nearest interpolation
/// of an 8-bit mask [height, width] to the token grid [token_height,
/// token_width] (the latent size / 2), then > 0.
std::vector<std::uint8_t> TokenMask(std::span<const std::uint8_t> mask,
                                    int width, int height, int token_width,
                                    int token_height);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_INFINITETALK_HPP_
