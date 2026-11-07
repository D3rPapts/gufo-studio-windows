// Speaker embedding for Ming-3.0-tts: torchaudio-compatible resampling to
// 16 kHz, Kaldi fbank (80 mel), and a native CPU port of the CAM++ network
// whose weights are read directly from the checkpoint's campplus.onnx.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_SPEAKER_HPP_
#define GUFO_MODELS_MING_TTS_SPEAKER_HPP_

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ming_tts {

// torchaudio.transforms.Resample(orig, new) defaults (sinc_interp_hann,
// lowpass_filter_width 6, rolloff 0.99; kernel built in float64, applied in
// float32).
[[nodiscard]] std::vector<float> ResampleLikeTorchaudio(
    std::span<const float> input, std::int64_t orig_rate,
    std::int64_t new_rate);

// torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, dither=0,
// sample_frequency=16000) with all other defaults; returns [frames, 80].
[[nodiscard]] std::vector<float> KaldiFbank80(std::span<const float> wave16k,
                                              std::int64_t* frames);

class CamPlusPlus {
public:
  [[nodiscard]] static std::unique_ptr<CamPlusPlus> Load(
      const std::filesystem::path& onnx_path, std::string* error);
  ~CamPlusPlus();

  // features: [frames, 80] mean-normalised fbank -> 192-dim embedding.
  [[nodiscard]] std::vector<float> Embed(std::span<const float> features,
                                         std::int64_t frames) const;

  // Full SpkembExtractor path: mono waveform at `sample_rate` -> embedding.
  [[nodiscard]] std::vector<float> EmbedWaveform(std::span<const float> mono,
                                                 std::int64_t sample_rate) const;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_SPEAKER_HPP_
