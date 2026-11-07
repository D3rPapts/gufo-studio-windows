// Audio front end of IndexTTS-2.5: reference loading/resampling as in
// IndexTTS2.infer_generator (torchaudio Resample to 22.05/16 kHz, librosa
// soxr_hq to 16 kHz for the emotion reference), the 80-band 22.05 kHz log-mel
// of s2mel (librosa slaney filters, reflect pad, STFT 1024/256, no center),
// the SeamlessM4T/w2v-BERT 2.0 fbank features (Kaldi mel, Povey window,
// per-bin normalisation, stride-2 stacking) and the Kaldi fbank for CAM++.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_INDEXTTS_AUDIO_HPP_
#define GUFO_MODELS_INDEXTTS_AUDIO_HPP_

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::indextts {

struct MonoAudio {
  std::vector<float> samples;
  std::int64_t rate{0};
};

// 16-bit PCM / 32-bit float WAV, channels averaged (librosa.load mono=True).
[[nodiscard]] bool ReadWavMono(const std::filesystem::path& path, MonoAudio* out,
                               std::string* error);
[[nodiscard]] bool WriteWavPcm16(const std::filesystem::path& path,
                                 std::span<const std::int16_t> samples, std::int64_t rate);

// torchaudio.transforms.Resample(orig, new) (sinc_interp_hann, width 6).
[[nodiscard]] std::vector<float> ResampleTorchaudio(std::span<const float> in, std::int64_t from,
                                                    std::int64_t to);
// librosa.resample(res_type="soxr_hq") == python-soxr resample(quality="HQ").
[[nodiscard]] std::vector<float> ResampleSoxrHq(std::span<const float> in, std::int64_t from,
                                                std::int64_t to);

// s2mel mel_spectrogram(sr 22050, n_fft 1024, hop 256, win 1024, 80 mels,
// fmin 0, fmax None, center False) -> [80, frames] (channel-major).
[[nodiscard]] std::vector<float> MelSpectrogram22k(std::span<const float> wave22k,
                                                   std::int64_t* frames);

// SeamlessM4TFeatureExtractor(audio_16k) -> input_features [rows, 160] and
// attention mask [rows] (1 = valid; the last row is 0 when the frame count
// was odd and the pad frame landed there).
struct W2vFeatures {
  std::vector<float> features;
  std::vector<std::uint8_t> mask;
  std::int64_t rows{0};
};
[[nodiscard]] W2vFeatures W2vBertFeatures(std::span<const float> wave16k);

// torchaudio.compliance.kaldi.fbank(num_mel_bins=80, dither=0) minus the
// per-bin mean over frames -> [frames, 80] (CAM++ input).
[[nodiscard]] std::vector<float> CampplusFbank(std::span<const float> wave16k,
                                               std::int64_t* frames);

}  // namespace gufo::models::indextts

#endif  // GUFO_MODELS_INDEXTTS_AUDIO_HPP_
