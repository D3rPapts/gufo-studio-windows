#include "src/models/indextts/audio.hpp"

#include <soxr.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>

#include "src/models/ming_tts/cpu/fft.hpp"
#include "src/models/ming_tts/speaker.hpp"

namespace gufo::models::indextts {
namespace {

std::uint32_t Le32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8U) |
         (static_cast<std::uint32_t>(p[2]) << 16U) | (static_cast<std::uint32_t>(p[3]) << 24U);
}
std::uint16_t Le16(const unsigned char* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8U));
}

// librosa: slaney mel scale
double HzToMelSlaney(double f) {
  constexpr double f_sp = 200.0 / 3.0;
  constexpr double min_log_hz = 1000.0;
  constexpr double min_log_mel = min_log_hz / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (f >= min_log_hz) {
    return min_log_mel + std::log(f / min_log_hz) / logstep;
  }
  return f / f_sp;
}
double MelToHzSlaney(double m) {
  constexpr double f_sp = 200.0 / 3.0;
  constexpr double min_log_hz = 1000.0;
  constexpr double min_log_mel = min_log_hz / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (m >= min_log_mel) {
    return min_log_hz * std::exp(logstep * (m - min_log_mel));
  }
  return f_sp * m;
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) (norm slaney), float32.
std::vector<float> LibrosaMel(double sr, int n_fft, int n_mels, double fmin, double fmax) {
  const int bins = n_fft / 2 + 1;
  std::vector<double> fftfreqs(static_cast<std::size_t>(bins));
  for (int k = 0; k < bins; ++k) {
    fftfreqs[static_cast<std::size_t>(k)] = k * sr / n_fft;
  }
  const double mmin = HzToMelSlaney(fmin);
  const double mmax = HzToMelSlaney(fmax);
  std::vector<double> mel_f(static_cast<std::size_t>(n_mels + 2));
  for (int i = 0; i < n_mels + 2; ++i) {
    const double m = mmin + (mmax - mmin) * i / (n_mels + 1);
    mel_f[static_cast<std::size_t>(i)] = MelToHzSlaney(m);
  }
  std::vector<float> w(static_cast<std::size_t>(n_mels * bins));
  for (int i = 0; i < n_mels; ++i) {
    const double fd0 = mel_f[i + 1] - mel_f[i];
    const double fd1 = mel_f[i + 2] - mel_f[i + 1];
    const double enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
    for (int k = 0; k < bins; ++k) {
      const double lower = -(mel_f[i] - fftfreqs[k]) / fd0;
      const double upper = (mel_f[i + 2] - fftfreqs[k]) / fd1;
      w[static_cast<std::size_t>(i * bins + k)] =
          static_cast<float>(std::max(0.0, std::min(lower, upper)) * enorm);
    }
  }
  return w;
}

}  // namespace

bool ReadWavMono(const std::filesystem::path& path, MonoAudio* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)), {});
  auto fail = [&](const std::string& m) {
    if (error != nullptr) {
      *error = path.string() + ": " + m;
    }
    return false;
  };
  if (buf.size() < 12 || std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    return fail("not a RIFF/WAVE file");
  }
  std::uint16_t format = 0;
  std::uint16_t channels = 0;
  std::uint16_t bits = 0;
  std::uint32_t rate = 0;
  std::size_t pos = 12;
  const unsigned char* data = nullptr;
  std::size_t data_size = 0;
  while (pos + 8 <= buf.size()) {
    const std::uint32_t size = Le32(buf.data() + pos + 4);
    if (std::memcmp(buf.data() + pos, "fmt ", 4) == 0 && pos + 8 + 16 <= buf.size()) {
      format = Le16(buf.data() + pos + 8);
      channels = Le16(buf.data() + pos + 10);
      rate = Le32(buf.data() + pos + 12);
      bits = Le16(buf.data() + pos + 22);
      if (format == 0xFFFE && size >= 40) {
        format = Le16(buf.data() + pos + 8 + 24);
      }
    } else if (std::memcmp(buf.data() + pos, "data", 4) == 0) {
      data = buf.data() + pos + 8;
      data_size = std::min<std::size_t>(size, buf.size() - pos - 8);
    }
    pos += 8 + size + (size & 1U);
  }
  if (data == nullptr || channels == 0 || rate == 0) {
    return fail("missing fmt/data chunk");
  }
  const std::size_t bytes = bits / 8;
  if (!((format == 1 && (bits == 16 || bits == 32 || bits == 24)) || (format == 3 && bits == 32))) {
    return fail("unsupported sample format");
  }
  const std::size_t frames = data_size / (bytes * channels);
  out->rate = rate;
  out->samples.assign(frames, 0.0F);
  for (std::size_t f = 0; f < frames; ++f) {
    float acc = 0.0F;
    for (std::size_t c = 0; c < channels; ++c) {
      const unsigned char* p = data + (f * channels + c) * bytes;
      float v = 0.0F;
      if (format == 3) {
        std::memcpy(&v, p, 4);
      } else if (bits == 16) {
        v = static_cast<float>(static_cast<std::int16_t>(Le16(p))) / 32768.0F;
      } else if (bits == 24) {
        std::int32_t s = static_cast<std::int32_t>(p[0] | (p[1] << 8U) | (p[2] << 16U));
        if ((s & 0x800000) != 0) {
          s -= 0x1000000;
        }
        v = static_cast<float>(s) / 8388608.0F;
      } else {
        v = static_cast<float>(static_cast<std::int32_t>(Le32(p))) / 2147483648.0F;
      }
      acc += v;
    }
    out->samples[f] = channels == 1 ? acc : acc / static_cast<float>(channels);
  }
  return true;
}

bool WriteWavPcm16(const std::filesystem::path& path, std::span<const std::int16_t> samples,
                   std::int64_t rate) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
  auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
  out.write("RIFF", 4);
  u32(36 + data_bytes);
  out.write("WAVEfmt ", 8);
  u32(16);
  u16(1);
  u16(1);
  u32(static_cast<std::uint32_t>(rate));
  u32(static_cast<std::uint32_t>(rate * 2));
  u16(2);
  u16(16);
  out.write("data", 4);
  u32(data_bytes);
  out.write(reinterpret_cast<const char*>(samples.data()), data_bytes);
  return static_cast<bool>(out);
}

std::vector<float> ResampleTorchaudio(std::span<const float> in, std::int64_t from,
                                      std::int64_t to) {
  if (from == to) {
    return {in.begin(), in.end()};
  }
  return ming_tts::ResampleLikeTorchaudio(in, from, to);
}

std::vector<float> ResampleSoxrHq(std::span<const float> in, std::int64_t from, std::int64_t to) {
  if (from == to) {
    return {in.begin(), in.end()};
  }
  // python-soxr resample(): soxr_create + soxr_process in chunks + flush.
  soxr_error_t err = nullptr;
  const soxr_io_spec_t io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
  const soxr_quality_spec_t q = soxr_quality_spec(SOXR_HQ, 0);
  soxr_t s = soxr_create(static_cast<double>(from), static_cast<double>(to), 1, &err, &io, &q,
                         nullptr);
  if (s == nullptr || err != nullptr) {
    throw std::runtime_error("soxr_create failed");
  }
  const auto expected = static_cast<std::size_t>(
      std::ceil(static_cast<double>(in.size()) * static_cast<double>(to) / static_cast<double>(from)));
  std::vector<float> out(expected + 16);
  std::size_t idone = 0;
  std::size_t odone = 0;
  err = soxr_process(s, in.data(), in.size(), &idone, out.data(), out.size(), &odone);
  std::size_t total = odone;
  while (err == nullptr && total < out.size()) {
    std::size_t more = 0;
    err = soxr_process(s, nullptr, 0, nullptr, out.data() + total, out.size() - total, &more);
    if (more == 0) {
      break;
    }
    total += more;
  }
  soxr_delete(s);
  if (err != nullptr) {
    throw std::runtime_error("soxr_process failed");
  }
  out.resize(std::min(total, expected));
  return out;
}

std::vector<float> MelSpectrogram22k(std::span<const float> wave, std::int64_t* frames) {
  constexpr int kFft = 1024;
  constexpr int kHop = 256;
  constexpr int kPad = (kFft - kHop) / 2;
  constexpr int kBins = kFft / 2 + 1;
  constexpr int kMels = 80;
  static const std::vector<float> mel = LibrosaMel(22050.0, kFft, kMels, 0.0, 11025.0);
  static const std::vector<float> window = [] {
    std::vector<float> w(kFft);
    for (int i = 0; i < kFft; ++i) {  // torch.hann_window(1024) (periodic)
      w[static_cast<std::size_t>(i)] =
          static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / kFft));
    }
    return w;
  }();
  const auto n = static_cast<std::int64_t>(wave.size());
  if (n <= kPad) {
    throw std::invalid_argument("reference audio too short for the mel spectrogram");
  }
  std::vector<float> padded(static_cast<std::size_t>(n + 2 * kPad));
  for (std::int64_t i = 0; i < n + 2 * kPad; ++i) {
    std::int64_t src = i - kPad;
    if (src < 0) {
      src = -src;
    } else if (src >= n) {
      src = 2 * (n - 1) - src;
    }
    padded[static_cast<std::size_t>(i)] = wave[static_cast<std::size_t>(src)];
  }
  const std::int64_t t = (static_cast<std::int64_t>(padded.size()) - kFft) / kHop + 1;
  *frames = t;
  std::vector<float> out(static_cast<std::size_t>(kMels * t));
  const ming_tts::cpu::ComplexFft fft(kFft, -1);
#pragma omp parallel
  {
    std::vector<std::complex<double>> buf(kFft);
    std::vector<std::complex<double>> spec(kFft);
    std::vector<float> mag(kBins);
#pragma omp for schedule(static)
    for (std::int64_t f = 0; f < t; ++f) {
      const float* x = padded.data() + f * kHop;
      for (int i = 0; i < kFft; ++i) {
        buf[static_cast<std::size_t>(i)] = {static_cast<double>(x[i] * window[static_cast<std::size_t>(i)]), 0.0};
      }
      fft.Run(buf.data(), spec.data());
      for (int k = 0; k < kBins; ++k) {
        const auto re = static_cast<float>(spec[static_cast<std::size_t>(k)].real());
        const auto im = static_cast<float>(spec[static_cast<std::size_t>(k)].imag());
        mag[static_cast<std::size_t>(k)] = std::sqrt(re * re + im * im + 1e-9F);
      }
      for (int m = 0; m < kMels; ++m) {
        const float* row = mel.data() + static_cast<std::size_t>(m) * kBins;
        float acc = 0.0F;
        for (int k = 0; k < kBins; ++k) {
          acc += row[k] * mag[static_cast<std::size_t>(k)];
        }
        out[static_cast<std::size_t>(m * t + f)] = std::log(std::max(acc, 1e-5F));
      }
    }
  }
  return out;
}

W2vFeatures W2vBertFeatures(std::span<const float> wave16k) {
  constexpr int kFrame = 400;
  constexpr int kHop = 160;
  constexpr int kFft = 512;
  constexpr int kBins = kFft / 2 + 1;
  constexpr int kMels = 80;
  static const std::vector<double> window = [] {
    std::vector<double> w(kFrame);
    for (int i = 0; i < kFrame; ++i) {
      const double hann = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / (kFrame - 1));
      w[static_cast<std::size_t>(i)] = std::pow(hann, 0.85);
    }
    return w;
  }();
  // mel_filter_bank(257, 80, 20, 8000, 16000, norm None, kaldi, triangularize in mel space)
  static const std::vector<double> filters = [] {
    auto mel = [](double f) { return 1127.0 * std::log(1.0 + f / 700.0); };
    const double mmin = mel(20.0);
    const double mmax = mel(8000.0);
    std::vector<double> mel_f(kMels + 2);
    for (int i = 0; i < kMels + 2; ++i) {
      mel_f[static_cast<std::size_t>(i)] = mmin + (mmax - mmin) * i / (kMels + 1);
    }
    const double bin_width = 16000.0 / ((kBins - 1) * 2);
    std::vector<double> f(static_cast<std::size_t>(kMels * kBins));
    for (int k = 0; k < kBins; ++k) {
      const double fk = mel(bin_width * k);
      for (int m = 0; m < kMels; ++m) {
        const double down = -(mel_f[m] - fk) / (mel_f[m + 1] - mel_f[m]);
        const double up = (mel_f[m + 2] - fk) / (mel_f[m + 2] - mel_f[m + 1]);
        f[static_cast<std::size_t>(m * kBins + k)] = std::max(0.0, std::min(down, up));
      }
    }
    return f;
  }();
  const auto n = static_cast<std::int64_t>(wave16k.size());
  if (n < kFrame) {
    throw std::invalid_argument("reference audio too short for w2v-BERT features");
  }
  const std::int64_t frames = 1 + (n - kFrame) / kHop;
  std::vector<float> feat(static_cast<std::size_t>(frames * kMels));
  const ming_tts::cpu::ComplexFft fft(kFft, -1);
#pragma omp parallel
  {
    std::vector<std::complex<double>> buf(kFft);
    std::vector<std::complex<double>> spec(kFft);
    std::vector<double> power(kBins);
#pragma omp for schedule(static)
    for (std::int64_t fr = 0; fr < frames; ++fr) {
      double frame[kFrame];
      double mean = 0.0;
      for (int i = 0; i < kFrame; ++i) {
        // waveform * 2**15 in float32, then promoted to float64
        frame[i] = static_cast<double>(wave16k[static_cast<std::size_t>(fr * kHop + i)] * 32768.0F);
        mean += frame[i];
      }
      mean /= kFrame;
      for (double& v : frame) {
        v -= mean;
      }
      for (int i = kFrame - 1; i >= 1; --i) {
        frame[i] -= 0.97 * frame[i - 1];
      }
      frame[0] *= 1.0 - 0.97;
      for (int i = 0; i < kFft; ++i) {
        buf[static_cast<std::size_t>(i)] = {i < kFrame ? frame[i] * window[static_cast<std::size_t>(i)] : 0.0, 0.0};
      }
      fft.Run(buf.data(), spec.data());
      for (int k = 0; k < kBins; ++k) {
        // stored as complex64, then |.|**2 in float64
        const auto re = static_cast<double>(static_cast<float>(spec[static_cast<std::size_t>(k)].real()));
        const auto im = static_cast<double>(static_cast<float>(spec[static_cast<std::size_t>(k)].imag()));
        power[static_cast<std::size_t>(k)] = re * re + im * im;
      }
      for (int m = 0; m < kMels; ++m) {
        double acc = 0.0;
        for (int k = 0; k < kBins; ++k) {
          acc += filters[static_cast<std::size_t>(m * kBins + k)] * power[static_cast<std::size_t>(k)];
        }
        feat[static_cast<std::size_t>(fr * kMels + m)] =
            static_cast<float>(std::log(std::max(1.192092955078125e-07, acc)));
      }
    }
  }
  // per mel bin: (x - mean) / sqrt(var(ddof=1) + 1e-7)
  for (int m = 0; m < kMels; ++m) {
    double sum = 0.0;
    for (std::int64_t f = 0; f < frames; ++f) {
      sum += feat[static_cast<std::size_t>(f * kMels + m)];
    }
    const auto mean = static_cast<float>(sum / static_cast<double>(frames));
    double sq = 0.0;
    for (std::int64_t f = 0; f < frames; ++f) {
      const double d = static_cast<double>(feat[static_cast<std::size_t>(f * kMels + m)]) - mean;
      sq += d * d;
    }
    const auto var = static_cast<float>(sq / static_cast<double>(std::max<std::int64_t>(1, frames - 1)));
    const float denom = std::sqrt(var + 1e-7F);
    for (std::int64_t f = 0; f < frames; ++f) {
      float& v = feat[static_cast<std::size_t>(f * kMels + m)];
      v = (v - mean) / denom;
    }
  }
  // pad_to_multiple_of=2 with padding_value 1.0, then stride-2 stacking
  W2vFeatures out;
  const std::int64_t padded = frames + (frames % 2);
  std::vector<std::uint8_t> frame_mask(static_cast<std::size_t>(padded), 1);
  if (padded != frames) {
    feat.resize(static_cast<std::size_t>(padded * kMels), 1.0F);
    frame_mask.back() = 0;
  }
  out.rows = padded / 2;
  out.features = std::move(feat);  // [padded, 80] == [rows, 160]
  out.mask.resize(static_cast<std::size_t>(out.rows));
  for (std::int64_t r = 0; r < out.rows; ++r) {
    out.mask[static_cast<std::size_t>(r)] = frame_mask[static_cast<std::size_t>(2 * r + 1)];
  }
  return out;
}

std::vector<float> CampplusFbank(std::span<const float> wave16k, std::int64_t* frames) {
  std::vector<float> feat = ming_tts::KaldiFbank80(wave16k, frames);
  const std::int64_t t = *frames;
  for (int f = 0; f < 80; ++f) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < t; ++i) {
      sum += feat[static_cast<std::size_t>(i * 80 + f)];
    }
    const auto mean = static_cast<float>(sum / static_cast<double>(t));
    for (std::int64_t i = 0; i < t; ++i) {
      feat[static_cast<std::size_t>(i * 80 + f)] -= mean;
    }
  }
  return feat;
}

}  // namespace gufo::models::indextts
