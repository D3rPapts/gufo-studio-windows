// Mixed-radix FFT in double precision for the Ming-3.0-tts ISTFT head
// (n_fft = 3528 = 2^3 * 3^2 * 7^2) and the Kaldi fbank front end (512).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_CPU_FFT_HPP_
#define GUFO_MODELS_MING_TTS_CPU_FFT_HPP_

#include <complex>
#include <cstddef>
#include <vector>

namespace gufo::models::ming_tts::cpu {

class ComplexFft {
public:
  // sign = +1: inverse (unnormalised), -1: forward.
  ComplexFft(std::size_t n, int sign);
  void Run(const std::complex<double>* in, std::complex<double>* out) const;
  [[nodiscard]] std::size_t size() const noexcept { return n_; }

private:
  void Transform(const std::complex<double>* in, std::size_t in_stride,
                 std::complex<double>* out, std::size_t n,
                 std::size_t factor_index) const;

  std::size_t n_;
  std::vector<std::size_t> factors_;
  std::vector<std::complex<double>> twiddles_;  // exp(sign * 2 pi i k / n)
};

// torch.fft.irfft(spec, n) semantics: `bins` = n / 2 + 1 Hermitian half
// spectrum in, n real samples out (1/n normalised). The imaginary parts of the
// DC and (even-n) Nyquist bins are ignored.
class RealInverseFft {
public:
  explicit RealInverseFft(std::size_t n);
  void Run(const std::complex<double>* spectrum, double* out) const;

private:
  std::size_t n_;
  ComplexFft fft_;
  mutable std::vector<std::complex<double>> full_;
  mutable std::vector<std::complex<double>> result_;
};

}  // namespace gufo::models::ming_tts::cpu

#endif  // GUFO_MODELS_MING_TTS_CPU_FFT_HPP_
