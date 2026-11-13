#include "ming_tts/cpu/fft.hpp"

#include <numbers>
#include <stdexcept>

namespace gufo::models::ming_tts::cpu {

ComplexFft::ComplexFft(std::size_t n, int sign) : n_(n) {
  if (n == 0) {
    throw std::invalid_argument("FFT size must be positive");
  }
  std::size_t rest = n;
  for (std::size_t p = 2; p * p <= rest; ++p) {
    while (rest % p == 0) {
      factors_.push_back(p);
      rest /= p;
    }
  }
  if (rest > 1) {
    factors_.push_back(rest);
  }
  twiddles_.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double angle = static_cast<double>(sign) * 2.0 * std::numbers::pi *
                         static_cast<double>(k) / static_cast<double>(n);
    twiddles_[k] = std::complex<double>(std::cos(angle), std::sin(angle));
  }
}

void ComplexFft::Transform(const std::complex<double>* in,
                           std::size_t in_stride, std::complex<double>* out,
                           std::size_t n, std::size_t factor_index) const {
  if (n == 1) {
    out[0] = in[0];
    return;
  }
  const std::size_t p = factors_[factor_index];
  const std::size_t m = n / p;
  for (std::size_t r = 0; r < p; ++r) {
    Transform(in + r * in_stride, in_stride * p, out + r * m, m,
              factor_index + 1);
  }
  // Twiddle index for W_n^x is x * (N / n) in the size-N table.
  const std::size_t scale = n_ / n;
  std::vector<std::complex<double>> column(p);
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t r = 0; r < p; ++r) {
      column[r] = out[r * m + k];
    }
    for (std::size_t q = 0; q < p; ++q) {
      const std::size_t output_index = k + q * m;
      std::complex<double> sum = column[0];
      for (std::size_t r = 1; r < p; ++r) {
        const std::size_t exponent = (r * output_index) % n;
        sum += column[r] * twiddles_[exponent * scale];
      }
      out[output_index] = sum;
    }
  }
}

void ComplexFft::Run(const std::complex<double>* in,
                     std::complex<double>* out) const {
  Transform(in, 1, out, n_, 0);
}

RealInverseFft::RealInverseFft(std::size_t n)
    : n_(n), fft_(n, +1), full_(n), result_(n) {}

void RealInverseFft::Run(const std::complex<double>* spectrum,
                         double* out) const {
  const std::size_t half = n_ / 2;
  full_[0] = std::complex<double>(spectrum[0].real(), 0.0);
  for (std::size_t k = 1; k <= half; ++k) {
    if (k == half && n_ % 2 == 0) {
      full_[k] = std::complex<double>(spectrum[k].real(), 0.0);
    } else {
      full_[k] = spectrum[k];
      full_[n_ - k] = std::conj(spectrum[k]);
    }
  }
  fft_.Run(full_.data(), result_.data());
  const double inverse = 1.0 / static_cast<double>(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    out[i] = result_[i].real() * inverse;
  }
}

}  // namespace gufo::models::ming_tts::cpu
