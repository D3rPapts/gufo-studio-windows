// MMAudio (hkchengrex/MMAudio) for gufo: shared device helpers.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MMAUDIO_COMMON_HPP_
#define GUFO_MODELS_MMAUDIO_COMMON_HPP_

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/platform/mapped_file.hpp"

namespace gufo::models::mmaudio {

/// Sets `error` (when not null) and returns false.
bool Fail(std::string* error, std::string message);
/// Turns a HIP status into `error`; true on success.
bool Check(hipError_t status, const char* operation, std::string* error);

/// Owning device allocation.
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer() { Release(); }
  bool Allocate(std::size_t bytes, std::string* error);
  /// Allocates only when the buffer is smaller than `bytes` (contents are
  /// not kept).
  bool Reserve(std::size_t bytes, std::string* error);
  void Release();
  [[nodiscard]] float* f32() const { return static_cast<float*>(data_); }
  [[nodiscard]] std::uint16_t* u16() const {
    return static_cast<std::uint16_t*>(data_);
  }
  [[nodiscard]] void* get() const { return data_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

/// FP32 <-> IEEE half / BF16 on the host.
float HalfToFloat(std::uint16_t value);
std::uint16_t FloatToHalf(float value);
float Bf16ToFloat(std::uint16_t value);

/// One safetensors file read on the host (F16, BF16 or F32 tensors).
class TensorFile {
public:
  struct Info {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t offset{0};  // absolute file offset
    std::uint64_t bytes{0};
    [[nodiscard]] std::size_t count() const;
  };

  TensorFile() = default;
  TensorFile(const TensorFile&) = delete;
  TensorFile& operator=(const TensorFile&) = delete;
  ~TensorFile();

  bool Open(const std::filesystem::path& path, std::string* error);
  [[nodiscard]] const Info* Find(std::string_view name) const;
  [[nodiscard]] bool Has(std::string_view name) const {
    return Find(name) != nullptr;
  }
  /// Reads the named tensor as FP32; fails when it is missing or (when
  /// `expected` > 0) has another element count.
  bool Read(std::string_view name, std::vector<float>* out, std::string* error,
            std::size_t expected = 0) const;
  [[nodiscard]] std::vector<std::string> Names() const;
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
  platform::PlatformFile file_{};
  bool open_{false};
  std::map<std::string, Info, std::less<>> tensors_;
};

/// Storage of the large weight matrices. kF16 (default) keeps them as IEEE
/// half and runs the GEMMs with half inputs, FP32 accumulation and half
/// output, like ComfyUI-MMAudio's fp16 precision; kF32 (environment
/// GUFO_MMAUDIO_F32=1) keeps FP32 everywhere.
enum class Precision { kF16, kF32 };
Precision DefaultPrecision();

/// A dense weight W[rows, columns] (row-major: torch Linear weight, or a
/// conv weight [out, in, k...] flattened to [out, in * k...]) on the device.
struct Matrix {
  DeviceBuffer data;
  int rows{0};
  int columns{0};
  bool f16{false};
  [[nodiscard]] std::size_t bytes() const { return data.bytes(); }
};

/// Uploads host FP32 values [rows, columns] in `precision`.
bool UploadMatrix(const std::vector<float>& host, int rows, int columns,
                  Precision precision, Matrix* out, std::string* error);
/// Reads a tensor of rows * columns elements and uploads it.
bool LoadMatrix(const TensorFile& file, std::string_view name, int rows,
                int columns, Precision precision, Matrix* out,
                std::string* error);
/// Uploads FP32 host values.
bool UploadF32(const std::vector<float>& host, DeviceBuffer* out,
               std::string* error);
/// Reads a tensor of `count` elements and uploads it as FP32.
bool LoadF32(const TensorFile& file, std::string_view name, std::size_t count,
             DeviceBuffer* out, std::string* error);

/// Linear layers and attention products through rocBLAS on one stream.
class Blas {
public:
  Blas() = default;
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;
  ~Blas();

  bool Init(hipStream_t stream, std::string* error);
  [[nodiscard]] rocblas_handle handle() const { return handle_; }
  [[nodiscard]] hipStream_t stream() const { return stream_; }

  /// out[tokens, w.rows] = in[tokens, w.columns] * W^T (+ bias[w.rows]),
  /// FP32 activations. With `accumulate` the product is added to `out`
  /// (bias included). F16 weights convert the input to half (clamped to
  /// +-65504) and widen the half product.
  bool Linear(const Matrix& w, const float* bias, const float* in, int tokens,
              float* out, std::string* error, bool accumulate = false);

  /// Multi-head attention in FP32: q/k/v rows of `stride` floats, head h at
  /// column offset h * head_dim (plus the pointer offset), out rows of
  /// `out_stride`. Scores softmax(scale * q k^T) with an optional causal
  /// mask; `scores` is scratch (grown to heads * lq * lk floats).
  bool Attention(const float* q, int q_stride, const float* k, int k_stride,
                 const float* v, int v_stride, float* out, int out_stride,
                 int lq, int lk, int heads, int head_dim, float scale,
                 bool causal, std::string* error);

private:
  rocblas_handle handle_{nullptr};
  hipStream_t stream_{nullptr};
  DeviceBuffer half_in_, half_out_, product_, scores_;
};

}  // namespace gufo::models::mmaudio

#endif  // GUFO_MODELS_MMAUDIO_COMMON_HPP_
