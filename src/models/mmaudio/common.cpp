// MMAudio for gufo: shared device helpers (see common.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include "src/core/json.hpp"
#include "src/models/mmaudio/kernels.hpp"

namespace gufo::models::mmaudio {
namespace {

// GEMM launches are split into this many rows to stay far below Windows'
// 2 s GPU watchdog and to bound the half scratch buffers.
constexpr int kRowChunk = 16384;

}  // namespace

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  return Fail(error, std::string("MMAudio ") + operation + ": " +
                         hipGetErrorString(status));
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

bool DeviceBuffer::Allocate(std::size_t bytes, std::string* error) {
  Release();
  const std::size_t size = std::max<std::size_t>(bytes, 256);
  if (!Check(hipMalloc(&data_, size), "device allocation", error)) {
    data_ = nullptr;
    return false;
  }
  bytes_ = size;
  return true;
}

bool DeviceBuffer::Reserve(std::size_t bytes, std::string* error) {
  if (data_ != nullptr && bytes_ >= bytes)
    return true;
  return Allocate(bytes, error);
}

void DeviceBuffer::Release() {
  if (data_ != nullptr)
    (void)hipFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x3FFU;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x3FFU;
      bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23U) | (mantissa << 13U);
  }
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

std::uint16_t FloatToHalf(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t magnitude = bits & 0x7FFFFFFFU;
  if (magnitude > 0x7F800000U)
    return static_cast<std::uint16_t>(sign | 0x7E00U);  // NaN
  if (magnitude >= 0x477FF000U)  // rounds to >= 65520: clamp to max finite
    return static_cast<std::uint16_t>(sign | 0x7BFFU);
  if (magnitude < 0x38800000U) {  // subnormal or zero in half
    if (magnitude < 0x33000000U)
      return sign;
    const std::uint32_t exponent = magnitude >> 23U;
    const std::uint32_t mantissa = (magnitude & 0x7FFFFFU) | 0x800000U;
    const std::uint32_t shift = 126U - exponent;  // 14 + (113 - exponent)
    const std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    std::uint32_t rounded = half_mantissa;
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U)))
      ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  const std::uint32_t rebased = magnitude - (112U << 23U);
  std::uint32_t half = rebased >> 13U;
  const std::uint32_t remainder = rebased & 0x1FFFU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U)))
    ++half;
  return static_cast<std::uint16_t>(sign | half);
}

float Bf16ToFloat(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

std::size_t TensorFile::Info::count() const {
  std::size_t total = 1;
  for (const std::int64_t dim : shape)
    total *= static_cast<std::size_t>(dim);
  return total;
}

TensorFile::~TensorFile() {
  if (open_)
    platform::CloseFile(file_);
}

bool TensorFile::Open(const std::filesystem::path& path, std::string* error) {
  path_ = path;
  std::string open_error;
  file_ = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file_))
    return Fail(error, "cannot open " + path.string() + ": " + open_error);
  open_ = true;
  const auto file_size = platform::FileSizeBytes(file_);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file_, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *file_size - 8)
    return Fail(error, "invalid safetensors header in " + path.string());
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file_, text.data(), text.size(), 8) != text.size())
    return Fail(error, "short read of " + path.string());
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (shape == nullptr || offsets == nullptr || offsets->size() != 2)
        return Fail(error, "invalid tensor entry " + name);
      Info info;
      info.dtype = entry.member_str("dtype");
      for (const auto& dim : shape->items())
        info.shape.push_back(static_cast<std::int64_t>(dim.as_size()));
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      const std::size_t element = info.dtype == "F32"                       ? 4
                                  : info.dtype == "F16" || info.dtype == "BF16" ? 2
                                                                            : 0;
      if (element == 0)
        continue;  // other dtypes are not read (none are needed)
      if (end < begin || end - begin != info.count() * element ||
          8 + header_size + end > *file_size)
        return Fail(error, "invalid tensor extent " + name);
      info.offset = 8 + header_size + begin;
      info.bytes = end - begin;
      tensors_.emplace(name, std::move(info));
    }
  } catch (const std::exception& e) {
    return Fail(error, path.string() + ": " + e.what());
  }
  return true;
}

const TensorFile::Info* TensorFile::Find(std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

std::vector<std::string> TensorFile::Names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& [name, info] : tensors_)
    names.push_back(name);
  return names;
}

bool TensorFile::Read(std::string_view name, std::vector<float>* out,
                      std::string* error, std::size_t expected) const {
  const Info* info = Find(name);
  if (info == nullptr)
    return Fail(error, "missing tensor " + std::string(name) + " in " +
                           path_.filename().string());
  const std::size_t count = info->count();
  if (expected != 0 && count != expected)
    return Fail(error, "tensor " + std::string(name) + " has " +
                           std::to_string(count) + " values, expected " +
                           std::to_string(expected));
  out->resize(count);
  if (info->dtype == "F32") {
    if (platform::ReadFileAt(file_, out->data(), info->bytes, info->offset) !=
        info->bytes)
      return Fail(error, "short read of " + std::string(name));
    return true;
  }
  std::vector<std::uint16_t> raw(count);
  if (platform::ReadFileAt(file_, raw.data(), info->bytes, info->offset) !=
      info->bytes)
    return Fail(error, "short read of " + std::string(name));
  const bool half = info->dtype == "F16";
  for (std::size_t i = 0; i < count; ++i)
    (*out)[i] = half ? HalfToFloat(raw[i]) : Bf16ToFloat(raw[i]);
  return true;
}

Precision DefaultPrecision() {
  const char* value = std::getenv("GUFO_MMAUDIO_F32");
  return value != nullptr && value[0] == '1' ? Precision::kF32
                                             : Precision::kF16;
}

bool UploadF32(const std::vector<float>& host, DeviceBuffer* out,
               std::string* error) {
  return out->Allocate(host.size() * 4, error) &&
         Check(hipMemcpy(out->get(), host.data(), host.size() * 4,
                         hipMemcpyHostToDevice),
               "upload", error);
}

bool LoadF32(const TensorFile& file, std::string_view name, std::size_t count,
             DeviceBuffer* out, std::string* error) {
  std::vector<float> host;
  return file.Read(name, &host, error, count) && UploadF32(host, out, error);
}

bool UploadMatrix(const std::vector<float>& host, int rows, int columns,
                  Precision precision, Matrix* out, std::string* error) {
  const std::size_t count = static_cast<std::size_t>(rows) * columns;
  if (host.size() != count)
    return Fail(error, "matrix size mismatch");
  out->rows = rows;
  out->columns = columns;
  out->f16 = precision == Precision::kF16;
  if (!out->f16)
    return UploadF32(host, &out->data, error);
  std::vector<std::uint16_t> half(count);
  for (std::size_t i = 0; i < count; ++i)
    half[i] = FloatToHalf(host[i]);
  return out->data.Allocate(count * 2, error) &&
         Check(hipMemcpy(out->data.get(), half.data(), count * 2,
                         hipMemcpyHostToDevice),
               "upload", error);
}

bool LoadMatrix(const TensorFile& file, std::string_view name, int rows,
                int columns, Precision precision, Matrix* out,
                std::string* error) {
  std::vector<float> host;
  return file.Read(name, &host, error,
                   static_cast<std::size_t>(rows) * columns) &&
         UploadMatrix(host, rows, columns, precision, out, error);
}

Blas::~Blas() {
  if (handle_ != nullptr)
    (void)rocblas_destroy_handle(handle_);
}

bool Blas::Init(hipStream_t stream, std::string* error) {
  stream_ = stream;
  if (rocblas_create_handle(&handle_) != rocblas_status_success ||
      rocblas_set_stream(handle_, stream) != rocblas_status_success)
    return Fail(error, "MMAudio rocBLAS handle");
  return true;
}

bool Blas::Linear(const Matrix& w, const float* bias, const float* in,
                  int tokens, float* out, std::string* error,
                  bool accumulate) {
  const float one = 1.0F, zero = 0.0F;
  for (int start = 0; start < tokens; start += kRowChunk) {
    const int n = std::min(kRowChunk, tokens - start);
    const float* input = in + static_cast<std::size_t>(start) * w.columns;
    float* output = out + static_cast<std::size_t>(start) * w.rows;
    if (w.f16) {
      const std::size_t in_count = static_cast<std::size_t>(n) * w.columns;
      const std::size_t out_count = static_cast<std::size_t>(n) * w.rows;
      if (!half_in_.Reserve(in_count * 2, error) ||
          !half_out_.Reserve(out_count * 2, error))
        return false;
      LaunchToHalf(input, in_count, half_in_.u16(), stream_);
      const rocblas_status status = rocblas_gemm_ex(
          handle_, rocblas_operation_transpose, rocblas_operation_none,
          w.rows, n, w.columns, &one, w.data.get(), rocblas_datatype_f16_r,
          w.columns, half_in_.get(), rocblas_datatype_f16_r, w.columns, &zero,
          half_out_.get(), rocblas_datatype_f16_r, w.rows, half_out_.get(),
          rocblas_datatype_f16_r, w.rows, rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, 0);
      if (status != rocblas_status_success)
        return Fail(error, "MMAudio half GEMM failed (" +
                               std::to_string(static_cast<int>(status)) +
                               ")");
      LaunchWiden(half_out_.u16(), static_cast<std::size_t>(n), w.rows, bias,
                  accumulate, output, stream_);
      continue;
    }
    const bool direct = bias == nullptr;
    float* target = output;
    if (!direct) {
      if (!product_.Reserve(static_cast<std::size_t>(n) * w.rows * 4, error))
        return false;
      target = product_.f32();
    }
    const float* beta = direct && accumulate ? &one : &zero;
    const rocblas_status status = rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none, w.rows,
        n, w.columns, &one, w.data.get(), rocblas_datatype_f32_r, w.columns,
        input, rocblas_datatype_f32_r, w.columns, beta, target,
        rocblas_datatype_f32_r, w.rows, target, rocblas_datatype_f32_r, w.rows,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success)
      return Fail(error, "MMAudio FP32 GEMM failed (" +
                             std::to_string(static_cast<int>(status)) + ")");
    if (!direct)
      LaunchAddBias(target, static_cast<std::size_t>(n), w.rows, bias,
                    accumulate, output, stream_);
  }
  return Check(hipGetLastError(), "linear", error);
}

bool Blas::Attention(const float* q, int q_stride, const float* k,
                     int k_stride, const float* v, int v_stride, float* out,
                     int out_stride, int lq, int lk, int heads, int head_dim,
                     float scale, bool causal, std::string* error) {
  const std::size_t score_count = static_cast<std::size_t>(heads) * lq * lk;
  if (!scores_.Reserve(score_count * 4, error))
    return false;
  const float one = 1.0F, zero = 0.0F;
  if (rocblas_sgemm_strided_batched(
          handle_, rocblas_operation_transpose, rocblas_operation_none, lk, lq,
          head_dim, &one, k, k_stride, head_dim, q, q_stride, head_dim, &zero,
          scores_.f32(), lk, static_cast<rocblas_stride>(lq) * lk, heads) !=
      rocblas_status_success)
    return Fail(error, "MMAudio attention scores failed");
  LaunchSoftmax(scores_.f32(), static_cast<std::size_t>(heads) * lq, lk, scale,
                causal ? lq : 0, stream_);
  if (rocblas_sgemm_strided_batched(
          handle_, rocblas_operation_none, rocblas_operation_none, head_dim,
          lq, lk, &one, v, v_stride, head_dim, scores_.f32(), lk,
          static_cast<rocblas_stride>(lq) * lk, &zero, out, out_stride,
          head_dim, heads) != rocblas_status_success)
    return Fail(error, "MMAudio attention values failed");
  return Check(hipGetLastError(), "attention", error);
}

}  // namespace gufo::models::mmaudio
