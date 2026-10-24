#ifndef GUFO_MODELS_QWEN3_OMNI_HIP_DEVICE_BUFFER_HPP_
#define GUFO_MODELS_QWEN3_OMNI_HIP_DEVICE_BUFFER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace gufo::models::qwen3_omni::hip {

inline void RequireHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

/// Owning device allocation of `count` elements.
template<typename Element>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Reset(count); }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void Reset(std::size_t count) {
    if (data_ != nullptr) {
      RequireHip(hipFree(data_), "hipFree Qwen3-Omni buffer");
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0U) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Qwen3-Omni HIP buffer size overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Qwen3-Omni buffer (" +
            std::to_string(count * sizeof(Element) >> 20) + " MiB)");
    count_ = count;
  }

  /// Grows (never shrinks) to at least `count` elements; contents are lost.
  void Ensure(std::size_t count) {
    if (count > count_) {
      Reset(count);
    }
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

}  // namespace gufo::models::qwen3_omni::hip

#endif  // GUFO_MODELS_QWEN3_OMNI_HIP_DEVICE_BUFFER_HPP_
