// The NPU runtime without XRT (GUFO_NPU=OFF, gufo-studio-windows): hosts
// without the XRT SDK build Gufo with this file instead of xrt.cpp and the
// XRT probe. No device is reported and opening one throws Error, so `gufo
// diagnose` warns "No XDNA NPU found" and every GPU path works as before.
// Instruction patching, the sysfs probe and dispatch libraries stay in
// common.cpp; the Q8 GEMM front end (q8_gemm.cpp) builds against this stub
// and fails at Device::Open.

#include "src/core/npu/npu.hpp"

namespace gufo::npu {
namespace {

[[noreturn]] void NoRuntime(const char* what) {
  throw Error(std::string(what) +
              ": this build has no NPU runtime (built without the XRT SDK)");
}

}  // namespace

struct Device::Impl {};
struct Buffer::Impl {};
struct Run::Impl {};
struct Program::Impl {};

std::vector<DeviceInfo> ProbeDevices() {
  return {};
}

Device::Device() = default;
Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

Device Device::Open(unsigned) {
  NoRuntime("NPU device");
}

std::string Device::name() const {
  NoRuntime("NPU device");
}

Buffer::Buffer() = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

std::size_t Buffer::size() const noexcept {
  return 0;
}

std::byte* Buffer::data() const {
  NoRuntime("NPU buffer");
}

void Buffer::ToDevice(std::size_t, std::size_t) {
  NoRuntime("NPU buffer");
}

void Buffer::FromDevice(std::size_t, std::size_t) {
  NoRuntime("NPU buffer");
}

Run::Run() = default;
Run::~Run() = default;
Run::Run(Run&&) noexcept = default;
Run& Run::operator=(Run&&) noexcept = default;

void Run::Wait() {
  NoRuntime("NPU run");
}

Program::Program() = default;
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Program Program::Load(Device&, const std::filesystem::path&, std::string_view) {
  NoRuntime("NPU program");
}

const std::string& Program::kernel_name() const noexcept {
  static const std::string none;
  return none;
}

Buffer Program::Allocate(std::size_t, unsigned, Memory) const {
  NoRuntime("NPU program");
}

Buffer Program::LoadBuffer(const std::filesystem::path&, unsigned,
                           Memory) const {
  NoRuntime("NPU program");
}

Instructions Program::AllocateInstructions(std::size_t) const {
  NoRuntime("NPU program");
}

Instructions Program::LoadInstructions(const std::filesystem::path&) const {
  NoRuntime("NPU program");
}

Run Program::Start(const Instructions&,
                   std::span<const Buffer* const>) const {
  NoRuntime("NPU program");
}

}  // namespace gufo::npu
