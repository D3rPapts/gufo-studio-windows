#include "src/models/minimax_h3/prompt_vision.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <bit>
#include <cstdint>
#include <exception>

#include "src/core/image.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#endif

namespace gufo::minimax_h3 {

#if defined(ENGINE_ENABLE_HIP)
namespace {

// The tower rounds its outputs to BF16 already; keep those bits.
bool ReadRows(const float* device, std::size_t elements,
              std::vector<std::uint16_t>* output, std::string* error) {
  std::vector<float> host(elements);
  const hipError_t status = hipMemcpy(
      host.data(), device, elements * sizeof(float), hipMemcpyDeviceToHost);
  if (status != hipSuccess) {
    *error = std::string("read H3 keyframe vision features: ") +
             hipGetErrorString(status);
    return false;
  }
  output->resize(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(host[index]);
    (*output)[index] = static_cast<std::uint16_t>(
        (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
  }
  return true;
}

}  // namespace

bool EncodeKeyframeVision(const std::filesystem::path& projector,
                          std::span<const KeyframeImage> images,
                          const CancellationToken* cancellation,
                          std::vector<PromptVisionBlock>* blocks,
                          std::string* error) {
  std::string local_error;
  if (error == nullptr) {
    error = &local_error;
  }
  blocks->clear();
  try {
    models::qwen::vision::Encoder encoder(
        projector, static_cast<std::uint32_t>(kPromptEncoderHiddenSize));
    for (const KeyframeImage& image : images) {
      if (image.width % 32 != 0 || image.height % 32 != 0) {
        *error = "MiniMax H3 keyframe vision needs 32-pixel canvas sides";
        return false;
      }
      core::Image canvas;
      canvas.width = static_cast<std::uint32_t>(image.width);
      canvas.height = static_cast<std::uint32_t>(image.height);
      canvas.pixels.assign(image.rgb.begin(), image.rgb.end());
      const auto embedding = encoder.Encode(canvas, {}, [cancellation] {
        return cancellation != nullptr && cancellation->IsCancelled();
      });
      PromptVisionBlock block;
      block.grid_height = canvas.height / 32U;
      block.grid_width = canvas.width / 32U;
      const std::size_t elements =
          static_cast<std::size_t>(embedding->rows()) * embedding->width();
      if (embedding->rows() != block.grid_height * block.grid_width ||
          embedding->width() != kPromptEncoderHiddenSize ||
          embedding->deepstack().size() != kVisionDeepstackLayers) {
        *error =
            "MiniMax H3 vision projector must be the Qwen3-VL tower with "
            "deepstack layers 8, 16 and 24";
        return false;
      }
      if (!ReadRows(embedding->data(), elements, &block.embedding, error)) {
        return false;
      }
      for (const float* features : embedding->deepstack()) {
        block.deepstack.emplace_back();
        if (!ReadRows(features, elements, &block.deepstack.back(), error)) {
          return false;
        }
      }
      blocks->push_back(std::move(block));
    }
  } catch (const std::exception& exception) {
    *error =
        std::string("MiniMax H3 keyframe vision failed: ") + exception.what();
    return false;
  }
  return true;
}
#else
bool EncodeKeyframeVision(const std::filesystem::path&,
                          std::span<const KeyframeImage>,
                          const CancellationToken*,
                          std::vector<PromptVisionBlock>*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 keyframe vision requires ENGINE_ENABLE_HIP";
  }
  return false;
}
#endif

namespace {

constexpr std::uint32_t kRequestMagic = 0x51503348U;   // "H3PQ"
constexpr std::uint32_t kResponseMagic = 0x52503348U;  // "H3PR"
constexpr std::uint64_t kMaximumPayloadBytes = 1ULL << 32U;

template<typename T>
void Put(std::ostream& output, T value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template<typename T>
bool Get(std::istream& input, T* value) {
  input.read(reinterpret_cast<char*>(value), sizeof(*value));
  return static_cast<bool>(input);
}

template<typename T>
void PutVector(std::ostream& output, std::span<const T> values) {
  Put<std::uint64_t>(output, values.size());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
}

template<typename T>
bool GetVector(std::istream& input, std::vector<T>* values) {
  std::uint64_t count = 0;
  if (!Get(input, &count) || count > kMaximumPayloadBytes / sizeof(T)) {
    return false;
  }
  values->resize(static_cast<std::size_t>(count));
  input.read(reinterpret_cast<char*>(values->data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  return static_cast<bool>(input);
}

void PutPath(std::ostream& output, const std::filesystem::path& path) {
  const auto& native = path.native();
  PutVector(output, std::span<const std::filesystem::path::value_type>(
                        native.data(), native.size()));
}

bool GetPath(std::istream& input, std::filesystem::path* path) {
  std::vector<std::filesystem::path::value_type> native;
  if (!GetVector(input, &native)) {
    return false;
  }
  *path = std::filesystem::path::string_type(native.begin(), native.end());
  return true;
}

struct PromptRequest {
  std::filesystem::path model_root;
  std::filesystem::path manifest;
  std::filesystem::path projector;
  std::vector<std::int32_t> sizes;  // width, height per image
  std::vector<std::vector<std::uint8_t>> pixels;
  std::vector<std::uint32_t> tokens;
};

bool WriteRequest(const std::filesystem::path& path,
                  const PromptIsolationInput& input) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Put(output, kRequestMagic);
  PutPath(output, input.model_root);
  PutPath(output, input.manifest);
  PutPath(output, input.projector);
  Put<std::uint32_t>(output, static_cast<std::uint32_t>(input.images.size()));
  for (const KeyframeImage& image : input.images) {
    Put<std::int32_t>(output, image.width);
    Put<std::int32_t>(output, image.height);
    PutVector(output, image.rgb);
  }
  PutVector(output, input.tokens);
  Put<std::uint32_t>(output, static_cast<std::uint32_t>(input.blocks.size()));
  for (const PromptVisionBlock& block : input.blocks) {
    Put<std::uint64_t>(output, block.begin);
    Put(output, block.grid_height);
    Put(output, block.grid_width);
  }
  output.close();
  return static_cast<bool>(output);
}

bool ReadRequest(const std::filesystem::path& path, PromptRequest* request,
                 std::vector<PromptVisionBlock>* blocks) {
  std::ifstream input(path, std::ios::binary);
  std::uint32_t magic = 0;
  std::uint32_t images = 0;
  if (!Get(input, &magic) || magic != kRequestMagic ||
      !GetPath(input, &request->model_root) ||
      !GetPath(input, &request->manifest) ||
      !GetPath(input, &request->projector) || !Get(input, &images) ||
      images > 2) {
    return false;
  }
  request->pixels.resize(images);
  for (std::uint32_t index = 0; index < images; ++index) {
    std::int32_t width = 0;
    std::int32_t height = 0;
    if (!Get(input, &width) || !Get(input, &height) || width < 1 ||
        height < 1 || !GetVector(input, &request->pixels[index]) ||
        request->pixels[index].size() != static_cast<std::size_t>(width) *
                                             static_cast<std::size_t>(height) *
                                             3U) {
      return false;
    }
    request->sizes.push_back(width);
    request->sizes.push_back(height);
  }
  std::uint32_t count = 0;
  if (!GetVector(input, &request->tokens) || !Get(input, &count) || count > 2) {
    return false;
  }
  blocks->assign(count, {});
  for (PromptVisionBlock& block : *blocks) {
    std::uint64_t begin = 0;
    if (!Get(input, &begin) || !Get(input, &block.grid_height) ||
        !Get(input, &block.grid_width)) {
      return false;
    }
    block.begin = static_cast<std::size_t>(begin);
  }
  return input.peek() == std::ifstream::traits_type::eof();
}

bool WriteResponse(const std::filesystem::path& path, double vision_ms,
                   const PromptEmbedding& embedding,
                   const PromptEncoderTelemetry& telemetry) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Put(output, kResponseMagic);
  Put(output, vision_ms);
  Put<std::uint64_t>(output, embedding.tokens);
  Put<std::uint64_t>(output, embedding.width);
  PutVector(output, std::span<const std::uint16_t>(embedding.values));
  Put<std::uint32_t>(output,
                     static_cast<std::uint32_t>(telemetry.layers.size()));
  for (const PromptEncoderLayerTelemetry& layer : telemetry.layers) {
    Put<std::uint64_t>(output, layer.layer);
    Put(output, layer.weight_bytes);
    Put(output, layer.load_ms);
    Put(output, layer.prefetch_wait_ms);
    Put(output, layer.submit_ms);
    Put(output, layer.gpu_ms);
    Put(output, layer.dispatches);
  }
  for (const std::uint64_t value :
       {telemetry.embedding_weight_bytes, telemetry.activation_bytes,
        telemetry.peak_device_bytes, telemetry.peak_registered_host_bytes,
        telemetry.live_registered_host_bytes_after,
        telemetry.total_dispatches}) {
    Put(output, value);
  }
  output.close();
  return static_cast<bool>(output);
}

bool ReadResponse(const std::filesystem::path& path, double* vision_ms,
                  PromptEmbedding* embedding,
                  PromptEncoderTelemetry* telemetry) {
  std::ifstream input(path, std::ios::binary);
  std::uint32_t magic = 0;
  std::uint64_t tokens = 0;
  std::uint64_t width = 0;
  std::uint32_t layers = 0;
  if (!Get(input, &magic) || magic != kResponseMagic ||
      !Get(input, vision_ms) || !Get(input, &tokens) || !Get(input, &width) ||
      !GetVector(input, &embedding->values) ||
      embedding->values.size() != tokens * width || !Get(input, &layers) ||
      layers > 4096) {
    return false;
  }
  embedding->tokens = static_cast<std::size_t>(tokens);
  embedding->width = static_cast<std::size_t>(width);
  telemetry->layers.assign(layers, {});
  for (PromptEncoderLayerTelemetry& layer : telemetry->layers) {
    std::uint64_t index = 0;
    if (!Get(input, &index) || !Get(input, &layer.weight_bytes) ||
        !Get(input, &layer.load_ms) || !Get(input, &layer.prefetch_wait_ms) ||
        !Get(input, &layer.submit_ms) || !Get(input, &layer.gpu_ms) ||
        !Get(input, &layer.dispatches)) {
      return false;
    }
    layer.layer = static_cast<std::size_t>(index);
  }
  return Get(input, &telemetry->embedding_weight_bytes) &&
         Get(input, &telemetry->activation_bytes) &&
         Get(input, &telemetry->peak_device_bytes) &&
         Get(input, &telemetry->peak_registered_host_bytes) &&
         Get(input, &telemetry->live_registered_host_bytes_after) &&
         Get(input, &telemetry->total_dispatches) &&
         input.peek() == std::ifstream::traits_type::eof();
}

#if defined(_WIN32)
std::string WindowsMessage(DWORD code) {
  return "Windows error " + std::to_string(code);
}

// Runs this executable as "video --h3-prompt-child <request> <response>".
// stdin/stdout are NUL, stderr is shared; nothing else is inherited (a serve
// process holds sockets). Cancellation terminates the child.
bool RunChild(const std::filesystem::path& request,
              const std::filesystem::path& response,
              const CancellationToken* cancellation, std::string* error) {
  std::wstring executable(32768, L'\0');
  const DWORD length = GetModuleFileNameW(
      nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (length == 0 || length >= executable.size()) {
    *error = "cannot locate gufo for the prompt child: " +
             WindowsMessage(GetLastError());
    return false;
  }
  executable.resize(length);
  std::wstring line;
  for (const std::wstring& argument :
       {executable, std::wstring(L"video"), std::wstring(L"--h3-prompt-child"),
        request.wstring(), response.wstring()}) {
    // Paths and fixed words never end in a backslash or contain quotes.
    line += L"\"" + argument + L"\" ";
  }
  SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
  const auto open_nul = [&attributes](DWORD access) {
    return CreateFileW(L"NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                       &attributes, OPEN_EXISTING, 0, nullptr);
  };
  const HANDLE nul_in = open_nul(GENERIC_READ);
  const HANDLE nul_out = open_nul(GENERIC_WRITE);
  HANDLE errors = nullptr;
  const HANDLE parent_errors = GetStdHandle(STD_ERROR_HANDLE);
  if (parent_errors == nullptr || parent_errors == INVALID_HANDLE_VALUE ||
      !DuplicateHandle(GetCurrentProcess(), parent_errors, GetCurrentProcess(),
                       &errors, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
    errors = open_nul(GENERIC_WRITE);
  }
  const auto close = [](HANDLE handle) {
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
      CloseHandle(handle);
    }
  };
  bool success = nul_in != INVALID_HANDLE_VALUE &&
                 nul_out != INVALID_HANDLE_VALUE && errors != nullptr &&
                 errors != INVALID_HANDLE_VALUE;
  PROCESS_INFORMATION info{};
  if (success) {
    std::array<HANDLE, 3> inherited = {nul_in, nul_out, errors};
    SIZE_T attribute_bytes = 0;
    (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
    std::vector<unsigned char> storage(attribute_bytes);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = nul_in;
    startup.StartupInfo.hStdOutput = nul_out;
    startup.StartupInfo.hStdError = errors;
    startup.lpAttributeList = list;
    const bool initialized =
        InitializeProcThreadAttributeList(list, 1, 0, &attribute_bytes) != 0;
    success =
        initialized &&
        UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                  inherited.data(), sizeof(inherited), nullptr,
                                  nullptr) &&
        CreateProcessW(executable.c_str(), line.data(), nullptr, nullptr, TRUE,
                       EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                       nullptr, &startup.StartupInfo, &info);
    if (!success) {
      *error =
          "cannot start the prompt child: " + WindowsMessage(GetLastError());
    }
    if (initialized) {
      DeleteProcThreadAttributeList(list);
    }
  } else {
    *error = "cannot prepare the prompt child handles";
  }
  close(nul_in);
  close(nul_out);
  close(errors);
  if (!success) {
    return false;
  }
  CloseHandle(info.hThread);
  while (true) {
    const DWORD wait = WaitForSingleObject(info.hProcess, 200);
    if (wait == WAIT_OBJECT_0) {
      break;
    }
    const bool cancelled =
        cancellation != nullptr && cancellation->IsCancelled();
    if (wait != WAIT_TIMEOUT || cancelled) {
      TerminateProcess(info.hProcess, 1);
      WaitForSingleObject(info.hProcess, INFINITE);
      CloseHandle(info.hProcess);
      *error = cancelled ? "MiniMax H3 prompt encoding cancelled"
                         : "cannot wait for the prompt child";
      return false;
    }
  }
  DWORD code = 1;
  GetExitCodeProcess(info.hProcess, &code);
  CloseHandle(info.hProcess);
  if (code != 0) {
    *error =
        "MiniMax H3 prompt child failed with exit code " + std::to_string(code);
    return false;
  }
  return true;
}
#endif

}  // namespace

bool PromptIsolationEnabled() {
#if defined(_WIN32) && defined(ENGINE_ENABLE_HIP)
  const char* inprocess = std::getenv("GUFO_H3_PROMPT_INPROCESS");
  return inprocess == nullptr || std::strcmp(inprocess, "1") != 0;
#else
  return false;
#endif
}

bool EncodePromptIsolated(const PromptIsolationInput& input,
                          const CancellationToken* cancellation,
                          PromptEmbedding* embedding,
                          PromptEncoderTelemetry* telemetry, double* vision_ms,
                          std::string* error) {
  std::string local_error;
  if (error == nullptr) {
    error = &local_error;
  }
#if defined(_WIN32)
  static std::atomic<unsigned> sequence{0};
  std::error_code ignored;
  const std::filesystem::path base =
      std::filesystem::temp_directory_path(ignored) /
      ("gufo-h3-prompt-" + std::to_string(GetCurrentProcessId()) + "-" +
       std::to_string(sequence.fetch_add(1)));
  const std::filesystem::path request = base.wstring() + L".request";
  const std::filesystem::path response = base.wstring() + L".response";
  bool success = WriteRequest(request, input);
  if (!success) {
    *error = "cannot write the prompt child request " + request.string();
  } else {
    success = RunChild(request, response, cancellation, error);
    if (success && !ReadResponse(response, vision_ms, embedding, telemetry)) {
      *error = "invalid prompt child response " + response.string();
      success = false;
    }
  }
  std::filesystem::remove(request, ignored);
  std::filesystem::remove(response, ignored);
  return success;
#else
  (void)input;
  (void)cancellation;
  (void)embedding;
  (void)telemetry;
  (void)vision_ms;
  *error = "MiniMax H3 prompt isolation is only available on Windows";
  return false;
#endif
}

int RunPromptChild(const std::filesystem::path& request_path,
                   const std::filesystem::path& response_path) {
  PromptRequest request;
  std::vector<PromptVisionBlock> layout;
  if (!ReadRequest(request_path, &request, &layout)) {
    std::fprintf(stderr, "h3 prompt child: invalid request\n");
    return 2;
  }
  std::string error;
  double vision_ms = 0.0;
  std::vector<PromptVisionBlock> blocks;
  if (!request.projector.empty() && !request.pixels.empty()) {
    std::vector<KeyframeImage> images;
    for (std::size_t index = 0; index < request.pixels.size(); ++index) {
      images.push_back({.width = request.sizes[index * 2U],
                        .height = request.sizes[index * 2U + 1U],
                        .rgb = request.pixels[index]});
    }
    const auto begin = std::chrono::steady_clock::now();
    if (!EncodeKeyframeVision(request.projector, images, nullptr, &blocks,
                              &error)) {
      std::fprintf(stderr, "h3 prompt child: %s\n", error.c_str());
      return 1;
    }
    vision_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin)
                    .count();
    if (blocks.size() != layout.size()) {
      std::fprintf(stderr, "h3 prompt child: vision block count differs\n");
      return 1;
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      if (blocks[index].grid_height != layout[index].grid_height ||
          blocks[index].grid_width != layout[index].grid_width) {
        std::fprintf(stderr, "h3 prompt child: vision grid differs\n");
        return 1;
      }
      blocks[index].begin = layout[index].begin;
    }
    TraceStage("prompt child vision encoded");
  } else if (!layout.empty()) {
    std::fprintf(stderr, "h3 prompt child: vision blocks without images\n");
    return 2;
  }
  auto inventory =
      ModelInventory::Inspect(request.model_root, request.manifest, &error);
  if (!inventory.has_value()) {
    std::fprintf(stderr, "h3 prompt child: %s\n", error.c_str());
    return 1;
  }
  PromptEncoderOptions options;
  options.vision_blocks = blocks;
  PromptEmbedding embedding;
  PromptEncoderTelemetry telemetry;
  if (!EncodePromptLayer50(*inventory, request.tokens, options, nullptr,
                           &embedding, &telemetry, &error)) {
    std::fprintf(stderr, "h3 prompt child: %s\n", error.c_str());
    return 1;
  }
  TraceStage("prompt child encoded");
  return WriteResponse(response_path, vision_ms, embedding, telemetry) ? 0 : 1;
}

}  // namespace gufo::minimax_h3
