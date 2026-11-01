// Real-ESRGAN body models in ONNX Runtime (see ort_upscaler.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/realesrgan/ort_upscaler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#include "onnxruntime_c_api.h"
#include "src/models/realesrgan/cpu_post.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <appmodel.h>
#include <objbase.h>
#else
#include <dlfcn.h>
#endif

namespace gufo::models::realesrgan {
namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

#if defined(_WIN32)
using Library = HMODULE;
Library OpenLibrary(const std::filesystem::path& path) {
  // Dependencies (onnxruntime_providers_shared.dll, DirectML.dll, the EP's
  // own DLLs) resolve from the library's folder, never from System32's old
  // Windows-internal onnxruntime.dll.
  return LoadLibraryExW(path.wstring().c_str(), nullptr,
                        LOAD_WITH_ALTERED_SEARCH_PATH);
}
void* Symbol(Library lib, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(lib, name));
}
std::string LastLoadError() {
  return "Windows error " + std::to_string(GetLastError());
}
#else
using Library = void*;
Library OpenLibrary(const std::filesystem::path& path) {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
void* Symbol(Library lib, const char* name) { return dlsym(lib, name); }
std::string LastLoadError() {
  const char* e = dlerror();
  return e != nullptr ? e : "dlopen failed";
}
#endif

#if defined(_WIN32)
// Windows ML (Microsoft.Windows.AI.MachineLearning.dll) flat C API for the
// execution provider catalog, declared here from its documented ABI
// (WinMLEpCatalog.h, Windows App SDK 1.8) because only these five calls are
// needed and the DLL is loaded at run time.
struct WinMLEpCatalog;
struct WinMLEp;
enum WinMLEpReadyState { kEpReady = 0, kEpNotReady = 1, kEpNotPresent = 2 };
struct WinMLEpInfo {
  const char* name;
  const char* version;
  const char* package_family_name;
  const char* library_path;
  const char* package_root_path;
  WinMLEpReadyState ready_state;
  int certification;
};
using WinMLEpEnumCallback = BOOL(CALLBACK*)(WinMLEp*, const WinMLEpInfo*,
                                             void*);
using CatalogCreateFn = HRESULT(WINAPI*)(WinMLEpCatalog**);
using CatalogReleaseFn = void(WINAPI*)(WinMLEpCatalog*);
using CatalogEnumFn = HRESULT(WINAPI*)(WinMLEpCatalog*, WinMLEpEnumCallback,
                                       void*);
using EpEnsureReadyFn = HRESULT(WINAPI*)(WinMLEp*);
using EpGetLibraryPathSizeFn = HRESULT(WINAPI*)(WinMLEp*, size_t*);
using EpGetLibraryPathFn = HRESULT(WINAPI*)(WinMLEp*, size_t, char*, size_t*);

std::string Hex(HRESULT hr) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08lX", static_cast<unsigned long>(hr));
  return text;
}

/// Install folder of the newest WindowsWorkload.WinMLShared.<M>.<m> package
/// that has onnxruntime.dll. Uses the package APIs, which work without
/// access to the WindowsApps folder listing.
std::filesystem::path FindWinMLShared(std::string* version) {
  for (int major = 20; major >= 4; --major) {
    for (int minor = 9; minor >= 0; --minor) {
      const std::wstring family = L"WindowsWorkload.WinMLShared." +
                                  std::to_wstring(major) + L"." +
                                  std::to_wstring(minor) + L"_8wekyb3d8bbwe";
      UINT32 count = 0, length = 0;
      LONG rc = FindPackagesByPackageFamily(
          family.c_str(), PACKAGE_FILTER_HEAD | PACKAGE_FILTER_DIRECT, &count,
          nullptr, &length, nullptr, nullptr);
      if (rc != ERROR_INSUFFICIENT_BUFFER || count == 0)
        continue;
      std::vector<PWSTR> names(count);
      std::vector<WCHAR> buffer(length);
      rc = FindPackagesByPackageFamily(
          family.c_str(), PACKAGE_FILTER_HEAD | PACKAGE_FILTER_DIRECT, &count,
          names.data(), &length, buffer.data(), nullptr);
      if (rc != ERROR_SUCCESS)
        continue;
      for (UINT32 i = 0; i < count; ++i) {
        UINT32 path_length = 0;
        if (GetPackagePathByFullName(names[i], &path_length, nullptr) !=
            ERROR_INSUFFICIENT_BUFFER)
          continue;
        std::wstring path(path_length, L'\0');
        if (GetPackagePathByFullName(names[i], &path_length, path.data()) !=
            ERROR_SUCCESS)
          continue;
        path.resize(wcslen(path.c_str()));
        const std::filesystem::path dir(path);
        std::error_code ec;
        if (std::filesystem::exists(dir / "onnxruntime.dll", ec)) {
          *version = std::to_string(major) + "." + std::to_string(minor);
          return dir;
        }
      }
    }
  }
  return {};
}

struct CatalogHit {
  WinMLEp* ep{nullptr};
  std::string version;
  std::string library;
  WinMLEpReadyState state{kEpNotPresent};
};

BOOL CALLBACK OnProvider(WinMLEp* ep, const WinMLEpInfo* info, void* context) {
  auto* hit = static_cast<CatalogHit*>(context);
  if (info != nullptr && info->name != nullptr &&
      std::strcmp(info->name, "VitisAIExecutionProvider") == 0) {
    hit->ep = ep;
    hit->version = info->version != nullptr ? info->version : "";
    hit->library = info->library_path != nullptr ? info->library_path : "";
    hit->state = info->ready_state;
    return FALSE;  // stop
  }
  return TRUE;
}

/// Asks Windows ML for the VitisAI EP library (it ships and updates the
/// AMD NPU provider; gufo never distributes it).
bool FindVitisAiEp(const std::filesystem::path& winml_dir,
                   std::filesystem::path* library, std::string* version,
                   std::string* error) {
  const std::filesystem::path dll =
      winml_dir / "Microsoft.Windows.AI.MachineLearning.dll";
  Library lib = OpenLibrary(dll);
  if (lib == nullptr) {
    *error = "cannot load " + dll.string() + " (" + LastLoadError() + ")";
    return false;
  }
  auto create = reinterpret_cast<CatalogCreateFn>(Symbol(lib, "WinMLEpCatalogCreate"));
  auto release = reinterpret_cast<CatalogReleaseFn>(Symbol(lib, "WinMLEpCatalogRelease"));
  auto enumerate = reinterpret_cast<CatalogEnumFn>(Symbol(lib, "WinMLEpCatalogEnumProviders"));
  auto ensure = reinterpret_cast<EpEnsureReadyFn>(Symbol(lib, "WinMLEpEnsureReady"));
  auto path_size = reinterpret_cast<EpGetLibraryPathSizeFn>(Symbol(lib, "WinMLEpGetLibraryPathSize"));
  auto get_path = reinterpret_cast<EpGetLibraryPathFn>(Symbol(lib, "WinMLEpGetLibraryPath"));
  if (!create || !release || !enumerate || !ensure || !path_size || !get_path) {
    *error = dll.string() + " has no execution provider catalog API";
    return false;
  }
  const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  WinMLEpCatalog* catalog = nullptr;
  HRESULT hr = create(&catalog);
  bool ok = false;
  if (FAILED(hr)) {
    *error = "WinMLEpCatalogCreate failed " + Hex(hr);
  } else {
    CatalogHit hit;
    hr = enumerate(catalog, &OnProvider, &hit);
    if (hit.ep == nullptr) {
      *error = "Windows ML has no VitisAIExecutionProvider (AMD NPU) on this "
               "PC" + (FAILED(hr) ? " (" + Hex(hr) + ")" : std::string());
    } else {
      if (hit.state != kEpReady) {
        // Lets Windows ML install / update the provider package.
        hr = ensure(hit.ep);
        if (FAILED(hr))
          *error = "WinMLEpEnsureReady(VitisAI) failed " + Hex(hr);
      }
      if (error->empty()) {
        size_t size = 0;
        if (SUCCEEDED(path_size(hit.ep, &size)) && size > 1) {
          std::string path(size, '\0');
          size_t used = 0;
          if (SUCCEEDED(get_path(hit.ep, size, path.data(), &used))) {
            path.resize(std::strlen(path.c_str()));
            hit.library = path;
          }
        }
        if (hit.library.empty()) {
          *error = "Windows ML reports no library for VitisAIExecutionProvider";
        } else {
          // The catalog returns UTF-8.
          *library = std::filesystem::path(
              std::u8string(hit.library.begin(), hit.library.end()));
          *version = hit.version;
          ok = true;
        }
      }
    }
    release(catalog);
  }
  if (SUCCEEDED(com))
    CoUninitialize();
  // The catalog DLL stays loaded: the EP package registration may rely on it.
  return ok;
}
#endif

}  // namespace

bool ParseOrtDevice(std::string_view text, OrtDevice* device) {
  if (text == "npu")
    *device = OrtDevice::kNpu;
  else if (text == "cpu")
    *device = OrtDevice::kCpu;
  else
    return false;
  return true;
}

std::string_view OrtDeviceName(OrtDevice device) {
  return device == OrtDevice::kNpu ? "npu" : "cpu";
}

bool ParseAreaFit(std::string_view text, AreaFit* fit) {
  if (text == "crop")
    *fit = AreaFit::kCrop;
  else if (text == "pad")
    *fit = AreaFit::kPad;
  else if (text == "stretch")
    *fit = AreaFit::kStretch;
  else
    return false;
  return true;
}

std::string_view AreaFitName(AreaFit fit) {
  return fit == AreaFit::kCrop  ? "crop"
         : fit == AreaFit::kPad ? "pad"
                                : "stretch";
}

struct OrtUpscaler::Impl {
  struct Model {
    OrtModelInfo info;
    OrtSession* session{nullptr};
    std::string input_name;
    std::string output_name;
  };

  OrtDevice device{OrtDevice::kNpu};
  Library library{nullptr};
  const OrtApi* api{nullptr};
  OrtEnv* env{nullptr};
  OrtMemoryInfo* memory{nullptr};
  std::string runtime_version;
  std::filesystem::path runtime_path;
  std::string ep_version;
  std::filesystem::path ep_path;
  bool ep_registered{false};
  std::vector<Model> models;
  std::vector<OrtModelInfo> infos;

  ~Impl() {
    if (api != nullptr) {
      for (Model& m : models)
        if (m.session != nullptr)
          api->ReleaseSession(m.session);
      if (memory != nullptr)
        api->ReleaseMemoryInfo(memory);
      if (env != nullptr) {
        if (ep_registered)
          ReleaseStatusQuiet(api->UnregisterExecutionProviderLibrary(
              env, "VitisAIExecutionProvider"));
        api->ReleaseEnv(env);
      }
    }
    // ONNX Runtime stays loaded: unloading it while EP threads exit is unsafe.
  }

  void ReleaseStatusQuiet(OrtStatus* status) const {
    if (status != nullptr)
      api->ReleaseStatus(status);
  }

  bool Ok(OrtStatus* status, std::string_view what, std::string* error) const {
    if (status == nullptr)
      return true;
    *error = std::string(what) + ": " + api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    return false;
  }

  bool LoadRuntime(const OrtUpscalerOptions& options, std::string* error) {
    std::filesystem::path dll = options.runtime_library;
#if defined(_WIN32)
    std::filesystem::path winml_dir;
    if (dll.empty()) {
      std::string winml_version;
      winml_dir = FindWinMLShared(&winml_version);
      if (!winml_dir.empty()) {
        dll = winml_dir / "onnxruntime.dll";
      }
    } else {
      winml_dir = dll.parent_path();
    }
#endif
    if (dll.empty())
      if (const char* env_dll = std::getenv("GUFO_ONNXRUNTIME_DLL"))
        dll = env_dll;
    if (dll.empty()) {
      *error =
          "the NPU / ONNX path needs ONNX Runtime 1.23+ (Windows ML on "
          "Windows 11 24H2+; or --onnxruntime <onnxruntime.dll>)";
      return false;
    }
    library = OpenLibrary(dll);
    if (library == nullptr) {
      *error = "cannot load " + dll.string() + " (" + LastLoadError() + ")";
      return false;
    }
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    auto get_base =
        reinterpret_cast<GetApiBaseFn>(Symbol(library, "OrtGetApiBase"));
    const OrtApiBase* base = get_base != nullptr ? get_base() : nullptr;
    api = base != nullptr ? base->GetApi(ORT_API_VERSION) : nullptr;
    if (api == nullptr) {
      *error = dll.string() + " is not ONNX Runtime 1." +
               std::to_string(ORT_API_VERSION) + " or newer" +
               (base != nullptr
                    ? std::string(" (it is ") + base->GetVersionString() + ")"
                    : std::string());
      return false;
    }
    runtime_version = base->GetVersionString();
    runtime_path = dll;
    if (!Ok(api->CreateEnv(static_cast<OrtLoggingLevel>(options.log_level),
                           "gufo-upscale", &env),
            "CreateEnv", error) ||
        !Ok(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                     &memory),
            "CreateCpuMemoryInfo", error))
      return false;
    if (device != OrtDevice::kNpu)
      return true;
    ep_path = options.ep_library;
#if defined(_WIN32)
    if (ep_path.empty() &&
        !FindVitisAiEp(winml_dir, &ep_path, &ep_version, error))
      return false;
#endif
    if (ep_path.empty()) {
      *error = "--npu-ep <onnxruntime_vitisai_ep.dll> is required here";
      return false;
    }
    if (!Ok(api->RegisterExecutionProviderLibrary(
                env, "VitisAIExecutionProvider", ep_path.c_str()),
            "RegisterExecutionProviderLibrary(" + ep_path.string() + ")",
            error))
      return false;
    ep_registered = true;
    return true;
  }

  bool NpuDevice(const OrtEpDevice** found, std::string* error) const {
    const OrtEpDevice* const* devices = nullptr;
    size_t count = 0;
    if (!Ok(api->GetEpDevices(env, &devices, &count), "GetEpDevices", error))
      return false;
    for (size_t i = 0; i < count; ++i) {
      const char* name = api->EpDevice_EpName(devices[i]);
      const OrtHardwareDevice* hw = api->EpDevice_Device(devices[i]);
      if (name != nullptr &&
          std::strcmp(name, "VitisAIExecutionProvider") == 0 && hw != nullptr &&
          api->HardwareDevice_Type(hw) == OrtHardwareDeviceType_NPU) {
        *found = devices[i];
        return true;
      }
    }
    *error = "VitisAIExecutionProvider reports no NPU device (driver?)";
    return false;
  }

  bool OpenModel(const OrtUpscalerOptions& options, const OrtModelSpec& spec,
                 std::string* error) {
    const Clock::time_point start = Clock::now();
    Model m;
    m.info.name = spec.name;
    m.info.path = spec.path;
    m.info.cache_key =
        spec.cache_key.empty() ? spec.path.stem().string() : spec.cache_key;
    OrtSessionOptions* so = nullptr;
    if (!Ok(api->CreateSessionOptions(&so), "CreateSessionOptions", error))
      return false;
    struct Release {
      const OrtApi* api;
      OrtSessionOptions* so;
      ~Release() { api->ReleaseSessionOptions(so); }
    } release{api, so};
    if (options.threads > 0 &&
        !Ok(api->SetIntraOpNumThreads(so, options.threads),
            "SetIntraOpNumThreads", error))
      return false;
    if (!Ok(api->SetSessionLogSeverityLevel(so, options.log_level),
            "SetSessionLogSeverityLevel", error))
      return false;
    if (device == OrtDevice::kNpu) {
      const OrtEpDevice* npu = nullptr;
      if (!NpuDevice(&npu, error))
        return false;
      std::filesystem::path cache = options.cache_dir;
      if (cache.empty())
        cache = spec.path.parent_path() / "npu-cache";
      std::error_code ec;
      std::filesystem::create_directories(cache, ec);
      // Take One's npu_session.py options.
      const std::string cache_dir = cache.string();
      const std::string config = options.ep_config_file.string();
      std::vector<const char*> keys{"cache_dir", "cache_key",
                                    "enable_cache_file_io_in_mem"};
      std::vector<const char*> values{cache_dir.c_str(),
                                      m.info.cache_key.c_str(), "0"};
      if (!config.empty()) {
        keys.push_back("config_file");
        values.push_back(config.c_str());
      }
      if (!Ok(api->SessionOptionsAppendExecutionProvider_V2(
                  so, env, &npu, 1, keys.data(), values.data(), keys.size()),
              "SessionOptionsAppendExecutionProvider_V2(VitisAI)", error))
        return false;
    }
    if (!Ok(api->CreateSession(env, spec.path.c_str(), so, &m.session),
            "CreateSession(" + spec.path.string() + ")", error))
      return false;
    // One input [1, 3, rows, width], one output [1, 48, rows, width].
    size_t inputs = 0, outputs = 0;
    if (!Ok(api->SessionGetInputCount(m.session, &inputs), "inputs", error) ||
        !Ok(api->SessionGetOutputCount(m.session, &outputs), "outputs", error)) {
      api->ReleaseSession(m.session);
      return false;
    }
    OrtAllocator* allocator = nullptr;
    char* name = nullptr;
    std::vector<int64_t> in_shape, out_shape;
    const auto shape_of = [&](bool input, std::vector<int64_t>* shape) {
      OrtTypeInfo* type = nullptr;
      OrtStatus* st = input ? api->SessionGetInputTypeInfo(m.session, 0, &type)
                            : api->SessionGetOutputTypeInfo(m.session, 0, &type);
      if (!Ok(st, "type info", error))
        return false;
      const OrtTensorTypeAndShapeInfo* tensor = nullptr;
      size_t dims = 0;
      bool ok = Ok(api->CastTypeInfoToTensorInfo(type, &tensor), "tensor", error) &&
                tensor != nullptr &&
                Ok(api->GetDimensionsCount(tensor, &dims), "dims", error);
      if (ok) {
        shape->resize(dims);
        ok = Ok(api->GetDimensions(tensor, shape->data(), dims), "dims", error);
      }
      api->ReleaseTypeInfo(type);
      return ok;
    };
    bool ok = inputs == 1 && outputs == 1 &&
              Ok(api->GetAllocatorWithDefaultOptions(&allocator), "allocator",
                 error) &&
              shape_of(true, &in_shape) && shape_of(false, &out_shape);
    if (ok && Ok(api->SessionGetInputName(m.session, 0, allocator, &name),
                 "input name", error)) {
      m.input_name = name;
      ReleaseStatusQuiet(api->AllocatorFree(allocator, name));
    } else {
      ok = false;
    }
    if (ok && Ok(api->SessionGetOutputName(m.session, 0, allocator, &name),
                 "output name", error)) {
      m.output_name = name;
      ReleaseStatusQuiet(api->AllocatorFree(allocator, name));
    } else {
      ok = false;
    }
    if (ok && (in_shape.size() != 4 || in_shape[1] != 3 || in_shape[2] <= 0 ||
               in_shape[3] <= 0 || out_shape.size() != 4 ||
               out_shape[1] != cpu::kBodyChannels ||
               out_shape[2] != in_shape[2] || out_shape[3] != in_shape[3])) {
      *error = spec.path.string() +
               ": expected a body model [1,3,H,W] -> [1,48,H,W] with a fixed "
               "size (Real-ESRGAN before the pixel shuffle)";
      ok = false;
    }
    if (!ok) {
      if (error->empty())
        *error = spec.path.string() + ": needs one input and one output";
      api->ReleaseSession(m.session);
      return false;
    }
    m.info.tile_rows = static_cast<int>(in_shape[2]);
    m.info.width = static_cast<int>(in_shape[3]);
    m.info.load_seconds = Seconds(start);
    infos.push_back(m.info);
    models.push_back(std::move(m));
    return true;
  }

  const Model* Pick(std::string_view name, int w, int h) const {
    for (const Model& m : models)
      if ((name.empty() || m.info.name == name) && m.info.width == w &&
          !cpu::PlanRowTiles(h, m.info.tile_rows).empty())
        return &m;
    return nullptr;
  }

  bool RunPass(const Model& m, float* input, float* output,
               std::string* error) const {
    const int64_t in_shape[4] = {1, 3, m.info.tile_rows, m.info.width};
    const int64_t out_shape[4] = {1, cpu::kBodyChannels, m.info.tile_rows,
                                  m.info.width};
    const size_t pixels =
        static_cast<size_t>(m.info.tile_rows) * static_cast<size_t>(m.info.width);
    OrtValue* in = nullptr;
    OrtValue* out = nullptr;
    if (!Ok(api->CreateTensorWithDataAsOrtValue(
                memory, input, pixels * 3 * sizeof(float), in_shape, 4,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
            "input tensor", error))
      return false;
    if (!Ok(api->CreateTensorWithDataAsOrtValue(
                memory, output, pixels * cpu::kBodyChannels * sizeof(float),
                out_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out),
            "output tensor", error)) {
      api->ReleaseValue(in);
      return false;
    }
    const char* in_name = m.input_name.c_str();
    const char* out_name = m.output_name.c_str();
    const bool ok = Ok(api->Run(m.session, nullptr, &in_name, &in, 1,
                                &out_name, 1, &out),
                       "Run", error);
    api->ReleaseValue(in);
    api->ReleaseValue(out);
    return ok;
  }
};

OrtUpscaler::OrtUpscaler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
OrtUpscaler::~OrtUpscaler() = default;

std::unique_ptr<OrtUpscaler> OrtUpscaler::Load(
    const OrtUpscalerOptions& options, const std::vector<OrtModelSpec>& models,
    std::string* error) {
  if (models.empty()) {
    *error = "no model";
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->device = options.device;
  if (!impl->LoadRuntime(options, error))
    return nullptr;
  for (const OrtModelSpec& spec : models)
    if (!impl->OpenModel(options, spec, error))
      return nullptr;
  return std::unique_ptr<OrtUpscaler>(new OrtUpscaler(std::move(impl)));
}

const OrtModelInfo* OrtUpscaler::Find(std::string_view name, int w,
                                      int h) const {
  const Impl::Model* m = impl_->Pick(name, w, h);
  return m != nullptr ? &m->info : nullptr;
}

bool OrtUpscaler::UpscaleX4(std::string_view name, const std::uint8_t* rgb24,
                            int w, int h, std::vector<std::uint8_t>* up24,
                            UpscaleTimings* timings, std::string* error,
                            std::vector<float>* body_out) {
  const Impl::Model* m = impl_->Pick(name, w, h);
  if (m == nullptr) {
    std::string sizes;
    for (const OrtModelInfo& info : impl_->infos)
      if (name.empty() || info.name == name)
        sizes += (sizes.empty() ? "" : ", ") + std::to_string(info.width) +
                 " wide (" + std::to_string(info.tile_rows) + "-row passes)";
    *error = "no " + std::string(name.empty() ? "" : std::string(name) + " ") +
             "model for a " + std::to_string(w) + "x" + std::to_string(h) +
             " frame" + (sizes.empty() ? "" : "; loaded: " + sizes);
    return false;
  }
  const std::vector<cpu::TilePass> plan =
      cpu::PlanRowTiles(h, m->info.tile_rows);
  const int rows = m->info.tile_rows;
  std::vector<float> input(static_cast<size_t>(3) * rows * w);
  std::vector<float> output(static_cast<size_t>(cpu::kBodyChannels) * rows * w);
  std::vector<float> local;
  std::vector<float>& body = body_out != nullptr ? *body_out : local;
  body.resize(static_cast<size_t>(cpu::kBodyChannels) * h * w);
  const Clock::time_point start = Clock::now();
  for (const cpu::TilePass& pass : plan) {
    cpu::PackTileInput(rgb24, w, pass.src_row, rows, input.data());
    if (!impl_->RunPass(*m, input.data(), output.data(), error))
      return false;
    cpu::StoreTileOutput(output.data(), w, rows, pass, h, body.data());
  }
  const Clock::time_point post = Clock::now();
  up24->resize(static_cast<size_t>(w) * h * cpu::kScale * cpu::kScale * 3);
  cpu::ShuffleResidual(body.data(), rgb24, w, h, up24->data());
  if (timings != nullptr) {
    timings->body_seconds +=
        std::chrono::duration<double>(post - start).count();
    timings->post_seconds += Seconds(post);
    timings->passes += static_cast<int>(plan.size());
  }
  return true;
}

bool OrtUpscaler::Upscale(std::string_view name, const std::uint8_t* rgb24,
                          int w, int h, int out_w, int out_h, AreaFit fit,
                          std::vector<std::uint8_t>* out24,
                          UpscaleTimings* timings, std::string* error) {
  std::vector<std::uint8_t> up;
  if (!UpscaleX4(name, rgb24, w, h, &up, timings, error))
    return false;
  const int up_w = w * cpu::kScale, up_h = h * cpu::kScale;
  if ((out_w == 0 && out_h == 0) || (out_w == up_w && out_h == up_h)) {
    *out24 = std::move(up);
    return true;
  }
  const Clock::time_point start = Clock::now();
  bool ok = false;
  switch (fit) {
  case AreaFit::kCrop:
    ok = cpu::FitAreaCrop(up.data(), up_w, up_h, out_w, out_h, out24, error);
    break;
  case AreaFit::kPad:
    ok = cpu::FitAreaPad(up.data(), up_w, up_h, out_w, out_h, out24, error);
    break;
  case AreaFit::kStretch:
    out24->resize(static_cast<size_t>(out_w) * out_h * 3);
    ok = cpu::ResizeAreaRgb8(up.data(), up_w, up_h, up_w * 3, out_w, out_h,
                             out24->data(), error);
    break;
  }
  if (timings != nullptr)
    timings->post_seconds += Seconds(start);
  return ok;
}

OrtDevice OrtUpscaler::device() const { return impl_->device; }
const std::vector<OrtModelInfo>& OrtUpscaler::models() const {
  return impl_->infos;
}
const std::string& OrtUpscaler::runtime_version() const {
  return impl_->runtime_version;
}
const std::filesystem::path& OrtUpscaler::runtime_library() const {
  return impl_->runtime_path;
}
const std::string& OrtUpscaler::ep_version() const { return impl_->ep_version; }
const std::filesystem::path& OrtUpscaler::ep_library() const {
  return impl_->ep_path;
}

}  // namespace gufo::models::realesrgan
