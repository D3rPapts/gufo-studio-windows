#include "src/cli/serve/router/einrichtung.hpp"

#include "src/core/platform/net.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/jobs.hpp"
#include "src/cli/serve/router/strings.hpp"

namespace gufo::router::einrichtung {

using server::HttpRequest;
using server::HttpResponse;
using server::Logger;

namespace {

// ---------------------------------------------------------------------------
// Small helpers

bool ReadText(const std::filesystem::path& file, std::string* out) {
  std::ifstream in(file, std::ios::binary);
  if (!in)
    return false;
  out->assign(std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>());
  if (out->starts_with("\xEF\xBB\xBF"))
    out->erase(0, 3);
  return true;
}

/// Python's str() of a JSON value: numbers without a trailing ".0",
/// strings as they are, true/false/null as "" (null) or the word.
std::string Text(const json::Value* value) {
  if (value == nullptr || value->is_null())
    return "";
  if (value->is_string())
    return value->str();
  if (value->is_bool())
    return value->as_bool() ? "True" : "False";
  if (value->is_number()) {
    const double d = value->as_double();
    if (std::isfinite(d) && std::floor(d) == d && std::fabs(d) < 1e15)
      return std::to_string(static_cast<long long>(d));
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.17g", d);
    return buffer;
  }
  return value->dump();
}

std::string Member(const json::Value& object, const std::string& key) {
  return Text(object.find(key));
}

json::Value ErrorBody(std::string_view code, std::string_view message) {
  json::Value detail = json::Value::object();
  detail["code"] = std::string(code);
  detail["message"] = std::string(message);
  detail["type"] = "invalid_request_error";
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return body;
}

HttpResponse Json(int status, std::string_view reason,
                  const json::Value& body) {
  return HttpResponse{.status = status,
                      .reason = std::string(reason),
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"},
                                  {"Cache-Control", "no-store"}},
                      .streaming_body = {}};
}

HttpResponse JsonOk(const json::Value& body) { return Json(200, "OK", body); }

HttpResponse Error(int status, std::string_view reason, std::string_view code,
                   std::string_view message) {
  return Json(status, reason, ErrorBody(code, message));
}

HttpResponse NotFound() {
  return Error(404, "Not Found", "not_found", "no such route");
}

}  // namespace

// ---------------------------------------------------------------------------
// Public helpers

std::filesystem::path FindDir(const std::filesystem::path& exe,
                              const std::filesystem::path& override) {
  std::error_code ec;
  if (!override.empty())
    return override;
  const std::filesystem::path bin = exe.parent_path();
  const std::filesystem::path candidates[] = {
      bin / ".." / "examples" / "einrichtung", bin / "einrichtung",
      bin / ".." / "einrichtung", bin / ".." / ".." / "einrichtung"};
  for (const std::filesystem::path& dir : candidates) {
    if (std::filesystem::is_regular_file(dir / "katalog.json", ec))
      return std::filesystem::weakly_canonical(dir, ec);
  }
  return {};
}

bool IsSafePath(std::string_view path) {
  if (path.empty())
    return false;
#ifdef _WIN32
  if (path.size() < 3 || !std::isalpha(static_cast<unsigned char>(path[0])) ||
      path[1] != ':' || (path[2] != '\\' && path[2] != '/'))
    return false;
#else
  if (path[0] != '/')
    return false;
#endif
  std::size_t start = 0;
  while (start <= path.size()) {
    const auto end = std::min(path.find_first_of("\\/", start), path.size());
    if (path.substr(start, end - start) == "..")
      return false;
    start = end + 1;
  }
  return true;
}

namespace {

std::filesystem::path NativePath(std::string_view text) {
#ifdef _WIN32
  return std::filesystem::path(std::string(text));
#else
  std::string posix(text);
  std::replace(posix.begin(), posix.end(), '\\', '/');
  return std::filesystem::path(posix);
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// Machine check

namespace {

struct Check {
  std::string id;
  std::string status;  // ok | warn | err
  std::string text;
  std::string tun;
};

json::Value CheckJson(const Check& c) {
  json::Value out = json::Value::object();
  out["id"] = c.id;
  out["status"] = c.status;
  out["text"] = c.text;
  out["tun"] = c.tun;
  return out;
}

/// "32.0.32015.7029" < "32.0.32016.1" numerically per dotted part.
int CompareVersions(std::string_view a, std::string_view b) {
  const auto parts = [](std::string_view v) {
    std::vector<long long> out;
    for (const std::string& p : Split(v, '.')) {
      long long n = 0;
      for (const char c : p) {
        if (!std::isdigit(static_cast<unsigned char>(c)))
          break;
        n = n * 10 + (c - '0');
      }
      out.push_back(n);
    }
    return out;
  };
  const auto pa = parts(a), pb = parts(b);
  for (std::size_t i = 0; i < std::max(pa.size(), pb.size()); ++i) {
    const long long x = i < pa.size() ? pa[i] : 0;
    const long long y = i < pb.size() ? pb[i] : 0;
    if (x != y)
      return x < y ? -1 : 1;
  }
  return 0;
}

#ifdef _WIN32
std::string Narrow(const std::wstring& text) {
  if (text.empty())
    return {};
  const int size = WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                       static_cast<int>(text.size()), nullptr,
                                       0, nullptr, nullptr);
  std::string out(static_cast<std::size_t>(std::max(size, 0)), '\0');
  if (size > 0)
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        out.data(), size, nullptr, nullptr);
  return out;
}

std::optional<std::string> RegString(HKEY root, const wchar_t* key,
                                     const wchar_t* name) {
  HKEY handle = nullptr;
  if (RegOpenKeyExW(root, key, 0, KEY_READ | KEY_WOW64_64KEY, &handle) !=
      ERROR_SUCCESS)
    return std::nullopt;
  wchar_t buffer[1024];
  DWORD size = sizeof buffer;
  DWORD type = 0;
  const LSTATUS status = RegQueryValueExW(
      handle, name, nullptr, &type, reinterpret_cast<LPBYTE>(buffer), &size);
  RegCloseKey(handle);
  if (status != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ))
    return std::nullopt;
  std::wstring text(buffer, size / sizeof(wchar_t));
  while (!text.empty() && text.back() == L'\0')
    text.pop_back();
  return Narrow(text);
}

std::optional<DWORD> RegDword(HKEY root, const wchar_t* key,
                              const wchar_t* name) {
  HKEY handle = nullptr;
  if (RegOpenKeyExW(root, key, 0, KEY_READ | KEY_WOW64_64KEY, &handle) !=
      ERROR_SUCCESS)
    return std::nullopt;
  DWORD value = 0;
  DWORD size = sizeof value;
  DWORD type = 0;
  const LSTATUS status = RegQueryValueExW(
      handle, name, nullptr, &type, reinterpret_cast<LPBYTE>(&value), &size);
  RegCloseKey(handle);
  if (status != ERROR_SUCCESS || type != REG_DWORD)
    return std::nullopt;
  return value;
}

struct GpuInfo {
  std::string name;
  std::string arch;
  std::string driver;
  std::string device_id;
};

/// The display class in the registry: the first AMD adapter (VEN_1002).
/// DEV_1586 is the Strix Halo GPU (gfx1151; the .198 and .56 report
/// PCI\VEN_1002&DEV_1586).
GpuInfo ReadGpu() {
  GpuInfo info;
  const std::wstring base =
      L"SYSTEM\\CurrentControlSet\\Control\\Class\\"
      L"{4d36e968-e325-11ce-bfc1-08002be10318}\\";
  for (int i = 0; i < 16; ++i) {
    wchar_t index[8];
    std::swprintf(index, 8, L"%04d", i);
    const std::wstring key = base + index;
    const auto device = RegString(HKEY_LOCAL_MACHINE, key.c_str(),
                                  L"MatchingDeviceId");
    if (!device.has_value())
      continue;
    const std::string lower = Lower(*device);
    if (lower.find("ven_1002") == std::string::npos)
      continue;
    info.device_id = *device;
    info.name = RegString(HKEY_LOCAL_MACHINE, key.c_str(), L"DriverDesc")
                    .value_or("");
    info.driver = RegString(HKEY_LOCAL_MACHINE, key.c_str(), L"DriverVersion")
                      .value_or("");
    if (lower.find("dev_1586") != std::string::npos)
      info.arch = "gfx1151";
    break;
  }
  return info;
}

std::string WindowsVersion(long long* build) {
  *build = 0;
  using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
  if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
    if (auto fn = reinterpret_cast<RtlGetVersionFn>(
            reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")))) {
      RTL_OSVERSIONINFOW info{};
      info.dwOSVersionInfoSize = sizeof info;
      if (fn(&info) == 0)
        *build = info.dwBuildNumber;
    }
  }
  const std::string edition =
      RegString(HKEY_LOCAL_MACHINE,
                L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"EditionID")
          .value_or("");
  return std::string(*build >= 22000 ? "Windows 11" : "Windows 10") +
         (edition.empty() ? "" : " " + edition) + " " + std::to_string(*build);
}

std::string FindFfmpeg() {
  wchar_t buffer[MAX_PATH * 2];
  wchar_t* part = nullptr;
  const DWORD n = SearchPathW(nullptr, L"ffmpeg", L".exe", MAX_PATH * 2, buffer,
                              &part);
  if (n == 0 || n >= MAX_PATH * 2)
    return "";
  return Narrow(std::wstring(buffer, n));
}

std::string ComputerName() {
  wchar_t buffer[MAX_COMPUTERNAME_LENGTH + 1];
  DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
  if (!GetComputerNameW(buffer, &size))
    return "";
  return Lower(Narrow(std::wstring(buffer, size)));
}

json::Value Drives() {
  json::Value out = json::Value::array();
  const DWORD mask = GetLogicalDrives();
  for (int i = 0; i < 26; ++i) {
    if (!(mask & (1u << i)))
      continue;
    const wchar_t root[] = {static_cast<wchar_t>(L'A' + i), L':', L'\\', 0};
    if (GetDriveTypeW(root) != DRIVE_FIXED)
      continue;
    std::error_code ec;
    const auto space = std::filesystem::space(std::filesystem::path(root), ec);
    if (ec)
      continue;
    json::Value drive = json::Value::object();
    drive["laufwerk"] = std::string(1, static_cast<char>('A' + i)) + ":";
    drive["frei_bytes"] = static_cast<long long>(space.available);
    drive["gesamt_bytes"] = static_cast<long long>(space.capacity);
    out.push_back(std::move(drive));
  }
  return out;
}

/// True when a listener could bind the port right now.
bool PortFree(int port) {
  try {
    net::EnsureSocketRuntimeStarted();
  } catch (const std::exception&) {
    return true;
  }
  const net::Socket sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (sock == net::kInvalidSocket)
    return true;
  const int exclusive = 1;
  ::setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
               reinterpret_cast<const char*>(&exclusive), sizeof exclusive);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  const bool ok =
      ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0;
  net::CloseSocket(sock);
  return ok;
}
#else
std::string ComputerName() {
  char buffer[256]{};
  if (gethostname(buffer, sizeof buffer - 1) != 0)
    return "";
  return Lower(buffer);
}

json::Value Drives() {
  json::Value out = json::Value::array();
  std::error_code ec;
  const auto space = std::filesystem::space("/", ec);
  if (!ec) {
    json::Value drive = json::Value::object();
    drive["laufwerk"] = "/";
    drive["frei_bytes"] = static_cast<long long>(space.available);
    drive["gesamt_bytes"] = static_cast<long long>(space.capacity);
    out.push_back(std::move(drive));
  }
  return out;
}

bool PortFree(int port) {
  net::EnsureSocketRuntimeStarted();
  const net::Socket sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (sock == net::kInvalidSocket)
    return true;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  const bool ok =
      ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0;
  net::CloseSocket(sock);
  return ok;
}
#endif

std::string LocalAddress(int port) {
  // A connected UDP socket toward a public address picks the route;
  // nothing is sent (jobs::LocalUrlToward).
  std::string url = jobs::LocalUrlToward("http://1.1.1.1:80", port);
  if (url.starts_with("http://"))
    url.erase(0, 7);
  if (const auto colon = url.rfind(':'); colon != std::string::npos)
    url.resize(colon);
  return url;
}

}  // namespace

json::Value MachineCheck(const CheckOptions& options) {
  json::Value out = json::Value::object();
  std::vector<Check> checks;
  const SystemMemory& memory = options.memory;
  const double gib = static_cast<double>(kGiB);
  const double host_gb = static_cast<double>(memory.host_total) / gib;
  const double gpu_gb = static_cast<double>(memory.gpu_dedicated_total) / gib;
  const double total_gb = host_gb + (memory.gpu_valid ? gpu_gb : 0.0);
  const int ram_gb = total_gb >= 96.0 ? 128 : 64;
  out["ram_gb"] = ram_gb;
  out["ram_gesamt_gb"] = std::round(total_gb * 10.0) / 10.0;
  const std::filesystem::path app = options.exe.parent_path();
  out["app"] = app.string();
  out["rechner"] = ComputerName();
  out["adresse"] = LocalAddress(options.own_port > 0 ? options.own_port : 8081);
  // Worker builds the package ships beside the app folder: <app>\..\<name>.
  json::Value worker = json::Value::object();
  json::Value worker_da = json::Value::object();
  std::error_code ec;
  for (const auto& [key, folder] :
       std::vector<std::pair<std::string, std::string>>{
           {"app_3d", "3d"}, {"app_moss", "moss"}, {"app_trellis2", "trellis2"},
           {"app_35b", "35b"}}) {
    const std::filesystem::path dir =
        std::filesystem::weakly_canonical(app.parent_path() / folder, ec);
    worker[key] = (ec ? app.parent_path() / folder : dir).string();
    worker_da[key] = std::filesystem::is_regular_file(dir / "gufo.exe", ec);
  }
  out["worker"] = std::move(worker);
  out["worker_vorhanden"] = std::move(worker_da);

  // GPU and driver.
  json::Value gpu = json::Value::object();
#ifdef _WIN32
  const GpuInfo info = ReadGpu();
  gpu["name"] = info.name;
  gpu["arch"] = info.arch;
  gpu["treiber"] = info.driver;
  gpu["geraet"] = info.device_id;
  if (info.arch == "gfx1151") {
    checks.push_back({"gpu", "ok", info.name + " (gfx1151)", ""});
  } else if (!info.name.empty()) {
    checks.push_back({"gpu", "err", info.name + " (" + info.device_id + ")",
                      "gufo braucht einen AMD Strix Halo (Radeon 8060S/8050S, "
                      "gfx1151)."});
  } else {
    checks.push_back({"gpu", "err", "kein AMD-Grafiktreiber gefunden",
                      "AMD Adrenalin-Treiber installieren."});
  }
  if (info.driver.empty()) {
    checks.push_back({"treiber", "err", "keine Treiberversion lesbar",
                      "AMD Adrenalin-Treiber installieren."});
  } else if (CompareVersions(info.driver, kMinDriver) < 0) {
    checks.push_back({"treiber", "warn",
                      info.driver + " (geprueft mit " + kMinDriver + ")",
                      "Adrenalin-Treiber aktualisieren (Stand der .198: " +
                          std::string(kMinDriver) + ")."});
  } else {
    checks.push_back({"treiber", "ok",
                      info.driver + " (Mindestversion " + kMinDriver + ")", ""});
  }
#else
  gpu["name"] = "";
  gpu["arch"] = "";
  gpu["treiber"] = "";
  checks.push_back({"gpu", "warn", "nicht geprueft (kein Windows)", ""});
  checks.push_back({"treiber", "warn", "nicht geprueft (kein Windows)", ""});
#endif
  out["gpu"] = std::move(gpu);

  // RAM and the BIOS split.
  {
    char text[160];
    std::snprintf(text, sizeof text, "%d GB gesamt (Windows %.1f GB + Grafik %.1f GB)",
                  ram_gb, host_gb, memory.gpu_valid ? gpu_gb : 0.0);
    const bool plausible =
        (total_gb >= 56.0 && total_gb <= 72.0) ||
        (total_gb >= 112.0 && total_gb <= 144.0);
    checks.push_back({"ram", plausible ? "ok" : "warn", text,
                      plausible ? ""
                                : "gufo ist fuer 64 oder 128 GB gemessen; die "
                                  "Vorgaben passen eventuell nicht."});
    if (!memory.gpu_valid) {
      checks.push_back({"zuteilung", "warn", "Grafikspeicher nicht lesbar",
                        "Treiber pruefen (DXGI-Adapter nicht gefunden)."});
    } else {
      const double need = ram_gb == 128 ? 90.0 : 40.0;
      std::snprintf(text, sizeof text, "Grafik: %.0f GB zugeteilt", gpu_gb);
      checks.push_back({"zuteilung", gpu_gb < need ? "warn" : "ok", text,
                        gpu_gb < need
                            ? "Im BIOS die Zuteilung an die Grafik (UMA Frame "
                              "Buffer) auf " +
                                  std::to_string(static_cast<int>(need)) +
                                  " GB oder mehr stellen."
                            : ""});
    }
  }

  // Drives.
  json::Value drives = Drives();
  {
    double best = 0.0;
    std::string best_name;
    for (const json::Value& d : drives.items()) {
      if (d.member_double("frei_bytes", 0.0) > best) {
        best = d.member_double("frei_bytes", 0.0);
        best_name = d.member_str("laufwerk");
      }
    }
    char text[160];
    std::snprintf(text, sizeof text, "%s %.0f GB frei", best_name.c_str(),
                  best / 1e9);
    checks.push_back({"platte", best < 60e9 ? "err" : "ok",
                      best_name.empty() ? "kein Laufwerk lesbar" : text,
                      best < 60e9 ? "Mindestens 60 GB frei fuer Chat + Bilder; "
                                    "Platz schaffen oder eine zweite Platte."
                                  : ""});
  }
  out["laufwerke"] = std::move(drives);

  // Windows build.
#ifdef _WIN32
  {
    long long build = 0;
    const std::string version = WindowsVersion(&build);
    checks.push_back({"windows", build >= 22000 ? "ok" : "warn", version,
                      build >= 22000 ? "" : "Windows 11 empfohlen (gemessen "
                                            "auf Build 26200)."});
  }
  {
    const std::string ffmpeg = FindFfmpeg();
    checks.push_back({"ffmpeg", ffmpeg.empty() ? "warn" : "ok",
                      ffmpeg.empty() ? "nicht gefunden" : ffmpeg,
                      ffmpeg.empty()
                          ? "winget install Gyan.FFmpeg (fuer Video mit Ton "
                            "und MP3)"
                          : ""});
  }
  {
    const auto tdr = RegDword(HKEY_LOCAL_MACHINE,
                              L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers",
                              L"TdrDelay");
    if (!tdr.has_value()) {
      checks.push_back({"tdr", "warn", "TdrDelay nicht gesetzt (Standard 2 s)",
                        "Empfohlen: TdrDelay = 30 oder mehr (Registry, "
                        "tools\\windows\\tdr_delay.ps1), sonst bricht der "
                        "Treiber lange Bildrechnungen ab."});
    } else if (*tdr < 30) {
      checks.push_back({"tdr", "warn", "TdrDelay = " + std::to_string(*tdr) + " s",
                        "Empfohlen: TdrDelay = 30 oder mehr (Registry)."});
    } else {
      checks.push_back({"tdr", "ok", "TdrDelay = " + std::to_string(*tdr) + " s",
                        ""});
    }
  }
#else
  checks.push_back({"windows", "warn", "kein Windows", ""});
  checks.push_back({"ffmpeg", "warn", "nicht geprueft", ""});
  checks.push_back({"tdr", "warn", "nicht geprueft", ""});
#endif

  // Port.
  {
    const int port = options.wunsch_port > 0 ? options.wunsch_port : 8081;
    if (port == options.own_port) {
      checks.push_back({"port", "ok",
                        std::to_string(port) + " (dieser Einrichtungs-Router)",
                        ""});
    } else if (PortFree(port)) {
      checks.push_back({"port", "ok", std::to_string(port) + " frei", ""});
    } else {
      checks.push_back({"port", "err", std::to_string(port) + " belegt",
                        "Laeuft schon ein Router? Sonst einen anderen Port "
                        "waehlen (Schritt 5)."});
    }
  }

  // Commit limit (page file).
  if (memory.commit_valid) {
    const double limit_gb = static_cast<double>(memory.commit_limit) / gib;
    char text[96];
    std::snprintf(text, sizeof text, "Commit-Limit %.1f GB", limit_gb);
    const bool small = limit_gb < host_gb + 16.0;
    checks.push_back({"auslagerung", small ? "warn" : "ok", text,
                      small ? "Auslagerungsdatei auf systemverwaltet stellen "
                              "(grosse Modelle brauchen Commit-Reserve)."
                            : ""});
  } else {
    checks.push_back({"auslagerung", "warn", "nicht lesbar", ""});
  }

  json::Value list = json::Value::array();
  for (const Check& c : checks)
    list.push_back(CheckJson(c));
  out["pruefungen"] = std::move(list);
  out["einrichtung_version"] = kVersion;
  return out;
}

// ---------------------------------------------------------------------------
// The service

SetupService::SetupService(ServiceOptions options)
    : options_(std::move(options)) {}

bool SetupService::LoadKatalog(json::Value* out, std::string* error) const {
  if (options_.dir.empty()) {
    *error = "einrichtung\\katalog.json not found next to the package "
             "(examples\\einrichtung); --einrichtung-dir names the folder";
    return false;
  }
  std::string text;
  const std::filesystem::path file = options_.dir / "katalog.json";
  if (!ReadText(file, &text)) {
    *error = "cannot read " + file.string();
    return false;
  }
  try {
    *out = json::parse(text);
  } catch (const std::exception& e) {
    *error = file.string() + " does not parse: " + e.what();
    return false;
  }
  return true;
}

HttpResponse SetupService::Katalog() const {
  json::Value katalog;
  std::string error;
  if (!LoadKatalog(&katalog, &error))
    return Error(404, "Not Found", "katalog_missing", error);
  katalog["katalog_datei"] = (options_.dir / "katalog.json").string();
  katalog["einrichtung_version"] = kVersion;
  return JsonOk(katalog);
}

HttpResponse SetupService::Pruefen(const HttpRequest& req) {
  CheckOptions check;
  check.exe = options_.exe;
  check.own_port = options_.port;
  const std::string port = req.query_param("port");
  if (!port.empty())
    check.wunsch_port = std::atoi(port.c_str());
  if (options_.memory) {
    check.memory = options_.memory();
  } else {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (probe_ == nullptr)
      probe_ = MakeSystemMemoryProbe();
    check.memory = probe_->System();
  }
  return JsonOk(MachineCheck(check));
}

HttpResponse SetupService::PruefenPfade(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", "invalid_request",
                 std::string("bad JSON: ") + e.what());
  }
  const json::Value* pfade = body.find("pfade");
  if (pfade == nullptr || !pfade->is_object())
    return Error(400, "Bad Request", "invalid_request", "pfade fehlt");
  json::Value ordner = json::Value::array();
  for (const char* key : {"modelle", "daten"}) {
    const std::string pfad = Member(*pfade, key);
    if (pfad.empty())
      continue;
    if (!IsSafePath(pfad))
      return Error(400, "Bad Request", "invalid_path",
                   std::string("pfade.") + key +
                       ": absoluter Pfad mit Laufwerk, ohne '..' und ohne "
                       "UNC (" + pfad + ")");
    json::Value o = json::Value::object();
    o["pfad"] = pfad;
    std::error_code ec;
    const std::filesystem::path dir = NativePath(pfad);
    const bool existed = std::filesystem::is_directory(dir, ec);
    std::filesystem::create_directories(dir, ec);
    o["angelegt"] = !existed && !ec;
    bool writable = false;
    {
      const std::filesystem::path probe = dir / ".gufo-schreibprobe";
      std::ofstream out(probe, std::ios::binary | std::ios::trunc);
      if (out) {
        out << "ok";
        out.close();
        writable = static_cast<bool>(out);
      }
      std::filesystem::remove(probe, ec);
    }
    o["schreibbar"] = writable;
    const auto space = std::filesystem::space(dir, ec);
    o["frei_bytes"] = ec ? 0LL : static_cast<long long>(space.available);
    ordner.push_back(std::move(o));
  }
  json::Value out = json::Value::object();
  out["ordner"] = std::move(ordner);
  return JsonOk(out);
}

std::optional<HttpResponse> SetupService::Handle(const HttpRequest& req) {
  if (!req.path.starts_with("/router/setup/"))
    return std::nullopt;
  const std::string_view rest = std::string_view(req.path).substr(14);
  const bool get = req.method == "GET" || req.method == "HEAD";
  const bool post = req.method == "POST";
  if (rest == "katalog")
    return get ? Katalog()
               : Error(405, "Method Not Allowed", "method_not_allowed", "GET");
  if (rest == "pruefen") {
    if (get)
      return Pruefen(req);
    if (post)
      return PruefenPfade(req);
    return Error(405, "Method Not Allowed", "method_not_allowed", "GET or POST");
  }
  if (rest == "preset" || rest == "admin" || rest == "downloads" ||
      rest == "selbsttest")
    return Error(501, "Not Implemented", "not_implemented",
                 std::string(rest) + " comes with a later 2.1 commit");
  return NotFound();
}

}  // namespace gufo::router::einrichtung
