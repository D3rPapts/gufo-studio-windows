#include "src/cli/serve/router/selbsttest.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/rgba_image.hpp"
#include "src/cli/serve/router/workers.hpp"
#include "src/core/base64.hpp"

namespace gufo::router::einrichtung {

using server::Logger;

namespace {

/// The requests of modell_pruefung.py, taken over 1:1.
constexpr const char* kPrompt =
    "Beschreibe in fünf Sätzen einen Sonnenaufgang am Meer.";
constexpr const char* kSentence =
    "Am Abend kam der Regen, und die Stadt wurde still, als hätte jemand das "
    "Licht gedimmt.";
constexpr const char* kImagePrompt = "a lighthouse on a cliff at dusk, photo";
constexpr const char* kVideoPrompt = "waves rolling onto a beach at sunset";
constexpr const char* kMusicPrompt = "calm piano, warm strings";
constexpr const char* kResultFolder = "selbsttest";

void CurlInit() {
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

std::string Now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", &tm);
  return buf;
}

std::string Trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.remove_suffix(1);
  return std::string(s);
}

std::string Lower(std::string s) {
  for (char& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool Has(std::string_view text, std::string_view part) {
  return Lower(std::string(text)).find(Lower(std::string(part))) !=
         std::string::npos;
}

/// A file name part from a model id.
std::string Safe(std::string_view id) {
  std::string out;
  for (const char c : id)
    out.push_back(std::isalnum(static_cast<unsigned char>(c)) || c == '-' ||
                          c == '.' || c == '_'
                      ? c
                      : '_');
  return out;
}

bool WriteBytes(const std::filesystem::path& file, std::string_view bytes,
                std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(file.parent_path(), ec);
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) {
    *error = "cannot write " + file.string();
    return false;
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

bool ReadBytes(const std::filesystem::path& file, std::string* out) {
  std::ifstream in(file, std::ios::binary);
  if (!in)
    return false;
  out->assign((std::istreambuf_iterator<char>(in)),
              std::istreambuf_iterator<char>());
  if (out->starts_with("\xEF\xBB\xBF"))
    out->erase(0, 3);
  return true;
}

std::string JsonError(const std::string& body) {
  try {
    const json::Value v = json::parse(body);
    if (const json::Value* e = v.find("error")) {
      if (e->is_string())
        return e->str();
      const std::string m = e->member_str("message");
      if (!m.empty())
        return m;
      const std::string c = e->member_str("code");
      if (!c.empty())
        return c;
    }
    const std::string m = v.member_str("message");
    if (!m.empty())
      return m;
    const std::string d = v.member_str("detail");
    if (!d.empty())
      return d;
  } catch (const std::exception&) {
  }
  return Trim(body.substr(0, 300));
}

std::string OneLine(std::string s) {
  for (char& c : s)
    if (c == '\n' || c == '\r' || c == '\t')
      c = ' ';
  if (s.size() > 300)
    s.resize(300);
  return Trim(s);
}

std::string DataUrl(const std::string& png) {
  return "data:image/png;base64," + core::EncodeBase64(png);
}

std::size_t WriteCallback(char* data, std::size_t size, std::size_t n,
                          void* user) {
  static_cast<std::string*>(user)->append(data, size * n);
  return size * n;
}

std::size_t HeaderCallback(char* data, std::size_t size, std::size_t n,
                           void* user) {
  const std::string_view line(data, size * n);
  const auto colon = line.find(':');
  if (colon != std::string_view::npos &&
      Lower(std::string(line.substr(0, colon))) == "content-type")
    *static_cast<std::string*>(user) = Trim(line.substr(colon + 1));
  return size * n;
}

int ProgressCallback(void* user, curl_off_t, curl_off_t, curl_off_t,
                     curl_off_t) {
  return static_cast<std::atomic<bool>*>(user)->load() ? 1 : 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Planning

std::string TestKind(std::string_view baustein, std::string_view modell) {
  const std::string m = Lower(std::string(modell));
  if (baustein == "zweiter-rechner")
    return "partner";
  if (m.starts_with("lora-"))
    return "keiner";
  if (baustein == "chat")
    return "chat";
  if (baustein == "bilder")
    return "bild";
  if (baustein == "stimme") {
    // Clone and voice design need a reference / description, ASR an audio
    // file, voice conversion two files: no case in the self-test.
    if (m.find("asr") != std::string::npos ||
        m.find("clone") != std::string::npos ||
        m.find("voicedesign") != std::string::npos ||
        m.find("voice-convert") != std::string::npos)
      return "keiner";
    return "stimme";
  }
  if (baustein == "video") {
    // Uni3C and InfiniteTalk need an image / audio input.
    if (m.find("uni3c") != std::string::npos ||
        m.find("infinitetalk") != std::string::npos)
      return "keiner";
    return "video";
  }
  if (baustein == "musik")
    return "musik";
  if (baustein == "3d")
    return "3d";
  if (baustein == "werkzeuge") {
    if (m.find("upscale") != std::string::npos)
      return "upscale";
    return "keiner";
  }
  return "keiner";
}

std::vector<SelfTestCase> PlanSelfTest(const json::Value& katalog,
                                       const std::vector<std::string>& bausteine,
                                       const std::vector<std::string>& modelle,
                                       bool alle_modelle,
                                       std::string_view partner_url) {
  std::vector<SelfTestCase> out;
  const json::Value* blocks = katalog.find("bausteine");
  const auto chosen = [&](std::string_view id) {
    return std::find(modelle.begin(), modelle.end(), id) != modelle.end();
  };
  for (const std::string& b : bausteine) {
    if (b == "zweiter-rechner") {
      if (!partner_url.empty()) {
        SelfTestCase c;
        c.baustein = b;
        c.modell = std::string(partner_url);
        c.art = "partner";
        out.push_back(std::move(c));
      }
      continue;
    }
    const json::Value* block =
        blocks != nullptr && blocks->is_object() ? blocks->find(b) : nullptr;
    const json::Value* list =
        block != nullptr ? block->find("modelle") : nullptr;
    if (list == nullptr || !list->is_array())
      continue;
    std::vector<SelfTestCase> required;
    std::vector<SelfTestCase> optional;
    for (const json::Value& m : list->items()) {
      const std::string id = m.member_str("id");
      if (id.empty() || !chosen(id))
        continue;
      SelfTestCase c;
      c.baustein = b;
      c.modell = id;
      c.art = TestKind(b, id);
      if (c.art == "keiner")
        continue;
      const json::Value* pflicht = m.find("pflicht");
      (pflicht != nullptr && pflicht->as_bool() ? required : optional)
          .push_back(std::move(c));
    }
    if (alle_modelle) {
      for (auto* v : {&required, &optional})
        for (SelfTestCase& c : *v)
          out.push_back(std::move(c));
      continue;
    }
    if (required.empty()) {
      if (!optional.empty())
        out.push_back(std::move(optional.front()));
      continue;
    }
    // One per block: the first required model, the turbo one for pictures.
    std::size_t pick = 0;
    if (b == "bilder") {
      for (std::size_t i = 0; i < required.size(); ++i)
        if (Has(required[i].modell, "turbo")) {
          pick = i;
          break;
        }
    }
    out.push_back(std::move(required[pick]));
  }
  return out;
}

std::vector<std::string> SplitArgs(std::string_view line) {
  std::vector<std::string> out;
  std::string cur;
  bool quoted = false;
  bool any = false;
  for (const char c : line) {
    if (c == '"') {
      quoted = !quoted;
      any = true;
      continue;
    }
    if (!quoted && std::isspace(static_cast<unsigned char>(c))) {
      if (any) {
        out.push_back(cur);
        cur.clear();
        any = false;
      }
      continue;
    }
    cur.push_back(c);
    any = true;
  }
  if (any)
    out.push_back(cur);
  return out;
}

bool ParseStartLine(std::string_view preset_text, StartLine* out) {
  *out = StartLine{};
  bool found = false;
  std::size_t begin = 0;
  while (begin <= preset_text.size()) {
    const auto end = preset_text.find('\n', begin);
    std::string line = Trim(preset_text.substr(
        begin, end == std::string_view::npos ? std::string_view::npos
                                             : end - begin));
    begin = end == std::string_view::npos ? preset_text.size() + 1 : end + 1;
    if (!line.starts_with("#"))
      continue;
    std::string body = Trim(line.substr(1));
    if (body.starts_with("start:")) {
      std::vector<std::string> args = SplitArgs(Trim(body.substr(6)));
      if (args.empty())
        continue;
      out->program = args.front();
      args.erase(args.begin());
      for (std::size_t i = 0; i < args.size(); ++i) {
        if ((args[i] == "--host" || args[i] == "--port") &&
            i + 1 < args.size()) {
          ++i;
          continue;
        }
        out->args.push_back(args[i]);
      }
      found = true;
    } else if (body.starts_with("start-opt:")) {
      for (const std::string& a : SplitArgs(Trim(body.substr(10))))
        out->args.push_back(a);
    } else if (body.starts_with("env:")) {
      const std::string kv = Trim(body.substr(4));
      const auto eq = kv.find('=');
      if (eq == std::string::npos || eq == 0)
        continue;
      out->env.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
  }
  return found;
}

std::string TestImagePng() {
  image::RgbaImage img;
  img.width = 64;
  img.height = 64;
  img.rgba.assign(static_cast<std::size_t>(64 * 64 * 4), 0);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      const double dx = x - 31.5;
      const double dy = y - 31.5;
      const double r = std::sqrt(dx * dx + dy * dy);
      if (r > 24.0)
        continue;
      std::uint8_t* p = &img.rgba[static_cast<std::size_t>((y * 64 + x) * 4)];
      // Orange disc with a darker rim, so a 3D model has some shading.
      const double shade = 1.0 - 0.4 * (r / 24.0);
      p[0] = static_cast<std::uint8_t>(240 * shade);
      p[1] = static_cast<std::uint8_t>(140 * shade);
      p[2] = static_cast<std::uint8_t>(30 * shade);
      p[3] = 255;
    }
  }
  const std::vector<std::uint8_t> png = image::EncodePng(img);
  return std::string(png.begin(), png.end());
}

// ---------------------------------------------------------------------------
// SelfTest

SelfTest::SelfTest(SelfTestOptions options)
    : options_(std::move(options)), tests_(options_.tests) {
  CurlInit();
}

SelfTest::~SelfTest() {
  Cancel();
  if (thread_.joinable())
    thread_.join();
  StopChild();
}

bool SelfTest::Start() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (running_ || finished_ || tests_.empty())
    return false;
  running_ = true;
  zeit_start_ = Now();
  started_ = std::chrono::steady_clock::now();
  thread_ = std::thread([this] { Run(); });
  return true;
}

void SelfTest::Cancel() { cancel_.store(true); }

bool SelfTest::running() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

std::filesystem::path SelfTest::ordner() const {
  return options_.daten / "jobs" / kResultFolder;
}

json::Value SelfTest::Status() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  json::Value out = json::Value::object();
  out["laeuft"] = running_;
  out["fertig"] = finished_;
  out["abgebrochen"] = cancel_.load();
  out["ziel"] = base_url_;
  out["preset"] = options_.preset.string();
  out["ordner"] = ordner().string();
  out["zeit_start"] = zeit_start_;
  out["zeit_ende"] = zeit_ende_;
  out["hinweis"] = hinweis_;
  if (child_port_ != 0) {
    json::Value kind = json::Value::object();
    kind["port"] = child_port_;
    kind["pid"] = static_cast<double>(child_pid_);
    kind["laeuft"] = child_ != nullptr;
    out["kind"] = std::move(kind);
  }
  int ok = 0;
  int fehler = 0;
  json::Value list = json::Value::array();
  for (const SelfTestCase& t : tests_) {
    json::Value v = json::Value::object();
    v["baustein"] = t.baustein;
    v["modell"] = t.modell;
    v["art"] = t.art;
    v["status"] = t.status;
    v["dauer_s"] = std::round(t.dauer_s * 10.0) / 10.0;
    v["hinweis"] = t.hinweis;
    v["log"] = t.log;
    v["datei"] = t.datei;
    list.push_back(std::move(v));
    ok += t.status == "ok";
    fehler += t.status == "fehler";
  }
  out["ok"] = ok;
  out["fehler"] = fehler;
  out["tests"] = std::move(list);
  return out;
}

void SelfTest::Set(std::size_t index, const std::string& status,
                   const std::string& hint, const std::string& log,
                   const std::string& datei, double dauer_s) {
  const std::lock_guard<std::mutex> lock(mutex_);
  SelfTestCase& t = tests_[index];
  t.status = status;
  t.hinweis = hint;
  t.log = log;
  t.datei = datei;
  t.dauer_s = dauer_s;
}

SelfTest::Reply SelfTest::Request(const std::string& method,
                                  const std::string& url,
                                  const std::string& body, long timeout_s,
                                  bool own) {
  Reply r;
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                           curl_easy_cleanup);
  if (curl == nullptr) {
    r.error = "curl_easy_init failed";
    return r;
  }
  CURL* c = curl.get();
  std::array<char, CURL_ERROR_SIZE> error_buffer{};
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_s);
  curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(c, CURLOPT_USERAGENT,
                   "gufo-studio-windows-router/2.1 (selbsttest)");
  curl_easy_setopt(c, CURLOPT_ERRORBUFFER, error_buffer.data());
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &r.body);
  curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, HeaderCallback);
  curl_easy_setopt(c, CURLOPT_HEADERDATA, &r.content_type);
  curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
  curl_easy_setopt(c, CURLOPT_XFERINFODATA, &cancel_);
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
      nullptr, curl_slist_free_all);
  if (own)
    for (const std::string& header : options_.headers)
      headers.reset(curl_slist_append(headers.release(), header.c_str()));
  if (method == "POST") {
    headers.reset(
        curl_slist_append(headers.release(), "Content-Type: application/json"));
    headers.reset(curl_slist_append(headers.release(), "Expect:"));
    curl_easy_setopt(c, CURLOPT_POST, 1L);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(body.size()));
  }
  if (headers != nullptr)
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers.get());
  const CURLcode rc = curl_easy_perform(c);
  if (rc != CURLE_OK) {
    r.error = error_buffer[0] != '\0' ? std::string(error_buffer.data())
                                      : std::string(curl_easy_strerror(rc));
    if (rc == CURLE_ABORTED_BY_CALLBACK)
      r.error = "abgebrochen";
    return r;
  }
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
  r.ok = true;
  return r;
}

SelfTest::Reply SelfTest::Get(const std::string& path, long timeout_s) {
  return Request("GET", base_url_ + path, "", timeout_s, true);
}

SelfTest::Reply SelfTest::Post(const std::string& path, const json::Value& body,
                               long timeout_s) {
  return Request("POST", base_url_ + path, body.dump(), timeout_s, true);
}

void SelfTest::Explain(const Reply& r, std::string* hint, std::string* log) {
  if (!r.ok) {
    *hint = r.error == "abgebrochen" ? "abgebrochen"
                                     : "Router nicht erreichbar: " + r.error;
    *log = "";
    return;
  }
  const std::string message = JsonError(r.body);
  *log = OneLine(message);
  const std::string low = Lower(message);
  if (r.status == 507 || low.find("resourceexhausted") != std::string::npos ||
      low.find("out of memory") != std::string::npos ||
      low.find("bad allocation") != std::string::npos ||
      low.find("bad_alloc") != std::string::npos ||
      low.find("insufficient") != std::string::npos) {
    *hint = "zu wenig Speicher (HTTP " + std::to_string(r.status) + ")";
    return;
  }
  if (low.find("no such file") != std::string::npos ||
      low.find("cannot open") != std::string::npos ||
      low.find("not found") != std::string::npos ||
      low.find("does not exist") != std::string::npos ||
      low.find("missing") != std::string::npos) {
    *hint = "Datei oder Modell fehlt (HTTP " + std::to_string(r.status) +
            "): " + OneLine(message);
    return;
  }
  if (low.find("exited") != std::string::npos ||
      low.find("exit code") != std::string::npos ||
      low.find("worker") != std::string::npos) {
    *hint = "Worker-Fehler (HTTP " + std::to_string(r.status) +
            "): " + OneLine(message);
    return;
  }
  if (r.status == 401 || r.status == 403) {
    *hint = "Zugriff verweigert (HTTP " + std::to_string(r.status) +
            "): der Router verlangt eine Anmeldung (auth = all?)";
    return;
  }
  *hint = "HTTP " + std::to_string(r.status) + ": " + OneLine(message);
}

// ---------------------------------------------------------------------------
// The child router

bool SelfTest::StartChild(std::string* error) {
  std::string text;
  if (!ReadBytes(options_.preset, &text)) {
    *error = "cannot read " + options_.preset.string();
    return false;
  }
  StartLine start;
  if (!ParseStartLine(text, &start)) {
    // A hand-written preset without the header: this gufo.exe, preset only.
    start.program = options_.exe.string();
    start.args = {"router", "--models-preset", options_.preset.string()};
  }
  std::error_code ec;
  if (!std::filesystem::is_regular_file(start.program, ec)) {
    *error = "router program not found: " + start.program +
             " (\"# start:\" line of the preset)";
    return false;
  }
  const int port = ReserveLoopbackPort(error);
  if (port < 0)
    return false;
  WorkerSpec spec;
  spec.model_id = "selbsttest-router";
  spec.command = start.program;
  spec.args = start.args;
  spec.port = port;
  // The `# env:` lines go to the child only, never into this router's
  // own environment (review finding 2).
  spec.env = start.env;
  std::unique_ptr<Worker> child = Worker::Spawn(spec, options_.exe, error);
  if (child == nullptr)
    return false;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    child_ = std::move(child);
    child_pid_ = child_->pid();
    child_port_ = port;
    base_url_ = "http://127.0.0.1:" + std::to_string(port);
  }
  Logger::Info("router", "event=selbsttest_child_start port=" +
                             std::to_string(port) + " program=\"" +
                             start.program + "\" args=" +
                             std::to_string(start.args.size()));
  return true;
}

void SelfTest::StopChild() {
  std::unique_ptr<Worker> child;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    child = std::move(child_);
  }
  if (child == nullptr)
    return;
  Logger::Info("router", "event=selbsttest_child_stop port=" +
                             std::to_string(child->port()));
  child->Stop(std::chrono::seconds(20));
}

bool SelfTest::WaitReady(std::string* error) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options_.start_timeout_s);
  while (!Cancelled()) {
    Reply r = Get("/ready", 5);
    if (r.ok && r.status == 200)
      return true;
    int exit_status = 0;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (child_ != nullptr && child_->PollExit(&exit_status)) {
        *error = "der Router mit dem neuen Preset hat sich beendet (Code " +
                 std::to_string(exit_status) +
                 "); Log des Einrichtungs-Routers lesen "
                 "(model=selbsttest-router)";
        return false;
      }
    }
    if (std::chrono::steady_clock::now() > deadline) {
      *error = "der Router mit dem neuen Preset antwortet nicht nach " +
               std::to_string(options_.start_timeout_s) + " s";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(
        std::min(1000, std::max(50, options_.poll_ms))));
  }
  *error = "abgebrochen";
  return false;
}

void SelfTest::Unload(const std::string& model) {
  if (!options_.unload_after)
    return;
  json::Value body = json::Value::object();
  body["model"] = model;
  const Reply r = Post("/models/unload", body, 120);
  if (!r.ok || r.status >= 300)
    Logger::Warn("router", "event=selbsttest_unload model=" + model +
                               " status=" + std::to_string(r.status) +
                               " error=\"" + OneLine(r.error) + "\"");
}

// ---------------------------------------------------------------------------
// The cases

bool SelfTest::Chat(const SelfTestCase& c, std::string* hint, std::string* log,
                    std::string* out, std::string* ext) {
  json::Value body = json::Value::object();
  body["model"] = c.modell;
  body["max_tokens"] = 160;
  body["temperature"] = 0;
  json::Value messages = json::Value::array();
  json::Value m = json::Value::object();
  m["role"] = "user";
  m["content"] = kPrompt;
  messages.push_back(std::move(m));
  body["messages"] = std::move(messages);
  json::Value kwargs = json::Value::object();
  kwargs["enable_thinking"] = false;
  body["chat_template_kwargs"] = std::move(kwargs);
  const Reply r = Post("/v1/chat/completions", body, options_.test_timeout_s);
  if (!r.ok || r.status != 200) {
    Explain(r, hint, log);
    return false;
  }
  try {
    const json::Value v = json::parse(r.body);
    const json::Value* choices = v.find("choices");
    if (choices == nullptr || !choices->is_array() || choices->empty())
      throw std::runtime_error("no choices");
    const json::Value* msg = choices->items().front().find("message");
    const std::string text = msg != nullptr ? msg->member_str("content") : "";
    if (Trim(text).empty())
      throw std::runtime_error("empty answer");
    *out = text;
    *ext = "txt";
    const json::Value* usage = v.find("usage");
    const double tokens =
        usage != nullptr ? usage->member_double("completion_tokens") : 0.0;
    *hint = OneLine(text.substr(0, 80)) +
            (tokens > 0 ? " (" + std::to_string(static_cast<int>(tokens)) +
                              " Tokens)"
                        : "");
    return true;
  } catch (const std::exception& e) {
    *hint = std::string("Antwort unlesbar: ") + e.what();
    *log = OneLine(r.body);
    return false;
  }
}

bool SelfTest::Image(const SelfTestCase& c, std::string* hint,
                     std::string* log, std::string* out, std::string* ext) {
  json::Value body = json::Value::object();
  body["model"] = c.modell;
  body["prompt"] = kImagePrompt;
  body["size"] = "512x512";
  body["seed"] = 7;
  body["response_format"] = "b64_json";
  const Reply r = Post("/v1/images/generations", body, options_.test_timeout_s);
  if (!r.ok || r.status != 200) {
    Explain(r, hint, log);
    return false;
  }
  try {
    const json::Value v = json::parse(r.body);
    const json::Value* data = v.find("data");
    if (data == nullptr || !data->is_array() || data->empty())
      throw std::runtime_error("no data");
    const std::string b64 = data->items().front().member_str("b64_json");
    if (b64.empty())
      throw std::runtime_error("no b64_json");
    if (core::DecodeBase64(b64, core::Base64Options{}, *out) !=
        core::Base64Status::kOk)
      throw std::runtime_error("b64_json does not decode");
    *ext = "png";
    *hint = "512x512, " + std::to_string(out->size() / 1024) + " KB";
    return true;
  } catch (const std::exception& e) {
    *hint = std::string("Antwort unlesbar: ") + e.what();
    *log = OneLine(r.body);
    return false;
  }
}

bool SelfTest::Speech(const SelfTestCase& c, std::string* hint,
                      std::string* log, std::string* out, std::string* ext) {
  // The first voice the model lists (CustomVoice has fixed speakers,
  // cloning models the voices of the preset); none when the list is empty.
  std::string voice;
  {
    const Reply v = Get("/v1/audio/voices?model=" + c.modell,
                        options_.test_timeout_s);
    if (v.ok && v.status == 200) {
      try {
        const json::Value j = json::parse(v.body);
        const json::Value* list = j.find("voices");
        if (list == nullptr)
          list = j.find("data");
        if (list != nullptr && list->is_array() && !list->empty()) {
          const json::Value& first = list->items().front();
          voice = first.is_string() ? first.str()
                                    : first.member_str("id", first.member_str(
                                                                 "name"));
        }
      } catch (const std::exception&) {
      }
    }
  }
  json::Value body = json::Value::object();
  body["model"] = c.modell;
  body["input"] = kSentence;
  body["language"] = "german";
  body["response_format"] = "wav";
  if (!voice.empty())
    body["voice"] = voice;
  const Reply r = Post("/v1/audio/speech", body, options_.test_timeout_s);
  if (!r.ok || r.status != 200) {
    Explain(r, hint, log);
    return false;
  }
  if (r.body.size() < 44 || !r.body.starts_with("RIFF")) {
    *hint = "keine WAV-Antwort (" + r.content_type + ")";
    *log = OneLine(r.body);
    return false;
  }
  *out = r.body;
  *ext = "wav";
  *hint = std::to_string(r.body.size() / 1024) + " KB WAV" +
          (voice.empty() ? "" : ", Stimme " + voice);
  return true;
}

bool SelfTest::Music(const SelfTestCase& c, std::string* hint,
                     std::string* log, std::string* out, std::string* ext) {
  json::Value body = json::Value::object();
  body["model"] = c.modell;
  body["prompt"] = kMusicPrompt;
  body["lyrics"] = "[Instrumental]";
  body["duration"] = 5;
  body["seed"] = 7;
  body["audio_codes"] = false;
  const Reply r = Post("/v1/audio/music", body, options_.test_timeout_s);
  if (!r.ok || r.status != 200) {
    Explain(r, hint, log);
    return false;
  }
  if (r.body.size() < 44 || !r.body.starts_with("RIFF")) {
    *hint = "keine WAV-Antwort (" + r.content_type + ")";
    *log = OneLine(r.body);
    return false;
  }
  *out = r.body;
  *ext = "wav";
  *hint = "5 s, " + std::to_string(r.body.size() / 1024) + " KB WAV";
  return true;
}

bool SelfTest::Job(const SelfTestCase& c, const std::string& path,
                   const json::Value& body, std::string* hint,
                   std::string* log, std::string* out, std::string* ext) {
  const Reply r = Post(path, body, 300);
  if (!r.ok || (r.status != 200 && r.status != 201 && r.status != 202)) {
    Explain(r, hint, log);
    return false;
  }
  std::string id;
  try {
    id = json::parse(r.body).member_str("id");
  } catch (const std::exception&) {
  }
  if (id.empty()) {
    *hint = "Auftrag ohne id";
    *log = OneLine(r.body);
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options_.test_timeout_s);
  std::string status;
  while (!Cancelled()) {
    const Reply s = Get(path + "/" + id, 60);
    if (!s.ok) {
      Explain(s, hint, log);
      return false;
    }
    json::Value v;
    try {
      v = json::parse(s.body);
    } catch (const std::exception&) {
    }
    status = v.member_str("status");
    if (status == "completed")
      break;
    if (status == "failed" || status == "cancelled" || status == "expired" ||
        s.status >= 400) {
      const json::Value* e = v.find("error");
      const std::string message =
          e != nullptr ? (e->is_string() ? e->str() : e->member_str("message"))
                       : JsonError(s.body);
      Reply fake;
      fake.ok = true;
      fake.status = s.status >= 400 ? s.status : 500;
      json::Value eb = json::Value::object();
      eb["message"] = message.empty() ? "Auftrag " + status : message;
      json::Value wrap = json::Value::object();
      wrap["error"] = std::move(eb);
      fake.body = wrap.dump();
      Explain(fake, hint, log);
      *hint = "Auftrag " + (status.empty() ? std::string("fehlgeschlagen")
                                           : status) +
              ": " + *hint;
      return false;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      *hint = "Zeit ueberschritten (" +
              std::to_string(options_.test_timeout_s) + " s), Stand " +
              status;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(
        std::max(50, options_.poll_ms)));
  }
  if (Cancelled()) {
    *hint = "abgebrochen";
    return false;
  }
  const Reply content = Get(path + "/" + id + "/content", 300);
  if (!content.ok || content.status != 200) {
    Explain(content, hint, log);
    return false;
  }
  *out = content.body;
  *ext = c.art == "3d" ? "glb" : "mp4";
  *hint = std::to_string(content.body.size() / 1024) + " KB";
  return true;
}

bool SelfTest::Upscale(const SelfTestCase& c, std::string* hint,
                       std::string* log, std::string* out, std::string* ext) {
  json::Value body = json::Value::object();
  body["model"] = c.modell;
  body["image"] = DataUrl(TestImagePng());
  body["scale"] = 2;
  body["response_format"] = "b64_json";
  const Reply r = Post("/v1/images/upscale", body, options_.test_timeout_s);
  if (!r.ok || r.status != 200) {
    Explain(r, hint, log);
    return false;
  }
  try {
    const json::Value v = json::parse(r.body);
    const json::Value* data = v.find("data");
    if (data == nullptr || !data->is_array() || data->empty())
      throw std::runtime_error("no data");
    const std::string b64 = data->items().front().member_str("b64_json");
    if (b64.empty())
      throw std::runtime_error("no b64_json");
    if (core::DecodeBase64(b64, core::Base64Options{}, *out) !=
        core::Base64Status::kOk)
      throw std::runtime_error("b64_json does not decode");
    *ext = "png";
    *hint = "64x64 -> " + std::to_string(static_cast<int>(v.member_double("width"))) +
            "x" + std::to_string(static_cast<int>(v.member_double("height")));
    return true;
  } catch (const std::exception& e) {
    *hint = std::string("Antwort unlesbar: ") + e.what();
    *log = OneLine(r.body);
    return false;
  }
}

bool SelfTest::Partner(std::string* hint, std::string* log) {
  std::string url = options_.partner_url;
  while (!url.empty() && url.back() == '/')
    url.pop_back();
  const Reply r = Request("GET", url + "/health", "", 15);
  if (!r.ok) {
    *hint = "zweiter Rechner nicht erreichbar: " + r.error;
    return false;
  }
  if (r.status != 200) {
    Explain(r, hint, log);
    return false;
  }
  *hint = "erreichbar (" + url + ")";
  return true;
}

void SelfTest::RunCase(std::size_t index) {
  SelfTestCase c;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    c = tests_[index];
    tests_[index].status = "laeuft";
  }
  const auto t0 = std::chrono::steady_clock::now();
  const auto seconds = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  };
  std::string hint;
  std::string log;
  std::string out;
  std::string ext;
  bool ok = false;
  if (c.art == "chat") {
    ok = Chat(c, &hint, &log, &out, &ext);
  } else if (c.art == "bild") {
    ok = Image(c, &hint, &log, &out, &ext);
  } else if (c.art == "stimme") {
    ok = Speech(c, &hint, &log, &out, &ext);
  } else if (c.art == "musik") {
    ok = Music(c, &hint, &log, &out, &ext);
  } else if (c.art == "video") {
    json::Value body = json::Value::object();
    body["model"] = c.modell;
    body["prompt"] = kVideoPrompt;
    body["size"] = "640x352";
    body["seconds"] = 1;
    json::Value g = json::Value::object();
    g["seed"] = 7;
    g["steps"] = 20;
    body["gufo"] = std::move(g);
    ok = Job(c, "/v1/videos", body, &hint, &log, &out, &ext);
  } else if (c.art == "3d") {
    json::Value body = json::Value::object();
    body["model"] = c.modell;
    body["image"] = DataUrl(TestImagePng());
    body["seed"] = 7;
    ok = Job(c, "/v1/3d/generations", body, &hint, &log, &out, &ext);
  } else if (c.art == "upscale") {
    ok = Upscale(c, &hint, &log, &out, &ext);
  } else if (c.art == "partner") {
    ok = Partner(&hint, &log);
  } else {
    Set(index, "uebersprungen", "kein Testfall fuer dieses Modell", "", "",
        0.0);
    return;
  }
  std::string datei;
  if (ok && !out.empty()) {
    datei = c.baustein + "-" + Safe(c.modell) + "." + ext;
    std::string error;
    if (!WriteBytes(ordner() / datei, out, &error)) {
      ok = false;
      hint = error;
      datei.clear();
    }
  }
  if (Cancelled() && !ok) {
    Set(index, "abgebrochen", "abgebrochen", log, "", seconds());
    return;
  }
  Set(index, ok ? "ok" : "fehler", hint, log, datei, seconds());
  Logger::Info("router", std::string("event=selbsttest_case baustein=") +
                             c.baustein + " model=" + c.modell + " art=" +
                             c.art + " status=" + (ok ? "ok" : "fehler") +
                             " seconds=" + std::to_string(seconds()) +
                             " hint=\"" + OneLine(hint) + "\"");
  if (c.art != "partner")
    Unload(c.modell);
}

void SelfTest::WriteReport() {
  const json::Value status = Status();
  std::string error;
  WriteBytes(ordner() / "selbsttest.json", status.tojson() + "\n", &error);
  // einrichtung.json next to the preset carries the report, so the page
  // and the regie see it after a restart.
  const std::filesystem::path file =
      options_.preset.parent_path() / "einrichtung.json";
  std::string text;
  if (!ReadBytes(file, &text))
    return;
  json::Value ein;
  try {
    ein = json::parse(text);
  } catch (const std::exception&) {
    return;
  }
  if (!ein.is_object())
    return;
  ein["selbsttest"] = status;
  if (!WriteBytes(file, ein.tojson() + "\n", &error))
    Logger::Warn("router", "event=selbsttest_report error=\"" + error + "\"");
}

void SelfTest::Run() {
  std::string error;
  bool target_ok = true;
  if (options_.base_url.empty()) {
    target_ok = StartChild(&error) && WaitReady(&error);
  } else {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      base_url_ = options_.base_url;
      while (!base_url_.empty() && base_url_.back() == '/')
        base_url_.pop_back();
    }
    target_ok = WaitReady(&error);
  }
  if (!target_ok) {
    const std::lock_guard<std::mutex> lock(mutex_);
    hinweis_ = error;
    for (SelfTestCase& t : tests_) {
      if (t.status != "wartet")
        continue;
      t.status = Cancelled() ? "abgebrochen" : "fehler";
      t.hinweis = error;
    }
  } else {
    // Models the target does not know are skipped, not red: a running
    // router with the old preset answers them only after a restart.
    std::vector<std::string> known;
    {
      const Reply r = Get("/v1/models", 30);
      if (r.ok && r.status == 200) {
        try {
          const json::Value v = json::parse(r.body);
          if (const json::Value* data = v.find("data");
              data != nullptr && data->is_array())
            for (const json::Value& m : data->items())
              known.push_back(m.member_str("id"));
        } catch (const std::exception&) {
        }
      }
    }
    for (std::size_t i = 0; i < tests_.size() && !Cancelled(); ++i) {
      SelfTestCase c;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        c = tests_[i];
      }
      if (c.art != "partner" && !known.empty() &&
          std::find(known.begin(), known.end(), c.modell) == known.end()) {
        Set(i, "uebersprungen",
            "nicht im laufenden Preset (" + base_url_ +
                "); nach einem Neustart des Routers wieder pruefen",
            "", "", 0.0);
        continue;
      }
      RunCase(i);
    }
    if (Cancelled()) {
      const std::lock_guard<std::mutex> lock(mutex_);
      for (SelfTestCase& t : tests_)
        if (t.status == "wartet" || t.status == "laeuft")
          t.status = "abgebrochen";
    }
  }
  StopChild();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    finished_ = true;
    zeit_ende_ = Now();
  }
  WriteReport();
  const json::Value s = Status();
  Logger::Info("router", "event=selbsttest_done ok=" +
                             std::to_string(static_cast<int>(
                                 s.member_double("ok"))) +
                             " fehler=" +
                             std::to_string(static_cast<int>(
                                 s.member_double("fehler"))) +
                             " seconds=" +
                             std::to_string(std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                started_)
                                                .count()));
}

}  // namespace gufo::router::einrichtung
