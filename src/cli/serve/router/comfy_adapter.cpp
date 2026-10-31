#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/cli/serve/router/comfy_preview.hpp"
#include "src/cli/serve/websocket.hpp"

namespace gufo::router::comfy {

namespace fs = std::filesystem;
using server::HttpRequest;
using server::HttpResponse;
using server::Logger;

struct Adapter::Job {
  std::string id;
  long long number{0};
  json::Value prompt;
  json::Value extra_data = json::Value::object();
  std::string client_id;
  std::vector<std::string> outputs;
  Plan plan;
  std::atomic<bool> cancel{false};
  /// Warnings found while running (worker thread only).
  std::vector<std::string> warnings;
};

namespace {

/// A job stopped by /interrupt (ComfyUI's execution_interrupted).
struct Interrupted : std::runtime_error {
  Interrupted() : std::runtime_error("interrupted") {}
};

constexpr std::string_view kWorkDir = ".gufo-work";

fs::path U8Path(std::string_view text) {
  return fs::path(std::u8string(text.begin(), text.end()));
}

std::string PathU8(const fs::path& path) {
  const std::u8string text = path.u8string();
  return std::string(text.begin(), text.end());
}

long long NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string NewUuid() {
  static std::mutex mutex;
  static std::mt19937_64 engine{std::random_device{}() ^
                                static_cast<std::uint64_t>(NowMs())};
  std::uint64_t hi = 0;
  std::uint64_t lo = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    hi = engine();
    lo = engine();
  }
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  char text[40];
  std::snprintf(text, sizeof(text), "%08x-%04x-%04x-%04x-%012llx",
                static_cast<unsigned>(hi >> 32),
                static_cast<unsigned>((hi >> 16) & 0xFFFF),
                static_cast<unsigned>(hi & 0xFFFF),
                static_cast<unsigned>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return text;
}

HttpResponse JsonResponse(int status, const json::Value& body) {
  return HttpResponse{.status = status,
                      .reason = status == 200   ? "OK"
                                : status == 400 ? "Bad Request"
                                : status == 404 ? "Not Found"
                                : status == 405 ? "Method Not Allowed"
                                                : "Error",
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"}}};
}

HttpResponse ErrorResponse(int status, std::string_view type,
                           std::string_view message) {
  json::Value error = json::Value::object();
  error["type"] = std::string(type);
  error["message"] = std::string(message);
  error["details"] = "";
  error["extra_info"] = json::Value::object();
  json::Value body = json::Value::object();
  body["error"] = std::move(error);
  body["node_errors"] = json::Value::object();
  return JsonResponse(status, body);
}

HttpResponse EmptyOk() {
  return HttpResponse{.status = 200, .reason = "OK", .body = ""};
}

std::string ReadAll(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + PathU8(path));
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void WriteAll(const fs::path& path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out)
    throw std::runtime_error("cannot write " + PathU8(path));
}

std::string ContentTypeFor(std::string_view name) {
  const std::string ext = [&] {
    const auto dot = name.rfind('.');
    std::string e(dot == std::string_view::npos ? "" : name.substr(dot + 1));
    for (char& c : e)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e;
  }();
  if (ext == "png")
    return "image/png";
  if (ext == "jpg" || ext == "jpeg")
    return "image/jpeg";
  if (ext == "webp")
    return "image/webp";
  if (ext == "gif")
    return "image/gif";
  if (ext == "flac")
    return "audio/flac";
  if (ext == "wav")
    return "audio/wav";
  if (ext == "mp3")
    return "audio/mpeg";
  if (ext == "ogg")
    return "audio/ogg";
  if (ext == "mp4")
    return "video/mp4";
  if (ext == "webm")
    return "video/webm";
  if (ext == "json")
    return "application/json";
  return "application/octet-stream";
}

/// "image/png" or "image/jpeg" from the leading bytes.
std::string SniffImageType(std::string_view bytes) {
  return bytes.size() > 2 && static_cast<unsigned char>(bytes[0]) == 0xFF &&
                 static_cast<unsigned char>(bytes[1]) == 0xD8
             ? "image/jpeg"
             : "image/png";
}

/// ComfyUI reports subfolders with the platform's separator.
std::string ReportedSubfolder(std::string subfolder) {
#if defined(_WIN32)
  for (char& c : subfolder)
    if (c == '/')
      c = '\\';
#endif
  return subfolder;
}

std::string ErrorMessage(const FetchResult& result) {
  try {
    const json::Value body = json::parse(result.body);
    if (const json::Value* error = body.find("error")) {
      if (error->is_string())
        return error->str();
      const std::string message = error->member_str("message");
      if (!message.empty())
        return message;
    }
  } catch (const std::exception&) {
  }
  return result.body.substr(0, 300);
}

#if defined(_WIN32)
void AppendArgument(std::wstring& command, const std::wstring& argument) {
  if (!command.empty())
    command.push_back(L' ');
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    command += argument;
    return;
  }
  command.push_back(L'"');
  std::size_t backslashes = 0;
  for (const wchar_t c : argument) {
    if (c == L'\\') {
      ++backslashes;
      continue;
    }
    if (c == L'"') {
      command.append(backslashes * 2 + 1, L'\\');
    } else {
      command.append(backslashes, L'\\');
    }
    backslashes = 0;
    command.push_back(c);
  }
  command.append(backslashes * 2, L'\\');
  command.push_back(L'"');
}

int RunProcess(const fs::path& exe, const std::vector<fs::path>& args,
               std::chrono::minutes timeout, std::string* error) {
  std::wstring command;
  AppendArgument(command, exe.wstring());
  for (const fs::path& arg : args)
    AppendArgument(command, arg.wstring());
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(exe.wstring().c_str(), command.data(), nullptr, nullptr,
                      FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                      &process)) {
    *error = "cannot start " + PathU8(exe) + " (error " +
             std::to_string(GetLastError()) + ")";
    return -1;
  }
  CloseHandle(process.hThread);
  const DWORD waited = WaitForSingleObject(
      process.hProcess,
      static_cast<DWORD>(
          std::chrono::duration_cast<std::chrono::milliseconds>(timeout)
              .count()));
  if (waited != WAIT_OBJECT_0) {
    TerminateProcess(process.hProcess, 1);
    WaitForSingleObject(process.hProcess, 5000);
    CloseHandle(process.hProcess);
    *error = PathU8(exe) + " timed out";
    return -1;
  }
  DWORD code = 1;
  GetExitCodeProcess(process.hProcess, &code);
  CloseHandle(process.hProcess);
  return static_cast<int>(code);
}

void MemoryStats(std::uint64_t* total, std::uint64_t* free) {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
  }
}
#else
int RunProcess(const fs::path& exe, const std::vector<fs::path>& args,
               std::chrono::minutes timeout, std::string* error) {
  std::vector<std::string> storage{exe.string()};
  for (const fs::path& arg : args)
    storage.push_back(arg.string());
  std::vector<char*> argv;
  for (std::string& item : storage)
    argv.push_back(item.data());
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) {
    *error = "fork failed";
    return -1;
  }
  if (pid == 0) {
    execv(argv[0], argv.data());
    _exit(127);
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    int status = 0;
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid)
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (std::chrono::steady_clock::now() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      *error = exe.string() + " timed out";
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void MemoryStats(std::uint64_t* total, std::uint64_t* free) {
  std::ifstream in("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (in >> key >> value >> unit) {
    if (key == "MemTotal:")
      *total = value * 1024;
    else if (key == "MemAvailable:")
      *free = value * 1024;
  }
}
#endif

void CollectFiles(const Plan& plan, std::vector<FileRef>* files) {
  for (const Task& task : plan.tasks) {
    if (const auto* image = std::get_if<ImageTask>(&task.spec)) {
      files->insert(files->end(), image->references.begin(),
                    image->references.end());
      if (image->size_source == SizeSource::kCanvas)
        files->push_back(image->canvas);
      if (image->strength)
        files->push_back(image->init_image);
    } else if (const auto* speech = std::get_if<SpeechTask>(&task.spec)) {
      if (speech->reference_audio)
        files->push_back(*speech->reference_audio);
    } else if (const auto* dialogue = std::get_if<DialogueTask>(&task.spec)) {
      for (const SpeechTask& line : dialogue->lines)
        if (line.reference_audio)
          files->push_back(*line.reference_audio);
    } else if (const auto* video = std::get_if<VideoTask>(&task.spec)) {
      if (video->first_frame)
        files->push_back(*video->first_frame);
      if (video->last_frame)
        files->push_back(*video->last_frame);
      for (const auto& [frame, ref] : video->keyframes)
        files->push_back(ref);
      if (video->audio_reference)
        files->push_back(*video->audio_reference);
      if (video->render_video)
        files->push_back(*video->render_video);
      if (video->source_video)
        files->push_back(*video->source_video);
      if (video->guide_video)
        files->push_back(*video->guide_video);
      if (video->control_video)
        files->push_back(*video->control_video);
      if (video->vace_mask)
        files->push_back(*video->vace_mask);
      files->insert(files->end(), video->reference_images.begin(),
                    video->reference_images.end());
    }
  }
}

}  // namespace

Adapter::Adapter(AdapterOptions options) : options_(std::move(options)) {
  for (const char* folder : {"input", "output", "temp"})
    fs::create_directories(options_.root / folder);
  LoadHistory();
}

Adapter::~Adapter() {
  Stop();
}

void Adapter::Start() {
  worker_ = std::thread([this] { WorkerMain(); });
}

void Adapter::Stop() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    if (running_)
      running_->cancel.store(true);
  }
  wake_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

std::optional<HttpResponse> Adapter::Handle(const HttpRequest& request) {
  std::string_view path = request.path;
  if (path.starts_with("/api/"))
    path.remove_prefix(4);
  const std::string& method = request.method;
  const auto only =
      [&](std::string_view allowed) -> std::optional<HttpResponse> {
    if (method == allowed)
      return std::nullopt;
    return ErrorResponse(405, "method_not_allowed", "method is not allowed");
  };
  try {
    if (path == "/prompt") {
      if (method == "POST")
        return Submit(request);
      if (auto bad = only("GET"))
        return bad;
      const std::lock_guard<std::mutex> lock(mutex_);
      json::Value info = json::Value::object();
      info["queue_remaining"] = QueueRemainingLocked();
      json::Value body = json::Value::object();
      body["exec_info"] = std::move(info);
      return JsonResponse(200, body);
    }
    if (path == "/queue")
      return method == "POST" ? QueuePost(request) : QueueGet();
    if (path == "/history")
      return method == "POST" ? HistoryPost(request) : HistoryGet(request, "");
    if (path.starts_with("/history/")) {
      if (auto bad = only("GET"))
        return bad;
      return HistoryGet(request, path.substr(9));
    }
    if (path == "/interrupt") {
      if (auto bad = only("POST"))
        return bad;
      return Interrupt(request);
    }
    if (path == "/free") {
      if (auto bad = only("POST"))
        return bad;
      return Free(request);
    }
    if (path == "/view") {
      if (auto bad = only("GET"))
        return bad;
      return View(request);
    }
    if (path == "/upload/image") {
      if (auto bad = only("POST"))
        return bad;
      return Upload(request);
    }
    if (path == "/upload/mask")
      return ErrorResponse(400, "unsupported",
                           "mask uploads are not supported by gufo");
    if (path == "/system_stats")
      return SystemStats();
    if (path == "/object_info")
      return ObjectInfo("");
    if (path.starts_with("/object_info/"))
      return ObjectInfo(path.substr(13));
    // ComfyUI's /embeddings is a GET list of textual-inversion names. A
    // POST to the bare /embeddings is the llama-server embeddings API and
    // routes to an embedding model by its JSON `model` instead.
    if (path == "/embeddings" && method != "GET" &&
        request.path == "/embeddings")
      return std::nullopt;
    if (path == "/embeddings" || path == "/extensions")
      return JsonResponse(200, json::Value::array());
    if (path == "/features")
      return JsonResponse(200, json::Value::object());
    if (path == "/ws")
      return Socket(request);
  } catch (const std::exception& error) {
    Logger::Warn("comfy", std::string("event=request_failed path=") +
                              request.path + " error=" + error.what());
    return ErrorResponse(500, "server_error", error.what());
  }
  return std::nullopt;
}

HttpResponse Adapter::Submit(const HttpRequest& request) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& error) {
    return ErrorResponse(400, "invalid_prompt",
                         std::string("invalid JSON: ") + error.what());
  }
  const json::Value* prompt = body.find("prompt");
  if (prompt == nullptr || !prompt->is_object())
    return ErrorResponse(400, "no_prompt", "No prompt provided");
  auto job = std::make_shared<Job>();
  job->prompt = *prompt;
  job->client_id = body.member_str("client_id");
  job->id = body.member_str("prompt_id");
  if (job->id.empty())
    job->id = NewUuid();
  if (const json::Value* extra = body.find("extra_data");
      extra != nullptr && extra->is_object())
    job->extra_data = *extra;
  if (!job->client_id.empty())
    job->extra_data["client_id"] = job->client_id;
  job->extra_data["create_time"] = NowMs();
  PlanError plan_error;
  if (!BuildPlan(job->prompt, options_.models, &job->plan, &plan_error)) {
    Logger::Warn("comfy", "event=prompt_rejected type=" + plan_error.type +
                              " message=" + plan_error.message);
    json::Value error = json::Value::object();
    error["type"] = plan_error.type;
    error["message"] = plan_error.message;
    error["details"] = plan_error.details;
    error["extra_info"] = json::Value::object();
    json::Value reply = json::Value::object();
    reply["error"] = std::move(error);
    reply["node_errors"] = plan_error.node_errors;
    return JsonResponse(400, reply);
  }
  // ComfyUI rejects missing input files at validation, before queueing.
  std::vector<FileRef> files;
  CollectFiles(job->plan, &files);
  for (const FileRef& file : files) {
    std::error_code ec;
    const fs::path path = FilePath(file);
    if (path.empty() || !fs::is_regular_file(path, ec))
      return ErrorResponse(400, "prompt_outputs_failed_validation",
                           "Invalid file: " + file.name + " [" + file.folder +
                               "] (upload it with /upload/image first)");
  }
  for (const Task& task : job->plan.tasks)
    for (const OutputTarget& target : task.outputs)
      job->outputs.push_back(target.node_id);
  json::Value reply = json::Value::object();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
      return ErrorResponse(503, "server_error", "adapter is shutting down");
    job->number = next_number_++;
    if (const json::Value* number = body.find("number");
        number != nullptr && number->is_number())
      job->number = static_cast<long long>(number->as_double());
    if (body.find("front") != nullptr && body.find("front")->as_bool())
      pending_.push_front(job);
    else
      pending_.push_back(job);
    reply["prompt_id"] = job->id;
    reply["number"] = job->number;
    reply["node_errors"] = json::Value::object();
    EmitStatusLocked();
  }
  wake_.notify_all();
  std::string families;
  for (const Task& task : job->plan.tasks)
    families += (families.empty() ? "" : ",") + task.class_type;
  Logger::Info("comfy", "event=prompt_queued prompt_id=" + job->id +
                            " tasks=" + families);
  return JsonResponse(200, reply);
}

json::Value Adapter::QueueItemLocked(const Job& job) const {
  json::Value item = json::Value::array();
  item.push_back(job.number);
  item.push_back(job.id);
  item.push_back(job.prompt);
  item.push_back(job.extra_data);
  json::Value outputs = json::Value::array();
  for (const std::string& id : job.outputs)
    outputs.push_back(id);
  item.push_back(std::move(outputs));
  return item;
}

int Adapter::QueueRemainingLocked() const {
  return static_cast<int>(pending_.size()) + (running_ ? 1 : 0);
}

HttpResponse Adapter::QueueGet() {
  const std::lock_guard<std::mutex> lock(mutex_);
  json::Value running = json::Value::array();
  if (running_)
    running.push_back(QueueItemLocked(*running_));
  json::Value pending = json::Value::array();
  for (const auto& job : pending_)
    pending.push_back(QueueItemLocked(*job));
  json::Value body = json::Value::object();
  body["queue_running"] = std::move(running);
  body["queue_pending"] = std::move(pending);
  return JsonResponse(200, body);
}

HttpResponse Adapter::QueuePost(const HttpRequest& request) {
  json::Value body = json::Value::object();
  if (!request.body.empty())
    body = json::parse(request.body);
  const std::lock_guard<std::mutex> lock(mutex_);
  if (body.find("clear") != nullptr && body.find("clear")->as_bool())
    pending_.clear();
  if (const json::Value* remove = body.find("delete");
      remove != nullptr && remove->is_array()) {
    for (const json::Value& id : remove->items())
      std::erase_if(pending_,
                    [&](const auto& job) { return job->id == id.str(); });
  }
  EmitStatusLocked();
  return EmptyOk();
}

HttpResponse Adapter::HistoryGet(const HttpRequest& request,
                                 std::string_view id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  json::Value body = json::Value::object();
  if (!id.empty()) {
    if (const auto found = history_.find(std::string(id));
        found != history_.end())
      body[found->first] = found->second;
    return JsonResponse(200, body);
  }
  std::size_t limit = history_order_.size();
  if (const std::string max_items = request.query_param("max_items");
      !max_items.empty())
    limit = std::min<std::size_t>(
        limit, std::strtoull(max_items.c_str(), nullptr, 10));
  for (std::size_t i = history_order_.size() - limit; i < history_order_.size();
       ++i)
    body[history_order_[i]] = history_.at(history_order_[i]);
  return JsonResponse(200, body);
}

HttpResponse Adapter::HistoryPost(const HttpRequest& request) {
  json::Value body = json::Value::object();
  if (!request.body.empty())
    body = json::parse(request.body);
  const std::lock_guard<std::mutex> lock(mutex_);
  if (body.find("clear") != nullptr && body.find("clear")->as_bool()) {
    history_.clear();
    history_order_.clear();
  }
  if (const json::Value* remove = body.find("delete");
      remove != nullptr && remove->is_array()) {
    for (const json::Value& id : remove->items()) {
      history_.erase(id.str());
      std::erase(history_order_, id.str());
    }
  }
  PersistLocked();
  return EmptyOk();
}

HttpResponse Adapter::Interrupt(const HttpRequest& request) {
  std::string wanted;
  if (!request.body.empty()) {
    try {
      wanted = json::parse(request.body).member_str("prompt_id");
    } catch (const std::exception&) {
    }
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  if (running_ && (wanted.empty() || wanted == running_->id)) {
    running_->cancel.store(true);
    Logger::Info("comfy", "event=interrupt prompt_id=" + running_->id);
  }
  return EmptyOk();
}

HttpResponse Adapter::Free(const HttpRequest& request) {
  bool unload = true;
  if (!request.body.empty()) {
    try {
      const json::Value body = json::parse(request.body);
      unload = body.find("unload_models") != nullptr
                   ? body.find("unload_models")->as_bool()
                   : body.find("free_memory") != nullptr &&
                         body.find("free_memory")->as_bool();
    } catch (const std::exception&) {
    }
  }
  if (unload && options_.unload) {
    Logger::Info("comfy", "event=free unload=idle_workers");
    options_.unload();
  }
  return EmptyOk();
}

fs::path Adapter::FilePath(const FileRef& ref) const {
  if (ref.folder != "input" && ref.folder != "output" && ref.folder != "temp")
    return {};
  const auto relative = SafeRelative(ref.name);
  if (!relative || relative->empty())
    return {};
  return options_.root / ref.folder / U8Path(*relative);
}

std::string Adapter::ReadFileRef(const FileRef& ref) const {
  const fs::path path = FilePath(ref);
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec))
    throw std::runtime_error("file not found: " + ref.name + " [" + ref.folder +
                             "]");
  return ReadAll(path);
}

HttpResponse Adapter::View(const HttpRequest& request) {
  const std::string filename = request.query_param("filename");
  std::string type = request.query_param("type");
  if (type.empty())
    type = "output";
  const std::string subfolder = request.query_param("subfolder");
  if (filename.empty())
    return ErrorResponse(400, "invalid_request", "filename is required");
  FileRef ref{subfolder.empty() ? filename : subfolder + "/" + filename, type};
  const fs::path path = FilePath(ref);
  if (path.empty())
    return ErrorResponse(400, "invalid_request", "invalid file path");
  std::error_code ec;
  if (!fs::is_regular_file(path, ec))
    return ErrorResponse(404, "not_found", "file not found");
  return HttpResponse{
      .status = 200,
      .reason = "OK",
      .body = ReadAll(path),
      .headers = {{"Content-Type", ContentTypeFor(filename)},
                  {"Content-Disposition",
                   "filename=\"" + PathU8(path.filename()) + "\""},
                  {"Cache-Control", "no-cache"}}};
}

HttpResponse Adapter::Upload(const HttpRequest& request) {
  std::vector<MultipartPart> parts;
  std::string error;
  if (!ParseMultipart(request.header("content-type"), request.body, &parts,
                      &error))
    return ErrorResponse(400, "invalid_request", error);
  const MultipartPart* image = nullptr;
  std::string subfolder;
  std::string type = "input";
  bool overwrite = false;
  for (const MultipartPart& part : parts) {
    if (part.name == "image")
      image = &part;
    else if (part.name == "subfolder")
      subfolder = part.data;
    else if (part.name == "type")
      type = part.data;
    else if (part.name == "overwrite")
      overwrite = part.data == "true" || part.data == "1";
  }
  if (image == nullptr || image->filename.empty())
    return ErrorResponse(400, "invalid_request", "image file is required");
  if (type != "input" && type != "output" && type != "temp")
    return ErrorResponse(400, "invalid_request", "invalid type");
  const auto safe_sub = SafeRelative(subfolder);
  std::string base = image->filename;
  if (const auto slash = base.find_last_of("/\\"); slash != std::string::npos)
    base = base.substr(slash + 1);
  if (!safe_sub || !SafeRelative(base) || base.empty() || base == "." ||
      base == "..")
    return ErrorResponse(400, "invalid_request", "invalid file name");
  const fs::path folder = options_.root / type / U8Path(*safe_sub);
  std::string name = base;
  std::error_code ec;
  if (!overwrite) {
    const auto dot = base.rfind('.');
    const std::string stem =
        dot == std::string::npos ? base : base.substr(0, dot);
    const std::string ext = dot == std::string::npos ? "" : base.substr(dot);
    for (int i = 1; fs::exists(folder / U8Path(name), ec); ++i) {
      if (ReadAll(folder / U8Path(name)) == image->data)
        break;  // same content: reuse, as ComfyUI does
      name = stem + " (" + std::to_string(i) + ")" + ext;
    }
  }
  WriteAll(folder / U8Path(name), image->data);
  Logger::Info("comfy", "event=upload name=" + name +
                            " bytes=" + std::to_string(image->data.size()));
  json::Value body = json::Value::object();
  body["name"] = name;
  body["subfolder"] = *safe_sub;
  body["type"] = type;
  return JsonResponse(200, body);
}

HttpResponse Adapter::SystemStats() {
  std::uint64_t total = 0;
  std::uint64_t free = 0;
  MemoryStats(&total, &free);
  json::Value system = json::Value::object();
#if defined(_WIN32)
  system["os"] = "win32";
#else
  system["os"] = "posix";
#endif
  system["ram_total"] = total;
  system["ram_free"] = free;
  system["comfyui_version"] = "gufo-comfy-adapter";
  system["python_version"] = "";
  system["pytorch_version"] = "";
  system["embedded_python"] = false;
  system["argv"] = json::Value::array();
  json::Value device = json::Value::object();
  device["name"] = "gufo router (shared memory)";
  device["type"] = "gufo";
  device["index"] = 0;
  device["vram_total"] = total;
  device["vram_free"] = free;
  device["torch_vram_total"] = 0;
  device["torch_vram_free"] = 0;
  json::Value devices = json::Value::array();
  devices.push_back(std::move(device));
  json::Value body = json::Value::object();
  body["system"] = std::move(system);
  body["devices"] = std::move(devices);
  return JsonResponse(200, body);
}

HttpResponse Adapter::ObjectInfo(std::string_view class_type) {
  const auto files = [&](const char* folder) {
    json::Value list = json::Value::array();
    const fs::path base = options_.root / folder;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec)) {
      if (!it->is_regular_file(ec))
        continue;
      const std::u8string text =
          fs::relative(it->path(), base, ec).generic_u8string();
      list.push_back(std::string(text.begin(), text.end()));
    }
    return list;
  };
  json::Value body = json::Value::object();
  for (const std::string& cls : SupportedClassTypes()) {
    if (!class_type.empty() && cls != class_type)
      continue;
    json::Value required = json::Value::object();
    if (cls == "LoadImage" || cls == "LoadAudio") {
      json::Value spec = json::Value::array();
      spec.push_back(files("input"));
      json::Value upload = json::Value::object();
      upload[cls == "LoadImage" ? "image_upload" : "audio_upload"] = true;
      spec.push_back(std::move(upload));
      required[cls == "LoadImage" ? "image" : "audio"] = std::move(spec);
    }
    json::Value input = json::Value::object();
    input["required"] = std::move(required);
    json::Value info = json::Value::object();
    info["input"] = std::move(input);
    info["output"] = json::Value::array();
    info["output_is_list"] = json::Value::array();
    info["output_name"] = json::Value::array();
    info["name"] = cls;
    info["display_name"] = cls;
    info["description"] = "served by gufo";
    info["python_module"] = "gufo";
    info["category"] = "gufo";
    info["output_node"] = IsOutputClass(cls);
    body[cls] = std::move(info);
  }
  return JsonResponse(200, body);
}

void Adapter::Emit(const std::string& client, std::string_view type,
                   json::Value data) {
  json::Value message = json::Value::object();
  message["type"] = std::string(type);
  message["data"] = std::move(data);
  const std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back(Event{next_event_++, client, message.dump()});
  while (events_.size() > 4096)
    events_.pop_front();
}

bool Adapter::WatchedLocked(const std::string& client) const {
  if (client.empty())
    return !sockets_.empty();
  const auto it = sockets_.find(client);
  return it != sockets_.end() && it->second > 0;
}

void Adapter::EmitPreview(const std::string& client,
                          const std::string& prompt_id,
                          const std::string& node_id, std::string jpeg) {
  const std::lock_guard<std::mutex> lock(mutex_);
  previews_[client] =
      Preview{next_preview_++, prompt_id, node_id, std::move(jpeg)};
}

void Adapter::RelayPreviews(const std::shared_ptr<Job>& job, const Task& task,
                            const std::string& preview_id,
                            const std::string& model, bool image_progress,
                            std::stop_token stop) {
  double seq = 0;
  while (!stop.stop_requested()) {
    // Short sleep slices: the end of the task stops the relay at once.
    const auto until = std::chrono::steady_clock::now() + options_.preview_poll;
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (stop.stop_requested())
      break;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!WatchedLocked(job->client_id))
        continue;  // nobody looks: the worker renders no previews either
    }
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/previews/" + preview_id;
    poll.query = "format=frame-json";
    if (!model.empty())
      poll.query += "&model=" + model;
    poll.request_id = "comfy-preview-" + job->id;
    FetchResult result;
    try {
      result = options_.fetch(poll);
    } catch (const std::exception&) {
      continue;
    }
    if (result.status != 200)
      continue;
    try {
      const json::Value parsed = json::parse(result.body);
      const double next = parsed.member_double("seq", 0);
      if (next <= seq || !parsed.contains("data"))
        continue;
      seq = next;
      auto jpeg = Base64Decode(parsed.member_str("data"));
      if (!jpeg || jpeg->empty())
        continue;
      EmitPreview(job->client_id, job->id, task.node_id, std::move(*jpeg));
      if (image_progress) {
        // Images have no other progress source: the preview's step.
        json::Value data = json::Value::object();
        data["value"] = static_cast<int>(parsed.member_double("step", 0));
        data["max"] = static_cast<int>(parsed.member_double("steps", 0));
        data["prompt_id"] = job->id;
        data["node"] = task.node_id;
        Emit(job->client_id, "progress", std::move(data));
      }
    } catch (const std::exception&) {
      // A malformed preview is skipped; the next poll tries again.
    }
  }
}

void Adapter::EmitStatusLocked() {
  json::Value info = json::Value::object();
  info["queue_remaining"] = QueueRemainingLocked();
  json::Value status = json::Value::object();
  status["exec_info"] = std::move(info);
  json::Value data = json::Value::object();
  data["status"] = std::move(status);
  json::Value message = json::Value::object();
  message["type"] = "status";
  message["data"] = std::move(data);
  events_.push_back(Event{next_event_++, "", message.dump()});
  while (events_.size() > 4096)
    events_.pop_front();
}

HttpResponse Adapter::Socket(const HttpRequest& request) {
  if (!server::IsWebSocketUpgrade(request))
    return ErrorResponse(400, "invalid_request", "WebSocket upgrade required");
  std::string client = request.query_param("clientId");
  if (client.empty())
    client = NewUuid();
  return server::UpgradeWebSocket(
      request, [this, client](server::WebSocket& socket) {
        std::uint64_t last = 0;
        std::uint64_t last_preview = 0;
        std::string hello;
        // Clients announce {"type": "feature_flags", "data":
        // {"supports_preview_metadata": true}} as their first message.
        std::atomic<bool> metadata{false};
        std::thread reader([&socket, &metadata] {
          std::string text;
          while (socket.Receive(&text)) {
            try {
              const json::Value message = json::parse(text);
              if (message.member_str("type") != "feature_flags")
                continue;
              const json::Value* data = message.find("data");
              const json::Value* flag =
                  data != nullptr ? data->find("supports_preview_metadata")
                                  : nullptr;
              metadata = flag != nullptr && flag->is_bool() && flag->as_bool();
              json::Value features = json::Value::object();
              features["supports_preview_metadata"] = true;
              json::Value reply = json::Value::object();
              reply["type"] = "feature_flags";
              reply["data"] = std::move(features);
              (void)socket.SendText(reply.dump());
            } catch (const std::exception&) {
              // Not JSON or not for us; ComfyUI ignores it as well.
            }
          }
        });
        struct Session {
          Adapter& adapter;
          server::WebSocket& socket;
          std::thread& reader;
          const std::string& client;
          ~Session() {
            socket.Close();
            reader.join();
            const std::lock_guard<std::mutex> lock(adapter.mutex_);
            if (--adapter.sockets_[client] <= 0)
              adapter.sockets_.erase(client);
          }
        } session{*this, socket, reader, client};
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          ++sockets_[client];
          last = next_event_ - 1;
          last_preview = next_preview_ - 1;
          json::Value info = json::Value::object();
          info["queue_remaining"] = QueueRemainingLocked();
          json::Value status = json::Value::object();
          status["exec_info"] = std::move(info);
          json::Value data = json::Value::object();
          data["status"] = std::move(status);
          data["sid"] = client;
          json::Value message = json::Value::object();
          message["type"] = "status";
          message["data"] = std::move(data);
          hello = message.dump();
        }
        if (!socket.SendText(hello))
          return;
        auto keepalive = std::chrono::steady_clock::now();
        for (;;) {
          std::vector<std::string> out;
          std::vector<Preview> previews;
          {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_)
              break;
            for (const Event& event : events_) {
              if (event.seq <= last)
                continue;
              last = event.seq;
              if (event.client.empty() || event.client == client)
                out.push_back(event.text);
            }
            std::uint64_t newest = last_preview;
            for (const std::string& key : {client, std::string()}) {
              const auto it = previews_.find(key);
              if (it != previews_.end() && it->second.seq > last_preview) {
                previews.push_back(it->second);
                newest = std::max(newest, it->second.seq);
              }
            }
            last_preview = newest;
          }
          if (socket.cancelled())
            break;
          bool ok = true;
          for (const std::string& text : out)
            ok = ok && socket.SendText(text);
          for (const Preview& preview : previews) {
            if (!metadata.load()) {
              ok = ok && socket.SendBinary(PreviewImageFrame(preview.jpeg));
              continue;
            }
            json::Value meta = json::Value::object();
            meta["node_id"] = preview.node_id;
            meta["prompt_id"] = preview.prompt_id;
            meta["display_node_id"] = preview.node_id;
            meta["parent_node_id"] = nullptr;
            meta["real_node_id"] = preview.node_id;
            meta["image_type"] = "image/jpeg";
            ok = ok && socket.SendBinary(
                           PreviewMetadataFrame(preview.jpeg, meta.dump()));
          }
          if (!previews.empty())
            keepalive = std::chrono::steady_clock::now();
          if (!ok)
            break;
          if (out.empty() && std::chrono::steady_clock::now() - keepalive >
                                 std::chrono::seconds(10)) {
            if (!socket.SendText(hello))
              break;
            keepalive = std::chrono::steady_clock::now();
          } else if (!out.empty()) {
            keepalive = std::chrono::steady_clock::now();
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      });
}

void Adapter::WorkerMain() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
      if (stopping_)
        return;
      job = pending_.front();
      pending_.pop_front();
      running_ = job;
      EmitStatusLocked();
    }
    Execute(job);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      running_.reset();
      EmitStatusLocked();
    }
  }
}

void Adapter::Execute(const std::shared_ptr<Job>& job) {
  const auto started = std::chrono::steady_clock::now();
  json::Value messages = json::Value::array();
  const auto message = [&](std::string_view type, json::Value data) {
    data["prompt_id"] = job->id;
    data["timestamp"] = NowMs();
    json::Value entry = json::Value::array();
    entry.push_back(std::string(type));
    entry.push_back(data);
    messages.push_back(std::move(entry));
    Emit(job->client_id, type, std::move(data));
  };
  message("execution_start", json::Value::object());
  json::Value cached = json::Value::object();
  cached["nodes"] = json::Value::array();
  message("execution_cached", std::move(cached));
  for (const std::string& warning : job->plan.warnings) {
    json::Value data = json::Value::object();
    data["message"] = warning;
    message("gufo_warning", std::move(data));
    Logger::Warn("comfy", "event=warning prompt_id=" + job->id + " " + warning);
  }
  json::Value outputs = json::Value::object();
  json::Value meta = json::Value::object();
  bool ok = true;
  json::Value executed = json::Value::array();
  for (const Task& task : job->plan.tasks) {
    json::Value executing = json::Value::object();
    executing["node"] = task.node_id;
    executing["display_node"] = task.node_id;
    executing["prompt_id"] = job->id;
    Emit(job->client_id, "executing", std::move(executing));
    try {
      if (job->cancel.load())
        throw Interrupted();
      const json::Value ui = RunTask(job, task);
      for (const std::string& warning : job->warnings) {
        json::Value data = json::Value::object();
        data["message"] = warning;
        message("gufo_warning", std::move(data));
        Logger::Warn("comfy",
                     "event=warning prompt_id=" + job->id + " " + warning);
      }
      job->warnings.clear();
      for (const auto& [node, value] : ui.members()) {
        outputs[node] = value;
        json::Value node_meta = json::Value::object();
        node_meta["node_id"] = node;
        node_meta["display_node"] = node;
        node_meta["parent_node"] = nullptr;
        node_meta["real_node_id"] = node;
        meta[node] = std::move(node_meta);
        json::Value done = json::Value::object();
        done["node"] = node;
        done["display_node"] = node;
        done["output"] = value;
        done["prompt_id"] = job->id;
        Emit(job->client_id, "executed", std::move(done));
        executed.push_back(node);
      }
      executed.push_back(task.node_id);
    } catch (const Interrupted&) {
      json::Value data = json::Value::object();
      data["node_id"] = task.node_id;
      data["node_type"] = task.class_type;
      data["executed"] = executed;
      message("execution_interrupted", std::move(data));
      Logger::Warn("comfy", "event=prompt_interrupted prompt_id=" + job->id);
      ok = false;
      break;
    } catch (const std::exception& error) {
      json::Value data = json::Value::object();
      data["node_id"] = task.node_id;
      data["node_type"] = task.class_type;
      data["executed"] = executed;
      data["exception_message"] = std::string(error.what());
      data["exception_type"] = "RuntimeError";
      data["traceback"] = json::Value::array();
      data["current_inputs"] = json::Value::object();
      data["current_outputs"] = json::Value::object();
      message("execution_error", std::move(data));
      Logger::Warn("comfy", "event=prompt_failed prompt_id=" + job->id +
                                " node=" + task.node_id +
                                " error=" + error.what());
      ok = false;
      break;
    }
  }
  if (ok)
    message("execution_success", json::Value::object());
  json::Value done = json::Value::object();
  done["node"] = nullptr;
  done["prompt_id"] = job->id;
  Emit(job->client_id, "executing", std::move(done));
  json::Value status = json::Value::object();
  status["status_str"] = ok ? "success" : "error";
  status["completed"] = ok;
  status["messages"] = std::move(messages);
  json::Value prompt_entry = json::Value::array();
  prompt_entry.push_back(job->number);
  prompt_entry.push_back(job->id);
  prompt_entry.push_back(job->prompt);
  prompt_entry.push_back(job->extra_data);
  json::Value outputs_to_execute = json::Value::array();
  for (const std::string& id : job->outputs)
    outputs_to_execute.push_back(id);
  prompt_entry.push_back(std::move(outputs_to_execute));
  json::Value entry = json::Value::object();
  entry["prompt"] = std::move(prompt_entry);
  entry["outputs"] = std::move(outputs);
  entry["status"] = std::move(status);
  entry["meta"] = std::move(meta);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (history_.find(job->id) == history_.end())
      history_order_.push_back(job->id);
    history_[job->id] = std::move(entry);
    while (history_order_.size() > options_.history_limit) {
      history_.erase(history_order_.front());
      history_order_.erase(history_order_.begin());
    }
    PersistLocked();
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  Logger::Info("comfy", "event=prompt_done prompt_id=" + job->id +
                            " status=" + (ok ? "success" : "error") +
                            " seconds=" + std::to_string(seconds));
}

json::Value Adapter::RunTask(const std::shared_ptr<Job>& job,
                             const Task& task) {
  if (const auto* image = std::get_if<ImageTask>(&task.spec))
    return RunImage(job, task, *image);
  if (const auto* speech = std::get_if<SpeechTask>(&task.spec))
    return RunSpeech(job, task, *speech);
  if (const auto* dialogue = std::get_if<DialogueTask>(&task.spec))
    return RunDialogue(job, task, *dialogue);
  if (const auto* music = std::get_if<MusicTask>(&task.spec))
    return RunMusic(job, task, *music);
  return RunVideo(job, task, std::get<VideoTask>(task.spec));
}

FetchResult Adapter::Call(const std::shared_ptr<Job>& job, std::string method,
                          std::string path, std::string body,
                          std::string content_type) {
  HttpRequest request;
  request.method = std::move(method);
  request.path = std::move(path);
  request.body = std::move(body);
  if (!content_type.empty())
    request.headers.emplace_back("Content-Type", std::move(content_type));
  request.request_id = "comfy-" + job->id;
  request.is_cancelled = [job] { return job->cancel.load(); };
  FetchResult result = options_.fetch(request);
  if (job->cancel.load())
    throw Interrupted();
  return result;
}

json::Value Adapter::RunImage(const std::shared_ptr<Job>& job, const Task& task,
                              const ImageTask& spec) {
  std::vector<std::string> references;
  for (const FileRef& ref : spec.references)
    references.push_back(ReadFileRef(ref));
  int width = spec.width;
  int height = spec.height;
  switch (spec.size_source) {
    case SizeSource::kExplicit:
      // ComfyUI empty latents are multiples of 16 (or 8); gufo Qwen-Image
      // sizes are multiples of 32.
      width = std::max(32, static_cast<int>(std::lround(width / 32.0)) * 32);
      height = std::max(32, static_cast<int>(std::lround(height / 32.0)) * 32);
      break;
    case SizeSource::kSquare:
      width = height = spec.resolution;
      break;
    case SizeSource::kFirstReference: {
      const auto dims = ImageDimensions(references.front());
      if (!dims)
        throw std::runtime_error("cannot read the size of " +
                                 spec.references.front().name);
      std::tie(width, height) =
          ScaleToArea(dims->first, dims->second, spec.resolution);
      break;
    }
    case SizeSource::kCanvas: {
      const auto dims = ImageDimensions(ReadFileRef(spec.canvas));
      if (!dims)
        throw std::runtime_error("cannot read the size of " + spec.canvas.name);
      // gufo Qwen-Image sizes are multiples of 32.
      width =
          std::max(32, static_cast<int>(std::lround(dims->first / 32.0)) * 32);
      height =
          std::max(32, static_cast<int>(std::lround(dims->second / 32.0)) * 32);
      break;
    }
  }
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  std::string init_image, mask;
  if (spec.strength || !spec.mask.name.empty())
    init_image = ReadFileRef(spec.init_image);
  if (!spec.mask.name.empty())
    mask = ReadFileRef(spec.mask);
  const json::Value extension = ImageExtension(spec);
  const json::Value loras = ImageLoras(spec);
  // Live preview under this id; the worker renders it only while the relay
  // below polls it for a connected /ws client.
  const std::string preview_id = PreviewId(job->id, task.node_id);
  const auto gufo_fields = [&](bool with_extension, bool with_preview) {
    json::Value fields =
        with_extension ? extension : json::Value(json::Value::object());
    if (with_preview) {
      json::Value preview = json::Value::object();
      preview["id"] = preview_id;
      fields["preview"] = std::move(preview);
    }
    return fields;
  };
  const auto send = [&](bool with_extension, bool with_preview) {
    const json::Value fields = gufo_fields(with_extension, with_preview);
    if (references.empty() && init_image.empty()) {
      json::Value body = json::Value::object();
      body["model"] = spec.model;
      body["prompt"] = spec.prompt;
      body["size"] = size;
      body["steps"] = spec.steps;
      body["seed"] = spec.seed;
      body["n"] = spec.batch;
      body["response_format"] = "b64_json";
      if (!fields.empty())
        body["gufo"] = fields;
      if (!loras.is_null())
        body["loras"] = loras;
      return Call(job, "POST", "/v1/images/generations", body.dump(),
                  "application/json");
    }
    std::vector<MultipartPart> parts = {
        {"model", "", "", spec.model},
        {"prompt", "", "", spec.prompt},
        {"size", "", "", size},
        {"steps", "", "", std::to_string(spec.steps)},
        {"seed", "", "", std::to_string(spec.seed)},
        {"n", "", "", std::to_string(spec.batch)},
        {"response_format", "", "", "b64_json"}};
    if (!fields.empty())
      parts.push_back({"gufo", "", "application/json", fields.dump()});
    if (!loras.is_null())
      parts.push_back({"loras", "", "application/json", loras.dump()});
    if (!init_image.empty()) {
      const std::string type = SniffImageType(init_image);
      // The start image is the single "image" part, or "init_image" next
      // to "image[]" references or a mask (which would otherwise also make
      // a single "image" a reference).
      parts.push_back(
          {references.empty() && mask.empty() ? "image" : "init_image",
           std::string("init") + (type == "image/jpeg" ? ".jpg" : ".png"), type,
           init_image});
    }
    if (!mask.empty()) {
      const std::string type = SniffImageType(mask);
      parts.push_back({"mask", type == "image/jpeg" ? "mask.jpg" : "mask.png",
                       type, mask});
    }
    for (std::size_t i = 0; i < references.size(); ++i) {
      const std::string type = SniffImageType(references[i]);
      parts.push_back({"image[]",
                       "reference-" + std::to_string(i + 1) +
                           (type == "image/jpeg" ? ".jpg" : ".png"),
                       type, references[i]});
    }
    const std::string boundary = "gufocomfy" + NewUuid();
    return Call(job, "POST", "/v1/images/edits",
                BuildMultipart(parts, boundary),
                "multipart/form-data; boundary=" + boundary);
  };
  std::jthread relay([&](std::stop_token stop) {
    RelayPreviews(job, task, preview_id, spec.model, true, stop);
  });
  FetchResult result = send(true, true);
  if (result.status == 400 &&
      ErrorMessage(result).find("preview") != std::string::npos) {
    // A worker without live previews renders as before.
    result = send(true, false);
  }
  relay.request_stop();
  relay.join();
  if (result.status == 400 && !extension.empty() && !spec.strength &&
      spec.mask.name.empty() &&
      ErrorMessage(result).find("gufo") != std::string::npos) {
    // An image worker without the gufo fields (negative prompt, guidance):
    // render without them and say so, as before.
    job->warnings.push_back("node " + task.node_id +
                            ": the image worker rejected " + extension.dump() +
                            " (" + ErrorMessage(result) +
                            "); rendered without negative prompt / guidance");
    result = send(false, false);
  }
  Logger::Info("comfy",
               "event=image prompt_id=" + job->id + " size=" + size +
                   " steps=" + std::to_string(spec.steps) +
                   " references=" + std::to_string(spec.references.size()) +
                   (loras.is_null() ? "" : " loras=" + loras.dump()) +
                   " status=" + std::to_string(result.status));
  if (result.status != 200)
    throw std::runtime_error("gufo image request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  const json::Value parsed = json::parse(result.body);
  const json::Value* data = parsed.find("data");
  if (data == nullptr || !data->is_array() || data->empty())
    throw std::runtime_error("gufo image response has no data");
  std::vector<std::string> images;
  for (const json::Value& item : data->items()) {
    auto png = Base64Decode(item.member_str("b64_json"));
    if (!png || png->empty())
      throw std::runtime_error("gufo image response has no b64_json");
    images.push_back(std::move(*png));
  }
  json::Value ui = json::Value::object();
  for (const OutputTarget& target : task.outputs) {
    json::Value list = json::Value::array();
    for (const std::string& png : images) {
      json::Value saved = SaveBytes(target, "png", png);
      list.push_back(saved["images"].items().front());
    }
    json::Value node = json::Value::object();
    node["images"] = std::move(list);
    ui[target.node_id] = std::move(node);
  }
  return ui;
}

std::string Adapter::Speak(const std::shared_ptr<Job>& job,
                           const SpeechTask& spec) {
  json::Value body = json::Value::object();
  body["model"] = spec.model;
  body["input"] = spec.text;
  body["voice"] = spec.voice;
  if (!spec.language.empty())
    body["language"] = spec.language;
  if (!spec.instructions.empty())
    body["instructions"] = spec.instructions;
  if (spec.seed)
    body["seed"] = *spec.seed;
  body["response_format"] = "wav";
  for (const auto& [key, value] : spec.sampling.members())
    body[key] = value;
  if (spec.reference_audio) {
    std::string audio = ReadFileRef(*spec.reference_audio);
    if (!audio.starts_with("RIFF")) {
      if (options_.ffmpeg.empty())
        throw std::runtime_error(
            "voice reference " + spec.reference_audio->name +
            " is not WAV; start the router with --comfy-ffmpeg to convert it");
      const fs::path wav = WorkFile("wav");
      std::string error;
      if (!RunFfmpeg({"-i", FilePath(*spec.reference_audio), "-ac", "1", "-c:a",
                      "pcm_s16le", wav},
                     &error))
        throw std::runtime_error("cannot convert the voice reference: " +
                                 error);
      audio = ReadAll(wav);
      fs::remove(wav);
    }
    body["reference_audio"] = Base64Encode(audio);
    if (spec.embedding_only)
      body["voice_clone_mode"] = "speaker_embedding_only";
    else
      body["reference_text"] = spec.reference_text;
  }
  const FetchResult result =
      Call(job, "POST", "/v1/audio/speech", body.dump(), "application/json");
  Logger::Info("comfy", "event=speech prompt_id=" + job->id +
                            " model=" + spec.model +
                            " status=" + std::to_string(result.status) +
                            " bytes=" + std::to_string(result.body.size()));
  if (result.status != 200 || !result.body.starts_with("RIFF"))
    throw std::runtime_error("gufo speech request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  return result.body;
}

json::Value Adapter::SaveSpeech(const Task& task,
                                const std::vector<std::string>& wavs) {
  json::Value ui = json::Value::object();
  for (const OutputTarget& target : task.outputs) {
    json::Value list = json::Value::array();
    for (const std::string& wav : wavs) {
      json::Value saved;
      if (options_.ffmpeg.empty()) {
        saved = SaveBytes(target, "wav", wav);
      } else {
        const fs::path file = WorkFile("wav");
        WriteAll(file, wav);
        saved = SaveAudioTrack(target, file);
        fs::remove(file);
      }
      list.push_back(saved["audio"].items().front());
    }
    json::Value node = json::Value::object();
    node["audio"] = std::move(list);
    ui[target.node_id] = std::move(node);
  }
  return ui;
}

json::Value Adapter::RunSpeech(const std::shared_ptr<Job>& job,
                               const Task& task, const SpeechTask& spec) {
  return SaveSpeech(task, {Speak(job, spec)});
}

json::Value Adapter::RunDialogue(const std::shared_ptr<Job>& job,
                                 const Task& task, const DialogueTask& spec) {
  std::vector<std::string> lines;
  for (std::size_t i = 0; i < spec.lines.size(); ++i) {
    if (job->cancel.load())
      throw Interrupted();
    lines.push_back(Speak(job, spec.lines[i]));
    json::Value data = json::Value::object();
    data["value"] = static_cast<int>(i + 1);
    data["max"] = static_cast<int>(spec.lines.size());
    data["prompt_id"] = job->id;
    data["node"] = task.node_id;
    Emit(job->client_id, "progress", std::move(data));
  }
  if (!spec.merge)
    return SaveSpeech(task, lines);
  auto joined = JoinWavs(lines, spec.pause_seconds);
  if (!joined)
    throw std::runtime_error(
        "cannot join the dialogue lines (different WAV formats)");
  return SaveSpeech(task, {*joined});
}

json::Value Adapter::RunMusic(const std::shared_ptr<Job>& job,
                              const Task& task, const MusicTask& spec) {
  const FetchResult result = Call(job, "POST", "/v1/audio/music",
                                  MusicRequestBody(spec).dump(),
                                  "application/json");
  Logger::Info("comfy", "event=music prompt_id=" + job->id +
                            " model=" + spec.model +
                            " status=" + std::to_string(result.status) +
                            " bytes=" + std::to_string(result.body.size()));
  if (result.status != 200 || !result.body.starts_with("RIFF"))
    throw std::runtime_error("gufo music request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  return SaveSpeech(task, {result.body});
}

json::Value Adapter::RunVideo(const std::shared_ptr<Job>& job, const Task& task,
                              const VideoTask& spec) {
  json::Value body = VideoRequestBody(spec);
  // ComfyUI stretches H3 guide images to the canvas; gufo centre-crops.
  // Stretch here (ffmpeg) so both give the same picture.
  bool warned_stretch = false;
  const auto data_url = [&](const FileRef& ref) {
    std::string bytes = ReadFileRef(ref);
    const auto dims = ImageDimensions(bytes);
    if (spec.family == VideoFamily::kH3 && dims &&
        (dims->first != spec.width || dims->second != spec.height)) {
      if (options_.ffmpeg.empty()) {
        if (!warned_stretch)
          job->warnings.push_back(
              "node " + task.node_id +
              ": guide images are centre-cropped by gufo, not stretched "
              "(start the router with --comfy-ffmpeg)");
        warned_stretch = true;
      } else {
        const fs::path png = WorkFile("png");
        std::string error;
        if (!RunFfmpeg({"-i", FilePath(ref), "-vf",
                        "scale=" + std::to_string(spec.width) + ":" +
                            std::to_string(spec.height) + ":flags=bicubic",
                        "-frames:v", "1", png},
                       &error))
          throw std::runtime_error("cannot resize " + ref.name + ": " + error);
        bytes = ReadAll(png);
        fs::remove(png);
      }
    }
    return "data:" + SniffImageType(bytes) + ";base64," + Base64Encode(bytes);
  };
  if (spec.first_frame)
    body["first_frame"] = data_url(*spec.first_frame);
  if (spec.last_frame)
    body["last_frame"] = data_url(*spec.last_frame);
  if (!spec.keyframes.empty()) {
    json::Value keyframes = json::Value::array();
    for (const auto& [frame, ref] : spec.keyframes) {
      json::Value keyframe = json::Value::object();
      keyframe["frame"] = frame;
      keyframe["image"] = data_url(ref);
      keyframes.push_back(std::move(keyframe));
    }
    body["gufo"]["keyframes"] = std::move(keyframes);
  }
  if (spec.render_video) {
    json::Value camera = json::Value::object();
    camera["render_video"] = "data:video/mp4;base64," +
                             Base64Encode(ReadFileRef(*spec.render_video));
    body["gufo"]["camera"] = std::move(camera);
  }
  if (spec.family == VideoFamily::kVace) {
    // VACE media into gufo.vace: videos as MP4 data URLs, images as they are
    // (gufo fits every input to the video size itself).
    json::Value& vace = body["gufo"]["vace"];
    if (spec.control_video)
      vace["control_video"] = "data:video/mp4;base64," +
                              Base64Encode(ReadFileRef(*spec.control_video));
    if (spec.vace_mask) {
      const std::string bytes = ReadFileRef(*spec.vace_mask);
      vace["mask"] = (ImageDimensions(bytes) ? "data:" + SniffImageType(bytes)
                                             : std::string("data:video/mp4")) +
                     ";base64," + Base64Encode(bytes);
    }
    if (!spec.reference_images.empty()) {
      json::Value references = json::Value::array();
      for (const FileRef& ref : spec.reference_images) {
        const std::string bytes = ReadFileRef(ref);
        references.push_back("data:" + SniffImageType(bytes) + ";base64," +
                             Base64Encode(bytes));
      }
      vace["reference_images"] = std::move(references);
    }
  }
  if (spec.source_video)
    body["source_video"] = "data:video/mp4;base64," +
                           Base64Encode(ReadFileRef(*spec.source_video));
  if (spec.guide_video) {
    // MiniMaxH3AddGuide image from LoadVideo -> GetVideoComponents: gufo
    // decodes the clip and anchors its first 17k + 5 frames.
    json::Value guide = json::Value::object();
    guide["frame"] = spec.guide_video_frame;
    guide["video"] = "data:video/mp4;base64," +
                     Base64Encode(ReadFileRef(*spec.guide_video));
    body["gufo"]["guide_video"] = std::move(guide);
  }
  if (spec.audio_reference && spec.audio_reference_from_video) {
    // The anchor clip's soundtrack (GetVideoComponents audio) as FLAC.
    if (options_.ffmpeg.empty())
      throw std::runtime_error(
          "the H3 audio guide comes from a video; start the router with "
          "--comfy-ffmpeg to extract it");
    const fs::path flac = WorkFile("flac");
    std::string error;
    if (RunFfmpeg({"-i", FilePath(*spec.audio_reference), "-vn", "-map",
                   "0:a:0", "-ac", "2", "-c:a", "flac", flac},
                  &error)) {
      body["audio_reference"] =
          "data:audio/flac;base64," + Base64Encode(ReadAll(flac));
    } else {
      // ComfyUI's GetVideoComponents yields no audio for a silent clip, so
      // the guide anchors the picture only.
      job->warnings.push_back("node " + task.node_id + ": " +
                              spec.audio_reference->name +
                              " has no usable audio track, anchoring the "
                              "picture only (" +
                              error + ")");
    }
    std::error_code ignored;
    fs::remove(flac, ignored);
  } else if (spec.audio_reference) {
    const std::string audio = ReadFileRef(*spec.audio_reference);
    body["audio_reference"] =
        "data:" + ContentTypeFor(spec.audio_reference->name) + ";base64," +
        Base64Encode(audio);
  }
  const FetchResult created =
      Call(job, "POST", "/v1/videos", body.dump(), "application/json");
  if (created.status != 200 && created.status != 202 && created.status != 201)
    throw std::runtime_error("gufo video request failed (HTTP " +
                             std::to_string(created.status) +
                             "): " + ErrorMessage(created));
  const std::string video_id = json::parse(created.body).member_str("id");
  if (video_id.empty())
    throw std::runtime_error("gufo video response has no id");
  Logger::Info("comfy", "event=video_created prompt_id=" + job->id +
                            " video=" + video_id + " model=" + spec.model);
  // The middle frame of the worker's preview loop, while a /ws client looks.
  std::jthread relay([&](std::stop_token stop) {
    RelayPreviews(job, task, video_id, "", false, stop);
  });
  int last_progress = -1;
  for (;;) {
    if (job->cancel.load()) {
      HttpRequest cancel;
      cancel.method = "DELETE";
      cancel.path = "/v1/videos/" + video_id;
      (void)options_.fetch(cancel);
      throw Interrupted();
    }
    std::this_thread::sleep_for(options_.video_poll);
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/videos/" + video_id;
    const FetchResult state = options_.fetch(poll);
    if (state.status != 200)
      throw std::runtime_error("gufo video status failed (HTTP " +
                               std::to_string(state.status) +
                               "): " + ErrorMessage(state));
    const json::Value parsed = json::parse(state.body);
    const std::string status = parsed.member_str("status");
    const int progress = static_cast<int>(parsed.member_double("progress", 0));
    if (progress != last_progress) {
      last_progress = progress;
      json::Value data = json::Value::object();
      data["value"] = progress;
      data["max"] = 100;
      data["prompt_id"] = job->id;
      data["node"] = task.node_id;
      Emit(job->client_id, "progress", std::move(data));
    }
    if (status == "completed")
      break;
    if (status == "failed" || status == "cancelled" || status == "expired") {
      const json::Value* error = parsed.find("error");
      throw std::runtime_error(
          "gufo video " + status +
          (error != nullptr ? ": " + error->member_str("message") : ""));
    }
  }
  const FetchResult content =
      Call(job, "GET", "/v1/videos/" + video_id + "/content", "", "");
  if (content.status != 200 || content.body.empty())
    throw std::runtime_error("gufo video content failed (HTTP " +
                             std::to_string(content.status) + ")");
  const fs::path mp4 = WorkFile("mp4");
  WriteAll(mp4, content.body);
  json::Value ui = json::Value::object();
  try {
    for (const OutputTarget& target : task.outputs) {
      if (target.kind == OutputKind::kVideo &&
          target.class_type == "VHS_VideoCombine") {
        // VideoHelperSuite reports its file under "gifs" with the format
        // and frame rate.
        json::Value saved = SaveBytes(target, "mp4", content.body);
        json::Value entry = saved["images"].items().front();
        entry["format"] = "video/h264-mp4";
        entry["frame_rate"] =
            spec.fps_num > 0
                ? static_cast<double>(spec.fps_num) / spec.fps_den
                : static_cast<double>(spec.fps);
        json::Value gifs = json::Value::array();
        gifs.push_back(std::move(entry));
        json::Value node = json::Value::object();
        node["gifs"] = std::move(gifs);
        ui[target.node_id] = std::move(node);
      } else if (target.kind == OutputKind::kVideo) {
        json::Value saved = SaveBytes(target, "mp4", content.body);
        saved["animated"] = json::Value::array();
        saved["animated"].push_back(true);
        ui[target.node_id] = std::move(saved);
      } else if (target.kind == OutputKind::kAudio) {
        ui[target.node_id] = SaveAudioTrack(target, mp4);
      } else {
        ui[target.node_id] = SaveVideoFrames(target, mp4);
      }
    }
  } catch (...) {
    std::error_code ec;
    fs::remove(mp4, ec);
    throw;
  }
  std::error_code ec;
  fs::remove(mp4, ec);
  return ui;
}

fs::path Adapter::WorkFile(std::string_view extension) {
  const fs::path dir = options_.root / "temp" / U8Path(kWorkDir);
  fs::create_directories(dir);
  return dir / U8Path("work-" + std::to_string(++work_counter_) + "-" +
                      NewUuid().substr(0, 8) + "." + std::string(extension));
}

bool Adapter::RunFfmpeg(const std::vector<fs::path>& args, std::string* error) {
  if (options_.ffmpeg.empty()) {
    *error = "no ffmpeg configured (--comfy-ffmpeg)";
    return false;
  }
  std::vector<fs::path> full = {"-hide_banner", "-loglevel", "error",
                                "-nostdin", "-y"};
  full.insert(full.end(), args.begin(), args.end());
  const int code =
      RunProcess(options_.ffmpeg, full, std::chrono::minutes(20), error);
  if (code != 0 && error->empty())
    *error = "ffmpeg exited with " + std::to_string(code);
  return code == 0;
}

namespace {

struct Destination {
  fs::path folder;
  std::string subfolder;
  std::string base;
  std::string type;
};

Destination DestinationFor(const fs::path& root, const OutputTarget& target) {
  const auto name = SplitPrefix(target.prefix);
  if (!name)
    throw std::runtime_error("filename_prefix '" + target.prefix +
                             "' leaves the output folder");
  Destination destination;
  destination.type = target.temp ? "temp" : "output";
  destination.subfolder = name->subfolder;
  destination.base = name->base;
  destination.folder = root / destination.type / U8Path(destination.subfolder);
  fs::create_directories(destination.folder);
  return destination;
}

int CounterIn(const Destination& destination) {
  std::vector<std::string> names;
  std::error_code ec;
  for (fs::directory_iterator it(destination.folder, ec), end; !ec && it != end;
       it.increment(ec))
    names.push_back(PathU8(it->path().filename()));
  return NextCounter(names, destination.base);
}

json::Value FileEntry(const Destination& destination,
                      const std::string& filename) {
  json::Value entry = json::Value::object();
  entry["filename"] = filename;
  entry["subfolder"] = ReportedSubfolder(destination.subfolder);
  entry["type"] = destination.type;
  return entry;
}

}  // namespace

json::Value Adapter::SaveBytes(const OutputTarget& target,
                               std::string_view extension,
                               const std::string& bytes) {
  const Destination destination = DestinationFor(options_.root, target);
  const std::string filename = OutputFilename(
      destination.base, CounterIn(destination), target.kind, extension);
  WriteAll(destination.folder / U8Path(filename), bytes);
  json::Value list = json::Value::array();
  list.push_back(FileEntry(destination, filename));
  json::Value ui = json::Value::object();
  ui[target.kind == OutputKind::kAudio ? "audio" : "images"] = std::move(list);
  return ui;
}

json::Value Adapter::SaveAudioTrack(const OutputTarget& target,
                                    const fs::path& media) {
  const Destination destination = DestinationFor(options_.root, target);
  const std::string filename = OutputFilename(
      destination.base, CounterIn(destination), OutputKind::kAudio, "flac");
  std::string error;
  if (!RunFfmpeg({"-i", media, "-vn", "-c:a", "flac",
                  destination.folder / U8Path(filename)},
                 &error))
    throw std::runtime_error("cannot write the audio track: " + error);
  json::Value list = json::Value::array();
  list.push_back(FileEntry(destination, filename));
  json::Value ui = json::Value::object();
  ui["audio"] = std::move(list);
  return ui;
}

json::Value Adapter::SaveVideoFrames(const OutputTarget& target,
                                     const fs::path& video) {
  const Destination destination = DestinationFor(options_.root, target);
  const int first = CounterIn(destination);
  std::string error;
  if (!RunFfmpeg({"-i", video, "-start_number", std::to_string(first),
                  destination.folder / U8Path(destination.base + "_%05d_.png")},
                 &error))
    throw std::runtime_error("cannot extract the video frames: " + error);
  json::Value list = json::Value::array();
  std::error_code ec;
  for (int counter = first;; ++counter) {
    const std::string filename =
        OutputFilename(destination.base, counter, OutputKind::kImage, "png");
    if (!fs::exists(destination.folder / U8Path(filename), ec))
      break;
    list.push_back(FileEntry(destination, filename));
  }
  json::Value ui = json::Value::object();
  ui["images"] = std::move(list);
  return ui;
}

void Adapter::PersistLocked() {
  json::Value all = json::Value::object();
  for (const std::string& id : history_order_)
    all.append_member(id, history_.at(id));
  const fs::path file = options_.root / "history.json";
  const fs::path temp = options_.root / "history.json.tmp";
  try {
    WriteAll(temp, all.dump());
    std::error_code ec;
    fs::rename(temp, file, ec);
    if (ec)
      Logger::Warn("comfy", "event=history_save_failed error=" + ec.message());
  } catch (const std::exception& error) {
    Logger::Warn("comfy", std::string("event=history_save_failed error=") +
                              error.what());
  }
}

void Adapter::LoadHistory() {
  const fs::path file = options_.root / "history.json";
  std::error_code ec;
  if (!fs::is_regular_file(file, ec))
    return;
  try {
    const json::Value all = json::parse(ReadAll(file));
    for (const auto& [id, entry] : all.members()) {
      history_order_.push_back(id);
      history_[id] = entry;
      if (const json::Value* prompt = entry.find("prompt");
          prompt != nullptr && prompt->is_array() && !prompt->empty())
        next_number_ = std::max(
            next_number_,
            static_cast<long long>(prompt->items()[0].as_double()) + 1);
    }
  } catch (const std::exception& error) {
    Logger::Warn("comfy", std::string("event=history_load_failed error=") +
                              error.what());
  }
}

}  // namespace gufo::router::comfy
