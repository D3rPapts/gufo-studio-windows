#include "src/cli/serve/router/download.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <system_error>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::router {

using server::Logger;

namespace {

void EnsureCurl() {
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

std::string Lower(std::string text) {
  for (char& c : text)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

std::string Trim(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  return std::string(text);
}

bool IsHex64(std::string_view text) {
  if (text.size() != 64)
    return false;
  return std::all_of(text.begin(), text.end(), [](unsigned char c) {
    return std::isxdigit(c) != 0;
  });
}

std::FILE* OpenFile(const std::filesystem::path& file, bool append) {
#ifdef _WIN32
  std::FILE* out = nullptr;
  if (_wfopen_s(&out, file.c_str(), append ? L"ab" : L"wb") != 0)
    return nullptr;
  return out;
#else
  return std::fopen(file.c_str(), append ? "ab" : "wb");
#endif
}

/// Headers of the last response of a transfer (redirects pass through too;
/// the X-Linked-* pair comes with the first Hugging Face answer).
struct Headers {
  long long content_length{-1};
  long long linked_size{-1};
  std::string linked_etag;
  bool accept_ranges{false};
};

std::size_t HeaderCallback(char* data, std::size_t size, std::size_t count,
                           void* user) {
  auto* headers = static_cast<Headers*>(user);
  const std::string_view line(data, size * count);
  const auto colon = line.find(':');
  if (colon == std::string_view::npos)
    return size * count;
  const std::string name = Lower(Trim(line.substr(0, colon)));
  const std::string value = Trim(line.substr(colon + 1));
  if (name == "content-length") {
    headers->content_length = std::strtoll(value.c_str(), nullptr, 10);
  } else if (name == "x-linked-size" && headers->linked_size < 0) {
    headers->linked_size = std::strtoll(value.c_str(), nullptr, 10);
  } else if (name == "x-linked-etag" && headers->linked_etag.empty()) {
    std::string etag = value;
    if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"')
      etag = etag.substr(1, etag.size() - 2);
    headers->linked_etag = Lower(etag);
  } else if (name == "accept-ranges" && Lower(value).find("bytes") != std::string::npos) {
    headers->accept_ranges = true;
  }
  return size * count;
}

std::string HttpText(long code) {
  switch (code) {
    case 401:
    case 403:
      return "Hugging Face verlangt einen Token (gated); Zugang beantragen "
             "und den Token eintragen, dann \"Weiter\"";
    case 404:
      return "nicht gefunden (Repo, Revision oder Datei)";
    case 429:
      return "zu viele Anfragen (HTTP 429)";
    default:
      return "HTTP " + std::to_string(code);
  }
}

}  // namespace

const char* ToString(DownloadFile::Status status) {
  switch (status) {
    case DownloadFile::Status::kWartet:
      return "wartet";
    case DownloadFile::Status::kLaeuft:
      return "laeuft";
    case DownloadFile::Status::kPause:
      return "pause";
    case DownloadFile::Status::kFertig:
      return "fertig";
    case DownloadFile::Status::kFehler:
      return "fehler";
    case DownloadFile::Status::kVorhanden:
      return "vorhanden";
    case DownloadFile::Status::kPruefung:
      return "pruefung";
    case DownloadFile::Status::kAbgebrochen:
      return "abgebrochen";
  }
  return "?";
}

// ---------------------------------------------------------------------------

/// Per-transfer state the libcurl callbacks see.
struct Downloader::Transfer {
  Downloader* self{nullptr};
  std::size_t index{0};
  CURL* handle{nullptr};  // the GET handle (response code for Write)
  std::FILE* file{nullptr};
  std::filesystem::path part;
  crypto::Sha256Hasher hasher;
  long long written{0};   // bytes in the .part (resumed part included)
  long long resumed_from{0};
  bool code_checked{false};
  bool write_error{false};
  bool restarted{false};  // the server ignored the Range: began at 0
  long response_code{0};
  std::chrono::steady_clock::time_point last_space_check{
      std::chrono::steady_clock::now()};
  Headers headers;

  static std::size_t Write(char* data, std::size_t size, std::size_t count,
                           void* user);
  static int Progress(void* user, curl_off_t, curl_off_t, curl_off_t,
                      curl_off_t);
};

std::size_t Downloader::Transfer::Write(char* data, std::size_t size,
                                        std::size_t count, void* user) {
  auto* t = static_cast<Transfer*>(user);
  const std::size_t n = size * count;
  if (t->write_error)
    return 0;
  if (!t->code_checked) {
    t->code_checked = true;
    if (t->handle != nullptr)
      curl_easy_getinfo(t->handle, CURLINFO_RESPONSE_CODE, &t->response_code);
    // A resumed request must be answered with 206; a 200 means the server
    // sent the whole file from the start, so the .part begins again.
    if (t->resumed_from > 0 && t->response_code == 200) {
      std::fclose(t->file);
      t->file = OpenFile(t->part, false);
      t->hasher = crypto::Sha256Hasher();
      t->self->AddBytes(t->index, -t->written);
      t->written = 0;
      t->restarted = true;
      if (t->file == nullptr) {
        t->write_error = true;
        return 0;
      }
    }
  }
  if (std::fwrite(data, 1, n, t->file) != n) {
    t->write_error = true;
    return 0;
  }
  t->hasher.Update(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(data), n));
  t->written += static_cast<long long>(n);
  t->self->AddBytes(t->index, static_cast<long long>(n));
  return n;
}

int Downloader::Transfer::Progress(void* user, curl_off_t, curl_off_t,
                                   curl_off_t, curl_off_t) {
  auto* t = static_cast<Transfer*>(user);
  Downloader* self = t->self;
  {
    const std::lock_guard<std::mutex> lock(self->mutex_);
    if (self->cancel_ || self->pause_)
      return 1;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now - t->last_space_check > std::chrono::seconds(60)) {
    t->last_space_check = now;
    if (self->TargetFull(t->part)) {
      {
        const std::lock_guard<std::mutex> lock(self->mutex_);
        self->pause_ = true;
        self->hinweis_ =
            "Die Platte ist fast voll (unter 1 GB frei); der Download "
            "wurde angehalten. Platz schaffen, dann \"Weiter\".";
      }
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------

Downloader::Downloader(DownloaderOptions options,
                       std::vector<DownloadFile> files)
    : options_(std::move(options)), files_(std::move(files)) {
  EnsureCurl();
  if (options_.parallel < 1)
    options_.parallel = 1;
}

Downloader::~Downloader() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    cancel_ = true;
    token_.clear();
  }
  cv_.notify_all();
  for (std::thread& t : threads_)
    if (t.joinable())
      t.join();
  threads_.clear();
}

bool Downloader::Start(const std::string& token) {
  bool spawn = false;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    token_ = token;
    cancel_ = false;
    pause_ = false;
    hinweis_.clear();
    bool waiting = false;
    for (const DownloadFile& f : files_)
      waiting = waiting || (f.status == DownloadFile::Status::kWartet &&
                            !f.url.empty());
    if (!waiting)
      return false;
    spawn = active_ == 0;
  }
  if (spawn) {
    JoinFinished();
    const std::lock_guard<std::mutex> lock(mutex_);
    active_ = options_.parallel;
    for (int i = 0; i < options_.parallel; ++i)
      threads_.emplace_back([this] { Worker(); });
  }
  return true;
}

void Downloader::Pause() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    pause_ = true;
  }
  cv_.notify_all();
}

bool Downloader::Resume(const std::string& token) {
  std::string use;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!token.empty())
      token_ = token;
    use = token_;
    pause_ = false;
    cancel_ = false;
    hinweis_.clear();
    for (DownloadFile& f : files_) {
      if (f.url.empty())
        continue;
      if (f.status == DownloadFile::Status::kPause ||
          f.status == DownloadFile::Status::kFehler ||
          f.status == DownloadFile::Status::kAbgebrochen) {
        f.status = DownloadFile::Status::kWartet;
        f.fehler.clear();
        f.fehler_code.clear();
        f.versuche = 0;
      }
    }
  }
  cv_.notify_all();
  Changed();
  return Start(use);
}

void Downloader::Cancel() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    cancel_ = true;
    pause_ = false;
    token_.clear();
    for (DownloadFile& f : files_)
      if (f.status == DownloadFile::Status::kWartet ||
          f.status == DownloadFile::Status::kPause)
        f.status = DownloadFile::Status::kAbgebrochen;
  }
  cv_.notify_all();
  JoinFinished();
  Changed();
}

bool Downloader::running() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return active_ > 0 && !pause_;
}

bool Downloader::paused() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return pause_;
}

std::vector<DownloadFile> Downloader::Files() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return files_;
}

long long Downloader::BytesOpen() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  long long open = 0;
  std::error_code ec;
  for (const DownloadFile& f : files_) {
    if (f.status != DownloadFile::Status::kWartet || f.url.empty())
      continue;
    std::filesystem::path part = f.ziel;
    part += ".part";
    const auto have = std::filesystem::is_regular_file(part, ec)
                          ? static_cast<long long>(
                                std::filesystem::file_size(part, ec))
                          : 0;
    open += std::max(0LL, f.bytes - have);
  }
  return open;
}

json::Value Downloader::Status() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  json::Value out = json::Value::object();
  out["laeuft"] = active_ > 0 && !pause_;
  out["pause"] = pause_;
  out["parallel"] = static_cast<double>(options_.parallel);
  out["rate_bytes_s"] = static_cast<double>(RateLocked());
  out["hinweis"] = hinweis_;
  long long gesamt = 0;
  long long geladen = 0;
  json::Value list = json::Value::array();
  for (const DownloadFile& f : files_) {
    gesamt += f.bytes;
    geladen += f.status == DownloadFile::Status::kVorhanden ||
                       f.status == DownloadFile::Status::kFertig
                   ? f.bytes
                   : std::min(f.geladen, f.bytes > 0 ? f.bytes : f.geladen);
    json::Value d = json::Value::object();
    d["pfad"] = f.pfad;
    d["modell"] = f.modell;
    d["bytes"] = static_cast<double>(f.bytes);
    d["geladen"] = static_cast<double>(
        f.status == DownloadFile::Status::kVorhanden ||
                f.status == DownloadFile::Status::kFertig
            ? f.bytes
            : f.geladen);
    d["status"] = ToString(f.status);
    d["sha256_status"] = f.sha256_status;
    d["sha256_quelle"] = f.sha256_quelle;
    d["sha256_ist"] = f.sha256_ist;
    d["fehler"] = f.fehler;
    d["fehler_code"] = f.fehler_code;
    d["versuche"] = static_cast<double>(f.versuche);
    if (!f.hf_repo.empty()) {
      json::Value q = json::Value::object();
      q["hf_repo"] = f.hf_repo;
      q["revision"] = f.revision.empty() ? std::string("main") : f.revision;
      d["quelle"] = std::move(q);
    } else {
      d["quelle"] = json::Value();
    }
    list.push_back(std::move(d));
  }
  out["gesamt_bytes"] = static_cast<double>(gesamt);
  out["geladen_bytes"] = static_cast<double>(geladen);
  out["dateien"] = std::move(list);
  return out;
}

// ---------------------------------------------------------------------------

void Downloader::JoinFinished() {
  std::vector<std::thread> done;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (active_ != 0)
      return;
    done.swap(threads_);
  }
  for (std::thread& t : done)
    if (t.joinable())
      t.join();
}

void Downloader::Worker() {
  for (;;) {
    std::size_t index = 0;
    bool found = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!cancel_ && !pause_) {
        for (std::size_t i = 0; i < files_.size(); ++i) {
          if (files_[i].status == DownloadFile::Status::kWartet &&
              !files_[i].url.empty()) {
            files_[i].status = DownloadFile::Status::kLaeuft;
            files_[i].geladen = 0;
            index = i;
            found = true;
            break;
          }
        }
      }
      if (!found) {
        --active_;
        break;
      }
    }
    Changed();
    LoadOne(index);
    Changed();
  }
  cv_.notify_all();
  Changed();
}

void Downloader::Set(std::size_t index, DownloadFile::Status status,
                     const std::string& code, const std::string& text) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    DownloadFile& f = files_[index];
    f.status = status;
    f.fehler_code = code;
    f.fehler = text;
  }
  Changed();
}

void Downloader::Changed() {
  if (options_.on_change)
    options_.on_change(*this);
}

void Downloader::AddBytes(std::size_t index, long long n) {
  const std::lock_guard<std::mutex> lock(mutex_);
  files_[index].geladen += n;
  const auto sec = static_cast<long long>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - epoch_)
          .count());
  Bucket& b = rate_[static_cast<std::size_t>(sec % 8)];
  if (b.second != sec) {
    b.second = sec;
    b.bytes = 0;
  }
  if (n > 0)
    b.bytes += n;
}

long long Downloader::RateLocked() const {
  const auto sec = static_cast<long long>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - epoch_)
          .count());
  long long sum = 0;
  for (const Bucket& b : rate_)
    if (b.second >= sec - 5 && b.second < sec)
      sum += b.bytes;
  return sum / 5;
}

bool Downloader::TargetFull(const std::filesystem::path& file) const {
  std::error_code ec;
  const auto space = std::filesystem::space(file.parent_path(), ec);
  if (ec)
    return false;
  return static_cast<long long>(space.available) < options_.min_free_bytes;
}

bool Downloader::WaitRetry(int attempt) {
  const int ms = attempt < static_cast<int>(options_.retry_wait_ms.size())
                     ? options_.retry_wait_ms[static_cast<std::size_t>(attempt)]
                     : options_.retry_wait_ms.empty()
                           ? 0
                           : options_.retry_wait_ms.back();
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, std::chrono::milliseconds(ms),
               [this] { return cancel_ || pause_; });
  return !(cancel_ || pause_);
}

bool Downloader::LoadOne(std::size_t index) {
  for (;;) {
    Transfer transfer;
    transfer.self = this;
    transfer.index = index;
    bool retry = false;
    if (Attempt(index, &transfer, &retry))
      return true;
    int attempt = 0;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (cancel_ || pause_) {
        DownloadFile& f = files_[index];
        f.status = cancel_ ? DownloadFile::Status::kAbgebrochen
                           : DownloadFile::Status::kPause;
        f.fehler_code.clear();
        f.fehler.clear();
        return false;
      }
      if (!retry)
        return false;
      ++files_[index].versuche;
      if (files_[index].versuche > static_cast<int>(options_.retry_wait_ms.size())) {
        files_[index].status = DownloadFile::Status::kFehler;
        files_[index].fehler_code = "network";
        files_[index].fehler =
            "nach " + std::to_string(files_[index].versuche) +
            " Versuchen aufgegeben: " + files_[index].fehler;
        Logger::Warn("router", "event=setup_download_failed file=" +
                                   files_[index].pfad + " error=\"" +
                                   files_[index].fehler + "\"");
        return false;
      }
      attempt = files_[index].versuche - 1;
    }
    Changed();
    if (!WaitRetry(attempt)) {
      const std::lock_guard<std::mutex> lock(mutex_);
      DownloadFile& f = files_[index];
      f.status = cancel_ ? DownloadFile::Status::kAbgebrochen
                         : DownloadFile::Status::kPause;
      return false;
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      files_[index].status = DownloadFile::Status::kLaeuft;
    }
  }
}

bool Downloader::Attempt(std::size_t index, Transfer* t, bool* retry) {
  *retry = false;
  DownloadFile f;
  std::string token;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    f = files_[index];
    token = token_;
  }
  std::error_code ec;
  std::filesystem::create_directories(f.ziel.parent_path(), ec);
  t->part = f.ziel;
  t->part += ".part";
  if (TargetFull(t->part)) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      pause_ = true;
      hinweis_ =
          "Die Platte ist fast voll (unter 1 GB frei); der Download wurde "
          "angehalten. Platz schaffen, dann \"Weiter\".";
    }
    return false;
  }

  const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
      curl_easy_init(), curl_easy_cleanup);
  if (curl == nullptr) {
    Set(index, DownloadFile::Status::kFehler, "network", "curl_easy_init failed");
    return false;
  }
  std::array<char, CURL_ERROR_SIZE> error_buffer{};
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
      nullptr, curl_slist_free_all);
  if (!token.empty()) {
    const std::string auth = "Authorization: Bearer " + token;
    headers.reset(curl_slist_append(nullptr, auth.c_str()));
  }
  const auto common = [&](CURL* c) {
    curl_easy_setopt(c, CURLOPT_URL, f.url.c_str());
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR,
                     options_.allow_http ? "http,https" : "https");
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS_STR,
                     options_.allow_http ? "http,https" : "https");
    curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, options_.connect_timeout_s);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, options_.stall_timeout_s);
    curl_easy_setopt(c, CURLOPT_USERAGENT, options_.user_agent.c_str());
    curl_easy_setopt(c, CURLOPT_ERRORBUFFER, error_buffer.data());
    curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, HeaderCallback);
    curl_easy_setopt(c, CURLOPT_HEADERDATA, &t->headers);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, Transfer::Progress);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, t);
    if (headers != nullptr)
      curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers.get());
  };
  const auto failed = [&](CURLcode rc, const char* phase) -> bool {
    // Returns true when the attempt is over (status set or pause/cancel);
    // *retry says whether another attempt makes sense.
    long code = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &code);
    if (rc == CURLE_ABORTED_BY_CALLBACK)
      return true;
    const std::string what =
        error_buffer[0] != '\0' ? std::string(error_buffer.data())
                                : std::string(curl_easy_strerror(rc));
    if (rc == CURLE_HTTP_RETURNED_ERROR) {
      if (code == 401 || code == 403) {
        Set(index, DownloadFile::Status::kFehler, "gated", HttpText(code));
        return true;
      }
      if (code == 404) {
        Set(index, DownloadFile::Status::kFehler, "not_found", HttpText(code));
        return true;
      }
      if (code >= 500 || code == 429) {
        *retry = true;
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          files_[index].fehler = std::string(phase) + ": " + HttpText(code);
        }
        return true;
      }
      Set(index, DownloadFile::Status::kFehler, "http",
          std::string(phase) + ": " + HttpText(code));
      return true;
    }
    if (rc == CURLE_WRITE_ERROR && t->write_error) {
      Set(index, DownloadFile::Status::kFehler, "write",
          "Schreiben fehlgeschlagen: " + t->part.string());
      return true;
    }
    *retry = true;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      files_[index].fehler = std::string(phase) + ": " + what;
    }
    return true;
  };

  // 1. HEAD: size against the catalogue, gated repos, the LFS digest.
  common(curl.get());
  curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 1L);
  CURLcode rc = curl_easy_perform(curl.get());
  if (rc != CURLE_OK) {
    failed(rc, "HEAD");
    return false;
  }
  long long remote_size = t->headers.linked_size;
  if (remote_size < 0) {
    curl_off_t length = -1;
    curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
    remote_size = static_cast<long long>(length);
  }
  if (f.bytes > 0 && remote_size > 0 && remote_size != f.bytes) {
    Set(index, DownloadFile::Status::kFehler, "size_mismatch",
        "Groesse anders als im Katalog (Hugging Face: " +
            std::to_string(remote_size) + " Bytes, Katalog: " +
            std::to_string(f.bytes) + ")");
    return false;
  }
  std::string expected = f.sha256;
  std::string quelle = expected.empty() ? "" : "katalog";
  if (expected.empty() && IsHex64(t->headers.linked_etag)) {
    expected = t->headers.linked_etag;
    quelle = "hf";
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (f.bytes <= 0 && remote_size > 0) {
      files_[index].bytes = remote_size;
      f.bytes = remote_size;
    }
    files_[index].sha256_quelle = quelle;
  }

  // 2. The .part on disk: resume when it is a prefix, start over when it
  // is longer than the file.
  long long have = 0;
  if (std::filesystem::is_regular_file(t->part, ec))
    have = static_cast<long long>(std::filesystem::file_size(t->part, ec));
  if (f.bytes > 0 && have > f.bytes) {
    std::filesystem::remove(t->part, ec);
    have = 0;
  }
  if (have > 0) {
    Set(index, DownloadFile::Status::kPruefung, "", "");
    std::ifstream in(t->part, std::ios::binary);
    std::vector<std::uint8_t> buffer(1 << 20);
    long long hashed = 0;
    while (in && hashed < have) {
      in.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
      const auto n = in.gcount();
      if (n <= 0)
        break;
      t->hasher.Update(std::span<const std::uint8_t>(
          buffer.data(), static_cast<std::size_t>(n)));
      hashed += n;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (cancel_ || pause_)
          return false;
      }
    }
    if (hashed != have) {
      std::filesystem::remove(t->part, ec);
      have = 0;
      t->hasher = crypto::Sha256Hasher();
    }
    Set(index, DownloadFile::Status::kLaeuft, "", "");
  }
  t->written = have;
  t->resumed_from = have;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    files_[index].geladen = have;
  }

  // 3. GET, streamed into the .part (skipped when the .part is complete).
  const bool complete = f.bytes > 0 && have == f.bytes;
  if (!complete) {
    const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> get(
        curl_easy_init(), curl_easy_cleanup);
    for (int pass = 0; pass < 2; ++pass) {
      t->file = OpenFile(t->part, have > 0);
      if (t->file == nullptr) {
        Set(index, DownloadFile::Status::kFehler, "write",
            "kann nicht schreiben: " + t->part.string());
        return false;
      }
      curl_easy_reset(get.get());
      t->headers = Headers();
      t->handle = get.get();
      error_buffer[0] = '\0';
      common(get.get());
      curl_easy_setopt(get.get(), CURLOPT_WRITEFUNCTION, Transfer::Write);
      curl_easy_setopt(get.get(), CURLOPT_WRITEDATA, t);
      if (have > 0)
        curl_easy_setopt(get.get(), CURLOPT_RESUME_FROM_LARGE,
                         static_cast<curl_off_t>(have));
      // The write callback needs the response code of the final response;
      // libcurl exposes it through getinfo once the headers are complete.
      t->code_checked = false;
      rc = curl_easy_perform(get.get());
      std::fclose(t->file);
      t->file = nullptr;
      if (rc == CURLE_RANGE_ERROR && have > 0 && pass == 0) {
        // The server answered the Range request with the whole file
        // (libcurl refuses to append that): the .part starts over.
        std::filesystem::remove(t->part, ec);
        t->hasher = crypto::Sha256Hasher();
        t->written = 0;
        t->resumed_from = 0;
        t->restarted = true;
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          files_[index].geladen = 0;
        }
        have = 0;
        continue;
      }
      break;
    }
    if (rc != CURLE_OK) {
      // Swap the handles for the error reading (code and buffer belong to
      // the GET handle).
      long code = 0;
      curl_easy_getinfo(get.get(), CURLINFO_RESPONSE_CODE, &code);
      if (rc == CURLE_ABORTED_BY_CALLBACK)
        return false;
      const std::string what =
          error_buffer[0] != '\0' ? std::string(error_buffer.data())
                                  : std::string(curl_easy_strerror(rc));
      if (rc == CURLE_HTTP_RETURNED_ERROR) {
        if (code == 416 && have > 0) {
          // Range not satisfiable: the server thinks the file is complete.
        } else if (code == 401 || code == 403) {
          Set(index, DownloadFile::Status::kFehler, "gated", HttpText(code));
          return false;
        } else if (code == 404) {
          Set(index, DownloadFile::Status::kFehler, "not_found",
              HttpText(code));
          return false;
        } else if (code >= 500 || code == 429) {
          *retry = true;
          const std::lock_guard<std::mutex> lock(mutex_);
          files_[index].fehler = "GET: " + HttpText(code);
          return false;
        } else {
          Set(index, DownloadFile::Status::kFehler, "http",
              "GET: " + HttpText(code));
          return false;
        }
      } else if (rc == CURLE_WRITE_ERROR && t->write_error) {
        Set(index, DownloadFile::Status::kFehler, "write",
            "Schreiben fehlgeschlagen: " + t->part.string());
        return false;
      } else {
        *retry = true;
        const std::lock_guard<std::mutex> lock(mutex_);
        files_[index].fehler = "GET: " + what;
        return false;
      }
    }
    if (f.bytes > 0 && t->written != f.bytes) {
      // Short read without an error: treat like a dropped connection.
      *retry = true;
      const std::lock_guard<std::mutex> lock(mutex_);
      files_[index].fehler = "GET: unvollstaendig (" +
                             std::to_string(t->written) + " von " +
                             std::to_string(f.bytes) + " Bytes)";
      return false;
    }
  }

  // 4. Digest and rename.
  const std::string digest = t->hasher.FinishHex();
  std::string sha_status;
  if (expected.empty()) {
    sha_status = "unbekannt";
  } else if (digest == expected) {
    sha_status = "ok";
  } else {
    sha_status = "falsch";
  }
  if (sha_status == "falsch") {
    std::filesystem::path wrong = f.ziel;
    wrong += ".falsch";
    std::filesystem::remove(wrong, ec);
    std::filesystem::rename(t->part, wrong, ec);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      DownloadFile& ff = files_[index];
      ff.sha256_status = sha_status;
      ff.sha256_ist = digest;
    }
    Set(index, DownloadFile::Status::kFehler, "sha256",
        "Pruefsumme falsch (" + quelle + "); Datei liegt als " +
            wrong.filename().string());
    Logger::Warn("router", "event=setup_download_sha256 file=" + f.pfad +
                               " expected=" + expected + " got=" + digest);
    return false;
  }
  std::filesystem::remove(f.ziel, ec);
  std::filesystem::rename(t->part, f.ziel, ec);
  if (ec) {
    Set(index, DownloadFile::Status::kFehler, "write",
        "Umbenennen fehlgeschlagen: " + f.ziel.string() + " (" +
            ec.message() + ")");
    return false;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    DownloadFile& ff = files_[index];
    ff.sha256_status = sha_status;
    ff.sha256_ist = digest;
    ff.geladen = ff.bytes > 0 ? ff.bytes : t->written;
    ff.status = DownloadFile::Status::kFertig;
    ff.fehler.clear();
    ff.fehler_code.clear();
  }
  Logger::Info("router", "event=setup_download_done file=" + f.pfad +
                             " bytes=" + std::to_string(t->written) +
                             " sha256=" + sha_status +
                             (t->resumed_from > 0 && !t->restarted
                                  ? " resumed_from=" +
                                        std::to_string(t->resumed_from)
                                  : std::string()));
  Changed();
  return true;
}

}  // namespace gufo::router
