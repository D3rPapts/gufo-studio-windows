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

constexpr const char* kBlockMarker = "# baustein:";
constexpr const char* kModelMarker = "# modell:";
constexpr const char* kKeptMarker =
    "# --- kept from the previous preset (not managed by einrichten.py) ---";
/// pfade.* keys the templates may use as placeholders (PFAD_SCHLUESSEL).
constexpr const char* kPathKeys[] = {
    "modelle", "app", "daten", "pages", "preset", "app_35b", "app_3d",
    "app_moss", "app_d", "app_trellis2", "comfy_dir", "comfy_ffmpeg",
    "names_file"};
/// Catalogue `worker` -> the pfade.* key holding its folder.
const std::map<std::string, std::string> kWorkerPath = {
    {"gufo.exe", "app"},     {"router", "app"},         {"35b", "app_35b"},
    {"gufo-3d", "app_3d"},   {"gufo-moss", "app_moss"}, {"gufo-d", "app_d"},
    {"trellis2", "app_trellis2"}};
/// Folders einrichten.py creates under pfade.daten.
constexpr const char* kDataFolders[] = {
    "jobs",
    "prompt-cache",
    "video-jobs",
    "video-jobs-wan",
    "video-jobs-camera",
    "video-jobs-talk",
    "video-jobs-rife",
    "jobs\\objekt",
    "jobs\\trellis2",
    "prompt-cache-35b",
    "prompt-cache-flashnext",
    "prompt-cache-flashnext-q3",
    "prompt-cache-hemmingway",
    "voices",
    "expert-profiles"};

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

bool WriteText(const std::filesystem::path& file, std::string_view text,
               std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(file.parent_path(), ec);
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) {
    *error = "cannot write " + file.string();
    return false;
  }
  out << text;
  if (!out) {
    *error = "write failed: " + file.string();
    return false;
  }
  return true;
}

std::vector<std::string> Lines(std::string_view text) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find('\n', start);
    std::string_view line =
        text.substr(start, end == std::string_view::npos ? std::string_view::npos
                                                          : end - start);
    if (line.ends_with('\r'))
      line.remove_suffix(1);
    if (end == std::string_view::npos) {
      if (!line.empty())
        out.emplace_back(line);
      break;
    }
    out.emplace_back(line);
    start = end + 1;
  }
  return out;
}

bool Blank(std::string_view line) {
  return Trim(line).empty();
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

std::vector<std::string> StringList(const json::Value* value) {
  std::vector<std::string> out;
  if (value == nullptr || !value->is_array())
    return out;
  for (const json::Value& item : value->items())
    out.push_back(Text(&item));
  return out;
}

json::Value ListOf(const std::vector<std::string>& list) {
  json::Value out = json::Value::array();
  for (const std::string& item : list)
    out.push_back(item);
  return out;
}

bool Contains(const std::vector<std::string>& list, std::string_view item) {
  return std::find(list.begin(), list.end(), item) != list.end();
}

std::string Now(const char* format) {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &now);
#else
  localtime_r(&now, &tm);
#endif
  char buffer[64];
  std::strftime(buffer, sizeof buffer, format, &tm);
  return buffer;
}

std::string Gb(double bytes) {
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%.2f GB", bytes / 1e9);
  return buffer;
}

std::string JoinStr(const std::vector<std::string>& parts,
                    std::string_view sep) {
  std::string out;
  for (const std::string& part : parts) {
    if (!out.empty())
      out += sep;
    out += part;
  }
  return out;
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

// ---------------------------------------------------------------------------
// Templates (lies_vorlage, fuelle)

struct Block {
  std::string baustein;
  std::string modell;
  std::vector<std::string> zeilen;  // starts with the "# baustein:" line
};

bool ReadTemplate(const std::filesystem::path& dir, const std::string& name,
                  std::vector<Block>* out, std::string* error) {
  std::string text;
  const std::filesystem::path file = dir / (name + ".ini");
  if (!ReadText(file, &text)) {
    *error = "Vorlage fehlt: " + file.string();
    return false;
  }
  out->clear();
  Block* current = nullptr;
  for (const std::string& line : Lines(text)) {
    if (line.starts_with(kBlockMarker)) {
      out->push_back({});
      current = &out->back();
      current->baustein =
          std::string(Trim(std::string_view(line).substr(
              std::strlen(kBlockMarker))));
      current->zeilen.push_back(line);
      continue;
    }
    if (current == nullptr)
      continue;
    if (line.starts_with(kModelMarker))
      current->modell = std::string(
          Trim(std::string_view(line).substr(std::strlen(kModelMarker))));
    current->zeilen.push_back(line);
  }
  for (Block& block : *out)
    while (!block.zeilen.empty() && Blank(block.zeilen.back()))
      block.zeilen.pop_back();
  return true;
}

bool PlaceholderChar(char c, bool first) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_' ||
         (!first && std::isdigit(static_cast<unsigned char>(c)));
}

/// `{name}` must have a value; `{name?}` empty drops the line; a line
/// `#@flag rest` is `rest` when the flag is set, else dropped.
bool Fill(const std::vector<std::string>& zeilen,
          const std::map<std::string, std::string>& werte,
          const std::set<std::string>& flags, std::vector<std::string>* out,
          std::string* error) {
  for (const std::string& raw : zeilen) {
    std::string line = raw;
    if (line.starts_with("#@")) {
      const auto space = line.find(' ');
      const std::string flag =
          line.substr(2, space == std::string::npos ? std::string::npos
                                                    : space - 2);
      if (!flags.contains(flag))
        continue;
      line = space == std::string::npos ? "" : line.substr(space + 1);
    }
    std::string filled;
    std::vector<std::string> missing;
    bool drop = false;
    std::size_t i = 0;
    while (i < line.size()) {
      if (line[i] != '{') {
        filled += line[i++];
        continue;
      }
      std::size_t j = i + 1;
      while (j < line.size() && PlaceholderChar(line[j], j == i + 1))
        ++j;
      const bool optional = j < line.size() && line[j] == '?';
      const std::size_t close = optional ? j + 1 : j;
      if (j == i + 1 || close >= line.size() || line[close] != '}') {
        filled += line[i++];
        continue;
      }
      const std::string name = line.substr(i + 1, j - i - 1);
      const auto it = werte.find(name);
      if (it == werte.end() || it->second.empty()) {
        if (optional) {
          drop = true;
        } else {
          missing.push_back(name);
          filled += line.substr(i, close + 1 - i);
        }
      } else {
        filled += it->second;
      }
      i = close + 1;
    }
    if (!missing.empty()) {
      *error = "Platzhalter ohne Wert: [";
      for (std::size_t k = 0; k < missing.size(); ++k)
        *error += (k ? ", '" : "'") + missing[k] + "'";
      *error += "] in Zeile: " + line;
      return false;
    }
    if (drop)
      continue;
    out->push_back(filled);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Catalogue view

struct Model {
  std::string baustein;
  const json::Value* json{nullptr};
};

/// id -> (baustein, model) in block order (einrichten.py's `alle`).
std::vector<std::pair<std::string, Model>> AllModels(const json::Value& katalog) {
  std::vector<std::pair<std::string, Model>> out;
  const json::Value* bausteine = katalog.find("bausteine");
  if (bausteine == nullptr)
    return out;
  for (const char* b : kBausteine) {
    const json::Value* block = bausteine->find(b);
    if (block == nullptr)
      continue;
    const json::Value* modelle = block->find("modelle");
    if (modelle == nullptr || !modelle->is_array())
      continue;
    for (const json::Value& m : modelle->items())
      out.emplace_back(m.member_str("id"), Model{b, &m});
  }
  return out;
}

const Model* FindModel(const std::vector<std::pair<std::string, Model>>& alle,
                       std::string_view id) {
  for (const auto& [mid, model] : alle)
    if (mid == id)
      return &model;
  return nullptr;
}

bool Truthy(const json::Value* value) {
  if (value == nullptr || value->is_null())
    return false;
  if (value->is_bool())
    return value->as_bool();
  if (value->is_number())
    return value->as_double() != 0.0;
  if (value->is_string())
    return !value->str().empty();
  return !value->empty();
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

std::string ParentOf(std::string_view path) {
  while (path.size() > 1 && (path.back() == '\\' || path.back() == '/'))
    path.remove_suffix(1);
  const auto cut = path.find_last_of("\\/");
  if (cut == std::string_view::npos)
    return "";
  if (cut == 0)
    return std::string(path.substr(0, 1));
  if (cut == 2 && path[1] == ':')
    return std::string(path.substr(0, 3));  // C:\x -> C:\ (pathlib: C:\)
  return std::string(path.substr(0, cut));
}

std::map<std::string, std::map<std::string, std::string>> PresetSections(
    std::string_view text) {
  std::map<std::string, std::map<std::string, std::string>> out;
  std::string current;
  bool in_section = false;
  for (const std::string& raw : Lines(text)) {
    const std::string_view line = Trim(raw);
    if (line.empty() || line.starts_with('#'))
      continue;
    if (line.starts_with('[')) {
      const auto close = line.find(']');
      current = std::string(Trim(line.substr(1, close == std::string_view::npos
                                                    ? std::string_view::npos
                                                    : close - 1)));
      out[current];
      in_section = true;
      continue;
    }
    if (!in_section)
      continue;
    const auto eq = line.find('=');
    const std::string key(Trim(line.substr(0, eq)));
    const std::string value(
        eq == std::string_view::npos ? "" : Trim(line.substr(eq + 1)));
    out[current][key] = value;
  }
  return out;
}

std::vector<std::string> ForeignBlocks(std::string_view old) {
  std::vector<std::string> fremd;
  bool eigen = false;
  bool kopf = false;
  for (const std::string& line : Lines(old)) {
    if (line.starts_with(kBlockMarker)) {
      eigen = true;
      kopf = false;
      continue;
    }
    if (eigen) {
      if (line.starts_with('[')) {
        kopf = true;
        continue;
      }
      if (kopf && Blank(line)) {
        eigen = false;
        continue;
      }
      continue;
    }
    fremd.push_back(line);
  }
  while (!fremd.empty() && Blank(fremd.back()))
    fremd.pop_back();
  std::vector<std::string> out;
  for (const std::string& line : fremd)
    if (!line.starts_with("# --- kept from the previous preset"))
      out.push_back(line);
  return out;
}

// ---------------------------------------------------------------------------
// The plan (einrichten.py: pruefe_antworten, waehle_modelle, pruefe_dateien,
// pruefe_worker, werte_fuer, flags_fuer, partner_modelle, baue_preset,
// start_cmd, bericht)

namespace {

bool CheckAnswers(json::Value* a, std::string* error) {
  for (const char* key : {"rechner", "ram_gb", "bausteine", "pfade"}) {
    if (a->find(key) == nullptr) {
      *error = std::string("antworten.json: '") + key + "' fehlt";
      return false;
    }
  }
  const double ram = a->member_double("ram_gb", 0.0);
  if (ram != 64.0 && ram != 128.0) {
    *error = "ram_gb muss 64 oder 128 sein";
    return false;
  }
  for (const std::string& b : StringList(a->find("bausteine"))) {
    bool known = false;
    for (const char* k : kBausteine)
      known = known || b == k;
    if (!known) {
      *error = "unbekannter Baustein '" + b +
               "' (erlaubt: ['chat', 'bilder', 'stimme', 'video', 'musik', "
               "'3d', 'werkzeuge', 'zweiter-rechner'])";
      return false;
    }
  }
  const json::Value* pfade = a->find("pfade");
  for (const char* key : {"modelle", "app", "daten"}) {
    if (pfade == nullptr || Member(*pfade, key).empty()) {
      *error = std::string("pfade.") + key + " fehlt";
      return false;
    }
  }
  json::Value& r = (*a)["router"];
  if (!r.is_object())
    r = json::Value::object();
  const auto def = [](json::Value& object, const char* key, json::Value value) {
    if (const json::Value* v = object.find(key); v == nullptr || v->is_null())
      object[key] = std::move(value);
  };
  def(r, "port", 8081);
  def(r, "erreichbarkeit", "nur-dieser-rechner");
  def(r, "idle_seconds", 1800);
  def(r, "reserve_mib", 2048);
  def(r, "webhook_allow", "127.0.0.1");
  def(r, "adresse", "");
  for (const char* key : {"schalter", "partner"})
    if (const json::Value* v = a->find(key); v == nullptr || !v->is_object())
      (*a)[key] = json::Value::object();
  for (const char* key :
       {"modelle_optional", "modelle_weglassen", "lizenzen_bestaetigt"})
    if (const json::Value* v = a->find(key); v == nullptr || !v->is_array())
      (*a)[key] = json::Value::array();
  return true;
}

bool ChooseModels(const json::Value& a,
                  const std::vector<std::pair<std::string, Model>>& alle,
                  std::vector<std::string>* reihe,
                  std::vector<std::string>* hinweise, std::string* error) {
  const std::vector<std::string> bausteine = StringList(a.find("bausteine"));
  const std::vector<std::string> optional = StringList(a.find("modelle_optional"));
  const std::vector<std::string> weglassen = StringList(a.find("modelle_weglassen"));
  const double ram = a.member_double("ram_gb", 0.0);
  for (const std::string& mid : optional) {
    if (FindModel(alle, mid) == nullptr) {
      *error = "unbekanntes Modell '" + mid + "' (IDs siehe katalog.json)";
      return false;
    }
  }
  for (const std::string& mid : weglassen) {
    if (FindModel(alle, mid) == nullptr) {
      *error = "unbekanntes Modell '" + mid + "' (IDs siehe katalog.json)";
      return false;
    }
  }
  std::set<std::string> gewaehlt;
  for (const auto& [mid, model] : alle) {
    if (!Contains(bausteine, model.baustein))
      continue;
    const json::Value* vorgabe = model.json->find("vorgabe_ram");
    if (Truthy(model.json->find("pflicht")) ||
        (vorgabe != nullptr && vorgabe->is_number() &&
         vorgabe->as_double() == ram) ||
        Contains(optional, mid))
      gewaehlt.insert(mid);
  }
  for (const std::string& mid : optional) {
    const Model* model = FindModel(alle, mid);
    if (!Contains(bausteine, model->baustein)) {
      *error = "Modell '" + mid + "' gehoert zum Baustein '" + model->baustein +
               "', der nicht gewaehlt ist";
      return false;
    }
  }
  for (const std::string& mid : weglassen)
    gewaehlt.erase(mid);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& [mid, model] : alle) {
      if (!gewaehlt.contains(mid))
        continue;
      for (const std::string& dep : StringList(model.json->find("benoetigt"))) {
        if (!gewaehlt.contains(dep) && FindModel(alle, dep) != nullptr) {
          gewaehlt.insert(dep);
          changed = true;
          hinweise->push_back(mid + " braucht " + dep + ": mit aufgenommen");
        }
      }
    }
  }
  reihe->clear();
  for (const auto& [mid, model] : alle)
    if (gewaehlt.contains(mid))
      reihe->push_back(mid);
  return true;
}

std::string JoinPath(std::string base, std::string_view rel) {
  if (!base.empty() && base.back() != '\\' && base.back() != '/')
    base += '\\';
  return base + std::string(rel);
}

std::filesystem::path NativePath(std::string_view text) {
#ifdef _WIN32
  return std::filesystem::path(std::string(text));
#else
  std::string posix(text);
  std::replace(posix.begin(), posix.end(), '\\', '/');
  return std::filesystem::path(posix);
#endif
}

struct FileCheck {
  struct Missing {
    std::string modell;
    std::string pfad;
    long long bytes{0};
    const json::Value* quellen{nullptr};
  };
  struct WrongSize {
    std::string modell;
    std::string pfad;
    long long soll{0};
    long long ist{0};
  };
  std::vector<Missing> fehlend;
  std::vector<WrongSize> falsch;
  long long bytes_da{0};
  long long bytes_fehlend{0};
  std::vector<std::pair<std::string, std::string>> daten_fehlend;
};

FileCheck CheckFiles(const json::Value& a, const std::vector<std::string>& reihe,
                     const std::vector<std::pair<std::string, Model>>& alle) {
  FileCheck out;
  const json::Value& pfade = *a.find("pfade");
  const std::string wurzel = Member(pfade, "modelle");
  const std::string daten = Member(pfade, "daten");
  std::set<std::string> geprueft;
  std::error_code ec;
  for (const std::string& mid : reihe) {
    const Model* model = FindModel(alle, mid);
    const json::Value* dateien = model->json->find("dateien");
    if (dateien != nullptr && dateien->is_array()) {
      for (const json::Value& d : dateien->items()) {
        const std::string pfad = d.member_str("pfad");
        if (!geprueft.insert(pfad).second)
          continue;
        const long long bytes =
            static_cast<long long>(d.member_double("bytes", 0.0));
        const long long soll =
            Truthy(d.find("hardlink_auf")) ? 0 : bytes;
        const std::filesystem::path file = NativePath(JoinPath(wurzel, pfad));
        if (!std::filesystem::is_regular_file(file, ec)) {
          out.fehlend.push_back({mid, pfad, bytes, model->json->find("quellen")});
          out.bytes_fehlend += soll;
          continue;
        }
        const auto size =
            static_cast<long long>(std::filesystem::file_size(file, ec));
        if (bytes != 0 && size != bytes) {
          out.falsch.push_back({mid, pfad, bytes, size});
          out.bytes_fehlend += soll;
        } else {
          out.bytes_da += soll;
        }
      }
    }
    for (const std::string& dd : StringList(model->json->find("daten_dateien"))) {
      if (!std::filesystem::is_regular_file(NativePath(JoinPath(daten, dd)), ec))
        out.daten_fehlend.emplace_back(mid, dd);
    }
  }
  return out;
}

std::map<std::string, std::vector<std::string>> CheckWorkers(
    const json::Value& a, const std::vector<std::string>& reihe,
    const std::vector<std::pair<std::string, Model>>& alle) {
  std::map<std::string, std::vector<std::string>> fehlend;
  const json::Value& pfade = *a.find("pfade");
  std::error_code ec;
  for (const std::string& mid : reihe) {
    const Model* model = FindModel(alle, mid);
    const std::string worker = model->json->member_str("worker");
    const auto key = kWorkerPath.find(worker);
    const std::string k = key == kWorkerPath.end() ? "app" : key->second;
    const std::string pfad = Member(pfade, k);
    if (pfad.empty() ||
        !std::filesystem::is_regular_file(NativePath(JoinPath(pfad, "gufo.exe")),
                                          ec))
      fehlend[k].push_back(mid);
  }
  return fehlend;
}

std::map<std::string, std::string> ValuesFor(
    const json::Value& a, const std::vector<std::string>& reihe,
    const std::vector<std::pair<std::string, Model>>& alle) {
  std::map<std::string, std::string> w;
  const json::Value& p = *a.find("pfade");
  const json::Value& r = *a.find("router");
  const json::Value& partner = *a.find("partner");
  for (const char* key : kPathKeys)
    w[key] = Member(p, key);
  if (w["preset"].empty())
    w["preset"] = JoinPath(w["daten"], "gufo-router.ini");
  if (w["pages"].empty())
    w["pages"] = JoinPath(ParentOf(w["app"]), "pages");
  w["port"] = Member(r, "port");
  w["idle_seconds"] = Member(r, "idle_seconds");
  w["reserve"] = Member(r, "reserve_mib");
  w["webhook_allow"] = Member(r, "webhook_allow");
  w["host"] = Member(r, "erreichbarkeit") == "heimnetz" ? "0.0.0.0" : "127.0.0.1";
  w["rechner"] = Member(a, "rechner");
  w["ram"] = Member(a, "ram_gb");
  w["zeit"] = Now("%Y-%m-%dT%H:%M");
  w["einrichtung_version"] = kVersion;
  const std::vector<std::string> bausteine = StringList(a.find("bausteine"));
  const std::string url =
      Contains(bausteine, "zweiter-rechner") ? Member(partner, "url") : "";
  w["partner"] = url;
  // ^http://([0-9.]+):\d+/?$ -> the IPv4
  std::string ip;
  if (url.starts_with("http://")) {
    std::string_view rest = std::string_view(url).substr(7);
    const auto colon = rest.find(':');
    if (colon != std::string_view::npos) {
      const std::string_view host = rest.substr(0, colon);
      std::string_view port = rest.substr(colon + 1);
      if (port.ends_with('/'))
        port.remove_suffix(1);
      const bool host_ok =
          !host.empty() &&
          std::all_of(host.begin(), host.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) || c == '.';
          });
      const bool port_ok =
          !port.empty() &&
          std::all_of(port.begin(), port.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
          });
      if (host_ok && port_ok)
        ip = std::string(host);
    }
  }
  w["partner_ip"] = ip;
  w["pool_wait"] = Member(partner, "pool_wait_seconds");
  std::vector<std::string> loras;
  for (const std::string& mid : reihe) {
    const Model* model = FindModel(alle, mid);
    if (model->json->member_str("art") != "lora" ||
        model->json->member_str("lora_fuer") != "qwen-image-2.1")
      continue;
    const json::Value* dateien = model->json->find("dateien");
    if (dateien == nullptr)
      continue;
    for (const json::Value& d : dateien->items()) {
      const std::string pfad = d.member_str("pfad");
      if (Lower(pfad).ends_with(".safetensors")) {
        loras.push_back(model->json->member_str("lora_name") + "=" +
                        JoinPath(w["modelle"], pfad));
        break;
      }
    }
  }
  w["lora_qwen_image"] = Join(loras, ';');
  return w;
}

std::set<std::string> FlagsFor(const json::Value& a) {
  std::set<std::string> flags;
  flags.insert(Member(a, "ram_gb"));
  for (const auto& [key, value] : a.find("schalter")->members())
    if (Truthy(&value))
      flags.insert(key);
  // heimnetz | nur-dieser-rechner: the [router] section of the template
  // (auth = lan + loopback admin at home, off on this machine only).
  flags.insert(Member(*a.find("router"), "erreichbarkeit"));
  return flags;
}

std::vector<std::string> PartnerModels(
    const json::Value& a, const std::vector<std::string>& reihe,
    const std::vector<std::pair<std::string, Model>>& alle) {
  if (!Contains(StringList(a.find("bausteine")), "zweiter-rechner"))
    return {};
  const json::Value* pm = a.find("partner")->find("modelle");
  if (pm == nullptr || pm->is_null() ||
      (pm->is_string() && pm->str() == "alle-lokalen")) {
    std::vector<std::string> out;
    for (const std::string& mid : reihe) {
      const Model* model = FindModel(alle, mid);
      if (!model->json->member_str("abschnitt").empty() &&
          model->json->member_str("art") != "lora")
        out.push_back(mid);
    }
    return out;
  }
  return StringList(pm);
}

bool BuildPreset(const std::vector<std::string>& reihe,
                 const std::vector<std::pair<std::string, Model>>& alle,
                 const std::map<std::string, std::string>& werte,
                 const std::set<std::string>& flags,
                 const std::filesystem::path& dir,
                 const std::vector<std::string>& partner, std::string* text,
                 std::string* error) {
  std::vector<std::string> out;
  std::vector<Block> router;
  if (!ReadTemplate(dir, "router", &router, error))
    return false;
  if (router.empty()) {
    *error = "bausteine\\router.ini hat keinen '# baustein:'-Block";
    return false;
  }
  if (!Fill(router.front().zeilen, werte, flags, &out, error))
    return false;
  out.emplace_back();
  std::map<std::string, std::vector<Block>> vorlagen;
  for (const char* b : kBausteine) {
    if (std::string_view(b) == "zweiter-rechner")
      continue;
    if (!ReadTemplate(dir, b, &vorlagen[b], error))
      return false;
  }
  for (const std::string& mid : reihe) {
    const Model* model = FindModel(alle, mid);
    if (model->json->member_str("art") == "lora")
      continue;
    const Block* block = nullptr;
    for (const Block& candidate : vorlagen[model->baustein])
      if (candidate.modell == mid)
        block = &candidate;
    if (block == nullptr) {
      *error = "keine Vorlage fuer " + mid + " in bausteine\\" +
               model->baustein + ".ini";
      return false;
    }
    if (!Fill(block->zeilen, werte, flags, &out, error))
      return false;
    out.emplace_back();
  }
  if (!partner.empty()) {
    std::vector<Block> twin;
    if (!ReadTemplate(dir, "zweiter-rechner", &twin, error))
      return false;
    if (twin.empty()) {
      *error = "bausteine\\zweiter-rechner.ini hat keinen Block";
      return false;
    }
    for (const std::string& mid : partner) {
      // [remote/<id>] must meet the local section (moss-music-8b runs as
      // [external/moss-music]); the router refuses a twin without one.
      std::string sid = mid;
      if (const Model* model = FindModel(alle, mid)) {
        const std::string abschnitt = model->json->member_str("abschnitt");
        if (const auto slash = abschnitt.rfind('/');
            !abschnitt.empty() && slash != std::string::npos)
          sid = abschnitt.substr(slash + 1);
        else if (!abschnitt.empty())
          sid = abschnitt;
      }
      std::map<std::string, std::string> with_id = werte;
      with_id["id"] = sid;
      if (!Fill(twin.front().zeilen, with_id, flags, &out, error))
        return false;
      out.emplace_back();
    }
  }
  *text = Join(out, '\n') + "\n";
  return true;
}

std::string StartCmd(const std::map<std::string, std::string>& werte,
                     std::string_view preset_text) {
  std::string start, opts;
  std::vector<std::string> env;
  for (const std::string& line : Lines(preset_text)) {
    if (line.starts_with("# start: "))
      start = line.substr(9);
    else if (line.starts_with("# start-opt: "))
      opts += " " + line.substr(13);
    else if (line.starts_with("# env: "))
      env.push_back(line.substr(7));
  }
  const std::string app = werte.at("app");
  const std::string log_dir = ParentOf(werte.at("preset"));
  std::string rest;
  if (const auto at = start.find("gufo.exe router"); at != std::string::npos)
    rest = std::string(Trim(std::string_view(start).substr(at + 15)));
  std::vector<std::string> lines = {
      "@echo off",
      "rem gufo router, written by the setup assistant (gufo router "
      "--einrichtung, einrichtung.cpp) " +
          werte.at("einrichtung_version") + " on " + werte.at("zeit") +
          " (KI-unterstuetzt).",
      "rem Logs: router.out.log / router.err.log next to the preset.",
      "chcp 65001 >nul", "cd /d \"" + app + "\""};
  for (const std::string& e : env)
    lines.push_back("set " + e);
  lines.push_back("\"" + app + "\\gufo.exe\" router " + rest + opts +
                  " 1> \"" + log_dir + "\\router.out.log\" 2> \"" + log_dir +
                  "\\router.err.log\"");
  std::string out;
  for (const std::string& line : lines)
    out += line + "\r\n";
  return out;
}

struct Licence {
  std::string modell;
  const json::Value* lizenz;
};

std::vector<Licence> Licences(const std::vector<std::string>& reihe,
                              const std::vector<std::pair<std::string, Model>>& alle) {
  std::vector<Licence> out;
  for (const char* key : {"lizenz", "lizenz_zusatz"}) {
    for (const std::string& mid : reihe) {
      const json::Value* l = FindModel(alle, mid)->json->find(key);
      if (l != nullptr && l->is_object() && Truthy(l->find("bestaetigen")))
        out.push_back({mid, l});
    }
  }
  return out;
}

bool Confirmed(const json::Value& a, std::string_view mid) {
  const std::vector<std::string> list = StringList(a.find("lizenzen_bestaetigt"));
  return Contains(list, mid) || Contains(list, "alle");
}

std::string Report(const json::Value& a, const json::Value& katalog,
                   const std::vector<std::string>& reihe,
                   const std::vector<std::pair<std::string, Model>>& alle,
                   const std::vector<std::string>& hinweise,
                   const FileCheck& pr,
                   const std::map<std::string, std::vector<std::string>>& worker_fehlend,
                   const std::vector<std::string>& pm,
                   const std::vector<Licence>& lizenzen,
                   const std::vector<std::string>& geschrieben) {
  const json::Value& pfade = *a.find("pfade");
  const json::Value& r = *a.find("router");
  std::vector<std::string> z;
  z.push_back("# Einrichtung " + Member(a, "rechner") + " (" + Member(a, "ram_gb") +
              " GB) – " + Now("%Y-%m-%dT%H:%M"));
  z.emplace_back();
  z.push_back(std::string("Router-Einrichtung ") + kVersion + ", Katalog " +
              katalog.member_str("katalog_version") + ". Modelle: " +
              Member(pfade, "modelle") + "; App: " + Member(pfade, "app") +
              "; Daten: " + Member(pfade, "daten") + ".");
  z.push_back("Router: Port " + Member(r, "port") + ", " +
              Member(r, "erreichbarkeit") + ", Leerlauf " +
              Member(r, "idle_seconds") + " s, GUFO_HOST_RESERVE_MIB=" +
              Member(r, "reserve_mib") + ".");
  z.emplace_back();
  z.push_back("## Bausteine und Modelle");
  z.emplace_back();
  const double ram = a.member_double("ram_gb", 0.0);
  for (const std::string& b : StringList(a.find("bausteine"))) {
    std::vector<std::string> ms;
    for (const std::string& mid : reihe)
      if (FindModel(alle, mid)->baustein == b)
        ms.push_back(mid);
    double summe = 0;
    std::set<std::string> seen;
    for (const std::string& mid : ms) {
      const json::Value* dateien = FindModel(alle, mid)->json->find("dateien");
      if (dateien == nullptr)
        continue;
      for (const json::Value& d : dateien->items()) {
        const std::string key = d.member_str("hardlink_auf").empty()
                                    ? d.member_str("pfad")
                                    : d.member_str("hardlink_auf");
        const double bytes = d.member_double("bytes", 0.0);
        if (bytes != 0.0 && seen.insert(key).second)
          summe += bytes;
      }
    }
    std::string titel;
    if (const json::Value* block = katalog.find("bausteine")->find(b))
      titel = block->member_str("titel");
    z.push_back("- **" + b + "** (" + titel + "): " + std::to_string(ms.size()) +
                " Modelle, " + Gb(summe));
    for (const std::string& mid : ms) {
      const json::Value& m = *FindModel(alle, mid)->json;
      const json::Value* vorgabe = m.find("vorgabe_ram");
      const std::string art =
          Truthy(m.find("pflicht")) ? "Pflicht"
          : (vorgabe != nullptr && vorgabe->is_number() &&
             vorgabe->as_double() == ram)
              ? "Vorgabe"
              : "gewaehlt";
      std::string lizenz;
      if (const json::Value* l = m.find("lizenz"))
        lizenz = l->member_str("name");
      z.push_back("  - " + mid + " (" + art + ", " +
                  Gb(m.member_double("bytes", 0.0)) + ", Worker " +
                  m.member_str("worker") + ", passt in 64 GB: " +
                  Member(m, "passt_64gb") + ") – " + lizenz);
    }
  }
  if (!pm.empty())
    z.push_back("- **zweiter-rechner**: " + std::to_string(pm.size()) +
                " [remote/...]-Zwillinge auf " +
                Member(*a.find("partner"), "url") + ": " + JoinStr(pm, ", "));
  for (const std::string& h : hinweise)
    z.push_back("- Hinweis: " + h);
  z.emplace_back();
  z.push_back("## Dateien im Zielbaum");
  z.emplace_back();
  z.push_back("- vorhanden: " + Gb(static_cast<double>(pr.bytes_da)) +
              "; fehlend oder falsche Groesse: " +
              Gb(static_cast<double>(pr.bytes_fehlend)) +
              " (nur geprueft, nicht geladen)");
  for (const FileCheck::Missing& f : pr.fehlend) {
    std::string q = "Quelle nicht gepinnt";
    if (f.quellen != nullptr && f.quellen->is_array()) {
      const json::Value* passend = nullptr;
      for (const json::Value& x : f.quellen->items()) {
        const std::string ordner = Lower(x.member_str("ordner")) + "\\";
        if (Lower(f.pfad).starts_with(ordner) && !x.member_str("hf_repo").empty()) {
          passend = &x;
          break;
        }
      }
      if (passend == nullptr)
        for (const json::Value& x : f.quellen->items())
          if (!x.member_str("hf_repo").empty()) {
            passend = &x;
            break;
          }
      if (passend != nullptr) {
        const std::string rev = passend->member_str("hf_revision");
        q = passend->member_str("hf_repo") + "@" +
            (rev.empty() ? "main (nicht gepinnt)" : rev);
      } else if (!f.quellen->items().empty()) {
        const json::Value& x = f.quellen->items().front();
        const std::string hinweis = x.member_str("hinweis");
        q += ": " + (hinweis.empty() ? x.member_str("angabe_von") : hinweis);
      }
    }
    z.push_back("  - FEHLT " + f.pfad + " (" + Gb(static_cast<double>(f.bytes)) +
                "; " + f.modell + "; " + q + ")");
  }
  for (const FileCheck::WrongSize& f : pr.falsch)
    z.push_back("  - GROESSE " + f.pfad + ": erwartet " + std::to_string(f.soll) +
                " Bytes, vorhanden " + std::to_string(f.ist) + " (" + f.modell +
                ")");
  for (const auto& [mid, dd] : pr.daten_fehlend)
    z.push_back("  - DATEN fehlt " + Member(pfade, "daten") + "\\" + dd + " (" +
                mid + "; Datendatei, von einem eingerichteten Rechner kopieren "
                "oder Zeile entfernen)");
  if (!worker_fehlend.empty()) {
    z.emplace_back();
    z.push_back("## Programme");
    z.emplace_back();
    for (const auto& [k, ms] : worker_fehlend) {
      const std::string pfad = Member(pfade, k);
      z.push_back("- pfade." + k + ": gufo.exe nicht gefunden (" +
                  (pfad.empty() ? "leer" : pfad) + ") – gebraucht von " +
                  JoinStr(ms, ", "));
    }
  }
  z.emplace_back();
  z.push_back("## Lizenzen, die bestaetigt werden muessen");
  z.emplace_back();
  for (const Licence& l : lizenzen)
    z.push_back(std::string("- [") +
                (Confirmed(a, l.modell) ? "bestaetigt" : "OFFEN") + "] " +
                l.modell + ": " + l.lizenz->member_str("name") + " – " +
                l.lizenz->member_str("satz"));
  if (lizenzen.empty())
    z.push_back("- keine");
  z.emplace_back();
  z.push_back("## Ergebnis");
  z.emplace_back();
  if (geschrieben.empty())
    z.push_back("- nur Plan, nichts geschrieben");
  for (const std::string& g : geschrieben)
    z.push_back("- " + g);
  return Join(z, '\n') + "\n";
}

}  // namespace

// ---------------------------------------------------------------------------
// Downloads (plan section 4): the file list of the chosen models with their
// Hugging Face address and the state on disk.

std::string DownloadUrl(std::string_view base, std::string_view hf_repo,
                        std::string_view revision, std::string_view hf_pfad) {
  std::string url(base);
  while (!url.empty() && url.back() == '/')
    url.pop_back();
  url += '/';
  url += hf_repo;
  url += "/resolve/";
  url += revision.empty() ? std::string("main") : std::string(revision);
  url += '/';
  static constexpr char kHex[] = "0123456789ABCDEF";
  for (const char c : hf_pfad) {
    const auto u = static_cast<unsigned char>(c);
    if (c == '\\' || c == '/') {
      url += '/';
    } else if (std::isalnum(u) || c == '-' || c == '.' || c == '_' ||
               c == '~') {
      url += c;
    } else {
      url += '%';
      url += kHex[u >> 4];
      url += kHex[u & 15];
    }
  }
  return url;
}

std::vector<DownloadFile> PlanDownloads(const json::Value& katalog,
                                        const std::vector<std::string>& modelle,
                                        std::string_view ziel,
                                        std::string_view hf_base_url,
                                        std::vector<std::string>* hinweise) {
  std::vector<DownloadFile> out;
  const auto alle = AllModels(katalog);
  std::set<std::string> seen;
  std::error_code ec;
  for (const std::string& mid : modelle) {
    const Model* model = FindModel(alle, mid);
    if (model == nullptr) {
      hinweise->push_back("unbekanntes Modell '" + mid + "' uebersprungen");
      continue;
    }
    const json::Value* quellen = model->json->find("quellen");
    const json::Value* dateien = model->json->find("dateien");
    if (dateien == nullptr || !dateien->is_array())
      continue;
    for (const json::Value& d : dateien->items()) {
      const std::string pfad = d.member_str("pfad");
      if (pfad.empty() || !seen.insert(Lower(pfad)).second)
        continue;
      DownloadFile f;
      f.pfad = pfad;
      f.modell = mid;
      f.bytes = static_cast<long long>(d.member_double("bytes", 0.0));
      const std::string sha = Lower(d.member_str("sha256"));
      if (sha.size() == 64)
        f.sha256 = sha;
      f.ziel = NativePath(JoinPath(std::string(ziel), pfad));

      // The source: the file's own hf_repo/hf_pfad when the catalogue has
      // them (null = not on Hugging Face), else the model's source whose
      // folder is the path's prefix (einrichten.py line 274), else any
      // source of the model with a repo.
      const json::Value* file_repo = d.find("hf_repo");
      const bool none = file_repo != nullptr && file_repo->is_null();
      std::string repo = file_repo != nullptr && file_repo->is_string()
                             ? file_repo->str()
                             : std::string();
      std::string hf_pfad = d.member_str("hf_pfad");
      std::string revision;
      std::string ordner;
      if (!none && quellen != nullptr && quellen->is_array()) {
        const json::Value* match = nullptr;
        for (const json::Value& q : quellen->items()) {
          const std::string folder = q.member_str("ordner");
          if (folder.empty() ||
              !Lower(pfad).starts_with(Lower(folder) + "\\"))
            continue;
          if (q.member_str("hf_repo").empty())
            continue;
          if (!repo.empty() && Lower(q.member_str("hf_repo")) != Lower(repo))
            continue;
          match = &q;
          break;
        }
        if (match == nullptr) {
          for (const json::Value& q : quellen->items()) {
            if (q.member_str("hf_repo").empty())
              continue;
            if (!repo.empty() &&
                Lower(q.member_str("hf_repo")) != Lower(repo))
              continue;
            match = &q;
            break;
          }
        }
        if (match != nullptr) {
          if (repo.empty())
            repo = match->member_str("hf_repo");
          revision = match->member_str("hf_revision");
          ordner = match->member_str("ordner");
        }
      }
      if (const json::Value* file_rev = d.find("hf_revision");
          file_rev != nullptr && file_rev->is_string() &&
          !file_rev->str().empty())
        revision = file_rev->str();
      if (hf_pfad.empty()) {
        hf_pfad = !ordner.empty() &&
                          Lower(pfad).starts_with(Lower(ordner) + "\\")
                      ? pfad.substr(ordner.size() + 1)
                      : pfad;
      }
      if (none || repo.empty()) {
        f.status = DownloadFile::Status::kFehler;
        f.fehler_code = "no_source";
        f.fehler = "keine Download-Quelle im Katalog (von einem "
                   "eingerichteten Rechner kopieren)";
      } else {
        f.hf_repo = repo;
        f.revision = revision;
        f.url = DownloadUrl(hf_base_url, repo, revision, hf_pfad);
      }
      if (std::filesystem::is_regular_file(f.ziel, ec)) {
        const auto size =
            static_cast<long long>(std::filesystem::file_size(f.ziel, ec));
        if (f.bytes <= 0 || size == f.bytes) {
          f.status = DownloadFile::Status::kVorhanden;
          f.geladen = size;
          f.fehler.clear();
          f.fehler_code.clear();
        }
      }
      out.push_back(std::move(f));
    }
  }
  return out;
}

bool BuildPlan(const json::Value& antworten, const json::Value& katalog,
               const std::filesystem::path& bausteine_dir, Plan* out,
               std::string* error) {
  *out = Plan{};
  out->antworten = antworten;
  if (!out->antworten.is_object()) {
    *error = "antworten.json: kein Objekt";
    return false;
  }
  json::Value& a = out->antworten;
  if (!CheckAnswers(&a, error))
    return false;
  const auto alle = AllModels(katalog);
  if (alle.empty()) {
    *error = "katalog.json: keine Bausteine";
    return false;
  }
  if (!ChooseModels(a, alle, &out->modelle, &out->hinweise, error))
    return false;
  out->werte = ValuesFor(a, out->modelle, alle);
  const std::set<std::string> flags = FlagsFor(a);
  const FileCheck pr = CheckFiles(a, out->modelle, alle);
  out->worker_fehlend = CheckWorkers(a, out->modelle, alle);
  out->partner_modelle = PartnerModels(a, out->modelle, alle);
  if (!BuildPreset(out->modelle, alle, out->werte, flags, bausteine_dir,
                   out->partner_modelle, &out->preset_text, error))
    return false;
  const std::vector<Licence> lizenzen = Licences(out->modelle, alle);
  for (const Licence& l : lizenzen)
    if (!Confirmed(a, l.modell) && !Contains(out->lizenzen_offen, l.modell))
      out->lizenzen_offen.push_back(l.modell);
  for (const FileCheck::Missing& f : pr.fehlend)
    out->dateien_fehlend.push_back(f.pfad);
  for (const FileCheck::WrongSize& f : pr.falsch)
    out->dateien_fehlend.push_back(f.pfad);
  for (const auto& [mid, dd] : pr.daten_fehlend)
    out->daten_fehlend.push_back(dd);
  out->bytes_fehlend = pr.bytes_fehlend;
  out->bytes_da = pr.bytes_da;
  out->start_cmd = StartCmd(out->werte, out->preset_text);
  out->bericht = Report(a, katalog, out->modelle, alle, out->hinweise, pr,
                        out->worker_fehlend, out->partner_modelle, lizenzen, {});
  return true;
}

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

SetupService::~SetupService() {
  // The self-test first (it stops its child router and joins), then the
  // download workers while the state-file lock and target are still alive.
  selbsttest_.reset();
  downloader_.reset();
}

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

void SetupService::BindUsers(const std::filesystem::path& preset) {
  if (!options_.users_unbound || options_.gate == nullptr)
    return;
  const std::filesystem::path file =
      preset.parent_path() / "nutzer" / "nutzer.json";
  std::string error;
  if (!options_.gate->store().Relocate(file, &error)) {
    Logger::Warn("router", "event=setup_users_file file=" + file.string() +
                               " error=\"" + error + "\"");
    return;
  }
  Logger::Info("router", "event=setup_users_file file=" + file.string() +
                             " admin=" +
                             (options_.gate->store().HasAdmin() ? "yes" : "no"));
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
  std::string preset = Member(*pfade, "preset");
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
    if (std::string_view(key) == "daten" && preset.empty())
      preset = JoinPath(pfad, "gufo-router.ini");
  }
  if (!preset.empty() && IsSafePath(preset)) {
    const std::lock_guard<std::mutex> lock(mutex_);
    preset_path_ = NativePath(preset);
    BindUsers(preset_path_);
  }
  json::Value out = json::Value::object();
  out["ordner"] = std::move(ordner);
  return JsonOk(out);
}

HttpResponse SetupService::Preset(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", "invalid_request",
                 std::string("bad JSON: ") + e.what());
  }
  const json::Value* antworten = body.find("antworten");
  if (antworten == nullptr || !antworten->is_object())
    return Error(400, "Bad Request", "invalid_request", "antworten fehlt");
  const bool schreiben = Truthy(body.find("schreiben"));
  json::Value katalog;
  std::string error;
  if (!LoadKatalog(&katalog, &error))
    return Error(404, "Not Found", "katalog_missing", error);
  Plan plan;
  if (!BuildPlan(*antworten, katalog, options_.dir / "bausteine", &plan, &error))
    return Error(400, "Bad Request", "invalid_request", error);

  const std::string preset_text_only = plan.preset_text;
  const std::filesystem::path preset_path = NativePath(plan.werte.at("preset"));
  const std::string daten = plan.werte.at("daten");
  const std::string modelle = plan.werte.at("modelle");
  for (const auto& [name, pfad] :
       std::vector<std::pair<std::string, std::string>>{
           {"pfade.preset", plan.werte.at("preset")},
           {"pfade.daten", daten},
           {"pfade.modelle", modelle}}) {
    if (!IsSafePath(pfad))
      return Error(400, "Bad Request", "invalid_path",
                   name + ": absoluter Pfad mit Laufwerk, ohne '..' und ohne "
                          "UNC (" + pfad + ")");
  }
  // The text that would land on disk: the blocks plus what the previous
  // preset held outside them.
  std::string text = plan.preset_text;
  std::string old;
  std::error_code ec;
  const bool exists = std::filesystem::is_regular_file(preset_path, ec);
  if (exists && ReadText(preset_path, &old)) {
    const std::vector<std::string> fremd = ForeignBlocks(old);
    if (!fremd.empty())
      text += std::string("\n") + kKeptMarker + "\n" + Join(fremd, '\n') + "\n";
  }
  // Parse probe: the router must accept the file before it is written.
  if (options_.check_preset) {
    const std::filesystem::path probe =
        std::filesystem::temp_directory_path() /
        ("gufo-einrichtung-" + auth::RandomHex(4) + ".ini");
    if (!WriteText(probe, text, &error))
      return Error(500, "Internal Server Error", "write_failed", error);
    const std::string message = options_.check_preset(probe);
    std::filesystem::remove(probe, ec);
    if (!message.empty())
      return Error(422, "Unprocessable Entity", "preset_invalid", message);
  }

  json::Value out = json::Value::object();
  out["preset_text"] = text;
  out["start_cmd"] = plan.start_cmd;
  out["lizenzen_offen"] = ListOf(plan.lizenzen_offen);
  out["dateien_fehlend"] = ListOf(plan.dateien_fehlend);
  out["bytes_fehlend"] = plan.bytes_fehlend;
  out["daten_fehlend"] = ListOf(plan.daten_fehlend);
  out["modelle"] = ListOf(plan.modelle);
  out["partner_modelle"] = ListOf(plan.partner_modelle);
  out["hinweise"] = ListOf(plan.hinweise);
  json::Value worker = json::Value::object();
  for (const auto& [key, ms] : plan.worker_fehlend)
    worker[key] = ListOf(ms);
  out["worker_fehlend"] = std::move(worker);
  out["preset_datei"] = preset_path.string();
  std::vector<std::string> geschrieben;
  if (!schreiben) {
    out["bericht"] = plan.bericht;
    out["geschrieben"] = json::Value::array();
    return JsonOk(out);
  }
  if (!plan.lizenzen_offen.empty()) {
    json::Value refusal = ErrorBody(
        "licenses_open",
        "Lizenzen nicht bestaetigt (lizenzen_bestaetigt): " +
            JoinStr(plan.lizenzen_offen, ", "));
    refusal["lizenzen_offen"] = ListOf(plan.lizenzen_offen);
    return Json(409, "Conflict", refusal);
  }
  const std::filesystem::path ziel = preset_path.parent_path();
  std::filesystem::create_directories(ziel, ec);
  if (exists) {
    const std::filesystem::path backup =
        ziel / (preset_path.filename().string() + ".bak-" +
                Now("%Y-%m-%d_%H-%M-%S"));
    std::filesystem::rename(preset_path, backup, ec);
    if (ec)
      return Error(500, "Internal Server Error", "write_failed",
                   "cannot back up " + preset_path.string() + ": " +
                       ec.message());
    geschrieben.push_back("Sicherung: " + backup.string());
  }
  if (!WriteText(preset_path, text, &error))
    return Error(500, "Internal Server Error", "write_failed", error);
  geschrieben.push_back("Preset: " + preset_path.string());
  const std::filesystem::path cmd = ziel / "start-router.cmd";
  if (!WriteText(cmd, plan.start_cmd, &error))
    return Error(500, "Internal Server Error", "write_failed", error);
  geschrieben.push_back("Start: " + cmd.string());
  for (const char* folder : kDataFolders)
    std::filesystem::create_directories(NativePath(JoinPath(daten, folder)), ec);
  geschrieben.push_back("Datenordner unter " + daten +
                        " angelegt (jobs, prompt-cache*, video-jobs*, voices, "
                        "expert-profiles)");
  {
    json::Value ein = json::Value::object();
    ein["einrichtung_version"] = kVersion;
    ein["katalog_version"] = katalog.member_str("katalog_version");
    ein["zeit"] = Now("%Y-%m-%dT%H:%M:%S");
    ein["gufo_version"] = "";
    ein["antworten"] = plan.antworten;
    ein["modelle"] = ListOf(plan.modelle);
    ein["partner_modelle"] = ListOf(plan.partner_modelle);
    ein["dateien_fehlend"] = ListOf(plan.dateien_fehlend);
    ein["bytes_fehlend"] = plan.bytes_fehlend;
    const std::filesystem::path file = ziel / "einrichtung.json";
    if (!WriteText(file, ein.tojson() + "\n", &error))
      return Error(500, "Internal Server Error", "write_failed", error);
    geschrieben.push_back("Stand: " + file.string());
  }
  // Autostart and the firewall rule need an elevated process (schtasks,
  // netsh); they come with gufo-einrichten.cmd (C5). Said so, not done.
  if (Truthy(antworten->find("autostart"))) {
    out["autostart"] = "uebersprungen";
    out["autostart_hinweis"] =
        "Geplante Aufgabe und Firewall-Regel legt gufo-einrichten.cmd mit "
        "Admin-Rechten an (noch nicht in dieser Version); bis dahin "
        "start-router.cmd von Hand starten.";
  }
  // The report names what was written; rebuild it with the list.
  std::string bericht = plan.bericht;
  if (const auto at = bericht.find("- nur Plan, nichts geschrieben\n");
      at != std::string::npos) {
    std::string lines;
    for (const std::string& g : geschrieben)
      lines += "- " + g + "\n";
    bericht.replace(at, std::strlen("- nur Plan, nichts geschrieben\n"), lines);
  }
  if (WriteText(ziel / "einrichtung-bericht.md", bericht, &error))
    geschrieben.push_back("Bericht: " + (ziel / "einrichtung-bericht.md").string());
  out["bericht"] = bericht;
  out["geschrieben"] = ListOf(geschrieben);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    preset_path_ = preset_path;
    BindUsers(preset_path);
  }
  Logger::Info("router", "event=setup_preset_written file=" +
                             preset_path.string() + " models=" +
                             std::to_string(plan.modelle.size()));
  return JsonOk(out);
}

HttpResponse SetupService::Admin(const HttpRequest& req) {
  // Only the machine itself, and only while no admin exists: the one
  // proof of "I sit at this computer" before there is anyone to log in.
  if (!auth::IsLoopback(req.client_id))
    return NotFound();
  if (options_.gate == nullptr)
    return Error(409, "Conflict", "no_data_dir", "no users file");
  if (options_.users_unbound) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (preset_path_.empty())
      return Error(409, "Conflict", "no_data_dir",
                   "erst Schritt 3 (POST /router/setup/pruefen mit pfade) "
                   "nennt den Datenordner mit der Nutzerdatei");
  }
  if (options_.gate->store().HasAdmin())
    return Error(409, "Conflict", "admin_exists", "admin exists");
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", "invalid_request",
                 std::string("bad JSON: ") + e.what());
  }
  auth::User user;
  user.name = Lower(Trim(body.member_str("name")));
  user.rolle = "admin";
  if (!auth::IsValidName(user.name))
    return auth::AuthError(400, "Bad Request",
                           "bad user name (1-32 of a-z 0-9 . _ -)",
                           "invalid_name");
  if (auth::IsReservedName(user.name))
    return auth::AuthError(400, "Bad Request",
                           "the name is reserved (lokal = the router itself, "
                           "partner = the partner router)",
                           "name_reserved");
  const std::string password = body.member_str("passwort");
  if (password.size() < 4)
    return auth::AuthError(400, "Bad Request", "passwort: at least 4 characters",
                           "invalid_password");
  auth::UserStore& store = options_.gate->store();
  if (store.Find(user.name).has_value())
    return auth::AuthError(409, "Conflict", "user exists", "user_exists");
  user.erstellt = auth::NowUnix();
  std::string error;
  if (!store.Put(user, &error))
    return auth::AuthError(500, "Internal Server Error", error, "users_file");
  if (!store.SetPassword(user.name, password, &error))
    return auth::AuthError(500, "Internal Server Error", error, "users_file");
  Logger::Info("router", "event=setup_admin name=" + user.name +
                             " file=" + store.path().string());
  json::Value out = json::Value::object();
  out["name"] = user.name;
  out["rolle"] = "admin";
  out["datei"] = store.path().string();
  return Json(201, "Created", out);
}

// ---------------------------------------------------------------------------
// Downloads

namespace {

constexpr const char* kDownloadState = ".einrichtung-downloads.json";

json::Value EmptyDownloads(int parallel) {
  json::Value out = json::Value::object();
  out["laeuft"] = false;
  out["pause"] = false;
  out["ziel"] = "";
  out["parallel"] = static_cast<double>(parallel);
  out["rate_bytes_s"] = 0.0;
  out["gesamt_bytes"] = 0.0;
  out["geladen_bytes"] = 0.0;
  out["hinweis"] = "";
  out["dateien"] = json::Value::array();
  return out;
}

}  // namespace

void SetupService::WriteDownloadState(const Downloader& job) {
  // Called by the download workers (outside their lock) and by the
  // handlers; never takes mutex_ (a handler may hold it while it joins)
  // and never reads downloader_ (a handler may be replacing it).
  const std::lock_guard<std::mutex> lock(download_state_mutex_);
  if (download_ziel_.empty())
    return;
  json::Value state = job.Status();
  // Finished files of earlier jobs under the same target stay in the file
  // (the next `planen` takes their digests), the current job's entries
  // replace theirs.
  {
    std::string text;
    if (ReadText(download_ziel_ / kDownloadState, &text)) {
      try {
        const json::Value old = json::parse(text);
        const json::Value* list = old.find("dateien");
        json::Value* now = &state["dateien"];
        std::set<std::string> current;
        for (const json::Value& d : now->items())
          current.insert(Lower(d.member_str("pfad")));
        if (list != nullptr && list->is_array()) {
          for (const json::Value& d : list->items()) {
            const std::string st = d.member_str("status");
            if ((st == "fertig" || st == "vorhanden") &&
                !current.contains(Lower(d.member_str("pfad"))))
              now->push_back(d);
          }
        }
      } catch (const std::exception&) {
      }
    }
  }
  state["ziel"] = download_ziel_.string();
  state["zeit"] = Now("%Y-%m-%dT%H:%M:%S");
  state["einrichtung_version"] = kVersion;
  std::string error;
  std::error_code ec;
  std::filesystem::create_directories(download_ziel_, ec);
  if (!WriteText(download_ziel_ / kDownloadState, state.dump() + "\n", &error))
    Logger::Warn("router", "event=setup_download_state error=\"" + error + "\"");
}

HttpResponse SetupService::Downloads() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (downloader_ == nullptr)
    return JsonOk(EmptyDownloads(options_.download_parallel));
  json::Value out = downloader_->Status();
  out["ziel"] = download_ziel_.string();
  return JsonOk(out);
}

HttpResponse SetupService::DownloadsPost(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", "invalid_request",
                 std::string("bad JSON: ") + e.what());
  }
  std::string aktion = body.member_str("aktion", "planen");
  if (aktion == "fortsetzen")
    aktion = "weiter";
  // The token stays in this request and the downloader's memory; it is
  // never written to the state file, the report or the log.
  const std::string token = body.member_str("hf_token");
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto status = [this](int code, const char* reason) {
    json::Value out = downloader_->Status();
    out["ziel"] = download_ziel_.string();
    return Json(code, reason, out);
  };
  if (aktion == "planen" || aktion == "start") {
    const std::vector<std::string> modelle = StringList(body.find("modelle"));
    const std::string ziel = Member(body, "ziel");
    if (ziel.empty() || !IsSafePath(ziel))
      return Error(400, "Bad Request", "invalid_path",
                   "ziel: absoluter Pfad mit Laufwerk, ohne '..' und ohne "
                   "UNC (" + ziel + ")");
    if (modelle.empty())
      return Error(400, "Bad Request", "invalid_request", "modelle fehlt");
    if (downloader_ != nullptr &&
        (downloader_->running() || downloader_->paused())) {
      if (aktion == "planen")
        return status(200, "OK");
      return Error(409, "Conflict", "download_running",
                   "ein Download laeuft schon (pause / weiter / abbrechen)");
    }
    json::Value katalog;
    std::string error;
    if (!LoadKatalog(&katalog, &error))
      return Error(404, "Not Found", "katalog_missing", error);
    std::vector<std::string> hinweise;
    std::vector<DownloadFile> files = PlanDownloads(
        katalog, modelle, ziel, options_.hf_base_url, &hinweise);
    // Digests of files loaded earlier (state file of the last job).
    const std::filesystem::path ziel_dir = NativePath(ziel);
    {
      std::string text;
      if (ReadText(ziel_dir / kDownloadState, &text)) {
        try {
          const json::Value old = json::parse(text);
          if (const json::Value* list = old.find("dateien");
              list != nullptr && list->is_array()) {
            for (const json::Value& d : list->items()) {
              const std::string st = d.member_str("status");
              if (st != "fertig" && st != "vorhanden")
                continue;
              for (DownloadFile& f : files) {
                if (f.status != DownloadFile::Status::kVorhanden ||
                    Lower(f.pfad) != Lower(d.member_str("pfad")) ||
                    static_cast<long long>(d.member_double("bytes", -1.0)) !=
                        f.bytes)
                  continue;
                if (d.member_str("sha256_status") == "ok" ||
                    d.member_str("sha256_status") == "unbekannt") {
                  f.sha256_status = d.member_str("sha256_status");
                  f.sha256_quelle = d.member_str("sha256_quelle");
                  f.sha256_ist = d.member_str("sha256_ist");
                }
              }
            }
          }
        } catch (const std::exception&) {
        }
      }
    }
    // The old job (finished or cancelled) goes first, so its last
    // on_change cannot overwrite the new plan's state file.
    downloader_.reset();
    {
      const std::lock_guard<std::mutex> state_lock(download_state_mutex_);
      download_ziel_ = ziel_dir;
    }
    DownloaderOptions o;
    o.parallel = options_.download_parallel;
    o.retry_wait_ms = options_.download_retry_wait_ms;
    o.allow_http = options_.download_allow_http;
    o.on_change = [this](const Downloader& job) { WriteDownloadState(job); };
    downloader_ = std::make_unique<Downloader>(std::move(o), std::move(files));
    if (aktion == "planen") {
      json::Value out = downloader_->Status();
      out["ziel"] = download_ziel_.string();
      out["hinweise"] = ListOf(hinweise);
      return JsonOk(out);
    }
    const long long open = downloader_->BytesOpen();
    std::error_code ec;
    std::filesystem::create_directories(ziel_dir, ec);
    const auto space = std::filesystem::space(ziel_dir, ec);
    const long long frei = ec ? -1 : static_cast<long long>(space.available);
    if (frei >= 0 && open + options_.download_reserve_bytes > frei) {
      json::Value out =
          ErrorBody("disk_full",
                    "zu wenig Platz auf " + ziel + ": " +
                        std::to_string(open) + " Bytes zu laden + " +
                        std::to_string(options_.download_reserve_bytes) +
                        " Bytes Luft, frei " + std::to_string(frei));
      out["fehlt_bytes"] = static_cast<double>(
          open + options_.download_reserve_bytes - frei);
      out["offen_bytes"] = static_cast<double>(open);
      out["frei_bytes"] = static_cast<double>(frei);
      out["ziel"] = ziel;
      return Json(507, "Insufficient Storage", out);
    }
    const bool started = downloader_->Start(token);
    Logger::Info("router", "event=setup_download_start ziel=" + ziel +
                               " dateien=" +
                               std::to_string(downloader_->Files().size()) +
                               " offen_bytes=" + std::to_string(open) +
                               " token=" + (token.empty() ? "no" : "yes"));
    WriteDownloadState(*downloader_);
    json::Value out = downloader_->Status();
    out["ziel"] = download_ziel_.string();
    out["hinweise"] = ListOf(hinweise);
    return Json(started ? 202 : 200, started ? "Accepted" : "OK", out);
  }
  if (aktion == "pause" || aktion == "weiter" || aktion == "abbrechen") {
    if (downloader_ == nullptr)
      return Error(409, "Conflict", "no_download",
                   "kein Download geplant (erst planen oder start)");
    if (aktion == "pause")
      downloader_->Pause();
    else if (aktion == "weiter")
      downloader_->Resume(token);
    else
      downloader_->Cancel();
    Logger::Info("router", "event=setup_download_" + aktion);
    WriteDownloadState(*downloader_);
    return status(200, "OK");
  }
  return Error(400, "Bad Request", "invalid_request",
               "aktion: planen | start | pause | weiter | abbrechen");
}

// ---------------------------------------------------------------------------
// Self-test (selbsttest.hpp)

HttpResponse SetupService::Selbsttest() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (selbsttest_ != nullptr)
    return JsonOk(selbsttest_->Status());
  if (preset_path_.empty())
    return Error(409, "Conflict", "no_preset",
                 "noch kein Preset geschrieben (Schritt 5); der Selbsttest "
                 "braucht eins");
  json::Value out = json::Value::object();
  out["laeuft"] = false;
  out["fertig"] = false;
  out["preset"] = preset_path_.string();
  out["tests"] = json::Value::array();
  return JsonOk(out);
}

HttpResponse SetupService::SelbsttestPost(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Error(400, "Bad Request", "invalid_request",
                 std::string("bad JSON: ") + e.what());
  }
  const std::string aktion = body.member_str("aktion", "start");
  const std::lock_guard<std::mutex> lock(mutex_);
  if (aktion == "abbrechen") {
    if (selbsttest_ == nullptr)
      return Error(409, "Conflict", "no_selbsttest",
                   "kein Selbsttest gestartet");
    selbsttest_->Cancel();
    Logger::Info("router", "event=selbsttest_cancel");
    return JsonOk(selbsttest_->Status());
  }
  if (aktion != "start")
    return Error(400, "Bad Request", "invalid_request",
                 "aktion: start | abbrechen");
  // The preset: named in the request (the page remembers step 5's file
  // across a restart of this router) or the one step 5 wrote.
  std::filesystem::path preset = preset_path_;
  if (const std::string named = Member(body, "preset"); !named.empty()) {
    if (!IsSafePath(named))
      return Error(400, "Bad Request", "invalid_path",
                   "preset: absoluter Pfad mit Laufwerk, ohne '..' und ohne "
                   "UNC (" + named + ")");
    preset = NativePath(named);
  }
  std::error_code ec;
  if (preset.empty() || !std::filesystem::is_regular_file(preset, ec))
    return Error(409, "Conflict", "no_preset",
                 "noch kein Preset geschrieben (Schritt 5)" +
                     (preset.empty() ? std::string()
                                     : ": " + preset.string() + " fehlt"));
  if (selbsttest_ != nullptr && selbsttest_->running())
    return Error(409, "Conflict", "selbsttest_running",
                 "ein Selbsttest laeuft schon (abbrechen oder abwarten)");
  if (downloader_ != nullptr && downloader_->running())
    return Error(409, "Conflict", "download_running",
                 "erst den Download abwarten oder pausieren; der Selbsttest "
                 "startet den Router mit dem neuen Preset");
  // einrichtung.json next to the preset: the chosen models, the data
  // folder and the partner. A hand-written preset has none; then the
  // request names its cases in `tests`.
  json::Value ein;
  const json::Value* antworten = nullptr;
  {
    std::string text;
    if (ReadText(preset.parent_path() / "einrichtung.json", &text)) {
      try {
        ein = json::parse(text);
      } catch (const std::exception&) {
        ein = json::Value();
      }
      if (ein.is_object())
        antworten = ein.find("antworten");
    }
  }
  std::vector<std::string> bausteine = StringList(body.find("bausteine"));
  if (bausteine.empty() && antworten != nullptr)
    bausteine = StringList(antworten->find("bausteine"));
  const std::vector<std::string> modelle =
      ein.is_object() ? StringList(ein.find("modelle"))
                      : std::vector<std::string>{};
  std::string daten;
  std::string partner_url;
  if (antworten != nullptr) {
    if (const json::Value* pfade = antworten->find("pfade"))
      daten = pfade->member_str("daten");
    if (const json::Value* partner = antworten->find("partner"))
      partner_url = partner->member_str("url");
  }
  const bool alle = Truthy(body.find("alle_modelle"));
  std::vector<SelfTestCase> tests;
  if (const json::Value* list = body.find("tests");
      list != nullptr && list->is_array()) {
    for (const json::Value& t : list->items()) {
      SelfTestCase c;
      c.baustein = t.member_str("baustein");
      c.modell = t.member_str("modell");
      if (c.baustein.empty() || c.modell.empty())
        return Error(400, "Bad Request", "invalid_request",
                     "tests[]: baustein und modell fehlen");
      c.art = t.member_str("art", TestKind(c.baustein, c.modell));
      tests.push_back(std::move(c));
    }
  } else {
    json::Value katalog;
    std::string error;
    if (!LoadKatalog(&katalog, &error))
      return Error(404, "Not Found", "katalog_missing", error);
    tests = PlanSelfTest(katalog, bausteine, modelle, alle, partner_url);
  }
  if (tests.empty())
    return Error(400, "Bad Request", "invalid_request",
                 "keine Pruefpunkte: bausteine und modelle (einrichtung.json "
                 "neben dem Preset) oder tests[] fehlen");
  SelfTestOptions o;
  o.preset = preset;
  o.daten = daten.empty() ? preset.parent_path() : NativePath(daten);
  o.exe = options_.exe;
  o.base_url = options_.selbsttest_base_url;
  o.partner_url = partner_url;
  o.tests = std::move(tests);
  o.start_timeout_s = options_.selbsttest_start_timeout_s;
  o.test_timeout_s = options_.selbsttest_test_timeout_s;
  o.poll_ms = options_.selbsttest_poll_ms;
  const std::size_t count = o.tests.size();
  selbsttest_ = std::make_unique<SelfTest>(std::move(o));
  selbsttest_->Start();
  preset_path_ = preset;
  BindUsers(preset);
  Logger::Info("router", "event=selbsttest_start preset=" + preset.string() +
                             " tests=" + std::to_string(count) +
                             (options_.selbsttest_base_url.empty()
                                  ? std::string(" target=child")
                                  : " target=" +
                                        options_.selbsttest_base_url));
  return Json(202, "Accepted", selbsttest_->Status());
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
  if (rest == "preset")
    return post ? Preset(req)
                : Error(405, "Method Not Allowed", "method_not_allowed", "POST");
  if (rest == "admin")
    return post ? Admin(req)
                : Error(405, "Method Not Allowed", "method_not_allowed", "POST");
  if (rest == "downloads") {
    if (get)
      return Downloads();
    if (post)
      return DownloadsPost(req);
    return Error(405, "Method Not Allowed", "method_not_allowed", "GET or POST");
  }
  if (rest == "selbsttest") {
    if (get)
      return Selbsttest();
    if (post)
      return SelbsttestPost(req);
    return Error(405, "Method Not Allowed", "method_not_allowed", "GET or POST");
  }
  return NotFound();
}

}  // namespace gufo::router::einrichtung
