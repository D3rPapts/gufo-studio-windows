#include "src/cli/serve/router/auth.hpp"

#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
#include <system_error>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/strings.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::router::auth {
namespace {

using server::HttpRequest;
using server::HttpResponse;
using server::Logger;

bool Contains(const std::vector<std::string>& list, std::string_view item) {
  return std::find(list.begin(), list.end(), item) != list.end();
}

/// Civil date (proleptic Gregorian, UTC) of unix seconds; Howard
/// Hinnant's days-to-civil, so no locale or time zone library is needed.
struct Civil {
  int year, month, day, hour, minute, second;
};

Civil CivilOf(long long t) {
  long long days = t / 86400;
  long long rem = t % 86400;
  if (rem < 0) {
    rem += 86400;
    --days;
  }
  const long long z = days + 719468;
  const long long era = (z >= 0 ? z : z - 146096) / 146097;
  const long long doe = z - era * 146097;
  const long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const long long y = yoe + era * 400;
  const long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const long long mp = (5 * doy + 2) / 153;
  const long long d = doy - (153 * mp + 2) / 5 + 1;
  const long long m = mp < 10 ? mp + 3 : mp - 9;
  Civil out;
  out.year = static_cast<int>(m <= 2 ? y + 1 : y);
  out.month = static_cast<int>(m);
  out.day = static_cast<int>(d);
  out.hour = static_cast<int>(rem / 3600);
  out.minute = static_cast<int>(rem % 3600 / 60);
  out.second = static_cast<int>(rem % 60);
  return out;
}

std::string DayOf(long long t) {
  const Civil c = CivilOf(t);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", c.year, c.month, c.day);
  return buf;
}

std::string IsoOf(long long t) {
  const Civil c = CivilOf(t);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ", c.year,
                c.month, c.day, c.hour, c.minute, c.second);
  return buf;
}

long long NumberOf(const json::Value& object, const std::string& key,
                   long long def = 0) {
  const json::Value* v = object.find(key);
  return v != nullptr && v->is_number()
             ? static_cast<long long>(v->as_double())
             : def;
}

std::vector<std::string> StringsOf(const json::Value& object,
                                   const std::string& key) {
  std::vector<std::string> out;
  const json::Value* v = object.find(key);
  if (v == nullptr || !v->is_array())
    return out;
  for (const json::Value& item : v->items())
    if (item.is_string() && !item.str().empty())
      out.push_back(item.str());
  return out;
}

json::Value ArrayOf(const std::vector<std::string>& list) {
  json::Value out = json::Value::array();
  for (const std::string& item : list)
    out.push_back(item);
  return out;
}

std::string UrlEncode(std::string_view text) {
  std::string out;
  for (const unsigned char c : text) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' ||
        c == '/') {
      out += static_cast<char>(c);
    } else {
      char buf[4];
      std::snprintf(buf, sizeof(buf), "%%%02X", c);
      out += buf;
    }
  }
  return out;
}

std::string CookieValue(const HttpRequest& req, std::string_view name) {
  const std::string header = req.header("Cookie");
  std::size_t begin = 0;
  while (begin < header.size()) {
    const auto end = std::min(header.find(';', begin), header.size());
    const std::string_view pair = Trim(std::string_view(header).substr(
        begin, end - begin));
    const auto eq = pair.find('=');
    if (eq != std::string_view::npos && Trim(pair.substr(0, eq)) == name)
      return std::string(Trim(pair.substr(eq + 1)));
    begin = end + 1;
  }
  return "";
}

std::string BearerOf(const HttpRequest& req) {
  const std::string header = req.header("Authorization");
  constexpr std::string_view kBearer = "Bearer ";
  if (header.size() <= kBearer.size())
    return "";
  for (std::size_t i = 0; i < kBearer.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(header[i])) !=
        std::tolower(static_cast<unsigned char>(kBearer[i])))
      return "";
  }
  return std::string(Trim(std::string_view(header).substr(kBearer.size())));
}

std::string MultipartBoundary(const HttpRequest& req) {
  const std::string type = req.header("Content-Type");
  const std::string lower = Lower(type);
  if (lower.find("multipart/form-data") == std::string::npos)
    return "";
  const auto at = lower.find("boundary=");
  if (at == std::string::npos)
    return "";
  std::string_view value = std::string_view(type).substr(at + 9);
  if (const auto semi = value.find(';'); semi != std::string_view::npos)
    value = value.substr(0, semi);
  value = Trim(value);
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
    value = value.substr(1, value.size() - 2);
  return std::string(value);
}

void CollectStrings(const json::Value& value, bool all, int depth,
                    std::vector<std::string>* out) {
  if (depth > 32)
    return;
  if (value.is_object()) {
    for (const auto& [key, member] : value.members()) {
      const bool prompt_key = key == "prompt" || key == "negative_prompt" ||
                              key == "negativ_prompt" ||
                              key == "negativer_prompt";
      if (member.is_string()) {
        if (all || prompt_key)
          out->push_back(member.str());
      } else {
        CollectStrings(member, all, depth + 1, out);
      }
    }
  } else if (value.is_array()) {
    for (const json::Value& item : value.items()) {
      if (item.is_string()) {
        if (all)
          out->push_back(item.str());
      } else {
        CollectStrings(item, all, depth + 1, out);
      }
    }
  }
}

HttpResponse Json(int status, std::string_view reason, const json::Value& body,
                  std::vector<std::pair<std::string, std::string>> headers = {}) {
  headers.emplace_back("Content-Type", "application/json");
  headers.emplace_back("Cache-Control", "no-store");
  return HttpResponse{.status = status,
                      .reason = std::string(reason),
                      .body = body.dump(),
                      .headers = std::move(headers)};
}

HttpResponse JsonOk(const json::Value& body) { return Json(200, "OK", body); }

HttpResponse NotFound() {
  return AuthError(404, "Not Found", "no route for this path", "not_found");
}

std::string SessionCookie(std::string_view id, long long max_age) {
  std::string out = std::string(kSessionCookie) + "=" + std::string(id) +
                    "; Path=/; HttpOnly; SameSite=Lax; Max-Age=" +
                    std::to_string(max_age);
  return out;
}

std::string RequestAddress(const HttpRequest& req) {
  return req.client_id.empty() ? std::string("?") : req.client_id;
}

const char* kDefaultFilterWords =
    "# Wortsperre fuer Bild- und Video-Prompts (gufo 2.0.3): ein Wort je "
    "Zeile,\n"
    "# Gross-/Kleinschreibung egal, * am Ende = auch Wortformen (nackt* "
    "trifft nackte).\n"
    "# Gilt fuer Nutzer mit filter = an; admin ist frei. Zeilen mit # sind "
    "Kommentare.\n"
    "# Word filter for image and video prompts: one word per line, "
    "case-insensitive,\n"
    "# trailing * = any word form. Applies to users with filter = on.\n"
    "nackt*\n"
    "nude*\n"
    "naked\n"
    "nsfw\n"
    "porn*\n"
    "sex*\n"
    "erotik*\n"
    "erotic*\n"
    "hentai\n"
    "xxx\n"
    "blut*\n"
    "blood*\n"
    "gore\n"
    "gewalt*\n"
    "violen*\n"
    "waffe*\n"
    "weapon*\n"
    "toet*\n"
    "töt*\n"
    "kill*\n"
    "leiche*\n"
    "corpse*\n"
    "drogen\n"
    "drugs\n"
    "nazi*\n"
    "hakenkreuz\n"
    "swastika\n";

}  // namespace

// ---------------------------------------------------------------------------
// Small helpers

const char* ModeName(Mode mode) {
  switch (mode) {
    case Mode::kLan:
      return "lan";
    case Mode::kAll:
      return "all";
    default:
      return "off";
  }
}

bool ParseMode(std::string_view text, Mode* out) {
  const std::string value = Lower(Trim(text));
  if (value == "off")
    *out = Mode::kOff;
  else if (value == "lan")
    *out = Mode::kLan;
  else if (value == "all")
    *out = Mode::kAll;
  else
    return false;
  return true;
}

bool ParseLoopbackPolicy(std::string_view text, LoopbackPolicy* out) {
  const std::string value = Lower(Trim(text));
  if (value == "admin")
    *out = LoopbackPolicy::kAdmin;
  else if (value == "login")
    *out = LoopbackPolicy::kLogin;
  else
    return false;
  return true;
}

bool IsArea(std::string_view area) {
  for (const std::string_view known : kAreas)
    if (known == area)
      return true;
  return false;
}

bool IsRole(std::string_view role) {
  for (const char* known : kRoles)
    if (role == known)
      return true;
  return false;
}

bool IsValidName(std::string_view name) {
  if (name.empty() || name.size() > 32)
    return false;
  for (const unsigned char c : name) {
    if (!(std::islower(c) || std::isdigit(c) || c == '.' || c == '_' ||
          c == '-'))
      return false;
  }
  return true;
}

bool IsLoopback(std::string_view address) {
  return address == "::1" || address == "[::1]" ||
         address.starts_with("127.") || address.starts_with("::ffff:127.");
}

long long NowUnix() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string HexOf(std::string_view bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const unsigned char c : bytes) {
    out += kDigits[c >> 4];
    out += kDigits[c & 15];
  }
  return out;
}

std::string BytesOfHex(std::string_view hex) {
  std::string out;
  if (hex.size() % 2 != 0)
    return out;
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
      return c - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    const int hi = nibble(hex[i]);
    const int lo = nibble(hex[i + 1]);
    if (hi < 0 || lo < 0)
      return {};
    out += static_cast<char>(hi * 16 + lo);
  }
  return out;
}

std::string RandomHex(std::size_t bytes) {
  std::string raw(bytes, '\0');
  if (RAND_bytes(reinterpret_cast<unsigned char*>(raw.data()),
                 static_cast<int>(bytes)) != 1) {
    // OpenSSL's generator refused (never seen on Windows or Linux); the
    // caller must not get a predictable value.
    return {};
  }
  return HexOf(raw);
}

std::string Sha256Hex(std::string_view text) {
  return crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

std::string Pbkdf2Hex(std::string_view password, std::string_view salt,
                      long long rounds) {
  unsigned char out[32];
  if (rounds <= 0 || rounds > 10000000)
    return {};
  if (PKCS5_PBKDF2_HMAC(password.data(), static_cast<int>(password.size()),
                        reinterpret_cast<const unsigned char*>(salt.data()),
                        static_cast<int>(salt.size()),
                        static_cast<int>(rounds), EVP_sha256(), sizeof(out),
                        out) != 1)
    return {};
  return HexOf(std::string_view(reinterpret_cast<const char*>(out), sizeof(out)));
}

bool ConstantTimeEquals(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return false;
  unsigned char diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    diff |= static_cast<unsigned char>(a[i] ^ b[i]);
  return diff == 0;
}

HttpResponse AuthError(int status, std::string_view reason,
                       std::string_view message, std::string_view code) {
  json::Value detail = json::Value::object();
  detail["message"] = std::string(message);
  detail["type"] = status == 401   ? "authentication_error"
                   : status == 403 ? "permission_error"
                   : status == 429 ? "rate_limit_error"
                                   : "invalid_request_error";
  detail["code"] = std::string(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return Json(status, reason, body);
}

// ---------------------------------------------------------------------------
// Rights

bool Rights::AllowsArea(std::string_view area) const {
  if (area.empty() || bereiche.empty())
    return true;
  // "bild|video": either side is enough (previews of both kinds).
  std::size_t begin = 0;
  while (begin <= area.size()) {
    const auto end = std::min(area.find('|', begin), area.size());
    if (Contains(bereiche, area.substr(begin, end - begin)))
      return true;
    begin = end + 1;
  }
  return false;
}

bool Rights::AllowsModel(std::string_view model) const {
  return model.empty() || modelle.empty() || Contains(modelle, model);
}

bool Rights::FilterOn(std::string_view role) const {
  if (filter.has_value())
    return *filter;
  return role != "admin";
}

std::string Rights::Encode() const {
  std::string out = "bereiche=" + Join(bereiche, ',') +
                    ";modelle=" + Join(modelle, ',') + ";filter=";
  out += filter.has_value() ? (*filter ? "an" : "aus") : "rolle";
  out += ";tageslimit=" + std::to_string(tageslimit);
  out += std::string(";pool=") + (pool ? "an" : "aus");
  return out;
}

Rights Rights::Decode(std::string_view text) {
  Rights out;
  for (const std::string& pair : Split(text, ';')) {
    const auto eq = pair.find('=');
    if (eq == std::string::npos)
      continue;
    const std::string key = pair.substr(0, eq);
    const std::string value = pair.substr(eq + 1);
    if (key == "bereiche") {
      for (const std::string& area : Split(value, ','))
        if (IsArea(area))
          out.bereiche.push_back(area);
    } else if (key == "modelle") {
      out.modelle = Split(value, ',');
    } else if (key == "filter") {
      if (value == "an")
        out.filter = true;
      else if (value == "aus")
        out.filter = false;
    } else if (key == "tageslimit") {
      try {
        out.tageslimit = std::stoll(value);
      } catch (const std::exception&) {
        out.tageslimit = 0;
      }
    } else if (key == "pool") {
      out.pool = value != "aus";
    }
  }
  return out;
}

json::Value Rights::ToJson() const {
  json::Value out = json::Value::object();
  out["bereiche"] = ArrayOf(bereiche);
  out["modelle"] = ArrayOf(modelle);
  out["filter"] = filter.has_value() ? json::Value(*filter) : json::Value();
  out["tageslimit"] = tageslimit;
  out["pool"] = pool;
  return out;
}

Rights Rights::FromJson(const json::Value& value) {
  Rights out;
  if (!value.is_object())
    return out;
  for (const std::string& area : StringsOf(value, "bereiche"))
    if (IsArea(area))
      out.bereiche.push_back(area);
  out.modelle = StringsOf(value, "modelle");
  if (const json::Value* f = value.find("filter"); f != nullptr && f->is_bool())
    out.filter = f->as_bool();
  out.tageslimit = std::max<long long>(0, NumberOf(value, "tageslimit", 0));
  if (const json::Value* p = value.find("pool"); p != nullptr && p->is_bool())
    out.pool = p->as_bool();
  return out;
}

json::Value User::PublicJson() const {
  json::Value out = json::Value::object();
  out["name"] = name;
  out["rolle"] = rolle;
  out["rechte"] = rechte.ToJson();
  json::Value tokens = json::Value::array();
  for (const Token& t : token) {
    json::Value item = json::Value::object();
    item["name"] = t.name;
    item["erstellt"] = t.erstellt;
    item["ablauf"] = t.ablauf;
    item["zuletzt"] = t.zuletzt;
    tokens.push_back(std::move(item));
  }
  out["token"] = std::move(tokens);
  out["erstellt"] = erstellt;
  out["letzte_anmeldung"] = letzte_anmeldung;
  out["gesperrt"] = gesperrt;
  out["passwort"] = !hash_hex.empty();
  return out;
}

// ---------------------------------------------------------------------------
// UserStore

UserStore::UserStore(std::filesystem::path file, long long rounds)
    : file_(std::move(file)), rounds_(rounds) {}

bool UserStore::exists() const {
  std::error_code ec;
  return std::filesystem::is_regular_file(file_, ec);
}

bool UserStore::Load(std::string* error) {
  std::ifstream in(file_, std::ios::binary);
  if (!in) {
    if (!exists()) {
      const std::lock_guard<std::mutex> lock(mutex_);
      users_.clear();
      return true;
    }
    *error = "cannot read users file " + file_.string();
    return false;
  }
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  if (text.starts_with("\xEF\xBB\xBF"))
    text.erase(0, 3);
  json::Value doc;
  try {
    doc = json::parse(text);
  } catch (const std::exception& e) {
    *error = "users file " + file_.string() + " does not parse: " + e.what();
    return false;
  }
  return FromJson(doc, error);
}

bool UserStore::FromJson(const json::Value& doc, std::string* error) {
  if (!doc.is_object()) {
    *error = "users file: expected a JSON object";
    return false;
  }
  const json::Value* list = doc.find("nutzer");
  if (list == nullptr)
    list = doc.find("users");
  std::vector<User> users;
  if (list != nullptr && list->is_array()) {
    for (const json::Value& item : list->items()) {
      if (!item.is_object())
        continue;
      User user;
      user.name = Lower(item.member_str("name"));
      if (!IsValidName(user.name)) {
        *error = "users file: bad user name '" + user.name + "'";
        return false;
      }
      user.rolle = item.member_str("rolle", "nutzer");
      if (!IsRole(user.rolle))
        user.rolle = "nutzer";
      if (const json::Value* pw = item.find("passwort");
          pw != nullptr && pw->is_object()) {
        user.salt_hex = pw->member_str("salz");
        user.hash_hex = pw->member_str("hash");
        user.runden = NumberOf(*pw, "runden", kPasswordRounds);
      }
      if (const json::Value* r = item.find("rechte"); r != nullptr)
        user.rechte = Rights::FromJson(*r);
      if (const json::Value* tokens = item.find("token");
          tokens != nullptr && tokens->is_array()) {
        for (const json::Value& t : tokens->items()) {
          if (!t.is_object())
            continue;
          Token token;
          token.name = t.member_str("name");
          token.hash = t.member_str("hash");
          token.erstellt = NumberOf(t, "erstellt");
          token.ablauf = NumberOf(t, "ablauf");
          token.zuletzt = NumberOf(t, "zuletzt");
          if (!token.hash.empty())
            user.token.push_back(std::move(token));
        }
      }
      if (const json::Value* sessions = item.find("sitzungen");
          sessions != nullptr && sessions->is_array()) {
        for (const json::Value& s : sessions->items()) {
          if (!s.is_object())
            continue;
          RememberedSession session;
          session.hash = s.member_str("hash");
          session.ablauf = NumberOf(s, "ablauf");
          if (!session.hash.empty())
            user.sitzungen.push_back(std::move(session));
        }
      }
      user.erstellt = NumberOf(item, "erstellt");
      user.letzte_anmeldung = NumberOf(item, "letzte_anmeldung");
      if (const json::Value* g = item.find("gesperrt");
          g != nullptr && g->is_bool())
        user.gesperrt = g->as_bool();
      users.push_back(std::move(user));
    }
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  users_ = std::move(users);
  return true;
}

namespace {
json::Value DocumentOf(const std::vector<User>& users) {
  json::Value doc = json::Value::object();
  doc["version"] = 1;
  json::Value list = json::Value::array();
  for (const User& user : users) {
    json::Value item = json::Value::object();
    item["name"] = user.name;
    item["rolle"] = user.rolle;
    json::Value pw = json::Value::object();
    pw["algo"] = "pbkdf2-sha256";
    pw["runden"] = user.runden;
    pw["salz"] = user.salt_hex;
    pw["hash"] = user.hash_hex;
    item["passwort"] = std::move(pw);
    item["rechte"] = user.rechte.ToJson();
    json::Value tokens = json::Value::array();
    for (const Token& t : user.token) {
      json::Value tok = json::Value::object();
      tok["name"] = t.name;
      tok["hash"] = t.hash;
      tok["erstellt"] = t.erstellt;
      tok["ablauf"] = t.ablauf;
      tok["zuletzt"] = t.zuletzt;
      tokens.push_back(std::move(tok));
    }
    item["token"] = std::move(tokens);
    json::Value sessions = json::Value::array();
    for (const RememberedSession& s : user.sitzungen) {
      json::Value ses = json::Value::object();
      ses["hash"] = s.hash;
      ses["ablauf"] = s.ablauf;
      sessions.push_back(std::move(ses));
    }
    item["sitzungen"] = std::move(sessions);
    item["erstellt"] = user.erstellt;
    item["letzte_anmeldung"] = user.letzte_anmeldung;
    item["gesperrt"] = user.gesperrt;
    list.push_back(std::move(item));
  }
  doc["nutzer"] = std::move(list);
  return doc;
}
}  // namespace

json::Value UserStore::ToJson() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return DocumentOf(users_);
}

bool UserStore::SaveLocked(std::string* error) const {
  const json::Value doc = DocumentOf(users_);
  std::error_code ec;
  std::filesystem::create_directories(file_.parent_path(), ec);
  const std::filesystem::path tmp = file_.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      *error = "cannot write " + tmp.string();
      return false;
    }
    out << doc.tojson() << '\n';
    if (!out) {
      *error = "write failed: " + tmp.string();
      return false;
    }
  }
  std::filesystem::rename(tmp, file_, ec);
  if (ec) {
    *error = "cannot replace " + file_.string() + ": " + ec.message();
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

bool UserStore::Save(std::string* error) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return SaveLocked(error);
}

bool UserStore::empty() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return users_.empty();
}

bool UserStore::HasAdmin() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const User& user : users_)
    if (user.admin() && !user.gesperrt && !user.hash_hex.empty())
      return true;
  return false;
}

std::vector<User> UserStore::List() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return users_;
}

User* UserStore::FindLocked(std::string_view name) {
  for (User& user : users_)
    if (user.name == name)
      return &user;
  return nullptr;
}

const User* UserStore::FindLocked(std::string_view name) const {
  for (const User& user : users_)
    if (user.name == name)
      return &user;
  return nullptr;
}

std::optional<User> UserStore::Find(std::string_view name) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (const User* user = FindLocked(name))
    return *user;
  return std::nullopt;
}

void UserStore::SetPasswordLocked(User* user, std::string_view password) const {
  const std::string salt = BytesOfHex(RandomHex(16));
  user->salt_hex = HexOf(salt);
  user->runden = rounds_;
  user->hash_hex = Pbkdf2Hex(password, salt, rounds_);
}

bool UserStore::Put(User user, std::string* error) {
  user.name = Lower(user.name);
  if (!IsValidName(user.name)) {
    *error = "bad user name (1-32 of a-z 0-9 . _ -)";
    return false;
  }
  if (!IsRole(user.rolle)) {
    *error = "bad role (admin|nutzer|dienst)";
    return false;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  if (User* existing = FindLocked(user.name)) {
    *existing = std::move(user);
  } else {
    if (user.erstellt == 0)
      user.erstellt = NowUnix();
    users_.push_back(std::move(user));
  }
  return SaveLocked(error);
}

bool UserStore::Remove(std::string_view name, std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = std::find_if(users_.begin(), users_.end(),
                               [&](const User& u) { return u.name == name; });
  if (it == users_.end()) {
    *error = "unknown user";
    return false;
  }
  users_.erase(it);
  return SaveLocked(error);
}

bool UserStore::SetPassword(std::string_view name, std::string_view password,
                            std::string* error) {
  if (password.size() < 4) {
    *error = "password too short (at least 4 characters)";
    return false;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  User* user = FindLocked(name);
  if (user == nullptr) {
    *error = "unknown user";
    return false;
  }
  SetPasswordLocked(user, password);
  if (user->hash_hex.empty()) {
    *error = "password hashing failed";
    return false;
  }
  // A new password ends every remembered session.
  user->sitzungen.clear();
  return SaveLocked(error);
}

std::optional<User> UserStore::Verify(std::string_view name,
                                      std::string_view password) const {
  User copy;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const User* user = FindLocked(Lower(name));
    if (user == nullptr || user->gesperrt || user->hash_hex.empty())
      return std::nullopt;
    copy = *user;
  }
  // The slow hash runs without the lock.
  const std::string hash =
      Pbkdf2Hex(password, BytesOfHex(copy.salt_hex), copy.runden);
  if (hash.empty() || !ConstantTimeEquals(hash, copy.hash_hex))
    return std::nullopt;
  return copy;
}

std::string UserStore::CreateToken(std::string_view name,
                                   std::string_view token_name, long long ablauf,
                                   std::string* error) {
  if (!IsValidName(token_name)) {
    *error = "bad token name (1-32 of a-z 0-9 . _ -)";
    return {};
  }
  const std::string random = RandomHex(32);
  if (random.empty()) {
    *error = "random generator failed";
    return {};
  }
  const std::string plaintext = std::string(kTokenPrefix) + random;
  const std::lock_guard<std::mutex> lock(mutex_);
  User* user = FindLocked(name);
  if (user == nullptr) {
    *error = "unknown user";
    return {};
  }
  for (const Token& t : user->token) {
    if (t.name == token_name) {
      *error = "a token of that name exists; revoke it first";
      return {};
    }
  }
  Token token;
  token.name = std::string(token_name);
  token.hash = Sha256Hex(plaintext);
  token.erstellt = NowUnix();
  token.ablauf = ablauf;
  user->token.push_back(std::move(token));
  if (!SaveLocked(error))
    return {};
  return plaintext;
}

bool UserStore::RevokeToken(std::string_view name, std::string_view token_name,
                            std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  User* user = FindLocked(name);
  if (user == nullptr) {
    *error = "unknown user";
    return false;
  }
  const auto it =
      std::find_if(user->token.begin(), user->token.end(),
                   [&](const Token& t) { return t.name == token_name; });
  if (it == user->token.end()) {
    *error = "unknown token";
    return false;
  }
  user->token.erase(it);
  return SaveLocked(error);
}

std::optional<User> UserStore::FindByToken(std::string_view token) {
  if (!token.starts_with(kTokenPrefix))
    return std::nullopt;
  const std::string hash = Sha256Hex(token);
  const long long now = NowUnix();
  const std::lock_guard<std::mutex> lock(mutex_);
  for (User& user : users_) {
    for (Token& t : user.token) {
      if (!ConstantTimeEquals(t.hash, hash))
        continue;
      if (user.gesperrt || (t.ablauf > 0 && t.ablauf < now))
        return std::nullopt;
      // The last use is noted in memory only; the file is written on the
      // next change (a token check must never cost a disk write).
      t.zuletzt = now;
      return user;
    }
  }
  return std::nullopt;
}

bool UserStore::RememberSession(std::string_view name,
                                std::string_view session_id, long long ablauf,
                                std::string* error) {
  const long long now = NowUnix();
  const std::lock_guard<std::mutex> lock(mutex_);
  User* user = FindLocked(name);
  if (user == nullptr) {
    *error = "unknown user";
    return false;
  }
  // Expired ones go with the write.
  std::erase_if(user->sitzungen, [now](const RememberedSession& s) {
    return s.ablauf > 0 && s.ablauf < now;
  });
  user->sitzungen.push_back({Sha256Hex(session_id), ablauf});
  return SaveLocked(error);
}

std::optional<User> UserStore::FindBySession(std::string_view session_id,
                                             long long* ablauf) const {
  const std::string hash = Sha256Hex(session_id);
  const long long now = NowUnix();
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const User& user : users_) {
    for (const RememberedSession& s : user.sitzungen) {
      if (!ConstantTimeEquals(s.hash, hash))
        continue;
      if (user.gesperrt || (s.ablauf > 0 && s.ablauf < now))
        return std::nullopt;
      if (ablauf != nullptr)
        *ablauf = s.ablauf;
      return user;
    }
  }
  return std::nullopt;
}

void UserStore::ForgetSession(std::string_view session_id) {
  const std::string hash = Sha256Hex(session_id);
  const std::lock_guard<std::mutex> lock(mutex_);
  bool changed = false;
  for (User& user : users_) {
    const auto before = user.sitzungen.size();
    std::erase_if(user.sitzungen, [&](const RememberedSession& s) {
      return ConstantTimeEquals(s.hash, hash);
    });
    changed = changed || before != user.sitzungen.size();
  }
  std::string error;
  if (changed && !SaveLocked(&error))
    Logger::Warn("router", "event=users_file_write_failed reason=" + error);
}

void UserStore::NoteLogin(std::string_view name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (User* user = FindLocked(name)) {
    user->letzte_anmeldung = NowUnix();
    std::string error;
    if (!SaveLocked(&error))
      Logger::Warn("router", "event=users_file_write_failed reason=" + error);
  }
}

// ---------------------------------------------------------------------------
// Identity, pool secret

const char* Identity::SourceName() const {
  switch (source) {
    case Source::kOff:
      return "off";
    case Source::kLoopback:
      return "loopback";
    case Source::kInternal:
      return "internal";
    case Source::kPool:
      return "pool";
    case Source::kBearer:
      return "bearer";
    case Source::kCookie:
      return "cookie";
    default:
      return "none";
  }
}

namespace {
std::mutex pool_secret_mutex;
std::string pool_secret_value;
}  // namespace

void SetPoolSecret(std::string secret) {
  const std::lock_guard<std::mutex> lock(pool_secret_mutex);
  pool_secret_value = std::move(secret);
}

const std::string& PoolSecret() {
  const std::lock_guard<std::mutex> lock(pool_secret_mutex);
  return pool_secret_value;
}

// ---------------------------------------------------------------------------
// Classify

std::string PageArea(std::string_view path) {
  std::string_view rel = path;
  if (rel.starts_with('/'))
    rel.remove_prefix(1);
  const auto slash = rel.find('/');
  const auto query = rel.find('?');
  const std::string page(rel.substr(0, std::min(slash, query)));
  static const std::map<std::string, std::string> kPages = {
      {"bild", "bild"},          {"picture", "bild"},
      {"hochskalieren", "bild"}, {"schritte", "bild"},
      {"steps", "bild"},         {"video", "video"},
      {"lipsync", "video"},      {"objekt", "3d"},
      {"object", "3d"},          {"stimme", "stimme"},
      {"voice", "stimme"},       {"stimmwandel", "stimme"},
      {"transkript", "stimme"},  {"transcript", "stimme"},
      {"hoertest", "stimme"},    {"musik", "musik"},
      {"music", "musik"},        {"ablage", "ablage"},
      {"ergebnisse", "ablage"},  {"chat", "text"},
      {"speicher", "status"},    {"memory", "status"},
      {"pool", "status"},        {"status", "status"},
      {"nacht", "status"},       {"night", "status"}};
  const auto it = kPages.find(page);
  return it == kPages.end() ? std::string() : it->second;
}

Route Classify(const HttpRequest& req) {
  const std::string& p = req.path;
  const std::string& m = req.method;
  const bool get = m == "GET" || m == "HEAD";
  const bool post = m == "POST";
  Route route;
  const auto set = [&route](Route::Kind kind, std::string area = {},
                            int counts = 0, bool filtered = false) {
    route.kind = kind;
    route.area = std::move(area);
    route.counts = counts;
    route.filtered = filtered;
    return route;
  };
  if (get && (p == "/health" || p == "/healthz" || p == "/v1/health" ||
              p == "/ready" || p == "/readyz" || p == "/v1/ready"))
    return set(Route::Kind::kFree);
  if (post && p == "/login")
    return set(Route::Kind::kLogin);
  if (post && p == "/logout")
    return set(Route::Kind::kLogout);
  if (get && p == "/router/me")
    return set(Route::Kind::kMe);
  if (p == "/router/users" || p.starts_with("/router/users/") ||
      p == "/router/zugriff")
    return set(Route::Kind::kUsers);
  if (p == "/verwaltung" || p.starts_with("/verwaltung/"))
    return set(Route::Kind::kAdminPage);
  if (get && (p == "/login" || p.starts_with("/login/") ||
              p.starts_with("/gemeinsam/") || p == "/favicon.ico" ||
              p == "/favicon.svg"))
    return set(Route::Kind::kFree);
  if (p == "/v1/jobs" || p.starts_with("/v1/jobs/"))
    return set(Route::Kind::kJobs, "ablage");
  if (post && (p == "/models/load" || p == "/models/unload" ||
               p == "/router/placement" || p == "/router/pool/pull"))
    return set(Route::Kind::kAdminControl);
  if (p == "/router/status" || p == "/router/capabilities" ||
      p == "/router/placement" || p == "/v1/models" || p == "/models" ||
      p == "/v1/metrics" || p == "/metrics" || p == "/v1/names" ||
      p.starts_with("/v1/names/") || p == "/v1/systemone")
    return set(Route::Kind::kAnyUser);
  // Generation endpoints by prefix.
  struct Prefix {
    const char* prefix;
    const char* area;
    int counts;
    bool filtered;
  };
  static constexpr Prefix kWork[] = {
      {"/v1/chat/completions", "text", 1, false},
      {"/v1/completions", "text", 1, false},
      {"/v1/responses", "text", 1, false},
      {"/v1/messages", "text", 1, false},
      {"/v1/embeddings", "text", 0, false},
      {"/embeddings", "text", 0, false},
      {"/embedding", "text", 0, false},
      {"/completion", "text", 1, false},
      {"/apply-template", "text", 0, false},
      {"/v1/rerank", "text", 0, false},
      {"/v1/reranking", "text", 0, false},
      {"/v1/realtime", "text", 0, false},
      {"/v1/figuren", "text", 0, false},
      // GETs that load the model named by ?model= (router IsModelQueryGet
      // and the introspection paths): the model's area and model list.
      {"/props", "text", 0, false},
      {"/v1/slots", "text", 0, false},
      {"/slots", "text", 0, false},
      {"/sdapi/v1/loras", "bild", 0, false},
      {"/v1/images/", "bild", 1, true},
      {"/v1/images", "bild", 1, true},
      {"/v1/previews/", "bild|video", 0, false},
      {"/v1/videos", "video", 2, true},
      {"/v1/audio/speech", "stimme", 1, false},
      {"/v1/audio/transcriptions", "stimme", 1, false},
      {"/v1/audio/convert", "stimme", 1, false},
      {"/v1/audio/voices", "stimme", 0, false},
      {"/v1/audio/separate", "stimme", 1, false},
      {"/v1/audio/music", "musik", 1, false},
      {"/v1/audio/foley", "musik", 1, false},
      {"/v1/3d/", "3d", 1, false},
      // ComfyUI adapter: graphs make images and videos; counted and
      // filtered like an image request.
      {"/prompt", "bild", 1, true},
      {"/upload/", "bild", 0, false},
      {"/history", "bild", 0, false},
      {"/view", "bild", 0, false},
      {"/queue", "bild", 0, false},
      {"/ws", "bild", 0, false},
      {"/object_info", "bild", 0, false},
      {"/system_stats", "bild", 0, false},
      {"/interrupt", "bild", 0, false},
  };
  for (const Prefix& entry : kWork) {
    const std::string_view prefix = entry.prefix;
    if (p == prefix || (p.starts_with(prefix) &&
                        (prefix.ends_with('/') || p[prefix.size()] == '/' ||
                         p[prefix.size()] == '?'))) {
      return set(Route::Kind::kWork, entry.area, post ? entry.counts : 0,
                 entry.filtered && post);
    }
  }
  if (get && !p.starts_with("/v1/") && !p.starts_with("/router/") &&
      !p.starts_with("/models"))
    return set(Route::Kind::kPage, PageArea(p));
  return set(Route::Kind::kOther);
}

// ---------------------------------------------------------------------------
// Word filter

std::vector<std::string> PromptTexts(const HttpRequest& req, bool all_strings) {
  std::vector<std::string> out;
  if (req.body.empty())
    return out;
  if (IsJsonBody(req)) {
    try {
      CollectStrings(json::parse(req.body), all_strings, 0, &out);
    } catch (const std::exception&) {
      // Not JSON after all: the worker will refuse it.
    }
    return out;
  }
  const std::string boundary = MultipartBoundary(req);
  if (boundary.empty())
    return out;
  const std::string marker = "--" + boundary;
  std::size_t pos = req.body.find(marker);
  while (pos != std::string::npos) {
    pos += marker.size();
    if (req.body.compare(pos, 2, "--") == 0)
      break;
    const auto head_end = req.body.find("\r\n\r\n", pos);
    if (head_end == std::string::npos)
      break;
    const std::string head = Lower(req.body.substr(pos, head_end - pos));
    const auto next = req.body.find(marker, head_end);
    const std::size_t body_begin = head_end + 4;
    const std::size_t body_end =
        next == std::string::npos ? req.body.size() : next;
    const bool is_file = head.find("filename=") != std::string::npos;
    const bool prompt = head.find("name=\"prompt\"") != std::string::npos ||
                        head.find("name=\"negative_prompt\"") !=
                            std::string::npos;
    if (!is_file && (all_strings || prompt) && body_end >= body_begin) {
      std::string text = req.body.substr(body_begin, body_end - body_begin);
      while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
        text.pop_back();
      out.push_back(std::move(text));
    }
    pos = next;
  }
  return out;
}

WordFilter::WordFilter(std::filesystem::path file) : file_(std::move(file)) {}

void WordFilter::EnsureDefaults() {
  std::error_code ec;
  if (std::filesystem::is_regular_file(file_, ec))
    return;
  std::filesystem::create_directories(file_.parent_path(), ec);
  std::ofstream out(file_, std::ios::binary | std::ios::trunc);
  if (out)
    out << kDefaultFilterWords;
}

void WordFilter::Reload() {
  std::error_code ec;
  const auto stamp = std::filesystem::last_write_time(file_, ec);
  if (ec) {
    if (loaded_)
      words_.clear();
    loaded_ = true;
    return;
  }
  if (loaded_ && stamp == loaded_at_)
    return;
  std::ifstream in(file_, std::ios::binary);
  std::vector<std::string> words;
  std::string line;
  bool first = true;
  while (std::getline(in, line)) {
    if (first) {
      first = false;
      if (line.starts_with("\xEF\xBB\xBF"))
        line.erase(0, 3);
    }
    const std::string_view trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#' || trimmed.front() == ';')
      continue;
    words.push_back(LowerUtf8(trimmed));
  }
  words_ = std::move(words);
  loaded_at_ = stamp;
  loaded_ = true;
}

std::size_t WordFilter::size() {
  const std::lock_guard<std::mutex> lock(mutex_);
  Reload();
  return words_.size();
}

namespace {
bool IsWordChar(unsigned char c) {
  return std::isalnum(c) || c >= 0x80 || c == '_';
}
}  // namespace

std::string LowerUtf8(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    // U+00C0..U+00DE (except U+00D7, the multiplication sign) are the
    // Latin-1 capitals; their small letters sit 0x20 higher: C3 80..9E ->
    // C3 A0..BE.
    if (c == 0xC3 && i + 1 < text.size()) {
      const unsigned char d = static_cast<unsigned char>(text[i + 1]);
      if (d >= 0x80 && d <= 0x9E && d != 0x97) {
        out += static_cast<char>(0xC3);
        out += static_cast<char>(d + 0x20);
        ++i;
        continue;
      }
    }
    out += static_cast<char>(std::tolower(c));
  }
  return out;
}

std::string WordFilter::HitText(std::string_view text) {
  const std::lock_guard<std::mutex> lock(mutex_);
  Reload();
  if (words_.empty() || text.empty())
    return {};
  const std::string lower = LowerUtf8(text);
  for (const std::string& entry : words_) {
    std::string_view word = entry;
    const bool stem = word.ends_with('*');
    if (stem)
      word.remove_suffix(1);
    if (word.empty())
      continue;
    std::size_t at = lower.find(word);
    while (at != std::string::npos) {
      const bool start_ok =
          at == 0 || !IsWordChar(static_cast<unsigned char>(lower[at - 1]));
      const std::size_t after = at + word.size();
      const bool end_ok =
          stem || after >= lower.size() ||
          !IsWordChar(static_cast<unsigned char>(lower[after]));
      if (start_ok && end_ok)
        return entry;
      at = lower.find(word, at + 1);
    }
  }
  return {};
}

std::string WordFilter::Hit(const HttpRequest& req, bool comfy) {
  for (const std::string& text : PromptTexts(req, comfy)) {
    if (const std::string word = HitText(text); !word.empty())
      return word;
  }
  return {};
}

// ---------------------------------------------------------------------------
// Gate

Gate::Gate(Settings settings, std::shared_ptr<UserStore> store, Hooks hooks)
    : settings_(std::move(settings)),
      store_(std::move(store)),
      hooks_(std::move(hooks)),
      filter_(store_->dir() / "filter-woerter.txt"),
      internal_key_(RandomHex(32)) {
  if (settings_.mode != Mode::kOff)
    filter_.EnsureDefaults();
  // The day of the last line decides when the log rotates next.
  std::ifstream in(log_file(), std::ios::binary);
  std::string line;
  std::string last;
  while (std::getline(in, line))
    if (line.size() >= 10)
      last = line.substr(0, 10);
  log_day_ = last;
}

std::filesystem::path Gate::log_file() const {
  return store_->dir() / "zugriff.log";
}

bool Gate::SetupPending() const { return !store_->HasAdmin(); }

bool Gate::Blocked(std::string_view address, long long* retry_after) {
  const long long now = Now();
  const std::lock_guard<std::mutex> lock(mutex_);
  auto it = failures_.find(std::string(address));
  if (it == failures_.end())
    return false;
  std::deque<long long>& times = it->second;
  while (!times.empty() && times.front() < now - 60)
    times.pop_front();
  if (times.size() < static_cast<std::size_t>(kFailuresPerMinute)) {
    if (times.empty())
      failures_.erase(it);
    return false;
  }
  const long long until = times.back() + kFailureBlock.count();
  if (until <= now) {
    failures_.erase(it);
    return false;
  }
  *retry_after = until - now;
  return true;
}

void Gate::NoteFailure(std::string_view address) {
  const long long now = Now();
  const std::lock_guard<std::mutex> lock(mutex_);
  std::deque<long long>& times = failures_[std::string(address)];
  times.push_back(now);
  while (times.size() > 32)
    times.pop_front();
}

void Gate::ClearFailures(std::string_view address) {
  const std::lock_guard<std::mutex> lock(mutex_);
  failures_.erase(std::string(address));
}

std::string Gate::NewSession(const User& user, bool remember) {
  const std::string id = "s" + RandomHex(24);
  const long long ttl =
      remember ? kRememberTtl.count() : kSessionTtl.count();
  const long long ablauf = Now() + ttl;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    // Expired sessions go with every new one.
    const long long now = Now();
    for (auto it = sessions_.begin(); it != sessions_.end();) {
      if (it->second.ablauf < now)
        it = sessions_.erase(it);
      else
        ++it;
    }
    sessions_[id] = SessionInfo{user.name, ablauf, remember};
  }
  if (remember) {
    std::string error;
    if (!store_->RememberSession(user.name, id, ablauf, &error))
      Logger::Warn("router", "event=session_remember_failed reason=" + error);
  }
  return id;
}

void Gate::EndSession(std::string_view id) {
  bool remembered = false;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = sessions_.find(std::string(id));
    if (it != sessions_.end()) {
      remembered = it->second.remembered;
      sessions_.erase(it);
    } else {
      remembered = true;  // maybe from a previous router run
    }
  }
  if (remembered)
    store_->ForgetSession(id);
}

void Gate::EndSessionsOf(std::string_view name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = sessions_.begin(); it != sessions_.end();) {
    if (it->second.name == name)
      it = sessions_.erase(it);
    else
      ++it;
  }
}

std::optional<Identity> Gate::SessionIdentity(std::string_view id) {
  if (id.empty())
    return std::nullopt;
  std::string name;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = sessions_.find(std::string(id));
    if (it != sessions_.end()) {
      if (it->second.ablauf < Now()) {
        sessions_.erase(it);
        return std::nullopt;
      }
      name = it->second.name;
    }
  }
  std::optional<User> user;
  if (!name.empty()) {
    user = store_->Find(name);
    if (user.has_value() && user->gesperrt)
      user.reset();
  } else {
    // A remembered session from before a restart: cache it in memory with
    // the expiry the file holds.
    long long ablauf = 0;
    user = store_->FindBySession(id, &ablauf);
    if (user.has_value() && ablauf > 0 && ablauf < Now())
      user.reset();
    if (user.has_value()) {
      const std::lock_guard<std::mutex> lock(mutex_);
      sessions_[std::string(id)] = SessionInfo{user->name, ablauf, true};
    }
  }
  if (!user.has_value())
    return std::nullopt;
  Identity who;
  who.source = Identity::Source::kCookie;
  who.name = user->name;
  who.rolle = user->rolle;
  who.rechte = user->rechte;
  return who;
}

Identity Gate::Identify(const HttpRequest& req) {
  const std::string address = RequestAddress(req);
  const bool loopback = IsLoopback(address);
  // The router's own calls (jobs runner, figure check, ComfyUI adapter):
  // a 256-bit secret of this process, whatever address they arrive from
  // (self_fetch connects to --host, which may be the LAN address).
  {
    const std::string internal = req.header(kInternalHeader);
    if (!internal.empty() && !internal_key_.empty() &&
        ConstantTimeEquals(internal, internal_key_)) {
      Identity who;
      who.source = Identity::Source::kInternal;
      who.rolle = "admin";
      who.name = "lokal";
      const std::string job = req.header("X-Gufo-Job");
      if (!job.empty() && hooks_.job_owner) {
        if (const auto owner = hooks_.job_owner(job);
            owner.has_value() && !owner->empty())
          who.name = *owner;
      }
      return who;
    }
  }
  // The partner router with the shared pool secret.
  if (const std::string key = req.header(kPoolKeyHeader); !key.empty()) {
    if (!settings_.pool_secret.empty() &&
        ConstantTimeEquals(key, settings_.pool_secret)) {
      Identity who;
      who.source = Identity::Source::kPool;
      const std::string user = Lower(req.header(kUserHeader));
      if (!IsValidName(user)) {
        // The partner router itself (probe, capabilities, pull, the
        // queue it reads for the shared order): trusted like the
        // router's own calls, nothing is filtered for it.
        who.name = "partner";
        who.rolle = "admin";
        return who;
      }
      // A relayed request for one of the partner's users: its rights as
      // the entry router checked them, the own-jobs rule applies here.
      who.name = user;
      who.rolle = "nutzer";
      const std::string rights = req.header(kRightsHeader);
      who.rechte = Rights::Decode(rights);
      for (const std::string& pair : Split(rights, ';'))
        if (pair.starts_with("rolle=") && IsRole(pair.substr(6)))
          who.rolle = pair.substr(6);
      return who;
    }
    // A wrong key never falls back to loopback or a cookie.
    return Identity{};
  }
  // Bearer token.
  if (const std::string bearer = BearerOf(req); !bearer.empty()) {
    if (const auto user = store_->FindByToken(bearer); user.has_value()) {
      Identity who;
      who.source = Identity::Source::kBearer;
      who.name = user->name;
      who.rolle = user->rolle;
      who.rechte = user->rechte;
      return who;
    }
    // An unknown token is a refusal (401 invalid_token), except with
    // auth = off, where a client's stray key (an SDK default) is ignored.
    if (settings_.mode != Mode::kOff)
      return Identity{};
  }
  // Session cookie.
  if (const auto who = SessionIdentity(CookieValue(req, kSessionCookie));
      who.has_value())
    return *who;
  if (loopback && settings_.mode == Mode::kLan &&
      settings_.loopback == LoopbackPolicy::kAdmin) {
    Identity who;
    who.source = Identity::Source::kLoopback;
    who.name = "lokal";
    who.rolle = "admin";
    return who;
  }
  if (settings_.mode == Mode::kOff) {
    Identity who;
    who.source = Identity::Source::kOff;
    who.name = "lokal";
    who.rolle = "admin";
    return who;
  }
  return Identity{};
}

bool Gate::CountDaily(const Identity& who, int weight, long long limit) {
  const std::string day = DayOf(Now());
  const std::lock_guard<std::mutex> lock(mutex_);
  auto& entry = daily_[who.name];
  if (entry.first != day) {
    entry.first = day;
    entry.second = 0;
  }
  if (limit > 0 && entry.second + weight > limit)
    return false;
  entry.second += weight;
  return true;
}

void Gate::RememberOwner(std::string_view key, std::string_view name) {
  if (key.empty() || name.empty())
    return;
  const long long now = Now();
  const std::lock_guard<std::mutex> lock(mutex_);
  if (owners_.size() >= 20000) {
    for (auto it = owners_.begin(); it != owners_.end();)
      it = it->second.second < now - 86400 ? owners_.erase(it) : std::next(it);
    if (owners_.size() >= 20000)
      owners_.clear();  // a day's worth of ids is gone: unknown = open
  }
  owners_[std::string(key)] = {std::string(name), now};
}

std::optional<std::string> Gate::OwnerOf(std::string_view key) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = owners_.find(std::string(key));
  if (it == owners_.end() || it->second.second < Now() - 86400)
    return std::nullopt;
  return it->second.first;
}

long long Gate::DailyCount(std::string_view name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = daily_.find(std::string(name));
  return it == daily_.end() ? 0 : it->second.second;
}

namespace {

HttpResponse Forbidden(std::string_view message, std::string_view code) {
  return AuthError(403, "Forbidden", message, code);
}

HttpResponse LoginRedirect(const HttpRequest& req) {
  std::string back = req.path;
  if (!req.query.empty())
    back += "?" + req.query;
  return HttpResponse{.status = 302,
                      .reason = "Found",
                      .headers = {{"Location",
                                   "/login/?zurueck=" + UrlEncode(back)},
                                  {"Cache-Control", "no-store"}}};
}

HttpResponse ForbiddenPage(std::string_view area) {
  const std::string html =
      "<!doctype html><html lang=\"de\"><meta charset=\"utf-8\">"
      "<meta name=\"viewport\" content=\"width=device-width, "
      "initial-scale=1\"><title>gufo</title>"
      "<body style=\"font:16px system-ui;padding:24px;max-width:600px\">"
      "<h1 style=\"font-size:20px\">Nicht freigegeben / not enabled</h1>"
      "<p>Der Bereich <b>" +
      std::string(area) +
      "</b> ist f&uuml;r diesen Nutzer nicht freigegeben. / The area <b>" +
      std::string(area) +
      "</b> is not enabled for this user.</p>"
      "<p><a href=\"/start/\">Zur Startseite / Home</a></p></body></html>";
  return HttpResponse{.status = 403,
                      .reason = "Forbidden",
                      .body = html,
                      .headers = {{"Content-Type", "text/html; charset=utf-8"},
                                  {"Cache-Control", "no-store"}}};
}

std::string AreaMessage(std::string_view area) {
  return "Bereich " + std::string(area) +
         " ist für diesen Nutzer nicht freigegeben (area " +
         std::string(area) + " is not enabled for this user)";
}

/// The job id of /v1/jobs/{id}[/...] ("" for the collection).
std::string JobIdOf(const std::string& path) {
  constexpr std::string_view kPrefix = "/v1/jobs/";
  if (!path.starts_with(kPrefix))
    return "";
  std::string id = path.substr(kPrefix.size());
  if (const auto slash = id.find('/'); slash != std::string::npos)
    id.resize(slash);
  if (const auto q = id.find('?'); q != std::string::npos)
    id.resize(q);
  return id;
}

}  // namespace

std::optional<HttpResponse> Gate::Refuse(const HttpRequest& req,
                                         const Route& route,
                                         const Identity& who,
                                         const json::Value* body) {
  if (route.kind == Route::Kind::kFree)
    return std::nullopt;
  // A pool key that did not match: 401 whatever else the request carries,
  // so the partner marks this router unreachable instead of running work
  // under a cookie or as loopback.
  if (!req.header(kPoolKeyHeader).empty() &&
      who.source != Identity::Source::kPool) {
    return AuthError(401, "Unauthorized",
                     "pool key missing or wrong (pool-secret-env)",
                     "pool_key_invalid");
  }
  if (route.kind == Route::Kind::kLogin || route.kind == Route::Kind::kLogout ||
      route.kind == Route::Kind::kMe)
    return std::nullopt;
  if (!who.authenticated()) {
    if (!BearerOf(req).empty())
      return AuthError(401, "Unauthorized", "unknown or expired API token",
                       "invalid_token");
    if (route.kind == Route::Kind::kPage || route.kind == Route::Kind::kAdminPage)
      return LoginRedirect(req);
    return AuthError(401, "Unauthorized",
                     SetupPending()
                         ? "login required; no admin user yet (create one "
                           "from the machine itself)"
                         : "login required (cookie gufo_sitzung or "
                           "Authorization: Bearer gufo_...)",
                     SetupPending() ? "no_admin" : "not_authenticated");
  }
  // The partner's relayed job requests run for the user the entry router
  // named: its rights were checked there, but which jobs are its own only
  // this router knows (the owner check below applies).
  const bool pool_user =
      who.source == Identity::Source::kPool && !who.admin();
  if (who.admin() ||
      (who.trusted() && !(pool_user && route.kind == Route::Kind::kJobs)))
    return std::nullopt;
  switch (route.kind) {
    case Route::Kind::kAdminControl:
      return Forbidden("nur admin: Modelle laden/entladen, Platzierung, "
                       "Pool-Abholen (admin only)",
                       "forbidden");
    case Route::Kind::kUsers:
    case Route::Kind::kAdminPage:
      return NotFound();
    case Route::Kind::kPage:
      if (!who.rechte.AllowsArea(route.area))
        return ForbiddenPage(route.area);
      return std::nullopt;
    case Route::Kind::kAnyUser:
    case Route::Kind::kOther:
      return std::nullopt;
    case Route::Kind::kJobs: {
      if (!pool_user && !who.rechte.AllowsArea("ablage"))
        return Forbidden(AreaMessage("ablage"), "forbidden");
      if (req.method == "DELETE" && req.path == "/v1/jobs")
        return Forbidden("Sammel-Löschen nur admin (bulk delete is admin "
                         "only)",
                         "forbidden");
      if (req.method == "POST" && req.path == "/v1/jobs") {
        if (pool_user)
          return std::nullopt;  // checked by the entry router
        // The wrapped request is checked like a direct one: a JSON body
        // or a base64 body (multipart edits, upscale) with its type.
        HttpRequest inner;
        inner.method = "POST";
        inner.client_id = req.client_id;
        json::Value parsed;
        if (body == nullptr) {
          try {
            parsed = json::parse(req.body);
          } catch (const std::exception&) {
            return std::nullopt;  // the job API answers the parse error
          }
          body = &parsed;
        }
        const json::Value* request = body->find("request");
        if (request != nullptr && request->is_object()) {
          inner.path = request->member_str("path");
          if (const json::Value* b = request->find("body");
              b != nullptr && !b->is_null()) {
            inner.body = b->dump();
            inner.headers.emplace_back("Content-Type", "application/json");
          } else if (const json::Value* raw = request->find("body_base64");
                     raw != nullptr && raw->is_string()) {
            auto decoded = comfy::Base64Decode(raw->str());
            if (!decoded.has_value())
              return std::nullopt;  // the job API answers invalid_request
            inner.body = std::move(*decoded);
            inner.headers.emplace_back("Content-Type",
                                       request->member_str("content_type"));
          }
        }
        if (inner.path.empty())
          return std::nullopt;
        Route inner_route = Classify(inner);
        if (inner_route.kind != Route::Kind::kWork)
          return std::nullopt;
        return Refuse(inner, inner_route, who);
      }
      const std::string job_id = JobIdOf(req.path);
      if (!job_id.empty() && hooks_.job_owner) {
        if (const auto owner = hooks_.job_owner(job_id);
            owner.has_value() && *owner != who.name)
          return NotFound();
      }
      return std::nullopt;
    }
    case Route::Kind::kWork: {
      int counts = route.counts;
      std::optional<std::vector<WorkItem>> items;
      if (req.method == "POST" && req.path == "/prompt" && hooks_.comfy_plan) {
        // A ComfyUI graph: every task it resolves to is checked like a
        // direct request (area, model, filter), the daily limit counts
        // them all (videos twice). A graph that does not build gets the
        // plain checks below (area bild, every string through the
        // filter) and the adapter's error; nothing is counted for it.
        items = hooks_.comfy_plan(req);
        if (!items.has_value())
          counts = 0;
      }
      if (items.has_value()) {
        counts = 0;
        for (const WorkItem& item : *items) {
          if (!who.rechte.AllowsArea(item.area))
            return Forbidden(AreaMessage(item.area), "forbidden");
          if (!who.rechte.AllowsModel(item.model))
            return Forbidden("Modell " + item.model +
                                 " ist für diesen Nutzer nicht freigegeben "
                                 "(model " +
                                 item.model + " is not enabled for this user)",
                             "model_forbidden");
          if (who.rechte.FilterOn(who.rolle)) {
            for (const std::string& text : item.texts) {
              if (const std::string word = filter_.HitText(text);
                  !word.empty()) {
                Log(req, who, "abgelehnt:content_filtered", item.model);
                return AuthError(400, "Bad Request",
                                 "Der Prompt enthält ein gesperrtes Wort "
                                 "(the prompt contains a blocked word)",
                                 "content_filtered");
              }
            }
          }
          counts += item.counts;
        }
        if (counts > 0 && who.rechte.tageslimit > 0 &&
            !CountDaily(who, counts, who.rechte.tageslimit))
          return AuthError(429, "Too Many Requests",
                           "Tageslimit von " +
                               std::to_string(who.rechte.tageslimit) +
                               " Aufträgen erreicht (daily limit reached)",
                           "daily_limit");
        return std::nullopt;
      }
      if (!who.rechte.AllowsArea(route.area))
        return Forbidden(AreaMessage(route.area), "forbidden");
      // Things of other users the job store does not know: a direct video
      // job (/v1/videos/<id>[/content]), the job files a concat names, a
      // ComfyUI prompt's history entry (POST /history deletes).
      if (req.path.starts_with("/v1/videos/") && req.method != "POST") {
        std::string id = req.path.substr(std::string("/v1/videos/").size());
        if (const auto slash = id.find('/'); slash != std::string::npos)
          id.resize(slash);
        if (const auto owner = OwnerOf("video:" + id);
            owner.has_value() && *owner != who.name)
          return NotFound();
      }
      if (req.method == "POST" &&
          (req.path == "/v1/videos/concat" || req.path == "/history")) {
        try {
          const json::Value body = json::parse(req.body);
          const json::Value* list =
              body.find(req.path == "/history" ? "delete" : "videos");
          if (list != nullptr && list->is_array()) {
            for (const json::Value& item : list->items()) {
              if (!item.is_string())
                continue;
              std::optional<std::string> owner;
              if (req.path == "/history") {
                owner = OwnerOf("prompt:" + item.str());
              } else if (const auto at = item.str().find("/v1/jobs/");
                         at != std::string::npos && hooks_.job_owner) {
                std::string id = item.str().substr(at + 9);
                if (const auto slash = id.find('/'); slash != std::string::npos)
                  id.resize(slash);
                owner = hooks_.job_owner(id);
              }
              if (owner.has_value() && *owner != who.name)
                return NotFound();
            }
          }
        } catch (const std::exception&) {
          // The adapter / concat answers the parse error.
        }
      }
      std::string model;
      if (hooks_.model_of)
        model = hooks_.model_of(req);
      if (!who.rechte.AllowsModel(model))
        return Forbidden("Modell " + model +
                             " ist für diesen Nutzer nicht freigegeben "
                             "(model " +
                             model + " is not enabled for this user)",
                         "model_forbidden");
      if (route.filtered && who.rechte.FilterOn(who.rolle)) {
        const std::string word = filter_.Hit(req, req.path == "/prompt");
        if (!word.empty()) {
          Log(req, who, "abgelehnt:content_filtered", model);
          return AuthError(400, "Bad Request",
                           "Der Prompt enthält ein gesperrtes Wort (the "
                           "prompt contains a blocked word)",
                           "content_filtered");
        }
      }
      if (counts > 0 && who.rechte.tageslimit > 0 &&
          !CountDaily(who, counts, who.rechte.tageslimit)) {
        return AuthError(429, "Too Many Requests",
                         "Tageslimit von " +
                             std::to_string(who.rechte.tageslimit) +
                             " Aufträgen erreicht (daily limit reached)",
                         "daily_limit");
      }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

void Gate::Rewrite(HttpRequest* req, const Identity& who) {
  std::erase_if(req->headers, [](const auto& header) {
    const std::string name = Lower(header.first);
    return name == Lower(kUserHeader) || name == Lower(kRightsHeader) ||
           name == Lower(kInternalHeader) || name == Lower(kPoolKeyHeader);
  });
  req->headers.emplace_back(kUserHeader, who.name);
  req->headers.emplace_back(kRightsHeader,
                            "rolle=" + who.rolle + ";" + who.rechte.Encode());
  if (!who.trusted() && !who.admin() && !who.rechte.pool) {
    // The job API keeps the pin on the job (pin_local), the pool decision
    // keeps a direct request here.
    std::erase_if(req->headers, [](const auto& header) {
      return Lower(header.first) == "x-gufo-pool";
    });
    req->headers.emplace_back("X-Gufo-Pool", "local");
  }
  // The job API stamps metadata.nutzer from kUserHeader itself (no second
  // parse and no re-serialisation of a submit body here).
}

void Gate::After(const HttpRequest& req, const Route& route,
                 const Identity& who, HttpResponse* response) {
  const bool pool_user =
      who.source == Identity::Source::kPool && !who.admin();
  // Owners of ComfyUI prompts and direct video jobs come from the answer.
  if (req.method == "POST" && response->status < 300 && !who.name.empty() &&
      route.kind == Route::Kind::kWork) {
    if (req.path == "/prompt" && !response->streaming_body) {
      try {
        const std::string id = json::parse(response->body).member_str("prompt_id");
        RememberOwner("prompt:" + id, who.name);
      } catch (const std::exception&) {
      }
    } else if (req.path == "/v1/videos") {
      const std::string name = who.name;
      const auto record = [this, name](const std::string& body) {
        try {
          const std::string id = json::parse(body).member_str("id");
          RememberOwner("video:" + id, name);
        } catch (const std::exception&) {
        }
      };
      if (response->streaming_body) {
        // The worker's answer streams through: read the id on the way.
        auto inner = std::move(response->streaming_body);
        response->streaming_body =
            [inner, record](const HttpResponse::BodyWriter& write) {
              std::string seen;
              inner([&](std::string_view chunk) {
                if (seen.size() < 65536)
                  seen.append(chunk);
                return write(chunk);
              });
              record(seen);
            };
      } else {
        record(response->body);
      }
    }
    return;
  }
  const bool restricted = !who.admin() && !(who.trusted() && !pool_user);
  if (restricted && req.method == "GET" && route.kind == Route::Kind::kWork &&
      (req.path == "/history" || req.path.starts_with("/history/")) &&
      response->status == 200 && !response->streaming_body) {
    // ComfyUI history: an object keyed by prompt id; a user sees its own.
    try {
      const json::Value history = json::parse(response->body);
      if (history.is_object()) {
        json::Value kept = json::Value::object();
        for (const auto& [id, entry] : history.members()) {
          const auto owner = OwnerOf("prompt:" + id);
          if (owner.has_value() && *owner == who.name)
            kept[id] = entry;
        }
        response->body = kept.dump();
      }
    } catch (const std::exception&) {
    }
    return;
  }
  if (route.kind != Route::Kind::kJobs || !restricted || req.method != "GET" ||
      response->status != 200 || response->streaming_body)
    return;
  if (req.path == "/v1/jobs") {
    try {
      json::Value body = json::parse(response->body);
      const json::Value* data = body.find("data");
      if (data == nullptr || !data->is_array())
        return;
      json::Value kept = json::Value::array();
      for (const json::Value& job : data->items()) {
        const json::Value* metadata = job.find("metadata");
        if (metadata != nullptr &&
            metadata->member_str("nutzer") == who.name)
          kept.push_back(job);
      }
      body["data"] = std::move(kept);
      response->body = body.dump();
    } catch (const std::exception&) {
    }
    return;
  }
  const std::string job_id = JobIdOf(req.path);
  if (job_id.empty() || req.path != "/v1/jobs/" + job_id)
    return;
  // A job known here was checked before the call; this covers the
  // partner's jobs (relayed reads), whose owner only the answer names.
  if (hooks_.job_owner && hooks_.job_owner(job_id).has_value())
    return;
  try {
    const json::Value job = json::parse(response->body);
    const json::Value* metadata = job.find("metadata");
    if (metadata == nullptr || metadata->member_str("nutzer") != who.name)
      *response = NotFound();
  } catch (const std::exception&) {
  }
}

json::Value Gate::MeJson(const Identity& who) const {
  json::Value out = json::Value::object();
  out["auth"] = ModeName(settings_.mode);
  out["name"] = who.name;
  out["rolle"] = who.rolle;
  out["quelle"] = who.SourceName();
  json::Value areas = json::Value::array();
  if (who.admin() || who.rechte.bereiche.empty()) {
    for (const std::string_view area : kAreas)
      areas.push_back(std::string(area));
  } else {
    areas = ArrayOf(who.rechte.bereiche);
  }
  out["bereiche"] = std::move(areas);
  out["modelle"] = ArrayOf(who.rechte.modelle);
  out["filter"] = who.rechte.FilterOn(who.rolle);
  out["tageslimit"] = who.rechte.tageslimit;
  out["pool"] = who.rechte.pool;
  out["verwaltung"] = who.admin() && who.source != Identity::Source::kOff;
  out["setup"] = SetupPending();
  return out;
}

HttpResponse Gate::Login(const HttpRequest& req) {
  const std::string address = RequestAddress(req);
  long long retry_after = 0;
  if (Blocked(address, &retry_after)) {
    HttpResponse r = AuthError(429, "Too Many Requests",
                               "zu viele Fehlversuche, bitte " +
                                   std::to_string(retry_after) +
                                   " s warten (too many failed logins)",
                               "too_many_attempts");
    r.headers.emplace_back("Retry-After", std::to_string(retry_after));
    return r;
  }
  std::string name;
  std::string password;
  bool remember = false;
  try {
    const json::Value body = json::parse(req.body.empty() ? "{}" : req.body);
    name = Lower(Trim(body.member_str("name")));
    password = body.member_str("passwort", body.member_str("password"));
    if (const json::Value* b = body.find("bleiben"); b != nullptr && b->is_bool())
      remember = b->as_bool();
    if (const json::Value* b = body.find("remember");
        b != nullptr && b->is_bool())
      remember = b->as_bool();
  } catch (const std::exception&) {
    return AuthError(400, "Bad Request",
                     "JSON {\"name\", \"passwort\", \"bleiben\"} expected",
                     "invalid_request");
  }
  if (name.empty() || password.empty())
    return AuthError(400, "Bad Request", "name and passwort are required",
                     "invalid_request");
  const auto user = store_->Verify(name, password);
  if (!user.has_value()) {
    NoteFailure(address);
    Identity who;
    who.name = name;
    Log(req, who, "login_fehlgeschlagen");
    json::Value body = json::Value::object();
    json::Value detail = json::Value::object();
    detail["message"] =
        "Name oder Passwort falsch (wrong name or password)";
    detail["type"] = "authentication_error";
    detail["code"] = "login_failed";
    body["error"] = std::move(detail);
    body["setup"] = SetupPending();
    return Json(401, "Unauthorized", body);
  }
  if (user->rolle == "dienst") {
    Identity who;
    who.name = name;
    Log(req, who, "login_dienst_abgelehnt");
    return Forbidden("Dienstkonten melden sich nur mit Token an (service "
                     "accounts use a token)",
                     "login_not_allowed");
  }
  ClearFailures(address);
  const std::string id = NewSession(*user, remember);
  store_->NoteLogin(user->name);
  Identity who;
  who.source = Identity::Source::kCookie;
  who.name = user->name;
  who.rolle = user->rolle;
  who.rechte = user->rechte;
  Log(req, who, remember ? "login_ok_bleiben" : "login_ok");
  json::Value body = MeJson(who);
  body["ok"] = true;
  return Json(200, "OK", body,
              {{"Set-Cookie",
                SessionCookie(id, remember ? kRememberTtl.count()
                                           : kSessionTtl.count())}});
}

HttpResponse Gate::Logout(const HttpRequest& req, const Identity& who) {
  const std::string id = CookieValue(req, kSessionCookie);
  if (!id.empty())
    EndSession(id);
  if (who.authenticated())
    Log(req, who, "logout");
  json::Value body = json::Value::object();
  body["ok"] = true;
  return Json(200, "OK", body, {{"Set-Cookie", SessionCookie("", 0)}});
}

HttpResponse Gate::UsersApi(const HttpRequest& req, const Identity& who) {
  const std::string& p = req.path;
  if (p == "/router/zugriff") {
    if (req.method != "GET")
      return AuthError(405, "Method Not Allowed", "GET only",
                       "method_not_allowed");
    std::size_t n = 200;
    try {
      if (const std::string q = req.query_param("n"); !q.empty())
        n = std::clamp<std::size_t>(std::stoull(q), 1, 5000);
    } catch (const std::exception&) {
    }
    json::Value body = json::Value::object();
    body["zeilen"] = ArrayOf(LogTail(n));
    body["datei"] = log_file().string();
    return JsonOk(body);
  }
  if (p == "/router/users") {
    if (req.method == "GET") {
      json::Value body = json::Value::object();
      json::Value list = json::Value::array();
      for (const User& user : store_->List())
        list.push_back(user.PublicJson());
      body["users"] = std::move(list);
      body["datei"] = store_->path().string();
      body["auth"] = ModeName(settings_.mode);
      body["filter_woerter"] = static_cast<long long>(filter_.size());
      body["setup"] = SetupPending();
      return JsonOk(body);
    }
    if (req.method == "POST") {
      json::Value body;
      try {
        body = json::parse(req.body);
      } catch (const std::exception& e) {
        return AuthError(400, "Bad Request", std::string("bad JSON: ") + e.what(),
                         "invalid_request");
      }
      User user;
      user.name = Lower(Trim(body.member_str("name")));
      user.rolle = body.member_str("rolle", "nutzer");
      if (!IsValidName(user.name))
        return AuthError(400, "Bad Request",
                         "bad user name (1-32 of a-z 0-9 . _ -)",
                         "invalid_name");
      if (!IsRole(user.rolle))
        return AuthError(400, "Bad Request", "rolle takes admin|nutzer|dienst",
                         "invalid_role");
      if (store_->Find(user.name).has_value())
        return AuthError(409, "Conflict", "user exists", "user_exists");
      if (const json::Value* r = body.find("rechte"); r != nullptr)
        user.rechte = Rights::FromJson(*r);
      user.erstellt = Now();
      const std::string password = body.member_str("passwort");
      if (user.rolle != "dienst" && password.size() < 4)
        return AuthError(400, "Bad Request",
                         "passwort: at least 4 characters", "invalid_password");
      std::string error;
      if (!store_->Put(user, &error))
        return AuthError(500, "Internal Server Error", error, "users_file");
      if (!password.empty() && !store_->SetPassword(user.name, password, &error))
        return AuthError(500, "Internal Server Error", error, "users_file");
      Log(req, who, "nutzer_angelegt:" + user.name);
      return Json(201, "Created", store_->Find(user.name)->PublicJson());
    }
    return AuthError(405, "Method Not Allowed", "GET or POST",
                     "method_not_allowed");
  }
  // /router/users/<name>[/tokens[/<token>]]
  constexpr std::string_view kPrefix = "/router/users/";
  const std::string rest = p.substr(kPrefix.size());
  const auto slash = rest.find('/');
  const std::string name = Lower(rest.substr(0, slash));
  const std::string tail = slash == std::string::npos ? "" : rest.substr(slash);
  auto user = store_->Find(name);
  if (!user.has_value())
    return AuthError(404, "Not Found", "unknown user", "user_not_found");
  std::string error;
  if (tail.empty()) {
    if (req.method == "GET")
      return JsonOk(user->PublicJson());
    if (req.method == "PUT" || req.method == "PATCH") {
      json::Value body;
      try {
        body = json::parse(req.body);
      } catch (const std::exception& e) {
        return AuthError(400, "Bad Request", std::string("bad JSON: ") + e.what(),
                         "invalid_request");
      }
      if (const std::string rolle = body.member_str("rolle"); !rolle.empty()) {
        if (!IsRole(rolle))
          return AuthError(400, "Bad Request", "rolle takes admin|nutzer|dienst",
                           "invalid_role");
        user->rolle = rolle;
      }
      if (const json::Value* r = body.find("rechte"); r != nullptr)
        user->rechte = Rights::FromJson(*r);
      if (const json::Value* g = body.find("gesperrt");
          g != nullptr && g->is_bool())
        user->gesperrt = g->as_bool();
      if (!store_->Put(*user, &error))
        return AuthError(500, "Internal Server Error", error, "users_file");
      if (const std::string password = body.member_str("passwort");
          !password.empty()) {
        if (!store_->SetPassword(name, password, &error))
          return AuthError(400, "Bad Request", error, "invalid_password");
        // Whoever held the old password is out, in memory as well.
        EndSessionsOf(name);
      }
      if (user->gesperrt)
        EndSessionsOf(name);
      Log(req, who, "nutzer_geaendert:" + name);
      return JsonOk(store_->Find(name)->PublicJson());
    }
    if (req.method == "DELETE") {
      if (req.query_param("loeschen") == "1" ||
          req.query_param("delete") == "1") {
        if (!store_->Remove(name, &error))
          return AuthError(500, "Internal Server Error", error, "users_file");
        Log(req, who, "nutzer_geloescht:" + name);
        json::Value body = json::Value::object();
        body["ok"] = true;
        body["geloescht"] = name;
        return JsonOk(body);
      }
      user->gesperrt = true;
      if (!store_->Put(*user, &error))
        return AuthError(500, "Internal Server Error", error, "users_file");
      EndSessionsOf(name);
      Log(req, who, "nutzer_gesperrt:" + name);
      return JsonOk(store_->Find(name)->PublicJson());
    }
    return AuthError(405, "Method Not Allowed", "GET, PUT or DELETE",
                     "method_not_allowed");
  }
  if (tail == "/tokens" || tail == "/tokens/") {
    if (req.method != "POST")
      return AuthError(405, "Method Not Allowed", "POST creates a token",
                       "method_not_allowed");
    json::Value body;
    try {
      body = json::parse(req.body.empty() ? "{}" : req.body);
    } catch (const std::exception& e) {
      return AuthError(400, "Bad Request", std::string("bad JSON: ") + e.what(),
                       "invalid_request");
    }
    const std::string token_name = Lower(Trim(body.member_str("name")));
    const long long days = NumberOf(body, "tage", 0);
    const long long ablauf = days > 0 ? Now() + days * 86400 : 0;
    const std::string token =
        store_->CreateToken(name, token_name, ablauf, &error);
    if (token.empty())
      return AuthError(400, "Bad Request", error, "invalid_token_request");
    Log(req, who, "token_erzeugt:" + name + "/" + token_name);
    json::Value out = json::Value::object();
    out["token"] = token;
    out["name"] = token_name;
    out["nutzer"] = name;
    out["ablauf"] = ablauf;
    out["hinweis"] =
        "Nur einmal sichtbar; im Router liegt nur der Hash (shown once).";
    return Json(201, "Created", out);
  }
  if (tail.starts_with("/tokens/")) {
    if (req.method != "DELETE")
      return AuthError(405, "Method Not Allowed", "DELETE revokes a token",
                       "method_not_allowed");
    const std::string token_name = tail.substr(8);
    if (!store_->RevokeToken(name, token_name, &error))
      return AuthError(404, "Not Found", error, "token_not_found");
    Log(req, who, "token_widerrufen:" + name + "/" + token_name);
    json::Value out = json::Value::object();
    out["ok"] = true;
    return JsonOk(out);
  }
  return NotFound();
}

std::optional<HttpResponse> Gate::AuthEndpoint(const HttpRequest& req,
                                               const Route& route,
                                               const Identity& who) {
  switch (route.kind) {
    case Route::Kind::kMe: {
      if (who.authenticated())
        return JsonOk(MeJson(who));
      json::Value body = json::Value::object();
      json::Value detail = json::Value::object();
      detail["message"] = SetupPending()
                              ? "no admin user yet"
                              : "login required";
      detail["type"] = "authentication_error";
      detail["code"] = SetupPending() ? "no_admin" : "not_authenticated";
      body["error"] = std::move(detail);
      body["auth"] = ModeName(settings_.mode);
      body["setup"] = SetupPending();
      return Json(401, "Unauthorized", body);
    }
    case Route::Kind::kLogin:
      return Login(req);
    case Route::Kind::kLogout:
      return Logout(req, who);
    case Route::Kind::kUsers:
    case Route::Kind::kAdminPage: {
      // Loopback while it counts as admin (auth = lan with auth-loopback
      // = admin, or auth = off) or an admin session; everyone else sees
      // 404 so the pages stay invisible from outside. With auth-loopback
      // = login the emergency path is the preset: set it to admin, restart.
      const bool loopback_ok = IsLoopback(RequestAddress(req)) &&
                               settings_.mode != Mode::kAll &&
                               settings_.loopback == LoopbackPolicy::kAdmin;
      const bool admin_ok = who.authenticated() && who.admin() &&
                            who.source != Identity::Source::kOff;
      if (!loopback_ok && !admin_ok)
        return NotFound();
      if (route.kind == Route::Kind::kAdminPage)
        return std::nullopt;
      if (req.method != "GET" && req.method != "HEAD") {
        // A page from elsewhere in a browser on this machine must not
        // reach the API (no-cors POST): a cross-site fetch, or an Origin
        // that is not this router, is refused; a body must be JSON (a
        // preflight stops a cross-site one before it is sent).
        if (Lower(req.header("Sec-Fetch-Site")) == "cross-site")
          return AuthError(403, "Forbidden",
                           "cross-site request refused (Sec-Fetch-Site)",
                           "csrf_rejected");
        const std::string origin = Lower(req.header("Origin"));
        const std::string host = Lower(req.header("Host"));
        if (!origin.empty() && origin != "null") {
          std::string origin_host = origin;
          if (const auto scheme = origin_host.find("://");
              scheme != std::string::npos)
            origin_host.erase(0, scheme + 3);
          if (host.empty() || origin_host != host)
            return AuthError(403, "Forbidden",
                             "cross-site request refused (Origin " + origin +
                                 " is not this router)",
                             "csrf_rejected");
        }
        if (!req.body.empty() &&
            !Lower(req.header("Content-Type")).starts_with("application/json"))
          return AuthError(415, "Unsupported Media Type",
                           "the users API takes application/json",
                           "json_required");
      }
      return UsersApi(req, who);
    }
    default:
      return std::nullopt;
  }
}

HttpResponse Gate::Serve(
    const HttpRequest& req,
    const std::function<HttpResponse(const HttpRequest&)>& next) {
  const Route route = Classify(req);
  if (route.kind == Route::Kind::kFree)
    return next(req);
  const Identity who = Identify(req);
  if (settings_.mode == Mode::kOff) {
    // Stage 0: only the auth endpoints exist. A client's identity headers
    // still go (they would name a job's owner here and reach a partner
    // with auth = lan beside the real pool key); nothing else is touched.
    if (auto answer = AuthEndpoint(req, route, who))
      return std::move(*answer);
    const bool spoofed = std::any_of(
        req.headers.begin(), req.headers.end(), [](const auto& header) {
          const std::string name = Lower(header.first);
          return name == Lower(kUserHeader) || name == Lower(kRightsHeader) ||
                 name == Lower(kInternalHeader);
        });
    if (!spoofed)
      return next(req);
    HttpRequest copy = req;
    std::erase_if(copy.headers, [](const auto& header) {
      const std::string name = Lower(header.first);
      return name == Lower(kUserHeader) || name == Lower(kRightsHeader) ||
             name == Lower(kInternalHeader);
    });
    return next(copy);
  }
  if (auto answer = AuthEndpoint(req, route, who))
    return std::move(*answer);
  // A job submit is parsed once here for the inner-request check (the job
  // API parses its own copy; nothing is re-serialised).
  json::Value submit;
  const json::Value* submit_body = nullptr;
  if (req.method == "POST" && req.path == "/v1/jobs" && !who.admin() &&
      !who.trusted()) {
    try {
      submit = json::parse(req.body);
      submit_body = &submit;
    } catch (const std::exception&) {
      // The job API answers the parse error.
    }
  }
  if (auto refusal = Refuse(req, route, who, submit_body)) {
    std::string model;
    if (hooks_.model_of && route.kind == Route::Kind::kWork)
      model = hooks_.model_of(req);
    std::string code = "abgelehnt:" + std::to_string(refusal->status);
    if (refusal->status != 302) {
      try {
        const json::Value body = json::parse(refusal->body);
        if (const json::Value* error = body.find("error"))
          code = "abgelehnt:" + error->member_str("code", code);
      } catch (const std::exception&) {
      }
      Log(req, who, code, model);
    }
    return std::move(*refusal);
  }
  HttpRequest copy = req;
  Rewrite(&copy, who);
  HttpResponse response = next(copy);
  After(req, route, who, &response);
  if (req.method != "GET" && req.method != "HEAD" &&
      (route.kind == Route::Kind::kWork || route.kind == Route::Kind::kJobs ||
       route.kind == Route::Kind::kAdminControl)) {
    std::string model;
    if (hooks_.model_of)
      model = hooks_.model_of(req);
    Log(req, who,
        response.status < 400 ? "ok" : "fehler:" + std::to_string(response.status),
        model);
  }
  return response;
}

// ---------------------------------------------------------------------------
// Access log

void Gate::RotateLogLocked() {
  const std::string today = DayOf(Now());
  if (log_day_ == today)
    return;
  std::error_code ec;
  const std::filesystem::path file = log_file();
  if (!log_day_.empty() && std::filesystem::is_regular_file(file, ec)) {
    std::filesystem::path dated = file;
    dated.replace_filename("zugriff-" + log_day_ + ".log");
    std::filesystem::rename(file, dated, ec);
  }
  log_day_ = today;
  // Files older than kLogKeepDays: the name carries the day.
  const std::string cutoff = DayOf(Now() - kLogKeepDays * 86400LL);
  for (const auto& entry : std::filesystem::directory_iterator(
           file.parent_path(), ec)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with("zugriff-") && name.ends_with(".log") &&
        name.size() == std::string("zugriff-YYYY-MM-DD.log").size() &&
        name.substr(8, 10) < cutoff)
      std::filesystem::remove(entry.path(), ec);
  }
}

void Gate::Log(const HttpRequest& req, const Identity& who,
               std::string_view result, std::string_view model) {
  const std::lock_guard<std::mutex> lock(log_mutex_);
  RotateLogLocked();
  std::error_code ec;
  std::filesystem::create_directories(log_file().parent_path(), ec);
  std::ofstream out(log_file(), std::ios::binary | std::ios::app);
  if (!out)
    return;
  const auto clean = [](std::string_view text) {
    std::string s(text.substr(0, 200));
    for (char& c : s)
      if (c == '\t' || c == '\n' || c == '\r')
        c = ' ';
    return s;
  };
  out << IsoOf(Now()) << '\t' << clean(who.name.empty() ? "-" : who.name)
      << '\t' << clean(RequestAddress(req)) << '\t' << clean(req.method) << ' '
      << clean(req.path) << '\t' << clean(model.empty() ? "-" : model) << '\t'
      << clean(result) << '\n';
}

std::vector<std::string> Gate::LogTail(std::size_t lines) {
  const std::lock_guard<std::mutex> lock(log_mutex_);
  std::ifstream in(log_file(), std::ios::binary);
  std::deque<std::string> tail;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    tail.push_back(line);
    if (tail.size() > lines)
      tail.pop_front();
  }
  return std::vector<std::string>(tail.begin(), tail.end());
}

}  // namespace gufo::router::auth
