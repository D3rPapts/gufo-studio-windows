#include "src/cli/serve/router/names.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::router::names {

namespace {

constexpr int kMaxCount = 100;

server::HttpResponse JsonAnswer(int status, std::string_view reason,
                        const json::Value& body) {
  return server::HttpResponse{.status = status,
                      .reason = std::string(reason),
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"},
                                  {"Cache-Control", "no-store"}}};
}

server::HttpResponse Error(int status, std::string_view reason,
                   std::string_view message, std::string_view code) {
  json::Value detail = json::Value::object();
  detail["message"] = std::string(message);
  detail["type"] = "invalid_request_error";
  detail["code"] = std::string(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return JsonAnswer(status, reason, body);
}

// SplitMix64: small, fast and identical on every platform and compiler.
struct Random {
  std::uint64_t state;
  std::uint64_t Next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  std::size_t Below(std::size_t n) { return static_cast<std::size_t>(Next() % n); }
};

std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(c < 128 ? std::tolower(c) : c);
  });
  return text;
}

bool ParseUnsigned(const std::string& text, std::uint64_t* value) {
  if (text.empty() || text.size() > 20)
    return false;
  std::uint64_t v = 0;
  for (const char c : text) {
    if (c < '0' || c > '9')
      return false;
    v = v * 10 + static_cast<std::uint64_t>(c - '0');
  }
  *value = v;
  return true;
}

}  // namespace

NameService::NameService(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read names file " + file.string());
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF)
      line.erase(0, 3);  // UTF-8 BOM
    std::vector<std::string> cols;
    std::size_t start = 0;
    for (;;) {
      const std::size_t tab = line.find('\t', start);
      cols.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
      if (tab == std::string::npos)
        break;
      start = tab + 1;
    }
    if (cols.size() < 3 || cols[1].empty() || cols[2].empty())
      continue;
    NameLists& lists = origins_[Lower(cols[2])];
    const std::string gender = cols.size() > 3 ? Lower(cols[3]) : "";
    if (cols[0] == "nach") {
      lists.family.push_back(cols[1]);
    } else if (cols[0] == "vor" && gender == "m") {
      lists.given_m.push_back(cols[1]);
    } else if (cols[0] == "vor" && (gender == "w" || gender == "f")) {
      lists.given_f.push_back(cols[1]);
    } else {
      continue;
    }
    ++size_;
  }
  std::erase_if(origins_, [](const auto& entry) {
    return entry.second.family.empty() && entry.second.given_m.empty() &&
           entry.second.given_f.empty();
  });
  if (std::none_of(origins_.begin(), origins_.end(),
                   [](const auto& entry) { return !entry.second.family.empty(); }))
    throw std::runtime_error("names file has no family names: " + file.string());
  // Licence note: the whole LIZENZ.txt/LICENSE.txt on one line (CR/LF -> space).
  for (const char* name : {"LIZENZ.txt", "LICENSE.txt"}) {
    std::ifstream licence(file.parent_path() / name, std::ios::binary);
    if (!licence)
      continue;
    std::string text((std::istreambuf_iterator<char>(licence)), std::istreambuf_iterator<char>());
    for (char& c : text)
      if (c == '\r' || c == '\n')
        c = ' ';
    std::string joined;
    for (const char c : text)
      if (c != ' ' || (!joined.empty() && joined.back() != ' '))
        joined += c;
    while (!joined.empty() && joined.back() == ' ')
      joined.pop_back();
    license_ = joined.substr(0, 600);
    break;
  }
  if (license_.empty())
    license_ = "see the names file";
}

std::optional<server::HttpResponse> NameService::Handle(const server::HttpRequest& request) const {
  if (request.path != "/v1/names" && request.path != "/v1/names/origins")
    return std::nullopt;
  if (request.method != "GET")
    return Error(405, "Method Not Allowed", "only GET", "method_not_allowed");

  if (request.path == "/v1/names/origins") {
    json::Value data = json::Value::array();
    for (const auto& [origin, lists] : origins_) {
      json::Value item = json::Value::object();
      item["origin"] = origin;
      item["family"] = static_cast<long long>(lists.family.size());
      item["given_m"] = static_cast<long long>(lists.given_m.size());
      item["given_f"] = static_cast<long long>(lists.given_f.size());
      data.push_back(std::move(item));
    }
    json::Value body = json::Value::object();
    body["object"] = "list";
    body["data"] = std::move(data);
    body["license"] = license_;
    return JsonAnswer(200, "OK", body);
  }

  std::uint64_t count = 1;
  if (const std::string text = request.query_param("count"); !text.empty()) {
    if (!ParseUnsigned(text, &count) || count < 1 || count > kMaxCount)
      return Error(400, "Bad Request", "count must be 1..100", "invalid_count");
  }
  std::uint64_t seed = 0;
  const std::string seed_text = request.query_param("seed");
  if (!seed_text.empty()) {
    if (!ParseUnsigned(seed_text, &seed))
      return Error(400, "Bad Request", "seed must be a non-negative integer", "invalid_seed");
  } else {
    seed = static_cast<std::uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
  }
  seed &= (1ULL << 53) - 1;  // JSON numbers are doubles: the reported seed reproduces the list
  const std::string origin = Lower(request.query_param("origin"));
  if (!origin.empty() && !origins_.contains(origin))
    return Error(400, "Bad Request", "unknown origin '" + origin + "' (GET /v1/names/origins)",
                 "invalid_origin");
  std::string gender = Lower(request.query_param("gender"));
  if (gender == "w")
    gender = "f";
  if (!gender.empty() && gender != "m" && gender != "f")
    return Error(400, "Bad Request", "gender must be m or f", "invalid_gender");

  std::vector<const std::string*> with_family;  // origins to draw from when none is given
  for (const auto& [name, lists] : origins_)
    if (!lists.family.empty())
      with_family.push_back(&name);
  const auto english = origins_.find("english");

  Random random{seed};
  json::Value data = json::Value::array();
  for (std::uint64_t i = 0; i < count; ++i) {
    const std::string& pick = origin.empty() ? *with_family[random.Below(with_family.size())] : origin;
    const NameLists& lists = origins_.at(pick);
    const std::string sex = gender.empty() ? (random.Below(2) == 0 ? "m" : "f") : gender;
    const std::vector<std::string>* given = sex == "m" ? &lists.given_m : &lists.given_f;
    if (given->empty() && english != origins_.end())
      given = sex == "m" ? &english->second.given_m : &english->second.given_f;
    json::Value item = json::Value::object();
    item["given"] = given->empty() ? std::string() : (*given)[random.Below(given->size())];
    item["family"] = lists.family.empty() ? std::string() : lists.family[random.Below(lists.family.size())];
    item["origin"] = pick;
    item["gender"] = sex;
    data.push_back(std::move(item));
  }
  json::Value body = json::Value::object();
  body["object"] = "list";
  body["seed"] = static_cast<long long>(seed);
  body["data"] = std::move(data);
  body["license"] = license_;
  return JsonAnswer(200, "OK", body);
}

}  // namespace gufo::router::names
