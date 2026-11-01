// Name service (router --names-file): parsing, filters, seed reproducibility,
// errors. Written with AI assistance (Claude Code).
#undef NDEBUG
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "src/cli/serve/router/names.hpp"
#include "src/core/json.hpp"

namespace fs = std::filesystem;
using gufo::server::HttpRequest;
using gufo::router::names::NameService;

namespace {

HttpRequest Get(const std::string& path, const std::string& query = "") {
  HttpRequest request;
  request.method = "GET";
  request.path = path;
  request.query = query;
  return request;
}

}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() / "gufo_router_names_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    std::ofstream out(dir / "namen.tsv", std::ios::binary);
    out << "\xEF\xBB\xBF" << "nach\tYilmaz\tturkish\t\r\n"
        << "nach\tKaya\tturkish\t\n"
        << "vor\tAyşe\tturkish\tw\n"
        << "vor\tMehmet\tturkish\tm\n"
        << "nach\tIvanov\trussian\t\n"
        << "vor\tJohn\tenglish\tm\n"
        << "vor\tMary\tenglish\tw\n"
        << "nach\tSmith\tenglish\t\n"
        << "kaputt\n";
    std::ofstream licence(dir / "LIZENZ.txt");
    licence << "Wiktionary, CC BY-SA 4.0\n";
  }
  const NameService service(dir / "namen.tsv");
  assert(service.Size() == 8);
  assert(service.Origins().size() == 3);

  // Not our path: nullopt.
  assert(!service.Handle(Get("/v1/models")).has_value());

  // Origins list.
  auto origins = service.Handle(Get("/v1/names/origins"));
  assert(origins && origins->status == 200);
  auto list = gufo::json::parse(origins->body);
  assert(list["data"].items().size() == 3);
  assert(list["license"].str() == "Wiktionary, CC BY-SA 4.0");

  // Filters and seed: same seed, same names.
  auto a = service.Handle(Get("/v1/names", "count=5&origin=turkish&gender=f&seed=42"));
  auto b = service.Handle(Get("/v1/names", "count=5&origin=turkish&gender=w&seed=42"));
  assert(a && b && a->status == 200 && a->body == b->body);
  auto names = gufo::json::parse(a->body);
  assert(names["data"].items().size() == 5);
  for (const auto& item : names["data"].items()) {
    assert(item.find("origin")->str() == "turkish");
    assert(item.find("gender")->str() == "f");
    assert(item.find("given")->str() == "Ayşe");
    assert(item.find("family")->str() == "Yilmaz" || item.find("family")->str() == "Kaya");
  }
  assert(names["seed"].as_double() == 42);

  // An origin without given names borrows them from english.
  auto ru = service.Handle(Get("/v1/names", "origin=russian&gender=m&seed=1"));
  assert(ru && ru->status == 200);
  { auto ru_body = gufo::json::parse(ru->body); assert(ru_body["data"].items()[0].find("given")->str() == "John"); }

  // Errors.
  assert(service.Handle(Get("/v1/names", "count=0"))->status == 400);
  assert(service.Handle(Get("/v1/names", "count=101"))->status == 400);
  assert(service.Handle(Get("/v1/names", "origin=klingon"))->status == 400);
  assert(service.Handle(Get("/v1/names", "gender=x"))->status == 400);
  assert(service.Handle(Get("/v1/names", "seed=-3"))->status == 400);
  HttpRequest post = Get("/v1/names");
  post.method = "POST";
  assert(service.Handle(post)->status == 405);

  // A file without family names is refused.
  {
    std::ofstream out(dir / "leer.tsv");
    out << "vor\tJohn\tenglish\tm\n";
  }
  bool threw = false;
  try {
    NameService empty(dir / "leer.tsv");
  } catch (const std::exception&) {
    threw = true;
  }
  assert(threw);

  fs::remove_all(dir);
  std::cout << "router_names_test: all passed\n";
  return 0;
}
