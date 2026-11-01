#ifndef GUFO_CLI_SERVE_ROUTER_NAMES_HPP_
#define GUFO_CLI_SERVE_ROUTER_NAMES_HPP_

// Name service for the router (`gufo router --names-file FILE`): random
// person names by origin and gender from a word list, without a model.
//
//   GET /v1/names[?count=N&origin=O&gender=m|f&seed=S]
//       -> {"object": "list", "seed": S, "data": [{"given", "family",
//          "origin", "gender"}], "license": "..."}
//   GET /v1/names/origins
//       -> {"object": "list", "data": [{"origin", "family", "given_m",
//          "given_f"}]}
//
// FILE is UTF-8, one name per line: kind<TAB>name<TAB>origin<TAB>gender with
// kind "nach" (family name) or "vor" (given name) and gender "m" or "w" for
// given names. A LIZENZ.txt/LICENSE.txt beside the file is reported as the
// licence line. The same seed and query give the same names on every
// platform (own 64-bit generator, no std distributions). Origins without
// given names take them from "english".

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"

namespace gufo::router::names {

struct NameLists {
  std::vector<std::string> family;
  std::vector<std::string> given_m;
  std::vector<std::string> given_f;
};

class NameService {
 public:
  /// Reads FILE; throws std::runtime_error when it cannot be read or holds no
  /// family names.
  explicit NameService(const std::filesystem::path& file);

  /// Answers /v1/names and /v1/names/origins, nullopt for other paths.
  std::optional<server::HttpResponse> Handle(const server::HttpRequest& request) const;

  std::size_t Size() const { return size_; }
  const std::map<std::string, NameLists>& Origins() const { return origins_; }

 private:
  std::map<std::string, NameLists> origins_;
  std::string license_;
  std::size_t size_ = 0;
};

}  // namespace gufo::router::names

#endif  // GUFO_CLI_SERVE_ROUTER_NAMES_HPP_
