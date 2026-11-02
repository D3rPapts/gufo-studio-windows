#ifndef GUFO_CLI_SERVE_ROUTER_FIGUREN_HPP_
#define GUFO_CLI_SERVE_ROUTER_FIGUREN_HPP_

// Figure check for film keyframes (`POST /v1/figuren`): finds the people in a
// frame, assigns them to the film's figures and rates how closely each head
// matches that figure's head reference. Port of the Python prototype
// figuren.py, mode --zweistufig.
//
//   POST /v1/figuren
//     {"bild": "<data URL or base64 PNG/JPEG/WebP>",
//      "referenzen": [{"name": "Lars", "kopf": "<data URL>"}, ...],
//      "steckbrief": {"Lars": "short profile"},   optional
//      "schwelle": 0.6,                            optional
//      "modell": "qwen3.8-27b"}                    optional
//   -> {"figuren": [{"name", "box": [x0,y0,x1,y1], "kopf_box": [...],
//        "schild", "zuordnung": "schild"|"reihenfolge", "aehnlichkeit": 0..1,
//        "abweichungen": [...], "nachgeprueft": bool[, "schnell": 0..1]}],
//       "sekunden": n}
//
// Steps: (1) one vision chat lists every person with person_box, head_box
// (0..1000 relative) and badge, thinking off; (2) badge assignment (3-letter
// prefix, case-insensitive, letters only), else left to right in reference
// order; (3) per figure one compare chat (reference head + head crop padded by
// a third of the box, both at most 768 px), thinking off; (4) figures below
// `schwelle` are compared again with thinking and that verdict counts.
// Every chat goes through the router's own /v1/chat/completions (the fetch
// below), so model loading, the memory guard and busy handling apply as for
// any client.

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/core/json.hpp"

namespace gufo::router::figuren {

using Box = std::array<int, 4>;

/// Removes <think>...</think> blocks; a lone closing </think> (template that
/// opened the block in the prompt) drops everything before it.
std::string StripThinking(std::string_view text);

/// The outermost JSON array (`want_array`) or object after the last closing
/// bracket of the answer, after any thinking (figuren.py json_from).
std::optional<json::Value> JsonFrom(std::string_view text, bool want_array);

/// Relative 0..1000 box to pixels, rounding half to even like Python round().
/// Nullopt unless `box` is an array of four numbers (or numeric strings).
std::optional<Box> ToPixels(const json::Value& box, int width, int height);

/// Head crop rectangle (x0, y0, x1, y1, exclusive end): the head box padded
/// by a third of its size (floor division) on every side, clamped to the
/// image, at least one pixel.
Box HeadCrop(const Box& head, int width, int height);

/// Longer side scaled down to `max_side` (never up), aspect kept.
std::pair<int, int> FitSize(int width, int height, int max_side);

struct Person {
  Box box{};
  Box kopf_box{};
  std::string schild;
};

struct Assignment {
  std::string name;
  std::size_t person{0};  // index into the people list
  bool by_badge{false};
};

/// Badge first (left to right; prefix match either way over three letters
/// tolerates misspelt badges), then the remaining people left to right in the
/// order of `names`. Badge hits come first in the result, as in figuren.py.
std::vector<Assignment> Assign(const std::vector<Person>& people,
                               const std::vector<std::string>& names);

struct Options {
  /// Sends one request to the router's own model API.
  comfy::Fetch fetch;
  std::string default_model = "qwen3.8-27b";
};

class FigurenService {
public:
  explicit FigurenService(Options options);
  /// Answers POST /v1/figuren, nullopt for other paths.
  std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& request) const;

private:
  Options options_;
};

}  // namespace gufo::router::figuren

#endif  // GUFO_CLI_SERVE_ROUTER_FIGUREN_HPP_
