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
//      "nur_kostuem": false,                       optional
//      "modell": "qwen3.8-27b"}                    optional
//     A referenz may also carry "front": "<data URL>" (front reference of the
//     figure); then its costume is checked as well. With nur_kostuem the head
//     comparison is skipped (boxes + costume only) and "kopf" is optional.
//   -> {"figuren": [{"name", "box": [x0,y0,x1,y1], "kopf_box": [...],
//        "schild", "zuordnung": "schild"|"reihenfolge", "aehnlichkeit": 0..1,
//        "abweichungen": [...], "nachgeprueft": bool[, "schnell": 0..1]
//        [, "kostuem_ok": bool, "kostuem_abweichungen": [...],
//           "kostuem_teile": [{"teil", "soll", "ist", "ok"[, "regel"]}],
//           "steckbrief_widerspricht_referenz": [...]]}],
//       "sekunden": n}
//   (aehnlichkeit, abweichungen, nachgeprueft and schnell are left out with
//   nur_kostuem; the kostuem fields appear only for figures with a front.)
//
// Steps: (1) one vision chat lists every person with person_box, head_box
// (0..1000 relative) and badge, thinking off; (2) badge assignment (3-letter
// prefix, case-insensitive, letters only), else left to right in reference
// order; (3) per figure one compare chat (reference head + head crop padded by
// a third of the box, both at most 768 px), thinking off; (4) figures below
// `schwelle` are compared again with thinking and that verdict counts.
// Every chat goes through the router's own /v1/chat/completions (the fetch
// below), so model loading, the memory guard and busy handling apply as for
// any client. An answer without parsable JSON gets one repair pass
// (RepairJson), then one new request at temperature 0.3, before the 502.
// Costume (figuren.py kostueme): per figure with a front reference one chat
// with thinking (max_tokens 14000) compares the front reference with the
// head-to-belt crop of the frame (shirt, tie, glasses); shirt and tie items
// whose wanted and seen colour name different colour families are then
// marked wrong even when the model passed them.
//
// Opt-in method "verfahren": "ccip" (1.4.5; agreed with Regie #56-496/#56-498;
// spec = Python prototype ccip_merkmale.py). Without "verfahren" (or with
// "llm") everything above applies unchanged.
//   POST /v1/figuren
//     {"verfahren": "ccip",
//      "bild": "<frame>",
//      "referenzen": [{"name": "Lars", "kopf": "<image>", "front": "<image>",
//                      "bilder": ["<image>", ...]}, ...],   front/bilder optional
//      "galerie": [{"name": "Lars", "bild": "<head crop from another shot>"}],
//      "steckbrief": {"Lars": "appearance text"},  optional
//      "ccip_cap": 0.05, "ccip_abstand": 0.01,       optional
//      "modell": "qwen3.8-27b"}                      optional
//   -> {"verfahren": "ccip", "figuren": [{"name": figure | "unbekannt",
//        "box": face box [x0,y0,x1,y1], "kopf_box": head crop, "gesicht": score,
//        "beste", "zweitbeste", "abstand", "diff": {figure: 0..1},
//        "kandidaten": [top 3 names], "passend": [kept names],
//        "merkmale": {...} | null, "llm_gefragt": bool, "begruendung"}],
//       "sekunden": n}
//   People come from the anime face detector (YOLOv8n, letterbox 640, conf
//   0.25, NMS 0.7), sorted left to right; the head crop is the face box grown
//   to a square of 1.6 x its longer side. CCIP (CAFormer, 768 features) per
//   head; diff = clip((1 - cos) / 2, 0, 1), per figure the minimum over every
//   reference image (whole image AND its own centre-face head crop, for kopf,
//   front and bilder) and every gallery image (centre-face head crop, the
//   whole image when no face is found). The decision is DecideCcip
//   (figuren_ccip.hpp): top 3 by diff, the 27B (own /v1/chat/completions,
//   thinking off, temperature 0, head crop at most 512 px) reads moustache,
//   full beard, glasses and hair; candidates whose Steckbrief features (read
//   once per text by the 27B, cached in the process) contradict are dropped;
//   the best kept one is assigned when diff < ccip_cap and it is alone or the
//   next kept one is >= ccip_abstand farther, else "unbekannt". The 27B is
//   only asked when it can change the result (best diff < cap and a top-3
//   figure has Steckbrief features); otherwise merkmale is null and
//   llm_gefragt false. Never assigned by position or badge. Models: router
//   options --ccip-model-dir / --ccip-face-dir (converted safetensors, see
//   src/models/ccip), loaded on the first ccip request and kept (~165 MB host);
//   503 "ccip_unavailable" when missing. Embeddings of reference and gallery
//   images are cached by SHA-256 of the image string, so only the first
//   request pays for them (~1.1 s CPU per reference image). CPU work is
//   serialised over one lock and uses --ccip-threads OpenMP threads.

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
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

/// figuren.py repair_json: number forms the model sometimes writes that JSON
/// rejects (lauf_170: "[75, 3, 702, .948]"). After "[", ",", ":" or
/// whitespace a leading dot gets a zero (".948" -> "0.948", "-.5" -> "-0.5")
/// and a plus sign is dropped ("+3" -> "3"); a dot after a digit and before
/// whitespace, ",", "]" or "}" is dropped ("1." -> "1"). Everything else is
/// kept. Whitespace is Python's str \s; digits are ASCII 0-9.
std::string RepairJson(std::string_view text);

/// Relative 0..1000 box to pixels, rounding half to even like Python round().
/// Nullopt unless `box` is an array of four numbers (or numeric strings).
/// As in figuren.py to_px: when the largest value is above 2, single values
/// strictly between 0 and 1 are fractions of 1000 and scaled up (".948" ->
/// 948); when no value is above 1, the whole box is 0..1 and scaled by 1000.
std::optional<Box> ToPixels(const json::Value& box, int width, int height);

/// Head crop rectangle (x0, y0, x1, y1, exclusive end): the head box padded
/// by a third of its size (floor division) on every side, clamped to the
/// image, at least one pixel.
Box HeadCrop(const Box& head, int width, int height);

/// Torso crop rectangle (figuren.py torso_crop): person box width, from the
/// head top down to the head bottom plus 3.2 head heights (truncated), at
/// most the person box bottom but at least 8 px below the head; clamped to
/// the image, at least one pixel.
Box TorsoCrop(const Box& box, const Box& kopf_box, int width, int height);

/// Colour family of a colour word (figuren.py FAMILIEN/familie): the first
/// family in table order with a word contained in the lowercased text.
std::optional<std::string> ColourFamily(std::string_view text);

/// figuren.py nachpruefen: shirt ("hemd", "shirt") and tie ("krawatte",
/// "tie") items whose soll and ist name different colour families get
/// ok = false and regel = "Farbfamilie a != b" unless already wrong. Non-object
/// entries are dropped.
json::Value CheckCostumeItems(const json::Value& items);

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
  /// "verfahren": "ccip": dirs with ccip_caformer.{safetensors,json} and
  /// face_yolov8n.{safetensors,json}; OpenMP threads for the CPU work.
  std::filesystem::path ccip_model_dir;
  std::filesystem::path ccip_face_dir;
  int ccip_threads = 8;
};

struct CcipState;

class FigurenService {
public:
  explicit FigurenService(Options options);
  ~FigurenService();
  /// Answers POST /v1/figuren, nullopt for other paths.
  std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& request) const;

private:
  std::optional<server::HttpResponse> HandleCcip(
      const json::Value& body,
      std::chrono::steady_clock::time_point started) const;

  Options options_;
  std::shared_ptr<CcipState> ccip_;
};

}  // namespace gufo::router::figuren

#endif  // GUFO_CLI_SERVE_ROUTER_FIGUREN_HPP_
