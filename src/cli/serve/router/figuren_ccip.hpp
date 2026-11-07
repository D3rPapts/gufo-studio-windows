#ifndef GUFO_CLI_SERVE_ROUTER_FIGUREN_CCIP_HPP_
#define GUFO_CLI_SERVE_ROUTER_FIGUREN_CCIP_HPP_

// Decision rule of POST /v1/figuren with "verfahren": "ccip" (see figuren.hpp),
// kept free of images and models so it can be tested on its own. Port of the
// measured prototype ccip_merkmale.py (07./08.11.2026: top 3, cap 0.05,
// M 0.01 -> 216 of 286 figures right, 0 wrong, 4 of 4 statists unknown):
//   - CCIP diff per figure (min over its references and gallery crops),
//     sorted ascending (ties keep reference order);
//   - the 27B reads hard features of the head crop; a candidate among the
//     best `top` is dropped when its Steckbrief features contradict them
//     (moustache, full beard, or glasses "keine" vs any glasses);
//   - the best remaining candidate is assigned when its diff < cap and it is
//     the only one left or the next one is at least `margin` farther;
//     otherwise "unbekannt". Never by position or badge.
// Written with AI assistance (Claude Code).

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router::figuren {

inline constexpr double kCcipCap = 0.05;
inline constexpr double kCcipMargin = 0.01;
inline constexpr std::size_t kCcipTop = 3;
inline constexpr const char* kUnknownFigure = "unbekannt";

/// Feature schema the 27B fills (ccip_merkmale.SCHEMA).
inline constexpr const char* kCcipFeatureSchema =
    "{\"schnurrbart\": true|false, \"vollbart\": true|false, \"brille\": "
    "\"keine\"|\"klar\"|\"sonnenbrille\", \"haar\": \"weiss\"|\"grau\"|"
    "\"gruenlich-grau\"|\"braun\"|\"dunkel\"|\"blond\"|\"andere\"}";

/// Prompt that reads the features from a Steckbrief text.
std::string CcipFigurePrompt(const std::string& steckbrief);
/// Prompt (after the head-crop image) that reads the features of a head.
std::string CcipCropPrompt();

struct CcipCandidate {
  std::string name;
  double diff{1.0};
};

/// ccip_merkmale.contradicts: names of the hard features in which `figure`
/// (from the Steckbrief) and `seen` (from the head crop) disagree. A feature
/// only counts when both sides have it; "schnurrbart"/"vollbart" compare by
/// truthiness, "brille" only as none vs any glasses.
std::vector<std::string> FeatureContradictions(const json::Value& figure,
                                               const json::Value& seen);

/// Short German text of read features ("kein Schnurrbart, Sonnenbrille, ...").
std::string FeatureText(const json::Value& seen);

/// Ascending by diff; equal diffs keep the input (reference) order.
std::vector<CcipCandidate> SortCandidates(std::vector<CcipCandidate> diffs);

/// Whether asking the 27B can change the result: the best diff is below
/// `cap` and at least one of the best `top` figures has Steckbrief features.
bool CcipNeedsFeatures(const std::vector<CcipCandidate>& sorted,
                       const std::map<std::string, json::Value>& figure_features,
                       double cap, std::size_t top = kCcipTop);

struct CcipDecision {
  std::string name = kUnknownFigure;
  std::vector<CcipCandidate> top;   // the best `top` by diff
  std::vector<CcipCandidate> kept;  // top without contradictions
  std::string beste;
  std::string zweitbeste;
  double abstand{0};  // second best - best diff (all figures)
  std::string begruendung;
};

/// The decision for one person. `sorted` comes from SortCandidates;
/// `seen` is null when the 27B was not asked (then nothing is dropped).
/// Figures without an entry (or with an empty object) in `figure_features`
/// are never dropped.
CcipDecision DecideCcip(const std::vector<CcipCandidate>& sorted,
                        const std::map<std::string, json::Value>& figure_features,
                        const json::Value* seen, double cap, double margin,
                        std::size_t top = kCcipTop);

}  // namespace gufo::router::figuren

#endif  // GUFO_CLI_SERVE_ROUTER_FIGUREN_CCIP_HPP_
