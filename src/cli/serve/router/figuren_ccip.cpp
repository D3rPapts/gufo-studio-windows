// Decision rule of POST /v1/figuren "verfahren": "ccip" (see figuren_ccip.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/figuren_ccip.hpp"

#include <algorithm>
#include <cstdio>

namespace gufo::router::figuren {
namespace {

/// Python bool() of a JSON value.
bool PyTruthy(const json::Value& value) {
  if (value.is_null())
    return false;
  if (value.is_bool())
    return value.as_bool();
  if (value.is_number())
    return value.as_double() != 0;
  if (value.is_string())
    return !value.str().empty();
  return !value.empty();
}

bool IsNone(const json::Value& value) {
  return value.is_string() && value.str() == "keine";
}

std::string Fixed3(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.3f", value);
  return buffer;
}

std::string CandidateList(const std::vector<CcipCandidate>& list) {
  std::string out;
  for (const CcipCandidate& c : list) {
    if (!out.empty())
      out += ", ";
    out += c.name + " " + Fixed3(c.diff);
  }
  return out;
}

}  // namespace

std::string CcipFigurePrompt(const std::string& steckbrief) {
  return "Character description: " + steckbrief +
         "\nExtract only these features as JSON " + kCcipFeatureSchema +
         ". Answer only with JSON.";
}

std::string CcipCropPrompt() {
  return std::string(
             "Look only at the head in the centre of this anime image. Read "
             "these features as JSON ") +
         kCcipFeatureSchema +
         ". Only what is clearly visible. Answer only with JSON.";
}

std::vector<std::string> FeatureContradictions(const json::Value& figure,
                                               const json::Value& seen) {
  std::vector<std::string> bad;
  if (!figure.is_object() || !seen.is_object())
    return bad;
  for (const char* key : {"schnurrbart", "vollbart"}) {
    const json::Value* f = figure.find(key);
    const json::Value* s = seen.find(key);
    if (f != nullptr && s != nullptr && PyTruthy(*f) != PyTruthy(*s))
      bad.emplace_back(key);
  }
  const json::Value* gf = figure.find("brille");
  const json::Value* gs = seen.find("brille");
  if (gf != nullptr && gs != nullptr && PyTruthy(*gf) && PyTruthy(*gs) &&
      IsNone(*gf) != IsNone(*gs))
    bad.emplace_back("brille");
  return bad;
}

std::string FeatureText(const json::Value& seen) {
  if (!seen.is_object() || seen.empty())
    return "keine gelesen";
  std::vector<std::string> parts;
  if (const json::Value* v = seen.find("schnurrbart"))
    parts.push_back(PyTruthy(*v) ? "Schnurrbart" : "kein Schnurrbart");
  if (const json::Value* v = seen.find("vollbart"))
    parts.push_back(PyTruthy(*v) ? "Vollbart" : "kein Vollbart");
  if (const json::Value* v = seen.find("brille")) {
    const std::string kind = v->is_string() ? v->str() : v->dump();
    parts.push_back(kind == "keine"          ? "keine Brille"
                    : kind == "sonnenbrille" ? "Sonnenbrille"
                    : kind == "klar"         ? "Brille"
                                             : "Brille " + kind);
  }
  if (const json::Value* v = seen.find("haar"))
    parts.push_back("Haar " + (v->is_string() ? v->str() : v->dump()));
  std::string out;
  for (const std::string& p : parts) {
    if (!out.empty())
      out += ", ";
    out += p;
  }
  return out.empty() ? "keine gelesen" : out;
}

std::vector<CcipCandidate> SortCandidates(std::vector<CcipCandidate> diffs) {
  std::stable_sort(diffs.begin(), diffs.end(),
                   [](const CcipCandidate& a, const CcipCandidate& b) {
                     return a.diff < b.diff;
                   });
  return diffs;
}

bool CcipNeedsFeatures(const std::vector<CcipCandidate>& sorted,
                       const std::map<std::string, json::Value>& figure_features,
                       double cap, std::size_t top) {
  if (sorted.empty() || !(sorted.front().diff < cap))
    return false;
  for (std::size_t i = 0; i < sorted.size() && i < top; ++i) {
    const auto it = figure_features.find(sorted[i].name);
    if (it != figure_features.end() && it->second.is_object() &&
        !it->second.empty())
      return true;
  }
  return false;
}

CcipDecision DecideCcip(const std::vector<CcipCandidate>& sorted,
                        const std::map<std::string, json::Value>& figure_features,
                        const json::Value* seen, double cap, double margin,
                        std::size_t top) {
  CcipDecision d;
  if (sorted.empty()) {
    d.begruendung = "keine Figur";
    return d;
  }
  d.beste = sorted[0].name;
  if (sorted.size() > 1) {
    d.zweitbeste = sorted[1].name;
    d.abstand = sorted[1].diff - sorted[0].diff;
  }
  std::vector<std::string> excluded;
  for (std::size_t i = 0; i < sorted.size() && i < top; ++i) {
    d.top.push_back(sorted[i]);
    std::vector<std::string> bad;
    if (seen != nullptr) {
      const auto it = figure_features.find(sorted[i].name);
      if (it != figure_features.end())
        bad = FeatureContradictions(it->second, *seen);
    }
    if (bad.empty()) {
      d.kept.push_back(sorted[i]);
    } else {
      std::string joined;
      for (const std::string& b : bad)
        joined += (joined.empty() ? "" : ", ") + b;
      excluded.push_back(sorted[i].name + " (" + joined + ")");
    }
  }
  std::string why = "CCIP " + CandidateList(d.top);
  if (seen != nullptr) {
    why += "; Merkmale: " + FeatureText(*seen);
    if (!excluded.empty()) {
      why += "; widerspricht: ";
      for (std::size_t i = 0; i < excluded.size(); ++i)
        why += (i ? ", " : "") + excluded[i];
    }
    std::string kept;
    for (const CcipCandidate& c : d.kept)
      kept += (kept.empty() ? "" : ", ") + c.name;
    why += "; passt: " + (kept.empty() ? std::string("keiner") : kept);
  } else {
    why += "; 27B nicht gefragt";
  }
  if (d.kept.empty()) {
    why += " -> unbekannt (keine passende Figur)";
  } else if (!(d.kept[0].diff < cap)) {
    why += " -> unbekannt (" + d.kept[0].name + " " + Fixed3(d.kept[0].diff) +
           " nicht unter " + Fixed3(cap) + ")";
  } else if (d.kept.size() == 1) {
    d.name = d.kept[0].name;
    why += " -> " + d.name + " (einzige passende Figur)";
  } else {
    const double gap = d.kept[1].diff - d.kept[0].diff;
    if (gap >= margin) {
      d.name = d.kept[0].name;
      why += " -> " + d.name + " (Abstand " + Fixed3(gap) + " zu " +
             d.kept[1].name + ")";
    } else {
      why += " -> unbekannt (" + d.kept[0].name + " und " + d.kept[1].name +
             " zu nah, Abstand " + Fixed3(gap) + " < " + Fixed3(margin) + ")";
    }
  }
  d.begruendung = std::move(why);
  return d;
}

}  // namespace gufo::router::figuren
