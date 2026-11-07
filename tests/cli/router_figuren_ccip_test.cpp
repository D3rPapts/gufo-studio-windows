// POST /v1/figuren "verfahren": "ccip": decision rule against the measured
// prototype (ccip_merkmale.py, log merkmale2_top3_cap0.05_M0.01.json and
// _M0.003.json of 07./08.11.2026; CCIP values as logged, rounded to 3 places).
// Written with AI assistance (Claude Code).
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "src/cli/serve/router/figuren_ccip.hpp"

namespace figuren = gufo::router::figuren;
using figuren::CcipCandidate;
using gufo::json::Value;

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

Value Features(bool moustache, bool beard, const char* glasses, const char* hair) {
  Value v = Value::object();
  v["schnurrbart"] = moustache;
  v["vollbart"] = beard;
  v["brille"] = glasses;
  v["haar"] = hair;
  return v;
}

/// Steckbrief features as the 27B read them in the prototype run.
std::map<std::string, Value> Figures() {
  std::map<std::string, Value> f;
  f["bernd"] = Features(false, false, "sonnenbrille", "gruenlich-grau");
  f["fabian"] = Features(true, true, "klar", "grau");
  f["holger"] = Features(false, false, "sonnenbrille", "grau");
  f["lars"] = Features(false, false, "keine", "grau");
  f["mayk"] = Features(true, false, "keine", "grau");
  return f;
}

/// All five figures; the three logged ones first, the other two farther away.
std::vector<CcipCandidate> Diffs(std::vector<CcipCandidate> logged) {
  const char* names[] = {"bernd", "fabian", "holger", "lars", "mayk"};
  std::vector<CcipCandidate> all;
  for (const char* n : names) {
    double d = 0.2;
    for (const auto& c : logged)
      if (c.name == n)
        d = c.diff;
    all.push_back({n, d});
  }
  return figuren::SortCandidates(all);
}

struct Case {
  const char* what;
  std::vector<CcipCandidate> logged;
  Value seen;
  const char* expect_m001;   // cap 0.05, M 0.01
  const char* expect_m0003;  // cap 0.05, M 0.003
};

}  // namespace

int main() {
  const auto figures = Figures();
  const std::vector<Case> cases = {
      {"lars, clear", {{"lars", 0.036}, {"holger", 0.063}, {"mayk", 0.080}},
       Features(false, false, "keine", "grau"), "lars", "lars"},
      {"bernd drawn, CCIP says mayk; only bernd fits but 0.088 >= cap",
       {{"mayk", 0.052}, {"bernd", 0.088}, {"lars", 0.089}},
       Features(false, false, "sonnenbrille", "gruenlich-grau"), "unbekannt",
       "unbekannt"},
      {"mayk: moustache removes bernd and holger",
       {{"bernd", 0.036}, {"mayk", 0.039}, {"holger", 0.040}},
       Features(true, false, "keine", "weiss"), "mayk", "mayk"},
      {"holger vs bernd 0.012/0.014: too close",
       {{"holger", 0.012}, {"bernd", 0.014}, {"mayk", 0.049}},
       Features(false, false, "sonnenbrille", "weiss"), "unbekannt", "unbekannt"},
      {"bernd/holger 0.026/0.034 (the one M 0.003 error)",
       {{"holger", 0.026}, {"bernd", 0.034}, {"mayk", 0.049}},
       Features(false, false, "sonnenbrille", "weiss"), "unbekannt", "holger"},
      {"statist person2: best 0.144",
       {{"lars", 0.144}, {"bernd", 0.180}, {"fabian", 0.182}},
       Features(false, false, "sonnenbrille", "dunkel"), "unbekannt", "unbekannt"},
      {"statist person4: only lars fits, 0.058 >= cap",
       {{"lars", 0.058}, {"holger", 0.084}, {"bernd", 0.088}},
       Features(false, false, "keine", "dunkel"), "unbekannt", "unbekannt"},
      {"statist person5 (mayk double, clear glasses): nobody fits",
       {{"mayk", 0.030}, {"holger", 0.058}, {"bernd", 0.061}},
       Features(true, false, "klar", "grau"), "unbekannt", "unbekannt"},
      {"statist person1 (badge FABIAN, sunglasses): fabian kept but 0.070",
       {{"lars", 0.053}, {"fabian", 0.070}, {"mayk", 0.076}},
       Features(true, true, "sonnenbrille", "grau"), "unbekannt", "unbekannt"},
  };
  for (const Case& c : cases) {
    const auto sorted = Diffs(c.logged);
    const auto a = figuren::DecideCcip(sorted, figures, &c.seen, 0.05, 0.01);
    const auto b = figuren::DecideCcip(sorted, figures, &c.seen, 0.05, 0.003);
    std::printf("%-62s M0.01 %-9s M0.003 %-9s | %s\n", c.what, a.name.c_str(),
                b.name.c_str(), a.begruendung.c_str());
    CHECK(a.name == c.expect_m001);
    CHECK(b.name == c.expect_m0003);
    CHECK(a.beste == c.logged[0].name);
    CHECK(a.zweitbeste == c.logged[1].name);
    CHECK(a.top.size() == 3);
  }

  // Contradiction rule details.
  Value fig = Features(true, false, "keine", "grau");
  CHECK(figuren::FeatureContradictions(fig, Features(true, false, "keine", "x")).empty());
  CHECK(figuren::FeatureContradictions(fig, Features(false, false, "keine", "x")) ==
        std::vector<std::string>{"schnurrbart"});
  // klar vs sonnenbrille is no contradiction; keine vs any glasses is.
  CHECK(figuren::FeatureContradictions(Features(false, false, "klar", "x"),
                                       Features(false, false, "sonnenbrille", "x"))
            .empty());
  CHECK(figuren::FeatureContradictions(fig, Features(true, false, "klar", "x")) ==
        std::vector<std::string>{"brille"});
  // A feature only counts when both sides have it.
  Value partial = Value::object();
  partial["brille"] = "sonnenbrille";
  CHECK(figuren::FeatureContradictions(fig, partial) ==
        std::vector<std::string>{"brille"});
  CHECK(figuren::FeatureContradictions(Value::object(), partial).empty());
  // An empty "brille" does not count (Python truthiness).
  Value empty_glasses = Value::object();
  empty_glasses["brille"] = "";
  CHECK(figuren::FeatureContradictions(fig, empty_glasses).empty());

  // Without features nothing is dropped; cap is strict, margin inclusive.
  {
    const auto sorted = figuren::SortCandidates({{"a", 0.03}, {"b", 0.04}, {"c", 0.2}});
    CHECK(figuren::DecideCcip(sorted, {}, nullptr, 0.05, 0.01).name == "a");
    CHECK(figuren::DecideCcip(sorted, {}, nullptr, 0.03, 0.01).name == "unbekannt");
    CHECK(figuren::DecideCcip(sorted, {}, nullptr, 0.05, 0.011).name == "unbekannt");
    const auto tie = figuren::SortCandidates({{"x", 0.02}, {"y", 0.02}});
    CHECK(tie[0].name == "x");
    CHECK(figuren::DecideCcip(tie, {}, nullptr, 0.05, 0.0).name == "x");
    CHECK(figuren::DecideCcip({}, {}, nullptr, 0.05, 0.01).name == "unbekannt");
  }
  // When the 27B is asked.
  {
    std::map<std::string, Value> profiles;
    profiles["lars"] = Features(false, false, "keine", "grau");
    CHECK(figuren::CcipNeedsFeatures(Diffs({{"lars", 0.03}}), profiles, 0.05));
    CHECK(!figuren::CcipNeedsFeatures(Diffs({{"lars", 0.06}}), profiles, 0.05));
    CHECK(!figuren::CcipNeedsFeatures(Diffs({{"lars", 0.03}}), {}, 0.05));
  }
  CHECK(figuren::CcipCropPrompt().find("Look only at the head in the centre") == 0);
  CHECK(figuren::CcipFigurePrompt("x").find("Character description: x\nExtract") == 0);
  std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
