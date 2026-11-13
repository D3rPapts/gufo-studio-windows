// Written with AI assistance (Claude Code).
#include "indextts/ext_request.hpp"

#include <gufo/ext/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>

namespace gufo::models::indextts {
namespace {

namespace json = gufo::json;

constexpr const char* kPrefix = "IndexTTS instructions: ";

bool Fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

bool Space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

std::string_view Trim(std::string_view text) {
  while (!text.empty() && Space(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && Space(text.back())) {
    text.remove_suffix(1);
  }
  return text;
}

std::string LowerAscii(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool IdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

enum class Key { kEmotion, kAlpha, kUnknown };

Key KeyOf(std::string_view key) {
  const std::string k = LowerAscii(Trim(key));
  if (k == "emotion" || k == "emo" || k == "emo_vector") {
    return Key::kEmotion;
  }
  if (k == "emo_alpha" || k == "alpha") {
    return Key::kAlpha;
  }
  return Key::kUnknown;
}

constexpr const char* kKnownKeys = "emotion, emo_alpha";

// ParseEmotionSpec with the instructions prefix on errors; an empty spec is
// an error here (the key was given without a value).
bool SetEmotion(const std::string& spec, InstructControls* controls, std::string* error) {
  if (controls->emotion.has_value()) {
    return Fail(error, std::string(kPrefix) + "emotion given twice");
  }
  if (Trim(spec).empty()) {
    return Fail(error, std::string(kPrefix) + "empty emotion");
  }
  std::string parse_error;
  if (!ParseEmotionSpec(spec, &controls->emotion, &parse_error)) {
    return Fail(error, std::string(kPrefix) + parse_error);
  }
  return true;
}

bool SetAlpha(double value, InstructControls* controls, std::string* error) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    return Fail(error, std::string(kPrefix) + "emo_alpha must be a number between 0 and 1");
  }
  controls->emo_alpha = static_cast<float>(value);
  return true;
}

bool ParseAlphaText(std::string_view text, InstructControls* controls, std::string* error) {
  const std::string value(Trim(text));
  char* end = nullptr;
  const double d = value.empty() ? 0.0 : std::strtod(value.c_str(), &end);
  if (value.empty() || end == nullptr || *end != '\0') {
    return Fail(error, std::string(kPrefix) + "emo_alpha must be a number between 0 and 1");
  }
  return SetAlpha(d, controls, error);
}

// JSON array of exactly 8 numbers -> "a,b,...,h" (validated by ParseEmotionSpec).
bool VectorSpec(const json::Value& value, std::string* spec, std::string* error) {
  if (!value.is_array() || value.size() != 8) {
    return Fail(error, std::string(kPrefix) + "the emotion vector must have 8 numbers");
  }
  spec->clear();
  for (const json::Value& item : value.items()) {
    if (!item.is_number()) {
      return Fail(error, std::string(kPrefix) + "the emotion vector must have 8 numbers");
    }
    *spec += (spec->empty() ? "" : ",") + std::to_string(item.as_double());
  }
  return true;
}

bool ParseJson(std::string_view text, InstructControls* controls, std::string* error) {
  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& e) {
    return Fail(error, std::string(kPrefix) + "invalid JSON: " + e.what());
  }
  std::string spec;
  if (root.is_array()) {
    return VectorSpec(root, &spec, error) && SetEmotion(spec, controls, error);
  }
  if (!root.is_object()) {
    return Fail(error, std::string(kPrefix) + "JSON must be an object or an array of 8 numbers");
  }
  std::string named;  // {"Wut": 0.6, "Angst": 0.2}
  for (const auto& [key, value] : root.members()) {
    switch (KeyOf(key)) {
      case Key::kEmotion:
        if (value.is_string()) {
          spec = value.str();
        } else if (value.is_array()) {
          if (!VectorSpec(value, &spec, error)) {
            return false;
          }
        } else if (value.is_object() && !value.empty()) {
          spec.clear();
          for (const auto& [name, weight] : value.members()) {
            if (!weight.is_number()) {
              return Fail(error, std::string(kPrefix) + "emotion weight for '" + name +
                                     "' must be a number");
            }
            spec += (spec.empty() ? "" : ",") + name + ":" + std::to_string(weight.as_double());
          }
        } else {
          return Fail(error, std::string(kPrefix) +
                                 "'emotion' must be a name, an object of name weights or an "
                                 "array of 8 numbers");
        }
        if (!SetEmotion(spec, controls, error)) {
          return false;
        }
        break;
      case Key::kAlpha:
        if (!value.is_number()) {
          return Fail(error, std::string(kPrefix) + "emo_alpha must be a number between 0 and 1");
        }
        if (!SetAlpha(value.as_double(), controls, error)) {
          return false;
        }
        break;
      case Key::kUnknown:
        if (EmotionIndex(key) < 0) {
          return Fail(error, std::string(kPrefix) + "unknown key '" + key + "' (known: " +
                                 kKnownKeys + ", or emotion names with weights)");
        }
        if (!value.is_number()) {
          return Fail(error, std::string(kPrefix) + "emotion weight for '" + key +
                                 "' must be a number");
        }
        named += (named.empty() ? "" : ",") + key + ":" + std::to_string(value.as_double());
        break;
    }
  }
  if (!named.empty()) {
    if (controls->emotion.has_value()) {
      return Fail(error, std::string(kPrefix) +
                             "use either 'emotion' or emotion-name keys, not both");
    }
    return SetEmotion(named, controls, error);
  }
  return true;
}

// Position after "ident\s*=" when `text` (after leading spaces) starts with a
// key assignment, else npos.
std::size_t KeyAssignment(std::string_view text, std::size_t pos, std::string* key) {
  while (pos < text.size() && Space(text[pos])) {
    ++pos;
  }
  const std::size_t begin = pos;
  while (pos < text.size() && IdentifierChar(text[pos])) {
    ++pos;
  }
  if (pos == begin) {
    return std::string_view::npos;
  }
  const std::size_t end = pos;
  while (pos < text.size() && Space(text[pos])) {
    ++pos;
  }
  if (pos >= text.size() || text[pos] != '=') {
    return std::string_view::npos;
  }
  *key = std::string(text.substr(begin, end - begin));
  return pos + 1;
}

// "emotion=[0,0,0.7,0,0,0.3,0,0]; emo_alpha=0.8", "emotion=Wut:0.6, Angst:0.2".
// A pair ends at "," or ";" followed by the next "key=", or at the end.
bool ParseKeyValues(std::string_view text, InstructControls* controls, std::string* error) {
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::string key;
    std::size_t value_begin = KeyAssignment(text, pos, &key);
    if (value_begin == std::string_view::npos) {
      return Fail(error, std::string(kPrefix) +
                             "expected key=value pairs (emotion=..., emo_alpha=...)");
    }
    std::size_t value_end = text.size();
    std::size_t next = text.size();
    std::size_t scan = value_begin;
    while (scan < text.size() && Space(text[scan])) {
      ++scan;
    }
    if (scan < text.size() && text[scan] == '[') {
      const std::size_t close = text.find(']', scan);
      if (close == std::string_view::npos) {
        return Fail(error, std::string(kPrefix) + "missing ']' in the emotion vector");
      }
      value_begin = scan + 1;
      value_end = close;
      std::size_t after = close + 1;
      while (after < text.size() && Space(text[after])) {
        ++after;
      }
      if (after < text.size() && text[after] != ',' && text[after] != ';') {
        return Fail(error, std::string(kPrefix) + "unexpected text after ']'");
      }
      next = after < text.size() ? after + 1 : after;
    } else {
      for (std::size_t i = value_begin; i < text.size(); ++i) {
        std::string ignored;
        if ((text[i] == ',' || text[i] == ';') &&
            KeyAssignment(text, i + 1, &ignored) != std::string_view::npos) {
          value_end = i;
          next = i + 1;
          break;
        }
      }
    }
    std::string value(Trim(text.substr(value_begin, value_end - value_begin)));
    while (!value.empty() && (value.back() == ',' || value.back() == ';')) {
      value.pop_back();
    }
    switch (KeyOf(key)) {
      case Key::kEmotion:
        std::ranges::replace(value, ';', ',');
        if (!SetEmotion(value, controls, error)) {
          return false;
        }
        break;
      case Key::kAlpha:
        if (!ParseAlphaText(value, controls, error)) {
          return false;
        }
        break;
      case Key::kUnknown:
        return Fail(error, std::string(kPrefix) + "unknown key '" + key + "' (known: " +
                               kKnownKeys + ")");
    }
    pos = next;
  }
  return true;
}

}  // namespace

bool ParseInstructControls(std::string_view instruct, InstructControls* controls,
                           std::string* error) {
  *controls = InstructControls{};
  const std::string_view text = Trim(instruct);
  if (text.empty()) {
    return true;
  }
  if (text.front() == '{' || text.front() == '[') {
    return ParseJson(text, controls, error);
  }
  if (text.find('=') != std::string_view::npos) {
    return ParseKeyValues(text, controls, error);
  }
  // The 1.x "emotion" string: a name, "name:weight, ..." or 8 numbers.
  return SetEmotion(std::string(text), controls, error);
}

bool ResolveLanguage(std::string_view language, std::string* out, std::string* error) {
  const std::string lang = LowerAscii(Trim(language));
  // "english" is what the 2.0 core fills in when the client sends none.
  if (lang.empty() || lang == "english" || lang == "de" || lang == "german" ||
      lang == "deutsch" || lang == "de-de") {
    *out = "de";
    return true;
  }
  return Fail(error, "IndexTTS-2.5-German supports language 'de' only (got '" +
                         std::string(language) + "')");
}

SamplingOptions ResolveSampling(const ext::TtsSamplingOptions& sampling,
                                std::size_t max_new_tokens) {
  const ext::TtsSamplingOptions core;  // the core's fill-in defaults
  SamplingOptions out;                 // indextts_de.py defaults
  if (sampling.temperature != core.temperature) {
    out.temperature = sampling.temperature;
  }
  if (sampling.top_k != core.top_k) {
    out.top_k = static_cast<std::int32_t>(
        std::min<std::size_t>(sampling.top_k, static_cast<std::size_t>(INT32_MAX)));
  }
  if (sampling.top_p != core.top_p) {
    out.top_p = sampling.top_p;
  }
  if (sampling.repetition_penalty != core.repetition_penalty) {
    out.repetition_penalty = sampling.repetition_penalty;
  }
  out.max_tokens = static_cast<std::int32_t>(
      std::min<std::size_t>(max_new_tokens, static_cast<std::size_t>(kMaximumMelTokens)));
  return out;
}

bool BuildEngineRequest(const ext::SynthesisRequest& request, SynthesisRequest* out,
                        std::string* error) {
  SynthesisRequest req;
  req.text = request.text;
  if (!ResolveLanguage(request.language, &req.language, error)) {
    return false;
  }
  InstructControls controls;
  if (!ParseInstructControls(request.instruct, &controls, error)) {
    return false;
  }
  req.emotion = controls.emotion;
  req.emo_alpha = controls.emo_alpha;
  const ext::TtsAudioBuffer& audio = request.reference_audio;
  if (audio.samples.empty() || audio.channels == 0 || audio.sample_rate == 0) {
    return Fail(error, "IndexTTS-2.5 needs a reference voice (a --voice preset or "
                       "'reference_audio')");
  }
  const std::size_t frames = audio.samples.size() / audio.channels;
  const bool inline_clip = request.speaker.empty() || request.speaker == kCloneVoice;
  if (inline_clip &&
      frames > static_cast<std::size_t>(audio.sample_rate) * kMaximumReferenceSeconds) {
    return Fail(error, "IndexTTS-2.5 reference audio must not exceed " +
                           std::to_string(kMaximumReferenceSeconds) + " seconds");
  }
  req.reference.resize(frames);
  for (std::size_t f = 0; f < frames; ++f) {
    double sum = 0.0;
    for (std::uint32_t c = 0; c < audio.channels; ++c) {
      sum += audio.samples[f * audio.channels + c];
    }
    req.reference[f] = static_cast<float>(sum / static_cast<double>(audio.channels));
  }
  req.reference_rate = audio.sample_rate;
  req.seed = request.sampling.seed;
  req.sampling = ResolveSampling(request.sampling, request.max_new_tokens);
  *out = std::move(req);
  return true;
}

}  // namespace gufo::models::indextts
