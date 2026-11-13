// Written with AI assistance (Claude Code).
#include "ming_tts/ext_request.hpp"

#include <gufo/ext/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ming_tts/prompt.hpp"

namespace gufo::models::ming_tts {
namespace {

namespace json = gufo::json;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string_view Trim(std::string_view text) {
  const auto space = [](char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
  };
  while (!text.empty() && space(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && space(text.back())) {
    text.remove_suffix(1);
  }
  return text;
}

std::string Lower(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

// Slot for a control key, or nullptr when the key is unknown.
std::string* Slot(InstructControls* controls, std::string_view key) {
  const std::string k = Lower(Trim(key));
  if (k == "emotion") return &controls->emotion;
  if (k == "tempo" || k == "speed") return &controls->speed;
  if (k == "pitch") return &controls->pitch;
  if (k == "volume") return &controls->volume;
  if (k == "dialect") return &controls->dialect;
  if (k == "instruction" || k == "instructions" || k == "caption") {
    return &controls->instruction;
  }
  return nullptr;
}

constexpr const char* kKnownKeys =
    "emotion, tempo, pitch, volume, dialect, instruction";

bool ParseJsonControls(std::string_view text, InstructControls* controls,
                       std::string* error) {
  json::Value root;
  try {
    root = json::parse(std::string(text));
  } catch (const std::exception& e) {
    SetError(error, std::string("Ming instructions: invalid JSON: ") + e.what());
    return false;
  }
  if (!root.is_object()) {
    SetError(error, "Ming instructions: JSON controls must be an object");
    return false;
  }
  for (const auto& [key, value] : root.members()) {
    std::string* slot = Slot(controls, key);
    if (slot == nullptr) {
      SetError(error, "Ming instructions: unknown control '" + key + "' (" +
                          kKnownKeys + ")");
      return false;
    }
    if (!value.is_string()) {
      SetError(error, "Ming instructions: control '" + key + "' must be a string");
      return false;
    }
    *slot = std::string(Trim(value.str()));
  }
  return true;
}

// "key=value, key=value". Returns nullopt when no part names a known key (the
// text is then not a control list), false on a malformed list.
std::optional<bool> ParseListControls(std::string_view text,
                                      InstructControls* controls,
                                      std::string* error) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || text[i] == ',' || text[i] == ';') {
      const std::string_view part = Trim(text.substr(start, i - start));
      if (!part.empty()) {
        parts.push_back(part);
      }
      start = i + 1;
    }
  }
  bool any_key = false;
  for (const std::string_view part : parts) {
    const std::size_t eq = part.find('=');
    InstructControls probe;
    if (eq != std::string_view::npos && Slot(&probe, part.substr(0, eq)) != nullptr) {
      any_key = true;
    }
  }
  if (!any_key) {
    return std::nullopt;
  }
  for (const std::string_view part : parts) {
    const std::size_t eq = part.find('=');
    std::string* slot =
        eq == std::string_view::npos ? nullptr : Slot(controls, part.substr(0, eq));
    if (slot == nullptr) {
      SetError(error, "Ming instructions: '" + std::string(part) +
                          "' is not a key=value control (" + kKnownKeys + ")");
      return false;
    }
    *slot = std::string(Trim(part.substr(eq + 1)));
  }
  return true;
}

}  // namespace

bool ParseInstructControls(std::string_view instruct, InstructControls* controls,
                           std::string* error) {
  *controls = {};
  const std::string_view text = Trim(instruct);
  if (text.empty()) {
    return true;
  }
  if (text.front() == '{') {
    return ParseJsonControls(text, controls, error);
  }
  if (const auto list = ParseListControls(text, controls, error); list.has_value()) {
    return *list;
  }
  if (EmotionLabel(text).has_value()) {
    controls->emotion = std::string(text);
  } else {
    controls->instruction = std::string(text);
  }
  return true;
}

bool BuildEngineRequest(const ext::SynthesisRequest& request,
                        SynthesisRequest* out, std::string* error) {
  InstructControls controls;
  if (!ParseInstructControls(request.instruct, &controls, error)) {
    return false;
  }
  SynthesisRequest ming;
  ming.text = request.text;
  ming.language = request.language;
  ming.emotion = std::move(controls.emotion);
  ming.instruction = std::move(controls.instruction);
  ming.speed = std::move(controls.speed);
  ming.pitch = std::move(controls.pitch);
  ming.volume = std::move(controls.volume);
  ming.dialect = std::move(controls.dialect);
  // Validate here, before the engine loads audio, so a bad value fails fast.
  std::optional<std::string> task_instruction;
  if (!ResolveTaskInstruction(ming, &task_instruction, error)) {
    return false;
  }
  // As in the 1.x API, a free-text instruction wins over an emotion label.
  if (!ming.emotion.empty() && !EmotionLabel(ming.emotion).has_value()) {
    SetError(error, "unknown Ming emotion '" + ming.emotion +
                        "' (happy, sad, angry, fear, disgust, surprise, neutral)");
    return false;
  }
  const ext::TtsAudioBuffer& audio = request.reference_audio;
  if (!audio.samples.empty() && audio.channels > 0) {
    const std::size_t frames = audio.samples.size() / audio.channels;
    ming.reference_audio.resize(frames);
    for (std::size_t f = 0; f < frames; ++f) {
      double sum = 0.0;
      for (std::uint32_t c = 0; c < audio.channels; ++c) {
        sum += audio.samples[f * audio.channels + c];
      }
      ming.reference_audio[f] =
          static_cast<float>(sum / static_cast<double>(audio.channels));
    }
    ming.reference_sample_rate = audio.sample_rate;
    // Ming always clones from the CAM++ speaker embedding; a transcript adds
    // the ZERO_SHOT prompt latents unless the caller asked for the embedding
    // alone.
    if (!request.speaker_embedding_only) {
      ming.reference_text = request.reference_text;
    }
  }
  ming.seed = request.sampling.seed;
  ming.max_decode_steps = static_cast<std::int32_t>(
      std::min<std::size_t>(request.max_new_tokens, 1000));
  ming.on_audio = request.on_audio;
  *out = std::move(ming);
  return true;
}

}  // namespace gufo::models::ming_tts
