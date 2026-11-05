// Ming-3.0-tts prompt contract (tts_tasks.py / prompt_contract.py,
// commit 9332fb94) and the text segmentation of frontend/text_segment_cut.py.
// The strings and JSON key order are part of the trained input protocol.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MING_TTS_PROMPT_HPP_
#define GUFO_MODELS_MING_TTS_PROMPT_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/ming_tts/tokenizer.hpp"

namespace gufo::models::ming_tts {

inline constexpr std::uint32_t kImStartId = 151644;
inline constexpr std::uint32_t kImEndId = 151645;
inline constexpr std::uint32_t kVisionStartId = 151652;
inline constexpr std::uint32_t kVisionEndId = 151653;
inline constexpr std::uint32_t kVisionPadId = 151654;
inline constexpr std::uint32_t kAudioPatchId = 151705;
inline constexpr std::uint32_t kAudioStartId = 151706;

// "Please generate speech based on the following description.\n" plus
// "Target language: <Name>.\n" when `language` is set. Short codes (de, en, ...)
// and full names resolve like tts_tasks._TARGET_LANGUAGE_NAMES; an unknown
// value is used verbatim.
[[nodiscard]] std::string BuildTtsPrompt(std::string_view language);

// Chinese emotion label for an English/German/Chinese name, e.g. "happy",
// "fröhlich", "wut" -> 高兴/愤怒; nullopt when unknown. The seven labels are
// 高兴 悲伤 愤怒 恐惧 厌恶 惊讶 中性.
[[nodiscard]] std::optional<std::string> EmotionLabel(std::string_view name);

// BASIC task speaking rate (tts_tasks.BASIC_ATTRIBUTE_VALUES["speed"]):
// 慢速/中速/快速 for slow/medium/fast (also normal; German langsam, mittel,
// normal, schnell; the Chinese values themselves); nullopt when unknown.
[[nodiscard]] std::optional<std::string> SpeedLabel(std::string_view name);

// BASIC task pitch or volume level: 低/中/高 for low/medium/high (also
// normal; German tief/niedrig, mittel/normal, hoch; for volume also
// quiet/soft/loud and leise/laut); nullopt when unknown.
[[nodiscard]] std::optional<std::string> PitchLabel(std::string_view name);
[[nodiscard]] std::optional<std::string> VolumeLabel(std::string_view name);

// DIALECT task value (tts_tasks.DIALECT_VALUES): "cantonese" (also yue,
// kantonesisch, 粤语) or "chuanyu dialect" (also chuanyu, sichuanese,
// 川渝话); nullopt when unknown.
[[nodiscard]] std::optional<std::string> DialectLabel(std::string_view name);

// Structured caption (tts_tasks.build_structured_instruction). The reference
// uses the zh schema for every task except DIALECT, which uses the en schema
// (build_task_instruction).
enum class InstructionSchema { kChinese, kEnglish };
struct StructuredControls {
  std::optional<std::string> dialect, style, speed, pitch, volume, emotion, ip;
};
[[nodiscard]] std::string BuildStructuredInstruction(
    const StructuredControls& controls,
    InstructionSchema schema = InstructionSchema::kChinese);

struct InputLayout {
  std::vector<std::uint32_t> ids;
  std::vector<std::size_t> speaker_positions;  // rows replaced by spk_head
  // First of the prompt-latent rows (after <audio>), or -1 without them.
  std::int64_t latent_position{-1};
};

// prompt_contract.build_input_ids for the dense checkpoint. Every piece is
// tokenized separately, as the reference does. With a prompt transcript and
// prompt_latent_steps > 0 the transcript precedes the text and that many
// <audioPatch> rows follow <audio>.
[[nodiscard]] bool BuildInputIds(const Tokenizer& tokenizer,
                                 std::string_view prompt, std::string_view text,
                                 std::size_t num_speakers,
                                 const std::optional<std::string>& instruction,
                                 InputLayout* layout, std::string* error,
                                 const std::optional<std::string>& prompt_text = {},
                                 std::size_t prompt_latent_steps = 0);

// MingTTS.prepare_text for languages outside zh/en (no FST normalisation):
// strip Markdown inline code, trim, cut_text_by_semantic_length(text, 24).
[[nodiscard]] std::vector<std::string> SplitTextFragments(
    std::string_view text, std::size_t max_length = 24);

// frontend.text_segment_cut.get_semantic_length on NFC text.
[[nodiscard]] std::size_t SemanticLength(std::string_view text);

}  // namespace gufo::models::ming_tts

#endif  // GUFO_MODELS_MING_TTS_PROMPT_HPP_
