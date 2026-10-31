// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_VIDEO_TALK_HPP_
#define GUFO_SERVER_VIDEO_TALK_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/generation.hpp"

namespace gufo::server {

// Video job runner for Wan 2.1 I2V 14B + InfiniteTalk
// ("wan2.1-infinitetalk-14b"): talking video from a start image and one or
// two speakers' audio, 25 fps, any length up to 60 s (windows of 81 frames
// continued from the last 9). The model directory is a bundle whose
// gufo-talk.json names the parts relative to it (or absolute):
//   {"_class_name": "GufoWan21InfiniteTalk",
//    "wan21": Diffusers Wan2.1-I2V-14B-480P (vae/, image_encoder/),
//    "t5": Diffusers root with tokenizer/ and text_encoder/ (umT5-XXL),
//    "dit": Kijai FP8 checkpoint, "lora": lightx2v LoRA, "lora_strength",
//    "talk": InfiniteTalk single model patch,
//    "talk_multi": InfiniteTalk multi model patch (two speakers; optional,
//    else "talk"), "wav2vec2": chinese-wav2vec2-base safetensors,
//    "shift" (5), "window" (81), "motion_frames" (9)}
// request.audio_guides are the speakers (frame = index), request.talk_json
// the parsed "gufo.talk" options, request.speaker_masks the two masks (bright
// = the speaker). The MP4 carries the speakers' audio (placed as the video
// uses it).
[[nodiscard]] bool RunWan21TalkJob(
    const minimax_h3::GenerationRequest& request,
    const minimax_h3::CancellationToken* cancellation,
    minimax_h3::GenerationProgress progress, void* progress_opaque,
    minimax_h3::GenerationTelemetry* telemetry, std::string* error);

// "gufo.talk": {"audio_scale": -10..10 (1), "audio_mode": "add" | "para"
// ("add": speaker 2 speaks after speaker 1), "motion_frames": 1..33 (4k + 1,
// bundle default)}. Writes the normalized object to `normalized`.
[[nodiscard]] bool ParseTalkOptions(const json::Value& talk,
                                    std::string* normalized,
                                    std::string* error);

// Video frames at 25 fps one speaker's audio drives (wav2vec2 at 50 Hz,
// halved as WanInfiniteTalkToVideo does).
[[nodiscard]] int TalkFramesForAudio(const minimax_h3::AudioWaveform& audio);

// Mono 16 kHz from channel-major PCM at any rate (channel mean, windowed-sinc
// resampling).
[[nodiscard]] std::vector<float> TalkMono16k(
    const minimax_h3::AudioWaveform& audio);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_TALK_HPP_
