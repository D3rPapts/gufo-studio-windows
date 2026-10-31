// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_TALK21_HPP_
#define GUFO_MODELS_WAN22_TALK21_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/pipeline21.hpp"
#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// One speaker of a talking-head video.
struct TalkSpeaker {
  /// Mono 16 kHz speech.
  std::vector<float> audio;
  /// Two speakers: 8-bit mask [mask_height, mask_width] of where this
  /// speaker is in the first frame (non-zero = the speaker), any size.
  std::vector<std::uint8_t> mask;
  int mask_width{0};
  int mask_height{0};
};

/// InfiniteTalk on Wan 2.1 I2V 14B (+ lightx2v), as ComfyUI runs a chain of
/// WanInfiniteTalkToVideo windows with KSampler (LCM): umT5 prompt, CLIP-H
/// of the start image, VAE condition (start image then grey, 4-channel
/// first-frame mask), wav2vec2 features at 25 fps (two speakers in turn,
/// "add", or together, "para"), and windows of `window` frames: the first
/// pins latent frame 0 to the start image, each next one continues from the
/// last `motion_frames` decoded frames (re-encoded and pinned as the first
/// latent frames at every step, then dropped from the output). 25 fps.
struct Talk21Request {
  std::filesystem::path wan21_root;  // Diffusers Wan2.1-I2V-14B-480P: vae,
                                     // image_encoder
  std::filesystem::path t5_root;     // Diffusers root with tokenizer/ and
                                     // text_encoder/ (umT5-XXL)
  std::filesystem::path dit_file;    // Kijai FP8 checkpoint
  std::filesystem::path lora_file;   // optional
  float lora_strength{1.0F};
  std::filesystem::path talk_file;      // InfiniteTalk model patch
  std::filesystem::path wav2vec2_file;  // chinese-wav2vec2-base
  std::string prompt;
  int width{832};
  int height{480};
  /// Output frames at 25 fps; 0 = the length of the audio.
  int frames{0};
  int window{81};        // 4k + 1
  int motion_frames{9};  // 4k + 1, < window
  int steps{4};
  float shift{5.0F};
  std::uint64_t seed{42};  // window w samples with seed + w
  float audio_scale{1.0F};
  std::string audio_mode{"add"};  // "add" (in turn) or "para" (together)
  /// RGB8 [height, width, 3] at the video size.
  std::span<const std::uint8_t> first_frame;
  /// One or two speakers (two need masks).
  std::vector<TalkSpeaker> speakers;
};

struct Talk21Telemetry {
  double text_ms{0}, clip_ms{0}, audio_ms{0}, condition_ms{0};
  double dit_load_ms{0}, context_ms{0}, motion_ms{0}, decode_ms{0};
  double total_ms{0};
  int windows{0};
  int audio_frames{0};  // 25 fps
  std::vector<double> step_ms;
  double preview_ms{0};
  int previews{0};
  std::size_t peak_device_bytes{0};
};

/// Video frames at 25 fps the speakers' audio fills (`audio_mode` "add":
/// the sum, "para": the longest).
int TalkAudioFrames(const std::vector<TalkSpeaker>& speakers,
                    const std::string& audio_mode);

/// Windows a request of `frames` output frames needs.
int TalkWindows(int frames, int window, int motion_frames);

bool GenerateTalk21(const Talk21Request& request,
                    const I2V21Callbacks& callbacks, DecodedVideo* video,
                    Talk21Telemetry* telemetry, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_TALK21_HPP_
