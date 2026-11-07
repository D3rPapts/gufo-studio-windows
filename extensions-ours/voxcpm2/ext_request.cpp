// Written with AI assistance (Claude Code).
#include "voxcpm2/ext_request.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace gufo::models::voxcpm2 {

bool BuildEngineRequest(const ext::SynthesisRequest& request,
                        SynthesisRequest* out, std::string* error) {
  SynthesisRequest vox;
  vox.text = request.text;
  vox.instruction = request.instruct;
  const ext::TtsAudioBuffer& audio = request.reference_audio;
  if (!audio.samples.empty() && audio.channels > 0) {
    const std::size_t frames = audio.samples.size() / audio.channels;
    if (audio.sample_rate > 0 &&
        frames > static_cast<std::size_t>(audio.sample_rate) * kMaximumReferenceSeconds) {
      if (error != nullptr) {
        *error = "VoxCPM2 reference audio must not exceed " +
                 std::to_string(kMaximumReferenceSeconds) + " seconds";
      }
      return false;
    }
    vox.reference_audio.resize(frames);
    for (std::size_t f = 0; f < frames; ++f) {
      double sum = 0.0;
      for (std::uint32_t c = 0; c < audio.channels; ++c) {
        sum += audio.samples[f * audio.channels + c];
      }
      vox.reference_audio[f] =
          static_cast<float>(sum / static_cast<double>(audio.channels));
    }
    vox.reference_sample_rate = audio.sample_rate;
    if (!request.reference_text.empty() && !request.speaker_embedding_only) {
      // "Ultimate cloning": the same clip also continues as prompt.
      vox.prompt_audio = vox.reference_audio;
      vox.prompt_sample_rate = audio.sample_rate;
      vox.prompt_text = request.reference_text;
    }
  }
  vox.seed = request.sampling.seed;
  vox.max_len = static_cast<std::int32_t>(
      std::min<std::size_t>(request.max_new_tokens, 4096));
  vox.on_audio = request.on_audio;
  *out = std::move(vox);
  return true;
}

}  // namespace gufo::models::voxcpm2
