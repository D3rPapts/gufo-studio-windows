// ACE-Step sampler mathematics (src/models/ace_step/sampler, audio_io):
// the timestep schedule, APG / CFG guidance, repaint frames, masks,
// injection and blends, the waveform splice, the cover start, task rules,
// WAV decoding and resampling. CPU only, no model. Expected values follow
// the official ACE-Step 1.5 code (modeling_acestep_v15_xl_base.py,
// apg_guidance.py, conditioning_masks.py, repaint_waveform_splice.py).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/ace_step/audio_io.hpp"
#include "src/models/ace_step/sampler.hpp"

namespace ace = gufo::models::ace_step;

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL ace_step_sampler_test: " << what << '\n';
    ++failures;
  }
}

bool Near(double a, double b, double tolerance = 1e-5) {
  return std::fabs(a - b) <= tolerance;
}

// linspace(1, 0, steps + 1) shifted: the turbo schedule of 8 steps, shift 3
// starts at 1, ends at 0 and is strictly decreasing; its values match the
// closed form.
void TestTimesteps() {
  const auto turbo = ace::MusicTimesteps(8, 3.0F);
  Check(turbo.size() == 9, "turbo schedule has 9 values");
  Check(Near(turbo.front(), 1.0), "turbo starts at 1");
  Check(turbo.back() == 0.0F, "turbo ends at 0");
  for (std::size_t i = 0; i + 1 < turbo.size(); ++i)
    Check(turbo[i] > turbo[i + 1], "turbo decreasing at " + std::to_string(i));
  // t = 0.5 -> 3 * 0.5 / (1 + 2 * 0.5) = 0.75
  Check(Near(turbo[4], 0.75), "turbo shift of 0.5 is 0.75");
  // t = 0.875 (i = 1) -> 2.625 / 2.75
  Check(Near(turbo[1], 2.625 / 2.75), "turbo second value");
  const auto base = ace::MusicTimesteps(50, 1.0F);
  Check(base.size() == 51, "base schedule has 51 values");
  Check(Near(base[25], 0.5), "shift 1 is linear");
  Check(Near(base[1], 0.98), "shift 1 step");
  // Euler with these dt values reaches exactly 0 after the last step.
  double t = 1.0;
  for (std::size_t i = 0; i + 1 < turbo.size(); ++i)
    t -= double(turbo[i]) - double(turbo[i + 1]);
  Check(Near(t, 0.0), "the schedule integrates to 0");
}

// APG with equal predictions changes nothing; with a difference orthogonal
// to the conditional prediction the full difference is added (scale - 1)
// times; a parallel difference is dropped (eta 0).
void TestApg() {
  constexpr int kFrames = 2;
  std::vector<float> cond(kFrames * 64, 0.0F), uncond(kFrames * 64, 0.0F);
  for (int f = 0; f < kFrames; ++f)
    for (int c = 0; c < 64; ++c)
      cond[f * 64 + c] = uncond[f * 64 + c] = 0.1F * float(c + 1);
  std::vector<float> out;
  ace::ApgGuidance same;
  same.Apply(cond, uncond, 7.0F, kFrames, &out);
  Check(out.size() == cond.size(), "APG output size");
  bool equal = true;
  for (std::size_t i = 0; i < out.size(); ++i)
    equal = equal && Near(out[i], cond[i], 1e-6);
  Check(equal, "APG with identical predictions is the conditional one");

  // Channel 0: cond = (1, 0) over time, uncond = (1, -0.5): the difference
  // (0, 0.5) is orthogonal to cond -> out = cond + 6 * (0, 0.5) = (1, 3).
  // Channel 1: cond = (1, 1), uncond = (0.5, 0.5): the difference (0.5,
  // 0.5) is parallel -> out = cond.
  std::vector<float> c2(kFrames * 64, 0.0F), u2(kFrames * 64, 0.0F);
  c2[0 * 64 + 0] = 1.0F;
  c2[1 * 64 + 0] = 0.0F;
  u2[0 * 64 + 0] = 1.0F;
  u2[1 * 64 + 0] = -0.5F;
  c2[0 * 64 + 1] = 1.0F;
  c2[1 * 64 + 1] = 1.0F;
  u2[0 * 64 + 1] = 0.5F;
  u2[1 * 64 + 1] = 0.5F;
  ace::ApgGuidance apg;
  apg.Apply(c2, u2, 7.0F, kFrames, &out);
  Check(Near(out[0 * 64 + 0], 1.0) && Near(out[1 * 64 + 0], 3.0),
        "APG adds the orthogonal difference (scale - 1) times");
  Check(Near(out[0 * 64 + 1], 1.0) && Near(out[1 * 64 + 1], 1.0),
        "APG drops the parallel difference");
  // Momentum: the second call with the same inputs sees diff + (-0.75) *
  // diff = 0.25 diff -> channel 0 becomes (1, 6 * 0.125) = (1, 0.75).
  apg.Apply(c2, u2, 7.0F, kFrames, &out);
  Check(Near(out[1 * 64 + 0], 0.75), "APG momentum -0.75 accumulates");
  // Norm threshold: a difference of norm 5 over time is scaled to 2.5.
  std::vector<float> c3(kFrames * 64, 0.0F), u3(kFrames * 64, 0.0F);
  c3[0 * 64 + 2] = 1.0F;
  u3[0 * 64 + 2] = 1.0F;
  u3[1 * 64 + 2] = -5.0F;  // diff (0, 5), orthogonal to cond (1, 0)
  ace::ApgGuidance clamp;
  clamp.Apply(c3, u3, 2.0F, kFrames, &out);
  Check(Near(out[1 * 64 + 2], 2.5), "APG clamps the difference norm to 2.5");

  std::vector<float> cfg;
  ace::CfgGuidance(c2, u2, 7.0F, &cfg);
  Check(Near(cfg[1 * 64 + 0], -0.5 + 7.0 * 0.5), "plain CFG");
}

// RepaintFrames: floor(seconds * 25), clamped like the official code.
void TestRepaintFrames() {
  const auto r = ace::RepaintFrames(10.0, 20.0, 750);
  Check(r.start == 250 && r.end == 500, "10..20 s -> frames 250..500");
  const auto whole = ace::RepaintFrames(0.0, 30.0, 750);
  Check(whole.start == 0 && whole.end == 750, "full span");
  const auto beyond = ace::RepaintFrames(40.0, 50.0, 750);
  Check(beyond.start == 749 && beyond.end == 750, "span past the end clamps");
  const auto tiny = ace::RepaintFrames(1.0, 1.01, 750);
  Check(tiny.start == 25 && tiny.end == 26, "a span below one frame is one");
  const auto fractional = ace::RepaintFrames(0.1, 0.9, 750);
  Check(fractional.start == 2 && fractional.end == 22, "fractional seconds");
}

// The repaint context: chunk mask 1 inside the span, source latents replaced
// by silence inside (not for lego), repaint mask marks the span.
void TestRepaintContext() {
  constexpr int kFrames = 10;
  std::vector<float> src(kFrames * 64, 2.0F), silence(kFrames * 64, -1.0F);
  std::vector<float> chunk;
  std::vector<std::uint8_t> mask;
  ace::BuildRepaintContext(ace::FrameRange{3, 6}, kFrames, false, silence,
                           &src, &chunk, &mask);
  Check(chunk.size() == kFrames && mask.size() == kFrames, "mask sizes");
  bool ok = true;
  for (int f = 0; f < kFrames; ++f) {
    const bool inside = f >= 3 && f < 6;
    ok = ok && chunk[f] == (inside ? 1.0F : 0.0F) && (mask[f] != 0) == inside;
    for (int c = 0; c < 64; ++c)
      ok = ok && src[f * 64 + c] == (inside ? -1.0F : 2.0F);
  }
  Check(ok, "repaint context masks and silences the span");
  std::vector<float> lego(kFrames * 64, 2.0F);
  ace::BuildRepaintContext(ace::FrameRange{3, 6}, kFrames, true, silence,
                           &lego, &chunk, &mask);
  Check(lego[4 * 64] == 2.0F, "lego keeps the source inside the span");
}

// Injection replaces x outside the mask with the renoised clean source;
// the boundary blend ramps linearly over the crossfade.
void TestInjectionAndBlend() {
  constexpr int kFrames = 8;
  std::vector<float> x(kFrames * 64, 5.0F), clean(kFrames * 64, 1.0F),
      noise(kFrames * 64, 3.0F);
  std::vector<std::uint8_t> mask(kFrames, 0);
  mask[3] = mask[4] = 1;
  ace::RepaintInjection(x, clean, mask, 0.25F, noise);
  Check(Near(x[0], 0.25 * 3.0 + 0.75 * 1.0), "injection outside the mask");
  Check(x[3 * 64] == 5.0F, "injection keeps the mask");
  const auto weights = ace::RepaintBlendWeights(mask, 2);
  // Span [3, 5): ramps over frames 1, 2 (0.333, 0.667) and 5, 6 (0.667,
  // 0.333); frame 0 and 7 stay 0.
  Check(Near(weights[1], 1.0 / 3.0) && Near(weights[2], 2.0 / 3.0),
        "left ramp");
  Check(Near(weights[5], 2.0 / 3.0) && Near(weights[6], 1.0 / 3.0),
        "right ramp");
  Check(weights[0] == 0.0F && weights[7] == 0.0F && weights[3] == 1.0F,
        "ramp ends");
  std::vector<std::uint8_t> all(kFrames, 1);
  const auto ones = ace::RepaintBlendWeights(all, 2);
  Check(ones[0] == 1.0F && ones[7] == 1.0F, "all ones: no ramp");
  std::vector<float> y(kFrames * 64, 4.0F);
  ace::RepaintBoundaryBlend(y, clean, mask, 0);
  Check(y[0] == 1.0F && y[3 * 64] == 4.0F, "blend without crossfade");
}

// _resolve_repaint_config.
void TestRepaintConfig() {
  const auto balanced = ace::ResolveRepaintConfig("balanced", 0.5);
  Check(Near(balanced.injection_ratio, 0.5) && balanced.crossfade_frames == 12 &&
            Near(balanced.wav_crossfade_seconds, 0.025),
        "balanced 0.5 -> (0.5, round(12.5) = 12, 0.025)");
  const auto conservative = ace::ResolveRepaintConfig("conservative", 0.9);
  Check(conservative.injection_ratio == 1.0 &&
            conservative.crossfade_frames == 25 &&
            Near(conservative.wav_crossfade_seconds, 0.05),
        "conservative");
  const auto aggressive = ace::ResolveRepaintConfig("aggressive", 0.1);
  Check(aggressive.injection_ratio == 0.0 && aggressive.crossfade_frames == 0 &&
            aggressive.wav_crossfade_seconds == 0.0,
        "aggressive");
  const auto strong = ace::ResolveRepaintConfig("balanced", 1.0);
  Check(strong.injection_ratio == 0.0 && strong.crossfade_frames == 0,
        "balanced 1.0 is aggressive");
}

// Waveform splice: the source replaces the prediction outside the span with
// 10 ms ramps; a span covering everything changes nothing.
void TestWaveformSplice() {
  const int rate = 1000;
  const std::size_t samples = 100;
  std::vector<float> pred(2 * samples, 1.0F), src(2 * samples, 0.0F);
  ace::RepaintWaveformSplice(&pred, src, 0.04, 0.06, rate, 0.01);
  Check(pred[0] == 0.0F && pred[samples + 0] == 0.0F, "source outside");
  Check(pred[50] == 1.0F, "prediction inside");
  // Ramp of 10 samples before 40: frames 30..39 = 1/11 .. 10/11.
  Check(Near(pred[30], 1.0 / 11.0) && Near(pred[39], 10.0 / 11.0),
        "front ramp");
  Check(Near(pred[60], 10.0 / 11.0) && Near(pred[69], 1.0 / 11.0),
        "back ramp");
  const auto mask = ace::WaveformCrossfadeMask(100, 40, 60, 10);
  Check(mask[29] == 0.0F && mask[70] == 0.0F && mask[45] == 1.0F,
        "crossfade mask");
  std::vector<float> whole(2 * samples, 1.0F);
  ace::RepaintWaveformSplice(&whole, src, 0.0, 0.1, rate, 0.01);
  Check(whole[0] == 1.0F && whole[99] == 1.0F, "full span: untouched");
  std::vector<float> longer(2 * 150, 1.0F);
  ace::RepaintWaveformSplice(&longer, src, 0.0, 0.05, rate, 0.0);
  Check(longer[99] == 0.0F && longer[100] == 1.0F,
        "prediction longer than the source keeps its tail");
}

// Cover noise start: the schedule value nearest to 1 - strength.
void TestCoverStart() {
  const auto times = ace::MusicTimesteps(8, 1.0F);  // 1, .875, .75, ...
  Check(ace::CoverNoiseStartIndex(times, 0.0F) == 0, "strength 0 starts at 1");
  Check(ace::CoverNoiseStartIndex(times, 0.3F) == 2,
        "strength 0.3 -> nearest to 0.7 is 0.75");
  Check(ace::CoverNoiseStartIndex(times, 1.0F) == 7,
        "strength 1 -> the last value before 0");
}

// Tasks, variants, defaults and instructions.
void TestTasks() {
  ace::MusicTask task;
  Check(ace::ParseMusicTask("Cover-NoFSQ", &task) &&
            task == ace::MusicTask::kCoverNoFsq,
        "cover-nofsq parses");
  Check(ace::ParseMusicTask("cover_nofsq", &task), "cover_nofsq alias");
  Check(!ace::ParseMusicTask("remix", &task), "unknown task");
  Check(ace::ParseMusicTask("", &task) && task == ace::MusicTask::kText2Music,
        "empty task is text2music");
  Check(ace::TaskUnsupportedReason(ace::MusicTask::kExtract,
                                   ace::DitVariant::kTurbo)
                .find("base") != std::string::npos,
        "turbo refuses extract");
  Check(!ace::TaskUnsupportedReason(ace::MusicTask::kLego,
                                    ace::DitVariant::kSft)
             .empty(),
        "sft refuses lego");
  Check(ace::TaskUnsupportedReason(ace::MusicTask::kComplete,
                                   ace::DitVariant::kBase)
            .empty(),
        "base allows complete");
  Check(ace::TaskUnsupportedReason(ace::MusicTask::kRepaint,
                                   ace::DitVariant::kTurbo)
            .empty(),
        "turbo allows repaint");
  Check(ace::TaskNeedsSource(ace::MusicTask::kCover) &&
            !ace::TaskNeedsSource(ace::MusicTask::kText2Music),
        "source requirement");
  Check(ace::TaskIsCover(ace::MusicTask::kCover) &&
            !ace::TaskIsCover(ace::MusicTask::kCoverNoFsq),
        "only cover goes through the FSQ round trip");
  Check(ace::TaskRepaints(ace::MusicTask::kLego) &&
            !ace::TaskRepaints(ace::MusicTask::kExtract),
        "repaint tasks");
  const auto turbo = ace::DefaultSampler(ace::DitVariant::kTurbo);
  const auto sft = ace::DefaultSampler(ace::DitVariant::kSft);
  const auto base = ace::DefaultSampler(ace::DitVariant::kBase);
  Check(turbo.steps == 8 && turbo.guidance_scale == 1.0F && turbo.shift == 3.0F,
        "turbo defaults");
  Check(sft.steps == 50 && sft.guidance_scale == 7.0F, "sft defaults");
  Check(base.steps == 32 && base.guidance_scale == 7.0F, "base defaults");
  ace::DitVariant variant;
  Check(ace::GuessDitVariant("acestep_v1.5_xl_turbo_bf16.safetensors", &variant) &&
            variant == ace::DitVariant::kTurbo,
        "variant from the ComfyUI name");
  Check(ace::GuessDitVariant("dit-sft.safetensors", &variant) &&
            variant == ace::DitVariant::kSft,
        "variant from the gufo name");
  Check(!ace::GuessDitVariant("dit.safetensors", &variant),
        "no variant in the default name");
  const std::vector<std::string> classes = {"drums", "bass"};
  Check(ace::MusicInstruction(ace::MusicTask::kExtract, "vocals", {}) ==
            "Extract the VOCALS track from the audio:",
        "extract instruction");
  Check(ace::MusicInstruction(ace::MusicTask::kLego, "", {}) ==
            "Generate the track based on the audio context:",
        "lego default instruction");
  Check(ace::MusicInstruction(ace::MusicTask::kComplete, "", classes) ==
            "Complete the input track with DRUMS | BASS:",
        "complete instruction");
  Check(ace::MusicInstruction(ace::MusicTask::kCoverNoFsq, "", {}) ==
            "Generate audio semantic tokens based on the given conditions:",
        "cover instruction");
  Check(ace::MusicInstruction(ace::MusicTask::kText2Music, "", {}) ==
            ace::DefaultMusicInstruction(),
        "text2music instruction");
  // The plain branch of cover_strength < 1 re-encodes the prompt with the
  // default instruction, whose word (and token) count differs from the
  // cover instruction's: the pipeline must pass that prompt's own row count.
  const auto words = [](const std::string& text) {
    int count = 0;
    for (std::size_t i = 0; i < text.size(); ++i)
      if (text[i] != ' ' && (i == 0 || text[i - 1] == ' '))
        ++count;
    return count;
  };
  Check(words(ace::MusicInstruction(ace::MusicTask::kCover, "", {})) !=
            words(ace::DefaultMusicInstruction()),
        "cover and default instructions differ in length");
}

// WAV round trip through the writer and the decoder, mono duplication, and
// the resampler's length and tone.
void TestAudioIo() {
  const std::size_t samples = 480;
  std::vector<float> planar(2 * samples);
  for (std::size_t i = 0; i < samples; ++i) {
    planar[i] = 0.5F * std::sin(2.0F * 3.14159265F * float(i) / 48.0F);
    planar[samples + i] = -planar[i];
  }
  const std::string wav = ace::EncodeStereoWav(planar, 48000);
  ace::StereoAudio decoded;
  std::string error;
  Check(ace::DecodeWavStereo(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(wav.data()), wav.size()),
            &decoded, &error),
        "decode the written WAV: " + error);
  Check(decoded.sample_rate == 48000 && decoded.samples == samples,
        "WAV rate and length");
  bool close = true;
  for (std::size_t i = 0; i < samples && close; ++i)
    close = Near(decoded.planar[i], planar[i], 1e-4) &&
            Near(decoded.planar[samples + i], planar[samples + i], 1e-4);
  Check(close, "WAV samples survive the 16-bit round trip");
  // A mono 8-bit WAV: both channels get the same samples.
  std::string mono = "RIFF";
  auto u32 = [&](std::string* s, std::uint32_t v) {
    for (int k = 0; k < 4; ++k)
      s->push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
  };
  auto u16 = [&](std::string* s, std::uint16_t v) {
    s->push_back(static_cast<char>(v & 0xFF));
    s->push_back(static_cast<char>(v >> 8));
  };
  u32(&mono, 36 + 4);
  mono += "WAVEfmt ";
  u32(&mono, 16);
  u16(&mono, 1);
  u16(&mono, 1);
  u32(&mono, 22050);
  u32(&mono, 22050);
  u16(&mono, 1);
  u16(&mono, 8);
  mono += "data";
  u32(&mono, 4);
  mono.push_back(static_cast<char>(128));
  mono.push_back(static_cast<char>(255));
  mono.push_back(static_cast<char>(0));
  mono.push_back(static_cast<char>(192));
  Check(ace::DecodeWavStereo(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(mono.data()),
                mono.size()),
            &decoded, &error),
        "decode mono 8-bit: " + error);
  Check(decoded.samples == 4 && decoded.sample_rate == 22050 &&
            decoded.planar[0] == 0.0F && decoded.planar[4] == 0.0F &&
            Near(decoded.planar[1], 127.0 / 128.0) &&
            decoded.planar[2] == -1.0F && decoded.planar[6] == -1.0F &&
            Near(decoded.planar[3], 0.5) && Near(decoded.planar[7], 0.5),
        "mono 8-bit values and duplication");
  Check(!ace::DecodeWavStereo(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>("ID3xxxxxxxxxxxxx"), 16),
            &decoded, &error),
        "an mp3 is not a WAV");
  // Resampling a 1 kHz tone from 44.1 kHz to 48 kHz keeps the tone.
  const std::size_t n = 4410;
  std::vector<float> tone(n);
  for (std::size_t i = 0; i < n; ++i)
    tone[i] = std::sin(2.0F * 3.14159265F * 1000.0F * float(i) / 44100.0F);
  const auto resampled = ace::ResampleLikeTorchaudio(tone, 44100, 48000);
  Check(resampled.size() == 4800, "resampled length ceil(n * 48000 / 44100)");
  double worst = 0;
  for (std::size_t i = 200; i + 200 < resampled.size(); ++i) {
    const double expected =
        std::sin(2.0 * 3.14159265358979 * 1000.0 * double(i) / 48000.0);
    worst = std::max(worst, std::fabs(double(resampled[i]) - expected));
  }
  Check(worst < 0.02, "resampled tone matches (worst " + std::to_string(worst) + ")");
  Check(ace::ResampleLikeTorchaudio(tone, 48000, 48000).size() == n,
        "identical rates copy");
  ace::StereoAudio audio;
  audio.sample_rate = 44100;
  audio.samples = n;
  audio.planar.resize(2 * n);
  std::copy(tone.begin(), tone.end(), audio.planar.begin());
  std::copy(tone.begin(), tone.end(), audio.planar.begin() + n);
  ace::NormalizeStereo48k(&audio);
  Check(audio.sample_rate == 48000 && audio.samples == 4800 &&
            audio.planar.size() == 2 * 4800,
        "normalize to 48 kHz stereo");
  Check(ace::IsSilentAudio(std::vector<float>(10, 0.0F)) &&
            !ace::IsSilentAudio(tone),
        "silence detection");
  // The reference excerpt is 30 s even for a 1 s input (repeated).
  ace::StereoAudio reference;
  reference.sample_rate = 48000;
  reference.samples = 48000;
  reference.planar.assign(2 * 48000, 0.25F);
  const auto excerpt = ace::ReferenceExcerpt(reference, 7);
  Check(excerpt.size() == 2 * 30 * 48000 && excerpt[0] == 0.25F &&
            excerpt.back() == 0.25F,
        "reference excerpt covers 30 s");
}

}  // namespace

int main() {
  TestTimesteps();
  TestApg();
  TestRepaintFrames();
  TestRepaintContext();
  TestInjectionAndBlend();
  TestRepaintConfig();
  TestWaveformSplice();
  TestCoverStart();
  TestTasks();
  TestAudioIo();
  if (failures != 0) {
    std::cerr << failures << " ace_step_sampler_test failure(s)\n";
    return 1;
  }
  std::cout << "ace_step_sampler_test passed\n";
  return 0;
}
