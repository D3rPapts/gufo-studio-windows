// IndexTTS-2.5 HIP runtime (gfx1151) vs. the Python CPU fp32 oracle and the
// CPU runtime, plus speed: GPT teacher forcing (scores cosine, sampled tokens
// with the oracle uniforms, ms per AR step), free run with oracle uniforms,
// DiT velocity, CFM with oracle noise, BigVGAN, and the full engine (RTF,
// first audio) on the oracle sentence and a longer one.
//
//   GUFO_INDEXTTS_MODEL / GUFO_INDEXTTS_REF / GUFO_INDEXTTS_VOICE as in the oracle test
//   GUFO_INDEXTTS_OUT  optional directory for WAVs
//
// Exits 77 (skip) without model, oracle or GPU. Run only in a GPU window.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/indextts/audio.hpp"
#include "src/models/indextts/bigvgan.hpp"
#include "src/models/indextts/gpt.hpp"
#include "src/models/indextts/hip/runtime.hpp"
#include "src/models/indextts/s2mel.hpp"
#include "src/models/indextts/synthesis.hpp"
#include "src/models/indextts/text.hpp"
#include "src/models/ming_tts/safetensors.hpp"

namespace itts = gufo::models::indextts;
namespace json = gufo::json;
using gufo::models::ming_tts::SafetensorsFile;

namespace {

int g_failures = 0;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

double Cos(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    return -2.0;
  }
  double ab = 0.0, aa = 0.0, bb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb + 1e-300);
}

void Check(const std::string& name, double cos, double min_cos) {
  const bool ok = std::isfinite(cos) && cos >= min_cos;
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-40s cos=%.6f (min %.4f)\n", ok ? "OK" : "FAIL", name.c_str(), cos, min_cos);
  std::fflush(stdout);
}

void Expect(const std::string& name, bool ok, const std::string& detail) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-40s %s\n", ok ? "OK" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
}

std::vector<float> Tr(const std::vector<float>& x, std::int64_t rows, std::int64_t cols) {
  std::vector<float> out(x.size());
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(c * rows + r)] = x[static_cast<std::size_t>(r * cols + c)];
    }
  }
  return out;
}

std::string ReadText(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

int main() {
  const char* model_env = std::getenv("GUFO_INDEXTTS_MODEL");
  const char* ref_env = std::getenv("GUFO_INDEXTTS_REF");
  const char* voice_env = std::getenv("GUFO_INDEXTTS_VOICE");
  const char* out_env = std::getenv("GUFO_INDEXTTS_OUT");
  if (model_env == nullptr || ref_env == nullptr || voice_env == nullptr) {
    std::printf("SKIP: set GUFO_INDEXTTS_MODEL, GUFO_INDEXTTS_REF and GUFO_INDEXTTS_VOICE\n");
    return 77;
  }
  const std::filesystem::path model_dir(model_env);
  const std::filesystem::path ref_dir(ref_env);
  std::string error;
  auto oracle = SafetensorsFile::Open(ref_dir / "oracle.safetensors", &error);
  const bool have_oracle = oracle != nullptr && std::filesystem::exists(ref_dir / "oracle.json");
  if (!have_oracle) {
    std::printf("INFO oracle missing (%s): only the engine speed part runs\n", error.c_str());
  }
  const json::Value meta = have_oracle ? json::parse(ReadText(ref_dir / "oracle.json")) : json::Value();
  auto O = [&](const char* name) { return oracle->ToF32(name); };
  const auto t_load = std::chrono::steady_clock::now();
  auto gpt = itts::UnifiedVoice::Load(model_dir / "german" / "gpt.pth", &error);
  auto s2mel = itts::S2Mel::Load(model_dir / "s2mel.pth", &error);
  auto voc = itts::BigVgan::Load(model_dir / "bigvgan_v2_22khz_80band_256x", &error);
  if (gpt == nullptr || s2mel == nullptr || voc == nullptr) {
    std::printf("FAIL load: %s\n", error.c_str());
    return 1;
  }
  auto rt = itts::hip::Runtime::Create(*gpt, *s2mel, *voc, &error);
  if (rt == nullptr) {
    std::printf("SKIP: no GPU runtime: %s\n", error.c_str());
    return 77;
  }
  std::printf("     load + upload %.1f s\n", Seconds(t_load));
  if (have_oracle) {
  // ------------------------------------------------------------- GPT ----
  const std::vector<float> style = O("style");
  const std::vector<float> emovec = O("emovec");
  std::vector<std::int32_t> text_ids;
  for (const std::int64_t v : oracle->ToI64("text_ids")) {
    text_ids.push_back(static_cast<std::int32_t>(v));
  }
  std::vector<std::int32_t> want_tokens;
  for (const std::int64_t v : oracle->ToI64("gpt_tokens")) {
    want_tokens.push_back(static_cast<std::int32_t>(v));
  }
  std::vector<double> uniforms;
  for (const auto& v : meta.find("gpt_u")->items()) {
    uniforms.push_back(v.as_double());
  }
  std::int64_t rows = 0;
  const std::vector<float> prefix =
      gpt->BuildPrefix(gpt->SpeakerLatent(style), emovec, text_ids, itts::LanguageToken("de"), &rows);
  const std::vector<float> scores_want = O("gpt_scores");
  const std::int64_t vocab = gpt->config().mel_codes;
  const auto steps = static_cast<std::int64_t>(scores_want.size()) / vocab;
  itts::SamplingOptions sampling;
  {
    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    rt->GptPrefill(prefix, rows, steps + 4, &logits);
    const double prefill_s = Seconds(t0);
    std::vector<std::int32_t> history = {1, gpt->config().start_mel};
    double worst = 1.0;
    int mismatch = 0;
    t0 = std::chrono::steady_clock::now();
    for (std::int64_t s = 0; s < steps; ++s) {
      itts::ApplyRepetitionPenalty(logits, history, sampling.repetition_penalty);
      const std::vector<float> want(scores_want.begin() + s * vocab, scores_want.begin() + (s + 1) * vocab);
      worst = std::min(worst, Cos(logits, want));
      const std::int32_t tok = itts::SampleToken(logits, sampling, uniforms[static_cast<std::size_t>(s)]);
      if (tok != want_tokens[static_cast<std::size_t>(s)]) {
        ++mismatch;
      }
      history.push_back(want_tokens[static_cast<std::size_t>(s)]);
      if (s + 1 < steps) {
        rt->GptStep(want_tokens[static_cast<std::size_t>(s)], &logits);
      }
    }
    const double per_step = Seconds(t0) / static_cast<double>(std::max<std::int64_t>(1, steps - 1));
    Check("GPT teacher forcing (worst step)", worst, 0.999);
    std::printf("INFO GPT sampled tokens with oracle u: %lld/%lld equal\n",
                static_cast<long long>(steps - mismatch), static_cast<long long>(steps));
    std::printf("     GPT prefill %lld rows %.1f ms, %.2f ms per AR step\n", static_cast<long long>(rows + 1),
                prefill_s * 1000.0, per_step * 1000.0);
  }
  {
    std::vector<float> logits;
    rt->GptPrefill(prefix, rows, 600, &logits);
    std::vector<std::int32_t> history = {1, gpt->config().start_mel};
    std::vector<std::int32_t> codes;
    bool stopped = false;
    for (std::size_t s = 0; s < 600; ++s) {
      itts::ApplyRepetitionPenalty(logits, history, sampling.repetition_penalty);
      const std::int32_t tok = itts::SampleToken(logits, sampling, s < uniforms.size() ? uniforms[s] : 0.5);
      history.push_back(tok);
      if (tok == gpt->config().stop_mel) {
        stopped = true;
        break;
      }
      codes.push_back(tok);
      rt->GptStep(tok, &logits);
    }
    std::vector<std::int32_t> want;
    for (const std::int64_t v : oracle->ToI64("codes")) {
      want.push_back(static_cast<std::int32_t>(v));
    }
    std::size_t same = 0;
    while (same < codes.size() && same < want.size() && codes[same] == want[same]) {
      ++same;
    }
    std::printf("INFO GPT free run (oracle u): %zu codes (oracle %zu), first %zu equal, %s\n", codes.size(),
                want.size(), same, stopped ? "stopped" : "no stop");
    Expect("GPT free run stops", stopped && codes.size() + 40 > want.size() && codes.size() < want.size() + 40,
           std::to_string(codes.size()) + " codes");
  }

  // ------------------------------------------------------------ s2mel ---
  const std::vector<float> ref_mel = O("ref_mel");
  const auto ref_frames = static_cast<std::int64_t>(ref_mel.size() / 80);
  const std::vector<float> prompt_cond = O("prompt_condition");
  const std::vector<float> cond = O("cond");
  const auto target = static_cast<std::int64_t>(cond.size() / 512);
  const std::int64_t total = ref_frames + target;
  std::vector<float> mu(prompt_cond);
  mu.insert(mu.end(), cond.begin(), cond.end());
  // DiT / CFM: the first port's kernels ("legacy", GUFO_INDEXTTS_DIT_FAST=0) first, then the fast path
  // (CFG batch 2, hipBLASLt BF16, GEMM attention). Checks apply to the fast path; legacy is INFO only.
  std::vector<float> velocity_legacy, mel_legacy;
  for (const bool fast : {false, true}) {
  rt->SetFastDit(fast);
  const char* tag = fast ? "fast" : "legacy";
  {
    const std::vector<float> x0 = O("cfm_call0_x");
    const std::vector<float> p0 = O("cfm_call0_prompt_x");
    const std::vector<float> v0 = O("cfm_velocity_call0");
    const std::size_t half = static_cast<std::size_t>(80 * total);
    const std::vector<float> xa = Tr(std::vector<float>(x0.begin(), x0.begin() + half), 80, total);
    const std::vector<float> pa = Tr(std::vector<float>(p0.begin(), p0.begin() + half), 80, total);
    (void)rt->Velocity(xa.data(), pa.data(), 0.0F, style.data(), mu.data(), total);  // warm-up / GEMM tuning
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> vc = rt->Velocity(xa.data(), pa.data(), 0.0F, style.data(), mu.data(), total);
    const double one = Seconds(t0);
    const double cos = Cos(Tr(vc, total, 80), std::vector<float>(v0.begin(), v0.begin() + half));
    if (fast) {
      Check("DiT velocity (cond) step 0 [fast]", cos, 0.999);
      std::printf("INFO DiT velocity fast vs legacy           cos=%.6f\n", Cos(vc, velocity_legacy));
    } else {
      std::printf("INFO DiT velocity (cond) step 0 [legacy]   cos=%.6f\n", cos);
      velocity_legacy = vc;
    }
    std::printf("     DiT evaluation [%s] (%lld frames, 1 branch, incl. upload) %.1f ms\n", tag,
                static_cast<long long>(total), one * 1000.0);
  }
  std::vector<float> mel;
  {
    const std::vector<float> noise = Tr(O("cfm_noise"), 80, total);
    const std::vector<float> ref_tm = Tr(ref_mel, 80, ref_frames);
    const auto t0 = std::chrono::steady_clock::now();
    mel = rt->CfmSample(noise, mu.data(), total, ref_tm.data(), ref_frames, style.data(), itts::CfmOptions{});
    const double secs = Seconds(t0);
    const double cos = Cos(Tr(mel, total, 80), O("cfm_out"));
    if (fast) {
      Check("CFM 25 steps (oracle noise) [fast]", cos, 0.995);
      std::printf("INFO CFM fast vs legacy                    cos=%.6f\n", Cos(mel, mel_legacy));
    } else {
      std::printf("INFO CFM 25 steps (oracle noise) [legacy]  cos=%.6f\n", cos);
      mel_legacy = mel;
    }
    std::printf("TIME CFM [%s] %.1f ms (%lld frames, 25 steps x CFG 2)\n", tag, secs * 1000.0,
                static_cast<long long>(total));
    if (!rt->LastCfmProfile().empty()) {
      std::printf("PROF %s\n", rt->LastCfmProfile().c_str());
    }
  }
  }  // legacy / fast
  {
    const std::vector<float> vc_target = Tr(O("vc_target"), 80, target);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> wav = rt->Vocode(vc_target.data(), target);
    const double secs = Seconds(t0);
    Check("BigVGAN waveform", Cos(wav, O("wav_float")), 0.99);
    std::printf("     BigVGAN %.1f ms for %.2f s audio\n", secs * 1000.0, static_cast<double>(wav.size()) / 22050.0);
  }
  }  // have_oracle
  rt.reset();
  gpt.reset();
  s2mel.reset();
  voc.reset();

  // ----------------------------------------------------------- engine ---
  auto engine = itts::Engine::Create(model_dir, &error);
  if (engine == nullptr) {
    std::printf("FAIL engine: %s\n", error.c_str());
    return 1;
  }
  itts::MonoAudio voice;
  if (!itts::ReadWavMono(voice_env, &voice, &error)) {
    std::printf("FAIL voice: %s\n", error.c_str());
    return 1;
  }
  std::printf("     engine backend: %s\n", engine->backend_name());
  const struct {
    const char* name;
    const char* text;
    std::array<float, 8> emo;
  } cases[] = {
      {"indextts_hip_oracle_text.wav", "Ich kann nicht mehr. Es tut so weh.", {0, 0, 0.6F, 0, 0, 0.2F, 0, 0}},
      {"indextts_hip_long.wav",
       "Als der Zug endlich hielt, war es schon dunkel. Niemand wartete auf dem Bahnsteig, und der Wind trug "
       "den Geruch von Regen über die leeren Gleise.",
       {0, 0, 0.3F, 0, 0, 0.3F, 0, 0}},
      {"indextts_hip_triumph.wav", "Wir haben es geschafft! Endlich haben wir es geschafft!",
       {0.7F, 0, 0, 0, 0, 0, 0.1F, 0}}};
  for (int run = 0; run < 2; ++run) {
    for (const auto& c : cases) {
      itts::SynthesisRequest req;
      req.text = c.text;
      req.reference = voice.samples;
      req.reference_rate = voice.rate;
      req.emotion = c.emo;
      req.seed = 4242;
      itts::SynthesisResult res;
      const auto t0 = std::chrono::steady_clock::now();
      if (!engine->Generate(req, {}, {}, &res, &error)) {
        Expect(std::string("engine ") + c.name, false, error);
        continue;
      }
      const double secs = Seconds(t0);
      const double audio = static_cast<double>(res.samples.size()) / 22050.0;
      std::printf("%s engine %-30s %.2f s audio in %.2f s, RTF %.2f, first audio %.2f s "
                  "(reference %.2f, gpt %.2f / %d steps, s2mel %.2f, vocoder %.2f)\n",
                  run == 0 ? "INFO" : "TIME", c.name, audio, secs, secs / std::max(audio, 1e-9),
                  res.first_audio_seconds, res.reference_seconds, res.gpt_seconds, res.gpt_steps, res.s2mel_seconds,
                  res.vocoder_seconds);
      if (out_env != nullptr && run == 1) {
        (void)itts::WriteWavPcm16(std::filesystem::path(out_env) / c.name, res.samples, 22050);
      }
    }
  }
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
