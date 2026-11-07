// VoxCPM2 CPU runtime vs. the Python CPU fp32 oracle
// (src/models/voxcpm2/reference/voxcpm2_oracle.py, voxcpm 2.0.3). Every stage
// must reach cosine >= 0.999 against the oracle.
//
//   GUFO_VOXCPM_MODEL  checkpoint directory (openbmb/VoxCPM2)
//   GUFO_VOXCPM_REF    directory with oracle.safetensors + oracle.json
//   GUFO_VOXCPM_STAGES optional comma list: text,vae_enc,resample,prefill,dit,ar,vae_dec,free
//   GUFO_VOXCPM_AR_STEPS optional cap for the teacher-forced AR stage
//   GUFO_VOXCPM_OUT    optional directory for the free-running WAV
//
// Exits 77 (skip) when the model or the oracle is not configured.
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

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "src/core/json.hpp"
#include "src/models/ming_tts/safetensors.hpp"
#include "src/models/ming_tts/speaker.hpp"
#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/voxcpm2/config.hpp"
#include "src/models/voxcpm2/cpu/audio_vae.hpp"
#include "src/models/voxcpm2/cpu/model.hpp"
#include "src/models/voxcpm2/synthesis.hpp"
#include "src/models/voxcpm2/tokenizer.hpp"

namespace vox = gufo::models::voxcpm2;
namespace json = gufo::json;
using gufo::models::ming_tts::SafetensorsFile;

namespace {

// Peak host memory of this run (the router's admission measure on Windows is
// the peak working set, which includes resident file-mapping pages, #829).
void PrintPeakMemory() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(),
                           reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                           sizeof(counters)) != 0) {
    std::printf("host memory: peak working set %.3f GiB, working set %.3f GiB, private %.3f GiB\n",
                static_cast<double>(counters.PeakWorkingSetSize) / (1ULL << 30U),
                static_cast<double>(counters.WorkingSetSize) / (1ULL << 30U),
                static_cast<double>(counters.PrivateUsage) / (1ULL << 30U));
  }
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    std::printf("host memory: peak RSS %.3f GiB\n",
                static_cast<double>(usage.ru_maxrss) / (1ULL << 20U));
  }
#endif
}

int g_failures = 0;
constexpr double kMinCos = 0.999;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

double Cosine(const float* got, const float* want, std::size_t count, double* max_abs,
              double* ref_max) {
  double dot = 0.0, a2 = 0.0, b2 = 0.0;
  *max_abs = 0.0;
  *ref_max = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    dot += static_cast<double>(got[i]) * want[i];
    a2 += static_cast<double>(got[i]) * got[i];
    b2 += static_cast<double>(want[i]) * want[i];
    *max_abs = std::max(*max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
    *ref_max = std::max(*ref_max, std::fabs(static_cast<double>(want[i])));
  }
  return (a2 == 0.0 && b2 == 0.0) ? 1.0 : dot / std::sqrt(a2 * b2 + 1e-300);
}

void Compare(const std::string& name, const float* got, const float* want, std::size_t count,
             double min_cos = kMinCos) {
  double max_abs = 0.0, ref_max = 0.0;
  const double cos = Cosine(got, want, count, &max_abs, &ref_max);
  const bool ok = cos >= min_cos && std::isfinite(cos);
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-36s cos=%.7f max_abs=%.3g (ref max %.3g, n=%zu)\n", ok ? "OK" : "FAIL",
              name.c_str(), cos, max_abs, ref_max, count);
  std::fflush(stdout);
}

void Compare(const std::string& name, const std::vector<float>& got,
             const std::vector<float>& want, double min_cos = kMinCos) {
  if (got.size() != want.size()) {
    std::printf("FAIL %-36s size %zu vs %zu\n", name.c_str(), got.size(), want.size());
    ++g_failures;
    const std::size_t n = std::min(got.size(), want.size());
    if (n > 0) {
      Compare(name + " (common prefix)", got.data(), want.data(), n, min_cos);
    }
    return;
  }
  Compare(name, got.data(), want.data(), got.size(), min_cos);
}

void Info(const std::string& name, const std::vector<float>& got, const std::vector<float>& want) {
  const std::size_t n = std::min(got.size(), want.size());
  double max_abs = 0.0, ref_max = 0.0;
  const double cos = n > 0 ? Cosine(got.data(), want.data(), n, &max_abs, &ref_max) : 0.0;
  std::printf("INFO %-36s cos=%.7f max_abs=%.3g (n=%zu vs %zu)\n", name.c_str(), cos, max_abs,
              got.size(), want.size());
  std::fflush(stdout);
}

void Expect(const std::string& name, bool ok, const std::string& detail = {}) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-36s %s\n", ok ? "OK" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool Want(const std::string& stages, const char* stage) {
  return stages.empty() ||
         ("," + stages + ",").find("," + std::string(stage) + ",") != std::string::npos;
}

std::vector<float> Slice(const std::vector<float>& all, std::size_t index, std::size_t size) {
  return {all.begin() + static_cast<std::ptrdiff_t>(index * size),
          all.begin() + static_cast<std::ptrdiff_t>((index + 1) * size)};
}

// [64, 4] (channel-major patch) -> [4, 64]
std::vector<float> Transpose(const float* src, std::int64_t rows, std::int64_t cols) {
  std::vector<float> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(c * rows + r)] = src[r * cols + c];
    }
  }
  return out;
}

// First `keep` columns of every row of a [rows, cols] matrix.
std::vector<float> Columns(const std::vector<float>& m, std::int64_t rows, std::int64_t cols,
                           std::int64_t keep) {
  std::vector<float> out;
  for (std::int64_t r = 0; r < rows; ++r) {
    out.insert(out.end(), m.begin() + r * cols, m.begin() + r * cols + keep);
  }
  return out;
}

void WriteWav(const std::filesystem::path& path, const std::vector<float>& samples,
              std::uint32_t rate) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
  const auto bytes = static_cast<std::uint32_t>(samples.size() * 2);
  out.write("RIFF", 4);
  u32(36 + bytes);
  out.write("WAVEfmt ", 8);
  u32(16);
  u16(1);
  u16(1);
  u32(rate);
  u32(rate * 2);
  u16(2);
  u16(16);
  out.write("data", 4);
  u32(bytes);
  for (const float s : samples) {
    const auto v = static_cast<std::int16_t>(std::lround(std::clamp(s, -1.0F, 1.0F) * 32767.0F));
    u16(static_cast<std::uint16_t>(v));
  }
}

}  // namespace

int main() {
  const char* model_env = std::getenv("GUFO_VOXCPM_MODEL");
  const char* ref_env = std::getenv("GUFO_VOXCPM_REF");
  if (model_env == nullptr || ref_env == nullptr) {
    std::printf("SKIP: set GUFO_VOXCPM_MODEL and GUFO_VOXCPM_REF\n");
    return 77;
  }
  const std::filesystem::path model_dir(model_env);
  const std::filesystem::path ref_dir(ref_env);
  const char* stages_env = std::getenv("GUFO_VOXCPM_STAGES");
  const std::string stages = stages_env != nullptr ? stages_env : "";
  std::string error;
  auto oracle = SafetensorsFile::Open(ref_dir / "oracle.safetensors", &error);
  if (oracle == nullptr || !std::filesystem::exists(ref_dir / "oracle.json")) {
    std::printf("SKIP: oracle missing: %s\n", error.c_str());
    return 77;
  }
  const json::Value meta = json::parse(ReadText(ref_dir / "oracle.json"));
  const auto t0 = std::chrono::steady_clock::now();
  auto config = vox::LoadModelConfigFromPath(model_dir.string(), &error);
  if (!config.has_value()) {
    std::printf("FAIL config: %s\n", error.c_str());
    return 1;
  }
  const std::int64_t P = config->patch_size;
  const std::int64_t D = config->feat_dim;
  const std::int64_t H = config->base_lm.hidden_size;
  const std::int64_t PD = P * D;
  const std::string text = meta.member_str("text");

  // ------------------------------------------------------------ text ----
  vox::Tokenizer tokenizer;
  if (!vox::Tokenizer::Load(model_dir.string(), &tokenizer, &error)) {
    std::printf("FAIL tokenizer load: %s\n", error.c_str());
    return 1;
  }
  const std::vector<float> ref_feat = oracle->ToF32("ref_feat");
  const auto ref_len = static_cast<std::int64_t>(ref_feat.size()) / PD;
  const std::vector<std::int64_t> want_ids = oracle->ToI64("input_ids");
  const std::vector<std::int64_t> want_mask = oracle->ToI64("audio_mask");
  if (Want(stages, "text")) {
    int index = 0;
    for (const auto& c : meta.find("tokenizer_cases")->items()) {
      std::vector<std::int64_t> want;
      for (const auto& v : c.find("ids")->items()) {
        want.push_back(static_cast<std::int64_t>(v.as_double()));
      }
      const auto got = tokenizer.Encode(c.member_str("text"));
      Expect("tokenizer case " + std::to_string(index++), got == want,
             std::to_string(got.size()) + " ids vs " + std::to_string(want.size()));
    }
    std::vector<std::int64_t> ids{vox::kRefAudioStartToken};
    ids.insert(ids.end(), static_cast<std::size_t>(ref_len), 0);
    ids.push_back(vox::kRefAudioEndToken);
    const auto text_ids = tokenizer.Encode(vox::NormalizeWhitespace(text));
    ids.insert(ids.end(), text_ids.begin(), text_ids.end());
    ids.push_back(vox::kAudioStartToken);
    Expect("prompt layout (reference mode)", ids == want_ids,
           std::to_string(ids.size()) + " ids vs " + std::to_string(want_ids.size()));
  }

  // --------------------------------------------------------- AudioVAE ----
  auto vae = vox::cpu::AudioVae::Load(model_dir.string(), config->vae, &error);
  if (vae == nullptr) {
    std::printf("FAIL AudioVAE load: %s\n", error.c_str());
    return 1;
  }
  std::printf("load AudioVAE %.1f s\n", Seconds(t0));
  const std::vector<float> wave16k = oracle->ToF32("ref_wave16k");
  if (Want(stages, "vae_enc")) {
    const auto t = std::chrono::steady_clock::now();
    vox::cpu::AudioVae::EncodeTrace trace;
    std::int64_t frames = 0;
    const std::vector<float> mu = vae->Encode(wave16k.data(),
                                              static_cast<std::int64_t>(wave16k.size()), &frames,
                                              &trace);
    std::printf("vae encode %.2f s (%lld frames)\n", Seconds(t), static_cast<long long>(frames));
    const std::int64_t ch0 = config->vae.encoder_dim;
    Compare("vae_enc block0 (first 4000 samples)",
            Columns(trace.block0, ch0, static_cast<std::int64_t>(wave16k.size()), 4000),
            oracle->ToF32("vae_enc_block0"));
    Compare("vae_enc block4", trace.block4, oracle->ToF32("vae_enc_block4"));
    Compare("vae_enc mu", mu, oracle->ToF32("vae_enc_mu"));
  }
  if (Want(stages, "resample")) {
    const std::filesystem::path voice(meta.member_str("voice"));
    const std::string bytes = ReadText(voice);
    gufo::models::qwen3_tts::AudioBuffer audio;
    if (!gufo::models::qwen3_tts::DecodeWav(
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(bytes.data()),
                                       bytes.size()),
            &audio, &error)) {
      Expect("reference wav decode", false, error);
    } else {
      std::vector<float> mono(audio.samples.size() / audio.channels);
      for (std::size_t f = 0; f < mono.size(); ++f) {
        double sum = 0.0;
        for (std::uint32_t c = 0; c < audio.channels; ++c) {
          sum += audio.samples[f * audio.channels + c];
        }
        mono[f] = static_cast<float>(sum / audio.channels);
      }
      const std::vector<float> ours =
          audio.sample_rate == 16000
              ? mono
              : gufo::models::ming_tts::ResampleLikeTorchaudio(mono, audio.sample_rate, 16000);
      // librosa (soxr_hq) vs torchaudio-style sinc: not bit-equal, reported as info
      Info("resample " + std::to_string(audio.sample_rate) + "->16k vs librosa", ours,
           oracle->ToF32("ref_wave16k_raw"));
      std::vector<float> padded = ours;
      const std::size_t patch_len = static_cast<std::size_t>(P * config->vae.chunk_size());
      if (padded.size() % patch_len != 0) {
        padded.resize(padded.size() + patch_len - padded.size() % patch_len, 0.0F);
      }
      std::int64_t frames = 0;
      const std::vector<float> mu =
          vae->Encode(padded.data(), static_cast<std::int64_t>(padded.size()), &frames);
      Info("latents from our resampler", mu, oracle->ToF32("vae_enc_mu"));
    }
  }

  // ------------------------------------------------------------ model ----
  const auto tl = std::chrono::steady_clock::now();
  auto model = vox::cpu::Model::Load(model_dir.string(), &error);
  if (model == nullptr) {
    std::printf("FAIL model load: %s\n", error.c_str());
    return 1;
  }
  std::printf("load model %.1f s\n", Seconds(tl));
  const auto rows = static_cast<std::int64_t>(want_ids.size());
  std::vector<float> feats(static_cast<std::size_t>(rows * PD), 0.0F);
  std::vector<std::uint8_t> audio_mask(static_cast<std::size_t>(rows));
  for (std::int64_t r = 0; r < rows; ++r) {
    audio_mask[static_cast<std::size_t>(r)] = want_mask[static_cast<std::size_t>(r)] != 0 ? 1 : 0;
  }
  for (std::int64_t i = 0; i < ref_len; ++i) {  // [ref_start, ref..., ref_end, text...]
    std::copy(ref_feat.begin() + i * PD, ref_feat.begin() + (i + 1) * PD,
              feats.begin() + (i + 1) * PD);
  }
  const std::vector<float> step_lm = oracle->ToF32("step_lm_hidden");
  const std::vector<float> step_res = oracle->ToF32("step_residual_hidden");
  const std::vector<float> step_dit = oracle->ToF32("step_dit_hidden");
  const std::vector<float> step_cond = oracle->ToF32("step_cond");
  const std::vector<float> step_noise = oracle->ToF32("step_noise");
  const std::vector<float> step_pred = oracle->ToF32("step_pred_feat");
  const std::vector<float> step_cls = oracle->ToF32("step_locenc_cls");
  const std::vector<float> step_curr = oracle->ToF32("step_curr_embed");
  const std::vector<float> step_stop = oracle->ToF32("step_stop_logits");
  const auto steps = static_cast<std::int64_t>(step_pred.size()) / PD;
  const std::int64_t EH = config->encoder.hidden_size;
  vox::FlowOptions flow;
  flow.cfg = static_cast<float>(meta.member_double("cfg", 2.0));
  flow.timesteps = static_cast<std::int32_t>(meta.member_double("timesteps", 10));

  std::vector<float> lm(static_cast<std::size_t>(H));
  std::vector<float> res(static_cast<std::size_t>(H));
  const bool need_prefill = Want(stages, "prefill") || Want(stages, "ar");
  if (need_prefill) {
    const auto t = std::chrono::steady_clock::now();
    vox::cpu::PrefillTrace trace;
    model->PrefillTraced(want_ids.data(), feats.data(), audio_mask.data(), rows, lm.data(),
                         res.data(), &trace);
    std::printf("prefill %lld rows %.2f s\n", static_cast<long long>(rows), Seconds(t));
    Compare("locenc layer0 (row 0)", trace.locenc_l0_row0, oracle->ToF32("locenc_l0_row0"));
    Compare("locenc out", trace.locenc_out, oracle->ToF32("locenc_out"));
    Compare("feat_embed", trace.feat_embed, oracle->ToF32("feat_embed"));
    Compare("combined_embed", trace.combined, oracle->ToF32("combined_embed"));
    Compare("base_lm layer0", trace.base_l0, oracle->ToF32("base_l0"));
    Compare("base_lm out", trace.base_out, oracle->ToF32("base_out"));
    Compare("enc_outputs (FSQ mix)", trace.enc_outputs, oracle->ToF32("enc_outputs"));
    Compare("residual inputs", trace.residual_inputs, oracle->ToF32("residual_inputs"));
    Compare("residual_lm out", trace.residual_out, oracle->ToF32("residual_out"));
    Compare("lm_hidden step 0", lm, Slice(step_lm, 0, H));
    Compare("residual_hidden step 0", res, Slice(step_res, 0, H));
  }

  if (Want(stages, "dit")) {
    Compare("t_span", vox::cpu::Model::TimeSpan(flow.timesteps, 1.0F), oracle->ToF32("t_span"));
    const std::vector<float> ex = oracle->ToF32("est0_x");
    const std::vector<float> emu = oracle->ToF32("est0_mu");
    const std::vector<float> et = oracle->ToF32("est0_t");
    const std::vector<float> ec = oracle->ToF32("est0_cond");
    const std::vector<float> eo = oracle->ToF32("est0_out");
    const auto calls = static_cast<std::int64_t>(et.size()) / 2;
    const auto t = std::chrono::steady_clock::now();
    for (std::int64_t c = 0; c < calls; ++c) {
      const auto x = Transpose(ex.data() + c * 2 * PD, D, P);
      const auto cond = Transpose(ec.data() + c * 2 * PD, D, P);
      std::vector<float> out(static_cast<std::size_t>(2 * PD));
      model->Velocity(x.data(), emu.data() + c * 2 * H, cond.data(),
                      et[static_cast<std::size_t>(c * 2)], out.data());
      std::vector<float> want;
      for (int b = 0; b < 2; ++b) {
        const auto w = Transpose(eo.data() + (c * 2 + b) * PD, D, P);
        want.insert(want.end(), w.begin(), w.end());
      }
      Compare("dit velocity call " + std::to_string(c), out, want);
    }
    std::printf("dit %lld velocity calls %.2f s\n", static_cast<long long>(calls), Seconds(t));
    std::vector<float> mu(static_cast<std::size_t>(H));
    model->DitCondition(Slice(step_lm, 0, H).data(), Slice(step_res, 0, H).data(), mu.data());
    Compare("dit_hidden step 0", mu, Slice(step_dit, 0, H));
    std::vector<float> patch(static_cast<std::size_t>(PD));
    const auto ts = std::chrono::steady_clock::now();
    model->SampleFlow(Slice(step_dit, 0, H).data(), Slice(step_cond, 0, PD).data(),
                      Slice(step_noise, 0, PD).data(), flow, patch.data());
    std::printf("flow sampler (9 CFG evaluations) %.2f s\n", Seconds(ts));
    Compare("flow sampler step 0", patch, Slice(step_pred, 0, PD));
  }

  if (Want(stages, "ar")) {
    std::int64_t limit = steps;
    if (const char* cap = std::getenv("GUFO_VOXCPM_AR_STEPS"); cap != nullptr) {
      limit = std::min<std::int64_t>(limit, std::atoll(cap));
    }
    double worst_pred = 1.0, worst_lm = 1.0, worst_res = 1.0, worst_curr = 1.0, worst_dit = 1.0;
    const auto t = std::chrono::steady_clock::now();
    for (std::int64_t i = 0; i < limit; ++i) {
      double max_abs = 0.0, ref_max = 0.0;
      worst_lm = std::min(worst_lm, Cosine(lm.data(), step_lm.data() + i * H,
                                           static_cast<std::size_t>(H), &max_abs, &ref_max));
      worst_res = std::min(worst_res, Cosine(res.data(), step_res.data() + i * H,
                                             static_cast<std::size_t>(H), &max_abs, &ref_max));
      std::vector<float> mu(static_cast<std::size_t>(H));
      model->DitCondition(lm.data(), res.data(), mu.data());
      worst_dit = std::min(worst_dit, Cosine(mu.data(), step_dit.data() + i * H,
                                             static_cast<std::size_t>(H), &max_abs, &ref_max));
      std::vector<float> patch(static_cast<std::size_t>(PD));
      model->SampleFlow(mu.data(), step_cond.data() + i * PD, step_noise.data() + i * PD, flow,
                        patch.data());
      worst_pred = std::min(worst_pred, Cosine(patch.data(), step_pred.data() + i * PD,
                                               static_cast<std::size_t>(PD), &max_abs, &ref_max));
      float logits[2];
      model->StopLogits(lm.data(), logits);
      const bool stop = logits[1] > logits[0];
      const bool want_stop = step_stop[static_cast<std::size_t>(2 * i + 1)] >
                             step_stop[static_cast<std::size_t>(2 * i)];
      if (stop != want_stop) {
        Expect("stop decision step " + std::to_string(i), false);
      }
      // teacher forcing: feed the oracle patch
      std::vector<float> cls(static_cast<std::size_t>(EH));
      model->LocalEncode(step_pred.data() + i * PD, 1, cls.data());
      worst_curr = std::min(worst_curr, Cosine(cls.data(), step_cls.data() + i * EH,
                                               static_cast<std::size_t>(EH), &max_abs, &ref_max));
      if (i + 1 < steps) {
        model->Advance(step_pred.data() + i * PD, lm.data(), res.data());
      }
    }
    std::printf("teacher-forced AR %lld steps %.1f s (%.2f s/step)\n",
                static_cast<long long>(limit), Seconds(t), Seconds(t) / std::max<std::int64_t>(1, limit));
    Expect("AR lm_hidden (worst step)", worst_lm >= kMinCos, "cos=" + std::to_string(worst_lm));
    Expect("AR residual_hidden (worst step)", worst_res >= kMinCos, "cos=" + std::to_string(worst_res));
    Expect("AR dit_hidden (worst step)", worst_dit >= kMinCos, "cos=" + std::to_string(worst_dit));
    Expect("AR flow patch (worst step)", worst_pred >= kMinCos, "cos=" + std::to_string(worst_pred));
    Expect("AR locenc cls (worst step)", worst_curr >= kMinCos, "cos=" + std::to_string(worst_curr));
    (void)step_curr;
  }

  if (Want(stages, "vae_dec")) {
    const std::vector<float> latent = oracle->ToF32("latent");
    const std::int64_t frames = static_cast<std::int64_t>(latent.size()) / D;
    vox::cpu::AudioVae::DecoderState state;
    vox::cpu::AudioVae::DecodeTrace trace;
    const auto t = std::chrono::steady_clock::now();
    const std::vector<float> wave = vae->DecodeChunk(&state, latent.data(), frames, &trace);
    std::printf("vae decode %lld frames %.2f s\n", static_cast<long long>(frames), Seconds(t));
    Compare("vae_dec model.1", trace.model1, oracle->ToF32("vae_dec_model1"));
    Compare("vae_dec model.2 (first block)", trace.model2, oracle->ToF32("vae_dec_model2"));
    const std::int64_t samples = static_cast<std::int64_t>(wave.size());
    Compare("vae_dec last block (4000 samples)", Columns(trace.last_block, 32, samples, 4000),
            oracle->ToF32("vae_dec_model7_head"));
    Compare("vae_dec wave", wave, oracle->ToF32("wave"));
    // streaming: one AR patch (4 frames) at a time must equal the full decode
    vox::cpu::AudioVae::DecoderState stream;
    std::vector<float> chunked;
    for (std::int64_t f = 0; f < frames; f += P) {
      const std::int64_t n = std::min(P, frames - f);
      std::vector<float> z(static_cast<std::size_t>(D * n));
      for (std::int64_t c = 0; c < D; ++c) {
        for (std::int64_t j = 0; j < n; ++j) {
          z[static_cast<std::size_t>(c * n + j)] = latent[static_cast<std::size_t>(c * frames + f + j)];
        }
      }
      const auto part = vae->DecodeChunk(&stream, z.data(), n);
      chunked.insert(chunked.end(), part.begin(), part.end());
    }
    Compare("vae_dec streaming == full", chunked, wave, 0.9999999);
  }

  if (Want(stages, "free")) {
    model.reset();
    vae.reset();
    auto engine = vox::Engine::Create(model_dir.string(), &error);
    if (engine == nullptr) {
      std::printf("FAIL engine: %s\n", error.c_str());
      return 1;
    }
    vox::SynthesisRequest request;
    request.text = text;
    request.reference_audio = oracle->ToF32("ref_wave16k_raw");
    request.reference_sample_rate = 16000;
    request.cfg = flow.cfg;
    request.timesteps = flow.timesteps;
    request.retry_badcase = false;
    // The oracle AR loop is capped (VOXCPM_ORACLE_STEPS); without its natural stop, cap ours too.
    if (!meta.find("stopped")->as_bool()) {
      request.max_len = static_cast<std::int32_t>(steps);
    }
    request.noise_source = [&](std::int32_t, std::int64_t step, float* noise) {
      if (step < steps) {
        std::copy(step_noise.begin() + step * PD, step_noise.begin() + (step + 1) * PD, noise);
      } else {
        std::fill(noise, noise + PD, 0.0F);
      }
    };
    vox::SynthesisResult result;
    const auto t = std::chrono::steady_clock::now();
    if (!engine->Generate(request, {}, &result, &error)) {
      Expect("free run", false, error);
    } else {
      const double seconds = Seconds(t);
      const double audio_s = static_cast<double>(result.samples.size()) / result.sample_rate;
      std::printf("free run %.1f s for %.2f s audio (RTF %.2f, %s)\n", seconds, audio_s,
                  seconds / std::max(1e-9, audio_s), engine->backend_name());
      Expect("free run input ids", result.input_ids == want_ids);
      Expect("free run step count", result.steps == steps,
             std::to_string(result.steps) + " vs " + std::to_string(steps));
      const std::size_t n = std::min(result.patches.size(), step_pred.size());
      Compare("free run latents (common steps)", result.patches.data(), step_pred.data(), n, 0.999);
      const std::vector<float> want_wave = oracle->ToF32("wave");
      const std::size_t m = std::min(result.samples.size(), want_wave.size());
      Compare("free run wave (common samples)", result.samples.data(), want_wave.data(), m, 0.99);
      if (const char* out = std::getenv("GUFO_VOXCPM_OUT"); out != nullptr) {
        WriteWav(std::filesystem::path(out) / "cpp_free.wav", result.samples, result.sample_rate);
      }
    }
  }

  std::printf("total %.1f s, %d failure(s)\n", Seconds(t0), g_failures);
  PrintPeakMemory();
  return g_failures == 0 ? 0 : 1;
}
