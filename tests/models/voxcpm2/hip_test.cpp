// VoxCPM2 gfx1151 HIP runtime vs. the Python CPU fp32 oracle and the CPU
// runtime, plus a before/after benchmark: local encoder, DiT velocity, flow
// sampler (eager/graph/replay) + prefill + teacher-forced AR steps with the
// old 64x64 tile GEMM ("vorher") and the split-K skinny GEMM ("nachher"),
// AudioVAE encoder/streaming decoder on the GPU vs. oracle and CPU timing,
// engine free run with the oracle noise and a natural streamed sentence
// (RTF, ms per step) for both variants (vorher: tile GEMM + CPU AudioVAE).
//
//   GUFO_VOXCPM_MODEL  checkpoint directory (openbmb/VoxCPM2)
//   GUFO_VOXCPM_REF    directory with oracle.safetensors + oracle.json
//   GUFO_VOXCPM_OUT    optional directory for hip_free.wav
//
// Exits 77 (skip) without model, oracle or HIP device.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/ming_tts/safetensors.hpp"
#include "src/models/voxcpm2/cpu/audio_vae.hpp"
#include "src/models/voxcpm2/cpu/model.hpp"
#include "src/models/voxcpm2/hip/audio_vae.hpp"
#include "src/models/voxcpm2/hip/model.hpp"
#include "src/models/voxcpm2/synthesis.hpp"

namespace vox = gufo::models::voxcpm2;
namespace json = gufo::json;
using gufo::models::ming_tts::SafetensorsFile;

namespace {

int g_failures = 0;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

double Cos(const float* a, const float* b, std::size_t n) {
  double dot = 0.0, a2 = 0.0, b2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    a2 += static_cast<double>(a[i]) * a[i];
    b2 += static_cast<double>(b[i]) * b[i];
  }
  return (a2 == 0.0 && b2 == 0.0) ? 1.0 : dot / std::sqrt(a2 * b2 + 1e-300);
}

void Check(const std::string& name, double cos, double min_cos) {
  const bool ok = std::isfinite(cos) && cos >= min_cos;
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-40s cos=%.7f (min %.4f)\n", ok ? "OK" : "FAIL", name.c_str(), cos, min_cos);
  std::fflush(stdout);
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::vector<float> Transpose(const float* src, std::int64_t rows, std::int64_t cols) {
  std::vector<float> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(c * rows + r)] = src[r * cols + c];
    }
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
  if (!vox::hip::DeviceAvailable()) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  const std::filesystem::path ref_dir(ref_env);
  std::string error;
  auto oracle = SafetensorsFile::Open(ref_dir / "oracle.safetensors", &error);
  if (oracle == nullptr) {
    std::printf("SKIP: oracle missing: %s\n", error.c_str());
    return 77;
  }
  const json::Value meta = json::parse(ReadText(ref_dir / "oracle.json"));
  auto cpu = vox::cpu::Model::Load(model_env, &error);
  if (cpu == nullptr) {
    std::printf("FAIL cpu model: %s\n", error.c_str());
    return 1;
  }
  const auto tc = std::chrono::steady_clock::now();
  auto gpu = vox::hip::Model::Create(*cpu, &error);
  if (gpu == nullptr) {
    std::printf("FAIL hip model: %s\n", error.c_str());
    return 1;
  }
  std::printf("upload %.1f s\n", Seconds(tc));
  const vox::ModelConfig& c = cpu->config();
  const std::int64_t P = c.patch_size, D = c.feat_dim, H = c.base_lm.hidden_size;
  const std::int64_t PD = P * D, EH = c.encoder.hidden_size;
  vox::FlowOptions flow;
  flow.cfg = static_cast<float>(meta.member_double("cfg", 2.0));
  flow.timesteps = static_cast<std::int32_t>(meta.member_double("timesteps", 10));

  const auto ids = oracle->ToI64("input_ids");
  const auto mask64 = oracle->ToI64("audio_mask");
  const auto ref_feat = oracle->ToF32("ref_feat");
  const auto step_lm = oracle->ToF32("step_lm_hidden");
  const auto step_res = oracle->ToF32("step_residual_hidden");
  const auto step_dit = oracle->ToF32("step_dit_hidden");
  const auto step_cond = oracle->ToF32("step_cond");
  const auto step_noise = oracle->ToF32("step_noise");
  const auto step_pred = oracle->ToF32("step_pred_feat");
  const auto step_cls = oracle->ToF32("step_locenc_cls");
  const auto steps = static_cast<std::int64_t>(step_pred.size()) / PD;
  const auto rows = static_cast<std::int64_t>(ids.size());
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(rows));
  std::vector<float> feats(static_cast<std::size_t>(rows * PD), 0.0F);
  for (std::int64_t r = 0; r < rows; ++r) {
    mask[static_cast<std::size_t>(r)] = mask64[static_cast<std::size_t>(r)] != 0 ? 1 : 0;
  }
  const std::int64_t ref_len = static_cast<std::int64_t>(ref_feat.size()) / PD;
  std::copy(ref_feat.begin(), ref_feat.end(), feats.begin() + PD);
  (void)ref_len;

  // local encoder
  {
    std::vector<float> cls(static_cast<std::size_t>(EH));
    gpu->LocalEncode(step_pred.data(), 1, cls.data());
    Check("locenc cls (step 0 patch)", Cos(cls.data(), step_cls.data(), cls.size()), 0.999);
  }
  // velocity
  {
    const auto ex = oracle->ToF32("est0_x");
    const auto emu = oracle->ToF32("est0_mu");
    const auto et = oracle->ToF32("est0_t");
    const auto ec = oracle->ToF32("est0_cond");
    const auto eo = oracle->ToF32("est0_out");
    const auto x = Transpose(ex.data(), D, P);
    const auto cond = Transpose(ec.data(), D, P);
    std::vector<float> out(static_cast<std::size_t>(2 * PD));
    gpu->Velocity(x.data(), emu.data(), cond.data(), et[0], out.data());
    std::vector<float> want;
    for (int b = 0; b < 2; ++b) {
      const auto w = Transpose(eo.data() + b * PD, D, P);
      want.insert(want.end(), w.begin(), w.end());
    }
    Check("dit velocity call 0", Cos(out.data(), want.data(), out.size()), 0.999);
  }
  // flow sampler, prefill and teacher-forced AR for both GEMM paths:
  // "vorher" = 64x64 tile GEMM (run of 03.11. 01:53), "nachher" = split-K skinny GEMM.
  for (const bool skinny : {false, true}) {
    gpu->SetSkinnyGemm(skinny);
    const std::string tag = skinny ? "[nachher skinny] " : "[vorher tile] ";
    {
      std::vector<float> lm(step_lm.begin(), step_lm.begin() + H);
      std::vector<float> res(step_res.begin(), step_res.begin() + H);
      std::vector<float> patch(static_cast<std::size_t>(PD));
      for (int run = 0; run < 3; ++run) {  // eager, graph capture, graph replay
        const auto t = std::chrono::steady_clock::now();
        gpu->SamplePatch(lm.data(), res.data(), step_cond.data(), step_noise.data(), flow,
                         patch.data());
        const double ms = Seconds(t) * 1000.0;
        char name[160];
        std::snprintf(name, sizeof(name), "%sflow sampler step 0 run %d (%.1f ms)", tag.c_str(),
                      run, ms);
        Check(name, Cos(patch.data(), step_pred.data(), patch.size()), 0.999);
      }
    }
    std::vector<float> lm(static_cast<std::size_t>(H)), res(static_cast<std::size_t>(H));
    const auto t = std::chrono::steady_clock::now();
    gpu->Prefill(ids.data(), feats.data(), mask.data(), rows, lm.data(), res.data());
    std::printf("%sprefill %lld rows %.1f ms\n", tag.c_str(), static_cast<long long>(rows),
                Seconds(t) * 1000.0);
    Check(tag + "prefill lm_hidden", Cos(lm.data(), step_lm.data(), lm.size()), 0.999);
    Check(tag + "prefill residual_hidden", Cos(res.data(), step_res.data(), res.size()), 0.999);
    double worst_lm = 1.0, worst_res = 1.0, worst_patch = 1.0;
    double t_sample = 0.0, t_advance = 0.0, t_stop = 0.0;
    for (std::int64_t i = 0; i < steps; ++i) {
      worst_lm = std::min(worst_lm, Cos(lm.data(), step_lm.data() + i * H, static_cast<std::size_t>(H)));
      worst_res = std::min(worst_res, Cos(res.data(), step_res.data() + i * H, static_cast<std::size_t>(H)));
      std::vector<float> patch(static_cast<std::size_t>(PD));
      auto ts = std::chrono::steady_clock::now();
      gpu->SamplePatch(lm.data(), res.data(), step_cond.data() + i * PD,
                       step_noise.data() + i * PD, flow, patch.data());
      t_sample += Seconds(ts);
      worst_patch = std::min(worst_patch, Cos(patch.data(), step_pred.data() + i * PD, patch.size()));
      float logits[2];
      ts = std::chrono::steady_clock::now();
      gpu->StopLogits(lm.data(), logits);
      t_stop += Seconds(ts);
      if (i + 1 < steps) {
        ts = std::chrono::steady_clock::now();
        gpu->Advance(step_pred.data() + i * PD, lm.data(), res.data());
        t_advance += Seconds(ts);
      }
    }
    std::printf("%sper AR step: sampler %.1f ms, stop %.1f ms, advance %.1f ms\n", tag.c_str(),
                1000.0 * t_sample / steps, 1000.0 * t_stop / steps,
                1000.0 * t_advance / std::max<std::int64_t>(1, steps - 1));
    Check(tag + "AR lm_hidden (worst step)", worst_lm, 0.999);
    Check(tag + "AR residual_hidden (worst step)", worst_res, 0.999);
    Check(tag + "AR flow patch (worst step)", worst_patch, 0.999);
  }
  (void)step_dit;

  // AudioVAE on the GPU vs. the oracle and the CPU port
  {
    auto cpu_vae = vox::cpu::AudioVae::Load(model_env, c.vae, &error);
    auto gpu_vae = cpu_vae != nullptr ? vox::hip::AudioVae::Create(*cpu_vae, &error) : nullptr;
    if (gpu_vae == nullptr) {
      std::printf("FAIL hip AudioVAE: %s\n", error.c_str());
      ++g_failures;
    } else {
      const auto wave16k = oracle->ToF32("ref_wave16k");
      const auto want_mu = oracle->ToF32("vae_enc_mu");
      std::int64_t frames = 0;
      for (int run = 0; run < 2; ++run) {
        const auto t = std::chrono::steady_clock::now();
        const auto mu = gpu_vae->Encode(wave16k.data(), static_cast<std::int64_t>(wave16k.size()), &frames);
        char name[160];
        std::snprintf(name, sizeof(name), "hip vae encode run %d (%.1f ms, %.1f s audio)", run,
                      Seconds(t) * 1000.0, static_cast<double>(wave16k.size()) / 16000.0);
        Check(name, mu.size() == want_mu.size() ? Cos(mu.data(), want_mu.data(), mu.size()) : 0.0, 0.9999);
      }
      {
        const auto t = std::chrono::steady_clock::now();
        std::int64_t cpu_frames = 0;
        (void)cpu_vae->Encode(wave16k.data(), static_cast<std::int64_t>(wave16k.size()), &cpu_frames);
        std::printf("cpu vae encode %.1f ms (vorher)\n", Seconds(t) * 1000.0);
      }
      const auto latent = oracle->ToF32("latent");
      const auto want_wave = oracle->ToF32("wave");
      const std::int64_t lf = static_cast<std::int64_t>(latent.size()) / D;
      gpu_vae->ResetDecoder();
      auto t = std::chrono::steady_clock::now();
      const auto full = gpu_vae->DecodeChunk(latent.data(), lf);
      std::printf("hip vae decode %lld frames in one call %.1f ms\n", static_cast<long long>(lf),
                  Seconds(t) * 1000.0);
      Check("hip vae decode wave", full.size() == want_wave.size()
                                       ? Cos(full.data(), want_wave.data(), full.size()) : 0.0, 0.9999);
      // streaming, one AR patch (4 frames) per call, like the engine
      gpu_vae->ResetDecoder();
      std::vector<float> chunked;
      double t_chunks = 0.0;
      for (std::int64_t f = 0; f < lf; f += P) {
        const std::int64_t n = std::min(P, lf - f);
        std::vector<float> z(static_cast<std::size_t>(D * n));
        for (std::int64_t ch = 0; ch < D; ++ch) {
          for (std::int64_t j = 0; j < n; ++j) {
            z[static_cast<std::size_t>(ch * n + j)] = latent[static_cast<std::size_t>(ch * lf + f + j)];
          }
        }
        t = std::chrono::steady_clock::now();
        const auto part = gpu_vae->DecodeChunk(z.data(), n);
        t_chunks += Seconds(t);
        chunked.insert(chunked.end(), part.begin(), part.end());
      }
      std::printf("hip vae decode per AR patch (4 frames, 160 ms audio): %.1f ms\n",
                  1000.0 * t_chunks / static_cast<double>((lf + P - 1) / P));
      Check("hip vae streaming == oracle wave", chunked.size() == want_wave.size()
                ? Cos(chunked.data(), want_wave.data(), chunked.size()) : 0.0, 0.9999);
      vox::cpu::AudioVae::DecoderState state;
      t = std::chrono::steady_clock::now();
      for (std::int64_t f = 0; f < lf; f += P) {
        const std::int64_t n = std::min(P, lf - f);
        std::vector<float> z(static_cast<std::size_t>(D * n));
        for (std::int64_t ch = 0; ch < D; ++ch) {
          for (std::int64_t j = 0; j < n; ++j) {
            z[static_cast<std::size_t>(ch * n + j)] = latent[static_cast<std::size_t>(ch * lf + f + j)];
          }
        }
        (void)cpu_vae->DecodeChunk(&state, z.data(), n);
      }
      std::printf("cpu vae decode per AR patch %.1f ms (vorher)\n",
                  1000.0 * Seconds(t) / static_cast<double>((lf + P - 1) / P));
    }
  }

  // engine: oracle-noise free run (correctness) and a natural sentence (RTF), both variants
  {
    cpu.reset();
    gpu.reset();
    auto engine = vox::Engine::Create(model_env, &error);
    if (engine == nullptr) {
      std::printf("FAIL engine: %s\n", error.c_str());
      return 1;
    }
    const auto ref_raw = oracle->ToF32("ref_wave16k_raw");
    const auto want_wave = oracle->ToF32("wave");
    const char* out = std::getenv("GUFO_VOXCPM_OUT");
    for (const bool after : {false, true}) {
      engine->SetSkinnyGemm(after);
      engine->SetVaeOnGpu(after);
      const std::string tag = after ? "[nachher] " : "[vorher] ";
      std::printf("%sAR backend %s, VAE %s\n", tag.c_str(), engine->backend_name(),
                  engine->vae_backend_name());
      for (int run = 0; run < 2; ++run) {  // run 0: reference encode included, run 1: cached
        if (run == 0) {
          engine->ClearAudioCache();
        }
        vox::SynthesisRequest request;
        request.text = meta.member_str("text");
        request.reference_audio = ref_raw;
        request.reference_sample_rate = 16000;
        request.cfg = flow.cfg;
        request.timesteps = flow.timesteps;
        request.retry_badcase = false;
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
          std::printf("FAIL %sfree run: %s\n", tag.c_str(), error.c_str());
          ++g_failures;
          continue;
        }
        const double seconds = Seconds(t);
        const double audio_s = static_cast<double>(result.samples.size()) / result.sample_rate;
        std::printf("%sfree run %s: %lld steps (oracle %lld), %.2f s audio in %.2f s, RTF %.2f\n",
                    tag.c_str(), run == 0 ? "with reference encode" : "reference cached",
                    static_cast<long long>(result.steps), static_cast<long long>(steps), audio_s,
                    seconds, seconds / audio_s);
        if (run == 1) {
          const bool same_steps = std::llabs(result.steps - steps) <= 2;
          if (!same_steps) {
            ++g_failures;
          }
          std::printf("%-4s %sfree run step count within +-2\n", same_steps ? "OK" : "FAIL", tag.c_str());
          const std::size_t n = std::min(result.patches.size(), step_pred.size());
          Check(tag + "free run latents (common steps)", Cos(result.patches.data(), step_pred.data(), n), 0.98);
          const std::size_t w = std::min(result.samples.size(), want_wave.size());
          Check(tag + "free run wave vs oracle (common samples)", Cos(result.samples.data(), want_wave.data(), w), 0.95);
          if (out != nullptr) {
            WriteWav(std::filesystem::path(out) / (after ? "hip_free_nachher.wav" : "hip_free_vorher.wav"),
                     result.samples, result.sample_rate);
          }
        }
      }
      // natural sentence with its own stop (streaming as served: one chunk per AR step)
      vox::SynthesisRequest request;
      request.text = "Guten Morgen! Heute scheint die Sonne, und wir fahren ans Meer. "
                     "Ich freue mich schon sehr darauf.";
      request.instruction = "fröhlich und begeistert";
      request.reference_audio = ref_raw;
      request.reference_sample_rate = 16000;
      request.seed = 4242;
      std::size_t chunks = 0;
      double first_chunk_s = -1.0;
      const auto t = std::chrono::steady_clock::now();
      request.on_audio = [&](std::span<const float>) {
        if (chunks++ == 0) {
          first_chunk_s = Seconds(t);
        }
        return true;
      };
      vox::SynthesisResult result;
      if (!engine->Generate(request, {}, &result, &error)) {
        std::printf("FAIL %ssentence: %s\n", tag.c_str(), error.c_str());
        ++g_failures;
        continue;
      }
      const double seconds = Seconds(t);
      const double audio_s = static_cast<double>(result.samples.size()) / result.sample_rate;
      std::printf("%ssentence (streamed, reference cached): %lld steps, %.2f s audio in %.2f s, "
                  "RTF %.2f, first chunk after %.0f ms, %.1f ms per step\n",
                  tag.c_str(), static_cast<long long>(result.steps), audio_s, seconds,
                  seconds / audio_s, first_chunk_s * 1000.0,
                  1000.0 * seconds / std::max<std::int64_t>(1, result.steps));
      if (out != nullptr) {
        WriteWav(std::filesystem::path(out) / (after ? "hip_satz_nachher.wav" : "hip_satz_vorher.wav"),
                 result.samples, result.sample_rate);
      }
    }
  }
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
