// VoxCPM2 gfx1151 HIP runtime vs. the Python CPU fp32 oracle and the CPU
// runtime: local encoder, DiT velocity, flow sampler (graph and eager),
// prefill, teacher-forced AR steps, free running with the oracle noise
// (step count, latents, wave) and timings per AR step.
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
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/ming_tts/safetensors.hpp"
#include "src/models/voxcpm2/cpu/model.hpp"
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
  // flow sampler: eager, graph capture, graph replay
  {
    std::vector<float> lm(step_lm.begin(), step_lm.begin() + H);
    std::vector<float> res(step_res.begin(), step_res.begin() + H);
    std::vector<float> patch(static_cast<std::size_t>(PD));
    for (int run = 0; run < 3; ++run) {
      const auto t = std::chrono::steady_clock::now();
      gpu->SamplePatch(lm.data(), res.data(), step_cond.data(), step_noise.data(), flow,
                       patch.data());
      const double ms = Seconds(t) * 1000.0;
      Check("flow sampler step 0 run " + std::to_string(run) + " (" + std::to_string(ms) + " ms)",
            Cos(patch.data(), step_pred.data(), patch.size()), 0.999);
    }
  }
  // prefill + teacher-forced AR
  std::vector<float> lm(static_cast<std::size_t>(H)), res(static_cast<std::size_t>(H));
  {
    const auto t = std::chrono::steady_clock::now();
    gpu->Prefill(ids.data(), feats.data(), mask.data(), rows, lm.data(), res.data());
    std::printf("prefill %lld rows %.1f ms\n", static_cast<long long>(rows), Seconds(t) * 1000.0);
    Check("prefill lm_hidden", Cos(lm.data(), step_lm.data(), lm.size()), 0.999);
    Check("prefill residual_hidden", Cos(res.data(), step_res.data(), res.size()), 0.999);
    double worst_lm = 1.0, worst_res = 1.0, worst_patch = 1.0;
    double t_sample = 0.0, t_advance = 0.0;
    for (std::int64_t i = 0; i < steps; ++i) {
      worst_lm = std::min(worst_lm, Cos(lm.data(), step_lm.data() + i * H, static_cast<std::size_t>(H)));
      worst_res = std::min(worst_res, Cos(res.data(), step_res.data() + i * H, static_cast<std::size_t>(H)));
      std::vector<float> patch(static_cast<std::size_t>(PD));
      auto ts = std::chrono::steady_clock::now();
      gpu->SamplePatch(lm.data(), res.data(), step_cond.data() + i * PD,
                       step_noise.data() + i * PD, flow, patch.data());
      t_sample += Seconds(ts);
      worst_patch = std::min(worst_patch, Cos(patch.data(), step_pred.data() + i * PD, patch.size()));
      if (i + 1 < steps) {
        ts = std::chrono::steady_clock::now();
        gpu->Advance(step_pred.data() + i * PD, lm.data(), res.data());
        t_advance += Seconds(ts);
      }
    }
    std::printf("per AR step: sampler %.1f ms, advance %.1f ms\n", 1000.0 * t_sample / steps,
                1000.0 * t_advance / std::max<std::int64_t>(1, steps - 1));
    Check("AR lm_hidden (worst step)", worst_lm, 0.999);
    Check("AR residual_hidden (worst step)", worst_res, 0.999);
    Check("AR flow patch (worst step)", worst_patch, 0.999);
    (void)step_dit;
  }
  // free running through the engine (GPU backend), oracle noise
  {
    cpu.reset();
    gpu.reset();
    auto engine = vox::Engine::Create(model_env, &error);
    if (engine == nullptr) {
      std::printf("FAIL engine: %s\n", error.c_str());
      return 1;
    }
    vox::SynthesisRequest request;
    request.text = meta.member_str("text");
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
      std::printf("FAIL free run: %s\n", error.c_str());
      ++g_failures;
    } else {
      const double seconds = Seconds(t);
      const double audio_s = static_cast<double>(result.samples.size()) / result.sample_rate;
      std::printf("free run (%s): %lld steps (oracle %lld), %.2f s audio in %.2f s, RTF %.2f\n",
                  engine->backend_name(), static_cast<long long>(result.steps),
                  static_cast<long long>(steps), audio_s, seconds, seconds / audio_s);
      const bool same_steps = std::llabs(result.steps - steps) <= 2;
      if (!same_steps) {
        ++g_failures;
      }
      std::printf("%-4s free run step count within +-2\n", same_steps ? "OK" : "FAIL");
      const std::size_t n = std::min(result.patches.size(), step_pred.size());
      Check("free run latents (common steps)", Cos(result.patches.data(), step_pred.data(), n), 0.98);
      if (const char* out = std::getenv("GUFO_VOXCPM_OUT"); out != nullptr) {
        WriteWav(std::filesystem::path(out) / "hip_free.wav", result.samples, result.sample_rate);
      }
    }
  }
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
