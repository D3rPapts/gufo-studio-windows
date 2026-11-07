// Ming-3.0-tts gfx1151 HIP runtime vs. the CPU fp32 oracle and the CPU
// runtime, stage by stage, plus a timed free-running request.
//
//   GUFO_MING_MODEL  checkpoint directory
//   GUFO_MING_REF    directory with oracle.safetensors + oracle.json
//   GUFO_MING_WAV_OUT optional path for the free-running WAV
//
// Thresholds: paths with FP32 activations (GEMV: backbone decode, DiT,
// aggregator) cosine >= 0.999; paths through rocBLAS with BF16 activations
// (prefill, VAE decoder) >= 0.995 on hidden states and >= 0.98 on the
// waveform. Exits 77 without a model/oracle or HIP device.
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
#include "src/models/ming_tts/cpu/model.hpp"
#include "src/models/ming_tts/hip/model.hpp"
#include "src/models/ming_tts/safetensors.hpp"
#include "src/models/ming_tts/synthesis.hpp"

namespace ming = gufo::models::ming_tts;
namespace json = gufo::json;

namespace {

int g_failures = 0;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

double Cos(const float* a, const float* b, std::size_t n) {
  double d = 0, x = 0, y = 0;
  for (std::size_t i = 0; i < n; ++i) {
    d += static_cast<double>(a[i]) * b[i];
    x += static_cast<double>(a[i]) * a[i];
    y += static_cast<double>(b[i]) * b[i];
  }
  return (x == 0 && y == 0) ? 1.0 : d / std::sqrt(x * y + 1e-300);
}

void Compare(const std::string& name, const std::vector<float>& got,
             const std::vector<float>& want, double min_cos) {
  const std::size_t n = std::min(got.size(), want.size());
  const double cos = n == 0 ? 0.0 : Cos(got.data(), want.data(), n);
  double max_abs = 0;
  for (std::size_t i = 0; i < n; ++i) {
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
  }
  const bool ok = got.size() == want.size() && std::isfinite(cos) && cos >= min_cos;
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-38s cos=%.6f max_abs=%.3g (n=%zu/%zu, gate %.3f)\n",
              ok ? "OK" : "FAIL", name.c_str(), cos, max_abs, got.size(), want.size(),
              min_cos);
  std::fflush(stdout);
}

void Expect(const std::string& name, bool ok, const std::string& detail) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-38s %s\n", ok ? "OK" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::vector<float> Envelope(const std::vector<float>& wav) {
  std::vector<float> env;
  for (std::size_t start = 0; start + 882 <= wav.size(); start += 882) {
    double sum = 0;
    for (std::size_t i = 0; i < 882; ++i) {
      sum += static_cast<double>(wav[start + i]) * wav[start + i];
    }
    env.push_back(static_cast<float>(std::sqrt(sum / 882.0)));
  }
  return env;
}

// Free-running AR on a backend with the oracle's start noise; returns the
// latents. Mirrors Engine::Generate (stop > 0.5 after step 3).
std::vector<float> FreeRun(ming::Backend& backend, const std::vector<float>& embeds,
                           std::int64_t rows, std::int64_t hidden, std::int64_t lat,
                           const std::vector<float>& noises, double* ms_per_step) {
  backend.ResetBackbone();
  std::vector<float> normed(static_cast<std::size_t>(rows * hidden));
  backend.RunBackbone(embeds.data(), rows, normed.data());
  std::vector<float> z(normed.end() - hidden, normed.end());
  std::vector<float> history(static_cast<std::size_t>(lat), 0.0F), latent(history.size()),
      next(static_cast<std::size_t>(hidden)), out;
  const auto t = std::chrono::steady_clock::now();
  std::int64_t step = 0;
  for (; step < 512 - rows; ++step) {
    const std::size_t at = static_cast<std::size_t>(step * lat);
    std::vector<float> noise(static_cast<std::size_t>(lat), 0.0F);
    for (std::size_t i = 0; i < noise.size() && at + i < noises.size(); ++i) {
      noise[i] = noises[at + i];
    }
    backend.SampleFlow(z.data(), history.data(), noise.data(), nullptr, {}, latent.data());
    out.insert(out.end(), latent.begin(), latent.end());
    if (backend.StopAt(z.data()) > 0.5F && step > 3) {
      ++step;
      break;
    }
    backend.AggregateLatent(latent.data(), next.data());
    backend.RunBackbone(next.data(), 1, z.data());
    history = latent;
  }
  *ms_per_step = Seconds(t) * 1e3 / static_cast<double>(std::max<std::int64_t>(1, step));
  return out;
}

// Per-step latent cosine against a reference trajectory.
void Divergence(const std::string& name, const std::vector<float>& got,
                const std::vector<float>& want, std::size_t lat) {
  std::string curve;
  long first_bad = -1;
  const std::size_t steps = std::min(got.size(), want.size()) / lat;
  for (std::size_t s = 0; s < steps; ++s) {
    const double c = Cos(got.data() + s * lat, want.data() + s * lat, lat);
    if (first_bad < 0 && c < 0.999) {
      first_bad = static_cast<long>(s);
    }
    if (s % 4 == 0 || s + 1 == steps) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), " %zu:%.4f", s, c);
      curve += buffer;
    }
  }
  std::printf("     %s: %zu vs %zu steps, first step < 0.999: %ld;%s\n", name.c_str(),
              got.size() / lat, want.size() / lat, first_bad, curve.c_str());
  std::fflush(stdout);
}

}  // namespace

int main() {
  const char* model_env = std::getenv("GUFO_MING_MODEL");
  const char* ref_env = std::getenv("GUFO_MING_REF");
  if (model_env == nullptr || ref_env == nullptr) {
    std::printf("SKIP: set GUFO_MING_MODEL and GUFO_MING_REF\n");
    return 77;
  }
  if (!ming::hip::DeviceAvailable()) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  const std::filesystem::path ref_dir(ref_env);
  std::string error;
  auto oracle = ming::SafetensorsFile::Open(ref_dir / "oracle.safetensors", &error);
  if (oracle == nullptr) {
    std::printf("SKIP: %s\n", error.c_str());
    return 77;
  }
  const json::Value meta = json::parse(ReadText(ref_dir / "oracle.json"));
  auto t = std::chrono::steady_clock::now();
  auto cpu_model = ming::cpu::Model::Load(model_env, &error);
  if (cpu_model == nullptr) {
    std::printf("FAIL cpu model: %s\n", error.c_str());
    return 1;
  }
  auto gpu = ming::hip::Model::Create(*cpu_model, &error);
  if (gpu == nullptr) {
    std::printf("FAIL hip model: %s\n", error.c_str());
    return 1;
  }
  std::printf("     load + upload %.1f s\n", Seconds(t));
  const std::int64_t hidden = cpu_model->config().llm.hidden_size;
  const std::int64_t lat = cpu_model->config().latent_dim;
  const auto h = static_cast<std::size_t>(hidden);
  const auto l = static_cast<std::size_t>(lat);
  const std::vector<float> embeds = oracle->ToF32("inputs_embeds");
  const auto rows = static_cast<std::int64_t>(embeds.size()) / hidden;
  const std::vector<float> cond = oracle->ToF32("cond");
  const std::vector<float> latents = oracle->ToF32("latents");
  const std::vector<float> noises = oracle->ToF32("noise");
  const std::vector<float> agg = oracle->ToF32("agg");
  const std::vector<float> stop = oracle->ToF32("stop_prob");
  const auto steps = static_cast<std::int64_t>(latents.size()) / lat;

  // --- backbone prefill (rocBLAS, BF16 activations)
  gpu->ResetBackbone();
  std::vector<float> normed(static_cast<std::size_t>(rows) * h);
  t = std::chrono::steady_clock::now();
  gpu->RunBackbone(embeds.data(), rows, normed.data());
  std::printf("     prefill %lld rows %.1f ms\n", static_cast<long long>(rows), Seconds(t) * 1e3);
  Compare("prefill final normed vs oracle", normed, oracle->ToF32("prefill_hidden_last"), 0.995);

  // --- DiT velocity (GEMV, FP32 activations)
  const std::vector<float> zero(l, 0.0F);
  for (int k = 0; k < 10; k += 3) {
    const std::string s = std::to_string(k);
    std::vector<float> out(2 * l);
    gpu->FlowVelocity(oracle->ToF32("dit0_x_" + s).data(), oracle->ToF32("dit0_t_" + s).at(0),
                      cond.data(), zero.data(), out.data());
    Compare("DiT velocity eval " + s, out, oracle->ToF32("dit0_out_" + s), 0.999);
  }

  // --- teacher-forced AR on the GPU
  std::vector<float> got_cond(normed.end() - hidden, normed.end());
  std::vector<float> got_lat, got_agg, got_stop, want_stop;
  std::vector<float> history(l, 0.0F), latent(l), next(h), z(h);
  double sample_ms = 0, agg_ms = 0, backbone_ms = 0;
  for (std::int64_t s = 0; s < steps; ++s) {
    t = std::chrono::steady_clock::now();
    gpu->SampleFlow(cond.data() + s * hidden, history.data(), noises.data() + s * lat, nullptr,
                    {}, latent.data());
    sample_ms += Seconds(t) * 1e3;
    got_lat.insert(got_lat.end(), latent.begin(), latent.end());
    got_stop.push_back(gpu->StopAt(cond.data() + s * hidden));
    want_stop.push_back(stop[static_cast<std::size_t>(s) * 2 + 1]);
    history.assign(latents.begin() + s * lat, latents.begin() + (s + 1) * lat);
    if (s + 1 < steps) {
      t = std::chrono::steady_clock::now();
      gpu->AggregateLatent(latents.data() + s * lat, next.data());
      agg_ms += Seconds(t) * 1e3;
      got_agg.insert(got_agg.end(), next.begin(), next.end());
      t = std::chrono::steady_clock::now();
      gpu->RunBackbone(agg.data() + s * hidden, 1, z.data());
      backbone_ms += Seconds(t) * 1e3;
      got_cond.insert(got_cond.end(), z.begin(), z.end());
    }
  }
  Compare("AR cond z (teacher forced)", got_cond, cond, 0.999);
  Compare("AR sampled latents", got_lat, latents, 0.999);
  Compare("AR aggregator out", got_agg, agg, 0.999);
  Compare("AR stop probability", got_stop, want_stop, 0.999);
  std::printf("     per step: flow %.2f ms, aggregator %.2f ms, backbone %.2f ms -> %.2f ms "
              "(80 ms audio)\n",
              sample_ms / steps, agg_ms / std::max<std::int64_t>(1, steps - 1),
              backbone_ms / std::max<std::int64_t>(1, steps - 1),
              (sample_ms + agg_ms + backbone_ms) / steps);

  // --- VAE decoder
  t = std::chrono::steady_clock::now();
  const std::vector<float> wav = gpu->DecodeAudio(latents.data(), steps);
  std::printf("     VAE decode %lld latents %.1f ms\n", static_cast<long long>(steps),
              Seconds(t) * 1e3);
  Compare("VAE waveform vs oracle", wav, oracle->ToF32("wav"), 0.98);
  Compare("VAE waveform vs CPU runtime", wav,
          cpu_model->DecodeLatents(latents.data(), steps), 0.98);
  const std::vector<float> probe = oracle->ToF32("vae_probe_latents");
  Compare("VAE probe 300 frames", gpu->DecodeAudio(probe.data(), static_cast<std::int64_t>(probe.size() / l)),
          oracle->ToF32("vae_probe_wav"), 0.98);
  // --- free running directly on the backends: where do trajectories split?
  {
    double ms = 0;
    gpu->SetBf16Gemm(false);
    gpu->SetGraphs(true);
    const std::vector<float> fp32_graph = FreeRun(*gpu, embeds, rows, hidden, lat, noises, &ms);
    std::printf("     GPU free run (FP32 GEMM, graphs %s): %.2f ms/step\n",
                gpu->graphs_enabled() ? "on" : "off(capture failed)", ms);
    Divergence("GPU FP32 vs oracle", fp32_graph, latents, l);
    Expect("GPU FP32 free-run step count",
           std::llabs(static_cast<long long>(fp32_graph.size() / l) - steps) <= 1,
           std::to_string(fp32_graph.size() / l) + " vs " + std::to_string(steps));
    const std::vector<float> again = FreeRun(*gpu, embeds, rows, hidden, lat, noises, &ms);
    std::printf("     GPU free run again (graphs replayed): %.2f ms/step\n", ms);
    gpu->SetGraphs(false);
    const std::vector<float> eager = FreeRun(*gpu, embeds, rows, hidden, lat, noises, &ms);
    std::printf("     GPU free run eager (no graphs): %.2f ms/step\n", ms);
    Compare("graph replay == eager (latents)", again, eager, 0.99999);
    Compare("graph first run == replay (latents)", fp32_graph, again, 0.99999);
    gpu->SetGraphs(true);
    gpu->SetBf16Gemm(true);
    const std::vector<float> bf16 = FreeRun(*gpu, embeds, rows, hidden, lat, noises, &ms);
    Divergence("GPU BF16-GEMM prefill vs oracle (info)", bf16, latents, l);
    gpu->SetBf16Gemm(false);
    const std::vector<float> cpu = FreeRun(*cpu_model, embeds, rows, hidden, lat, noises, &ms);
    Divergence("CPU vs oracle (info)", cpu, latents, l);
  }
  gpu.reset();

  // --- free-running engine on the GPU (oracle start noise)
  t = std::chrono::steady_clock::now();
  auto engine = ming::Engine::Create(model_env, &error);
  if (engine == nullptr) {
    std::printf("FAIL engine: %s\n", error.c_str());
    return 1;
  }
  std::printf("     engine (%s) ready in %.1f s\n", engine->backend_name(), Seconds(t));
  Expect("engine uses the GPU", std::string(engine->backend_name()) == "hip", engine->backend_name());
  ming::SynthesisRequest request;
  request.text = meta.member_str("text");
  request.emotion = meta.member_str("emotion");
  request.speaker_embedding = oracle->ToF32("spk_emb");
  request.raw_text = true;
  request.noise_source = [&](std::size_t, std::int64_t step, float* noise) {
    for (std::size_t i = 0; i < l; ++i) {
      const std::size_t at = static_cast<std::size_t>(step) * l + i;
      noise[i] = at < noises.size() ? noises[at] : 0.0F;
    }
  };
  ming::SynthesisResult result;
  for (int round = 0; round < 2; ++round) {  // round 0 warms up kernels
    t = std::chrono::steady_clock::now();
    const bool ok = engine->Generate(request, {}, &result, &error);
    const double seconds = Seconds(t);
    if (!ok) {
      Expect("free-running generate", false, error);
      break;
    }
    const double audio_s = static_cast<double>(result.samples.size()) / 44100.0;
    std::printf("     free-running round %d: %.2f s for %.2f s audio (RTF %.3f), %lld steps\n",
                round, seconds, audio_s, seconds / std::max(audio_s, 1e-9),
                static_cast<long long>(result.fragment_steps.at(0)));
  }
  const auto want_steps = static_cast<std::int64_t>(meta.find("steps")->as_double());
  Expect("free-running step count",
         !result.fragment_steps.empty() &&
             std::llabs(result.fragment_steps[0] - want_steps) <= 2,
         std::to_string(result.fragment_steps.empty() ? -1 : result.fragment_steps[0]) +
             " vs " + std::to_string(want_steps));
  Compare("free-running RMS envelope", Envelope(result.samples),
          Envelope(oracle->ToF32("wav_trimmed")), 0.98);
  if (const char* out = std::getenv("GUFO_MING_WAV_OUT"); out != nullptr) {
    std::ofstream f(out, std::ios::binary);
    const auto n = static_cast<std::uint32_t>(result.samples.size());
    const std::uint32_t data_bytes = n * 2, riff = 36 + data_bytes, rate = 44100,
                        byte_rate = 88200, fmt = 16;
    const std::uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
    f.write("RIFF", 4).write(reinterpret_cast<const char*>(&riff), 4).write("WAVEfmt ", 8);
    f.write(reinterpret_cast<const char*>(&fmt), 4).write(reinterpret_cast<const char*>(&pcm), 2);
    f.write(reinterpret_cast<const char*>(&ch), 2).write(reinterpret_cast<const char*>(&rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4).write(reinterpret_cast<const char*>(&align), 2);
    f.write(reinterpret_cast<const char*>(&bits), 2).write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_bytes), 4);
    for (const float v : result.samples) {
      const auto s = static_cast<std::int16_t>(std::lround(std::clamp(v, -1.0F, 1.0F) * 32767.0F));
      f.write(reinterpret_cast<const char*>(&s), 2);
    }
  }
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
