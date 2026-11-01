// Ming-3.0-tts CPU runtime vs. the Python CPU fp32 oracle
// (tools: D:\claude\gufo-arbeit\ming-port\ming_oracle.py, Ming-3.0-tts commit
// 9332fb94). Every stage must reach cosine >= 0.999 against the oracle.
//
//   GUFO_MING_MODEL  checkpoint directory (Ming-3.0-tts-0.5B)
//   GUFO_MING_REF    directory with oracle.safetensors + oracle.json
//   GUFO_MING_STAGES optional comma list: text,speaker,backbone,dit,ar,vae,free
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

#include "src/core/json.hpp"
#include "src/models/ming_tts/cpu/model.hpp"
#include "src/models/ming_tts/prompt.hpp"
#include "src/models/ming_tts/safetensors.hpp"
#include "src/models/ming_tts/speaker.hpp"
#include "src/models/ming_tts/synthesis.hpp"
#include "src/models/ming_tts/tokenizer.hpp"

namespace ming = gufo::models::ming_tts;
namespace json = gufo::json;

namespace {

int g_failures = 0;
constexpr double kMinCos = 0.999;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

void Compare(const std::string& name, const float* got, const float* want,
             std::size_t count, double min_cos = kMinCos) {
  double dot = 0.0, a2 = 0.0, b2 = 0.0, max_abs = 0.0, ref_max = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    dot += static_cast<double>(got[i]) * want[i];
    a2 += static_cast<double>(got[i]) * got[i];
    b2 += static_cast<double>(want[i]) * want[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
    ref_max = std::max(ref_max, std::fabs(static_cast<double>(want[i])));
  }
  const double cos = (a2 == 0.0 && b2 == 0.0) ? 1.0 : dot / std::sqrt(a2 * b2 + 1e-300);
  const bool ok = cos >= min_cos && std::isfinite(cos);
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-34s cos=%.7f max_abs=%.3g (ref max %.3g, n=%zu)\n",
              ok ? "OK" : "FAIL", name.c_str(), cos, max_abs, ref_max, count);
  std::fflush(stdout);
}

void Compare(const std::string& name, const std::vector<float>& got,
             const std::vector<float>& want, double min_cos = kMinCos) {
  if (got.size() != want.size()) {
    std::printf("FAIL %-34s size %zu vs %zu\n", name.c_str(), got.size(), want.size());
    ++g_failures;
    const std::size_t n = std::min(got.size(), want.size());
    if (n > 0) {
      Compare(name + " (common prefix)", got.data(), want.data(), n, min_cos);
    }
    return;
  }
  Compare(name, got.data(), want.data(), got.size(), min_cos);
}

void Expect(const std::string& name, bool ok, const std::string& detail = {}) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-34s %s\n", ok ? "OK" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::vector<std::uint32_t> Ids(const json::Value& array) {
  std::vector<std::uint32_t> ids;
  for (const auto& v : array.items()) {
    ids.push_back(static_cast<std::uint32_t>(v.as_double()));
  }
  return ids;
}

bool Want(const std::string& stages, const char* stage) {
  if (stages.empty()) {
    return true;
  }
  return ("," + stages + ",").find("," + std::string(stage) + ",") != std::string::npos;
}

}  // namespace

int main() {
  const char* model_env = std::getenv("GUFO_MING_MODEL");
  const char* ref_env = std::getenv("GUFO_MING_REF");
  if (model_env == nullptr || ref_env == nullptr) {
    std::printf("SKIP: set GUFO_MING_MODEL and GUFO_MING_REF\n");
    return 77;
  }
  const std::filesystem::path model_dir(model_env);
  const std::filesystem::path ref_dir(ref_env);
  const char* stages_env = std::getenv("GUFO_MING_STAGES");
  const std::string stages = stages_env != nullptr ? stages_env : "";
  std::string error;
  auto oracle = ming::SafetensorsFile::Open(ref_dir / "oracle.safetensors", &error);
  if (oracle == nullptr || !std::filesystem::exists(ref_dir / "oracle.json")) {
    std::printf("SKIP: oracle missing: %s\n", error.c_str());
    return 77;
  }
  const json::Value meta = json::parse(ReadText(ref_dir / "oracle.json"));
  const auto t0 = std::chrono::steady_clock::now();

  // ------------------------------------------------------------ text ----
  ming::Tokenizer tokenizer;
  if (!ming::Tokenizer::Load(model_dir, &tokenizer, &error)) {
    std::printf("FAIL tokenizer load: %s\n", error.c_str());
    return 1;
  }
  const std::string emotion = meta.member_str("emotion");
  ming::StructuredControls controls;
  controls.emotion = emotion;
  const std::string instruction = ming::BuildStructuredInstruction(controls);
  if (Want(stages, "text")) {
    for (const auto& c : meta.find("tokenizer_cases")->items()) {
      std::vector<std::uint32_t> ids;
      const bool ok = tokenizer.Encode(c.member_str("text"), &ids, &error);
      Expect("tokenize '" + c.member_str("text").substr(0, 24) + "'",
             ok && ids == Ids(*c.find("ids")),
             std::to_string(ids.size()) + " ids");
    }
    for (const auto& [label, value] : meta.find("emotion_instructions")->members()) {
      ming::StructuredControls ec;
      ec.emotion = label;
      Expect("instruction " + label, ming::BuildStructuredInstruction(ec) == value.str());
    }
    Expect("emotion name 'Wut'", ming::EmotionLabel("Wut").value_or("") == "愤怒");
    Expect("prompt German", ming::BuildTtsPrompt("de") == meta.member_str("prompt"));
    ming::InputLayout layout;
    const bool ok = ming::BuildInputIds(tokenizer, ming::BuildTtsPrompt("de"),
                                        meta.member_str("text"), 1, instruction,
                                        &layout, &error);
    Expect("input_ids", ok && layout.ids == Ids(*meta.find("input_ids")),
           std::to_string(layout.ids.size()) + " ids, speaker at " +
               (layout.speaker_positions.empty()
                    ? std::string("-")
                    : std::to_string(layout.speaker_positions[0])));
    for (const auto& seg : meta.find("segments")->items()) {
      const auto fragments = ming::SplitTextFragments(seg.member_str("text"));
      std::vector<std::string> want;
      for (const auto& f : seg.find("fragments")->items()) {
        want.push_back(f.str());
      }
      std::string detail = std::to_string(fragments.size()) + " fragments";
      if (fragments != want) {
        for (const auto& f : fragments) {
          detail += "\n       got:  " + f;
        }
        for (const auto& f : want) {
          detail += "\n       want: " + f;
        }
      }
      Expect("segments '" + seg.member_str("text").substr(0, 20) + "'",
             fragments == want, detail);
    }
  }

  // --------------------------------------------------------- speaker ----
  if (Want(stages, "speaker")) {
    const auto start = std::chrono::steady_clock::now();
    const std::vector<float> ref_wav = oracle->ToF32("ref_wav");
    const auto rate = static_cast<std::int64_t>(meta.find("ref_sample_rate")->as_double());
    const std::vector<float> wav16 = ming::ResampleLikeTorchaudio(ref_wav, rate, 16000);
    Compare("resample to 16 kHz", wav16, oracle->ToF32("ref_wav16k"));
    std::int64_t frames = 0;
    const std::vector<float> fbank = ming::KaldiFbank80(oracle->ToF32("ref_wav16k"), &frames);
    Compare("kaldi fbank 80", fbank, oracle->ToF32("fbank_raw"));
    auto campp = ming::CamPlusPlus::Load(model_dir / "campplus.onnx", &error);
    if (campp == nullptr) {
      Expect("campplus load", false, error);
    } else {
      const std::vector<float> feat = oracle->ToF32("fbank");
      const auto feat_frames = static_cast<std::int64_t>(feat.size() / 80);
      Compare("CAM++ embedding (oracle fbank)", campp->Embed(feat, feat_frames),
              oracle->ToF32("spk_emb"));
      Compare("CAM++ embedding (full path)", campp->EmbedWaveform(ref_wav, rate),
              oracle->ToF32("spk_emb"));
    }
    std::printf("     speaker stage %.1f s\n", Seconds(start));
  }

  const bool need_model = Want(stages, "backbone") || Want(stages, "dit") ||
                          Want(stages, "ar") || Want(stages, "vae");
  if (need_model) {
    const auto start = std::chrono::steady_clock::now();
    auto model = ming::cpu::Model::Load(model_dir.string(), &error);
    if (model == nullptr) {
      std::printf("FAIL model load: %s\n", error.c_str());
      return 1;
    }
    std::printf("     model mapped in %.1f s\n", Seconds(start));
    const std::int64_t hidden = model->config().llm.hidden_size;
    const std::int64_t latent_dim = model->config().latent_dim;
    const std::vector<std::int64_t> input_ids = oracle->ToI64("input_ids");
    const auto rows = static_cast<std::int64_t>(input_ids.size());
    const std::vector<float> cond = oracle->ToF32("cond");
    const std::vector<float> latents = oracle->ToF32("latents");
    const std::vector<float> noises = oracle->ToF32("noise");
    const std::vector<float> agg = oracle->ToF32("agg");
    const std::vector<float> stop = oracle->ToF32("stop_prob");
    const auto steps = static_cast<std::int64_t>(latents.size()) / latent_dim;

    // ------------------------------------------------------ backbone ----
    if (Want(stages, "backbone") || Want(stages, "ar")) {
      const auto bstart = std::chrono::steady_clock::now();
      std::vector<float> spk(static_cast<std::size_t>(hidden));
      model->ProjectSpeaker(oracle->ToF32("spk_emb"), spk.data());
      Compare("spk_head projection", spk, oracle->ToF32("spk_proj"));
      std::vector<float> embeds(static_cast<std::size_t>(rows * hidden));
      model->Embed(input_ids, embeds.data());
      for (const auto& p : meta.find("speaker_positions")->items()) {
        const auto pos = static_cast<std::int64_t>(p.as_double());
        std::copy(spk.begin(), spk.end(), embeds.begin() + pos * hidden);
      }
      Compare("inputs_embeds", embeds, oracle->ToF32("inputs_embeds"));
      model->ResetCache();
      std::vector<float> normed(static_cast<std::size_t>(rows * hidden));
      std::vector<std::vector<float>> layers;
      model->BackboneForward(embeds.data(), rows, normed.data(), &layers);
      Compare("prefill hidden layer 0", layers.at(0), oracle->ToF32("prefill_hidden_l0"));
      Compare("prefill hidden layer 11", layers.at(11), oracle->ToF32("prefill_hidden_l12"));
      Compare("prefill final normed", normed, oracle->ToF32("prefill_hidden_last"));
      std::printf("     prefill %lld rows %.2f s\n", static_cast<long long>(rows),
                  Seconds(bstart));

      if (Want(stages, "ar")) {
        // Teacher forcing: oracle aggregator outputs drive the backbone.
        std::vector<float> z(static_cast<std::size_t>(hidden));
        std::vector<float> zs(normed.end() - hidden, normed.end());
        std::vector<float> agg_out(static_cast<std::size_t>(hidden));
        std::vector<float> latent(static_cast<std::size_t>(latent_dim));
        std::vector<float> all_cond(zs), all_lat, all_agg, all_stop;
        std::vector<float> history(static_cast<std::size_t>(latent_dim), 0.0F);
        const ming::cpu::FlowOptions flow;
        const auto astart = std::chrono::steady_clock::now();
        for (std::int64_t s = 0; s < steps; ++s) {
          const float* cs = cond.data() + s * hidden;
          model->SampleLatent(cs, history.data(), noises.data() + s * latent_dim,
                              nullptr, flow, latent.data());
          all_lat.insert(all_lat.end(), latent.begin(), latent.end());
          all_stop.push_back(model->StopProbability(cs));
          const float* oracle_latent = latents.data() + s * latent_dim;
          history.assign(oracle_latent, oracle_latent + latent_dim);
          if (s + 1 < steps) {
            model->Aggregate(oracle_latent, agg_out.data());
            all_agg.insert(all_agg.end(), agg_out.begin(), agg_out.end());
            model->BackboneForward(agg.data() + s * hidden, 1, z.data());
            all_cond.insert(all_cond.end(), z.begin(), z.end());
          }
        }
        std::printf("     teacher-forced AR %lld steps %.2f s\n",
                    static_cast<long long>(steps), Seconds(astart));
        Compare("AR cond z (all steps)", all_cond, cond);
        Compare("AR sampled latents (all steps)", all_lat, latents);
        Compare("AR aggregator out (all steps)", all_agg, agg);
        std::vector<float> stop1;
        for (std::int64_t s = 0; s < steps; ++s) {
          stop1.push_back(stop[static_cast<std::size_t>(s * 2 + 1)]);
        }
        Compare("AR stop probability", all_stop, stop1);
        double worst = 1.0;
        for (std::int64_t s = 0; s < steps; ++s) {
          double d = 0, a = 0, b = 0;
          for (std::int64_t i = 0; i < latent_dim; ++i) {
            const double x = all_lat[static_cast<std::size_t>(s * latent_dim + i)];
            const double y = latents[static_cast<std::size_t>(s * latent_dim + i)];
            d += x * y;
            a += x * x;
            b += y * y;
          }
          worst = std::min(worst, d / std::sqrt(a * b));
        }
        Expect("AR worst per-step latent cos", worst >= kMinCos, std::to_string(worst));
      }
    }

    // ----------------------------------------------------------- dit ----
    if (Want(stages, "dit")) {
      Compare("flow timesteps", ming::cpu::Model::FlowTimesteps(10),
              oracle->ToF32("dit_timesteps"));
      const std::vector<float> history(static_cast<std::size_t>(latent_dim), 0.0F);
      std::vector<float> out(static_cast<std::size_t>(2 * latent_dim));
      for (int k = 0; k < 10; ++k) {
        const std::string suffix = std::to_string(k);
        const std::vector<float> x = oracle->ToF32("dit0_x_" + suffix);
        const float t = oracle->ToF32("dit0_t_" + suffix).at(0);
        model->FlowVelocity(x.data(), t, cond.data(), history.data(), out.data());
        Compare("DiT velocity step0 eval " + suffix, out, oracle->ToF32("dit0_out_" + suffix));
      }
      std::vector<float> latent(static_cast<std::size_t>(latent_dim));
      model->SampleLatent(cond.data(), history.data(), noises.data(), nullptr, {},
                          latent.data());
      Compare("DiT sampled latent step 0", latent.data(), latents.data(),
              static_cast<std::size_t>(latent_dim));
    }

    // ----------------------------------------------------------- vae ----
    if (Want(stages, "vae")) {
      const auto vstart = std::chrono::steady_clock::now();
      ming::cpu::Model::VaeTrace trace;
      const std::vector<float> wav = model->DecodeLatents(latents.data(), steps, &trace);
      std::printf("     VAE decode %lld latents %.2f s\n", static_cast<long long>(steps),
                  Seconds(vstart));
      Compare("VAE fc1", trace.fc1, oracle->ToF32("vae_fc1"));
      Compare("VAE upsampled", trace.upsampled, oracle->ToF32("vae_upsampled"));
      Compare("VAE hidden layer 0", trace.hidden_l0, oracle->ToF32("vae_hidden_l0"));
      Compare("VAE hidden last (normed)", trace.hidden_last, oracle->ToF32("vae_hidden_last"));
      Compare("VAE ISTFT head out", trace.head_out, oracle->ToF32("vae_head_out"));
      Compare("VAE waveform", wav, oracle->ToF32("wav"));
      const std::vector<float> trimmed = ming::TrimTrailingSilence(wav, 44100);
      Expect("trailing-silence trim length",
             trimmed.size() == static_cast<std::size_t>(
                                   meta.find("wav_trimmed_samples")->as_double()),
             std::to_string(trimmed.size()));
      const std::vector<float> probe = oracle->ToF32("vae_probe_latents");
      const auto probe_frames = static_cast<std::int64_t>(probe.size()) / latent_dim;
      Compare("VAE probe 300 frames (window)",
              model->DecodeLatents(probe.data(), probe_frames),
              oracle->ToF32("vae_probe_wav"));
    }
  }

  // ---------------------------------------------------- free running ----
  if (Want(stages, "free")) {
    const auto fstart = std::chrono::steady_clock::now();
    auto engine = ming::Engine::Create(model_dir.string(), &error);
    if (engine == nullptr) {
      std::printf("FAIL engine: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> noises = oracle->ToF32("noise");
    ming::SynthesisRequest request;
    request.text = meta.member_str("text");
    request.language = "de";
    request.emotion = emotion;
    request.speaker_embedding = oracle->ToF32("spk_emb");
    request.raw_text = true;
    request.noise_source = [&](std::size_t, std::int64_t step, float* noise) {
      const std::size_t offset = static_cast<std::size_t>(step) * 64;
      for (std::size_t i = 0; i < 64; ++i) {
        noise[i] = offset + i < noises.size() ? noises[offset + i] : 0.0F;
      }
    };
    ming::SynthesisResult result;
    const bool ok = engine->Generate(request, {}, &result, &error);
    const double seconds = Seconds(fstart);
    Expect("free-running generate", ok, error);
    if (ok) {
      const auto want_steps = static_cast<std::int64_t>(meta.find("steps")->as_double());
      Expect("free-running step count", result.fragment_steps.at(0) == want_steps,
             std::to_string(result.fragment_steps.at(0)) + " vs " + std::to_string(want_steps));
      // Free running accumulates FP32 summation-order differences through the
      // AR loop; per-step latents stay close, the waveform phase does not.
      // Gate on latents and on the frame-RMS envelope, report raw wav cos.
      const std::vector<float> want_latents = oracle->ToF32("latents");
      Compare("free-running latents", result.latents, want_latents, 0.999);
      if (result.latents.size() == want_latents.size()) {
        std::string curve;
        for (std::size_t s = 0; s * 64 < want_latents.size(); s += 4) {
          double d = 0, a = 0, b = 0;
          for (std::size_t i = 0; i < 64; ++i) {
            d += static_cast<double>(result.latents[s * 64 + i]) * want_latents[s * 64 + i];
            a += static_cast<double>(result.latents[s * 64 + i]) * result.latents[s * 64 + i];
            b += static_cast<double>(want_latents[s * 64 + i]) * want_latents[s * 64 + i];
          }
          char buffer[32];
          std::snprintf(buffer, sizeof(buffer), " %zu:%.5f", s, d / std::sqrt(a * b));
          curve += buffer;
        }
        std::printf("     free-running latent cos per step%s\n", curve.c_str());
      }
      const std::vector<float> want_wav = oracle->ToF32("wav_trimmed");
      auto envelope = [](const std::vector<float>& wav) {
        std::vector<float> env;
        for (std::size_t start = 0; start + 882 <= wav.size(); start += 882) {
          double sum = 0.0;
          for (std::size_t i = 0; i < 882; ++i) {
            sum += static_cast<double>(wav[start + i]) * wav[start + i];
          }
          env.push_back(static_cast<float>(std::sqrt(sum / 882.0)));
        }
        return env;
      };
      Compare("free-running wav RMS envelope", envelope(result.samples), envelope(want_wav), 0.99);
      const std::size_t common = std::min(result.samples.size(), want_wav.size());
      double d = 0, a = 0, b = 0;
      for (std::size_t i = 0; i < common; ++i) {
        d += static_cast<double>(result.samples[i]) * want_wav[i];
        a += static_cast<double>(result.samples[i]) * result.samples[i];
        b += static_cast<double>(want_wav[i]) * want_wav[i];
      }
      std::printf("     free-running raw wav cos %.5f (info, phase-sensitive)\n",
                  d / std::sqrt(a * b + 1e-300));
      const double audio_s = static_cast<double>(result.samples.size()) / 44100.0;
      std::printf("     free-running %.2f s for %.2f s audio (RTF %.2f)\n", seconds,
                  audio_s, seconds / std::max(audio_s, 1e-9));
      const char* wav_out = std::getenv("GUFO_MING_WAV_OUT");
      if (wav_out != nullptr) {
        std::ofstream f(wav_out, std::ios::binary);
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
    }
  }

  std::printf("%s: %d failure(s), %.1f s\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, Seconds(t0));
  return g_failures == 0 ? 0 : 1;
}
