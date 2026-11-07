#include "src/models/indextts/synthesis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numbers>
#include <random>
#include <sstream>
#include <unordered_map>

#include "src/models/indextts/audio.hpp"
#include "src/models/indextts/bigvgan.hpp"
#include "src/models/indextts/campplus.hpp"
#include "src/models/indextts/hip/runtime.hpp"
#include "src/models/indextts/s2mel.hpp"
#include "src/models/indextts/text.hpp"
#include "src/models/indextts/w2vbert.hpp"

namespace gufo::models::indextts {
namespace {

using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point a) {
  return std::chrono::duration<double>(Clock::now() - a).count();
}

std::filesystem::path GptPath(const std::filesystem::path& dir) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(dir / "german" / "gpt.pth", ec)) {
    return dir / "german" / "gpt.pth";
  }
  return dir / "gpt.pth";
}

std::string Lower(std::string s) {
  // ASCII + the German capitals (UTF-8) used in emotion names
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c == 0xC3 && i + 1 < s.size()) {
      unsigned char d = static_cast<unsigned char>(s[i + 1]);
      if (d == 0x84 || d == 0x96 || d == 0x9C) {  // Ä Ö Ü -> ä ö ü
        d = static_cast<unsigned char>(d + 0x20);
      }
      out.push_back(static_cast<char>(c));
      out.push_back(static_cast<char>(d));
      ++i;
      continue;
    }
    out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

struct RefCond {
  std::vector<float> style;          // [192]
  std::vector<float> spk_cond;       // [rows, 1024]
  std::int64_t spk_rows{0};
  std::vector<float> emo_cond;       // [rows, 1024]
  std::int64_t emo_rows{0};
  std::vector<float> ref_mel;        // [frames, 80]
  std::int64_t ref_frames{0};
  std::vector<float> prompt_condition;  // [frames, 512]
  std::vector<float> merged_emovec;  // base + 1 * (emo - base)
};

std::uint64_t HashSamples(std::span<const float> s, std::int64_t rate) {
  std::uint64_t h = 1469598103934665603ULL ^ static_cast<std::uint64_t>(rate);
  for (const float v : s) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    h = (h ^ bits) * 1099511628211ULL;
  }
  return h ^ (s.size() * 0x9E3779B97F4A7C15ULL);
}

}  // namespace

bool LooksLikeIndexTts25(const std::filesystem::path& dir) {
  std::error_code ec;
  const auto exists = [&](const std::filesystem::path& p) {
    return std::filesystem::is_regular_file(p, ec);
  };
  if (!exists(dir / "config.yaml") || !exists(GptPath(dir)) || !exists(dir / "s2mel.pth") ||
      !exists(dir / "codec.pth") || !exists(dir / "multilingual_zh_ja_yue_char_del.tiktoken") ||
      !exists(dir / "w2v-bert-2.0" / "model.safetensors") ||
      !exists(dir / "campplus" / "campplus_cn_common.bin") ||
      !exists(dir / "bigvgan_v2_22khz_80band_256x" / "bigvgan_generator.pt")) {
    return false;
  }
  std::ifstream in(dir / "config.yaml");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("version:", 0) == 0 && line.find("2.5") != std::string::npos) {
      return true;
    }
  }
  return false;
}

int EmotionIndex(const std::string& raw) {
  static const std::vector<std::pair<std::string, int>> names = {
      {"happy", 0},        {"happiness", 0},     {"joy", 0},           {"freude", 0},
      {"fröhlich", 0},     {"froehlich", 0},     {"glücklich", 0},     {"gluecklich", 0},
      {"angry", 1},        {"anger", 1},         {"wut", 1},           {"wütend", 1},
      {"wuetend", 1},      {"zorn", 1},          {"ärger", 1},         {"aerger", 1},
      {"sad", 2},          {"sadness", 2},       {"trauer", 2},        {"traurig", 2},
      {"afraid", 3},       {"fear", 3},          {"angst", 3},         {"ängstlich", 3},
      {"aengstlich", 3},   {"furcht", 3},        {"disgusted", 4},     {"disgust", 4},
      {"ekel", 4},         {"angeekelt", 4},     {"abscheu", 4},       {"melancholic", 5},
      {"melancholy", 5},   {"melancholie", 5},   {"melancholisch", 5}, {"niedergeschlagen", 5},
      {"surprised", 6},    {"surprise", 6},      {"überraschung", 6},  {"ueberraschung", 6},
      {"überrascht", 6},   {"ueberrascht", 6},   {"calm", 7},          {"neutral", 7},
      {"ruhe", 7},         {"ruhig", 7},         {"gelassen", 7}};
  const std::string n = Lower(raw);
  for (const auto& [k, v] : names) {
    if (n == k) {
      return v;
    }
  }
  return -1;
}

bool ParseEmotionSpec(const std::string& spec, std::optional<std::array<float, 8>>* out,
                      std::string* error) {
  out->reset();
  auto trim = [](std::string s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    const auto e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
  };
  const std::string s = trim(spec);
  if (s.empty()) {
    return true;
  }
  std::vector<std::string> parts;
  {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
      parts.push_back(trim(item));
    }
  }
  auto number = [](const std::string& t, float* v) {
    if (t.empty()) {
      return false;
    }
    char* end = nullptr;
    const double d = std::strtod(t.c_str(), &end);
    if (end == nullptr || *end != '\0' || !std::isfinite(d) || d < 0.0 || d > 1.2) {
      return false;
    }
    *v = static_cast<float>(d);
    return true;
  };
  std::array<float, 8> v{};
  float probe = 0.0F;
  if (parts.size() == 8 && number(parts[0], &probe)) {
    for (std::size_t i = 0; i < 8; ++i) {
      if (!number(parts[i], &v[i])) {
        if (error != nullptr) {
          *error = "emotion vector values must be numbers between 0 and 1.2";
        }
        return false;
      }
    }
    *out = v;
    return true;
  }
  for (const std::string& part : parts) {
    const auto colon = part.find(':');
    const std::string name = trim(part.substr(0, colon));
    float weight = 1.0F;
    if (colon != std::string::npos && !number(trim(part.substr(colon + 1)), &weight)) {
      if (error != nullptr) {
        *error = "emotion weight for '" + name + "' must be a number between 0 and 1.2";
      }
      return false;
    }
    const int index = EmotionIndex(name);
    if (index < 0) {
      if (error != nullptr) {
        *error = "unknown emotion '" + name +
                 "' (happy/Freude, angry/Wut, sad/Trauer, afraid/Angst, disgusted/Ekel, "
                 "melancholic/Melancholie, surprised/Überraschung, calm/Ruhe)";
      }
      return false;
    }
    v[static_cast<std::size_t>(index)] += weight;
  }
  *out = v;
  return true;
}

struct Engine::Impl {
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<CamPlus> campplus;
  std::unique_ptr<W2vBert> w2v;
  std::unique_ptr<UnifiedVoice> gpt;
  std::unique_ptr<EmotionMatrix> emo_matrix;
  std::unique_ptr<SemanticCodec> codec;
  std::unique_ptr<S2Mel> s2mel;
  std::unique_ptr<BigVgan> vocoder;
  std::unique_ptr<hip::Runtime> gpu;  // GPT, CFM and vocoder on the GPU when available
  std::mutex mutex;
  std::uint64_t cached_key{0};
  bool has_cache{false};
  RefCond cache;
  // 25 Euler steps, CFG 0.7 as in infer_v2_5.py; GUFO_INDEXTTS_CFM_STEPS (4..50) trades quality for
  // speed (opt-in, not the reference).
  CfmOptions cfm;

  void BuildReference(std::span<const float> mono, std::int64_t rate, RefCond* out) {
    // _load_and_cut_audio(spk, 15)
    const std::size_t max_samples = static_cast<std::size_t>(15 * rate);
    std::span<const float> audio = mono.subspan(0, std::min(mono.size(), max_samples));
    const std::vector<float> a22 = ResampleTorchaudio(audio, rate, 22050);
    const std::vector<float> a16 = ResampleTorchaudio(audio, rate, 16000);
    std::int64_t frames = 0;
    const std::vector<float> mel_cm = MelSpectrogram22k(a22, &frames);  // [80, frames]
    out->ref_frames = frames;
    out->ref_mel.resize(static_cast<std::size_t>(frames * 80));
    for (std::int64_t f = 0; f < frames; ++f) {
      for (int m = 0; m < 80; ++m) {
        out->ref_mel[static_cast<std::size_t>(f * 80 + m)] = mel_cm[static_cast<std::size_t>(m * frames + f)];
      }
    }
    std::int64_t fb_frames = 0;
    const std::vector<float> fbank = CampplusFbank(a16, &fb_frames);
    out->style = campplus->Embed(fbank, fb_frames);
    const W2vFeatures spk_feat = W2vBertFeatures(a16);
    out->spk_cond = w2v->Embed(spk_feat);
    out->spk_rows = spk_feat.rows;
    // emotion reference: librosa.load(sr=16000) (soxr_hq) of the same clip, cut at 15 s
    std::vector<float> e16 = ResampleSoxrHq(mono, rate, 16000);
    e16.resize(std::min<std::size_t>(e16.size(), 15 * 16000));
    if (e16 == a16) {
      out->emo_cond = out->spk_cond;
      out->emo_rows = out->spk_rows;
    } else {
      const W2vFeatures emo_feat = W2vBertFeatures(e16);
      out->emo_cond = w2v->Embed(emo_feat);
      out->emo_rows = emo_feat.rows;
    }
    out->prompt_condition = s2mel->Regulate(out->spk_cond.data(), out->spk_rows, frames);
    const std::vector<float> emo_vec = gpt->EmoVec(out->emo_cond.data(), out->emo_rows);
    const std::vector<float> base_vec = gpt->EmoVec(out->spk_cond.data(), out->spk_rows);
    out->merged_emovec.resize(emo_vec.size());
    for (std::size_t i = 0; i < emo_vec.size(); ++i) {
      out->merged_emovec[i] = base_vec[i] + 1.0F * (emo_vec[i] - base_vec[i]);
    }
  }
};

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::Create(const std::filesystem::path& dir, std::string* error) {
  if (!LooksLikeIndexTts25(dir)) {
    if (error != nullptr) {
      *error = dir.string() + " is not an IndexTTS-2.5 bundle";
    }
    return nullptr;
  }
  auto e = std::unique_ptr<Engine>(new Engine());
  e->impl_ = std::make_unique<Impl>();
  Impl& m = *e->impl_;
  m.tokenizer = Tokenizer::Load(dir / "multilingual_zh_ja_yue_char_del.tiktoken", error);
  if (m.tokenizer == nullptr) return nullptr;
  m.campplus = CamPlus::Load(dir / "campplus" / "campplus_cn_common.bin", error);
  if (m.campplus == nullptr) return nullptr;
  m.w2v = W2vBert::Load(dir / "w2v-bert-2.0", dir / "wav2vec2bert_stats.pt", error);
  if (m.w2v == nullptr) return nullptr;
  m.gpt = UnifiedVoice::Load(GptPath(dir), error);
  if (m.gpt == nullptr) return nullptr;
  m.emo_matrix = EmotionMatrix::Load(dir / "feat1.pt", dir / "feat2.pt", error);
  if (m.emo_matrix == nullptr) return nullptr;
  m.codec = SemanticCodec::Load(dir / "codec.pth", error);
  if (m.codec == nullptr) return nullptr;
  m.s2mel = S2Mel::Load(dir / "s2mel.pth", error);
  if (m.s2mel == nullptr) return nullptr;
  m.vocoder = BigVgan::Load(dir / "bigvgan_v2_22khz_80band_256x", error);
  if (m.vocoder == nullptr) return nullptr;
  if (const char* steps = std::getenv("GUFO_INDEXTTS_CFM_STEPS"); steps != nullptr) {
    m.cfm.steps = std::clamp(std::atoi(steps), 4, 50);
  }
  // GUFO_INDEXTTS_BACKEND=cpu forces the CPU path, =hip makes a missing GPU an error.
  const char* backend = std::getenv("GUFO_INDEXTTS_BACKEND");
  const std::string want = backend != nullptr ? backend : "";
  if (want != "cpu") {
    std::string gpu_error;
    m.gpu = hip::Runtime::Create(*m.gpt, *m.s2mel, *m.vocoder, &gpu_error);
    if (m.gpu == nullptr && want == "hip") {
      if (error != nullptr) {
        *error = "IndexTTS-2.5 HIP runtime unavailable: " + gpu_error;
      }
      return nullptr;
    }
  }
  return e;
}

const char* Engine::backend_name() const noexcept { return impl_ != nullptr && impl_->gpu != nullptr ? "hip" : "cpu"; }

bool Engine::Generate(const SynthesisRequest& req, const CancelFn& cancelled,
                      const ChunkFn& on_chunk, SynthesisResult* result, std::string* error) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) {
      *error = msg;
    }
    return false;
  };
  Impl& m = *impl_;
  std::lock_guard<std::mutex> lock(m.mutex);
  const auto t_start = Clock::now();
  *result = SynthesisResult{};
  try {
    if (req.reference.empty() || req.reference_rate <= 0) {
      return fail("IndexTTS-2.5 needs a reference voice");
    }
    const std::string lang = Lower(req.language);
    if (lang != "de" && lang != "german" && lang != "deutsch") {
      return fail("IndexTTS-2.5-German supports language \"de\" only");
    }
    const PreparedText text = PrepareForSynthesis(*m.tokenizer, req.text, "de",
                                                  req.max_text_tokens_per_segment,
                                                  static_cast<std::int32_t>(m.gpt->config().max_text_positions));
    if (text.final_text.empty()) {
      return fail("text is empty after the German text frontend");
    }
    const std::uint64_t key = HashSamples(req.reference, req.reference_rate);
    auto t0 = Clock::now();
    if (!m.has_cache || m.cached_key != key) {
      m.has_cache = false;
      m.BuildReference(req.reference, req.reference_rate, &m.cache);
      m.cached_key = key;
      m.has_cache = true;
    }
    result->reference_seconds = Seconds(t0);
    const RefCond& ref = m.cache;
    // emotion vector (normalize_emo_vec, emo_alpha scaling) and emovec
    std::vector<float> emovec = ref.merged_emovec;
    if (req.emotion.has_value()) {
      std::vector<float> w = NormalizeEmotionVector(*req.emotion);
      const float scale = std::clamp(req.emo_alpha, 0.0F, 1.0F);
      if (scale != 1.0F) {
        for (float& x : w) {
          x = static_cast<float>(static_cast<std::int64_t>(static_cast<double>(x) * scale * 10000.0)) / 10000.0F;
        }
      }
      const std::vector<float> mat = m.emo_matrix->Mix(w, ref.style);
      float wsum = 0.0F;
      for (const float x : w) {
        wsum += x;
      }
      for (std::size_t i = 0; i < emovec.size(); ++i) {
        emovec[i] = mat[i] + (1.0F - wsum) * ref.merged_emovec[i];
      }
    }
    const std::vector<float> spk_latent = m.gpt->SpeakerLatent(ref.style);
    const std::int32_t lang_token = LanguageToken("de");
    std::mt19937_64 rng(req.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::size_t forced_u = 0;
    const std::vector<std::int16_t> silence(
        static_cast<std::size_t>(kOutputRate * req.interval_silence_ms / 1000), 0);
    for (std::size_t seg = 0; seg < text.ids.size(); ++seg) {
      if (cancelled && cancelled()) {
        return fail("cancelled");
      }
      t0 = Clock::now();
      std::int64_t rows = 0;
      const std::vector<float> prefix =
          m.gpt->BuildPrefix(spk_latent, emovec, text.ids[seg], lang_token, &rows);
      UnifiedVoice::State st;
      std::vector<float> logits;
      if (m.gpu != nullptr) {
        m.gpu->GptPrefill(prefix, rows, req.sampling.max_tokens, &logits);
      } else {
        m.gpt->Prefill(prefix, rows, &st, &logits, req.sampling.max_tokens);
      }
      std::vector<std::int32_t> history = {1, m.gpt->config().start_mel};
      std::vector<std::int32_t> codes;
      for (std::int32_t step = 0; step < req.sampling.max_tokens; ++step) {
        if (cancelled && cancelled()) {
          return fail("cancelled");
        }
        ApplyRepetitionPenalty(logits, history, req.sampling.repetition_penalty);
        double u;
        if (req.forced_uniforms != nullptr && forced_u < req.forced_uniforms->size()) {
          u = (*req.forced_uniforms)[forced_u++];
        } else {
          u = uniform(rng);
        }
        const std::int32_t tok = SampleToken(logits, req.sampling, u);
        ++result->gpt_steps;
        history.push_back(tok);
        if (tok == m.gpt->config().stop_mel) {
          break;
        }
        codes.push_back(tok);
        if (step + 1 < req.sampling.max_tokens) {
          if (m.gpu != nullptr) {
            m.gpu->GptStep(tok, &logits);
          } else {
            m.gpt->Step(tok, &st, &logits);
          }
        }
      }
      result->gpt_seconds += Seconds(t0);
      if (codes.empty()) {
        continue;
      }
      result->codes = codes;
      t0 = Clock::now();
      const std::vector<float> s_infer = m.codec->Decode(codes);
      const auto s_rows = static_cast<std::int64_t>(codes.size()) * 2;
      const auto target = static_cast<std::int64_t>(static_cast<double>(s_rows) * 1.72 *
                                                    static_cast<double>(req.duration_factor));
      const std::vector<float> cond = m.s2mel->Regulate(s_infer.data(), s_rows, target);
      const std::int64_t total = ref.ref_frames + target;
      std::vector<float> mu(static_cast<std::size_t>(total * 512));
      std::copy(ref.prompt_condition.begin(), ref.prompt_condition.end(), mu.begin());
      std::copy(cond.begin(), cond.end(), mu.begin() + ref.ref_frames * 512);
      std::vector<float> noise(static_cast<std::size_t>(total * 80));
      if (req.forced_noise != nullptr && req.forced_noise->size() == noise.size()) {
        for (std::int64_t c = 0; c < 80; ++c) {  // [80, t] -> [t, 80]
          for (std::int64_t r = 0; r < total; ++r) {
            noise[static_cast<std::size_t>(r * 80 + c)] = (*req.forced_noise)[static_cast<std::size_t>(c * total + r)];
          }
        }
      } else {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (float& v : noise) {
          v = normal(rng);
        }
      }
      const std::vector<float> mel =
          m.gpu != nullptr
              ? m.gpu->CfmSample(noise, mu.data(), total, ref.ref_mel.data(), ref.ref_frames, ref.style.data(),
                                 m.cfm)
              : m.s2mel->Sample(std::move(noise), mu.data(), total, ref.ref_mel.data(), ref.ref_frames,
                                ref.style.data(), m.cfm);
      result->s2mel_seconds += Seconds(t0);
      t0 = Clock::now();
      const std::vector<float> wav = m.gpu != nullptr
                                         ? m.gpu->Vocode(mel.data() + ref.ref_frames * 80, target)
                                         : m.vocoder->Synthesize(mel.data() + ref.ref_frames * 80, target);
      result->vocoder_seconds += Seconds(t0);
      std::vector<std::int16_t> pcm(wav.size());
      for (std::size_t i = 0; i < wav.size(); ++i) {
        const float v = std::clamp(32767.0F * wav[i], -32767.0F, 32767.0F);
        pcm[i] = static_cast<std::int16_t>(v);  // .type(torch.int16) truncates
      }
      if (result->segments > 0) {
        result->samples.insert(result->samples.end(), silence.begin(), silence.end());
        if (on_chunk) {
          on_chunk(silence);
        }
      }
      result->samples.insert(result->samples.end(), pcm.begin(), pcm.end());
      if (result->segments == 0) {
        result->first_audio_seconds = Seconds(t_start);
      }
      ++result->segments;
      if (on_chunk) {
        on_chunk(pcm);
      }
    }
  } catch (const std::exception& e) {
    return fail(std::string("IndexTTS-2.5: ") + e.what());
  }
  if (result->samples.empty()) {
    return fail("IndexTTS-2.5 produced no audio");
  }
  return true;
}

}  // namespace gufo::models::indextts
