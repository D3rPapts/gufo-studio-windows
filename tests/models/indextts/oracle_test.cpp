// IndexTTS-2.5 (German) CPU runtime vs. the Python CPU fp32 oracle
// (src/models/indextts/reference/indextts_oracle.py). Every float stage must
// reach cosine >= 0.999; token ids, AR tokens and code counts must be exact.
//
//   GUFO_INDEXTTS_MODEL   bundle directory (IndexTTS-2.5 + german/gpt.pth + aux models)
//   GUFO_INDEXTTS_REF     directory with oracle.safetensors + oracle.json
//   GUFO_INDEXTTS_VOICE   reference WAV used by the oracle
//   GUFO_INDEXTTS_STAGES  optional comma list: text,audio,campplus,w2v,emo,gpt,free,codec,s2mel,dit,cfm,vocoder,engine
//   GUFO_INDEXTTS_OUT     optional directory for WAVs
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
#include "src/models/indextts/audio.hpp"
#include "src/models/indextts/bigvgan.hpp"
#include "src/models/indextts/campplus.hpp"
#include "src/models/indextts/gpt.hpp"
#include "src/models/indextts/s2mel.hpp"
#include "src/models/indextts/synthesis.hpp"
#include "src/models/indextts/text.hpp"
#include "src/models/indextts/w2vbert.hpp"
#include "src/models/ming_tts/safetensors.hpp"

namespace itts = gufo::models::indextts;
namespace json = gufo::json;
using gufo::models::ming_tts::SafetensorsFile;

namespace {

int g_failures = 0;
constexpr double kMinCos = 0.999;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

void Compare(const std::string& name, const float* got, const float* want, std::size_t n,
             double min_cos = kMinCos, bool info = false) {
  double dot = 0.0, a2 = 0.0, b2 = 0.0, max_abs = 0.0, ref_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(got[i]) * want[i];
    a2 += static_cast<double>(got[i]) * got[i];
    b2 += static_cast<double>(want[i]) * want[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
    ref_max = std::max(ref_max, std::fabs(static_cast<double>(want[i])));
  }
  const double cos = (a2 == 0.0 && b2 == 0.0) ? 1.0 : dot / std::sqrt(a2 * b2 + 1e-300);
  const bool ok = info || (cos >= min_cos && std::isfinite(cos));
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-34s cos=%.7f max_abs=%.3g (ref max %.3g, n=%zu)\n", info ? "INFO" : (ok ? "OK" : "FAIL"),
              name.c_str(), cos, max_abs, ref_max, n);
  std::fflush(stdout);
}

void Compare(const std::string& name, const std::vector<float>& got, const std::vector<float>& want,
             double min_cos = kMinCos, bool info = false) {
  if (got.size() != want.size()) {
    std::printf("%s %-34s size %zu vs %zu\n", info ? "INFO" : "FAIL", name.c_str(), got.size(), want.size());
    if (!info) {
      ++g_failures;
    }
    const std::size_t n = std::min(got.size(), want.size());
    if (n > 0) {
      Compare(name + " (prefix)", got.data(), want.data(), n, min_cos, true);
    }
    return;
  }
  Compare(name, got.data(), want.data(), got.size(), min_cos, info);
}

void Expect(const std::string& name, bool ok, const std::string& detail) {
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-4s %-34s %s\n", ok ? "OK" : "FAIL", name.c_str(), detail.c_str());
  std::fflush(stdout);
}

bool Want(const std::string& stages, const char* stage) {
  if (stages.empty()) {
    return true;
  }
  std::stringstream ss(stages);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == stage) {
      return true;
    }
  }
  return false;
}

std::string ReadText(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// [rows, cols] -> [cols, rows]
std::vector<float> Tr(const std::vector<float>& x, std::int64_t rows, std::int64_t cols) {
  std::vector<float> out(x.size());
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(c * rows + r)] = x[static_cast<std::size_t>(r * cols + c)];
    }
  }
  return out;
}

std::vector<std::int32_t> Ids(const json::Value& v) {
  std::vector<std::int32_t> out;
  for (const auto& x : v.items()) {
    out.push_back(static_cast<std::int32_t>(x.as_double()));
  }
  return out;
}

std::string Join(const std::vector<std::int32_t>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size() && i < 24; ++i) {
    s += (i ? " " : "") + std::to_string(v[i]);
  }
  return s + (v.size() > 24 ? " ..." : "");
}

}  // namespace

int main() {
  const char* model_env = std::getenv("GUFO_INDEXTTS_MODEL");
  const char* ref_env = std::getenv("GUFO_INDEXTTS_REF");
  const char* voice_env = std::getenv("GUFO_INDEXTTS_VOICE");
  const char* stages_env = std::getenv("GUFO_INDEXTTS_STAGES");
  const char* out_env = std::getenv("GUFO_INDEXTTS_OUT");
  if (model_env == nullptr || ref_env == nullptr || voice_env == nullptr) {
    std::printf("SKIP: set GUFO_INDEXTTS_MODEL, GUFO_INDEXTTS_REF and GUFO_INDEXTTS_VOICE\n");
    return 77;
  }
  const std::filesystem::path model_dir(model_env);
  const std::filesystem::path ref_dir(ref_env);
  const std::string stages = stages_env != nullptr ? stages_env : "";
  std::string error;
  auto oracle = SafetensorsFile::Open(ref_dir / "oracle.safetensors", &error);
  if (oracle == nullptr || !std::filesystem::exists(ref_dir / "oracle.json")) {
    std::printf("SKIP: oracle missing: %s\n", error.c_str());
    return 77;
  }
  const json::Value meta = json::parse(ReadText(ref_dir / "oracle.json"));
  const auto t_all = std::chrono::steady_clock::now();
  auto O = [&](const char* name) { return oracle->ToF32(name); };

  // ------------------------------------------------------------ text ----
  auto tokenizer = itts::Tokenizer::Load(model_dir / "multilingual_zh_ja_yue_char_del.tiktoken", &error);
  if (tokenizer == nullptr) {
    std::printf("FAIL tokenizer: %s\n", error.c_str());
    return 1;
  }
  if (Want(stages, "text")) {
    Expect("tiktoken vocabulary", static_cast<double>(tokenizer->rank_count() + meta.find("special_tokens")->size()) ==
                                      meta.member_double("n_vocab_ranks"),
           std::to_string(tokenizer->rank_count()) + " ranks + specials");
    int bad_specials = 0;
    for (const auto& [name, id] : meta.find("special_tokens")->members()) {
      if (tokenizer->SpecialId(name) != static_cast<std::int32_t>(id.as_double())) {
        ++bad_specials;
      }
    }
    Expect("special tokens", bad_specials == 0, std::to_string(meta.find("special_tokens")->size()) +
                                                    " checked, " + std::to_string(bad_specials) + " wrong");
    Expect("lang token de", itts::LanguageToken("de") == static_cast<std::int32_t>(meta.member_double("lang_token")),
           std::to_string(itts::LanguageToken("de")));
    int index = 0;
    for (const auto& c : meta.find("text_cases")->items()) {
      const std::string input = c.member_str("input");
      const itts::PreparedText got = itts::PrepareForSynthesis(*tokenizer, input);
      const std::string tag = "text case " + std::to_string(index++);
      Expect(tag + " prepare_german", got.prepared == c.member_str("prepared"),
             got.prepared == c.member_str("prepared") ? "'" + got.prepared + "'"
                                                       : "got '" + got.prepared + "' want '" + c.member_str("prepared") + "'");
      Expect(tag + " final text", got.final_text == c.member_str("final"), got.final_text);
      std::vector<std::vector<std::int32_t>> want;
      for (const auto& seg : c.find("ids")->items()) {
        want.push_back(Ids(seg));
      }
      bool same = got.ids == want;
      Expect(tag + " ids", same,
             std::to_string(got.ids.size()) + " segment(s), " +
                 (got.ids.empty() ? std::string() : Join(got.ids[0])) +
                 (same || want.empty() ? std::string() : " | want " + Join(want[0])));
    }
  }

  // ----------------------------------------------------------- audio ----
  itts::MonoAudio voice;
  if (!itts::ReadWavMono(voice_env, &voice, &error)) {
    std::printf("FAIL voice: %s\n", error.c_str());
    return 1;
  }
  voice.samples.resize(std::min<std::size_t>(voice.samples.size(), static_cast<std::size_t>(15 * voice.rate)));
  const std::vector<float> a22 = itts::ResampleTorchaudio(voice.samples, voice.rate, 22050);
  const std::vector<float> a16 = itts::ResampleTorchaudio(voice.samples, voice.rate, 16000);
  std::vector<float> e16 = itts::ResampleSoxrHq(voice.samples, voice.rate, 16000);
  e16.resize(std::min<std::size_t>(e16.size(), 15 * 16000));
  const std::vector<float> ref_mel_want = O("ref_mel");  // [80, T]
  const auto ref_frames = static_cast<std::int64_t>(ref_mel_want.size() / 80);
  if (Want(stages, "audio")) {
    Compare("wav load", voice.samples, O("audio"));
    Compare("resample 22.05k (torchaudio)", a22, O("audio_22k"));
    Compare("resample 16k (torchaudio)", a16, O("audio_16k"));
    Compare("resample 16k (soxr_hq)", e16, O("emo_16k"));
    std::int64_t frames = 0;
    Compare("ref mel 22k", itts::MelSpectrogram22k(a22, &frames), ref_mel_want);
    std::int64_t fb = 0;
    Compare("campplus fbank", itts::CampplusFbank(a16, &fb), O("campplus_fbank"));
    const itts::W2vFeatures f = itts::W2vBertFeatures(a16);
    Compare("w2v features (spk)", f.features, O("spk_w2v_input"));
    const itts::W2vFeatures fe = itts::W2vBertFeatures(e16);
    Compare("w2v features (emo)", fe.features, O("emo_w2v_input"));
  }

  // --------------------------------------------------------- CAM++ ------
  std::vector<float> style = O("style");
  if (Want(stages, "campplus")) {
    auto cam = itts::CamPlus::Load(model_dir / "campplus" / "campplus_cn_common.bin", &error);
    if (cam == nullptr) {
      std::printf("FAIL campplus load: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> fbank = O("campplus_fbank");
    std::vector<float> head;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> got = cam->Embed(fbank, static_cast<std::int64_t>(fbank.size() / 80), &head);
    Compare("campplus head", head, O("campplus_head"));
    Compare("campplus style", got, style);
    std::printf("     campplus %.2f s\n", Seconds(t0));
  }

  if (oracle->Find("spk_cond_emb") == nullptr) {  // light oracle (text, audio, CAM++ only)
    std::printf("light oracle: later stages skipped\ntotal %.1f s, %d failure(s)\n", Seconds(t_all), g_failures);
    return g_failures == 0 ? 0 : 1;
  }

  // ------------------------------------------------------ w2v-BERT ------
  std::vector<float> spk_cond = O("spk_cond_emb");
  std::vector<float> emo_cond = O("emo_cond_emb");
  const auto cond_rows = static_cast<std::int64_t>(spk_cond.size() / 1024);
  if (Want(stages, "w2v")) {
    auto w2v = itts::W2vBert::Load(model_dir / "w2v-bert-2.0", model_dir / "wav2vec2bert_stats.pt", &error);
    if (w2v == nullptr) {
      std::printf("FAIL w2v load: %s\n", error.c_str());
      return 1;
    }
    for (const std::string tag : {"spk", "emo"}) {
      itts::W2vFeatures f;
      f.features = oracle->ToF32(tag + "_w2v_input");
      f.rows = static_cast<std::int64_t>(f.features.size() / 160);
      for (const std::int64_t m : oracle->ToI64(tag + "_w2v_mask")) {
        f.mask.push_back(static_cast<std::uint8_t>(m));
      }
      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<float> got = w2v->Embed(f, [&](const std::string& name, const std::vector<float>& v) {
        Compare(tag + " w2v " + name, v, oracle->ToF32(tag + "_w2v_" + name));
      });
      Compare(tag + " cond_emb", got, oracle->ToF32(tag + "_cond_emb"));
      std::printf("     w2v-BERT (%s, %lld rows) %.2f s\n", tag.c_str(), static_cast<long long>(f.rows), Seconds(t0));
    }
  }

  // -------------------------------------------------- GPT / emotion -----
  auto gpt = itts::UnifiedVoice::Load(model_dir / "german" / "gpt.pth", &error);
  if (gpt == nullptr) {
    std::printf("FAIL gpt load: %s\n", error.c_str());
    return 1;
  }
  const std::vector<float> emovec_want = O("emovec");
  if (Want(stages, "emo")) {
    std::vector<float> vecs[2];
    int idx = 0;
    for (const std::string tag : {"emo", "spk"}) {
      const std::vector<float>& cond = tag == "emo" ? emo_cond : spk_cond;
      const auto t0 = std::chrono::steady_clock::now();
      vecs[idx] = gpt->EmoVec(cond.data(), cond_rows, [&](const std::string& name, const std::vector<float>& v) {
        Compare(tag + " " + name, v, oracle->ToF32(tag + "_" + name));
      });
      Compare(tag + " emovec", vecs[idx], oracle->ToF32(tag + "_emovec"));
      std::printf("     emotion conditioning (%s) %.2f s\n", tag.c_str(), Seconds(t0));
      ++idx;
    }
    std::vector<float> merged(vecs[0].size());
    for (std::size_t i = 0; i < merged.size(); ++i) {
      merged[i] = vecs[1][i] + 1.0F * (vecs[0][i] - vecs[1][i]);
    }
    Compare("emovec merged", merged, O("emovec_merged"));
    auto em = itts::EmotionMatrix::Load(model_dir / "feat1.pt", model_dir / "feat2.pt", &error);
    std::vector<float> emo_in;
    for (const auto& v : meta.find("emo_vector")->items()) {
      emo_in.push_back(static_cast<float>(v.as_double()));
    }
    const std::vector<float> w = itts::NormalizeEmotionVector(emo_in);
    std::vector<std::int32_t> index;
    const std::vector<float> mat = em->Mix(w, style, &index);
    Expect("emotion matrix rows", index == Ids(*meta.find("emo_index")), Join(index));
    Compare("emovec_mat", mat, O("emovec_mat"));
    float wsum = 0.0F;
    for (const float x : w) {
      wsum += x;
    }
    std::vector<float> ev(mat.size());
    const std::vector<float> merged_want = O("emovec_merged");
    for (std::size_t i = 0; i < ev.size(); ++i) {
      ev[i] = mat[i] + (1.0F - wsum) * merged_want[i];
    }
    Compare("emovec (final)", ev, emovec_want);
  }

  const std::vector<std::int32_t> text_ids = [&] {
    std::vector<std::int32_t> v;
    for (const std::int64_t x : oracle->ToI64("text_ids")) {
      v.push_back(static_cast<std::int32_t>(x));
    }
    return v;
  }();
  const std::vector<std::int32_t> want_tokens = [&] {
    std::vector<std::int32_t> v;
    for (const std::int64_t x : oracle->ToI64("gpt_tokens")) {
      v.push_back(static_cast<std::int32_t>(x));
    }
    return v;
  }();
  std::vector<double> uniforms;
  for (const auto& v : meta.find("gpt_u")->items()) {
    uniforms.push_back(v.as_double());
  }
  itts::SamplingOptions sampling;
  const std::vector<float> spk_latent = gpt->SpeakerLatent(style);
  std::int64_t rows = 0;
  const std::vector<float> prefix =
      gpt->BuildPrefix(spk_latent, emovec_want, text_ids, itts::LanguageToken("de"), &rows);
  if (Want(stages, "gpt")) {
    Compare("spk latent", spk_latent, O("spk_latent"));
    const std::vector<float> want_prefix = O("gpt_prefix_emb");  // [pad][cond][text]
    const std::int64_t dim = gpt->config().dim;
    const std::int64_t pad = static_cast<std::int64_t>(want_prefix.size()) / dim - rows;
    Expect("prefix layout", pad >= 0, std::to_string(rows) + " rows + " + std::to_string(pad) + " masked pad row(s)");
    if (pad >= 0) {
      Compare("prefix embeddings", prefix.data(), want_prefix.data() + pad * dim, prefix.size());
    }
    // teacher forcing over all oracle steps
    const std::vector<float> scores_want = O("gpt_scores");
    const std::int64_t vocab = gpt->config().mel_codes;
    const auto steps = static_cast<std::int64_t>(scores_want.size()) / vocab;
    itts::UnifiedVoice::State st;
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    gpt->Prefill(prefix, rows, &st, &logits, steps + 4);
    const double prefill_s = Seconds(t0);
    std::vector<std::int32_t> history = {1, gpt->config().start_mel};
    double worst = 1.0;
    int token_mismatch = 0;
    const auto t1 = std::chrono::steady_clock::now();
    for (std::int64_t s = 0; s < steps; ++s) {
      itts::ApplyRepetitionPenalty(logits, history, sampling.repetition_penalty);
      double max_abs = 0.0;
      const float* want = scores_want.data() + s * vocab;
      double dot = 0.0, a2 = 0.0, b2 = 0.0;
      for (std::int64_t i = 0; i < vocab; ++i) {
        dot += static_cast<double>(logits[static_cast<std::size_t>(i)]) * want[i];
        a2 += static_cast<double>(logits[static_cast<std::size_t>(i)]) * logits[static_cast<std::size_t>(i)];
        b2 += static_cast<double>(want[i]) * want[i];
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(logits[static_cast<std::size_t>(i)]) - want[i]));
      }
      worst = std::min(worst, dot / std::sqrt(a2 * b2));
      const std::int32_t tok = itts::SampleToken(logits, sampling, uniforms[static_cast<std::size_t>(s)]);
      if (tok != want_tokens[static_cast<std::size_t>(s)]) {
        ++token_mismatch;
        std::printf("     step %lld: token %d vs oracle %d (max_abs %.3g)\n", static_cast<long long>(s), tok,
                    want_tokens[static_cast<std::size_t>(s)], max_abs);
      }
      const std::int32_t forced = want_tokens[static_cast<std::size_t>(s)];
      history.push_back(forced);
      if (s + 1 < steps) {
        gpt->Step(forced, &st, &logits);
      }
    }
    Expect("GPT teacher forcing scores", worst >= kMinCos,
           "worst step cos=" + std::to_string(worst) + " over " + std::to_string(steps) + " steps");
    Expect("GPT sampled tokens (oracle u)", token_mismatch == 0,
           std::to_string(steps - token_mismatch) + "/" + std::to_string(steps) + " equal");
    std::printf("     prefill %.2f s, %.3f s per AR step\n", prefill_s, Seconds(t1) / std::max<std::int64_t>(1, steps));
  }
  const std::vector<std::int32_t> codes_want = [&] {
    std::vector<std::int32_t> v;
    for (const std::int64_t x : oracle->ToI64("codes")) {
      v.push_back(static_cast<std::int32_t>(x));
    }
    return v;
  }();
  if (Want(stages, "free")) {
    itts::UnifiedVoice::State st;
    std::vector<float> logits;
    gpt->Prefill(prefix, rows, &st, &logits, 600);
    std::vector<std::int32_t> history = {1, gpt->config().start_mel};
    std::vector<std::int32_t> codes;
    bool stopped = false;
    for (std::size_t s = 0; s < 600; ++s) {
      itts::ApplyRepetitionPenalty(logits, history, sampling.repetition_penalty);
      const double u = s < uniforms.size() ? uniforms[s] : 0.5;
      const std::int32_t tok = itts::SampleToken(logits, sampling, u);
      history.push_back(tok);
      if (tok == gpt->config().stop_mel) {
        stopped = true;
        break;
      }
      codes.push_back(tok);
      gpt->Step(tok, &st, &logits);
    }
    Expect("GPT free run (oracle u)", codes == codes_want && stopped,
           std::to_string(codes.size()) + " codes vs " + std::to_string(codes_want.size()) + (stopped ? ", stopped" : ""));
  }
  gpt.reset();

  // ------------------------------------------------- codec / s2mel ------
  const std::vector<float> s_infer_want = O("S_infer");
  if (Want(stages, "codec")) {
    auto codec = itts::SemanticCodec::Load(model_dir / "codec.pth", &error);
    if (codec == nullptr) {
      std::printf("FAIL codec load: %s\n", error.c_str());
      return 1;
    }
    std::vector<float> vq, dec;
    const std::vector<float> s = codec->Decode(codes_want, &vq, &dec);
    Compare("codec vq2emb", Tr(vq, static_cast<std::int64_t>(codes_want.size()), 1024), O("codec_vq2emb"));
    Compare("codec decoder", dec, O("codec_decoder"));
    Compare("S_infer", s, s_infer_want);
  }
  auto s2mel = itts::S2Mel::Load(model_dir / "s2mel.pth", &error);
  if (s2mel == nullptr) {
    std::printf("FAIL s2mel load: %s\n", error.c_str());
    return 1;
  }
  const std::vector<float> prompt_want = O("prompt_condition");
  const std::vector<float> cond_want = O("cond");
  const auto target = static_cast<std::int64_t>(cond_want.size() / 512);
  if (Want(stages, "s2mel")) {
    const auto t0 = std::chrono::steady_clock::now();
    Compare("prompt_condition", s2mel->Regulate(spk_cond.data(), cond_rows, ref_frames), prompt_want);
    const auto srows = static_cast<std::int64_t>(s_infer_want.size() / 1024);
    const auto tl = static_cast<std::int64_t>(static_cast<double>(srows) * 1.72 * 1.0);
    Expect("target length", tl == target, std::to_string(tl) + " vs " + std::to_string(target));
    Compare("cond (length regulator)", s2mel->Regulate(s_infer_want.data(), srows, target), cond_want);
    std::printf("     length regulator %.2f s\n", Seconds(t0));
  }
  const std::int64_t total = ref_frames + target;
  std::vector<float> mu(static_cast<std::size_t>(total * 512));
  std::copy(prompt_want.begin(), prompt_want.end(), mu.begin());
  std::copy(cond_want.begin(), cond_want.end(), mu.begin() + ref_frames * 512);
  const std::vector<float> ref_mel_tm = Tr(ref_mel_want, 80, ref_frames);
  if (Want(stages, "dit")) {
    const std::vector<float> x0 = O("cfm_call0_x");        // [2, 80, T]
    const std::vector<float> p0 = O("cfm_call0_prompt_x");  // [2, 80, T]
    const std::vector<float> v0 = O("cfm_velocity_call0");  // [2, 80, T]
    const std::vector<float> t0v = O("cfm_call0_t");
    const std::size_t half = static_cast<std::size_t>(80 * total);
    const std::vector<float> xa = Tr(std::vector<float>(x0.begin(), x0.begin() + half), 80, total);
    const std::vector<float> pa = Tr(std::vector<float>(p0.begin(), p0.begin() + half), 80, total);
    const std::vector<float> zeros_p(half, 0.0F), zeros_mu(mu.size(), 0.0F), zeros_s(192, 0.0F);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> vc = s2mel->Velocity(xa.data(), pa.data(), t0v[0], style.data(), mu.data(), total);
    const double one = Seconds(t0);
    const std::vector<float> vu = s2mel->Velocity(xa.data(), zeros_p.data(), t0v[1], zeros_s.data(), zeros_mu.data(), total);
    Compare("DiT velocity (cond) step 0", Tr(vc, total, 80), std::vector<float>(v0.begin(), v0.begin() + half));
    Compare("DiT velocity (uncond) step 0", Tr(vu, total, 80), std::vector<float>(v0.begin() + half, v0.end()));
    const std::vector<float> x24 = O("cfm_call24_x");
    const std::vector<float> v24 = O("cfm_velocity_call24");
    const std::vector<float> xb = Tr(std::vector<float>(x24.begin(), x24.begin() + half), 80, total);
    const std::vector<float> span = itts::TorchLinspace01(25);
    float tt = span[0];
    for (int s = 1; s < 25; ++s) {
      tt = tt + (span[static_cast<std::size_t>(s)] - span[static_cast<std::size_t>(s - 1)]);
    }
    const std::vector<float> vc24 = s2mel->Velocity(xb.data(), pa.data(), tt, style.data(), mu.data(), total);
    Compare("DiT velocity (cond) step 24", Tr(vc24, total, 80), std::vector<float>(v24.begin(), v24.begin() + half));
    std::printf("     DiT one evaluation (%lld frames) %.2f s\n", static_cast<long long>(total), one);
  }
  std::vector<float> vc_target = Tr(O("vc_target"), 80, target);  // [target, 80]
  if (Want(stages, "cfm")) {
    const std::vector<float> noise = Tr(O("cfm_noise"), 80, total);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> mel = s2mel->Sample(noise, mu.data(), total, ref_mel_tm.data(), ref_frames, style.data(),
                                                 itts::CfmOptions{});
    Compare("CFM 25 steps (oracle noise)", Tr(mel, total, 80), O("cfm_out"));
    std::printf("     CFM %.1f s\n", Seconds(t0));
  }
  s2mel.reset();
  if (Want(stages, "vocoder")) {
    auto voc = itts::BigVgan::Load(model_dir / "bigvgan_v2_22khz_80band_256x", &error);
    if (voc == nullptr) {
      std::printf("FAIL bigvgan load: %s\n", error.c_str());
      return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> wav = voc->Synthesize(vc_target.data(), target);
    Compare("BigVGAN waveform", wav, O("wav_float"));
    std::printf("     BigVGAN %.1f s for %.2f s audio\n", Seconds(t0), static_cast<double>(wav.size()) / 22050.0);
  }
  if (Want(stages, "engine")) {
    auto engine = itts::Engine::Create(model_dir, &error);
    if (engine == nullptr) {
      std::printf("FAIL engine: %s\n", error.c_str());
      return 1;
    }
    itts::SynthesisRequest req;
    req.text = meta.member_str("text");
    req.reference = voice.samples;
    req.reference_rate = voice.rate;
    std::array<float, 8> emo{};
    int i = 0;
    for (const auto& v : meta.find("emo_vector")->items()) {
      emo[static_cast<std::size_t>(i++)] = static_cast<float>(v.as_double());
    }
    req.emotion = emo;
    req.forced_uniforms = &uniforms;
    const std::vector<float> noise_cm = O("cfm_noise");
    req.forced_noise = &noise_cm;
    itts::SynthesisResult res;
    const auto t0 = std::chrono::steady_clock::now();
    if (!engine->Generate(req, {}, {}, &res, &error)) {
      std::printf("FAIL engine generate: %s\n", error.c_str());
      return 1;
    }
    const double secs = Seconds(t0);
    Expect("engine codes", res.codes == codes_want,
           std::to_string(res.codes.size()) + " vs " + std::to_string(codes_want.size()));
    std::vector<float> got(res.samples.begin(), res.samples.end());
    std::vector<float> want;
    for (const std::int64_t v : oracle->ToI64("wav_int16")) {
      want.push_back(static_cast<float>(v));
    }
    Compare("engine int16 waveform", got, want);
    std::printf("     engine: %.2f s audio in %.1f s (reference %.1f, gpt %.1f, s2mel %.1f, vocoder %.1f)\n",
                static_cast<double>(res.samples.size()) / 22050.0, secs, res.reference_seconds, res.gpt_seconds,
                res.s2mel_seconds, res.vocoder_seconds);
    if (out_env != nullptr) {
      itts::WriteWavPcm16(std::filesystem::path(out_env) / "indextts_engine_oracle.wav", res.samples, 22050);
    }
  }
  std::printf("total %.1f s, %d failure(s)\n", Seconds(t_all), g_failures);
  return g_failures == 0 ? 0 : 1;
}
