// DSpine CPU reference against the numpy oracle (tools/dspine, dspine_mlx logic):
// one draft round with fixed inputs; draft tokens must match, block rows and
// candidate logits must be close. Needs the converted draft GGUF, the 27B
// target GGUF (embedding + lm_head) and the oracle directory; skips (77)
// without them. Env: GUFO_DSPINE_GGUF, GUFO_DSPINE_TARGET, GUFO_DSPINE_ORACLE.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/dspine_reference.hpp"

namespace {

template <class T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const auto n = static_cast<std::size_t>(f.tellg());
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

std::string Env(const char* k, const char* d) {
  const char* v = std::getenv(k);
  return v && *v ? v : d;
}

void TargetRow(const gufo::core::GgufTensorInfo* t, std::int64_t row, float* out) {
  const std::size_t ne0 = t->dimensions[0];
  const auto* base = static_cast<const std::uint8_t*>(t->data);
  if (t->type == gufo::core::GgmlType::kQ6_K)
    gufo::quant::DequantizeQ6_K(base + row * (ne0 / 256 * 210), out, ne0);
  else if (t->type == gufo::core::GgmlType::kQ4_K)
    gufo::quant::DequantizeQ4_K(base + row * (ne0 / 256 * 144), out, ne0);
  else
    std::abort();
}

}  // namespace

int main() {
  const std::string draft = Env("GUFO_DSPINE_GGUF",
      "D:/llm-modelle/tencent/Qwen3.8-27B-DSpine-b16-GGUF/Qwen3.8-27B-DSpine-b16-Q8_0.gguf");
  const std::string target_path = Env("GUFO_DSPINE_TARGET",
      "D:/llm-modelle/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf");
  const std::string oracle = Env("GUFO_DSPINE_ORACLE", "D:/claude/gufo-arbeit/dspine/orakel");

  std::string error;
  auto ref = gufo::models::qwen::DSpineReference::Open(draft, &error);
  auto tgt = gufo::core::GgufReader::OpenFile(target_path, &error);
  const auto ctx = ReadBin<float>(oracle + "/ctx_hidden.f32");
  if (!ref || !tgt || ctx.empty()) {
    std::printf("SKIP: %s\n", error.c_str());
    return 77;
  }
  int S = 0, T = 0;
  long long anchor = 0;
  {
    std::ifstream m(oracle + "/meta.txt");
    m >> S >> T >> anchor;
  }
  const auto* embed = tgt->FindTensor("token_embd.weight");
  const auto* head = tgt->FindTensor("output.weight");
  const auto& cfg = ref->config();
  const std::size_t H = cfg.hidden, V = cfg.vocab;

  gufo::models::qwen::DSpineTargetAccess access;
  access.embed_row = [&](std::int64_t v, float* out) { TargetRow(embed, v, out); };
  access.head_row = [&](std::int64_t v, float* out) { TargetRow(head, v, out); };
  access.head_logits = [&](const float* rows, std::size_t n, float* logits) {
#pragma omp parallel
    {
      std::vector<float> w(H);
#pragma omp for schedule(static)
      for (std::int64_t v = 0; v < static_cast<std::int64_t>(V); ++v) {
        TargetRow(head, v, w.data());
        for (std::size_t i = 0; i < n; ++i) {
          double acc = 0;
          for (std::size_t j = 0; j < H; ++j) acc += static_cast<double>(rows[i * H + j]) * w[j];
          logits[i * V + static_cast<std::size_t>(v)] = static_cast<float>(acc);
        }
      }
    }
  };

  const auto round = ref->Draft(ctx, static_cast<std::size_t>(S), anchor, access);

  const auto want_drafts = ReadBin<std::int32_t>(oracle + "/drafts.i32");
  const auto want_cand = ReadBin<std::int32_t>(oracle + "/cand.i32");
  const auto want_rows = ReadBin<float>(oracle + "/rows.f32");
  const auto want_logits = ReadBin<float>(oracle + "/top_logits.f32");
  int fails = 0;
  std::printf("drafts cpp:   ");
  for (auto d : round.drafts) std::printf("%d ", d);
  std::printf("\ndrafts numpy: ");
  for (auto d : want_drafts) std::printf("%d ", d);
  std::printf("\n");
  if (round.drafts != want_drafts) { std::printf("FAIL: draft tokens differ\n"); ++fails; }

  double max_rel = 0, ref_max = 0;
  for (std::size_t i = 0; i < want_rows.size(); ++i) {
    max_rel = std::max(max_rel, std::fabs(static_cast<double>(round.rows[i]) - want_rows[i]));
    ref_max = std::max(ref_max, std::fabs(static_cast<double>(want_rows[i])));
  }
  std::printf("rows: max abs diff %.4g of max %.4g (rel %.4g)\n", max_rel, ref_max, max_rel / ref_max);
  if (max_rel / ref_max > 0.03) { std::printf("FAIL: rows too far\n"); ++fails; }

  std::size_t same = 0;
  for (std::size_t i = 0; i < want_cand.size(); ++i) same += round.cand[i] == want_cand[i];
  std::printf("candidates equal: %zu / %zu\n", same, want_cand.size());
  // Top-1 candidate per row must match (the chain starts from the best ones).
  double lmax = 0;
  for (std::size_t i = 0; i < want_logits.size(); ++i)
    if (round.cand[i] == want_cand[i])
      lmax = std::max(lmax, std::fabs(static_cast<double>(round.cand_logits[i]) - want_logits[i]));
  std::printf("candidate logits (same ids): max abs diff %.4g\n", lmax);
  std::printf(fails ? "FAILED\n" : "PASS\n");
  return fails ? 1 : 0;
}
