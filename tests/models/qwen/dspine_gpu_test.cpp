// DSpine HIP draft against the C++ CPU reference (src/models/qwen/
// dspine_reference.*, itself checked against the numpy oracle): one round with
// the oracle's context (rounded to BF16 as both runners receive it) and anchor.
// Reports draft tokens, candidate overlap, table build time and ms per block.
// Needs the GPU, the draft GGUF, the 27B target GGUF and the oracle
// directory; skips (77) without them. Env as qwen_dspine_reference_test;
// GUFO_DSPINE_SKIP_CPU=1 compares with the oracle drafts only (faster, but
// those were computed on unrounded context rows).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/dspine_reference.hpp"
#include "src/models/qwen/hip/dspine.hpp"
#include "src/models/qwen/hip/executor.hpp"

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

float Bf(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  std::memcpy(&x, &u, 4);
  return x;
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
  const std::string draft_path = Env("GUFO_DSPINE_GGUF",
      "D:/llm-modelle/tencent/Qwen3.8-27B-DSpine-b16-GGUF/Qwen3.8-27B-DSpine-b16-Q8_0.gguf");
  const std::string target_path = Env("GUFO_DSPINE_TARGET",
      "D:/llm-modelle/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf");
  const std::string oracle = Env("GUFO_DSPINE_ORACLE", "D:/claude/gufo-arbeit/dspine/orakel");
  const bool skip_cpu = Env("GUFO_DSPINE_SKIP_CPU", "0") == "1";

  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  std::string error;
  auto ctx = ReadBin<float>(oracle + "/ctx_hidden.f32");
  std::shared_ptr<const gufo::core::GgufReader> target_reader(
      gufo::core::GgufReader::OpenFile(target_path, &error));
  std::shared_ptr<const gufo::core::GgufReader> draft_reader(
      gufo::core::GgufReader::OpenFile(draft_path, &error));
  if (!target_reader || !draft_reader || ctx.empty()) {
    std::printf("SKIP: %s\n", error.c_str());
    return 77;
  }
  int S = 0, T = 0;
  long long anchor = 0;
  {
    std::ifstream m(oracle + "/meta.txt");
    m >> S >> T >> anchor;
  }
  for (float& x : ctx) x = Bf(x);

  // GPU
  auto t0 = std::chrono::steady_clock::now();
  auto target = gufo::hip::QwenGpuModel::CreateFromGguf(target_reader, &error);
  if (!target) {
    std::printf("FAIL: target: %s\n", error.c_str());
    return 1;
  }
  auto t1 = std::chrono::steady_clock::now();
  auto model = gufo::hip::QwenDSpineGpuModel::Create(draft_reader, target, &error);
  if (!model) {
    std::printf("FAIL: draft model: %s\n", error.c_str());
    return 1;
  }
  auto t2 = std::chrono::steady_clock::now();
  std::printf("target load %.1f s, draft load %.1f s (tables %.1f s), draft device %.2f GiB\n",
              std::chrono::duration<double>(t1 - t0).count(),
              std::chrono::duration<double>(t2 - t1).count(), model->TableSeconds(),
              static_cast<double>(model->DeviceBytes()) / (1024.0 * 1024 * 1024));
  auto executor = gufo::hip::QwenDSpineGpuExecutor::Create(model, 4096, &error);
  if (!executor) {
    std::printf("FAIL: executor: %s\n", error.c_str());
    return 1;
  }
  if (!executor->InjectTargetContext(ctx, 0, static_cast<std::uint32_t>(S))) {
    std::printf("FAIL: inject\n");
    return 1;
  }
  std::vector<std::int32_t> gpu_cand;
  const auto gpu = executor->ForwardBlock(static_cast<gufo::tokenization::TokenId>(anchor),
                                          static_cast<std::uint32_t>(T - 1), &gpu_cand);
  // Timing: the same block again (context unchanged).
  constexpr int kRepeats = 20;
  const auto r0 = std::chrono::steady_clock::now();
  for (int r = 0; r < kRepeats; ++r)
    (void)executor->ForwardBlock(static_cast<gufo::tokenization::TokenId>(anchor),
                                 static_cast<std::uint32_t>(T - 1));
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - r0).count() / kRepeats;
  std::printf("GPU block (T=%d, context %d): %.2f ms\n", T, S, ms);

  // Reference
  std::vector<std::int32_t> want;
  std::vector<std::int32_t> want_cand;
  if (skip_cpu) {
    want = ReadBin<std::int32_t>(oracle + "/drafts.i32");
    want_cand = ReadBin<std::int32_t>(oracle + "/cand.i32");
  } else {
    auto ref = gufo::models::qwen::DSpineReference::Open(draft_path, &error);
    if (!ref) {
      std::printf("FAIL: reference: %s\n", error.c_str());
      return 1;
    }
    const auto* embed = target_reader->FindTensor("token_embd.weight");
    const auto* head = target_reader->FindTensor("output.weight");
    const std::size_t H = ref->config().hidden, V = ref->config().vocab;
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
    want = round.drafts;
    want_cand = round.cand;
  }

  std::printf("drafts gpu: ");
  for (auto d : gpu) std::printf("%d ", static_cast<int>(d));
  std::printf("\ndrafts ref: ");
  for (auto d : want) std::printf("%d ", d);
  std::printf("\n");
  std::size_t prefix = 0;
  while (prefix < gpu.size() && prefix < want.size() &&
         static_cast<std::int32_t>(gpu[prefix]) == want[prefix])
    ++prefix;
  std::size_t same = 0;
  for (std::size_t i = 0; i < gpu_cand.size() && i < want_cand.size(); ++i)
    same += gpu_cand[i] == want_cand[i];
  std::printf("equal draft prefix %zu / %zu, candidates equal %zu / %zu\n", prefix,
              want.size(), same, want_cand.size());
  // Near-ties of synthetic context can flip late chain links; the candidate
  // sets and the first tokens must agree.
  const bool ok = prefix >= 1 && same * 10 >= want_cand.size() * 9;
  std::printf(ok ? (prefix == want.size() ? "PASS\n" : "PASS (late divergence)\n")
                 : "FAILED\n");
  return ok ? 0 : 1;
}
