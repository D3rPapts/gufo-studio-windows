// Written with AI assistance (Claude Code).
#include "src/models/indextts/hip/blas.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/models/indextts/hip/ops.hpp"

namespace gufo::models::indextts::hip {
namespace {

constexpr int kMaxAlgorithms = 16;
constexpr int kTimedRuns = 2;
constexpr std::size_t kWorkspaceBytes = 32U << 20U;

void CheckHip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("IndexTTS HIP ") + what + ": " + hipGetErrorString(status));
  }
}

void CheckBlas(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("IndexTTS hipBLAS ") + what + " failed: " +
                             std::to_string(static_cast<int>(status)));
  }
}

struct Plan {
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t w_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};
  std::size_t workspace{0};
  bool usable{false};

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    if (y_layout != nullptr) (void)hipblasLtMatrixLayoutDestroy(y_layout);
    if (x_layout != nullptr) (void)hipblasLtMatrixLayoutDestroy(x_layout);
    if (w_layout != nullptr) (void)hipblasLtMatrixLayoutDestroy(w_layout);
    if (operation != nullptr) (void)hipblasLtMatmulDescDestroy(operation);
  }
};

}  // namespace

struct Blas::Impl {
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  hipblasLtHandle_t lt{nullptr};
  void* workspace{nullptr};
  // key: padded rows, m, k, accumulate
  std::map<std::tuple<std::size_t, std::size_t, std::size_t, bool>, std::unique_ptr<Plan>> plans;
  bool verbose{false};

  ~Impl() {
    plans.clear();
    if (workspace != nullptr) (void)hipFree(workspace);
    if (lt != nullptr) (void)hipblasLtDestroy(lt);
    if (blas != nullptr) (void)hipblasDestroy(blas);
  }

  hipblasStatus_t Launch(const Plan& p, const void* w, const void* x, float* y, bool accumulate) const {
    const float alpha = 1.0F;
    const float beta = accumulate ? 1.0F : 0.0F;
    return hipblasLtMatmul(lt, p.operation, &alpha, w, p.w_layout, x, p.x_layout, &beta, y, p.y_layout, y,
                           p.y_layout, &p.algorithm, p.workspace > 0 ? workspace : nullptr, p.workspace,
                           stream);
  }

  // Times every usable heuristic result on the caller's operands (output into
  // a scratch buffer, so an accumulating GEMM does not touch y) and keeps the
  // fastest.
  void Tune(Plan* p, std::size_t rows, std::size_t m, std::size_t k, bool accumulate, const void* w,
            const void* x) {
    if (lt == nullptr ||
        hipblasLtMatmulDescCreate(&p->operation, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&p->w_layout, HIP_R_16BF, k, m, k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&p->x_layout, HIP_R_16BF, k, rows, k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&p->y_layout, HIP_R_32F, m, rows, m) != HIPBLAS_STATUS_SUCCESS) {
      return;
    }
    // Row-major y[rows, m] is column-major y^T[m, rows] = W^T(col-major [k, m])^T * x^T([k, rows]).
    const hipblasOperation_t ta = HIPBLAS_OP_T;
    if (hipblasLtMatmulDescSetAttribute(p->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) !=
        HIPBLAS_STATUS_SUCCESS) {
      return;
    }
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
      return;
    }
    const std::uint64_t max_ws = kWorkspaceBytes;
    (void)hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                                                sizeof(max_ws));
    std::vector<hipblasLtMatmulHeuristicResult_t> results(kMaxAlgorithms);
    int count = 0;
    const hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(lt, p->operation, p->w_layout, p->x_layout,
                                                               p->y_layout, p->y_layout, pref, kMaxAlgorithms,
                                                               results.data(), &count);
    (void)hipblasLtMatmulPreferenceDestroy(pref);
    if (st != HIPBLAS_STATUS_SUCCESS || count <= 0) {
      return;
    }
    float* scratch = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&scratch), rows * m * sizeof(float)) != hipSuccess) {
      (void)hipGetLastError();
      return;
    }
    (void)hipMemsetAsync(scratch, 0, rows * m * sizeof(float), stream);
    hipEvent_t begin = nullptr, end = nullptr;
    (void)hipEventCreate(&begin);
    (void)hipEventCreate(&end);
    float best = -1.0F;
    hipblasLtMatmulAlgo_t best_algo{};
    std::size_t best_ws = 0;
    for (int i = 0; i < count; ++i) {
      if (results[i].state != HIPBLAS_STATUS_SUCCESS || results[i].workspaceSize > kWorkspaceBytes) {
        continue;
      }
      p->algorithm = results[i].algo;
      p->workspace = results[i].workspaceSize;
      bool ok = Launch(*p, w, x, scratch, accumulate) == HIPBLAS_STATUS_SUCCESS;  // warm-up, not timed
      float ms = -1.0F;
      if (ok && hipEventRecord(begin, stream) == hipSuccess) {
        for (int r = 0; r < kTimedRuns && ok; ++r) {
          ok = Launch(*p, w, x, scratch, accumulate) == HIPBLAS_STATUS_SUCCESS;
        }
        if (ok && hipEventRecord(end, stream) == hipSuccess && hipEventSynchronize(end) == hipSuccess) {
          (void)hipEventElapsedTime(&ms, begin, end);
        }
      }
      if (ms < 0.0F) {
        (void)hipGetLastError();
        continue;
      }
      if (best < 0.0F || ms < best) {
        best = ms;
        p->usable = true;
        best_algo = results[i].algo;
        best_ws = results[i].workspaceSize;
      }
    }
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    (void)hipStreamSynchronize(stream);
    (void)hipFree(scratch);
    if (p->usable) {
      p->algorithm = best_algo;
      p->workspace = best_ws;
      if (verbose) {
        const double tflops = 2.0 * static_cast<double>(rows * m * k) / (best / kTimedRuns * 1e-3) * 1e-12;
        std::fprintf(stderr, "indextts gemm plan rows=%zu m=%zu k=%zu acc=%d: %.3f ms (%.1f TFLOPS)\n", rows, m, k,
                     accumulate ? 1 : 0, best / kTimedRuns, tflops);
      }
    }
  }
};

Blas::Blas(hipStream_t stream) : impl_(std::make_unique<Impl>()) {
  impl_->stream = stream;
  const char* v = std::getenv("GUFO_INDEXTTS_PROFILE");
  impl_->verbose = v != nullptr && v[0] == '1';
  CheckBlas(hipblasCreate(&impl_->blas), "create");
  CheckBlas(hipblasSetStream(impl_->blas, stream), "set stream");
  if (hipblasLtCreate(&impl_->lt) != HIPBLAS_STATUS_SUCCESS) {
    impl_->lt = nullptr;
    std::fprintf(stderr, "IndexTTS: hipBLASLt unavailable, using hipBLAS GemmEx\n");
  }
  CheckHip(hipMalloc(&impl_->workspace, kWorkspaceBytes), "gemm workspace");
}

Blas::~Blas() = default;

std::size_t Blas::PaddedRows(std::size_t rows) { return (rows + 127U) / 128U * 128U; }

void Blas::Gemm(const std::uint16_t* w, const std::uint16_t* x, float* y, std::size_t rows, std::size_t m,
                std::size_t k, bool accumulate) {
  Impl& s = *impl_;
  const std::size_t padded = PaddedRows(rows);
  const auto key = std::make_tuple(padded, m, k, accumulate);
  auto found = s.plans.find(key);
  if (found == s.plans.end()) {
    auto plan = std::make_unique<Plan>();
    s.Tune(plan.get(), padded, m, k, accumulate, w, x);
    if (!plan->usable) {
      std::fprintf(stderr, "IndexTTS: no hipBLASLt algorithm for rows=%zu m=%zu k=%zu, using hipBLAS\n", padded,
                   m, k);
    }
    found = s.plans.emplace(key, std::move(plan)).first;
  }
  if (found->second->usable) {
    CheckBlas(s.Launch(*found->second, w, x, y, accumulate), "Lt matmul");
    return;
  }
  constexpr auto kLimit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (rows > kLimit || m > kLimit || k > kLimit) {
    throw std::length_error("IndexTTS GEMM exceeds library limits");
  }
  const float alpha = 1.0F;
  const float beta = accumulate ? 1.0F : 0.0F;
  CheckBlas(hipblasGemmEx(s.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m), static_cast<int>(rows),
                          static_cast<int>(k), &alpha, w, HIP_R_16BF, static_cast<int>(k), x, HIP_R_16BF,
                          static_cast<int>(k), &beta, y, HIP_R_32F, static_cast<int>(m), HIPBLAS_COMPUTE_32F,
                          HIPBLAS_GEMM_DEFAULT),
            "GemmEx");
}

void Blas::Attention64(const float* q, const float* k, const float* v, float* out, float* scores,
                       std::size_t rows, std::size_t heads) {
  Impl& s = *impl_;
  const int n = static_cast<int>(rows);
  const int ld = static_cast<int>(heads * 64);
  const int b = static_cast<int>(heads);
  const float scale = 0.125F;  // 1 / sqrt(64)
  const float one = 1.0F;
  const float zero = 0.0F;
  // scores[h] row-major [query, key] = column-major [key, query] = K_h * Q_h^T
  CheckBlas(hipblasSgemmStridedBatched(s.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, 64, &scale, k, ld, 64, q, ld, 64,
                                       &zero, scores, n, static_cast<hipblasStride>(rows) * n, b),
            "attention scores");
  LaunchSoftmaxRows(scores, heads * rows, rows, s.stream);
  // out_h row-major [query, 64] (ld heads*64) = column-major [64, query] = V_h^T(col [64, key]) * P_h^T
  CheckBlas(hipblasSgemmStridedBatched(s.blas, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, n, n, &one, v, ld, 64, scores, n,
                                       static_cast<hipblasStride>(rows) * n, &zero, out, ld, 64, b),
            "attention values");
}

}  // namespace gufo::models::indextts::hip

#endif  // defined(ENGINE_ENABLE_HIP)
