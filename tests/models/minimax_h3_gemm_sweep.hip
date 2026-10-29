// Times the four MiniMax H3 DiT block projections (BF16 weight [rows,
// columns] x BF16 input [batch, columns] -> BF16 [batch, rows], FP32
// accumulation) with every rocBLAS solution and the hipBLASLt heuristic
// algorithms on the current GPU. Prints per shape: the pinned rocBLAS
// solution, rocBLAS default, the fastest rocBLAS solution and the fastest
// hipBLASLt algorithm with its bitwise repeatability and its difference to
// the rocBLAS default result.
// Usage: minimax_h3_gemm_sweep [batch ...]   (default 5979)

#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr rocblas_int kPinned = -1162085109;
constexpr std::size_t kWorkspaceBytes = 128ULL << 20U;

#define CHECK_HIP(expression)                              \
  do {                                                     \
    const hipError_t status_ = (expression);               \
    if (status_ != hipSuccess) {                           \
      std::fprintf(stderr, "%s failed: %s\n", #expression, \
                   hipGetErrorString(status_));            \
      std::exit(1);                                        \
    }                                                      \
  } while (false)

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                    16);
}

float FromBf16(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct Shape {
  const char* name;
  int rows;
  int columns;
};

struct Buffers {
  std::uint16_t* weight{nullptr};
  std::uint16_t* input{nullptr};
  std::uint16_t* output{nullptr};
  void* workspace{nullptr};
};

hipEvent_t g_start = nullptr;
hipEvent_t g_stop = nullptr;

template<typename Launch>
double TimeMs(Launch&& launch, int repeats) {
  if (!launch()) {
    return -1.0;
  }
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipEventRecord(g_start, nullptr));
  for (int index = 0; index < repeats; ++index) {
    if (!launch()) {
      return -1.0;
    }
  }
  CHECK_HIP(hipEventRecord(g_stop, nullptr));
  CHECK_HIP(hipEventSynchronize(g_stop));
  float elapsed = 0.0F;
  CHECK_HIP(hipEventElapsedTime(&elapsed, g_start, g_stop));
  return elapsed / repeats;
}

std::vector<std::uint16_t> Download(const std::uint16_t* device,
                                    std::size_t elements) {
  std::vector<std::uint16_t> host(elements);
  CHECK_HIP(
      hipMemcpy(host.data(), device, elements * 2U, hipMemcpyDeviceToHost));
  return host;
}

// Max |a-b| / max |a| over all elements.
double RelativeDifference(const std::vector<std::uint16_t>& reference,
                          const std::vector<std::uint16_t>& candidate) {
  double maximum_difference = 0.0;
  double maximum_value = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double a = FromBf16(reference[index]);
    const double b = FromBf16(candidate[index]);
    if (!std::isfinite(b)) {
      return INFINITY;
    }
    maximum_difference = std::max(maximum_difference, std::fabs(a - b));
    maximum_value = std::max(maximum_value, std::fabs(a));
  }
  return maximum_value > 0.0 ? maximum_difference / maximum_value : 0.0;
}

rocblas_status RocblasRun(rocblas_handle handle, const Buffers& buffers,
                          int batch, const Shape& shape, rocblas_gemm_algo algo,
                          rocblas_int solution) {
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return rocblas_gemm_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, shape.rows,
      batch, shape.columns, &alpha, buffers.weight, rocblas_datatype_bf16_r,
      shape.columns, buffers.input, rocblas_datatype_bf16_r, shape.columns,
      &beta, buffers.output, rocblas_datatype_bf16_r, shape.rows,
      buffers.output, rocblas_datatype_bf16_r, shape.rows,
      rocblas_datatype_f32_r, algo, solution, 0);
}

void Sweep(rocblas_handle rocblas, hipblasLtHandle_t lt, int batch,
           const Shape& shape) {
  const std::size_t weight_elements =
      static_cast<std::size_t>(shape.rows) * shape.columns;
  const std::size_t input_elements =
      static_cast<std::size_t>(batch) * shape.columns;
  const std::size_t output_elements =
      static_cast<std::size_t>(batch) * shape.rows;
  Buffers buffers;
  CHECK_HIP(hipMalloc(&buffers.weight, weight_elements * 2U));
  CHECK_HIP(hipMalloc(&buffers.input, input_elements * 2U));
  CHECK_HIP(hipMalloc(&buffers.output, output_elements * 2U));
  CHECK_HIP(hipMalloc(&buffers.workspace, kWorkspaceBytes));
  {
    std::mt19937 random(1234);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> host(std::max(weight_elements, input_elements));
    const float weight_scale = 1.0F / std::sqrt(float(shape.columns));
    for (std::size_t index = 0; index < weight_elements; ++index) {
      host[index] = ToBf16(normal(random) * weight_scale);
    }
    CHECK_HIP(hipMemcpy(buffers.weight, host.data(), weight_elements * 2U,
                        hipMemcpyHostToDevice));
    for (std::size_t index = 0; index < input_elements; ++index) {
      host[index] = ToBf16(normal(random));
    }
    CHECK_HIP(hipMemcpy(buffers.input, host.data(), input_elements * 2U,
                        hipMemcpyHostToDevice));
  }
  const double flops = 2.0 * batch * shape.rows * shape.columns;
  const auto tflops = [&](double ms) { return ms > 0 ? flops / ms / 1e9 : 0; };

  // rocBLAS default and pinned.
  const double default_ms = TimeMs(
      [&] {
        return RocblasRun(rocblas, buffers, batch, shape,
                          rocblas_gemm_algo_standard,
                          0) == rocblas_status_success;
      },
      5);
  const std::vector<std::uint16_t> reference =
      Download(buffers.output, output_elements);
  const double pinned_ms = TimeMs(
      [&] {
        return RocblasRun(rocblas, buffers, batch, shape,
                          rocblas_gemm_algo_solution_index,
                          kPinned) == rocblas_status_success;
      },
      5);
  std::printf(
      "shape %s batch=%d rows=%d columns=%d: rocblas default %.2f ms (%.1f "
      "TFLOPS), pinned %d %.2f ms (%.1f TFLOPS)\n",
      shape.name, batch, shape.rows, shape.columns, default_ms,
      tflops(default_ms), kPinned, pinned_ms, tflops(pinned_ms));

  // Every rocBLAS solution.
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  rocblas_int count = 0;
  (void)rocblas_gemm_ex_get_solutions(
      rocblas, rocblas_operation_transpose, rocblas_operation_none, shape.rows,
      batch, shape.columns, &alpha, buffers.weight, rocblas_datatype_bf16_r,
      shape.columns, buffers.input, rocblas_datatype_bf16_r, shape.columns,
      &beta, buffers.output, rocblas_datatype_bf16_r, shape.rows,
      buffers.output, rocblas_datatype_bf16_r, shape.rows,
      rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr,
      &count);
  std::vector<rocblas_int> solutions(static_cast<std::size_t>(count));
  if (count > 0) {
    (void)rocblas_gemm_ex_get_solutions(
        rocblas, rocblas_operation_transpose, rocblas_operation_none,
        shape.rows, batch, shape.columns, &alpha, buffers.weight,
        rocblas_datatype_bf16_r, shape.columns, buffers.input,
        rocblas_datatype_bf16_r, shape.columns, &beta, buffers.output,
        rocblas_datatype_bf16_r, shape.rows, buffers.output,
        rocblas_datatype_bf16_r, shape.rows, rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, 0, solutions.data(), &count);
  }
  std::vector<std::pair<double, rocblas_int>> timed;
  for (const rocblas_int solution : solutions) {
    const double ms = TimeMs(
        [&] {
          return RocblasRun(rocblas, buffers, batch, shape,
                            rocblas_gemm_algo_solution_index,
                            solution) == rocblas_status_success;
        },
        1);
    if (ms > 0) {
      timed.emplace_back(ms, solution);
    }
  }
  std::sort(timed.begin(), timed.end());
  std::printf("  rocblas solutions: %d listed, %zu ran\n", count, timed.size());
  for (std::size_t index = 0; index < std::min<std::size_t>(timed.size(), 4);
       ++index) {
    const rocblas_int solution = timed[index].second;
    const auto launch = [&] {
      return RocblasRun(rocblas, buffers, batch, shape,
                        rocblas_gemm_algo_solution_index,
                        solution) == rocblas_status_success;
    };
    const double ms = TimeMs(launch, 5);
    const auto first = Download(buffers.output, output_elements);
    (void)launch();
    CHECK_HIP(hipDeviceSynchronize());
    const auto second = Download(buffers.output, output_elements);
    std::printf(
        "  rocblas solution %d: %.2f ms (%.1f TFLOPS) repeatable=%d "
        "rel_diff_vs_default=%.2e\n",
        solution, ms, tflops(ms), first == second ? 1 : 0,
        RelativeDifference(reference, first));
  }

  // hipBLASLt heuristic algorithms.
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t weight_layout = nullptr;
  hipblasLtMatrixLayout_t input_layout = nullptr;
  hipblasLtMatrixLayout_t output_layout = nullptr;
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  bool ok =
      hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F) ==
          HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmulDescSetAttribute(operation, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &transpose, sizeof(transpose)) ==
          HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatrixLayoutCreate(&weight_layout, HIP_R_16BF, shape.columns,
                                  shape.rows,
                                  shape.columns) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatrixLayoutCreate(&input_layout, HIP_R_16BF, shape.columns,
                                  batch,
                                  shape.columns) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatrixLayoutCreate(&output_layout, HIP_R_16BF, shape.rows, batch,
                                  shape.rows) == HIPBLAS_STATUS_SUCCESS;
  hipblasLtMatmulPreference_t preference = nullptr;
  std::array<hipblasLtMatmulHeuristicResult_t, 64> results{};
  int algorithms = 0;
  if (ok &&
      hipblasLtMatmulPreferenceCreate(&preference) == HIPBLAS_STATUS_SUCCESS) {
    std::size_t workspace = kWorkspaceBytes;
    (void)hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
        sizeof(workspace));
    (void)hipblasLtMatmulAlgoGetHeuristic(
        lt, operation, weight_layout, input_layout, output_layout,
        output_layout, preference, static_cast<int>(results.size()),
        results.data(), &algorithms);
    (void)hipblasLtMatmulPreferenceDestroy(preference);
  }
  std::vector<std::pair<double, int>> lt_timed;
  for (int index = 0; index < algorithms; ++index) {
    if (results[index].state != HIPBLAS_STATUS_SUCCESS) {
      continue;
    }
    const auto launch = [&] {
      constexpr float lt_alpha = 1.0F;
      constexpr float lt_beta = 0.0F;
      return hipblasLtMatmul(lt, operation, &lt_alpha, buffers.weight,
                             weight_layout, buffers.input, input_layout,
                             &lt_beta, buffers.output, output_layout,
                             buffers.output, output_layout,
                             &results[index].algo, buffers.workspace,
                             results[index].workspaceSize,
                             nullptr) == HIPBLAS_STATUS_SUCCESS;
    };
    const double ms = TimeMs(launch, 1);
    if (ms > 0) {
      lt_timed.emplace_back(ms, index);
    }
  }
  std::sort(lt_timed.begin(), lt_timed.end());
  std::printf("  hipblaslt algorithms: %d heuristic, %zu ran\n", algorithms,
              lt_timed.size());
  for (std::size_t rank = 0; rank < std::min<std::size_t>(lt_timed.size(), 4);
       ++rank) {
    const int index = lt_timed[rank].second;
    const auto launch = [&] {
      constexpr float lt_alpha = 1.0F;
      constexpr float lt_beta = 0.0F;
      return hipblasLtMatmul(lt, operation, &lt_alpha, buffers.weight,
                             weight_layout, buffers.input, input_layout,
                             &lt_beta, buffers.output, output_layout,
                             buffers.output, output_layout,
                             &results[index].algo, buffers.workspace,
                             results[index].workspaceSize,
                             nullptr) == HIPBLAS_STATUS_SUCCESS;
    };
    const double ms = TimeMs(launch, 5);
    const auto first = Download(buffers.output, output_elements);
    (void)launch();
    CHECK_HIP(hipDeviceSynchronize());
    const auto second = Download(buffers.output, output_elements);
    std::printf(
        "  hipblaslt heuristic #%d: %.2f ms (%.1f TFLOPS) workspace=%zu "
        "repeatable=%d rel_diff_vs_default=%.2e\n",
        index, ms, tflops(ms), results[index].workspaceSize,
        first == second ? 1 : 0, RelativeDifference(reference, first));
  }
  (void)hipblasLtMatrixLayoutDestroy(output_layout);
  (void)hipblasLtMatrixLayoutDestroy(input_layout);
  (void)hipblasLtMatrixLayoutDestroy(weight_layout);
  (void)hipblasLtMatmulDescDestroy(operation);
  CHECK_HIP(hipFree(buffers.workspace));
  CHECK_HIP(hipFree(buffers.output));
  CHECK_HIP(hipFree(buffers.input));
  CHECK_HIP(hipFree(buffers.weight));
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> batches;
  for (int index = 1; index < argc; ++index) {
    batches.push_back(std::atoi(argv[index]));
  }
  if (batches.empty()) {
    batches.push_back(5979);
  }
  CHECK_HIP(hipEventCreate(&g_start));
  CHECK_HIP(hipEventCreate(&g_stop));
  rocblas_handle rocblas = nullptr;
  hipblasLtHandle_t lt = nullptr;
  if (rocblas_create_handle(&rocblas) != rocblas_status_success ||
      hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cannot create BLAS handles\n");
    return 1;
  }
  (void)rocblas_set_atomics_mode(rocblas, rocblas_atomics_not_allowed);
  constexpr std::array<Shape, 4> kShapes = {{
      {"qkv", 21504, 5376},
      {"attention_out", 5376, 7168},
      {"fc1", 28672, 5376},
      {"fc2", 5376, 14336},
  }};
  for (const int batch : batches) {
    for (const Shape& shape : kShapes) {
      Sweep(rocblas, lt, batch, shape);
    }
  }
  (void)hipblasLtDestroy(lt);
  (void)rocblas_destroy_handle(rocblas);
  return 0;
}
