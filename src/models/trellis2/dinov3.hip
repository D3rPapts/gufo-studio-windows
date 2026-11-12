// TRELLIS.2: DINOv3 ViT-L/16 on HIP (FP32 rocBLAS, derived from triposg/dinov2.hip):
// CLS + 4 register tokens + patches, 2-D RoPE on the patch rows of q/k, LayerScale,
// final LayerNorm without affine. Weights are uploaded once from Dinov3Weights; the
// token count is fixed per instance (512 px -> 1029, 1024 px -> 4101).
// AI-assisted (Claude Code), 2026-11-12.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstddef>
#include <utility>

#include "src/models/trellis2/dinov3.hpp"

namespace gufo::trellis2 {
namespace {

using Cfg = DinoV3Config;
constexpr int kDim = Cfg::width;
constexpr int kThreads = 256;
constexpr int kPatchValues = 3 * Cfg::patch * Cfg::patch;
constexpr int kPrefix = 1 + Cfg::registers;
constexpr int kHalf = Cfg::head_dim / 2;
static_assert(kDim % kThreads == 0);

void SetError(std::string* error, std::string message) {
    if (error != nullptr) *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
    if (status == hipSuccess) return true;
    SetError(error, std::string("TRELLIS.2 DINOv3 ") + operation + ": " + hipGetErrorString(status));
    return false;
}

unsigned Blocks(std::size_t count) { return static_cast<unsigned>((count + kThreads - 1) / kThreads); }

__device__ __forceinline__ float BlockSum(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

// LayerNorm over rows of 1024 (eps 1e-5); weight == nullptr -> no affine.
__global__ void LayerNormKernel(const float* x, const float* weight, const float* bias, float* out) {
    __shared__ float shared[kThreads];
    const float* row = x + static_cast<std::size_t>(blockIdx.x) * kDim;
    float v[kDim / kThreads], sum = 0;
    for (int i = 0; i < kDim / kThreads; ++i) {
        v[i] = row[threadIdx.x + i * kThreads];
        sum += v[i];
    }
    const float mean = BlockSum(sum, shared) / kDim;
    float squares = 0;
    for (int i = 0; i < kDim / kThreads; ++i) {
        v[i] -= mean;
        squares += v[i] * v[i];
    }
    const float rstd = rsqrtf(BlockSum(squares, shared) / kDim + Cfg::layer_norm_eps);
    float* target = out + static_cast<std::size_t>(blockIdx.x) * kDim;
    for (int i = 0; i < kDim / kThreads; ++i) {
        const int c = threadIdx.x + i * kThreads;
        const float n = v[i] * rstd;
        target[c] = weight ? n * weight[c] + bias[c] : n;
    }
}

__global__ void BiasActKernel(float* x, const float* bias, std::size_t count, int columns, int act) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i] + bias[i % columns];
    if (act == 1) v = 0.5F * v * (1.0F + erff(v * 0.7071067811865476F));
    x[i] = v;
}

__global__ void ScaledAddKernel(float* hidden, const float* y, const float* gamma, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) hidden[i] += gamma[i % kDim] * y[i];
}

// RoPE on q and k of the patch rows of the [T, 3*1024] qkv buffer: one thread per
// (patch, head, pair d): pair (d, d+32) rotated by angles[patch, d].
__global__ void RopeKernel(float* qkv, const float* angles, int patches) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(patches) * Cfg::heads * kHalf;
    if (i >= total) return;
    const int d = static_cast<int>(i % kHalf);
    const int h = static_cast<int>((i / kHalf) % Cfg::heads);
    const int p = static_cast<int>(i / (kHalf * Cfg::heads));
    const float a = angles[static_cast<std::size_t>(p) * kHalf + d];
    const float c = cosf(a), s = sinf(a);
    float* row = qkv + static_cast<std::size_t>(kPrefix + p) * 3 * kDim;
    for (int which = 0; which < 2; ++which) {   // q, then k
        float* v = row + which * kDim + h * Cfg::head_dim;
        const float x1 = v[d], x2 = v[d + kHalf];
        v[d] = x1 * c - x2 * s;
        v[d + kHalf] = x2 * c + x1 * s;
    }
}

__global__ void SoftmaxKernel(float* scores, int columns, float scale) {
    __shared__ float shared[kThreads];
    float* row = scores + static_cast<std::size_t>(blockIdx.x) * columns;
    float maximum = -INFINITY;
    for (int c = threadIdx.x; c < columns; c += kThreads) maximum = fmaxf(maximum, row[c] * scale);
    shared[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = shared[0];
    __syncthreads();
    float sum = 0;
    for (int c = threadIdx.x; c < columns; c += kThreads) {
        const float e = expf(row[c] * scale - maximum);
        row[c] = e;
        sum += e;
    }
    sum = BlockSum(sum, shared);
    const float inverse = 1.0F / sum;
    for (int c = threadIdx.x; c < columns; c += kThreads) row[c] *= inverse;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)), bytes_(other.bytes_) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            Release();
            data_ = std::exchange(other.data_, nullptr);
            bytes_ = other.bytes_;
        }
        return *this;
    }
    ~DeviceBuffer() { Release(); }
    bool Allocate(std::size_t bytes, std::string* error) {
        Release();
        bytes_ = bytes;
        return Check(hipMalloc(&data_, std::max<std::size_t>(bytes, 256)), "device allocation", error);
    }
    bool Upload(const std::vector<float>& host, std::string* error) {
        return Allocate(host.size() * 4, error) &&
               Check(hipMemcpy(data_, host.data(), host.size() * 4, hipMemcpyHostToDevice), "weight upload", error);
    }
    float* f32() const { return static_cast<float*>(data_); }
    std::size_t bytes() const { return bytes_; }

private:
    void Release() {
        if (data_ != nullptr) (void)hipFree(data_);
        data_ = nullptr;
        bytes_ = 0;
    }
    void* data_{nullptr};
    std::size_t bytes_{0};
};

}  // namespace

struct Dinov3Hip::Impl {
    ~Impl() {
        if (blas != nullptr) (void)rocblas_destroy_handle(blas);
    }
    struct Layer {
        DeviceBuffer ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ls1, ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b, ls2;
    };
    DeviceBuffer prefix, patch_w, patch_b, angles;
    std::vector<Layer> layers;
    DeviceBuffer rows, hidden, normed, qkv, scores, attn, linear, mlp, output;
    rocblas_handle blas{nullptr};
    std::size_t device_bytes{0};
    int image{0}, grid{0}, patches{0}, tokens{0};

    bool Linear(const float* in, int rows_in, const DeviceBuffer& w, const DeviceBuffer& b, int rows_n, int columns,
                float* out, int act, std::string* error) {
        const float one = 1.0F, zero = 0.0F;
        if (rocblas_sgemm(blas, rocblas_operation_transpose, rocblas_operation_none, rows_n, rows_in, columns, &one,
                          w.f32(), columns, in, columns, &zero, out, rows_n) != rocblas_status_success) {
            SetError(error, "TRELLIS.2 DINOv3 GEMM failed");
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(rows_in) * static_cast<std::size_t>(rows_n);
        BiasActKernel<<<Blocks(count), kThreads>>>(out, b.f32(), count, rows_n, act);
        return Check(hipGetLastError(), "linear", error);
    }
};

Dinov3Hip::Dinov3Hip(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Dinov3Hip::~Dinov3Hip() = default;

std::size_t Dinov3Hip::device_bytes() const { return impl_->device_bytes; }

std::unique_ptr<Dinov3Hip> Dinov3Hip::Create(const Dinov3Weights& w, int image, std::string* error) {
    if (!w.Validate(error)) return nullptr;
    if (image <= 0 || image % Cfg::patch != 0) {
        SetError(error, "TRELLIS.2 DINOv3: image size must be a multiple of 16");
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    Impl& m = *impl;
    m.image = image;
    m.grid = image / Cfg::patch;
    m.patches = m.grid * m.grid;
    m.tokens = kPrefix + m.patches;
    if (rocblas_create_handle(&m.blas) != rocblas_status_success) {
        SetError(error, "TRELLIS.2 DINOv3: rocBLAS handle");
        return nullptr;
    }
    auto up = [&](DeviceBuffer& dst, const std::vector<float>& src) {
        if (!dst.Upload(src, error)) return false;
        m.device_bytes += dst.bytes();
        return true;
    };
    std::vector<float> prefix(w.cls);
    prefix.insert(prefix.end(), w.registers.begin(), w.registers.end());
    std::vector<float> angles;
    dinov3_rope_angles(m.grid, &angles);
    if (!up(m.prefix, prefix) || !up(m.patch_w, w.patch_w) || !up(m.patch_b, w.patch_b) || !up(m.angles, angles))
        return nullptr;
    m.layers.resize(w.layers.size());
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& s = w.layers[i];
        auto& d = m.layers[i];
        if (!up(d.ln1_w, s.ln1_w) || !up(d.ln1_b, s.ln1_b) || !up(d.qkv_w, s.qkv_w) || !up(d.qkv_b, s.qkv_b) ||
            !up(d.out_w, s.out_w) || !up(d.out_b, s.out_b) || !up(d.ls1, s.ls1) || !up(d.ln2_w, s.ln2_w) ||
            !up(d.ln2_b, s.ln2_b) || !up(d.fc1_w, s.fc1_w) || !up(d.fc1_b, s.fc1_b) || !up(d.fc2_w, s.fc2_w) ||
            !up(d.fc2_b, s.fc2_b) || !up(d.ls2, s.ls2))
            return nullptr;
    }
    const int T = m.tokens;
    const std::size_t TD = static_cast<std::size_t>(T) * kDim;
    if (!m.rows.Allocate(static_cast<std::size_t>(m.patches) * kPatchValues * 4, error) || !m.hidden.Allocate(TD * 4, error) ||
        !m.normed.Allocate(TD * 4, error) || !m.qkv.Allocate(TD * 3 * 4, error) ||
        !m.scores.Allocate(static_cast<std::size_t>(Cfg::heads) * T * T * 4, error) || !m.attn.Allocate(TD * 4, error) ||
        !m.linear.Allocate(TD * 4, error) || !m.mlp.Allocate(static_cast<std::size_t>(T) * Cfg::mlp * 4, error) ||
        !m.output.Allocate(TD * 4, error))
        return nullptr;
    return std::unique_ptr<Dinov3Hip>(new Dinov3Hip(std::move(impl)));
}

bool Dinov3Hip::Encode(const float* pixels, std::vector<float>* tokens, std::string* error) {
    Impl& m = *impl_;
    const int T = m.tokens;
    const std::size_t TD = static_cast<std::size_t>(T) * kDim;
    std::vector<float> rows;
    dinov3_patch_rows(pixels, m.image, &rows);
    if (!Check(hipMemcpy(m.rows.f32(), rows.data(), rows.size() * 4, hipMemcpyHostToDevice), "upload", error) ||
        !Check(hipMemcpy(m.hidden.f32(), m.prefix.f32(), static_cast<std::size_t>(kPrefix) * kDim * 4, hipMemcpyDeviceToDevice), "prefix", error) ||
        !m.Linear(m.rows.f32(), m.patches, m.patch_w, m.patch_b, kDim, kPatchValues, m.hidden.f32() + kPrefix * kDim, 0, error))
        return false;
    const float one = 1.0F, zero = 0.0F;
    const std::size_t rope_threads = static_cast<std::size_t>(m.patches) * Cfg::heads * kHalf;
    for (const auto& l : m.layers) {
        LayerNormKernel<<<T, kThreads>>>(m.hidden.f32(), l.ln1_w.f32(), l.ln1_b.f32(), m.normed.f32());
        if (!m.Linear(m.normed.f32(), T, l.qkv_w, l.qkv_b, 3 * kDim, kDim, m.qkv.f32(), 0, error)) return false;
        RopeKernel<<<Blocks(rope_threads), kThreads>>>(m.qkv.f32(), m.angles.f32(), m.patches);
        const float* q = m.qkv.f32();
        if (rocblas_sgemm_strided_batched(m.blas, rocblas_operation_transpose, rocblas_operation_none, T, T,
                                          Cfg::head_dim, &one, q + kDim, 3 * kDim, Cfg::head_dim, q, 3 * kDim,
                                          Cfg::head_dim, &zero, m.scores.f32(), T,
                                          static_cast<rocblas_stride>(T) * T, Cfg::heads) != rocblas_status_success) {
            SetError(error, "TRELLIS.2 DINOv3 attention scores failed");
            return false;
        }
        SoftmaxKernel<<<Cfg::heads * T, kThreads>>>(m.scores.f32(), T, 1.0F / std::sqrt(static_cast<float>(Cfg::head_dim)));
        if (rocblas_sgemm_strided_batched(m.blas, rocblas_operation_none, rocblas_operation_none, Cfg::head_dim, T, T,
                                          &one, q + 2 * kDim, 3 * kDim, Cfg::head_dim, m.scores.f32(), T,
                                          static_cast<rocblas_stride>(T) * T, &zero, m.attn.f32(), kDim, Cfg::head_dim,
                                          Cfg::heads) != rocblas_status_success) {
            SetError(error, "TRELLIS.2 DINOv3 attention values failed");
            return false;
        }
        if (!m.Linear(m.attn.f32(), T, l.out_w, l.out_b, kDim, kDim, m.linear.f32(), 0, error)) return false;
        ScaledAddKernel<<<Blocks(TD), kThreads>>>(m.hidden.f32(), m.linear.f32(), l.ls1.f32(), TD);
        LayerNormKernel<<<T, kThreads>>>(m.hidden.f32(), l.ln2_w.f32(), l.ln2_b.f32(), m.normed.f32());
        if (!m.Linear(m.normed.f32(), T, l.fc1_w, l.fc1_b, Cfg::mlp, kDim, m.mlp.f32(), 1, error) ||
            !m.Linear(m.mlp.f32(), T, l.fc2_w, l.fc2_b, kDim, Cfg::mlp, m.linear.f32(), 0, error))
            return false;
        ScaledAddKernel<<<Blocks(TD), kThreads>>>(m.hidden.f32(), m.linear.f32(), l.ls2.f32(), TD);
        if (!Check(hipGetLastError(), "encoder", error)) return false;
    }
    LayerNormKernel<<<T, kThreads>>>(m.hidden.f32(), nullptr, nullptr, m.output.f32());
    tokens->resize(TD);
    return Check(hipGetLastError(), "final norm", error) &&
           Check(hipMemcpy(tokens->data(), m.output.f32(), TD * 4, hipMemcpyDeviceToHost), "readback", error);
}

}  // namespace gufo::trellis2
