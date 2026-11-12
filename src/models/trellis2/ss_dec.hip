// TRELLIS.2: sparse-structure decoder on HIP (FP32 rocBLAS; voxel-major [V, C] buffers):
// dense im2col kernel (zero padding) + sgemm per 3^3 convolution, channel LayerNorm + SiLU
// kernel, pixel-shuffle kernel, residual add. Same maths as ss_dec_cpu.cpp.
// AI-assisted (Claude Code), 2026-11-12.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstddef>
#include <utility>

#include "src/models/trellis2/ss_dec.hpp"

namespace gufo::trellis2 {
namespace {

constexpr int kThreads = 256;

void SetError(std::string* error, std::string message) {
    if (error != nullptr) *error = std::move(message);
}
bool Check(hipError_t status, const char* operation, std::string* error) {
    if (status == hipSuccess) return true;
    SetError(error, std::string("TRELLIS.2 ss_dec ") + operation + ": " + hipGetErrorString(status));
    return false;
}
unsigned Blocks(std::size_t count) { return static_cast<unsigned>((count + kThreads - 1) / kThreads); }

// col[v, t*Ci + c] = in[neighbour(v, t), c] or 0 outside the grid.
__global__ void Im2colKernel(const float* in, float* col, int res, int ci) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(res) * res * res * 27 * ci;
    if (i >= total) return;
    const int c = static_cast<int>(i % ci);
    const int t = static_cast<int>((i / ci) % 27);
    const int v = static_cast<int>(i / (static_cast<std::size_t>(ci) * 27));
    const int x = v / (res * res), y = (v / res) % res, z = v % res;
    const int nx = x + t / 9 - 1, ny = y + (t / 3) % 3 - 1, nz = z + t % 3 - 1;
    float value = 0.f;
    if (nx >= 0 && ny >= 0 && nz >= 0 && nx < res && ny < res && nz < res)
        value = in[(static_cast<std::size_t>(nx) * res * res + static_cast<std::size_t>(ny) * res + nz) * ci + c];
    col[i] = value;
}

__global__ void BiasAddKernel(float* x, const float* bias, const float* residual, std::size_t count, int columns) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i] + bias[i % columns];
    if (residual) v += residual[i];
    x[i] = v;
}

// One block per voxel row: LayerNorm over C with affine, then SiLU.
__global__ void ChannelNormSiluKernel(const float* x, const float* w, const float* b, float* out, int channels, float eps) {
    __shared__ float shared[kThreads];
    const float* row = x + static_cast<std::size_t>(blockIdx.x) * channels;
    float* target = out + static_cast<std::size_t>(blockIdx.x) * channels;
    float sum = 0;
    for (int c = threadIdx.x; c < channels; c += kThreads) sum += row[c];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = shared[0] / channels;
    __syncthreads();
    float sq = 0;
    for (int c = threadIdx.x; c < channels; c += kThreads) { const float d = row[c] - mean; sq += d * d; }
    shared[threadIdx.x] = sq;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(shared[0] / channels + eps);
    for (int c = threadIdx.x; c < channels; c += kThreads) {
        const float y = (row[c] - mean) * rstd * w[c] + b[c];
        target[c] = y / (1.f + expf(-y));
    }
}

// [R^3, 8C] -> [(2R)^3, C]: child s = sx*4 + sy*2 + sz of voxel (x,y,z) takes channels c*8 + s.
__global__ void PixelShuffleKernel(const float* in, float* out, int res, int channels) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int R2 = 2 * res;
    const std::size_t total = static_cast<std::size_t>(R2) * R2 * R2 * channels;
    if (i >= total) return;
    const int c = static_cast<int>(i % channels);
    const std::size_t v = i / channels;
    const int X = static_cast<int>(v / (static_cast<std::size_t>(R2) * R2)), Y = static_cast<int>((v / R2) % R2), Z = static_cast<int>(v % R2);
    const int s = ((X & 1) << 2) | ((Y & 1) << 1) | (Z & 1);
    const std::size_t src = ((static_cast<std::size_t>(X / 2) * res + (Y / 2)) * res + (Z / 2)) * 8 * channels + static_cast<std::size_t>(c) * 8 + s;
    out[i] = in[src];
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : data_(std::exchange(o.data_, nullptr)), bytes_(o.bytes_) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { Release(); data_ = std::exchange(o.data_, nullptr); bytes_ = o.bytes_; }
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

struct ConvDev { DeviceBuffer w, b; int co = 0, ci = 0; };
struct NormDev { DeviceBuffer w, b; };
struct ResDev { NormDev n1, n2; ConvDev c1, c2; };

}  // namespace

struct SsDecHip::Impl {
    ~Impl() { if (blas) (void)rocblas_destroy_handle(blas); }
    SsDecShape shape;
    ConvDev input, up0, up1, out;
    std::vector<ResDev> middle, level0, level1, level2;
    NormDev out_norm;
    DeviceBuffer col, a, b, h, h2;
    rocblas_handle blas{nullptr};
    std::size_t device_bytes{0};

    bool Upload(ConvDev& d, const SsDecConv& c, std::string* error) {
        d.co = c.co; d.ci = c.ci;
        if (!d.w.Upload(c.w, error) || !d.b.Upload(c.b, error)) return false;
        device_bytes += d.w.bytes() + d.b.bytes();
        return true;
    }
    bool Upload(NormDev& d, const SsDecNorm& n, std::string* error) {
        if (!d.w.Upload(n.w, error) || !d.b.Upload(n.b, error)) return false;
        device_bytes += d.w.bytes() + d.b.bytes();
        return true;
    }
    bool Upload(ResDev& d, const SsDecResBlock& r, std::string* error) {
        return Upload(d.n1, r.norm1, error) && Upload(d.n2, r.norm2, error) && Upload(d.c1, r.conv1, error) && Upload(d.c2, r.conv2, error);
    }
    // out[V, Co] = im2col(in)[V, 27Ci] . W[Co, 27Ci]^T + bias (+ residual)
    bool Conv(const float* in, int res, const ConvDev& c, float* out, const float* residual, std::string* error) {
        const int V = res * res * res;
        const std::size_t k = static_cast<std::size_t>(27) * c.ci;
        Im2colKernel<<<Blocks(static_cast<std::size_t>(V) * k), kThreads>>>(in, col.f32(), res, c.ci);
        if (!Check(hipGetLastError(), "im2col", error)) return false;
        const float one = 1.f, zero = 0.f;
        if (rocblas_sgemm(blas, rocblas_operation_transpose, rocblas_operation_none, c.co, V, static_cast<int>(k), &one,
                          c.w.f32(), static_cast<int>(k), col.f32(), static_cast<int>(k), &zero, out, c.co) != rocblas_status_success) {
            SetError(error, "TRELLIS.2 ss_dec GEMM failed");
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(V) * c.co;
        BiasAddKernel<<<Blocks(count), kThreads>>>(out, c.b.f32(), residual, count, c.co);
        return Check(hipGetLastError(), "bias", error);
    }
    bool Norm(const float* x, int V, int C, const NormDev& n, float* out, std::string* error) {
        ChannelNormSiluKernel<<<V, kThreads>>>(x, n.w.f32(), n.b.f32(), out, C, SsDecConfig::norm_eps);
        return Check(hipGetLastError(), "norm", error);
    }
    // a = res(a): h = norm1(a) -> conv1 -> h2 ; norm2(h2) -> conv2 + a -> b ; swap
    bool Res(const ResDev& r, int res, int C, std::string* error) {
        const int V = res * res * res;
        return Norm(a.f32(), V, C, r.n1, h.f32(), error) && Conv(h.f32(), res, r.c1, h2.f32(), nullptr, error) &&
               Norm(h2.f32(), V, C, r.n2, h.f32(), error) && Conv(h.f32(), res, r.c2, b.f32(), a.f32(), error) && (std::swap(a, b), true);
    }
};

SsDecHip::SsDecHip(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SsDecHip::~SsDecHip() = default;
std::size_t SsDecHip::device_bytes() const { return impl_->device_bytes; }

std::unique_ptr<SsDecHip> SsDecHip::Create(const SsDecWeights& w, std::string* error) {
    auto impl = std::make_unique<Impl>();
    Impl& m = *impl;
    m.shape = w.shape;
    if (rocblas_create_handle(&m.blas) != rocblas_status_success) {
        SetError(error, "TRELLIS.2 ss_dec: rocBLAS handle");
        return nullptr;
    }
    if (!m.Upload(m.input, w.input, error) || !m.Upload(m.up0, w.up0, error) || !m.Upload(m.up1, w.up1, error) ||
        !m.Upload(m.out, w.out, error) || !m.Upload(m.out_norm, w.out_norm, error))
        return nullptr;
    auto up_list = [&](std::vector<ResDev>& dst, const std::vector<SsDecResBlock>& src) {
        dst.resize(src.size());
        for (std::size_t i = 0; i < src.size(); ++i)
            if (!m.Upload(dst[i], src[i], error)) return false;
        return true;
    };
    if (!up_list(m.middle, w.middle) || !up_list(m.level0, w.level0) || !up_list(m.level1, w.level1) || !up_list(m.level2, w.level2))
        return nullptr;
    const auto& ch = w.shape.channels;
    const int R0 = w.shape.latent_res, R1 = 2 * R0, R2 = 4 * R0;
    const std::size_t V0 = static_cast<std::size_t>(R0) * R0 * R0, V1 = V0 * 8, V2 = V1 * 8;
    // Largest buffers: im2col at level 2 (V2 x 27 x ch2), activations V1 x 8 ch1 / V2 x ch2 ...
    const std::size_t col_max = std::max({V0 * 27 * static_cast<std::size_t>(ch[0]), V1 * 27 * static_cast<std::size_t>(ch[1]), V2 * 27 * static_cast<std::size_t>(ch[2]),
                                          V0 * 27 * static_cast<std::size_t>(w.shape.latent_channels)});
    const std::size_t act_max = std::max({V0 * static_cast<std::size_t>(ch[0]), V0 * 8 * static_cast<std::size_t>(ch[1]), V1 * static_cast<std::size_t>(ch[1]),
                                          V1 * 8 * static_cast<std::size_t>(ch[2]), V2 * static_cast<std::size_t>(ch[2])});
    if (!m.col.Allocate(col_max * 4, error) || !m.a.Allocate(act_max * 4, error) || !m.b.Allocate(act_max * 4, error) ||
        !m.h.Allocate(act_max * 4, error) || !m.h2.Allocate(act_max * 4, error))
        return nullptr;
    return std::unique_ptr<SsDecHip>(new SsDecHip(std::move(impl)));
}

bool SsDecHip::Forward(const float* latent, std::vector<float>* logits, std::string* error) {
    Impl& m = *impl_;
    const auto& ch = m.shape.channels;
    int res = m.shape.latent_res;
    std::size_t V = static_cast<std::size_t>(res) * res * res;
    if (!Check(hipMemcpy(m.h.f32(), latent, V * m.shape.latent_channels * 4, hipMemcpyHostToDevice), "latent upload", error)) return false;
    if (!m.Conv(m.h.f32(), res, m.input, m.a.f32(), nullptr, error)) return false;
    for (const auto& r : m.middle) if (!m.Res(r, res, ch[0], error)) return false;
    for (const auto& r : m.level0) if (!m.Res(r, res, ch[0], error)) return false;
    if (!m.Conv(m.a.f32(), res, m.up0, m.b.f32(), nullptr, error)) return false;
    PixelShuffleKernel<<<Blocks(V * 8 * ch[1]), kThreads>>>(m.b.f32(), m.a.f32(), res, ch[1]);
    if (!Check(hipGetLastError(), "shuffle 0", error)) return false;
    res *= 2; V *= 8;
    for (const auto& r : m.level1) if (!m.Res(r, res, ch[1], error)) return false;
    if (!m.Conv(m.a.f32(), res, m.up1, m.b.f32(), nullptr, error)) return false;
    PixelShuffleKernel<<<Blocks(V * 8 * ch[2]), kThreads>>>(m.b.f32(), m.a.f32(), res, ch[2]);
    if (!Check(hipGetLastError(), "shuffle 1", error)) return false;
    res *= 2; V *= 8;
    for (const auto& r : m.level2) if (!m.Res(r, res, ch[2], error)) return false;
    if (!m.Norm(m.a.f32(), static_cast<int>(V), ch[2], m.out_norm, m.h.f32(), error)) return false;
    if (!m.Conv(m.h.f32(), res, m.out, m.b.f32(), nullptr, error)) return false;
    logits->resize(V);
    return Check(hipDeviceSynchronize(), "sync", error) &&
           Check(hipMemcpy(logits->data(), m.b.f32(), V * 4, hipMemcpyDeviceToHost), "readback", error);
}

}  // namespace gufo::trellis2
