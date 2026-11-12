// TRELLIS.2: flow DiT on the qwen_image_21 HIP runtime (BF16 GEMMs via hipBLASLt, FP32
// accumulation, flash attention). Weights are mapped from one converted part file and
// uploaded lazily. Small custom kernels cover what the Qwen runtime lacks: AdaLN
// modulation with the per-block vector, gated residuals and the per-head qk gammas.
// AI-assisted (Claude Code), 2026-11-12.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "src/models/qwen_image_21/hip/runtime.hpp"
#include "src/models/qwen_image_21/weights.hpp"
#include "src/models/trellis2/dit.hpp"
#include "src/models/trellis2/scheduler.hpp"

namespace gufo::trellis2 {
namespace {

using models::qwen_image_21::SafetensorsSource;
using models::qwen_image_21::Weights;
using models::qwen_image_21::hip::Activation;
using models::qwen_image_21::hip::Matrix;
using models::qwen_image_21::hip::Norm;
using models::qwen_image_21::hip::Runtime;

__device__ __forceinline__ float bf(std::uint16_t v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }
__device__ __forceinline__ std::uint16_t tobf(float f) {
    const unsigned bits = __float_as_uint(f);
    if ((bits & 0x7F800000u) == 0x7F800000u) return static_cast<std::uint16_t>((bits >> 16) | ((bits & 0xFFFF) ? 0x40 : 0));
    return static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// out = x * (1 + scale) + shift with shift/scale = chunk `shift_chunk` / `scale_chunk` of (mod + bmod).
__global__ void ModulateKernel(const std::uint16_t* x, const std::uint16_t* mod, const std::uint16_t* bmod, std::uint16_t* out,
                               int rows, int cols, int shift_chunk, int scale_chunk) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * cols) return;
    const int c = static_cast<int>(i % cols);
    const float shift = bf(mod[shift_chunk * cols + c]) + bf(bmod[shift_chunk * cols + c]);
    const float scale = bf(mod[scale_chunk * cols + c]) + bf(bmod[scale_chunk * cols + c]);
    out[i] = tobf(bf(x[i]) * (1.f + scale) + shift);
}

// out = x + gate * h with gate = chunk `gate_chunk` of (mod + bmod).
__global__ void GateAddKernel(const std::uint16_t* x, const std::uint16_t* h, const std::uint16_t* mod, const std::uint16_t* bmod,
                              std::uint16_t* out, int rows, int cols, int gate_chunk) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * cols) return;
    const int c = static_cast<int>(i % cols);
    const float gate = bf(mod[gate_chunk * cols + c]) + bf(bmod[gate_chunk * cols + c]);
    out[i] = tobf(bf(x[i]) + gate * bf(h[i]));
}

// out = x * gamma[c]
__global__ void ScaleColumnsKernel(const std::uint16_t* x, const std::uint16_t* gamma, std::uint16_t* out, int rows, int cols) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * cols) return;
    out[i] = tobf(bf(x[i]) * bf(gamma[i % cols]));
}

unsigned Blocks(std::size_t n) { return static_cast<unsigned>((n + 255) / 256); }

void CheckLaunch(const char* what) {
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

}  // namespace

struct DitHip::Impl {
    explicit Impl(const std::filesystem::path& file)
        : weights(Weights::FromFiles(std::vector<SafetensorsSource>{{file, ""}})), rt(weights) {}
    Weights weights;
    Runtime rt;
    DitShape shape;

    Matrix Modulate(const Matrix& x, const Matrix& mod, const Matrix& bmod, int shift_chunk, int scale_chunk) {
        Matrix out = rt.New(x.rows, x.cols);
        ModulateKernel<<<Blocks(x.size()), 256, 0, rt.stream()>>>(x.data, mod.data, bmod.data, out.data, x.rows, x.cols, shift_chunk, scale_chunk);
        CheckLaunch("modulate");
        return out;
    }
    Matrix GateAdd(const Matrix& x, const Matrix& h, const Matrix& mod, const Matrix& bmod, int gate_chunk) {
        Matrix out = rt.New(x.rows, x.cols);
        GateAddKernel<<<Blocks(x.size()), 256, 0, rt.stream()>>>(x.data, h.data, mod.data, bmod.data, out.data, x.rows, x.cols, gate_chunk);
        CheckLaunch("gate add");
        return out;
    }
    // F.normalize per head * gamma * sqrt(d): RMS norm (no weight) on [rows*heads, d], then column gamma [heads*d].
    Matrix QkNorm(const Matrix& x, const std::string& gamma_name) {
        const int H = shape.heads, D = shape.head_dim();
        Matrix normed = rt.Normalize(x.Reshape(x.rows * H, D), Norm::kRms, "", 1e-12f).Reshape(x.rows, x.cols);
        Matrix gamma = rt.Weight(gamma_name);
        if (gamma.size() != static_cast<std::size_t>(x.cols)) throw std::logic_error("qk gamma size");
        Matrix out = rt.New(x.rows, x.cols);
        ScaleColumnsKernel<<<Blocks(x.size()), 256, 0, rt.stream()>>>(normed.data, gamma.data, out.data, x.rows, x.cols);
        CheckLaunch("qk gamma");
        return out;
    }
};

DitHip::DitHip(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DitHip::~DitHip() = default;

std::size_t DitHip::weight_bytes() const { return impl_->rt.WeightBytes(); }

std::unique_ptr<DitHip> DitHip::Create(const std::filesystem::path& file, const DitShape& shape, std::string* error) {
    try {
        auto impl = std::make_unique<Impl>(file);
        impl->shape = shape;
        if (!impl->weights.Contains("input_layer.weight") || !impl->weights.Contains("blocks.29.modulation")) {
            if (error) *error = "not a converted TRELLIS.2 flow file: " + file.string();
            return nullptr;
        }
        const auto& in = impl->weights.Get("input_layer.weight").shape;
        const auto& out = impl->weights.Get("out_layer.weight").shape;
        impl->shape.in_channels = in[1];
        impl->shape.out_channels = out[0];
        impl->rt.SetCacheLimit(std::size_t{1} << 30);   // several runtimes share the GPU (pipeline)
        // Upload everything now through the bounded pinned path: touching the mapped file lazily
        // would keep ~2.4 GB of checkpoint pages in the process working set (host RAM is tight).
        if (!impl->rt.UploadMissing("")) {
            if (error) *error = "TRELLIS.2 DiT: weight upload failed for " + file.string();
            return nullptr;
        }
        return std::unique_ptr<DitHip>(new DitHip(std::move(impl)));
    } catch (const std::exception& e) {
        if (error) *error = std::string("TRELLIS.2 DiT: ") + e.what();
        return nullptr;
    }
}

bool DitHip::Forward(const float* x, const std::vector<Coord>& coords, const float* cond, int cond_tokens, float t,
                     std::vector<float>* v, std::map<int, std::vector<float>>* taps, std::string* error,
                     const DitForwardOptions& options) {
    Impl& m = *impl_;
    Runtime& rt = m.rt;
    const DitShape& s = m.shape;
    const int N = static_cast<int>(coords.size()), W = s.width, H = s.heads, D = s.head_dim();
    try {
        const std::vector<float> sinus = timestep_embedding(t, s.time_freq);
        Matrix temb = rt.Activate(rt.Linear(rt.Activate(rt.Linear(rt.Upload(sinus, 1, s.time_freq), "t_embedder.mlp.0"), Activation::kSilu),
                                            "t_embedder.mlp.2"), Activation::kSilu);
        Matrix mod = rt.Linear(temb, "adaLN_modulation.1");   // [1, 6W]
        Matrix hidden = rt.Linear(rt.Upload(std::span<const float>(x, static_cast<std::size_t>(N) * s.in_channels), N, s.in_channels), "input_layer");
        Matrix image;
        std::vector<float> zeros;
        if (cond == nullptr) {
            zeros.assign(static_cast<std::size_t>(cond_tokens) * s.cond_dim, 0.f);
            cond = zeros.data();
        }
        image = rt.Upload(std::span<const float>(cond, static_cast<std::size_t>(cond_tokens) * s.cond_dim), cond_tokens, s.cond_dim);
        // RoPE table: (cos, sin) per pair, [N, D] floats.
        std::vector<float> freqs(static_cast<std::size_t>(N) * D);
        std::vector<float> ang(static_cast<std::size_t>(D / 2));
        for (int i = 0; i < N; ++i) {
            rope_angles(coords[static_cast<std::size_t>(i)].x, coords[static_cast<std::size_t>(i)].y, coords[static_cast<std::size_t>(i)].z, ang.data(), D);
            for (int p = 0; p < D / 2; ++p) {
                freqs[static_cast<std::size_t>(i) * D + 2 * p] = std::cos(ang[static_cast<std::size_t>(p)]);
                freqs[static_cast<std::size_t>(i) * D + 2 * p + 1] = std::sin(ang[static_cast<std::size_t>(p)]);
            }
        }
        Matrix rope = rt.UploadFloat(freqs, N, D);

        const int layers = options.layer_limit > 0 ? std::min(options.layer_limit, s.layers) : s.layers;
        for (int layer = 0; layer < layers; ++layer) {
            const std::string p = "blocks." + std::to_string(layer) + ".";
            Matrix bmod = rt.Weight(p + "modulation");
            // self-attention
            Matrix normed = m.Modulate(rt.Normalize(hidden, Norm::kLayer, "", DitConfig::block_norm_eps), mod, bmod, 0, 1);
            Matrix qkv = rt.Linear(normed, p + "self_attn.to_qkv");
            Matrix q = rt.Rope(m.QkNorm(rt.Columns(qkv, 0, W), p + "self_attn.q_rms_norm.gamma"), H, D, rope, 0);
            Matrix k = rt.Rope(m.QkNorm(rt.Columns(qkv, W, W), p + "self_attn.k_rms_norm.gamma"), H, D, rope, 0);
            Matrix val = rt.Columns(qkv, 2 * W, W);
            hidden = m.GateAdd(hidden, rt.Linear(rt.Attention(q, k, val, H, H, D), p + "self_attn.to_out"), mod, bmod, 2);
            // cross-attention
            normed = rt.Normalize(hidden, Norm::kLayer, p + "norm2", DitConfig::block_norm_eps);
            q = m.QkNorm(rt.Linear(normed, p + "cross_attn.to_q"), p + "cross_attn.q_rms_norm.gamma");
            Matrix kv = rt.Linear(image, p + "cross_attn.to_kv");
            k = m.QkNorm(rt.Columns(kv, 0, W), p + "cross_attn.k_rms_norm.gamma");
            val = rt.Columns(kv, W, W);
            hidden = rt.Add(hidden, rt.Linear(rt.Attention(q, k, val, H, H, D), p + "cross_attn.to_out"));
            // mlp
            normed = m.Modulate(rt.Normalize(hidden, Norm::kLayer, "", DitConfig::block_norm_eps), mod, bmod, 3, 4);
            hidden = m.GateAdd(hidden, rt.Linear(rt.Activate(rt.Linear(normed, p + "mlp.mlp.0"), Activation::kGeluTanh), p + "mlp.mlp.2"), mod, bmod, 5);
            if (taps && std::find(options.tap_layers.begin(), options.tap_layers.end(), layer) != options.tap_layers.end())
                (*taps)[layer] = rt.Download(hidden);
        }
        Matrix out = rt.Linear(rt.Normalize(hidden, Norm::kLayer, "", DitConfig::final_norm_eps), "out_layer");
        *v = rt.Download(out);
        rt.Synchronize();
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("TRELLIS.2 DiT forward: ") + e.what();
        return false;
    }
}

}  // namespace gufo::trellis2
