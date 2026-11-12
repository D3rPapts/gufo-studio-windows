// TripoSG: DiT on the qwen_image_21 HIP runtime (BF16 GEMMs, FP32 accumulation).
// Weights are mapped from the converted triposg.safetensors and uploaded lazily.
// AI-assisted (Claude Code), 2026-11-12.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "src/models/qwen_image_21/hip/runtime.hpp"
#include "src/models/qwen_image_21/weights.hpp"
#include "src/models/triposg/dit.hpp"
#include "src/models/triposg/scheduler.hpp"

namespace gufo::triposg {
namespace {

using models::qwen_image_21::SafetensorsSource;
using models::qwen_image_21::Weights;
using models::qwen_image_21::hip::Activation;
using models::qwen_image_21::hip::Matrix;
using models::qwen_image_21::hip::Norm;
using models::qwen_image_21::hip::Runtime;

// out[r, 0..ca) = a[r, :], out[r, ca..ca+cb) = b[r, :]  (BF16 bit copies).
__global__ void ConcatColumnsKernel(const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* out, int rows,
                                    int ca, int cb) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ca + cb);
    if (i >= total) return;
    const int r = static_cast<int>(i / static_cast<std::size_t>(ca + cb));
    const int c = static_cast<int>(i % static_cast<std::size_t>(ca + cb));
    out[i] = c < ca ? a[static_cast<std::size_t>(r) * ca + c] : b[static_cast<std::size_t>(r) * cb + (c - ca)];
}

Matrix concat_columns(Runtime& rt, const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows) throw std::logic_error("concat_columns rows");
    Matrix out = rt.New(a.rows, a.cols + b.cols);
    const std::size_t total = out.size();
    ConcatColumnsKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, rt.stream()>>>(a.data, b.data, out.data,
                                                                                            a.rows, a.cols, b.cols);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("concat_columns: ") + hipGetErrorString(status));
    return out;
}

}  // namespace

struct DitHip::Impl {
    explicit Impl(const std::filesystem::path& file)
        : weights(Weights::FromFiles(std::vector<SafetensorsSource>{{file, ""}})), rt(weights) {}
    Weights weights;
    Runtime rt;
    DitShape shape;
};

DitHip::DitHip(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DitHip::~DitHip() = default;

std::size_t DitHip::weight_bytes() const { return impl_->rt.WeightBytes(); }

std::unique_ptr<DitHip> DitHip::Create(const std::filesystem::path& file, std::string* error) {
    try {
        auto impl = std::make_unique<Impl>(file);
        if (!impl->weights.Contains("dit.proj_in.weight") || !impl->weights.Contains("dit.blocks.20.skip_norm.weight")) {
            if (error) *error = "not a converted TripoSG file (dit.* tensors missing): " + file.string();
            return nullptr;
        }
        return std::unique_ptr<DitHip>(new DitHip(std::move(impl)));
    } catch (const std::exception& e) {
        if (error) *error = std::string("TripoSG DiT: ") + e.what();
        return nullptr;
    }
}

bool DitHip::Forward(const float* x, int tokens, const float* cond, int cond_tokens, float t, std::vector<float>* v,
                     std::map<int, std::vector<float>>* taps, std::string* error, const DitForwardOptions& options) {
    Impl& m = *impl_;
    Runtime& rt = m.rt;
    const DitShape& s = m.shape;
    const int W = s.width, H = s.heads, D = s.head_dim();
    const int N = tokens + 1;
    try {
        // Time token.
        const std::vector<float> sinus = timestep_embedding(t, W);
        Matrix temb = rt.Linear(rt.Activate(rt.Linear(rt.Upload(sinus, 1, W), "dit.time_proj.linear_1"), Activation::kGelu),
                                "dit.time_proj.linear_2");
        Matrix latents = rt.Linear(rt.Upload(std::span<const float>(x, static_cast<std::size_t>(tokens) * s.in_channels), tokens, s.in_channels),
                                   "dit.proj_in");
        Matrix hidden = rt.New(N, W);
        rt.CopyRows(hidden, 0, temb);
        rt.CopyRows(hidden, 1, latents);
        Matrix image = rt.Upload(std::span<const float>(cond, static_cast<std::size_t>(cond_tokens) * s.cross_dim), cond_tokens, s.cross_dim);

        std::vector<Matrix> skips;
        for (int layer = 0; layer < s.layers; ++layer) {
            const std::string p = "dit.blocks." + std::to_string(layer) + ".";
            if (s.has_skip(layer)) {
                if (skips.empty()) throw std::logic_error("skip stack underflow");
                Matrix skip = skips.back();
                skips.pop_back();
                hidden = rt.Normalize(rt.Linear(concat_columns(rt, skip, hidden), p + "skip_linear"), Norm::kLayer,
                                      p + "skip_norm", DitConfig::norm_eps);
            }
            // Self-attention with per-head RMS norm on q and k.
            Matrix normed = rt.Normalize(hidden, Norm::kLayer, p + "norm1", DitConfig::norm_eps);
            Matrix q = rt.Normalize(rt.Linear(normed, p + "attn1.to_q").Reshape(N * H, D), Norm::kRms, p + "attn1.norm_q",
                                    DitConfig::qk_norm_eps).Reshape(N, W);
            Matrix k = rt.Normalize(rt.Linear(normed, p + "attn1.to_k").Reshape(N * H, D), Norm::kRms, p + "attn1.norm_k",
                                    DitConfig::qk_norm_eps).Reshape(N, W);
            Matrix val = rt.Linear(normed, p + "attn1.to_v");
            hidden = rt.Add(hidden, rt.Linear(rt.Attention(q, k, val, H, H, D), p + "attn1.to_out.0"));
            // Cross-attention on the image tokens.
            normed = rt.Normalize(hidden, Norm::kLayer, p + "norm2", DitConfig::norm_eps);
            q = rt.Normalize(rt.Linear(normed, p + "attn2.to_q").Reshape(N * H, D), Norm::kRms, p + "attn2.norm_q",
                             DitConfig::qk_norm_eps).Reshape(N, W);
            k = rt.Normalize(rt.Linear(image, p + "attn2.to_k").Reshape(cond_tokens * H, D), Norm::kRms, p + "attn2.norm_k",
                             DitConfig::qk_norm_eps).Reshape(cond_tokens, W);
            val = rt.Linear(image, p + "attn2.to_v");
            hidden = rt.Add(hidden, rt.Linear(rt.Attention(q, k, val, H, H, D), p + "attn2.to_out.0"));
            // FFN.
            normed = rt.Normalize(hidden, Norm::kLayer, p + "norm3", DitConfig::norm_eps);
            hidden = rt.Add(hidden, rt.Linear(rt.Activate(rt.Linear(normed, p + "ff.net.0.proj"), Activation::kGelu),
                                              p + "ff.net.2"));
            if (layer <= s.skip_last_push()) skips.push_back(hidden);
            if (taps && std::find(options.tap_layers.begin(), options.tap_layers.end(), layer) != options.tap_layers.end())
                (*taps)[layer] = rt.Download(hidden);
        }
        Matrix out = rt.Linear(rt.Normalize(hidden, Norm::kLayer, "dit.norm_out", DitConfig::out_norm_eps).Slice(1, tokens),
                               "dit.proj_out");
        *v = rt.Download(out);
        rt.Synchronize();
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("TripoSG DiT forward: ") + e.what();
        return false;
    }
}

}  // namespace gufo::triposg
