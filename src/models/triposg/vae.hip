// TripoSG: VAE decoder on the qwen_image_21 HIP runtime (BF16).
// AI-assisted (Claude Code), 2026-11-12.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "src/models/qwen_image_21/hip/runtime.hpp"
#include "src/models/qwen_image_21/weights.hpp"
#include "src/models/triposg/vae.hpp"

namespace gufo::triposg {
namespace {

using models::qwen_image_21::SafetensorsSource;
using models::qwen_image_21::Weights;
using models::qwen_image_21::hip::Activation;
using models::qwen_image_21::hip::Matrix;
using models::qwen_image_21::hip::Norm;
using models::qwen_image_21::hip::Runtime;

}  // namespace

struct VaeHip::Impl {
    explicit Impl(const std::filesystem::path& file)
        : weights(Weights::FromFiles(std::vector<SafetensorsSource>{{file, ""}})), rt(weights) {}
    Weights weights;
    Runtime rt;
    VaeShape shape;
    Matrix kv, keys, values;  // [tokens, width]
    int tokens{0};
    std::string query_prefix() const { return "vae.decoder.blocks." + std::to_string(shape.layers) + "."; }
};

VaeHip::VaeHip(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VaeHip::~VaeHip() = default;

std::unique_ptr<VaeHip> VaeHip::Create(const std::filesystem::path& file, std::string* error) {
    try {
        auto impl = std::make_unique<Impl>(file);
        if (!impl->weights.Contains("vae.post_quant.weight") || !impl->weights.Contains("vae.decoder.proj_out.weight")) {
            if (error) *error = "not a converted TripoSG file (vae.* tensors missing): " + file.string();
            return nullptr;
        }
        return std::unique_ptr<VaeHip>(new VaeHip(std::move(impl)));
    } catch (const std::exception& e) {
        if (error) *error = std::string("TripoSG VAE: ") + e.what();
        return nullptr;
    }
}

bool VaeHip::Prepare(const float* z, int tokens, std::string* error) {
    Impl& m = *impl_;
    Runtime& rt = m.rt;
    const VaeShape& s = m.shape;
    const int W = s.width, H = s.heads, D = s.head_dim();
    try {
        Matrix hidden = rt.Linear(rt.Upload(std::span<const float>(z, static_cast<std::size_t>(tokens) * s.latent_channels), tokens, s.latent_channels),
                                  "vae.post_quant");
        for (int layer = 0; layer < s.layers; ++layer) {
            const std::string p = "vae.decoder.blocks." + std::to_string(layer) + ".";
            Matrix normed = rt.Normalize(hidden, Norm::kLayer, p + "norm1", VaeConfig::norm_eps);
            Matrix q = rt.Linear(normed, p + "attn1.to_q");
            Matrix k = rt.Linear(normed, p + "attn1.to_k");
            Matrix v = rt.Linear(normed, p + "attn1.to_v");
            hidden = rt.Add(hidden, rt.Linear(rt.Attention(q, k, v, H, H, D), p + "attn1.to_out.0"));
            normed = rt.Normalize(hidden, Norm::kLayer, p + "norm3", VaeConfig::norm_eps);
            hidden = rt.Add(hidden, rt.Linear(rt.Activate(rt.Linear(normed, p + "ff.net.0.proj"), Activation::kGelu), p + "ff.net.2"));
        }
        m.kv = hidden;
        m.tokens = tokens;
        const std::string q = m.query_prefix();
        Matrix kvn = rt.Normalize(hidden, Norm::kLayer, q + "attn2.norm_cross", 1e-5f);
        m.keys = rt.Linear(kvn, q + "attn2.to_k");
        m.values = rt.Linear(kvn, q + "attn2.to_v");
        (void)W;
        rt.Synchronize();
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("TripoSG VAE prepare: ") + e.what();
        return false;
    }
}

std::vector<float> VaeHip::DownloadKv() { return impl_->rt.Download(impl_->kv); }

bool VaeHip::Query(const float* points, int count, std::vector<float>* logits, std::string* error, int chunk) {
    Impl& m = *impl_;
    Runtime& rt = m.rt;
    const VaeShape& s = m.shape;
    const int H = s.heads, D = s.head_dim();
    if (m.tokens == 0) {
        if (error) *error = "TripoSG VAE: Prepare() first";
        return false;
    }
    try {
        logits->resize(static_cast<std::size_t>(count));
        const std::string q = m.query_prefix();
        std::vector<float> embed;
        for (int start = 0; start < count; start += chunk) {
            const int n = std::min(chunk, count - start);
            vae_embed_points(points + static_cast<std::size_t>(start) * 3, n, s.freqs, &embed);
            Matrix hidden = rt.Linear(rt.Upload(embed, n, s.query_channels()), "vae.decoder.proj_query");
            Matrix normed = rt.Normalize(hidden, Norm::kLayer, q + "norm2", VaeConfig::norm_eps);
            Matrix qq = rt.Linear(normed, q + "attn2.to_q");
            hidden = rt.Add(hidden, rt.Linear(rt.Attention(qq, m.keys, m.values, H, H, D), q + "attn2.to_out.0"));
            normed = rt.Normalize(hidden, Norm::kLayer, q + "norm3", VaeConfig::norm_eps);
            hidden = rt.Add(hidden, rt.Linear(rt.Activate(rt.Linear(normed, q + "ff.net.0.proj"), Activation::kGelu), q + "ff.net.2"));
            Matrix out = rt.Linear(rt.Normalize(hidden, Norm::kLayer, "vae.decoder.norm_out", 1e-5f), "vae.decoder.proj_out");
            const std::vector<float> values = rt.Download(out);
            for (int i = 0; i < n; ++i) (*logits)[static_cast<std::size_t>(start + i)] = -values[static_cast<std::size_t>(i)];
            rt.Trim();
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("TripoSG VAE query: ") + e.what();
        return false;
    }
}

}  // namespace gufo::triposg
