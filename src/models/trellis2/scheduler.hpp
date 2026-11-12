// TRELLIS.2: flow-matching Euler sampler with rescaled time, classifier-free
// guidance with interval and x0 rescale (FlowEulerGuidanceIntervalSampler), the
// timestep sinusoid of TimestepEmbedder and the 3-D RoPE angle table of
// RotaryPositionEmbedder. Host FP32, written from the formulas in
// trellis2/pipelines/samplers and trellis2/modules/attention/rope.py.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "src/models/trellis2/config.hpp"

namespace gufo::trellis2 {

// t_i = r * s_i / (1 + (r - 1) * s_i), s_i = 1 - i / steps, i = 0..steps (t_0 = 1, t_steps = 0).
inline std::vector<float> flow_times(int steps, float rescale_t) {
    std::vector<float> t;
    t.reserve(static_cast<std::size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const double s = 1.0 - static_cast<double>(i) / steps;
        t.push_back(static_cast<float>(rescale_t * s / (1.0 + (rescale_t - 1.0) * s)));
    }
    return t;
}

// The DiT receives 1000 * t.
inline float model_timestep(float t) { return 1000.f * t; }

// x_{t_prev} = x_t - (t - t_prev) * v
inline void euler_step(std::vector<float>& x, const std::vector<float>& v, float t, float t_prev) {
    const float dt = t - t_prev;
    for (std::size_t k = 0; k < x.size(); ++k) x[k] -= dt * v[k];
}

inline bool guidance_active(float t, const SamplerParams& p) {
    return p.guidance != 1.f && p.interval_lo <= t && t <= p.interval_hi;
}

// x0 from a velocity prediction: (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
inline float x0_coefficient_v(float t, float sigma_min) { return sigma_min + (1.f - sigma_min) * t; }

// pred = g * pos + (1 - g) * neg, then the CFG rescale on x0 (unbiased std over all
// elements, batch 1). Result in *pos.
inline void apply_guidance(std::vector<float>* pos, const std::vector<float>& neg, const std::vector<float>& x_t,
                           float t, const SamplerParams& p) {
    const float g = p.guidance;
    std::vector<float>& out = *pos;
    if (p.rescale <= 0.f) {
        for (std::size_t k = 0; k < out.size(); ++k) out[k] = g * out[k] + (1.f - g) * neg[k];
        return;
    }
    const double a = 1.0 - p.sigma_min, b = x0_coefficient_v(t, p.sigma_min);
    const std::size_t n = out.size();
    std::vector<float> x0_pos(n), x0_cfg(n);
    double mean_pos = 0, mean_cfg = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const float pred = g * out[k] + (1.f - g) * neg[k];
        x0_pos[k] = static_cast<float>(a * x_t[k] - b * out[k]);
        x0_cfg[k] = static_cast<float>(a * x_t[k] - b * pred);
        mean_pos += x0_pos[k];
        mean_cfg += x0_cfg[k];
    }
    mean_pos /= static_cast<double>(n);
    mean_cfg /= static_cast<double>(n);
    double var_pos = 0, var_cfg = 0;
    for (std::size_t k = 0; k < n; ++k) {
        var_pos += (x0_pos[k] - mean_pos) * (x0_pos[k] - mean_pos);
        var_cfg += (x0_cfg[k] - mean_cfg) * (x0_cfg[k] - mean_cfg);
    }
    const double denom = n > 1 ? static_cast<double>(n - 1) : 1.0;   // torch.std: unbiased
    const double ratio = std::sqrt(var_pos / denom) / std::sqrt(var_cfg / denom);
    for (std::size_t k = 0; k < n; ++k) {
        const double x0 = p.rescale * (x0_cfg[k] * ratio) + (1.0 - p.rescale) * x0_cfg[k];
        out[k] = static_cast<float>((a * x_t[k] - x0) / b);
    }
}

// TimestepEmbedder.timestep_embedding(t, 256): [cos(t f_j) | sin(t f_j)], f_j = exp(-ln(1e4) j / 128).
inline std::vector<float> timestep_embedding(float t, int dim = DitConfig::time_freq) {
    const int half = dim / 2;
    std::vector<float> out(static_cast<std::size_t>(dim));
    for (int j = 0; j < half; ++j) {
        const float freq = std::exp(-std::log(10000.f) * static_cast<float>(j) / static_cast<float>(half));
        const float arg = t * freq;
        out[static_cast<std::size_t>(j)] = std::cos(arg);
        out[static_cast<std::size_t>(half + j)] = std::sin(arg);
    }
    return out;
}

// RoPE angles for one token at integer coordinates (x, y, z): 64 pairs, pair i rotates
// (q[2i], q[2i+1]) by angle[i]; i < 63: axis i / 21, j = i % 21, angle = c_axis / 10000^(j/21);
// i = 63: angle 0 (padding).
// head_dim may differ from 128 in tests: per_axis = head_dim / 2 / 3, pairs = head_dim / 2.
inline void rope_angles(int x, int y, int z, float* angles, int head_dim = DitConfig::head_dim) {
    const int c[3] = {x, y, z};
    const int per_axis = head_dim / 2 / 3, pairs = head_dim / 2;
    for (int i = 0; i < pairs; ++i) {
        if (i >= 3 * per_axis) { angles[i] = 0.f; continue; }
        const int axis = i / per_axis, j = i % per_axis;
        const float freq = 1.f / std::pow(DitConfig::rope_base, static_cast<float>(j) / static_cast<float>(per_axis));
        angles[i] = static_cast<float>(c[axis]) * freq;
    }
}

// In-place rotation of one head vector [head_dim] by the head_dim/2 angles (complex multiply per pair).
inline void apply_rope(float* v, const float* angles, int head_dim = DitConfig::head_dim) {
    for (int i = 0; i < head_dim / 2; ++i) {
        const float c = std::cos(angles[i]), s = std::sin(angles[i]);
        const float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * c - b * s;
        v[2 * i + 1] = a * s + b * c;
    }
}

// Dense sparse-structure token order: 16^3 grid flattened as x*256 + y*16 + z (meshgrid 'ij').
inline int ss_token_index(int x, int y, int z, int res = SsDecConfig::latent_res) { return (x * res + y) * res + z; }

}  // namespace gufo::trellis2
