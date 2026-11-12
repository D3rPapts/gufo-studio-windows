// TripoSG: rectified-flow schedule (RectifiedFlowScheduler, shift 1), host-side
// Euler step and classifier-free guidance in FP32 (the reference upcasts the
// sample to FP32 before each step), and the diffusers sinusoidal timestep
// embedding used by TripoSGDiTModel (Timesteps(2048, flip_sin_to_cos=False, shift 0)).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "src/models/triposg/config.hpp"

namespace gufo::triposg {

// sigma_i = shift * s / (1 + (shift - 1) * s) with s = 1 - i/steps, i = 0..steps-1, then 0.
inline std::vector<float> flow_sigmas(int steps, float shift = SchedulerConfig::shift) {
    std::vector<float> sigmas;
    sigmas.reserve(static_cast<std::size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const double s = 1.0 - static_cast<double>(i) / steps;
        sigmas.push_back(static_cast<float>(shift * s / (1.0 + (shift - 1.0) * s)));
    }
    sigmas.push_back(0.f);
    return sigmas;
}

// Timestep handed to the DiT for step i.
inline float flow_timestep(const std::vector<float>& sigmas, int i) {
    return sigmas[static_cast<std::size_t>(i)] * static_cast<float>(SchedulerConfig::train_timesteps);
}

// x <- x + (sigma_i - sigma_{i+1}) * v   (TripoSG's direction convention).
inline void euler_step(std::vector<float>& x, const std::vector<float>& v, const std::vector<float>& sigmas, int i) {
    const float dt = sigmas[static_cast<std::size_t>(i)] - sigmas[static_cast<std::size_t>(i) + 1];
    for (std::size_t k = 0; k < x.size(); ++k) x[k] += dt * v[k];
}

// v = uncond + scale * (cond - uncond), in place into *cond.
inline void apply_guidance(std::vector<float>* cond, const std::vector<float>& uncond, float scale) {
    for (std::size_t k = 0; k < cond->size(); ++k) (*cond)[k] = uncond[k] + scale * ((*cond)[k] - uncond[k]);
}

// diffusers get_timestep_embedding(t, dim, flip_sin_to_cos=False, downscale_freq_shift=0,
// max_period=10000): [sin(t * f_0..f_{half-1}), cos(...)], f_j = exp(-ln(1e4) * j / half).
inline std::vector<float> timestep_embedding(float t, int dim) {
    const int half = dim / 2;
    std::vector<float> out(static_cast<std::size_t>(dim));
    for (int j = 0; j < half; ++j) {
        const float freq = std::exp(-std::log(10000.f) * static_cast<float>(j) / static_cast<float>(half));
        const float arg = t * freq;
        out[static_cast<std::size_t>(j)] = std::sin(arg);
        out[static_cast<std::size_t>(half + j)] = std::cos(arg);
    }
    return out;
}

}  // namespace gufo::triposg
