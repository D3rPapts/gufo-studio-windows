// CPU tests for src/models/trellis2/scheduler.hpp: rescaled time grid, Euler step,
// guidance interval, CFG mixing with and without x0 rescale, the cos|sin timestep
// embedding and the 3-D RoPE angle layout.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/scheduler.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
bool Near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    // t_seq for 12 steps, rescale_t 5: t_1 = 5*(11/12) / (1 + 4*(11/12)) = 55/56
    auto t = tr::flow_times(12, 5.f);
    Expect(t.size() == 13 && t[0] == 1.f && t[12] == 0.f, "13 times from 1 to 0");
    Expect(Near(t[1], 55.f / 56.f) && Near(t[6], 5.f * 0.5f / (1.f + 4.f * 0.5f)), "rescale_t 5 formula");
    auto plain = tr::flow_times(4, 1.f);
    Expect(Near(plain[1], 0.75f) && Near(plain[3], 0.25f), "rescale_t 1 = linspace");
    Expect(tr::model_timestep(t[1]) == 1000.f * t[1], "model gets 1000 t");

    std::vector<float> x = {1.f, 2.f}, v = {10.f, -10.f};
    tr::euler_step(x, v, 1.f, 0.9f);
    Expect(Near(x[0], 0.f) && Near(x[1], 3.f), "x -= (t - t_prev) v");

    Expect(tr::guidance_active(0.7f, tr::kSsSampler) && !tr::guidance_active(0.5f, tr::kSsSampler) &&
               tr::guidance_active(0.6f, tr::kSsSampler) && tr::guidance_active(1.f, tr::kSsSampler),
           "ss guidance interval [0.6, 1]");
    Expect(!tr::guidance_active(0.95f, tr::kTexSampler) && !tr::guidance_active(0.7f, tr::kTexSampler),
           "tex sampler has guidance 1 -> never active");

    // CFG without rescale
    tr::SamplerParams p = tr::kSsSampler;
    p.rescale = 0.f;
    std::vector<float> pos = {1.f, 2.f, 3.f}, neg = {0.f, 0.f, 1.f}, xt = {0.5f, 0.5f, 0.5f};
    tr::apply_guidance(&pos, neg, xt, 0.8f, p);
    Expect(Near(pos[0], 7.5f) && Near(pos[1], 15.f) && Near(pos[2], 7.5f * 3.f - 6.5f), "g*pos + (1-g)*neg");

    // Rescale with pos == neg: the mixture equals pos, ratio 1 -> unchanged
    std::vector<float> a = {0.3f, -1.2f, 2.f, 0.1f}, same = a, xs = {1.f, 0.f, -1.f, 2.f};
    tr::apply_guidance(&a, same, xs, 0.8f, tr::kSsSampler);
    Expect(Near(a[0], 0.3f, 1e-4f) && Near(a[2], 2.f, 1e-4f), "rescale with identical branches is the identity");

    // Rescale 1 brings the std of x0 back to the positive branch's std
    tr::SamplerParams full = tr::kSsSampler;
    full.rescale = 1.f;
    std::vector<float> pb = {1.f, -1.f, 2.f, -2.f}, nb = {0.5f, -0.5f, 1.f, -1.f}, xz(4, 0.f);
    const float tt = 0.8f;
    std::vector<float> mixed = pb;
    tr::apply_guidance(&mixed, nb, xz, tt, full);
    const double b = tr::x0_coefficient_v(tt, full.sigma_min);
    double s_pos = 0, s_mix = 0, m_pos = 0, m_mix = 0;
    for (int i = 0; i < 4; ++i) { m_pos += -b * pb[i]; m_mix += -b * mixed[i]; }
    m_pos /= 4; m_mix /= 4;
    for (int i = 0; i < 4; ++i) { s_pos += std::pow(-b * pb[i] - m_pos, 2); s_mix += std::pow(-b * mixed[i] - m_mix, 2); }
    Expect(Near(static_cast<float>(std::sqrt(s_pos)), static_cast<float>(std::sqrt(s_mix)), 1e-3f), "rescale 1 matches std of the positive x0");

    // Timestep embedding: cos half first
    auto e = tr::timestep_embedding(0.f);
    Expect(e.size() == 256 && e[0] == 1.f && e[127] == 1.f && e[128] == 0.f, "t=0: cos=1 | sin=0");
    auto e2 = tr::timestep_embedding(1000.f);
    Expect(Near(e2[0], std::cos(1000.f)) && Near(e2[128], std::sin(1000.f)) &&
               Near(e2[129], std::sin(1000.f * std::exp(-std::log(10000.f) / 128.f))),
           "frequencies exp(-ln 1e4 j/128)");

    // RoPE angles
    float ang[64];
    tr::rope_angles(3, 5, 7, ang);
    Expect(ang[0] == 3.f && ang[21] == 5.f && ang[42] == 7.f && ang[63] == 0.f, "axis blocks of 21, pad last pair");
    Expect(Near(ang[1], 3.f / std::pow(10000.f, 1.f / 21.f)), "j/21 exponent");
    float vec[128];
    for (int i = 0; i < 128; ++i) vec[i] = static_cast<float>(i);
    float zero[64] = {};
    tr::apply_rope(vec, zero);
    Expect(vec[5] == 5.f && vec[127] == 127.f, "zero angles leave the vector");
    float ninety[64] = {};
    ninety[0] = static_cast<float>(M_PI / 2);
    float w[128] = {1.f, 0.f};
    tr::apply_rope(w, ninety);
    Expect(Near(w[0], 0.f) && Near(w[1], 1.f), "pair (x0, x1) rotates by the angle");

    Expect(tr::ss_token_index(1, 2, 3) == 256 + 32 + 3 && tr::ss_token_index(15, 15, 15) == 4095, "ss token order x*256+y*16+z");

    if (failures == 0) std::cout << "trellis2_scheduler_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
