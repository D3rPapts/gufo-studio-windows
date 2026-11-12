// CPU tests for src/models/triposg/dit_cpu.cpp and scheduler.hpp on a tiny DiT shape
// (width 64, 2 heads, 5 blocks, 8 latent channels): schedule and Euler/CFG formulas,
// the diffusers sinusoid, shapes/finite/deterministic output, timestep sensitivity,
// skip wiring (blocks 3 and 4 consume blocks 1 and 0; zeroed skip weights change the
// result; a 2-layer model has no skips), taps and the conditioning path.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "src/models/triposg/dit.hpp"
#include "src/models/triposg/scheduler.hpp"

namespace tp = gufo::triposg;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

tp::DitShape Tiny() {
    tp::DitShape s;
    s.width = 64; s.heads = 2; s.layers = 5; s.mlp = 128; s.cross_dim = 16; s.in_channels = 8;
    return s;
}

std::vector<float> Signal(std::size_t n, float scale, int phase) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(static_cast<float>(i * 7 + phase) * 0.37f) * scale;
    return v;
}

double Diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::fabs(a[i] - b[i]);
    return d / static_cast<double>(a.size());
}

void SchedulerTest() {
    const auto sig = tp::flow_sigmas(50);
    Expect(sig.size() == 51 && sig[0] == 1.f && sig[50] == 0.f, "51 sigmas from 1 to 0");
    Expect(std::fabs(sig[1] - 0.98f) < 1e-6f && std::fabs(sig[49] - 0.02f) < 1e-6f, "sigma_i = 1 - i/50");
    Expect(tp::flow_timestep(sig, 0) == 1000.f && std::fabs(tp::flow_timestep(sig, 49) - 20.f) < 1e-3f, "timesteps 1000 .. 20");
    const auto shifted = tp::flow_sigmas(4, 3.f);
    // shift 3: s = 0.75 -> 3*0.75 / (1 + 2*0.75) = 0.9
    Expect(std::fabs(shifted[1] - 0.9f) < 1e-6f, "static shift formula");
    std::vector<float> x = {1.f, 2.f}, v = {10.f, -10.f};
    tp::euler_step(x, v, sig, 0);  // dt = 1 - 0.98 = 0.02
    Expect(std::fabs(x[0] - 1.2f) < 1e-5f && std::fabs(x[1] - 1.8f) < 1e-5f, "euler x += (sigma_i - sigma_i+1) v");
    std::vector<float> cond = {2.f, 0.f}, uncond = {1.f, 1.f};
    tp::apply_guidance(&cond, uncond, 7.f);
    Expect(cond[0] == 8.f && cond[1] == -6.f, "cfg = uncond + 7 (cond - uncond)");
    const auto emb = tp::timestep_embedding(1000.f, 2048);
    Expect(emb.size() == 2048, "embedding dim");
    Expect(std::fabs(emb[0] - std::sin(1000.f)) < 1e-5f && std::fabs(emb[1024] - std::cos(1000.f)) < 1e-5f, "j = 0: sin/cos(t)");
    const float f1 = std::exp(-std::log(10000.f) / 1024.f);
    Expect(std::fabs(emb[1] - std::sin(1000.f * f1)) < 1e-4f, "j = 1 frequency");
    Expect(std::fabs(emb[1023] - std::sin(1000.f * std::exp(-std::log(10000.f) * 1023.f / 1024.f))) < 1e-5f, "last sin");
}

void ForwardTest() {
    const tp::DitShape s = Tiny();
    tp::DitHostWeights w = tp::DitHostWeights::Random(s, 11, 0.08f);
    const int tokens = 6, cond_tokens = 5;
    const auto x = Signal(static_cast<std::size_t>(tokens) * s.in_channels, 1.f, 0);
    const auto cond = Signal(static_cast<std::size_t>(cond_tokens) * s.cross_dim, 1.f, 3);
    std::vector<float> v, again;
    std::map<int, std::vector<float>> taps;
    tp::DitForwardOptions opt;
    opt.tap_layers = {0, 4};
    std::string error;
    Expect(tp::dit_forward_cpu(w, x.data(), tokens, cond.data(), cond_tokens, 1000.f, &v, &taps, &error, opt), "forward ok: " + error);
    Expect(v.size() == static_cast<std::size_t>(tokens) * s.in_channels, "v shape [tokens, in]");
    bool finite = true;
    for (float f : v) finite = finite && std::isfinite(f);
    Expect(finite, "finite");
    Expect(taps.count(0) == 1 && taps.count(4) == 1 && taps[0].size() == static_cast<std::size_t>(tokens + 1) * s.width, "taps [tokens+1, width]");
    Expect(tp::dit_forward_cpu(w, x.data(), tokens, cond.data(), cond_tokens, 1000.f, &again, nullptr, &error) && again == v, "deterministic");
    std::vector<float> v_t;
    Expect(tp::dit_forward_cpu(w, x.data(), tokens, cond.data(), cond_tokens, 500.f, &v_t, nullptr, &error) && Diff(v, v_t) > 1e-4, "timestep changes v");
    const auto cond2 = Signal(cond.size(), 1.f, 9);
    std::vector<float> v_c;
    Expect(tp::dit_forward_cpu(w, x.data(), tokens, cond2.data(), cond_tokens, 1000.f, &v_c, nullptr, &error) && Diff(v, v_c) > 1e-4, "conditioning changes v");
    // Zero conditioning (the CFG negative) is a valid input.
    const std::vector<float> zeros(cond.size(), 0.f);
    std::vector<float> v_z;
    Expect(tp::dit_forward_cpu(w, x.data(), tokens, zeros.data(), cond_tokens, 1000.f, &v_z, nullptr, &error) && Diff(v, v_z) > 1e-4, "zero conditioning differs");
}

void SkipTest() {
    const tp::DitShape s = Tiny();
    Expect(s.skip_last_push() == 1 && s.skip_first_pop() == 3 && !s.has_skip(2) && s.has_skip(3), "5 layers: push 0..1, pop 3..4");
    tp::DitShape real;
    Expect(real.skip_last_push() == 9 && real.skip_first_pop() == 11 && !real.has_skip(10) && real.has_skip(20), "21 layers: push 0..9, pop 11..20");
    tp::DitHostWeights w = tp::DitHostWeights::Random(s, 21, 0.08f);
    const int tokens = 4, cond_tokens = 3;
    const auto x = Signal(static_cast<std::size_t>(tokens) * s.in_channels, 1.f, 1);
    const auto cond = Signal(static_cast<std::size_t>(cond_tokens) * s.cross_dim, 1.f, 2);
    std::vector<float> v;
    std::string error;
    Expect(tp::dit_forward_cpu(w, x.data(), tokens, cond.data(), cond_tokens, 700.f, &v, nullptr, &error), "skip model ok: " + error);
    // Zeroing the skip_linear weights of block 3 (so only its bias + norm pass) changes v.
    tp::DitHostWeights z = w;
    const auto inner = w.load_block;
    z.load_block = [inner](int layer, tp::DitBlockWeights* b, std::string* e) {
        if (!inner(layer, b, e)) return false;
        if (layer == 3) for (auto& f : b->skip_w) f = 0.f;
        return true;
    };
    std::vector<float> v_z;
    Expect(tp::dit_forward_cpu(z, x.data(), tokens, cond.data(), cond_tokens, 700.f, &v_z, nullptr, &error) && Diff(v, v_z) > 1e-4, "skip weights matter");
    // Skip rows: block 4 consumes block 0's output. Capture block 0 and check that the
    // concat input would be it (indirectly: perturbing block 0's output only via a
    // different time token changes block 4 even when blocks 1..3 are frozen is hard
    // to isolate; instead assert the stack discipline by running a 2-layer model
    // (no skips) and a 3-layer model (layer 2 pops layer 0) without underflow).
    tp::DitShape two = s;
    two.layers = 2;
    tp::DitHostWeights w2 = tp::DitHostWeights::Random(two, 5, 0.08f);
    Expect(!two.has_skip(1) && tp::dit_forward_cpu(w2, x.data(), tokens, cond.data(), cond_tokens, 700.f, &v, nullptr, &error), "2 layers: no skip");
    tp::DitShape three = s;
    three.layers = 3;
    tp::DitHostWeights w3 = tp::DitHostWeights::Random(three, 6, 0.08f);
    Expect(three.skip_first_pop() == 2 && three.skip_last_push() == 0 &&
               tp::dit_forward_cpu(w3, x.data(), tokens, cond.data(), cond_tokens, 700.f, &v, nullptr, &error),
           "3 layers: layer 2 pops layer 0");
    // A loader failure propagates.
    tp::DitHostWeights bad = w;
    bad.load_block = [](int, tp::DitBlockWeights*, std::string* e) { if (e) *e = "boom"; return false; };
    Expect(!tp::dit_forward_cpu(bad, x.data(), tokens, cond.data(), cond_tokens, 700.f, &v, nullptr, &error) && error == "boom", "loader error propagates");
}

void TimeTokenTest() {
    // With proj_out reading only the latent rows, the time token influences v only
    // through attention: a model with zero attention output weights and zero FFN
    // output weights must give v = proj_out(norm_out(proj_in(x))) regardless of t.
    const tp::DitShape s = Tiny();
    tp::DitHostWeights w = tp::DitHostWeights::Random(s, 31, 0.08f);
    const auto inner = w.load_block;
    w.load_block = [inner](int layer, tp::DitBlockWeights* b, std::string* e) {
        if (!inner(layer, b, e)) return false;
        for (auto* m : {&b->o1_w, &b->o1_b, &b->o2_w, &b->o2_b, &b->ff2_w, &b->ff2_b}) for (auto& f : *m) f = 0.f;
        for (auto& f : b->skip_w) f = 0.f;
        for (auto& f : b->skip_b) f = 0.f;
        return true;
    };
    // Also neutralise the skip blocks: with zero skip weights the hidden state becomes
    // LN(bias = 0) = 0 for layers >= 3, so restrict to a 3-layer model where only layer 2
    // has a skip; compare v at two timesteps for layers 0..1 instead via a 2-layer model.
    tp::DitShape two = s;
    two.layers = 2;
    tp::DitHostWeights w2 = tp::DitHostWeights::Random(two, 31, 0.08f);
    const auto inner2 = w2.load_block;
    w2.load_block = [inner2](int layer, tp::DitBlockWeights* b, std::string* e) {
        if (!inner2(layer, b, e)) return false;
        for (auto* m : {&b->o1_w, &b->o1_b, &b->o2_w, &b->o2_b, &b->ff2_w, &b->ff2_b}) for (auto& f : *m) f = 0.f;
        return true;
    };
    const int tokens = 5, cond_tokens = 4;
    const auto x = Signal(static_cast<std::size_t>(tokens) * s.in_channels, 1.f, 4);
    const auto cond = Signal(static_cast<std::size_t>(cond_tokens) * s.cross_dim, 1.f, 5);
    std::vector<float> va, vb;
    std::string error;
    Expect(tp::dit_forward_cpu(w2, x.data(), tokens, cond.data(), cond_tokens, 1000.f, &va, nullptr, &error) &&
               tp::dit_forward_cpu(w2, x.data(), tokens, cond.data(), cond_tokens, 10.f, &vb, nullptr, &error),
           "frozen model ok: " + error);
    Expect(Diff(va, vb) < 1e-6, "time token only acts through attention/FFN (frozen model is t-invariant)");
    // And the latent rows alone feed proj_out: v equals the direct stem computation.
    std::vector<float> h(static_cast<std::size_t>(tokens) * s.width), n(h.size()), direct(va.size());
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < s.width; ++r) {
            float acc = w2.stem.proj_in_b[static_cast<std::size_t>(r)];
            for (int c = 0; c < s.in_channels; ++c) acc += x[static_cast<std::size_t>(t) * s.in_channels + c] * w2.stem.proj_in_w[static_cast<std::size_t>(r) * s.in_channels + c];
            h[static_cast<std::size_t>(t) * s.width + r] = acc;
        }
    for (int t = 0; t < tokens; ++t) {
        double mean = 0, var = 0;
        for (int r = 0; r < s.width; ++r) mean += h[static_cast<std::size_t>(t) * s.width + r];
        mean /= s.width;
        for (int r = 0; r < s.width; ++r) { const double d = h[static_cast<std::size_t>(t) * s.width + r] - mean; var += d * d; }
        var /= s.width;
        for (int r = 0; r < s.width; ++r) n[static_cast<std::size_t>(t) * s.width + r] = static_cast<float>((h[static_cast<std::size_t>(t) * s.width + r] - mean) / std::sqrt(var + 1e-5));
    }
    for (int t = 0; t < tokens; ++t)
        for (int o = 0; o < s.in_channels; ++o) {
            float acc = w2.stem.proj_out_b[static_cast<std::size_t>(o)];
            for (int r = 0; r < s.width; ++r) acc += n[static_cast<std::size_t>(t) * s.width + r] * w2.stem.proj_out_w[static_cast<std::size_t>(o) * s.width + r];
            direct[static_cast<std::size_t>(t) * s.in_channels + o] = acc;
        }
    Expect(Diff(va, direct) < 1e-4, "frozen model: v = proj_out(norm_out(proj_in(x))) (diff " + std::to_string(Diff(va, direct)) + ")");
}

}  // namespace

int main() {
    SchedulerTest();
    ForwardTest();
    SkipTest();
    TimeTokenTest();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "dit_cpu_test: ok\n";
    return 0;
}
