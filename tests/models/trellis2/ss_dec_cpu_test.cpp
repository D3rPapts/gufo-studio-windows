// CPU tests for src/models/trellis2/ss_dec_cpu.cpp on a tiny shape (channels 16/8/4,
// latent 2 at 2^3 -> logits 8^3): dense conv against a hand computation at one voxel
// (zero padding, tap order), pixel shuffle mapping, channel norm, forward shape /
// finite / deterministic, occupancy -> coordinates with max-pool.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/ss_dec.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
}  // namespace

int main() {
    // dense conv: res 3, ci 2, co 1, weight picks tap (dx=1,dy=0,dz=-1) channel 1 only
    const int res = 3, ci = 2;
    std::vector<float> in(27 * ci);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
    std::vector<float> w(27 * ci, 0.f);
    w[static_cast<std::size_t>(tr::tap_index(1, 0, -1)) * ci + 1] = 1.f;
    std::vector<float> b = {0.5f}, out(27);
    tr::dense_conv3(in.data(), res, ci, w.data(), b.data(), 1, out.data());
    // voxel (0,1,1): neighbour (1,1,0) -> index (1*3+1)*3+0 = 12, channel 1 -> 25 ; + 0.5
    Expect(out[(0 * 3 + 1) * 3 + 1] == 25.5f, "conv reads tap (1,0,-1) at voxel (0,1,1)");
    // voxel (2,1,1): neighbour (3,..) outside -> bias only
    Expect(out[(2 * 3 + 1) * 3 + 1] == 0.5f, "zero padding outside the grid");
    // voxel (0,1,0): neighbour z = -1 outside
    Expect(out[(0 * 3 + 1) * 3 + 0] == 0.5f, "zero padding at z = -1");

    // pixel shuffle: res 1, C 2 -> 8 voxels
    std::vector<float> ps_in(16);
    for (std::size_t i = 0; i < 16; ++i) ps_in[i] = static_cast<float>(i);   // channel index = c*8 + s
    std::vector<float> ps_out(16);
    tr::pixel_shuffle3(ps_in.data(), 1, 2, ps_out.data());
    // child (1,0,1): s = 4 + 0 + 1 = 5 -> channel 0 value 5, channel 1 value 13; voxel index (1*2+0)*2+1 = 5
    Expect(ps_out[5 * 2 + 0] == 5.f && ps_out[5 * 2 + 1] == 13.f, "pixel shuffle channel c*8 + (sx*4 + sy*2 + sz)");
    Expect(ps_out[0] == 0.f && ps_out[7 * 2 + 1] == 15.f, "corners");

    // channel norm + silu
    std::vector<float> row = {1.f, 3.f, 5.f, 7.f};
    std::vector<float> nw(4, 1.f), nb(4, 0.f);
    tr::channel_norm_silu(row.data(), 1, 4, nw.data(), nb.data(), 1e-5f, false);
    Expect(std::fabs(row[0] + 1.3416f) < 1e-3f && std::fabs(row[3] - 1.3416f) < 1e-3f, "channel layer norm");

    // tiny forward
    tr::SsDecShape shape;
    shape.latent_channels = 2;
    shape.latent_res = 2;
    shape.channels[0] = 16; shape.channels[1] = 8; shape.channels[2] = 4;
    auto weights = tr::SsDecWeights::Random(shape, 5, 0.1f);
    std::vector<float> latent(8 * 2);
    for (std::size_t i = 0; i < latent.size(); ++i) latent[i] = std::sin(static_cast<float>(i));
    std::vector<float> logits, logits2;
    std::string error;
    Expect(tr::ss_dec_forward_cpu(weights, latent.data(), &logits, &error), "forward: " + error);
    Expect(logits.size() == 512, "8^3 logits");
    bool finite = true;
    for (float v : logits) finite = finite && std::isfinite(v);
    Expect(finite, "finite");
    tr::ss_dec_forward_cpu(weights, latent.data(), &logits2, &error);
    Expect(logits == logits2, "deterministic");
    Expect(weights.parameter_count() == 2 * 27 * 16 + 16 + 4 * (2 * (16 * 27 * 16 + 16) + 4 * 16) + (8 * 8 * 27 * 16 + 64) +
                                            2 * (2 * (8 * 27 * 8 + 8) + 4 * 8) + (8 * 4 * 27 * 8 + 32) + 2 * (2 * (4 * 27 * 4 + 4) + 4 * 4) +
                                            8 + (27 * 4 + 1),
           "parameter count of the tiny shape");

    // occupancy -> coords with max pool 2 (res 4 -> 2)
    std::vector<float> occ(64, -1.f);
    occ[(1 * 4 + 0) * 4 + 3] = 1.f;   // (1,0,3) -> block (0,0,1)
    occ[(3 * 4 + 3) * 4 + 3] = 1.f;   // (3,3,3) -> block (1,1,1)
    const auto c = tr::occupancy_to_coords(occ.data(), 4, 2);
    Expect(c.size() == 2 && c[0] == tr::Coord{0, 0, 1} && c[1] == tr::Coord{1, 1, 1}, "max-pool coordinates in x,y,z order");
    const auto c4 = tr::occupancy_to_coords(occ.data(), 4, 4);
    Expect(c4.size() == 2 && c4[0] == tr::Coord{1, 0, 3}, "no pooling at full resolution");
    if (failures == 0) std::cout << "trellis2_ss_dec_cpu_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
