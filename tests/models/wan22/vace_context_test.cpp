// Written with AI assistance (Claude Code).
// Host-side VACE conditioning against ComfyUI's WanVaceToVideo /
// WAN21_Vace.extra_conds layout (tools/models/vace/vace_reference.py
// --context prints the same numbers from the Python formulas).
#include "src/models/wan22/vace_context.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace wan22 = gufo::models::wan22;

namespace {

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL wan22_vace_context_test: " << message << '\n';
    ++failures;
  }
}

bool Near(float a, float b) {
  return std::fabs(a - b) <= 1e-6F;
}

}  // namespace

int main() {
  // 5 pixel frames (2 latent frames) at 16x16 (latent 2x2), one reference.
  constexpr int kFrames = 5, kH = 16, kW = 16, kLf = 2, kLh = 2, kLw = 2;
  constexpr std::size_t kPlane = kLh * kLw;
  std::vector<float> inactive(16 * kLf * kPlane), reactive(inactive.size());
  for (std::size_t i = 0; i < inactive.size(); ++i) {
    inactive[i] = static_cast<float>(i);
    reactive[i] = -static_cast<float>(i);
  }
  std::vector<float> mask(static_cast<std::size_t>(kFrames) * kH * kW);
  for (int f = 0; f < kFrames; ++f)
    for (int y = 0; y < kH; ++y)
      for (int x = 0; x < kW; ++x)
        mask[(static_cast<std::size_t>(f) * kH + y) * kW + x] =
            static_cast<float>(f * 1000 + y * 16 + x);
  std::vector<float> reference(16 * kPlane);
  for (int c = 0; c < 16; ++c)
    for (std::size_t i = 0; i < kPlane; ++i)
      reference[c * kPlane + i] = 100.0F + static_cast<float>(c);

  wan22::VaceContextInputs inputs;
  inputs.frames = kFrames;
  inputs.height = kH;
  inputs.width = kW;
  inputs.inactive = inactive;
  inputs.reactive = reactive;
  inputs.mask = mask;
  inputs.references.emplace_back(reference);
  const std::vector<float> out = wan22::BuildVaceContext(inputs);
  constexpr int kOutFrames = 1 + kLf;
  constexpr std::size_t kChannel = kOutFrames * kPlane;
  Expect(out.size() == 96 * kChannel, "context size");
  if (out.size() == 96 * kChannel) {
    const auto at = [&](int c, int f, int y, int x) {
      return out[c * kChannel + f * kPlane + y * kLw + x];
    };
    // Reference frame: [latent | 0], mask 0.
    Expect(Near(at(3, 0, 1, 1), 103.0F), "reference latent");
    Expect(Near(at(20, 0, 0, 1), 0.0F), "reference reactive is zero");
    Expect(Near(at(40, 0, 1, 0), 0.0F), "reference mask is zero");
    // Video frames after it.
    Expect(Near(at(2, 1, 0, 1), inactive[(2 * kLf + 0) * kPlane + 1]),
           "inactive frame 0");
    Expect(Near(at(16 + 5, 2, 1, 0), reactive[(5 * kLf + 1) * kPlane + 2]),
           "reactive frame 1");
    // Mask channel kh * 8 + kw at latent frame f reads pixel frame
    // floor((f + 0.5) * 5 / 2) = 1, 3 at (y * 8 + kh, x * 8 + kw).
    const int kh = 3, kw = 6;
    Expect(Near(at(32 + kh * 8 + kw, 1, 1, 0),
                1000.0F * 1 + (8 + kh) * 16 + (0 + kw)),
           "mask unshuffle frame 0");
    Expect(Near(at(32 + kh * 8 + kw, 2, 0, 1),
                1000.0F * 3 + (0 + kh) * 16 + (8 + kw)),
           "mask unshuffle frame 1");
  }
  // No mask: ones; no references.
  inputs.mask = {};
  inputs.references.clear();
  const std::vector<float> plain = wan22::BuildVaceContext(inputs);
  Expect(plain.size() == 96 * kLf * kPlane, "plain size");
  if (!plain.empty())
    Expect(Near(plain[95 * kLf * kPlane + 3], 1.0F), "default mask ones");
  // Size mismatch.
  inputs.frames = 6;
  Expect(wan22::BuildVaceContext(inputs).empty(), "frames must be 4k+1");

  // Control split around grey (v - 0.5 with v in [0, 1]): inactive =
  // v' (1 - m) + 0.5, reactive = v' m + 0.5, in VAE range 2x - 1. One pixel,
  // grey padding for the second frame.
  const std::vector<std::uint8_t> video = {255, 0, 51};
  const std::vector<float> one_mask = {0.25F, 1.0F};
  std::vector<float> in, re;
  wan22::SplitVaceControl(video, 1, one_mask, 2, 1, 1, &in, &re);
  Expect(in.size() == 6 && re.size() == 6, "split sizes");
  if (in.size() == 6 && re.size() == 6) {
    // v = 0.5: inactive 2 * (0.5 * 0.75 + 0.5) - 1 = 0.75, reactive 0.25.
    Expect(Near(in[0], 0.75F) && Near(re[0], 0.25F), "split white");
    // v = -0.5: inactive -0.75, reactive -0.25.
    Expect(Near(in[1], -0.75F) && Near(re[1], -0.25F), "split black");
    // Padded grey frame: 0.
    Expect(Near(in[3], 0.0F) && Near(re[5], 0.0F), "split grey padding");
  }
  if (failures == 0)
    std::cout << "wan22_vace_context_test passed\n";
  return failures == 0 ? 0 : 1;
}
