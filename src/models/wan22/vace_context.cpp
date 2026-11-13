// Written with AI assistance (Claude Code).
#include "src/models/wan22/vace_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::models::wan22 {

void SplitVaceControl(std::span<const std::uint8_t> video, int video_frames,
                      std::span<const float> mask, int frames, int height,
                      int width, std::vector<float>* inactive,
                      std::vector<float>* reactive) {
  const std::size_t pixels = static_cast<std::size_t>(frames) * height * width;
  const std::size_t frame = static_cast<std::size_t>(height) * width;
  const std::size_t video_pixels =
      video.size() >= frame * 3 * std::max(video_frames, 0)
          ? frame * std::max(video_frames, 0)
          : 0;
  inactive->assign(pixels * 3, 0.0F);
  reactive->assign(pixels * 3, 0.0F);
  for (std::size_t p = 0; p < pixels; ++p) {
    const float m = mask.size() == pixels ? mask[p] : 1.0F;
    for (int c = 0; c < 3; ++c) {
      // ComfyUI: images in [0, 1], grey 0.5 without a control video; the
      // VAE sees 2x - 1.
      const float v = p < video_pixels
                          ? static_cast<float>(video[p * 3 + c]) / 255.0F - 0.5F
                          : 0.0F;
      (*inactive)[p * 3 + c] = 2.0F * (v * (1.0F - m) + 0.5F) - 1.0F;
      (*reactive)[p * 3 + c] = 2.0F * (v * m + 0.5F) - 1.0F;
    }
  }
}

std::vector<float> BuildVaceContext(const VaceContextInputs& in) {
  if (in.frames < 1 || (in.frames - 1) % 4 != 0 || in.height % 8 != 0 ||
      in.width % 8 != 0 || in.height < 8 || in.width < 8)
    return {};
  const int lf = 1 + (in.frames - 1) / 4, lh = in.height / 8,
            lw = in.width / 8;
  const int refs = static_cast<int>(in.references.size());
  const int frames = refs + lf;
  const std::size_t plane = static_cast<std::size_t>(lh) * lw;
  const std::size_t latent = static_cast<std::size_t>(16) * lf * plane;
  const std::size_t pixels =
      static_cast<std::size_t>(in.frames) * in.height * in.width;
  if (in.inactive.size() != latent || in.reactive.size() != latent ||
      (!in.mask.empty() && in.mask.size() != pixels))
    return {};
  for (const auto& ref : in.references)
    if (ref.size() != 16 * plane)
      return {};
  const std::size_t channel = static_cast<std::size_t>(frames) * plane;
  std::vector<float> out(static_cast<std::size_t>(96) * channel, 0.0F);
  // Reference frames: [latent | 0], mask 0.
  for (int r = 0; r < refs; ++r)
    for (int c = 0; c < 16; ++c)
      std::copy_n(in.references[r].begin() + c * plane, plane,
                  out.begin() + c * channel + r * plane);
  // Video frames: [inactive | reactive].
  for (int c = 0; c < 16; ++c)
    for (int f = 0; f < lf; ++f) {
      const std::size_t source = (static_cast<std::size_t>(c) * lf + f) * plane;
      const std::size_t target = c * channel + (refs + f) * plane;
      std::copy_n(in.inactive.begin() + source, plane, out.begin() + target);
      std::copy_n(in.reactive.begin() + source, plane,
                  out.begin() + 16 * channel + target);
    }
  // Mask: pixel unshuffle by 8 (channel kh * 8 + kw), nearest-exact in time
  // (source frame floor((f + 0.5) * frames / latent_frames)).
  for (int f = 0; f < lf; ++f) {
    const int source = std::min(
        static_cast<int>(std::floor((f + 0.5) * in.frames / lf)),
        in.frames - 1);
    for (int kh = 0; kh < 8; ++kh)
      for (int kw = 0; kw < 8; ++kw) {
        float* target = out.data() + (32 + kh * 8 + kw) * channel +
                        (refs + f) * plane;
        for (int y = 0; y < lh; ++y)
          for (int x = 0; x < lw; ++x)
            target[static_cast<std::size_t>(y) * lw + x] =
                in.mask.empty()
                    ? 1.0F
                    : in.mask[(static_cast<std::size_t>(source) * in.height +
                               y * 8 + kh) *
                                  in.width +
                              x * 8 + kw];
      }
  }
  return out;
}

}  // namespace gufo::models::wan22
