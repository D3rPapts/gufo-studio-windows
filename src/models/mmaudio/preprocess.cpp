// MMAudio for gufo: video frame selection and resizing (see preprocess.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>

namespace gufo::models::mmaudio {
namespace {

// PIL's bicubic (a = -0.5) coefficients for one axis, 22-bit fixed point,
// antialiased when shrinking (ImagingResample).
struct Coefficients {
  std::vector<int> first, count;
  std::vector<std::int32_t> weights;  // [out, ksize]
  int ksize{0};
};

double Bicubic(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0)
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

Coefficients MakeCoefficients(int in_size, int out_size) {
  constexpr int kPrecision = 22;
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = std::max(scale, 1.0);
  const double support = 2.0 * filterscale;
  Coefficients c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.first.resize(out_size);
  c.count.resize(out_size);
  c.weights.assign(static_cast<std::size_t>(out_size) * c.ksize, 0);
  std::vector<double> k(c.ksize);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    xmin = std::max(xmin, 0);
    int xmax = static_cast<int>(center + support + 0.5);
    xmax = std::min(xmax, in_size) - xmin;
    double ww = 0;
    for (int x = 0; x < xmax; ++x) {
      k[x] = Bicubic((x + xmin - center + 0.5) * ss);
      ww += k[x];
    }
    for (int x = 0; x < xmax; ++x) {
      const double w = ww != 0.0 ? k[x] / ww : k[x];
      c.weights[static_cast<std::size_t>(xx) * c.ksize + x] =
          static_cast<std::int32_t>(w < 0 ? w * (1 << kPrecision) - 0.5
                                          : w * (1 << kPrecision) + 0.5);
    }
    c.first[xx] = xmin;
    c.count[xx] = xmax;
  }
  return c;
}

std::uint8_t Clip8(std::int64_t value) {
  value >>= 22;
  return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value, 0, 255));
}

std::uint64_t SplitMix(std::uint64_t* state) {
  std::uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

}  // namespace

std::vector<std::uint8_t> ResizeBicubic(std::span<const std::uint8_t> rgb,
                                        int width, int height, int out_width,
                                        int out_height) {
  if (width == out_width && height == out_height)
    return {rgb.begin(), rgb.end()};
  const Coefficients cx = MakeCoefficients(width, out_width);
  const Coefficients cy = MakeCoefficients(height, out_height);
  std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(height) *
                                       out_width * 3);
  for (int y = 0; y < height; ++y)
    for (int xx = 0; xx < out_width; ++xx)
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = std::int64_t{1} << 21;
        const std::int32_t* w =
            cx.weights.data() + static_cast<std::size_t>(xx) * cx.ksize;
        const std::uint8_t* row =
            rgb.data() + (static_cast<std::size_t>(y) * width + cx.first[xx]) *
                             3 +
            c;
        for (int x = 0; x < cx.count[xx]; ++x)
          sum += static_cast<std::int64_t>(w[x]) * row[x * 3];
        horizontal[(static_cast<std::size_t>(y) * out_width + xx) * 3 + c] =
            Clip8(sum);
      }
  std::vector<std::uint8_t> out(static_cast<std::size_t>(out_height) *
                                out_width * 3);
  for (int yy = 0; yy < out_height; ++yy) {
    const std::int32_t* w =
        cy.weights.data() + static_cast<std::size_t>(yy) * cy.ksize;
    for (int x = 0; x < out_width; ++x)
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = std::int64_t{1} << 21;
        for (int y = 0; y < cy.count[yy]; ++y)
          sum += static_cast<std::int64_t>(w[y]) *
                 horizontal[(static_cast<std::size_t>(cy.first[yy] + y) *
                                 out_width +
                             x) *
                                3 +
                            c];
        out[(static_cast<std::size_t>(yy) * out_width + x) * 3 + c] =
            Clip8(sum);
      }
  }
  return out;
}

std::vector<float> ClipPixels(std::span<const std::uint8_t> rgb, int width,
                              int height) {
  const std::vector<std::uint8_t> resized =
      ResizeBicubic(rgb, width, height, kClipSize, kClipSize);
  constexpr float kMean[3] = {0.48145466F, 0.4578275F, 0.40821073F};
  constexpr float kStd[3] = {0.26862954F, 0.26130258F, 0.27577711F};
  std::vector<float> out(static_cast<std::size_t>(3) * kClipSize * kClipSize);
  for (int y = 0; y < kClipSize; ++y)
    for (int x = 0; x < kClipSize; ++x)
      for (int c = 0; c < 3; ++c)
        out[(static_cast<std::size_t>(c) * kClipSize + y) * kClipSize + x] =
            (resized[(static_cast<std::size_t>(y) * kClipSize + x) * 3 + c] /
                 255.0F -
             kMean[c]) /
            kStd[c];
  return out;
}

std::vector<float> SyncPixels(std::span<const std::uint8_t> rgb, int width,
                              int height) {
  // torchvision Resize(224): shorter side 224, the other int(224 * l / s).
  int out_w, out_h;
  if (width <= height) {
    out_w = kSyncSize;
    out_h = static_cast<int>(static_cast<double>(kSyncSize) * height / width);
  } else {
    out_h = kSyncSize;
    out_w = static_cast<int>(static_cast<double>(kSyncSize) * width / height);
  }
  const std::vector<std::uint8_t> resized =
      ResizeBicubic(rgb, width, height, out_w, out_h);
  // CenterCrop: offsets int(round((size - 224) / 2)) (round half to even).
  const int top = static_cast<int>(std::nearbyint((out_h - kSyncSize) / 2.0));
  const int left = static_cast<int>(std::nearbyint((out_w - kSyncSize) / 2.0));
  std::vector<float> out(static_cast<std::size_t>(3) * kSyncSize * kSyncSize);
  for (int y = 0; y < kSyncSize; ++y)
    for (int x = 0; x < kSyncSize; ++x)
      for (int c = 0; c < 3; ++c)
        out[(static_cast<std::size_t>(c) * kSyncSize + y) * kSyncSize + x] =
            (resized[(static_cast<std::size_t>(top + y) * out_w + left + x) *
                         3 +
                     c] /
                 255.0F -
             0.5F) /
            0.5F;
  return out;
}

VideoSampler::VideoSampler(double duration, double source_fps, Mode mode)
    : duration_(duration), source_fps_(source_fps), mode_(mode) {
  clip_target_ = static_cast<int>(kClipFps * duration);
  sync_target_ = static_cast<int>(kSyncFps * duration);
  workers_ = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
}

VideoSampler::~VideoSampler() {
  for (auto& job : pending_)
    if (job.valid())
      job.wait();
}

void VideoSampler::Queue(std::shared_ptr<const std::vector<std::uint8_t>> frame,
                         int width, int height, bool clip, int slot) {
  // Bound the number of frames in flight.
  while (pending_.size() >= workers_) {
    pending_.front().wait();
    pending_.erase(pending_.begin());
  }
  std::vector<float>* target = clip ? &clip_[slot] : &sync_[slot];
  pending_.push_back(std::async(std::launch::async,
                                [frame, width, height, clip, target] {
                                  *target =
                                      clip ? ClipPixels(*frame, width, height)
                                           : SyncPixels(*frame, width, height);
                                }));
}

bool VideoSampler::Add(std::span<const std::uint8_t> rgb, int width,
                       int height) {
  const int index = index_++;
  int clip_copies = 0, sync_copies = 0;
  if (mode_ == Mode::kConsecutive) {
    clip_copies = index < clip_target_ ? 1 : 0;
    sync_copies = index < sync_target_ ? 1 : 0;
  } else {
    const double time = static_cast<double>(index) / source_fps_;
    if (time > duration_)
      return false;
    const double deltas[2] = {1.0 / kClipFps, 1.0 / kSyncFps};
    int* copies[2] = {&clip_copies, &sync_copies};
    for (int i = 0; i < 2; ++i)
      while (time >= next_[i]) {
        ++*copies[i];
        next_[i] += deltas[i];
      }
  }
  if (clip_copies > 0 || sync_copies > 0) {
    auto frame =
        std::make_shared<const std::vector<std::uint8_t>>(rgb.begin(),
                                                          rgb.end());
    for (int c = 0; c < clip_copies; ++c) {
      clip_.emplace_back();
      Queue(frame, width, height, true, static_cast<int>(clip_.size()) - 1);
    }
    for (int c = 0; c < sync_copies; ++c) {
      sync_.emplace_back();
      Queue(frame, width, height, false, static_cast<int>(sync_.size()) - 1);
    }
  }
  if (mode_ == Mode::kConsecutive)
    return index + 1 < std::max(clip_target_, sync_target_);
  return true;
}

VideoSampler::Result VideoSampler::Finish() {
  for (auto& job : pending_)
    job.wait();
  pending_.clear();
  Result result;
  double duration = duration_;
  int clip_count = static_cast<int>(clip_.size());
  int sync_count = static_cast<int>(sync_.size());
  if (mode_ == Mode::kConsecutive) {
    // MMAudioSampler: too few frames shorten the duration (CLIP first).
    const int total = index_;
    if (total < clip_target_)
      duration = total / kClipFps;
    if (total < sync_target_)
      duration = total / kSyncFps;
  } else {
    // load_video: the shorter stream bounds the duration.
    if (clip_count / kClipFps < duration)
      duration = clip_count / kClipFps;
    if (sync_count / kSyncFps < duration)
      duration = sync_count / kSyncFps;
  }
  clip_count = std::min(clip_count, static_cast<int>(kClipFps * duration));
  sync_count = std::min(sync_count, static_cast<int>(kSyncFps * duration));
  result.duration = duration;
  result.clip_frames = clip_count;
  result.sync_frames = sync_count;
  for (int i = 0; i < clip_count; ++i)
    result.clip.insert(result.clip.end(), clip_[i].begin(), clip_[i].end());
  for (int i = 0; i < sync_count; ++i)
    result.sync.insert(result.sync.end(), sync_[i].begin(), sync_[i].end());
  clip_.clear();
  sync_.clear();
  return result;
}

std::vector<float> GaussianNoise(std::uint64_t seed, std::size_t count) {
  std::vector<float> out(count);
  std::uint64_t state = seed;
  constexpr double kTwoPi = 6.283185307179586;
  for (std::size_t i = 0; i < count; i += 2) {
    // Uniform in (0, 1] and [0, 1) from the top 53 bits.
    const double u1 =
        (static_cast<double>(SplitMix(&state) >> 11U) + 1.0) * 0x1.0p-53;
    const double u2 = static_cast<double>(SplitMix(&state) >> 11U) * 0x1.0p-53;
    const double radius = std::sqrt(-2.0 * std::log(u1));
    out[i] = static_cast<float>(radius * std::cos(kTwoPi * u2));
    if (i + 1 < count)
      out[i + 1] = static_cast<float>(radius * std::sin(kTwoPi * u2));
  }
  return out;
}

std::string EncodeMonoWav(std::span<const float> samples, int sample_rate) {
  const std::uint32_t data_bytes =
      static_cast<std::uint32_t>(samples.size() * 2);
  std::string wav;
  wav.reserve(44 + data_bytes);
  const auto u32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
      wav.push_back(static_cast<char>((v >> (8 * i)) & 0xFFU));
  };
  const auto u16 = [&](std::uint16_t v) {
    wav.push_back(static_cast<char>(v & 0xFFU));
    wav.push_back(static_cast<char>(v >> 8U));
  };
  wav += "RIFF";
  u32(36 + data_bytes);
  wav += "WAVEfmt ";
  u32(16);
  u16(1);  // PCM
  u16(1);  // mono
  u32(static_cast<std::uint32_t>(sample_rate));
  u32(static_cast<std::uint32_t>(sample_rate) * 2);
  u16(2);
  u16(16);
  wav += "data";
  u32(data_bytes);
  for (const float sample : samples) {
    const float clamped = std::clamp(sample, -1.0F, 1.0F);
    const auto value =
        static_cast<std::int16_t>(std::lround(clamped * 32767.0F));
    u16(static_cast<std::uint16_t>(value));
  }
  return wav;
}

}  // namespace gufo::models::mmaudio
