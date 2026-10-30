#include "src/cli/serve/preview.hpp"

#include <jpeglib.h>
#include <webp/encode.h>
#include <webp/mux.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <csetjmp>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {
namespace {

using Clock = std::chrono::steady_clock;

// Linear latent -> RGB maps (range -1..1), least-squares fitted per model
// against its own VAE: latents of VAE-encoded sample images, normalized
// exactly as the sampler sees them, versus the images averaged over each
// latent pixel (tools in the preview work notes; see preview_factors.inc).
#include "src/cli/serve/preview_factors.inc"

struct Factors {
  int channels;
  const float* weights;  // [channels][3]
  const float* bias;     // [3]
  int temporal;          // pixel frames per latent frame (after the first)
};

Factors FactorsFor(LatentFormat format) {
  switch (format) {
    case LatentFormat::kQwenImage21:
      return {64, kQwenImage21Weights, kQwenImage21Bias, 1};
    case LatentFormat::kFlux2Patched:
      return {128, kFlux2PatchedWeights, kFlux2PatchedBias, 1};
    case LatentFormat::kWan22:
      return {48, kWan22Weights, kWan22Bias, 4};
    case LatentFormat::kMinimaxH3:
      return {24, kMinimaxH3Weights, kMinimaxH3Bias, 4};
  }
  throw std::logic_error("unknown latent format");
}

std::string Base64(std::string_view bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const auto at = [&](std::size_t k) {
      return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[k]));
    };
    const std::uint32_t word = (at(i) << 16) |
                               (i + 1 < bytes.size() ? at(i + 1) << 8 : 0) |
                               (i + 2 < bytes.size() ? at(i + 2) : 0);
    output += alphabet[(word >> 18) & 63];
    output += alphabet[(word >> 12) & 63];
    output += i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=';
    output += i + 2 < bytes.size() ? alphabet[word & 63] : '=';
  }
  return output;
}

// Flux2 latents carry fine texture in channels a linear map turns into
// coloured noise. Keep each pixel's luma, take the colour from a 3x3 box
// average: rgb' = box(rgb) + (Y - box(Y)).
void SmoothChroma(std::vector<float>& rgb, int width, int height) {
  const std::vector<float> source = rgb;
  const auto luma = [&](std::size_t p) {
    return 0.299F * source[p * 3] + 0.587F * source[p * 3 + 1] +
           0.114F * source[p * 3 + 2];
  };
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float sum[3] = {0, 0, 0};
      int count = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= height || xx < 0 || xx >= width)
            continue;
          const auto q = static_cast<std::size_t>(yy) * width + xx;
          for (int k = 0; k < 3; ++k)
            sum[k] += source[q * 3 + k];
          ++count;
        }
      const auto p = static_cast<std::size_t>(y) * width + x;
      const float box_luma =
          (0.299F * sum[0] + 0.587F * sum[1] + 0.114F * sum[2]) / count;
      const float detail = luma(p) - box_luma;
      for (int k = 0; k < 3; ++k)
        rgb[p * 3 + k] = sum[k] / count + detail;
    }
}

struct JpegError {
  jpeg_error_mgr manager{};
  std::jmp_buf jump{};
};

}  // namespace

std::vector<PreviewRgb> LatentToRgb(LatentFormat format,
                                    const models::LatentPreview& latent) {
  const Factors factors = FactorsFor(format);
  const int frames = latent.frames, height = latent.height,
            width = latent.width, channels = latent.channels;
  if (channels != factors.channels || frames < 1 || height < 1 || width < 1 ||
      latent.data.size() != static_cast<std::size_t>(frames) * height * width *
                                channels)
    throw std::invalid_argument("preview latent shape does not match");
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  std::vector<PreviewRgb> images(static_cast<std::size_t>(frames));
  std::vector<float> rgb(plane * 3);
  for (int f = 0; f < frames; ++f) {
    for (std::size_t p = 0; p < plane; ++p)
      for (int k = 0; k < 3; ++k)
        rgb[p * 3 + k] = factors.bias[k];
    if (latent.channel_major) {
      // [C][F][H][W]: one contiguous plane per channel.
      for (int c = 0; c < channels; ++c) {
        const float* source =
            latent.data.data() +
            (static_cast<std::size_t>(c) * frames + f) * plane;
        const float w0 = factors.weights[c * 3];
        const float w1 = factors.weights[c * 3 + 1];
        const float w2 = factors.weights[c * 3 + 2];
        for (std::size_t p = 0; p < plane; ++p) {
          rgb[p * 3] += source[p] * w0;
          rgb[p * 3 + 1] += source[p] * w1;
          rgb[p * 3 + 2] += source[p] * w2;
        }
      }
    } else {
      // [F][H][W][C]: one token row per pixel.
      const float* frame =
          latent.data.data() + static_cast<std::size_t>(f) * plane * channels;
      for (std::size_t p = 0; p < plane; ++p) {
        const float* token = frame + p * channels;
        float r = rgb[p * 3], g = rgb[p * 3 + 1], b = rgb[p * 3 + 2];
        for (int c = 0; c < channels; ++c) {
          r += token[c] * factors.weights[c * 3];
          g += token[c] * factors.weights[c * 3 + 1];
          b += token[c] * factors.weights[c * 3 + 2];
        }
        rgb[p * 3] = r;
        rgb[p * 3 + 1] = g;
        rgb[p * 3 + 2] = b;
      }
    }
    if (format == LatentFormat::kFlux2Patched)
      SmoothChroma(rgb, width, height);
    auto& image = images[static_cast<std::size_t>(f)];
    image.width = width;
    image.height = height;
    image.rgb.resize(plane * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
      const float v = std::isfinite(rgb[i]) ? (rgb[i] + 1.0F) * 127.5F : 0.0F;
      image.rgb[i] =
          static_cast<std::uint8_t>(std::clamp(v + 0.5F, 0.0F, 255.0F));
    }
  }
  return images;
}

PreviewRgb FitPreview(const PreviewRgb& image, int max_side) {
  if (image.width < 1 || image.height < 1 || max_side < 1)
    throw std::invalid_argument("invalid preview size");
  const double scale =
      static_cast<double>(max_side) / std::max(image.width, image.height);
  PreviewRgb out;
  out.width = std::max(1, static_cast<int>(std::lround(image.width * scale)));
  out.height = std::max(1, static_cast<int>(std::lround(image.height * scale)));
  if (out.width == image.width && out.height == image.height)
    return image;
  out.rgb.resize(static_cast<std::size_t>(out.width) * out.height * 3);
  const double sx = static_cast<double>(image.width) / out.width,
               sy = static_cast<double>(image.height) / out.height;
  for (int y = 0; y < out.height; ++y) {
    const double fy = std::clamp((y + 0.5) * sy - 0.5, 0.0,
                                 static_cast<double>(image.height - 1));
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const double ty = fy - y0;
    for (int x = 0; x < out.width; ++x) {
      const double fx = std::clamp((x + 0.5) * sx - 0.5, 0.0,
                                   static_cast<double>(image.width - 1));
      const int x0 = static_cast<int>(fx),
                x1 = std::min(x0 + 1, image.width - 1);
      const double tx = fx - x0;
      for (int k = 0; k < 3; ++k) {
        const auto at = [&](int yy, int xx) {
          return static_cast<double>(
              image.rgb[(static_cast<std::size_t>(yy) * image.width + xx) * 3 +
                        k]);
        };
        const double top = at(y0, x0) * (1 - tx) + at(y0, x1) * tx;
        const double bottom = at(y1, x0) * (1 - tx) + at(y1, x1) * tx;
        out.rgb[(static_cast<std::size_t>(y) * out.width + x) * 3 + k] =
            static_cast<std::uint8_t>(std::lround(
                std::clamp(top * (1 - ty) + bottom * ty, 0.0, 255.0)));
      }
    }
  }
  return out;
}

std::string EncodePreviewJpeg(const PreviewRgb& image, int quality) {
  if (image.width < 1 || image.height < 1 ||
      image.rgb.size() !=
          static_cast<std::size_t>(image.width) * image.height * 3)
    throw std::invalid_argument("invalid preview image");
  // Heap state only between setjmp and longjmp.
  auto error = std::make_unique<JpegError>();
  auto compressor = std::make_unique<jpeg_compress_struct>();
  unsigned char* buffer = nullptr;
  unsigned long size = 0;
  compressor->err = jpeg_std_error(&error->manager);
  error->manager.error_exit = [](j_common_ptr info) {
    std::longjmp(reinterpret_cast<JpegError*>(info->err)->jump, 1);
  };
  if (setjmp(error->jump)) {
    jpeg_destroy_compress(compressor.get());
    std::free(buffer);
    throw std::runtime_error("cannot encode preview JPEG");
  }
  jpeg_create_compress(compressor.get());
  jpeg_mem_dest(compressor.get(), &buffer, &size);
  compressor->image_width = static_cast<JDIMENSION>(image.width);
  compressor->image_height = static_cast<JDIMENSION>(image.height);
  compressor->input_components = 3;
  compressor->in_color_space = JCS_RGB;
  jpeg_set_defaults(compressor.get());
  jpeg_set_quality(compressor.get(), quality, TRUE);
  jpeg_start_compress(compressor.get(), TRUE);
  while (compressor->next_scanline < compressor->image_height) {
    JSAMPROW row = const_cast<JSAMPROW>(
        image.rgb.data() +
        static_cast<std::size_t>(compressor->next_scanline) * image.width * 3);
    jpeg_write_scanlines(compressor.get(), &row, 1);
  }
  jpeg_finish_compress(compressor.get());
  std::string bytes(reinterpret_cast<const char*>(buffer), size);
  jpeg_destroy_compress(compressor.get());
  std::free(buffer);
  return bytes;
}

std::string EncodePreviewWebp(std::span<const PreviewRgb> frames, int frame_ms,
                              int quality) {
  if (frames.empty())
    throw std::invalid_argument("no preview frames");
  const int width = frames.front().width, height = frames.front().height;
  WebPAnimEncoderOptions options;
  WebPConfig config;
  if (!WebPAnimEncoderOptionsInit(&options) || !WebPConfigInit(&config))
    throw std::runtime_error("libwebp version mismatch");
  options.anim_params.loop_count = 0;
  config.quality = static_cast<float>(quality);
  config.method = 0;  // fastest
  WebPAnimEncoder* encoder = WebPAnimEncoderNew(width, height, &options);
  if (encoder == nullptr)
    throw std::runtime_error("cannot allocate the WebP animation encoder");
  WebPPicture picture;
  WebPPictureInit(&picture);
  picture.width = width;
  picture.height = height;
  bool ok = true;
  int timestamp = 0;
  for (const auto& frame : frames) {
    ok = frame.width == width && frame.height == height &&
         WebPPictureImportRGB(&picture, frame.rgb.data(), width * 3) &&
         WebPAnimEncoderAdd(encoder, &picture, timestamp, &config);
    WebPPictureFree(&picture);
    if (!ok)
      break;
    timestamp += std::max(1, frame_ms);
  }
  WebPData data;
  WebPDataInit(&data);
  ok = ok && WebPAnimEncoderAdd(encoder, nullptr, timestamp, nullptr) &&
       WebPAnimEncoderAssemble(encoder, &data);
  std::string bytes;
  if (ok)
    bytes.assign(reinterpret_cast<const char*>(data.bytes), data.size);
  WebPDataClear(&data);
  WebPAnimEncoderDelete(encoder);
  if (!ok)
    throw std::runtime_error("cannot encode the preview WebP");
  return bytes;
}

bool ValidPreviewId(std::string_view id) noexcept {
  if (id.empty() || id.size() > 128)
    return false;
  return std::all_of(id.begin(), id.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
  });
}

std::string ParsePreviewRequest(const json::Value& object,
                                PreviewOptions* options) {
  if (!object.is_object())
    throw std::invalid_argument("gufo.preview must be an object");
  std::string id;
  for (const auto& [name, value] : object.members()) {
    const auto integer = [&](int low, int high) {
      if (!value.is_number() ||
          std::floor(value.as_double()) != value.as_double() ||
          value.as_double() < low || value.as_double() > high)
        throw std::invalid_argument("gufo.preview." + name + " must be " +
                                    std::to_string(low) + ".." +
                                    std::to_string(high));
      return static_cast<int>(value.as_double());
    };
    if (name == "id") {
      if (!value.is_string() || !ValidPreviewId(value.str()))
        throw std::invalid_argument(
            "gufo.preview.id must be 1..128 characters [A-Za-z0-9._-]");
      id = value.str();
    } else if (name == "always") {
      if (!value.is_bool())
        throw std::invalid_argument("gufo.preview.always must be a boolean");
      options->always = value.as_bool();
    } else if (name == "every") {
      options->every = integer(1, 100);
    } else if (name == "max_side") {
      options->max_side = integer(32, 1024);
    } else if (name == "method") {
      if (!value.is_string() || value.str() != "latent2rgb")
        throw std::invalid_argument(
            "gufo.preview.method supports only \"latent2rgb\"");
    } else {
      throw std::invalid_argument("unsupported field gufo.preview." + name);
    }
  }
  return id;
}

struct PreviewHub::Job {
  std::string id;
  LatentFormat format{LatentFormat::kQwenImage21};
  PreviewOptions options;
  bool started{false};
  Clock::time_point watched_until{};
  int streams{0};
  Clock::time_point touched{Clock::now()};
  // One queued latent at a time: a step that finds the previous preview
  // still encoding skips its own (no read-back, no copy).
  bool queued{false};
  models::LatentPreview pending;
  std::vector<float> pending_data;
  Snapshot snapshot;
};

struct PreviewHub::State {
  mutable std::mutex mutex;
  std::condition_variable work;
  std::condition_variable changed;
  std::map<std::string, std::shared_ptr<Job>, std::less<>> jobs;
  std::deque<std::shared_ptr<Job>> queue;
  bool stopping{false};
  Stats stats;
  std::thread thread;

  std::shared_ptr<Job> FindOrAdd(const std::string& id) {
    auto it = jobs.find(id);
    if (it != jobs.end())
      return it->second;
    // Forget finished or never started entries after ten minutes; keep at
    // most 256 (oldest idle first).
    const auto now = Clock::now();
    for (auto entry = jobs.begin(); entry != jobs.end();) {
      const auto& job = *entry->second;
      const bool idle = job.snapshot.state != "running" && job.streams == 0;
      if (idle && now - job.touched > std::chrono::minutes(10))
        entry = jobs.erase(entry);
      else
        ++entry;
    }
    while (jobs.size() >= 256) {
      auto oldest = jobs.end();
      for (auto entry = jobs.begin(); entry != jobs.end(); ++entry)
        if (entry->second->snapshot.state != "running" &&
            entry->second->streams == 0 &&
            (oldest == jobs.end() ||
             entry->second->touched < oldest->second->touched))
          oldest = entry;
      if (oldest == jobs.end())
        break;
      jobs.erase(oldest);
    }
    auto job = std::make_shared<Job>();
    job->id = id;
    job->snapshot.state = "pending";
    jobs.emplace(id, job);
    return job;
  }
};

PreviewHub::PreviewHub() : state_(std::make_unique<State>()) {
  state_->thread = std::thread([this] {
    std::unique_lock lock(state_->mutex);
    while (!state_->stopping) {
      if (state_->queue.empty()) {
        state_->work.wait(lock);
        continue;
      }
      lock.unlock();
      EncodePending();
      lock.lock();
    }
  });
}

PreviewHub::~PreviewHub() {
  {
    const std::lock_guard lock(state_->mutex);
    state_->stopping = true;
  }
  state_->work.notify_all();
  if (state_->thread.joinable())
    state_->thread.join();
}

PreviewHub& PreviewHub::Global() {
  static PreviewHub hub;
  return hub;
}

models::PreviewHook PreviewHub::Begin(const std::string& id,
                                      LatentFormat format,
                                      PreviewOptions options) {
  if (!ValidPreviewId(id))
    throw std::invalid_argument("invalid preview id");
  options.every = std::max(1, options.every);
  options.max_side = std::clamp(options.max_side, 32, 1024);
  std::shared_ptr<Job> job;
  {
    const std::lock_guard lock(state_->mutex);
    job = state_->FindOrAdd(id);
    job->format = format;
    job->options = options;
    job->started = true;
    job->queued = false;
    job->touched = Clock::now();
    job->snapshot.state = "running";
    job->snapshot.steps = 0;
    job->snapshot.first_frame = 0;
    job->snapshot.last_frame = std::max(1, options.pixel_frames) - 1;
  }
  state_->changed.notify_all();
  State* state = state_.get();
  models::PreviewHook hook;
  hook.wanted = [state, job](int step, int steps) {
    // The final step's image follows anyway.
    if (step >= steps || step % job->options.every != 0)
      return false;
    const std::lock_guard lock(state->mutex);
    if (job->queued || job->snapshot.state != "running")
      return false;
    return job->options.always || job->streams > 0 ||
           Clock::now() < job->watched_until;
  };
  hook.publish = [state, job](const models::LatentPreview& latent) {
    {
      const std::lock_guard lock(state->mutex);
      if (job->queued || state->stopping)
        return;
      job->pending = latent;
      job->pending_data.assign(latent.data.begin(), latent.data.end());
      job->pending.data = job->pending_data;
      job->queued = true;
      state->queue.push_back(job);
      ++state->stats.published;
    }
    state->work.notify_one();
  };
  return hook;
}

void PreviewHub::End(const std::string& id, bool ok) {
  {
    const std::lock_guard lock(state_->mutex);
    const auto it = state_->jobs.find(id);
    if (it == state_->jobs.end())
      return;
    it->second->snapshot.state = ok ? "done" : "failed";
    it->second->touched = Clock::now();
  }
  state_->changed.notify_all();
}

bool PreviewHub::EncodePending() {
  std::shared_ptr<Job> job;
  models::LatentPreview latent;
  std::vector<float> data;
  LatentFormat format;
  PreviewOptions options;
  {
    const std::lock_guard lock(state_->mutex);
    if (state_->queue.empty())
      return false;
    job = std::move(state_->queue.front());
    state_->queue.pop_front();
    latent = job->pending;
    data = std::move(job->pending_data);
    latent.data = data;
    format = job->format;
    options = job->options;
  }
  const auto begin = Clock::now();
  Snapshot update;
  try {
    const auto frames = LatentToRgb(format, latent);
    if (frames.size() == 1) {
      update.mime = "image/jpeg";
      update.image = std::make_shared<const std::string>(
          EncodePreviewJpeg(FitPreview(frames.front(), options.max_side)));
      update.frame = update.image;
    } else {
      std::vector<PreviewRgb> scaled;
      scaled.reserve(frames.size());
      for (const auto& frame : frames)
        scaled.push_back(FitPreview(frame, options.max_side));
      // The loop lasts as long as the video.
      const int frame_ms = static_cast<int>(std::lround(
          1000.0 * std::max(1, options.pixel_frames) /
          (std::max(1, options.fps) * static_cast<double>(frames.size()))));
      update.mime = "image/webp";
      update.image = std::make_shared<const std::string>(
          EncodePreviewWebp(scaled, frame_ms));
      update.frame = std::make_shared<const std::string>(
          EncodePreviewJpeg(scaled[scaled.size() / 2]));
    }
  } catch (const std::exception&) {
    update.mime.clear();
  }
  const double ms =
      std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  {
    const std::lock_guard lock(state_->mutex);
    job->queued = false;
    if (!update.mime.empty()) {
      auto& snapshot = job->snapshot;
      ++snapshot.seq;
      snapshot.step = latent.step;
      snapshot.steps = latent.steps;
      snapshot.mime = update.mime;
      snapshot.image = update.image;
      snapshot.frame = update.frame;
      ++state_->stats.encoded;
      state_->stats.encode_ms += ms;
    }
  }
  state_->changed.notify_all();
  return true;
}

std::optional<PreviewHub::Snapshot> PreviewHub::Latest(const std::string& id,
                                                       bool watch) {
  const std::lock_guard lock(state_->mutex);
  if (!watch) {
    const auto it = state_->jobs.find(id);
    if (it == state_->jobs.end())
      return std::nullopt;
    return it->second->snapshot;
  }
  const auto job = state_->FindOrAdd(id);
  job->touched = Clock::now();
  job->watched_until = job->touched + kWatchWindow;
  return job->snapshot;
}

void PreviewHub::Watch(const std::string& id) {
  const std::lock_guard lock(state_->mutex);
  const auto job = state_->FindOrAdd(id);
  ++job->streams;
  job->touched = Clock::now();
}

void PreviewHub::Unwatch(const std::string& id) {
  const std::lock_guard lock(state_->mutex);
  const auto it = state_->jobs.find(id);
  if (it != state_->jobs.end() && it->second->streams > 0) {
    --it->second->streams;
    it->second->touched = Clock::now();
  }
}

std::optional<PreviewHub::Snapshot> PreviewHub::WaitNewer(
    const std::string& id, std::uint64_t seq,
    std::chrono::milliseconds timeout) {
  std::unique_lock lock(state_->mutex);
  const auto deadline = Clock::now() + timeout;
  for (;;) {
    const auto it = state_->jobs.find(id);
    if (it == state_->jobs.end())
      return std::nullopt;
    const auto& snapshot = it->second->snapshot;
    if (snapshot.seq > seq || snapshot.state == "done" ||
        snapshot.state == "failed")
      return snapshot;
    if (state_->changed.wait_until(lock, deadline) == std::cv_status::timeout)
      return snapshot;
  }
}

PreviewHub::Stats PreviewHub::stats() const {
  const std::lock_guard lock(state_->mutex);
  return state_->stats;
}

bool IsPreviewPath(std::string_view path) noexcept {
  return path.starts_with("/v1/previews/");
}

namespace {

HttpResponse PreviewError(int status, const std::string& message,
                          const std::string& code) {
  auto body = json::Value::object();
  auto error = json::Value::object();
  error["message"] = message;
  error["type"] = "invalid_request_error";
  error["code"] = code;
  body["error"] = std::move(error);
  return {.status = status,
          .reason = status == 404 ? "Not Found"
                    : status == 405 ? "Method Not Allowed"
                                    : "Bad Request",
          .body = body.dump(),
          .headers = {{"Content-Type", "application/json"}}};
}

json::Value SnapshotJson(const std::string& id,
                         const PreviewHub::Snapshot& snapshot, bool frame) {
  auto body = json::Value::object();
  body["id"] = id;
  body["state"] = snapshot.state;
  body["seq"] = static_cast<double>(snapshot.seq);
  body["step"] = snapshot.step;
  body["steps"] = snapshot.steps;
  auto frames = json::Value::array();
  frames.push_back(snapshot.first_frame);
  frames.push_back(snapshot.last_frame);
  body["frames"] = std::move(frames);
  if (snapshot.image) {
    const auto& bytes = frame ? *snapshot.frame : *snapshot.image;
    body["mime"] = frame ? std::string("image/jpeg") : snapshot.mime;
    body["data"] = Base64(bytes);
  }
  return body;
}

// Plain query value (no percent decoding: the preview keys are ASCII words).
std::string QueryValue(std::string_view query, std::string_view key) {
  while (!query.empty()) {
    const auto end = query.find('&');
    const auto pair = query.substr(0, end);
    const auto equal = pair.find('=');
    if (pair.substr(0, equal) == key)
      return equal == std::string_view::npos
                 ? std::string()
                 : std::string(pair.substr(equal + 1));
    if (end == std::string_view::npos)
      break;
    query.remove_prefix(end + 1);
  }
  return {};
}

void AddHeaders(HttpResponse* response, const PreviewHub::Snapshot& snapshot) {
  response->headers.emplace_back("Cache-Control", "no-store");
  response->headers.emplace_back("X-Gufo-Preview-Seq",
                                 std::to_string(snapshot.seq));
  response->headers.emplace_back("X-Gufo-Preview-Step",
                                 std::to_string(snapshot.step));
  response->headers.emplace_back("X-Gufo-Preview-Steps",
                                 std::to_string(snapshot.steps));
  response->headers.emplace_back("X-Gufo-Preview-Frames",
                                 std::to_string(snapshot.first_frame) + "-" +
                                     std::to_string(snapshot.last_frame));
  response->headers.emplace_back("X-Gufo-Preview-State", snapshot.state);
}

}  // namespace

HttpResponse HandlePreviewRequest(const HttpRequest& request, PreviewHub& hub) {
  std::string_view rest(request.path);
  rest.remove_prefix(std::string_view("/v1/previews/").size());
  bool events = false;
  if (rest.ends_with("/events")) {
    rest.remove_suffix(std::string_view("/events").size());
    events = true;
  }
  const std::string id(rest);
  if (!ValidPreviewId(id))
    return PreviewError(404, "unknown preview path", "not_found");
  if (request.method != "GET")
    return PreviewError(405, "previews are read with GET",
                        "method_not_allowed");
  const std::string format = QueryValue(request.query, "format");
  if (!format.empty() && format != "frame" && format != "json" &&
      format != "frame-json")
    return PreviewError(400, "format must be frame, json or frame-json",
                        "invalid_format");
  if (!events) {
    const auto snapshot = hub.Latest(id, true);
    HttpResponse response;
    if (format == "json" || format == "frame-json") {
      response.body =
          SnapshotJson(id, *snapshot, format == "frame-json").dump();
      response.headers.emplace_back("Content-Type", "application/json");
    } else if (!snapshot->image) {
      response.status = 204;
      response.reason = "No Content";
    } else {
      response.body = format == "frame" ? *snapshot->frame : *snapshot->image;
      response.headers.emplace_back(
          "Content-Type", format == "frame" ? "image/jpeg" : snapshot->mime);
    }
    AddHeaders(&response, *snapshot);
    return response;
  }
  HttpResponse response;
  response.headers = {{"Content-Type", "text/event-stream"},
                      {"Cache-Control", "no-cache"},
                      {"X-Accel-Buffering", "no"}};
  const bool frame = format == "frame" || format == "frame-json";
  response.streaming_body = [&hub, id, frame,
                             cancelled = request.is_cancelled](
                                const HttpResponse::BodyWriter& write) {
    hub.Watch(id);
    struct Unwatch {
      PreviewHub& hub;
      const std::string& id;
      ~Unwatch() { hub.Unwatch(id); }
    } unwatch{hub, id};
    std::uint64_t seq = 0;
    std::string state;
    auto idle_since = Clock::now();
    auto keepalive = Clock::now();
    for (;;) {
      if (cancelled && cancelled())
        return;
      const auto snapshot =
          hub.WaitNewer(id, seq, std::chrono::milliseconds(500));
      if (!snapshot)
        return;
      if (snapshot->seq > seq && snapshot->image) {
        seq = snapshot->seq;
        if (!write("event: preview\ndata: " +
                   SnapshotJson(id, *snapshot, frame).dump() + "\n\n"))
          return;
        keepalive = Clock::now();
      }
      if (snapshot->state != state) {
        state = snapshot->state;
        idle_since = Clock::now();
        if (state == "done" || state == "failed") {
          (void)write("event: done\ndata: " +
                      SnapshotJson(id, *snapshot, frame).dump() + "\n\n");
          return;
        }
      }
      // A job that never starts releases its watcher after five minutes.
      if (state == "pending" &&
          Clock::now() - idle_since > std::chrono::minutes(5))
        return;
      if (Clock::now() - keepalive > std::chrono::seconds(10)) {
        if (!write(": keepalive\n\n"))
          return;
        keepalive = Clock::now();
      }
    }
  };
  return response;
}

}  // namespace gufo::server
