// ACE-Step 1.5 text-to-music behind /v1/audio/music (`gufo serve music`).
// gufo-studio-windows; no upstream capability (see media_service.hpp).
#ifndef GUFO_SERVER_MEDIA_MUSIC_API_HPP_
#define GUFO_SERVER_MEDIA_MUSIC_API_HPP_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "src/cli/serve/media/media_service.hpp"
#include "src/models/ace_step/pipeline.hpp"

namespace gufo::server {

inline constexpr std::string_view kMusicApiPath = "/v1/audio/music";

/// How `gufo serve music` finds its files (see
/// models::ace_step::ModelPaths::Resolve): the directory, the DiT choice
/// ("" / "turbo" / "sft" / "base" / file), the LM choice ("" / "none" /
/// "0.6b" / "1.7b" / "4b" / file), the variant override and the public
/// model id ("" = "ace-step-1.5-xl-<variant>").
struct MusicServiceOptions {
  std::filesystem::path model_root;
  std::string dit;
  std::string lm;
  std::string variant;
  std::string model_id;
};

/// ACE-Step 1.5 music behind /v1/audio/music:
///   POST /v1/audio/music               {prompt|tags, lyrics, duration, seed,
///        steps, shift, guidance_scale, cfg_interval_start/end, bpm,
///        keyscale, timesignature, language, audio_codes, lm_cfg,
///        lm_temperature, lm_top_p, task, source_audio, reference_audio,
///        repaint_start, repaint_end, repaint_mode, repaint_strength,
///        cover_strength, cover_noise_strength, track, track_classes,
///        vae_sample, response_format, bitrate_kbps, async}
///        -> audio/wav (response_format "wav", default), audio/mpeg
///        ("mp3") or audio/mp4 ("aac" / "m4a"), with Content-Disposition
///        output.{wav,mp3,m4a}; mp3 / m4a are FFmpeg transcodes at
///        bitrate_kbps (96..320, default 192) and fail the job with
///        "<format> encoding needs ffmpeg" when FFmpeg is missing
///        (async: 202 + job JSON)
///   GET  /v1/audio/music/{id}          job status (with preview_url and
///        preview_step while a live preview exists)
///   GET  /v1/audio/music/{id}/content  the audio in the requested format
///   GET  /v1/audio/music/{id}/preview  newest live preview: mono 24 kHz
///        audio/wav of ~6 s around the middle of the song decoded from the
///        sampler's clean estimate (X-Gufo-Preview-Step / -Steps /
///        -Seconds / -Window headers), 204 before the first one
/// `task` is text2music (default), repaint, cover, cover-nofsq (every
/// model) or extract, lego, complete (base model only; 400
/// `unsupported_task` elsewhere). Every task but text2music needs
/// `source_audio` (data URL, raw base64 or https; WAV directly, other
/// formats through FFmpeg); `reference_audio` gives the timbre.
/// Previews are rendered every 2nd diffusion step (request "preview":
/// {"every": N}) while the request asked for them ("preview": true) or a
/// client fetched the preview within the last kPreviewWatchWindow.
/// One worker generates jobs in order; finished audio is kept for an hour
/// (at most kKeptJobs results).
class MusicService final : public MediaService {
public:
  static constexpr std::size_t kQueueCapacity = 8;
  static constexpr std::size_t kKeptJobs = 16;
  static constexpr std::chrono::seconds kPreviewWatchWindow{10};
  /// Largest encoded source / reference audio accepted (a 10 minute 48 kHz
  /// stereo 16-bit WAV is 115 MB).
  static constexpr std::size_t kMaxAudioBytes = 256U << 20U;

  explicit MusicService(const MusicServiceOptions& options);
  ~MusicService();
  MusicService(const MusicService&) = delete;
  MusicService& operator=(const MusicService&) = delete;

  [[nodiscard]] bool ready() const override { return pipeline_ != nullptr; }
  [[nodiscard]] const std::string& initialization_error() const {
    return init_error_;
  }
  [[nodiscard]] std::string model_id() const override { return model_id_; }
  [[nodiscard]] std::string_view capability() const override {
    return "music";
  }
  [[nodiscard]] std::string_view owned_by() const override {
    return "operator-supplied-ace-step";
  }
  [[nodiscard]] bool HandlesPath(std::string_view path) const override;
  [[nodiscard]] std::string_view unavailable_message() const override {
    return "ACE-Step music is not ready";
  }
  [[nodiscard]] std::string_view unavailable_code() const override {
    return "music_service_unavailable";
  }
  [[nodiscard]] std::string variant() const;

  HttpResponse Handle(const HttpRequest& request) override;

private:
  /// The loaded DiT variant (turbo without a pipeline).
  [[nodiscard]] models::ace_step::DitVariant dit_variant() const;
  struct Job;
  HttpResponse Submit(const HttpRequest& request);
  HttpResponse Status(const std::string& id, bool content);
  HttpResponse Preview(const std::string& id);
  void Worker();
  void ExpireLocked();

  std::string model_id_;
  std::string init_error_;
  // shared_ptr: its deleter comes from Load (HIP side), so a CPU build
  // does not need the pipeline destructor.
  std::shared_ptr<models::ace_step::MusicPipeline> pipeline_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::uint64_t next_id_{1};
  bool stop_{false};
  std::thread worker_;
};

[[nodiscard]] bool IsMusicApiPath(std::string_view path) noexcept;

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_MUSIC_API_HPP_
