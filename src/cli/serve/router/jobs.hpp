#ifndef GUFO_CLI_SERVE_ROUTER_JOBS_HPP_
#define GUFO_CLI_SERVE_ROUTER_JOBS_HPP_

// Job API for the router (`gufo router --jobs-dir DIR`): long requests
// (videos, large images, music, speech, chat) run in the background instead
// of on one open HTTP connection.
//
//   POST   /v1/jobs                    {"request": {"path": "/v1/videos",
//                                       "body": {...}}, "webhook_url": "...",
//                                       "metadata": {...}} -> 202 + the job
//                                      (Idempotency-Key: a repeated submit
//                                      returns the first job)
//   GET    /v1/jobs[?limit=N&status=S] newest first; filters q (text over
//                                      metadata.titel/.prompt, model, id),
//                                      seite, geraet, kind (image|music|
//                                      speech|video|other), since/before
//                                      (unix seconds, created_at)
//   GET    /v1/jobs/{id}               status, progress, result links
//   PATCH  /v1/jobs/{id}               {"keep": true|false}: a kept job's
//                                      files never expire (also POST
//                                      /v1/jobs/{id}/keep)
//   DELETE /v1/jobs/{id}               queued/running: cancel (also POST
//                                      /v1/jobs/{id}/cancel); finished:
//                                      remove the record and its files
//   DELETE /v1/jobs?older_than=..&..   remove every finished job matching
//                                      all given filters (older_than: age in
//                                      seconds, unix seconds or ISO 8601;
//                                      plus the list filters); at least one
//                                      filter; dry_run=1 only counts;
//                                      kept jobs need include_kept=1
//   GET    /v1/jobs/{id}/files/{name}  one result file
//   GET    /v1/jobs/{id}/thumb         a small preview of the job's first
//          [?w=256&file=name]          image, video or audio file: JPEG of
//                                      width w (64..1024) for images and
//                                      videos (frame at 10 % of the
//                                      length), a waveform PNG (w x w/3)
//                                      for audio. Made with ffmpeg
//                                      (--comfy-ffmpeg; without it 501
//                                      thumb_unavailable), cached next to
//                                      the job's files as thumb-<w>.jpg/
//                                      .png and served with Cache-Control
//                                      public, max-age=86400. The job JSON
//                                      carries "thumb_url" when it has
//                                      such a file.
//
// Pool relay (placement = pool, Options::relay): GET /v1/jobs?scope=pool
// (or pool=1) merges this router's list with the partner router's (same
// filters, 3 s), newest first, every job with "owner": "local" or the
// partner's "http://host:port"; the answer's "partner" names the partner,
// "unreachable" when it did not answer, "none" without a partner. A job id
// this router does not know is asked from the partner for GET /v1/jobs/{id},
// its files and thumb, PATCH/DELETE and POST keep/cancel (10 s for JSON,
// 10 min for files; 502 partner_unreachable when it cannot be reached);
// relayed thumbs are cached under DIR/relay-cache/<id>/ for a day. Every
// relayed request carries X-Gufo-Relay: 1 and a request with that header
// (or X-Gufo-Forwarded) is never relayed on.
//
// Shared queue (1.4.20, Options::capabilities + relay): the router that
// took a job stays its owner (files, result, progress, webhook); only the
// execution may move to the partner.
//  - Needs: a job needs its model and the LoRA names of its body
//    ("loras"[].name, "gufo"."lora"). Submit answers 400 no_capable_router
//    when neither router can run it (message names what each has); at
//    dispatch the job's request carries X-Gufo-Pool: local|remote when
//    only one side can run it, and the router's pool decision keeps to
//    that side (waits there when it is busy).
//  - Order: a free runner compares its oldest queued job with the
//    partner's (GET /v1/jobs?status=queued, cached pull_poll). When the
//    partner's is older and this machine can run it, the runner POSTs
//    <partner>/router/pool/pull {"from", "router_id", "capabilities"}; the
//    partner hands out its oldest queued job the requester can run (never
//    a job a runner already took, never a concat job, only models it has
//    in the pool), marks it running on the requester and executes it on
//    its own thread with X-Gufo-Pool: remote, so the requester's worker
//    does the work while the owner keeps everything else. The requester's
//    runner waits for that job (polls the owner) before it takes the next
//    local one. Answer {"job": id, ...} or {"job": null}. A partner that
//    answers 404 (older than 1.4.20) pauses pulling for pull_hold, logged
//    once; an unreachable partner means "as before".
//  - View: every job carries "request.loras" and "running_on" ("local",
//    the partner's url while it runs there, null when not running);
//    scope=pool items add "position" (1 = the oldest queued job of both
//    routers). /router/status gets "pool_queue".
//   POST   /v1/videos/concat           {"videos": [2..20 result files of
//                                      finished jobs of this router, as
//                                      "/v1/jobs/{id}/files/video.mp4" or
//                                      the same with scheme and host, or
//                                      data: URLs of at most 64 MiB],
//                                      "output_format": "mp4",
//                                      "reencode": false} -> the stitched
//                                      mp4 (Content-Disposition video.mp4).
//                                      Needs ffmpeg + ffprobe (Options).
//                                      Wrapped in a job the result file is
//                                      video.mp4. Every input is probed;
//                                      with the same codec, size, frame
//                                      rate, pixel format and audio layout
//                                      the clips are joined without
//                                      re-encoding (concat demuxer, -c
//                                      copy), otherwise (or with
//                                      "reencode": true) they are scaled to
//                                      the first clip and re-encoded with
//                                      libx264 -crf 18 and AAC; audio is
//                                      kept, clips without audio get
//                                      silence. Errors: 400 invalid_input,
//                                      500 concat_failed with ffmpeg's last
//                                      stderr lines. Temp files live under
//                                      DIR/temp/ and are removed.
//
// Expiry (--jobs-expire-hours, default 12, 0 = never): the result media of a
// finished job that is not kept is deleted that long after completed_at;
// the small text files (request.body, response.json) and the record itself
// (prompt, metadata, timestamps) stay, result.files drops the deleted
// entries and the job shows "expired_at".
//
// Progress: image jobs read the live preview's step, video and music jobs
// poll the worker's job (music runs with "async": true). Any other job
// (speech, foley, transcriptions, chat) reports an estimate once a job of
// the same path and model has completed: elapsed / learned duration, at
// most 95, flagged "progress_estimated": true. The learned durations live
// in jobs.json ("durations").
//
// Jobs run through the router's own API, like the ComfyUI adapter's prompts:
// a /v1/videos job creates the worker's video job and polls it (progress,
// cancel = DELETE), an image job's progress is its live preview's step.
// Results land in DIR/<id>/ and DIR/jobs.json keeps every job across router
// restarts: queued jobs run again, a job that was running is failed.
//
// Webhooks: with "webhook_url" the router POSTs the finished job (completed,
// failed or cancelled) there, signed like Standard Webhooks (webhook-id,
// webhook-timestamp, webhook-signature) when --jobs-webhook-secret is set,
// with a timeout per attempt and growing waits between retries. Only plain
// http targets the router allows (--jobs-webhook-allow: host names, IPv4
// addresses or CIDR ranges, each with an optional :port, or *) are accepted,
// so a request cannot make the router call arbitrary hosts (SSRF). The
// router connects to the address it checked and follows no redirects.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <stop_token>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/core/json.hpp"

namespace gufo::router::jobs {

using comfy::Fetch;
using comfy::FetchResult;

/// One --jobs-webhook-allow entry.
struct AllowEntry {
  enum class Kind { kAny, kHost, kNetwork };
  Kind kind{Kind::kAny};
  /// kHost: the host name, lower case.
  std::string host;
  /// kNetwork: IPv4 network and mask, host byte order.
  std::uint32_t network{0};
  std::uint32_t mask{0};
  /// 0 = any port.
  int port{0};
};

/// Parses a comma separated allow list such as
/// "192.168.1.0/24,studio.local:8090,10.0.0.5,*". Empty text is an empty
/// list (webhooks off).
bool ParseAllowList(std::string_view text, std::vector<AllowEntry>* out,
                    std::string* error);

/// A webhook target: http only, no user info, no IPv6 literal.
struct WebhookUrl {
  /// Lower case host name or dotted IPv4 address.
  std::string host;
  int port{80};
  /// Path and query ("/" at least), without the fragment.
  std::string target;
};

bool ParseWebhookUrl(std::string_view url, WebhookUrl* out,
                     std::string* error);

/// Strict dotted-quad IPv4 (four decimal parts 0..255), host byte order.
bool ParseIpv4(std::string_view text, std::uint32_t* out);
std::string FormatIpv4(std::uint32_t address);

/// IPv4 addresses (host byte order) of a host name or address.
using Resolver =
    std::function<std::vector<std::uint32_t>(const std::string& host)>;
std::vector<std::uint32_t> ResolveIpv4(const std::string& host);

/// Whether the router may POST to `url`: the IPv4 address to connect to, or
/// nullopt with the reason in *error. An empty allow list allows nothing;
/// "*" allows any address except 0.0.0.0/8, link-local 169.254.0.0/16
/// (cloud metadata) and multicast/reserved 224.0.0.0/3, which only an
/// explicit entry opens.
std::optional<std::uint32_t> CheckWebhookTarget(
    const WebhookUrl& url, const std::vector<AllowEntry>& allow,
    const Resolver& resolve, std::string* error);

/// Standard Webhooks signature "v1,<base64 HMAC-SHA256>" of
/// "<id>.<timestamp>.<body>". A "whsec_<base64>" secret is decoded, any
/// other secret is used as raw bytes.
std::string SignWebhook(std::string_view secret, std::string_view id,
                        long long timestamp, std::string_view body);

/// Sends one webhook POST to `address`; returns the HTTP status, or 0 with
/// *error when no response arrived (connect failure, timeout).
using Deliver = std::function<int(std::uint32_t address, const WebhookUrl& url,
                                  const server::HttpRequest& request,
                                  std::chrono::milliseconds timeout,
                                  std::string* error)>;
/// Default Deliver: plain HTTP/1.1 through the router's upstream client.
int DeliverHttp(std::uint32_t address, const WebhookUrl& url,
                const server::HttpRequest& request,
                std::chrono::milliseconds timeout, std::string* error);

struct Options {
  /// Holds jobs.json and one folder per job.
  std::filesystem::path root;
  /// The router's own API (the same fetch the ComfyUI adapter uses).
  Fetch fetch;
  /// Preset model ids; a job naming another model is rejected at submit.
  /// Empty = no check.
  std::vector<std::string> models;
  /// Jobs that run at the same time (each holds one request to the router).
  std::size_t parallel{1};
  std::vector<AllowEntry> webhook_allow;
  std::string webhook_secret;
  /// Waits before the 2nd, 3rd, ... attempt; attempts = size + 1.
  std::vector<std::chrono::milliseconds> webhook_retry{
      std::chrono::seconds(5), std::chrono::seconds(30),
      std::chrono::seconds(120), std::chrono::seconds(600)};
  std::chrono::milliseconds webhook_timeout{std::chrono::seconds(10)};
  /// Empty = ResolveIpv4 / DeliverHttp.
  Resolver resolve;
  Deliver deliver;
  std::chrono::milliseconds video_poll{std::chrono::seconds(2)};
  /// Preview poll for the progress of image jobs.
  std::chrono::milliseconds progress_poll{std::chrono::seconds(1)};
  /// Result media of finished, not kept jobs are deleted this long after
  /// completed_at; zero = never.
  std::chrono::seconds expire_after{std::chrono::hours(12)};
  /// ffmpeg and ffprobe for POST /v1/videos/concat; empty = the endpoint
  /// answers 500 concat_failed.
  std::filesystem::path ffmpeg;
  std::filesystem::path ffprobe;
  /// One request to the pool partner router (see the pool relay above):
  /// its answer, streamed as the proxy gives it, with the response head
  /// bounded by `head_timeout` and the whole transfer by `total_timeout`;
  /// nullopt with *error when the partner cannot be reached. Empty = no
  /// partner.
  using Relay = std::function<std::optional<server::HttpResponse>(
      const server::HttpRequest& request, std::chrono::milliseconds head_timeout,
      std::chrono::milliseconds total_timeout, std::string* error)>;
  Relay relay;
  /// "http://host:port" of the partner, as /router/status names it.
  std::string partner_name;
  /// What this router can run, as GET /router/capabilities answers it
  /// ({"router_id", "partner", "models": {id: {"loras", "placement",
  /// ...}}}); empty = no capability checks, no pulling. Called without the
  /// queue's lock held.
  std::function<json::Value()> capabilities;
  /// The partner's answer to GET /router/capabilities (cached by the
  /// caller), nullopt while unknown (unreachable, older version).
  std::function<std::optional<json::Value>()> partner_capabilities;
  /// "http://host:port" this router is reached under: sent as "from" on a
  /// pull; the partner's capabilities name it more exactly ("partner").
  std::string self_name;
  /// How often a free runner looks at the partner's queue; the partner's
  /// queued list and a pulled job's state are read at most this often.
  std::chrono::milliseconds pull_poll{std::chrono::seconds(2)};
  /// After a 404 on /router/pool/pull (partner older than 1.4.20) pulling
  /// pauses this long.
  std::chrono::milliseconds pull_hold{std::chrono::seconds(60)};
};

/// Set on every request relayed to the partner; never relayed on.
inline constexpr const char* kRelayHeader = "X-Gufo-Relay";
/// Set on a job's request when only one side of the pool can run it
/// ("local" | "remote"); the router's pool decision keeps to that side.
inline constexpr const char* kPoolSideHeader = "X-Gufo-Pool";
/// POST: the partner asks for a queued job to run on its worker.
inline constexpr const char* kPullPath = "/router/pool/pull";

/// The LoRA names a request body asks for: "loras"[].name (images) and
/// "gufo"."lora" (videos, a string); empty for anything else.
std::vector<std::string> LoraNames(const json::Value& body);

/// Whether a router with these capabilities (the "models" object of
/// /router/capabilities, or the whole answer) can run a job: the model is
/// served there, not placed remote, and every LoRA is registered. *why
/// (optional) gets the reason ("no model 'x'", "no LoRA 'winkel'").
bool CanRun(const json::Value& capabilities, const std::string& model,
            const std::vector<std::string>& loras,
            std::string* why = nullptr);

// ------------------------------------------------- video concat helpers

/// One input of POST /v1/videos/concat: a result file of a job.
struct JobFileRef {
  std::string id;
  std::string name;
};

/// "/v1/jobs/{id}/files/{name}", with or without scheme and host in front
/// ("http://192.168.178.198:8081/v1/jobs/job_.../files/video.mp4"); a query
/// or fragment is dropped. nullopt for anything else (another path, a bad
/// id, a name the file API would not serve).
std::optional<JobFileRef> ParseJobFileUrl(std::string_view url);

/// "data:<type>;base64,<payload>" -> the decoded bytes and the media type
/// (lower case, without parameters). nullopt when the text is no data URL
/// or the payload is not base64.
std::optional<std::string> ParseDataUrl(std::string_view url,
                                        std::string* media_type);

/// What ffprobe reports about one input (-show_streams -show_format).
struct ProbeInfo {
  std::string video_codec;
  int width{0};
  int height{0};
  /// r_frame_rate as ffprobe prints it ("30/1", "30000/1001").
  std::string frame_rate;
  std::string pix_fmt;
  bool has_audio{false};
  std::string audio_codec;
  int sample_rate{0};
  int channels{0};
  /// Seconds (format.duration, else the video stream's); 0 = unknown.
  double duration{0};
  /// Seconds of the video stream alone; 0 = unknown. The concat demuxer
  /// offsets the next clip by the longer (audio) stream otherwise, which
  /// leaves an uneven gap at the seam and a bogus r_frame_rate.
  double video_duration{0};
};

/// Parses ffprobe's JSON (-print_format json); false with *error when there
/// is no video stream.
bool ParseProbe(std::string_view text, ProbeInfo* out, std::string* error);

/// "a/b" -> a / b as double; 0 when the text is no positive fraction.
double FrameRateValue(std::string_view text);

/// "copy" when `reencode` is false and every input shares the video codec,
/// size, frame rate, pixel format and audio layout (none, or the same codec,
/// sample rate and channels); "reencode" otherwise.
std::string ConcatMode(const std::vector<ProbeInfo>& inputs, bool reencode);

/// GET /v1/jobs and DELETE /v1/jobs filters (empty = not set).
struct Filter {
  std::string status;
  /// Case-insensitive text over metadata.titel, metadata.prompt, model, id.
  std::string q;
  std::string seite;
  std::string geraet;
  /// image, music, speech, video, other (German names accepted too).
  std::string kind;
  /// created_at >= since, created_at < before (unix seconds).
  long long since{0};
  long long before{0};
  /// Finished before this unix second (bulk delete).
  long long older_than{0};

  bool any() const {
    return !status.empty() || !q.empty() || !seite.empty() ||
           !geraet.empty() || !kind.empty() || since > 0 || before > 0 ||
           older_than > 0;
  }
};

/// The job kind the pages derive from request.path: image, music, speech,
/// video or other.
std::string KindOf(std::string_view path);

/// "3600" (an age in seconds), "1762800000" (unix seconds) or ISO 8601
/// ("2026-11-10T08:00:00Z", "2026-11-10 08:00", "2026-11-10"): the cutoff
/// as unix seconds; nullopt for anything else.
std::optional<long long> ParseCutoff(std::string_view text, long long now);

class JobQueue {
public:
  explicit JobQueue(Options options);
  ~JobQueue();
  JobQueue(const JobQueue&) = delete;
  JobQueue& operator=(const JobQueue&) = delete;

  void Start();
  /// Stops the runners (a running job ends as failed "router_shutdown") and
  /// the webhook sender; undelivered webhooks are sent after a restart.
  void Stop();

  /// The answer for a /v1/jobs path, or nullopt for any other path.
  std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& request);

  /// Called from the router's supervision loop: runs the expiry about once
  /// a minute (the first pass shortly after the start).
  void Tick();
  /// Deletes the result media of finished, not kept jobs whose completed_at
  /// is at least `expire_after` before `now` (unix seconds); the number of
  /// jobs that expired. Never touches kept or younger jobs.
  std::size_t ExpireFiles(long long now);
  /// Removes relay-cache folders (thumbs of the partner's jobs) untouched
  /// for a day; runs with the expiry pass.
  void ExpireRelayCache();
  /// /router/status "pool_queue": {"local": queued here, "partner": queued
  /// there (null when unknown), "pulled": jobs this router fetched from the
  /// partner, "pushed": jobs handed to the partner, ...}.
  json::Value PoolQueueStatus();

  struct Job;

private:
  server::HttpResponse Submit(const server::HttpRequest& request);
  server::HttpResponse List(const server::HttpRequest& request);
  server::HttpResponse Get(const std::string& id);
  server::HttpResponse Cancel(const std::string& id);
  server::HttpResponse Keep(const std::string& id, const std::string& body);
  server::HttpResponse Delete(const std::string& id);
  server::HttpResponse BulkDelete(const server::HttpRequest& request);
  server::HttpResponse File(const std::string& id, const std::string& name);
  /// GET /v1/jobs/{id}/thumb: the cached preview, or made with ffmpeg.
  server::HttpResponse Thumb(const std::string& id,
                             const server::HttpRequest& request);
  /// GET /v1/jobs?scope=pool: this list merged with the partner's.
  server::HttpResponse PoolList(const server::HttpRequest& request);
  /// A job this router does not know: the same request to the partner.
  server::HttpResponse RelayJob(const std::string& id, std::string_view tail,
                                const server::HttpRequest& request);
  /// POST /router/pool/pull: hands the oldest queued job the requester can
  /// run to it (see the shared queue above).
  server::HttpResponse Pull(const server::HttpRequest& request);
  /// One queued job of the partner, as its list shows it.
  struct PartnerJob {
    std::string id;
    std::string model;
    long long created_at{0};
    std::vector<std::string> loras;
  };
  /// The partner's queued jobs (oldest first), cached pull_poll; nullopt
  /// when the partner did not answer.
  std::optional<std::vector<PartnerJob>> PartnerQueued();
  /// Runner side: pulls the partner's oldest queued job when it is older
  /// than `mine_created_at` (0 = nothing queued here) and runs here; waits
  /// for it to end. True when a job was pulled (the runner looks again).
  bool TryPull(long long mine_created_at);
  /// Waits (polling the partner) until the pulled job `id` left the
  /// requester's worker, or the partner stopped answering.
  void WaitForPulled(const std::string& id);
  /// Which side of the pool must run `job` ("local", "remote", "" = any),
  /// from the capabilities; `*refusal` gets the error of a job neither
  /// router can run (then "" is returned).
  std::string SideFor(const Job& job, json::Value* refusal);
  /// Joins pulled-job threads that ended (mutex held).
  void ReapPulledLocked();
  /// POST /v1/videos/concat answered at once: the mp4 bytes.
  server::HttpResponse Concat(const server::HttpRequest& request);
  /// Stitches the clips named by `body` into `output` (the job's temp
  /// folder is made under root/temp and removed again). Throws JobFailure
  /// (400 invalid_input, 500 concat_failed) or Cancelled.
  void ConcatVideos(const json::Value& body, const std::filesystem::path& output,
                    std::atomic<bool>* cancel, json::Value* summary);
  /// Drops the job from every index (mutex held); the caller removes its
  /// folder after unlocking.
  void ForgetLocked(const std::shared_ptr<Job>& job);
  bool MatchesLocked(const Job& job, const Filter& filter) const;

  void RunnerMain();
  void Execute(const std::shared_ptr<Job>& job);
  void RunVideo(const std::shared_ptr<Job>& job, const FetchResult& created);
  void RunMusic(const std::shared_ptr<Job>& job, const FetchResult& created);
  void RunConcat(const std::shared_ptr<Job>& job, const std::string& payload);
  FetchResult Call(const std::shared_ptr<Job>& job, std::string method,
                   std::string path, std::string query, std::string body,
                   std::string content_type);
  void PollPreview(const std::shared_ptr<Job>& job, const std::string& model,
                   std::stop_token stop);
  json::Value StoreResult(Job& job, const FetchResult& result);
  void Finish(const std::shared_ptr<Job>& job, std::string status,
              json::Value result, json::Value error);
  void WebhookMain();
  void DeliverOne(const std::shared_ptr<Job>& job);

  json::Value ViewLocked(const Job& job) const;
  json::Value RecordLocked(const Job& job) const;
  void PersistLocked();
  void Load();
  std::filesystem::path JobDir(const std::string& id) const;

  Options options_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable webhook_wake_;
  /// Every job in submit order (the index of jobs.json).
  std::vector<std::shared_ptr<Job>> order_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  std::map<std::string, std::string> idempotency_;
  std::deque<std::shared_ptr<Job>> pending_;
  /// Learned seconds per "<path> <model>" for the estimated progress.
  std::map<std::string, double> durations_;
  /// Shared queue bookkeeping (mutex held).
  long long pulled_{0};
  long long pushed_{0};
  /// Jobs this router pulled and waits for (ids on the partner).
  std::vector<std::string> pulling_;
  /// Pulling pauses until then after a 404 from the partner.
  std::chrono::steady_clock::time_point pull_hold_until_{};
  bool pull_hold_logged_{false};
  /// Last partner queued list (PartnerQueued cache).
  std::chrono::steady_clock::time_point partner_queued_at_{};
  std::optional<std::vector<PartnerJob>> partner_queued_;
  /// How the partner names this router (its capabilities' "partner").
  std::string self_name_;
  /// Pulled jobs run on their own threads (the runners are for local work).
  struct PulledThread {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<PulledThread> pulled_threads_;
  /// Next expiry pass (supervision thread only).
  std::chrono::steady_clock::time_point next_expiry_{};
  /// One ffmpeg thumbnail at a time.
  std::mutex thumb_mutex_;
  bool stopping_{false};
  std::vector<std::thread> runners_;
  std::thread webhook_thread_;
};

}  // namespace gufo::router::jobs

#endif  // GUFO_CLI_SERVE_ROUTER_JOBS_HPP_
