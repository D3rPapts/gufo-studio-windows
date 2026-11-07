// MMAudio for gufo: video / text to audio pipeline (see pipeline.hpp).
// Written with AI assistance (Claude Code).
#include "src/models/mmaudio/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "src/models/mmaudio/preprocess.hpp"

namespace gufo::models::mmaudio {
namespace {

using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// First file in `root` whose name starts with `prefix` and ends with
/// ".safetensors" (fp16 preferred over fp32 / others).
std::filesystem::path FindWeights(const std::filesystem::path& root,
                                  const std::string& prefix) {
  std::filesystem::path best;
  std::error_code ec;
  for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (!name.starts_with(prefix) || !name.ends_with(".safetensors"))
      continue;
    if (best.empty() || name.find("fp16") != std::string::npos)
      best = it->path();
  }
  return best;
}

}  // namespace

bool ModelPaths::FromRoot(const std::filesystem::path& root, ModelPaths* paths,
                          std::string* error) {
  ModelPaths p;
  p.network = FindWeights(root, "mmaudio_large_44k_v2");
  p.vae = FindWeights(root, "mmaudio_vae_44k");
  p.synchformer = FindWeights(root, "mmaudio_synchformer");
  p.clip = FindWeights(root, "apple_DFN5B-CLIP-ViT-H-14-384");
  p.vocoder = root / "bigvgan_v2_44khz_128band_512x.safetensors";
  p.tokenizer = root / "tokenizer";
  const std::pair<const char*, const std::filesystem::path*> required[] = {
      {"mmaudio_large_44k_v2_*.safetensors", &p.network},
      {"mmaudio_vae_44k_*.safetensors", &p.vae},
      {"mmaudio_synchformer_*.safetensors", &p.synchformer},
      {"apple_DFN5B-CLIP-ViT-H-14-384_*.safetensors", &p.clip},
      {"bigvgan_v2_44khz_128band_512x.safetensors", &p.vocoder},
      {"tokenizer/vocab.json", &p.tokenizer}};
  for (const auto& [what, path] : required) {
    std::error_code ec;
    if (path->empty() || !std::filesystem::exists(*path, ec))
      return Fail(error, "MMAudio model directory " + root.string() +
                             " lacks " + what);
  }
  *paths = std::move(p);
  return true;
}

std::unique_ptr<FoleyPipeline> FoleyPipeline::Load(const ModelPaths& paths,
                                                   std::string* error) {
  std::unique_ptr<FoleyPipeline> pipeline(new FoleyPipeline());
  const Precision precision = DefaultPrecision();
  if (!pipeline->tokenizer_.Load(paths.tokenizer, error))
    return nullptr;
  pipeline->clip_ = ClipEncoder::Load(paths.clip, precision, error);
  if (!pipeline->clip_)
    return nullptr;
  pipeline->synchformer_ =
      Synchformer::Load(paths.synchformer, precision, error);
  if (!pipeline->synchformer_)
    return nullptr;
  pipeline->network_ = FlowNetwork::Load(paths.network, precision, error);
  if (!pipeline->network_)
    return nullptr;
  pipeline->vae_ = VaeDecoder::Load(paths.vae, precision, error);
  if (!pipeline->vae_)
    return nullptr;
  pipeline->vocoder_ = BigVgan::Load(paths.vocoder, precision, error);
  if (!pipeline->vocoder_)
    return nullptr;
  return pipeline;
}

std::size_t FoleyPipeline::weight_bytes() const {
  return clip_->weight_bytes() + synchformer_->weight_bytes() +
         network_->weight_bytes() + vae_->weight_bytes() +
         vocoder_->weight_bytes();
}

bool FoleyPipeline::Generate(
    const FoleyRequest& request, std::vector<float>* audio,
    FoleyTelemetry* telemetry,
    const std::function<bool(double, std::string_view)>& progress,
    std::string* error) {
  FoleyTelemetry local;
  FoleyTelemetry& t = telemetry != nullptr ? *telemetry : local;
  const auto report = [&](double fraction, std::string_view stage) {
    return !progress || progress(fraction, stage);
  };
  if (!(request.duration > 0.0) || request.duration > 600.0)
    return Fail(error, "duration must be in (0, 600] seconds");
  if (request.steps < 1 || request.steps > 200)
    return Fail(error, "steps must be 1..200");
  const SequenceLengths lengths = LengthsFor(request.duration);
  if (lengths.clip < 1 || lengths.sync < 8)
    return Fail(error, "duration is too short for MMAudio (at least 1 s)");
  t.duration = request.duration;
  const auto start = Clock::now();
  if (!report(0.0, "features"))
    return Fail(error, "cancelled");

  // CLIP features: per frame, or one image repeated, or the empty ones.
  std::vector<float> clip_features;
  bool have_clip = false;
  if (request.clip_frames > 0 && !request.mask_away_clip) {
    if (!clip_->EncodeImages(request.clip_pixels, request.clip_frames,
                             &clip_features, error))
      return false;
    if (request.image_input) {
      const std::vector<float> one(clip_features.begin(),
                                   clip_features.begin() +
                                       ClipEncoder::kFeatures);
      clip_features.clear();
      for (int i = 0; i < lengths.clip; ++i)
        clip_features.insert(clip_features.end(), one.begin(), one.end());
    } else if (request.clip_frames != lengths.clip) {
      return Fail(error, "got " + std::to_string(request.clip_frames) +
                             " CLIP frames for " +
                             std::to_string(lengths.clip) + " positions");
    }
    have_clip = true;
  }
  std::vector<float> sync_features;
  bool have_sync = false;
  if (request.sync_frames > 0 && !request.image_input) {
    if (!synchformer_->Encode(request.sync_pixels, request.sync_frames,
                              &sync_features, error))
      return false;
    const int rows =
        static_cast<int>(sync_features.size() / Synchformer::kWidth);
    if (rows != lengths.sync)
      return Fail(error, "got " + std::to_string(rows) +
                             " Synchformer features for " +
                             std::to_string(lengths.sync) + " positions");
    have_sync = true;
  }
  std::vector<float> text, negative_text;
  try {
    if (!clip_->EncodeText(tokenizer_.Encode(request.prompt), &text, error) ||
        !clip_->EncodeText(tokenizer_.Encode(request.negative_prompt),
                           &negative_text, error))
      return false;
  } catch (const std::exception& e) {
    return Fail(error, std::string("tokenizer: ") + e.what());
  }
  FlowNetwork::Conditions positive, negative;
  if (!network_->Prepare(lengths, have_clip ? &clip_features : nullptr,
                         have_sync ? &sync_features : nullptr, &text,
                         &positive, error) ||
      !network_->Prepare(lengths, nullptr, nullptr, &negative_text, &negative,
                         error))
    return false;
  t.features_seconds = Since(start);

  const auto sampling = Clock::now();
  const std::vector<float> noise = GaussianNoise(
      request.seed,
      static_cast<std::size_t>(lengths.latent) * FlowNetwork::kLatentDim);
  std::vector<float> latent;
  if (!network_->Sample(
          noise, request.steps, request.cfg_strength, positive, negative,
          &latent,
          [&](int step, int steps) {
            return report(0.05 + 0.85 * step / steps, "sampling");
          },
          error))
    return false;
  t.sampling_seconds = Since(sampling);

  const auto decode = Clock::now();
  if (!report(0.9, "decoding"))
    return Fail(error, "cancelled");
  std::vector<float> mel;
  if (!vae_->Decode(latent, lengths.latent, &mel, error) ||
      !vocoder_->Vocode(mel, 2 * lengths.latent, audio, error))
    return false;
  t.decode_seconds = Since(decode);
  report(1.0, "done");
  return true;
}

}  // namespace gufo::models::mmaudio
