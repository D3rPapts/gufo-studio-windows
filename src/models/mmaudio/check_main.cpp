// gufo-mmaudio-check: compares each MMAudio component against the FP32
// references of tools/models/mmaudio/reference.py.
// Written with AI assistance (Claude Code).
//
//   gufo-mmaudio-check --model ROOT --reference mmaudio_reference.safetensors
//
// Prints max |a - b|, the reference's max |b| and the relative RMS error per
// component; exits 1 when a component exceeds its tolerance.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "src/models/mmaudio/pipeline.hpp"
#include "src/models/mmaudio/vocoder.hpp"

namespace mm = gufo::models::mmaudio;

namespace {

bool Compare(const char* name, const std::vector<float>& got,
             const std::vector<float>& want, double tolerance) {
  if (got.size() != want.size()) {
    std::printf("%-14s SIZE MISMATCH %zu vs %zu\n", name, got.size(),
                want.size());
    return false;
  }
  double max_diff = 0, max_ref = 0, err = 0, ref = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    max_diff = std::max(max_diff, std::fabs(d));
    max_ref = std::max(max_ref, std::fabs(static_cast<double>(want[i])));
    err += d * d;
    ref += static_cast<double>(want[i]) * want[i];
  }
  const double rel = std::sqrt(err / std::max(ref, 1e-30));
  const bool ok = rel <= tolerance && std::isfinite(rel);
  std::printf("%-14s max_abs_diff %.3e  max_ref %.3e  rel_rms %.3e  %s\n",
              name, max_diff, max_ref, rel, ok ? "ok" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  if (!args.count("--model") || !args.count("--reference")) {
    std::fprintf(stderr,
                 "usage: gufo-mmaudio-check --model ROOT --reference F\n");
    return 2;
  }
  std::string error;
  mm::ModelPaths paths;
  mm::TensorFile ref;
  if (!mm::ModelPaths::FromRoot(args["--model"], &paths, &error) ||
      !ref.Open(args["--reference"], &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const auto read = [&](const std::string& name) {
    std::vector<float> v;
    if (!ref.Read(name, &v, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      std::exit(1);
    }
    return v;
  };
  const auto ints = [](const std::vector<float>& v) {
    std::vector<int> out;
    for (const float x : v)
      out.push_back(static_cast<int>(std::lround(x)));
    return out;
  };
  // Half weights are expected to land near 1e-2 relative error, FP32 ones
  // (GUFO_MMAUDIO_F32=1) near 1e-4.
  const double tol = mm::DefaultPrecision() == mm::Precision::kF32 ? 2e-3 : 3e-2;
  bool ok = true;

  // Pure host parts first.
  ok &= Compare("kaiser", mm::KaiserSincFilter(0.25, 0.3, 12),
                read("kaiser.filter"), 1e-6);
  mm::ClipTokenizer tokenizer;
  if (!tokenizer.Load(paths.tokenizer, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  for (int i = 0; ref.Has("tokenizer." + std::to_string(i) + ".ids"); ++i) {
    // The prompts are stored as UTF-8 bytes in float tensors.
    std::string text;
    for (const float c : read("tokenizer." + std::to_string(i) + ".text"))
      text.push_back(static_cast<char>(static_cast<int>(c)));
    std::vector<float> got;
    for (const int id : tokenizer.Encode(text))
      got.push_back(static_cast<float>(id));
    ok &= Compare(("tokens." + std::to_string(i)).c_str(), got,
                  read("tokenizer." + std::to_string(i) + ".ids"), 0.0);
  }

  const mm::Precision precision = mm::DefaultPrecision();
  std::vector<float> out;
  {
    auto clip = mm::ClipEncoder::Load(paths.clip, precision, &error);
    if (!clip) {
      std::fprintf(stderr, "clip: %s\n", error.c_str());
      return 1;
    }
    if (!clip->EncodeText(ints(read("clip_text.tokens")), &out, &error)) {
      std::fprintf(stderr, "clip text: %s\n", error.c_str());
      return 1;
    }
    ok &= Compare("clip_text", out, read("clip_text.out"), tol);
    const std::vector<float> pixels = read("clip_image.pixels");
    const int frames =
        static_cast<int>(pixels.size() / (3 * 384 * 384));
    if (!clip->EncodeImages(pixels, frames, &out, &error)) {
      std::fprintf(stderr, "clip image: %s\n", error.c_str());
      return 1;
    }
    ok &= Compare("clip_image", out, read("clip_image.out"), tol);
  }
  {
    auto sync = mm::Synchformer::Load(paths.synchformer, precision, &error);
    if (!sync) {
      std::fprintf(stderr, "synchformer: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> frames = read("sync.frames");
    if (!sync->Encode(frames, static_cast<int>(frames.size() / (3 * 224 * 224)),
                      &out, &error)) {
      std::fprintf(stderr, "synchformer: %s\n", error.c_str());
      return 1;
    }
    ok &= Compare("synchformer", out, read("sync.out"), tol);
  }
  {
    auto net = mm::FlowNetwork::Load(paths.network, precision, &error);
    if (!net) {
      std::fprintf(stderr, "network: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> lengths_v = read("net.lengths");
    mm::SequenceLengths lengths;
    lengths.latent = static_cast<int>(lengths_v[0]);
    lengths.clip = static_cast<int>(lengths_v[1]);
    lengths.sync = static_cast<int>(lengths_v[2]);
    const std::vector<float> clip_f = read("net.clip_f");
    const std::vector<float> sync_f = read("net.sync_f");
    const std::vector<float> text_f = read("net.text_f");
    const std::vector<float> latent = read("net.latent");
    const float t = read("net.t")[0];
    mm::FlowNetwork::Conditions conditions;
    mm::DeviceBuffer x, flow;
    if (!net->Prepare(lengths, &clip_f, &sync_f, &text_f, &conditions,
                      &error) ||
        !x.Allocate(latent.size() * 4, &error) ||
        !flow.Allocate(latent.size() * 4, &error) ||
        hipMemcpy(x.get(), latent.data(), latent.size() * 4,
                  hipMemcpyHostToDevice) != hipSuccess ||
        !net->PredictFlow(x.f32(), t, conditions, flow.f32(), &error) ||
        hipStreamSynchronize(net->stream()) != hipSuccess) {
      std::fprintf(stderr, "network: %s\n", error.c_str());
      return 1;
    }
    out.resize(latent.size());
    (void)hipMemcpy(out.data(), flow.get(), out.size() * 4,
                    hipMemcpyDeviceToHost);
    ok &= Compare("network_flow", out, read("net.flow"), tol);
  }
  {
    auto vae = mm::VaeDecoder::Load(paths.vae, precision, &error);
    if (!vae) {
      std::fprintf(stderr, "vae: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> latent = read("vae.latent");
    if (!vae->Decode(latent, static_cast<int>(latent.size() / 40), &out,
                     &error)) {
      std::fprintf(stderr, "vae: %s\n", error.c_str());
      return 1;
    }
    ok &= Compare("vae_mel", out, read("vae.mel"), tol);
  }
  {
    auto vocoder = mm::BigVgan::Load(paths.vocoder, precision, &error);
    if (!vocoder) {
      std::fprintf(stderr, "bigvgan: %s\n", error.c_str());
      return 1;
    }
    const std::vector<float> mel = read("bigvgan.mel");
    if (!vocoder->Vocode(mel, static_cast<int>(mel.size() / 128), &out,
                         &error)) {
      std::fprintf(stderr, "bigvgan: %s\n", error.c_str());
      return 1;
    }
    ok &= Compare("bigvgan_audio", out, read("bigvgan.audio"), tol);
  }
  std::printf("%s\n", ok ? "ALL OK" : "SOME FAILED");
  return ok ? 0 : 1;
}
