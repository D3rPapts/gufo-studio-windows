// Written with AI assistance (Claude Code).
// HIP registration of the Wan video providers: the CPU providers from
// extension.cpp plus a runner that decodes the prepared request and runs the
// family's pipeline.
#include <filesystem>
#include <gufo/ext/generated/wan22_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/video.hpp>
#include <memory>
#include <string>
#include <utility>

#include "src/models/wan22/extension.hpp"
#include "src/models/wan22/request.hpp"
#include "src/models/wan22/runners.hpp"

namespace gufo::models::wan22 {
namespace {

class WanRunner : public ext::VideoRunner {
public:
  WanRunner(Family family, std::filesystem::path model_root)
      : family_(family), model_root_(std::move(model_root)) {}

  bool Generate(const ext::VideoRequest& request,
                const ext::CancellationCheck& is_cancelled,
                ext::VideoProgress progress, void* progress_opaque,
                ext::VideoTelemetry* telemetry, std::string* error) override {
    JobRequest job;
    if (!BuildJobRequest(family_, request, &job, error))
      return false;
    if (job.model_root.empty())
      job.model_root = model_root_;
    // Live preview: the host's sink (changes/0.2/live-previews.md).
    job.preview = request.preview;
    JobCallbacks callbacks;
    callbacks.progress = [&](const std::string& phase, int done, int total) {
      if (progress != nullptr)
        progress(phase, done, total, progress_opaque);
    };
    callbacks.keep_going = [&] { return !is_cancelled || !is_cancelled(); };
    switch (family_) {
      case Family::kWan22:
        return RunWan22Job(job, callbacks, telemetry, error);
      case Family::kWan21Uni3C:
        return RunWan21Job(job, callbacks, telemetry, error);
      case Family::kWan21Vace:
        return RunWan21VaceJob(job, callbacks, telemetry, error);
      case Family::kWan21Talk:
        return RunWan21TalkJob(job, callbacks, telemetry, error);
    }
    if (error != nullptr)
      *error = "unknown Wan family";
    return false;
  }

private:
  Family family_;
  std::filesystem::path model_root_;
};

template <Family F>
std::unique_ptr<ext::VideoRunner> Create(const std::string& model_root,
                                         std::string*) {
  return std::make_unique<WanRunner>(F, model_root);
}

}  // namespace
}  // namespace gufo::models::wan22

namespace generated = gufo::ext::generated::extension_wan22;
using gufo::models::wan22::Family;
using gufo::models::wan22::MakeVideoProvider;

void wan22_ti2v_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(generated::kProvider_wan22_ti2v_5b);
  registry.AddVideoProvider(MakeVideoProvider(
      Family::kWan22, &gufo::models::wan22::Create<Family::kWan22>));
}

void wan21_uni3c_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(generated::kProvider_wan21_uni3c);
  registry.AddVideoProvider(MakeVideoProvider(
      Family::kWan21Uni3C, &gufo::models::wan22::Create<Family::kWan21Uni3C>));
}

void wan21_vace_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(generated::kProvider_wan21_vace_14b);
  registry.AddVideoProvider(MakeVideoProvider(
      Family::kWan21Vace, &gufo::models::wan22::Create<Family::kWan21Vace>));
}

void wan21_talk_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(generated::kProvider_wan21_infinitetalk_14b);
  registry.AddVideoProvider(MakeVideoProvider(
      Family::kWan21Talk, &gufo::models::wan22::Create<Family::kWan21Talk>));
}
