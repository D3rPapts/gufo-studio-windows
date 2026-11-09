#include <gufo/ext/generated/qwen_image_21_constants.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/registry.hpp>

#include <filesystem>
#include <mutex>
#include <sstream>

#include "model.hpp"
#include "probe.hpp"
#include "request.hpp"
#include "src/core/logging/logging.hpp"

namespace gufo::models::qwen_image_21 {

#if defined(ENGINE_ENABLE_HIP)
namespace {

void Log(const std::string& line) {
  server::Logger::Log(server::LogLevel::kInfo, "image", line);
}

}  // namespace

/// The 1.4.9 Qwen-Image-2.1 service on the 2.0 image extension API:
/// startup options --lora / --lora-default / --controlnet
/// (ext::ImageRunnerOptions) and the request extension ("gufo", "loras",
/// named image parts) turned into the model's Request by BuildRequest.
class QwenImage21RunnerImpl : public ext::ImageRunner {
public:
  QwenImage21RunnerImpl(std::unique_ptr<Model> model, StartupOptions startup)
      : model_(std::move(model)), startup_(std::move(startup)) {
    model_->SetLog(&Log);
    if (!startup_.controlnet.empty()) {
      if (!std::filesystem::is_regular_file(startup_.controlnet))
        throw std::invalid_argument("--controlnet file not found: " +
                                    startup_.controlnet);
      const auto layout = model_->SetControlnet(startup_.controlnet);
      Log("event=controlnet_registered file=" +
          std::filesystem::path(startup_.controlnet).filename().string() +
          " blocks=" + std::to_string(layout.blocks) +
          " mib=" + std::to_string(layout.bytes >> 20));
      state_.controlnet = true;
    }
    for (const auto& [name, path] : startup_.loras) {
      const auto& info = infos_.emplace_back(model_->AddLora(name, path));
      state_.loras.insert(name);
      Log("event=lora_registered name=" + name + " file=" +
          std::filesystem::path(path).filename().string() +
          " format=" + info.format + " targets=" +
          std::to_string(info.targets) + " rank=" +
          std::to_string(info.min_rank) + '-' +
          std::to_string(info.max_rank));
    }
    state_.default_loras = startup_.default_loras;
  }

  ext::ImageResult Generate(
      const ext::ImageRequest& request,
      const ext::CancellationCheck& is_cancelled) override {
    const Request req = BuildRequest(request, state_);
    Result result;
    try {
      result = model_->Generate(req, is_cancelled);
    } catch (const InsufficientMemory& error) {
      throw ext::ImageResourceError(error.what());
    }
    std::ostringstream details;
    details << "event=generated_extension guidance=" << req.guidance
            << " strength=" << req.strength
            << " init_image=" << (req.init_image ? 1 : 0)
            << " mask=" << (req.mask ? 1 : 0);
    if (!req.regions.empty())
      details << " regions=" << req.regions.size()
              << " base_strength=" << req.base_strength
              << " region_rows=" << result.metrics.region_condition_rows
              << " region_passes=" << result.metrics.region_passes;
    if (req.mask)
      details << " mask_composite=" << (req.mask_composite ? 1 : 0)
              << " mask_feather=" << req.mask_feather
              << " mask_blur=" << req.mask_blur;
    if (!startup_.loras.empty())
      details << " loras=" << LoraList(req.loras)
              << " lora_ms=" << result.metrics.lora_ms;
    if (req.control)
      details << " control=1 control_strength=" << req.control->strength
              << " control_range=" << req.control->start << '-'
              << req.control->end
              << " control_inpaint=" << (req.control->inpaint ? 1 : 0)
              << " control_turbo=" << (model_->sampling().IsTurbo() ? 1 : 0);
    details << " condition_rows=" << result.metrics.condition_rows
            << " negative_rows=" << result.metrics.negative_condition_rows
            << " released_text_encoder="
            << (result.metrics.released_text_encoder ? 1 : 0);
    Log(details.str());
    return ext::ImageResult{
        .image =
            ext::ImageBuffer{
                .width = result.image.width,
                .height = result.image.height,
                .rgba = std::move(result.image.rgba),
            },
        .metrics =
            ext::ImageMetrics{
                .prompt_ms = result.metrics.prompt_ms,
                .denoise_ms = result.metrics.denoise_ms,
                .vae_ms = result.metrics.vae_ms,
                .total_ms = result.metrics.total_ms,
                .resident_weight_bytes = result.metrics.resident_weight_bytes,
                .steps = result.metrics.steps,
            },
    };
  }

  std::string Query(std::string_view topic) const override {
    if (topic != "loras")
      return {};
    return LorasJson(startup_.loras, infos_, startup_.default_loras,
                     model_->ActiveLoras());
  }

private:
  std::unique_ptr<Model> model_;
  StartupOptions startup_;
  ProviderState state_;
  std::vector<LoraInfo> infos_;
};

std::unique_ptr<ext::ImageRunner> CreateQwenImage21RunnerWithOptions(
    const ext::ImageRunnerOptions& options, std::string* error) {
  try {
    auto startup = ParseStartupOptions(options.options);
    auto model = std::make_unique<Model>(options.model_root);
    return std::make_unique<QwenImage21RunnerImpl>(std::move(model),
                                                   std::move(startup));
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = e.what();
    return nullptr;
  }
}

std::unique_ptr<ext::ImageRunner> CreateQwenImage21Runner(
    const std::string& model_root, std::string* error) {
  return CreateQwenImage21RunnerWithOptions(
      ext::ImageRunnerOptions{.model_root = model_root, .options = {}}, error);
}
#endif

}  // namespace gufo::models::qwen_image_21

void qwen_image_21_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen_image_21::kProvider_qwen_image_21);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::qwen_image_21;
  registry.AddImageProvider({
      .id = "qwen-image-21",
      .probe = &ProbeQwenImage21,
      .resolve = &ResolveQwenImage21Artifacts,
      .create = &CreateQwenImage21Runner,
      .create_with_options = &CreateQwenImage21RunnerWithOptions,
  });
#endif
}
