// Written with AI assistance (Claude Code).
// The Wan family as gufo video providers. This translation unit registers
// the CPU parts (probing, request parsing, validation); provider.cpp
// re-registers the same providers with HIP runners.
#include "src/models/wan22/extension.hpp"

#include <filesystem>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/video.hpp>
#include <string>
#include <string_view>

#include "src/models/wan22/request.hpp"

namespace gufo::models::wan22 {
namespace {

template <Family F>
bool Probe(std::string_view model_root) {
  try {
    const std::filesystem::path root(model_root);
    switch (F) {
      case Family::kWan22:
        return IsWan22ModelDirectory(root);
      case Family::kWan21Uni3C:
        return IsWan21CameraDirectory(root);
      case Family::kWan21Vace:
        return IsWan21VaceDirectory(root);
      case Family::kWan21Talk:
        return IsWan21TalkDirectory(root);
    }
  } catch (...) {
  }
  return false;
}

template <Family F>
bool Resolve(std::string_view model_root, std::string_view, std::string* error) {
  if (Probe<F>(model_root))
    return true;
  if (error != nullptr)
    *error = "not a " + std::string(ModelId(F)) +
             " model directory: " + std::string(model_root);
  return false;
}

template <Family F>
std::optional<int> Duration(std::string_view seconds, std::string* error) {
  return ResolveDurationFrames(F, seconds, error);
}

template <Family F>
bool Validate(const ext::VideoParameters& parameters, std::string* error) {
  return ValidateParameters(F, parameters, error);
}

template <Family F>
bool Prepare(const ext::VideoRequestInput& input,
             ext::VideoPreparedRequest* prepared,
             ext::VideoRequestError* error) {
  return PrepareRequest(F, input, prepared, error);
}

std::string Telemetry(const ext::VideoTelemetry& telemetry) {
  return telemetry.extra_json.empty() ? "{}" : telemetry.extra_json;
}

template <Family F>
ext::VideoProvider Make(ext::VideoRunnerFactory create) {
  return {
      .id = std::string(ModelId(F)),
      .probe = &Probe<F>,
      .resolve = &Resolve<F>,
      .create = create,
      .resolve_preset = nullptr,
      .resolve_duration_frames = &Duration<F>,
      .validate_parameters = &Validate<F>,
      .parameters_json = &ParametersJson,
      .telemetry_json = &Telemetry,
      .prepare_request = &Prepare<F>,
      // Above the generic probes (MiniMax H3 accepts any model_index.json,
      // which a Wan 2.2 Diffusers directory also has).
      .probe_priority = 10,
  };
}

}  // namespace

ext::VideoProvider MakeVideoProvider(Family family,
                                     ext::VideoRunnerFactory create) {
  switch (family) {
    case Family::kWan22:
      return Make<Family::kWan22>(create);
    case Family::kWan21Uni3C:
      return Make<Family::kWan21Uni3C>(create);
    case Family::kWan21Vace:
      return Make<Family::kWan21Vace>(create);
    case Family::kWan21Talk:
      break;
  }
  return Make<Family::kWan21Talk>(create);
}

}  // namespace gufo::models::wan22

void wan22_register(gufo::ext::Registry& registry) {
  using gufo::models::wan22::Family;
  for (const Family family : {Family::kWan22, Family::kWan21Uni3C,
                              Family::kWan21Vace, Family::kWan21Talk}) {
    registry.AddVideoProvider(
        gufo::models::wan22::MakeVideoProvider(family, nullptr));
  }
}
