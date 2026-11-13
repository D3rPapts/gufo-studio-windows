// gufo-studio-windows: command-line side of the media services without an
// upstream capability: `gufo serve music | foley | upscale` and the
// `gufo serve video --rife-model / --upscale-model` post-steps. serve.cpp
// only dispatches here, so these parts stay in one folder.
// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_MEDIA_MEDIA_SERVE_HPP_
#define GUFO_SERVER_MEDIA_MEDIA_SERVE_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/media/media_service.hpp"
#include "src/cli/serve/media/upscale_api.hpp"
#include "src/cli/serve/media/video_postprocess.hpp"

namespace gufo::server {

/// "music", "foley", "upscale", "omni" or "objekt".
[[nodiscard]] bool IsMediaServeCommand(std::string_view subcommand);
/// Help lines for `gufo serve --help` (one per command, newline-terminated).
[[nodiscard]] std::string MediaServeCommandsHelp();
/// The parser description of `gufo serve <subcommand>`.
[[nodiscard]] std::string MediaServeDescription(std::string_view subcommand);

struct MediaServeArgs {
  std::filesystem::path model;  // music / foley directory, omni text GGUF
  std::string name;             // --served-model-name
  // music: --dit FILE|turbo|sft|base, --lm FILE|none|0.6b|1.7b|4b, --variant.
  std::string music_dit, music_lm, music_variant;
  UpscaleServeOptions upscale;  // upscale
  // omni (Qwen3-Omni-30B-A3B-Captioner): projector GGUF and limits.
  std::filesystem::path mmproj;
  std::size_t context{8192};
  std::size_t max_tokens{1024};
  // objekt (TripoSG / TRELLIS.2 image -> 3D, feat/trellis2): job storage,
  // result TTL, the TRELLIS.2 weights directory and its texture stage.
  std::filesystem::path objekt_root;
  double objekt_ttl_seconds{86400};
  std::filesystem::path trellis2_dir;
  bool trellis2_texture{false};
};

/// Registers the model options of `gufo serve <subcommand>`.
void AddMediaServeOptions(std::string_view subcommand, cli::ArgParser& parser,
                          MediaServeArgs* args);
/// Checks the parsed options (paths exist, device names) and fills defaults.
[[nodiscard]] bool ValidateMediaServeArgs(std::string_view subcommand,
                                          MediaServeArgs* args,
                                          std::string* error);
/// The file or directory named in the load log.
[[nodiscard]] std::filesystem::path MediaServeArtifact(
    std::string_view subcommand, const MediaServeArgs& args);
/// Loads the model. nullptr and *error on failure; *details for the load log.
[[nodiscard]] std::shared_ptr<MediaService> LoadMediaService(
    std::string_view subcommand, const MediaServeArgs& args,
    std::string* details, std::string* error);

struct VideoPostProcessArgs {
  std::filesystem::path rife_model;
  std::vector<std::pair<std::string, std::filesystem::path>> upscale_models;
};

/// `--rife-model SAFETENSORS` and `--upscale-model [NAME=]SAFETENSORS`.
void AddVideoPostProcessOptions(cli::ArgParser& parser,
                                VideoPostProcessArgs* args);
/// RIFE / Real-ESRGAN post-processors for VideoJobServiceOptions. False with
/// *error when a file is missing.
[[nodiscard]] bool BuildVideoPostProcessors(const VideoPostProcessArgs& args,
                                            VideoPostProcessors* tools,
                                            std::string* error);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_MEDIA_SERVE_HPP_
