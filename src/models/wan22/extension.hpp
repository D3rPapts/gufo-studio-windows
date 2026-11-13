// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_WAN22_EXTENSION_HPP_
#define GUFO_MODELS_WAN22_EXTENSION_HPP_

#include <gufo/ext/video.hpp>

#include "src/models/wan22/request.hpp"

namespace gufo::models::wan22 {

/// One Wan video provider. Its id is the served model name
/// (e.g. "wan2.1-infinitetalk-14b"); extension.json lists the provider
/// under a dot-free id ("wan21-infinitetalk-14b") because metadata ids may
/// not contain dots, with the served name in "model_ids". `create` nullptr
/// registers it without a runner (CPU builds: requests parse, generation is
/// refused).
[[nodiscard]] ext::VideoProvider MakeVideoProvider(
    Family family, ext::VideoRunnerFactory create);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_EXTENSION_HPP_
