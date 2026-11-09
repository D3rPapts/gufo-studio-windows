#ifndef GUFO_MODELS_WAN22_PREVIEW_HOOK_HPP_
#define GUFO_MODELS_WAN22_PREVIEW_HOOK_HPP_

// The pipelines' live-preview hook is the extension API's PreviewSink
// (changes/0.2/live-previews.md): the runner hands the host's sink through.
#include <gufo/ext/image.hpp>

namespace gufo::models {

using LatentPreview = ext::LatentPreview;
using PreviewHook = ext::PreviewSink;

}  // namespace gufo::models
#endif
