#ifndef GUFO_MODELS_MINIMAX_H3_PREVIEW_HOOK_HPP_
#define GUFO_MODELS_MINIMAX_H3_PREVIEW_HOOK_HPP_

// The denoiser's live-preview hook is the extension API's PreviewSink
// (gufo-studio-windows, changes/0.2/live-previews.md): the provider hands the
// host's sink through.
#include <gufo/ext/image.hpp>

namespace gufo::models {

using LatentPreview = ext::LatentPreview;
using PreviewHook = ext::PreviewSink;

}  // namespace gufo::models
#endif
