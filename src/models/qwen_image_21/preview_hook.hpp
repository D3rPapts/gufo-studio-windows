#ifndef GUFO_MODELS_QWEN_IMAGE_21_PREVIEW_HOOK_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_PREVIEW_HOOK_HPP_

// The model's live-preview hook is the extension API's PreviewSink
// (changes/0.2/live-previews.md): the provider hands the host's sink through.
#include <gufo/ext/image.hpp>

namespace gufo::models {

using LatentPreview = ext::LatentPreview;
using PreviewHook = ext::PreviewSink;

}  // namespace gufo::models
#endif
