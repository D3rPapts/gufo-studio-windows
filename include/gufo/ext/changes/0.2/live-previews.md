# Live previews (gufo-studio-windows)

Image (`src/ext/image.hpp`): add `LatentPreview` and `PreviewSink`, the
`ImageRequest::preview` sink and `ImageProvider::preview_latent`. When a
request asks for a live preview (`"gufo": {"preview": {"id": ...}}`), the host
sets `ImageRequest::preview`; a provider that sets `preview_latent` (the latent
space for the host's latent->RGB map: `"qwen_image_21"` or `"flux2_patched"`)
asks `PreviewSink::Wants(step, steps)` once per denoising step and, when it is
true, hands the sampler's denoised estimate to `publish` (FP32, valid only
during the call). The host renders JPEG / animated WebP previews and serves
them at `GET /v1/previews/{id}[/events]` like gufo-studio-windows 1.x.
Previews never change the result. Additive: providers that leave
`preview_latent` null publish nothing, and an empty sink wants nothing.

Video (`src/ext/video.hpp`, which now includes `src/ext/image.hpp`): add
`VideoRequest::preview` and `VideoProvider::preview_latent` (`"wan22"`,
`"wan21"` or `"minimax_h3"`) / `preview_fps`. The host publishes each job's
previews under the job id (options from `"gufo": {"preview": {"always",
"every", "max_side"}}`, defaults every second step, 320 px) for providers
that set `preview_latent`. Wan 2.2 TI2V and the Wan 2.1 families (Uni3C,
VACE, InfiniteTalk) publish; MiniMax H3 does not yet.
