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
