# Image request extension and provider startup options (gufo-studio-windows)

Image (`src/ext/image.hpp`): `ImageRequest` gains `extension_json`,
`named_images` and `size_auto`. The host no longer drops what it does not
interpret: the request's `"gufo"` object (JSON body, a multipart `gufo` field,
or multipart form fields `gufo.<name>`) and its `"loras"` member reach the
provider as one JSON object text; multipart image parts `init_image`, `mask`,
`control_image` and `region_image[]` arrive decoded in request order as
`named_images`. `size_auto` tells the provider that the caller asked for size
`auto`, so it may derive the canvas from its own inputs (a start or control
image). Providers that ignore the three members behave as before; a request
whose extension a provider does not accept is rejected by that provider with
`std::invalid_argument` (HTTP 400).

`ImageResourceError` (a `std::runtime_error`) from `Generate` means the
request does not fit into the memory left now; the host answers HTTP 503
instead of 500.

`ImageRunner::Query(topic)` returns provider state as JSON text for host
endpoints, empty for unknown topics. `"loras"` serves `GET /sdapi/v1/loras`.

Startup options reach the provider through the upstream mechanism of
`image-provider-options.md`: `ImageRunnerOptions::options` is the
`ParsedOptions` of the options declared in `extension.json` for `serve image`
(keys with dashes, typed values). A manifest option cannot repeat, so
Qwen-Image-2.1 declares `--lora NAME=PATH[;NAME=PATH...]`,
`--lora-default NAME:SCALE[;...]` (string, `;`-separated lists) and
`--controlnet FILE` (path).
