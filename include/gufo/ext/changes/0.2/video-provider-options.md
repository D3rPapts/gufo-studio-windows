# Video provider startup options and helper processes (gufo-studio-windows)

Video (`src/ext/video.hpp`): `VideoRunnerOptions`, `VideoRunnerOptionsFactory`
and `VideoProvider::create_with_options` (preferred over `create`) pass the
provider's own `serve video` options, in command-line order and possibly
repeated (`--lora NAME=PATH` twice), as name/value pairs without dashes. The
host forwards them only to a provider that sets `create_with_options` and fails
startup otherwise; a provider that returns no runner from it fails startup with
its error. MiniMax H3 accepts `lora` (`NAME=PATH[,steps=N][,strength=S]`) and
`vision-projector` (Qwen3-VL tower GGUF), as gufo-studio-windows 1.4.9.

`VideoChildEntry` and `VideoProvider::child_entry` let a provider run an
internal helper process of the same executable: `gufo video <args...>` hands
`args` to the provider (MiniMax H3: `--h3-prompt-child <request> <response>`,
vision + text encoding in a short-lived child on Windows). The entry returns
the exit code, or -1 when it does not claim the arguments.

Host: MiniMax H3 keeps its 1.4.9 job contract with its own `prepare_request`
(preset-selecting model names `minimax-h3-fast` etc., PPM diagnostics,
`exact-WxH` canvases such as 832x512); other prepared providers keep theirs.
Additive; providers that set neither member behave as before.
