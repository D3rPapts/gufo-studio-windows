# gufo-studio-windows: Gufo as a creative AI studio on Windows 11 for AMD Strix Halo

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

> **Download [1.2.0](https://github.com/D3rPapts/gufo-studio-windows/releases/tag/v1.2.0)**
> (zip, branch `release/1.2`): text, images, speech, music and video from one
> OpenAI-compatible port on Windows 11, plus a ComfyUI-compatible API. Unpack
> it, edit `examples\router.ini`, run `start-router.cmd`; models load on the
> first request. The release notes list what was tested.
> Previous: [1.1.1](https://github.com/D3rPapts/gufo-studio-windows/releases/tag/v1.1.1), [1.1.0](https://github.com/D3rPapts/gufo-studio-windows/releases/tag/v1.1.0), [1.0.0](https://github.com/D3rPapts/gufo-studio-windows/releases/tag/v1.0.0).
> This repository was called `gufo_windows` until 31 October 2026; old links still work.

### Focus

gufo-studio-windows is built for one setup first: two GMKtec EVO-X2 (AMD Strix Halo,
64 GB) on Windows 11 that share the load for video and creative work. The
goal is one tool that covers everything a film pipeline needs: text, images,
speech, music and video. New features land here when that setup needs them,
and the measurements in the release notes come from it. Sharing the load
between the two machines is the next big step; it is not in 1.1.1 yet.
Other Strix Halo machines (64 and 128 GB) should work too, and we help in
the issues, but only this setup is tested here.

_(AI-assisted text, written with Claude Code.)_

### New in 1.1.1

- Qwen-Image 2.1 Turbo (fixed 8 steps, about 4.7x faster than the base at
  40 steps; weights are non-commercial only) as its own model, also chosen
  by ComfyUI workflows whose `UNETLoader` file name contains `turbo`.
- LoRA adapters for Qwen-Image 2.1 (`--lora NAME=PATH`, `"loras"` per
  request, ComfyUI `LoraLoader` / `LoraLoaderModelOnly`).
- Qwen3.8-27B with DFlash2 answers the first token about 0.7 s sooner on
  Windows (film JSON request 0.96 s instead of 1.70 s).

### New in 1.1

- The router is part of `gufo.exe` (`gufo router`); models load on demand and
  unload when idle; `POST /models/load` and `/models/unload`; an optional
  memory guard keeps several models loaded while they fit.
- Live previews for Qwen-Image, MOSS-Image, Wan 2.2 and MiniMax H3
  (`/v1/previews/<id>/events`), shown on the `/bild/` and `/video/` pages and
  at the KSampler node in ComfyUI.
- ACE-Step 1.5 music (`/v1/audio/music`, page `/musik/`), Wan 2.1 + Uni3C
  camera moves, RIFE to any frame rate (24 to 60 fps) and Real-ESRGAN
  upscaling for video delivery.
- A ComfyUI-compatible API in the router (`/prompt`, `/history`, `/view`,
  `/ws`) that runs ComfyUI API-format workflows for Qwen-Image, H3, Wan 2.2,
  Qwen3-TTS, ACE-Step, camera moves (incl. gufo's `GufoCameraMove` node), RIFE
  target fps and Real-ESRGAN upscale chains.
- Qwen3-TTS `speed`; Qwen3.8-27B reuses system prompts again (a new question
  under a 1.25K-token system prompt answers after 0.49 s instead of 3.29 s)
  and gets a Windows table of verification GEMM shapes; Qwen3.8-Flash-Next
  MTP decode +17 % on a story prompt; the Omni captioner streams its weights
  (about 1.7 GB host RAM instead of 12+ GB).

This is a fork of [Gufo](https://github.com/gufo-org/gufo), the local
inference engine for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S,
`gfx1151`). It carries the [pixmaate Windows port](https://github.com/pixmaate/gufo)
onto current upstream Gufo, so Gufo builds and runs natively on Windows 11:
no WSL, no Linux. It serves an OpenAI-compatible API, so clients written for
llama.cpp's server or the OpenAI API work against it.

The easiest way to use it is the release zip above. The step-by-step guide
further down builds `main` from source and covers Qwen3.8-27B only; the
release branches (`release/1.1`, `release/1.1.1-ttft`) carry everything in
the tables below. Everything on this page was run on one machine (below);
Linux users should use [upstream Gufo](https://github.com/gufo-org/gufo).

The port, the fixes and the measurements were done by Claude (Anthropic,
Claude Opus 5.5) in Claude Code, at the request of the repository owner. MIT
licensed, like upstream.

## What runs here now (1.1.1, October 2026)

All of this runs on the 64 GB EVO-X2 below, under Windows 11, behind one
OpenAI-compatible router on one port. Models load on the first request;
with the memory guard several stay loaded while they fit. Numbers are single
requests on this machine, measured on the version in the last column.

| Area | Model | Measured here | Version |
| --- | --- | --- | --- |
| Text | Qwen3.8-27B (UD-Q4_K_XL, DFlash2) | 28-31 tok/s decode, prefill ~515 tok/s (1.0); first token of a film JSON request 0.96 s, loads in ~14 s (1.1.1) | 1.0 / 1.1.1 |
| Text, 64 GB with SSD streaming | Qwen3.8-Flash-Next UD-Q4_K_XL (104 GB) | 12.8 tok/s (story), 13.5 tok/s (cached English), prefill ~282 tok/s | 1.1.0 |
| Text, 64 GB with SSD streaming | Qwen3.8-Flash-Next UD-Q3_K_XL (90 GB) | 15.6 tok/s, one request | 1.1.0 |
| Text | Qwen3.6-35B-A3B (Q4_K_M) | 51 tok/s on a 40-shot JSON list, 65 tok/s prose; with DFlash2 63 tok/s on the JSON list (+25 %), but 46 tok/s on prose | 1.0 `gufo-a3b.exe` |
| Creative writing | Hemmingway-1.5 (27B fine-tune) | ~25 tok/s | 1.0 |
| Audio description | Qwen3-Omni-30B-A3B-Captioner | 53.6 tok/s | 1.1.0 |
| Image | Qwen-Image 2.1 | 512² / 40 steps 20.2 s; 1024² / 40 steps 109-112 s; edit with up to 10 reference images | 1.1.1 |
| Image, fast | Qwen-Image 2.1 Turbo (8 steps) | 512² 4.3 s; 1024² 22.7-24.0 s; 1920x1088 63 s | 1.1.1 |
| Image style | LoRA adapters for Qwen-Image 2.1 | switching 2.9-4.2 s, no extra time per step | 1.1.1 |
| Image layout | MOSS-Image (JSON layout prompt with boxes) | 512² / 30 steps 27.1 s; 1024² / 50 steps 172 s | #907 build |
| Video with sound | MiniMax H3 (Turbo LoRA, first/last frame) | 1 s 512² in 60-66 s | 1.1.0 |
| Video | Wan 2.2 TI2V-5B | 1 s 480² / 20 steps in 41-46 s | 1.1.0 |
| Camera moves | Wan 2.1 I2V 14B + Uni3C | 3 s 832x480 orbit, 4 steps, in 197 s | 1.1.0 |
| Video delivery | RIFE 4.7, Real-ESRGAN | RIFE 5.9 ms per new frame (24 to 60 fps); 832x512 to 1920x1200 in 34-62 ms per frame | 1.1.0 |
| Speech | Qwen3-TTS 1.7B (CustomVoice, VoiceDesign, cloning, `speed`) | ~2x realtime | 1.0 |
| Music | ACE-Step 1.5 XL turbo | 10 s of music in 10.7 s | 1.1.0 |
| Pages | `/chat/` `/bild/` `/stimme/` `/video/` `/musik/` `/speicher/` | served by the router | 1.1.1 |

What made the bigger models fit on 64 GB:
- **SSD expert streaming for Flash-Next** (`--expert-cache-bytes`): routed
  experts live in a GPU slot cache and are read from the GGUF on a miss.
  Without it, neither Flash-Next quant loads on a 64 GB machine.
- **Reading weights from the file instead of a memory mapping** (Qwen-Image,
  27B, H3 prompt encoder): Windows keeps mapped pages resident, so a 30-40 GB
  model filled the 32 GB left to Windows.
- **H3 on Windows:** MP4/audio muxing through FFmpeg named pipes, and rocBLAS
  fallbacks where TheRock 10.0 lacks the pinned ROCm 10.1 solutions. Since
  29 October the video decoder runs in FP16 (132 s → 23 s for 5 s of video)
  and the picture encoder runs in a short-lived child process, so all 50 DiT
  blocks and picture understanding fit together.

Found on the way: **the Windows build did not inline GPU helper functions.**
clang-cl turns the RelWithDebInfo flag `/Ob1` into `-finline-hint-functions`
for the device compile too, so every `__device__` helper without `inline`
stayed a real call inside the kernels. Building HIP sources with `/Ob2`
makes Qwen-Image 30 % faster and MOSS-Image 8 % faster, with bit-identical
output (upstream [#872](https://github.com/gufo-org/gufo/pull/872), merged).

Also found: on Windows `hipMemsetAsync` cleared the 27B's 159 MB recurrent
state at only ~0.5 GB/s, twice before every prompt. A small zeroing kernel
does it in 1-4 ms; that is the 0.7 s faster first token in 1.1.1.

Not working here: Fish Audio Drama-3 TTS (its weights alone take 39.6 GB of
GPU memory; not yet tried on the 128 GB machine), DeepSeek V4 Flash (too
large for 64 GB; not tried yet).

### Want something from this?

**Open an [issue](https://github.com/D3rPapts/gufo-studio-windows/issues) in this
repository** and say what you'd like to use: a model, a fix, a feature from
one of the branches. We'll help. That can mean rebasing it onto your fork or
branch, building it for you, or testing on our hardware (a 64 GB and a 128 GB
Strix Halo, both under Windows 11).

Given back to upstream Gufo:
- Merged: [#773](https://github.com/gufo-org/gufo/pull/773),
  [#794](https://github.com/gufo-org/gufo/pull/794),
  [#795](https://github.com/gufo-org/gufo/pull/795),
  [#796](https://github.com/gufo-org/gufo/pull/796),
  [#810](https://github.com/gufo-org/gufo/pull/810) (model-bench on Windows),
  [#830](https://github.com/gufo-org/gufo/pull/830) (Qwen-Image loading),
  [#856](https://github.com/gufo-org/gufo/pull/856) (H3 on Windows),
  [#872](https://github.com/gufo-org/gufo/pull/872) (`/Ob2` for HIP sources).
- Open: [#885](https://github.com/gufo-org/gufo/pull/885) (Qwen-Image true CFG
  and multi-reference edits), [#904](https://github.com/gufo-org/gufo/pull/904)
  (Windows table for the 27B verification GEMMs),
  [#907](https://github.com/gufo-org/gufo/pull/907) (MOSS-Image).
- Elsewhere: [slimsami/strycast#7](https://github.com/slimsami/strycast/pull/7)
  (MMQ decode for the 35B experts, merged);
  [stockmind/gufo#1](https://github.com/stockmind/gufo/pull/1) (faster
  Qwen3.6-35B-A3B decode, open); [endyjasmi/gufo#35](https://github.com/endyjasmi/gufo/pull/35)
  (SSD expert streaming for that Windows port; declined, so it stays here).

## What was tested

Machine: **GMKtec EVO-X2** (Ryzen AI MAX+ 395, 64 GB: 32 GB dedicated GPU
memory, 32 GB left to Windows), run in its **normal performance mode, not the
high-performance mode**, Windows power plan "Balanced". Windows 11 Pro
(build 26200), AMD driver 32.0.32015.7029, TheRock ROCm 10.0.0. llama.cpp
and gufo ran under the same settings; the faster modes were not tried.

Qwen3.8-27B UD-Q4_K_XL with the DFlash2 draft, against llama.cpp on the same
machine, on a mix of German, English and strict JSON requests with and
without thinking:

| Server | Decode, JSON + thinking | Prefill, 17.5K prompt | Total, all test cases |
| --- | --- | --- | --- |
| llama.cpp Vulkan + MTP | 28.2 tok/s | 256 tok/s | 487 s |
| best llama.cpp setting found (Vulkan + DFlash2) | 28.3 tok/s | 257 tok/s | 431 s |
| **gufo-studio-windows** (1.0) | **28.2-31.2 tok/s** | **512-517 tok/s** | **332-349 s** |

- Prefill about 2x llama.cpp; 20-30% less total time on this workload.
- Up to 152K context on 32 GB GPU memory: a 147,913-token prompt was
  answered correctly (prefill 343 tok/s, decode 10.8 tok/s at that depth).
- Output at temperature 0 is identical with and without speculative decoding.
- One request at a time; concurrent requests were not measured.

Every configuration tried (gufo and llama.cpp), what worked and what did not,
and the repeat runs: [docs/WINDOWS-REBASE.md](docs/WINDOWS-REBASE.md).

## What you need

| What | Version used here | Where to get it |
| --- | --- | --- |
| Strix Halo PC (Radeon 8060S, `gfx1151`) with Windows 11 x64 | Windows 11 Pro 26200 | |
| AMD graphics driver | 32.0.32015.7029 | [AMD drivers](https://www.amd.com/en/support/download/drivers.html) (AMD Software: Adrenalin Edition) |
| Dedicated GPU memory of at least 32 GB | 32 GB | AMD Software > Performance > Tuning > Variable Graphics Memory, then restart |
| TheRock ROCm for Windows, gfx1151 | 10.0.0 | [therock-dist-windows-gfx1151-10.0.0.tar.gz](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz), extracted to `C:\TheRock\build` |
| Visual Studio Build Tools, workload "Desktop development with C++" | Build Tools 2026 (18.11) | [Visual Studio downloads](https://visualstudio.microsoft.com/downloads/) > Tools for Visual Studio > Build Tools |
| CMake and Ninja | CMake 4.4.4, Ninja 1.13.2 | `winget install Kitware.CMake Ninja-build.Ninja` |
| Git | 2.52 | `winget install Git.Git` |
| vcpkg with icu, curl, openssl, libpng, libjpeg-turbo | | see step 2 below |
| The model files (about 18 GB) | | see step 4 below |

TheRock is AMD's ROCm build for Windows; its clang compiles Gufo, Visual
Studio only supplies the C++ standard library and the Windows SDK. Gufo was
built and tested with TheRock 10.0.0 only. **Do not use TheRock 10.1 or newer
on Windows yet:** Gufo built with it runs about 50 times slower on gfx1151
(27B decode 0.2 instead of 11.4 tok/s), and the long-running kernels make
Windows reset the graphics driver (TDR).

## Step by step

All commands are for PowerShell. **After installing anything with winget,
open a new PowerShell window**, otherwise the new programs are not on the
PATH yet.

### 1. TheRock ROCm

Download the archive linked above (SHA-256
`1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`) and
extract it so that `C:\TheRock\build\bin` exists:

```powershell
mkdir C:\TheRock\build
tar -xzf $HOME\Downloads\therock-dist-windows-gfx1151-10.0.0.tar.gz -C C:\TheRock\build
Get-FileHash $HOME\Downloads\therock-dist-windows-gfx1151-10.0.0.tar.gz   # compare with the SHA-256 above
```

### 2. vcpkg and the libraries

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install icu curl openssl libpng libjpeg-turbo --triplet x64-windows
```

### 3. Get the code and build

```powershell
git clone https://github.com/D3rPapts/gufo-studio-windows
cd gufo-studio-windows
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # lists anything missing and how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # -> build\release\gufo.exe
```

`build.ps1` loads the Visual Studio environment, builds, and copies the ROCm
and vcpkg DLLs next to `gufo.exe`, so `build\release` runs as it is. If it
reports `ninja not found`, open a new PowerShell window (see above) or pass
`-Ninja <path to ninja.exe>`.

### 4. Download the model

Three files: the model, the DFlash2 draft (makes decoding about twice as
fast) and the vision file (for image input). Download them in the browser or
with `curl.exe`, into one folder, for example `D:\models`:

| File | Size | Link |
| --- | --- | --- |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 16.4 GB | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.1 GB | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/resolve/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) |
| `mmproj-BF16.gguf` | 0.9 GB | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/mmproj-BF16.gguf) (download button on the page) |

```powershell
mkdir D:\models
curl.exe -L -o D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf
curl.exe -L -o D:\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/resolve/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
curl.exe -L -o D:\models\mmproj-BF16.gguf https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf
```

`check.ps1 -ModelDir D:\models` then lists the model with "fits".

### 5. Start the server

This is the exact setting that was measured as fastest:

```powershell
$env:GUFO_QWEN27_LOOKUP = "4"
.\build\release\gufo.exe serve --host 127.0.0.1 --port 8080 llm `
  --model D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf `
  --mmproj D:\models\mmproj-BF16.gguf `
  --served-model-name qwen3.8-27b `
  --context 65536 `
  --speculative dflash2 --dflash-model D:\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --draft-tokens 7 --repeat-penalty 1.05
```

- The server is ready when the log in the window prints `event=listening`
  (about 6 s here, with the model files already in the Windows file cache).
  Closing the window stops the server.
- `GUFO_QWEN27_LOOKUP=4` turns on prompt lookup (reuses text from the prompt
  as a draft), the fastest setting measured.
- `--repeat-penalty 1.05`: without it Qwen3.8-27B sometimes loops into
  endless lists (on llama.cpp too); with it, no loops in 18 of 18 runs.
- `--context 155648` (152K) also works on 32 GB GPU memory; a prompt that
  long takes about 7 minutes to read in.
- `--host 0.0.0.0` makes the server reachable from other computers in the
  network. There is no password unless you add `--api-key`.

Do not run llama.cpp or a second Gufo at the same time: they compete for GPU
memory.

### 6. Use it

Any OpenAI-compatible client works with the base URL
`http://localhost:8080/v1` and the model name `qwen3.8-27b`. From PowerShell:

```powershell
$body = @{
  model = "qwen3.8-27b"
  messages = @(@{ role = "user"; content = "Sag Hallo auf Deutsch." })
  max_tokens = 200
  chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 5
(Invoke-RestMethod http://localhost:8080/v1/chat/completions -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body))).choices[0].message.content
```

Tested request features: text, thinking on and off
(`chat_template_kwargs.enable_thinking`, `reasoning_budget_tokens`), and
JSON output forced by a schema (`response_format`), in the OpenAI form and in
llama.cpp's form. Set `max_tokens` with thinking on: one thinking run went to
about 34K tokens.

### 7. Watch it (optional)

`tools\windows\gufo_monitor.py` (from the akamotaco/gufo_win fork) is a
read-only web page: prompt progress, decode tokens, the last requests with
cache hits and time to first token, GPU utilisation, dedicated and shared GPU
memory, free RAM, commit and page file. It needs Python 3.

```powershell
python tools\windows\gufo_monitor.py --port 8095 --servers 8080=Qwen3.8-27B --logs <path to the server log>
```

Then open `http://127.0.0.1:8095`. `--logs` takes the server's log file
(stderr redirected to a file) or a folder of `gufo-PORT-*.log` files.
Port 8090, the script's default, may be taken by other software.

## Known limitations

Measured on the test machine; see [docs/WINDOWS-REBASE.md](docs/WINDOWS-REBASE.md)
for the numbers.

- **Short answers without thinking decode slower than llama.cpp:** about
  19 tok/s against 20-23 tok/s. gufo's lead comes from prefill (about 2x)
  and from long thinking runs.
- **One request at a time per model.** A long request makes every other request wait
  in the queue (once 10 minutes during a 23-minute request).
- **Without `--repeat-penalty 1.05`, Qwen3.8-27B sometimes loops** into
  endless lists. llama.cpp does the same with this model.
- **Long context gets slow:** with 148K tokens in context, prefill drops to
  343 tok/s and decode to 10.8 tok/s.
- **`check.ps1` and `start.ps1` only look in fixed places:** the Hugging Face
  cache, a `models` folder in or beside the checkout, `%USERPROFILE%\models`,
  LM Studio's folders and `%LOCALAPPDATA%\llama.cpp`. Pass
  `-ModelDir <folder>` for anything else (`start.ps1` remembers it).
- **After installing CMake or Ninja with winget, open a new terminal**, or
  `check.ps1` and `build.ps1` report them missing.
- Plans and open questions are tracked in the
  [issues](https://github.com/D3rPapts/gufo-studio-windows/issues).

## Not tested here

- On this `main` branch: anything but Qwen3.8-27B. The other models in
  [What runs here now](#what-runs-here-now-111-october-2026) were tested on their
  branches and in the release packages. DeepSeek V4 Flash was never run here.
- The launchers `start.cmd` and `tools\windows\start.ps1` (only
  `start.ps1 -List` was run) and the prebuilt package script.
- Concurrent requests; the EVO-X2's high-performance mode and Windows power mode "Best performance".

[docs/WINDOWS.md](docs/WINDOWS.md) describes how the port works inside; its
measurements come from the pixmaate fork and were not re-checked here.

## Build from source

On Windows, follow [Step by step](#step-by-step) above; `tools\windows\build.ps1`
builds `build\release\gufo.exe`, and `-Preset gpu-test` adds the tests.
[docs/WINDOWS.md](docs/WINDOWS.md) explains the toolchain and how the port
works. The Linux build (Nix or system packages) is unchanged from upstream and
was not tested here; its prerequisites are in
[upstream's README](https://github.com/gufo-org/gufo#build-from-source).

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and
dependencies retain their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. Model
weights are not included and keep their publishers' terms.

## Credits

- [Gufo](https://github.com/gufo-org/gufo) and its contributors: the engine.
- [pixmaate/gufo](https://github.com/pixmaate/gufo): the original Windows port.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), and the projects Gufo
  credits in [NOTICE](NOTICE).
