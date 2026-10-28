# gufo_windows: Gufo on Windows 11 for AMD Strix Halo

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

> **Download [gufo_windows 1.0.0](https://github.com/D3rPapts/gufo_windows/releases/tag/v1.0.0)**
> (0.16 GB zip): text, images, speech and video from one OpenAI-compatible
> port on Windows 11. Unpack it, edit `examples\router.ini`, run
> `start-router.cmd`. The release notes list the 13 models tested.

This is a fork of [Gufo](https://github.com/gufo-org/gufo), the local
inference engine for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S,
`gfx1151`). It carries the [pixmaate Windows port](https://github.com/pixmaate/gufo)
onto current upstream Gufo, so Gufo builds and runs natively on Windows 11:
no WSL, no Linux. It serves an OpenAI-compatible API, so clients written for
llama.cpp's server or the OpenAI API work against it.

The step-by-step guide below covers Qwen3.8-27B. Since then many more models
run here on the same machine, each from its own branch. They are listed in
[What runs here now](#what-runs-here-now-october-2026). Everything on this
page was run on one machine (below); Linux users should use
[upstream Gufo](https://github.com/gufo-org/gufo).

The port, the fixes and the measurements were done by Claude (Anthropic,
Claude Opus 5.5) in Claude Code, at the request of the repository owner. MIT
licensed, like upstream.

## What runs here now (October 2026)

All of this runs on the 64 GB EVO-X2 below, under Windows 11, behind one
OpenAI-compatible router on one port (one model loaded at a time). Speeds are
single requests on this machine.

| Model | What it does | Speed here | Branch |
| --- | --- | --- | --- |
| Qwen3.8-27B (UD-Q4_K_XL, DFlash2) | text, images in | 28-31 tok/s, prefill ~515 tok/s (table below) | `win-merge-2026-10-24`, `verify-gemm-27b` |
| **Qwen3.8-Flash-Next on 64 GB** (UD-Q4_K_XL, 104 GB) | text, images in | 14-15 tok/s, prefill ~285 tok/s | `moe-ssd-stream` |
| **Qwen3.8-Flash-Next UD-Q3_K_XL** (90 GB) | text | ~16 tok/s, prefill ~290 tok/s | `moe-ssd-stream` |
| Qwen3.6-35B-A3B (Q4_K_M) | text | 70 / 62 / 53 tok/s at 0 / 8K / 32K context | `win-35b-decode-mmvq` |
| Hemmingway-1.5 (Qwen3.8-27B fine-tune for fiction) | creative writing | ~25 tok/s | runs on the 27B build |
| Qwen3-Omni-30B-A3B-Captioner | describes audio (speech, voices, sounds) | 47-59 tok/s | `win-main` |
| Qwen-Image-2.1 | text to image | 1024² / 40 steps ~150 s | `fix/qwen-image-weights-from-file` |
| MOSS-Image | text to image | 1024² / 50 steps ~200-260 s | `moss-image` |
| **MiniMax H3** | text to video with stereo sound | 1 s ~1.5 min, 5 s ~5 min (512²) | `feat/h3-windows-media` |
| Qwen3-TTS 1.7B (CustomVoice, VoiceDesign, Base) | speech, designed voices, voice cloning | ~2x realtime | upstream + `feat/h3-windows-media` build |
| Router with web pages | one port for all of the above; chat, image, voice and video pages | – | `win-router` |

What made the bigger models fit on 64 GB:
- **SSD expert streaming for Flash-Next** (`--expert-cache-bytes`): routed
  experts live in a GPU slot cache and are read from the GGUF on a miss.
  Without it, neither Flash-Next quant loads on a 64 GB machine.
- **Reading weights from the file instead of a memory mapping** (Qwen-Image,
  27B, H3 prompt encoder): Windows keeps mapped pages resident, so a 30-40 GB
  model filled the 32 GB left to Windows.
- **H3 on Windows:** MP4/audio muxing through FFmpeg named pipes, and rocBLAS
  fallbacks where TheRock 10.0 lacks the pinned ROCm 10.1 solutions.

Not working here: Fish Audio Drama-3 TTS (its weights alone take 39.6 GB of
GPU memory; not yet tried on the 128 GB machine), DeepSeek V4 Flash (too
large).

### Want something from this?

**Open an [issue](https://github.com/D3rPapts/gufo_windows/issues) in this
repository** and say what you'd like to use: a model, a fix, a feature from
one of the branches. We'll help. That can mean rebasing it onto your fork or
branch, building it for you, or testing on our hardware (a 64 GB and a 128 GB
Strix Halo, both under Windows 11).

Already offered elsewhere:
- Upstream: [#773](https://github.com/gufo-org/gufo/pull/773),
  [#794](https://github.com/gufo-org/gufo/pull/794),
  [#795](https://github.com/gufo-org/gufo/pull/795),
  [#796](https://github.com/gufo-org/gufo/pull/796) merged;
  [#810](https://github.com/gufo-org/gufo/pull/810) (model-bench on Windows),
  [#822](https://github.com/gufo-org/gufo/pull/822) (27B verify GEMMs),
  [#830](https://github.com/gufo-org/gufo/pull/830) (Qwen-Image loading),
  [#856](https://github.com/gufo-org/gufo/pull/856) (H3 on Windows) open.
- [endyjasmi/gufo#35](https://github.com/endyjasmi/gufo/pull/35): SSD
  expert streaming and UD-Q3_K_XL for the Windows port.
- [stockmind/gufo#1](https://github.com/stockmind/gufo/pull/1): faster
  Qwen3.6-35B-A3B decode.

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
| **gufo_windows** | **28.2-31.2 tok/s** | **512-517 tok/s** | **332-349 s** |

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
git clone https://github.com/D3rPapts/gufo_windows
cd gufo_windows
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
- **One request at a time.** A long request makes every other request wait
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
  [issues](https://github.com/D3rPapts/gufo_windows/issues).

## Not tested here

- On this `main` branch: anything but Qwen3.8-27B. The other models in
  [What runs here now](#what-runs-here-now-october-2026) were tested on their
  branches. DeepSeek V4 Flash and Qwen3-ASR were never run here.
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
