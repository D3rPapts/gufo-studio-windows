# Windows GPU timeouts (TDR)

**Windows only.** This page is about a watchdog in the Windows display driver
model (WDDM). Linux builds of Gufo are not affected by it, and nothing here
changes or applies to Linux.

## Symptom

A server that was working stops with:

```
Error: 503: {"message":"GPU context lost; restart required","type":"server_error","code":"device_lost"}
```

The server log shows the loss at the first GPU call after it, for example:

```
[ERROR] [scheduler] event=device_lost remedy=restart reason=component cache transfer failed
hipModuleUnload failed:
 error: unspecified launch failure
```

Windows records it as a "hardware problem" in Reliability Monitor and the
Application event log:

```
Problem event name: LiveKernelEvent
Code: 141
```

with a `WATCHDOG-*.dmp` file in `C:\Windows\LiveKernelReports`. Code 141 is
`VIDEO_ENGINE_TIMEOUT_DETECTED`: Windows saw the GPU not finish its work in time
and reset it. The display usually comes back after a few seconds; every GPU
program running at that moment (Gufo, llama.cpp, games) loses its GPU context
and has to be restarted.

Despite the wording, this is a timeout, not a hardware fault. A failing GPU or
memory shows up as WHEA errors in the System event log, not as code 141 alone.

## Why it happens

Windows' Timeout Detection and Recovery (TDR) resets the GPU when submitted work
runs longer than `TdrDelay`, **2 seconds by default**. The limit applies to the
batch of work the driver submitted, not to one kernel, so a run of short
kernels can still exceed it when the GPU is busy. LLM inference keeps the GPU
fully loaded for minutes at a time: long prefills at 100K+ tokens, large MoE
expert batches and cache copies on a unified-memory APU all add to the time one
submission takes, and the desktop shares the same GPU.

Linux has no WDDM watchdog. The `amdgpu` driver has its own hang detection with
different, longer limits, which is why Gufo's kernels, written and tuned on
Linux, do not need to fit a 2-second window there.

## AMD lists it as a known issue

AMD's ROCm documentation for Radeon and Ryzen lists driver timeouts during
inference as a **Windows** known issue for ROCm 7.1.1:

> Intermittent application crash or driver timeout may be observed while running inference workloads

The Linux known issues on the same page do not mention driver timeouts.
Source: [ROCm on Radeon and Ryzen: limitations and recommended settings (7.1.1)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/limitations/limitationsryz.html).

Owners of other Ryzen AI Max+ 395 machines report the same "AMD Software Detected
a Driver Timeout" under long GPU loads on Windows 11 across several driver and
BIOS versions
([Framework Community thread](https://community.frame.work/t/gpu-intense-tasks-like-gaming-either-softlock-or-hardfreeze-my-machine/79426)).

On the machine this port is developed on (Ryzen AI Max+ 395, Radeon 8060S,
driver 32.0.32015.2008), Windows recorded nine code-141 events with the same
signature between 2026-10-08 and 2026-10-20, several of them before this Gufo
build was in use. No WHEA errors were logged.

## Fix: raise the timeout

Raising `TdrDelay` is the usual setting for GPU compute on Windows; Intel's
oneAPI guide, for example, recommends 20 seconds or more
([Intel: adjust the TDR setting](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-windows/latest/gpu-adjust-timeout-detection-and-recovery-setting.html)).
`tools\windows\tdr_delay.bat` sets `TdrDelay` and `TdrDdiDelay` to 30 seconds
and can put them back:

| Command | Does |
| --- | --- |
| `tools\windows\tdr_delay.bat` | shows the current values and asks what to do |
| `tools\windows\tdr_delay.bat set` | saves the current values, then sets both to 30 s |
| `tools\windows\tdr_delay.bat restore` | puts the saved values back (Windows defaults if none were saved) |
| `tools\windows\tdr_delay.bat status` | shows the current values |

`set` and `restore` ask for administrator rights. **Reboot afterwards**; the
driver reads these values at startup. The previous values are kept in
`%ProgramData%\gufo-win\tdr-backup.json` until `restore` uses them.

The same change by hand, in an administrator prompt:

```
reg add "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 30 /f
reg add "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 30 /f
```

The trade-off: if the GPU really hangs, the screen can stay frozen for up to
30 seconds before Windows recovers it. Do not turn the watchdog off
(`TdrLevel = 0`): a real hang would then need a hard reset.

## If it still happens

- Check the System event log for WHEA errors; those point at hardware.
- Note what was running: the server log's last request (prompt and prefill
  sizes) and any game or video playing at the same time.
- Try another AMD driver version; the reports above point at the driver more
  than at any one program.
