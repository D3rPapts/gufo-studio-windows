# Sweeps the Qwen verification launch geometries for one GGUF on this GPU.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\verify-sweep\run.ps1 -Model <model.gguf>
#
# Needs build\release from tools\windows\build.ps1 (it links against its
# libraries) and Python 3. Stop every gufo or llama.cpp server first: the GPU
# must be idle for the timings to mean anything. Prints, per projection and
# row count, the fastest exact candidate when it beats production by at least
# -MinGain percent, in the Launch<> syntax of verify_gemm_27b.hip.
param(
  [Parameter(Mandatory)][string]$Model,
  [int[]]$Batches = @(2, 3, 4, 5, 6, 7, 8),
  [int]$Rounds = 7,
  [double]$MinGain = 5,
  [int]$MinLayers = 2,
  [string]$Rocm = "C:\TheRock\build",
  [string]$Out = ""
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$work = Join-Path $root "build\verify-sweep"
New-Item -ItemType Directory -Force $work | Out-Null
$exe = Join-Path $work "gufo-verify-sweep.exe"
$lib = Join-Path $root "build\release"
if (-not (Test-Path "$lib\gufo_core.lib")) { throw "build\release is missing; run tools\windows\build.ps1 first" }

if (-not (Test-Path $exe) -or (Get-Item $exe).LastWriteTime -lt (Get-Item "$lib\gufo_core.lib").LastWriteTime -or
    (Get-Item $exe).LastWriteTime -lt (Get-Item "$PSScriptRoot\sweep.hip").LastWriteTime) {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  $env:PATH = "$(Split-Path $vswhere);$env:PATH"
  cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
  }
  $env:HIP_PATH = $Rocm.Replace('\', '/')
  $env:HIP_DEVICE_LIB_PATH = "$Rocm/lib/llvm/amdgcn/bitcode".Replace('\', '/')
  $clang = "$Rocm\lib\llvm\bin\clang++.exe"
  $src = $root.Replace('\', '/')
  # Same defines and flags as gufo_core (build\release\compile_commands.json).
  & $clang -DENGINE_ENABLE_HIP=1 -DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_DECLARE_NONSTDC_NAMES=0 `
    -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 `
    "-I$src/compat/win32/include" "-I$src" -isystem "$Rocm/include" "-I$src/compat/win32/hip_include" `
    -O2 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt `
    -fno-slp-vectorize -x hip -c "$PSScriptRoot\sweep.hip" -o "$work\sweep.obj"
  if ($LASTEXITCODE -ne 0) { throw "compile failed" }
  $clangRt = Get-ChildItem "$Rocm\lib\llvm\lib\clang\*\lib\windows\clang_rt.builtins-x86_64.lib" | Select-Object -First 1
  & $clang -fuse-ld=lld --hip-link --offload-arch=gfx1151 "$work\sweep.obj" `
    "$lib\gufo_core.lib" "$lib\gufo_posix_compat.lib" "$lib\gufo_logging.lib" "$lib\gufo_hip_host.lib" `
    "$Rocm\lib\hipblas.lib" "$Rocm\lib\rocblas.lib" "$Rocm\lib\amdhip64.lib" "$Rocm\lib\libhipblaslt.dll.a" `
    $clangRt.FullName -ldxgi -lws2_32 -lpsapi -lbcrypt -lshlwapi -lole32 -o $exe -Xlinker /NODEFAULTLIB:libcmt
  if ($LASTEXITCODE -ne 0) { throw "link failed" }
}

$busy = Get-Process gufo, llama-server -ErrorAction SilentlyContinue
if ($busy) { Write-Warning "running: $($busy.ProcessName -join ', ') - timings are not meaningful while a server uses the GPU" }
$env:PATH = "$Rocm\bin;$env:PATH"
$shapes = @(& python -I "$PSScriptRoot\gguf_tensors.py" $Model)
$names = @{ 12 = "Q4_K"; 13 = "Q5_K"; 14 = "Q6_K"; 23 = "IQ4_XS" }
$log = @()
$suggestions = foreach ($shape in $shapes) {
  $type, $m, $k, $offset, $count, $example = $shape -split ' ', 6
  if ([int]$count -lt $MinLayers) { continue }
  foreach ($batch in $Batches) {
    $lines = @(& $exe $Model $offset $type $m $k $batch $Rounds)
    $log += $lines
    $best = $lines | Where-Object { $_ -match '^cand (.+) [\d.]+ us (\d+) GB/s ([+-][\d.]+)%$' } | ForEach-Object {
      [pscustomobject]@{ Launch = $matches[1]; GBs = [int]$matches[2]; Gain = [double]$matches[3] } } |
      Sort-Object Gain -Descending | Select-Object -First 1
    $base = ($lines | Select-String '^base .* ([\d.]+) us (\d+) GB/s$').Matches.Groups[2].Value
    $inexact = @($lines | Where-Object { $_ -match 'NOT-EXACT' }).Count
    [pscustomobject]@{
      Type = $names[[int]$type]; M = [int]$m; K = [int]$k; Layers = [int]$count; Example = $example
      Batch = $batch; BaseGBs = [int]$base; BestGBs = $best.GBs; Gain = $best.Gain; Launch = $best.Launch
      NotExact = $inexact; Keep = $best.Gain -ge $MinGain
    }
  }
}
if ($Out) { $log | Set-Content -Encoding utf8 $Out }
$suggestions | Format-Table Type, M, K, Layers, Batch, BaseGBs, BestGBs, Gain, Keep, Launch -AutoSize | Out-String -Width 220
"Gains of $MinGain% or more, as verify_gemm_27b.hip entries (check the shape is not also a draft shape there):"
$suggestions | Where-Object Keep | ForEach-Object {
  "  // {0} {1}x{2} ({3} layers, {4}), {5} rows: {6} -> {7} GB/s, +{8}%" -f $_.Type, $_.M, $_.K, $_.Layers, $_.Example, $_.Batch, $_.BaseGBs, $_.BestGBs, $_.Gain
  "  {0}(w, x, y, m, k, stream);" -f $_.Launch
}
