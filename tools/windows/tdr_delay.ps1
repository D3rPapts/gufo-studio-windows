# Sets or restores the Windows GPU timeout (TDR) for long Gufo GPU work.
# See docs/WINDOWS-TDR.md.
#
#   tdr_delay.ps1 set       TdrDelay and TdrDdiDelay = 30 s (the previous values are saved first)
#   tdr_delay.ps1 restore   put the saved values back (or the Windows defaults when none were saved)
#   tdr_delay.ps1 status    show the current values
# set and restore need an administrator shell and take effect after a reboot.
# Keep this file ASCII: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.
param(
  [Parameter(Mandatory = $true)][ValidateSet("set", "restore", "status")][string]$Action,
  [ValidateRange(3, 600)][int]$Seconds = 30
)
$ErrorActionPreference = "Stop"
$key = "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers"
$names = @("TdrDelay", "TdrDdiDelay")
# Windows defaults when a value is absent: TdrDelay 2 s, TdrDdiDelay 5 s.
$defaults = @{ TdrDelay = 2; TdrDdiDelay = 5 }
$backupDir = Join-Path $env:ProgramData "gufo-win"
$backup = Join-Path $backupDir "tdr-backup.json"

function Current($name) {
  $item = Get-ItemProperty -Path $key -Name $name -ErrorAction SilentlyContinue
  if ($item) { return [int]$item.$name }
  return $null
}

function Show() {
  foreach ($name in $names) {
    $value = Current $name
    if ($null -eq $value) { Write-Host ("  {0,-12} not set (Windows default {1} s)" -f $name, $defaults[$name]) }
    else { Write-Host ("  {0,-12} {1} s" -f $name, $value) }
  }
  if (Test-Path $backup) { Write-Host "  saved values to restore: $backup" }
}

function Assert-Admin() {
  $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
  if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this as administrator (tdr_delay.bat asks for it)."
  }
}

switch ($Action) {
  "status" {
    Write-Host "GPU timeout (TDR) settings:"
    Show
  }
  "set" {
    Assert-Admin
    # Save only the first time, so a second "set" does not overwrite the original values.
    if (-not (Test-Path $backup)) {
      New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
      $saved = @{}
      foreach ($name in $names) { $saved[$name] = Current $name }
      $saved | ConvertTo-Json | Out-File -Encoding ascii $backup
      Write-Host "Saved the previous values to $backup"
    }
    foreach ($name in $names) {
      New-ItemProperty -Path $key -Name $name -PropertyType DWord -Value $Seconds -Force | Out-Null
    }
    Write-Host "Set TdrDelay and TdrDdiDelay to $Seconds s:"
    Show
    Write-Host ""
    Write-Host "Reboot for the change to take effect."
  }
  "restore" {
    Assert-Admin
    $saved = $null
    if (Test-Path $backup) { $saved = Get-Content -Raw $backup | ConvertFrom-Json }
    else { Write-Host "No saved values; restoring the Windows defaults (values removed)." }
    foreach ($name in $names) {
      $value = $null
      if ($saved) { $value = $saved.$name }
      if ($null -eq $value) { Remove-ItemProperty -Path $key -Name $name -ErrorAction SilentlyContinue }
      else { New-ItemProperty -Path $key -Name $name -PropertyType DWord -Value ([int]$value) -Force | Out-Null }
    }
    if (Test-Path $backup) { Remove-Item -Force $backup }
    Write-Host "Restored:"
    Show
    Write-Host ""
    Write-Host "Reboot for the change to take effect."
  }
}
