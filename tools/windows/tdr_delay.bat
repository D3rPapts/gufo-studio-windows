@echo off
rem Windows GPU timeout (TDR) for Gufo: raises TdrDelay/TdrDdiDelay to 30 s, or puts them back.
rem   tdr_delay.bat set | restore | status     (no argument: asks)
rem Changing the values asks for administrator rights and needs a reboot. See docs\WINDOWS-TDR.md.
setlocal
set "ACTION=%~1"
if "%ACTION%"=="" (
  echo GPU timeout ^(TDR^) for Gufo. See docs\WINDOWS-TDR.md
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tdr_delay.ps1" status
  echo.
  echo   1 = set 30 s    2 = restore previous values    anything else = quit
  set /p "CHOICE=Choice: "
)
if "%ACTION%"=="" if "%CHOICE%"=="1" set "ACTION=set"
if "%ACTION%"=="" if "%CHOICE%"=="2" set "ACTION=restore"
if "%ACTION%"=="" goto :end
if /i "%ACTION%"=="status" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tdr_delay.ps1" status
  goto :end
)
net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0' -ArgumentList '%ACTION% elevated'"
  goto :done
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tdr_delay.ps1" %ACTION%
:end
echo.
pause
:done
endlocal
