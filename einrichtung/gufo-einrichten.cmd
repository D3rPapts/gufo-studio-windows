@echo off
setlocal EnableExtensions
rem gufo-studio-windows 2.1: Einrichtungs-Assistent starten.
rem Startet den Router ohne Preset (nur die Seiten und /router/setup/*), nur fuer
rem diesen Rechner (127.0.0.1), und oeffnet http://127.0.0.1:<port>/einrichtung/
rem im Browser. Dort: Rechner pruefen, Bausteine waehlen, Speicherort, Download,
rem Preset schreiben, Selbsttest. Danach startet start-router.cmd (im Datenordner)
rem den fertigen Router; fuer das Heimnetz bindet der an 0.0.0.0 und braucht
rem die Firewall-Regel (die Seite sagt es in Schritt 5).
rem
rem Liegt im Paket neben bin\ und pages\ (oder in examples\einrichtung\).
rem Aufruf: gufo-einrichten.cmd [port]   (Vorgabe 8081; belegt -> 8089)

set "HERE=%~dp0"
set "EXE="
for %%D in ("%HERE%bin" "%HERE%..\bin" "%HERE%..\..\bin" "%HERE%.") do (
  if not defined EXE if exist "%%~D\gufo.exe" set "EXE=%%~fD\gufo.exe"
)
if not defined EXE (
  echo gufo.exe nicht gefunden. Dieses Skript gehoert in den Paketordner neben bin\ und pages\.
  pause
  exit /b 1
)
for %%F in ("%EXE%") do set "BIN=%%~dpF"
set "PAGES="
for %%D in ("%BIN%..\pages" "%BIN%pages" "%HERE%pages") do (
  if not defined PAGES if exist "%%~D\einrichtung\index.html" set "PAGES=%%~fD"
)
if not defined PAGES (
  echo Seitenordner pages\einrichtung nicht gefunden ^(neben bin\^).
  pause
  exit /b 1
)
set "EINR="
for %%D in ("%BIN%..\examples\einrichtung" "%HERE%." "%BIN%..\einrichtung") do (
  if not defined EINR if exist "%%~D\katalog.json" set "EINR=%%~fD"
)

set "PORT=%~1"
if not defined PORT set "PORT=8081"
netstat -ano -p tcp | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo Port %PORT% ist belegt ^(laeuft schon ein Router?^). Der Assistent nimmt 8089.
  set "PORT=8089"
)

echo Einrichtungs-Router: %EXE%
echo Seiten: %PAGES%
if defined EINR echo Katalog: %EINR%
echo Adresse: http://127.0.0.1:%PORT%/einrichtung/   ^(nur dieser Rechner^)
echo Dieses Fenster offen lassen, bis die Einrichtung fertig ist. Beenden mit Strg+C.
echo.
start "" /b powershell -NoProfile -NonInteractive -WindowStyle Hidden -Command "$u='http://127.0.0.1:%PORT%'; for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri ($u + '/health') | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }; Start-Process ($u + '/einrichtung/')"
if defined EINR (
  "%EXE%" router --einrichtung --host 127.0.0.1 --port %PORT% --pages-dir "%PAGES%" --einrichtung-dir "%EINR%"
) else (
  "%EXE%" router --einrichtung --host 127.0.0.1 --port %PORT% --pages-dir "%PAGES%"
)
set "RC=%errorlevel%"
if not "%RC%"=="0" (
  echo.
  echo Der Router hat sich mit Code %RC% beendet. Meldung oben lesen.
  pause
)
endlocal & exit /b %RC%
