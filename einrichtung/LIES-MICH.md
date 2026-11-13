# Einrichtung: Bausteine-Katalog und unbeaufsichtigtes Einrichtungs-Skript

Teile 1 und 5 aus `D:\claude\gufo-arbeit\INSTALLER-PLAN-2026-11-13.md` (Stand 13.11.2026, KI-unterstützt mit Claude Code, Baumeister .198). Noch ohne Router-Endpunkte, ohne Seite `/einrichtung/`, ohne Download – das Skript **prüft nur**, welche Dateien fehlen, und **lädt nichts**.

## Was liegt hier

| Datei | Zweck |
|---|---|
| `katalog.json` | Bausteine-Katalog: je Baustein (chat, bilder, stimme, video, musik, 3d, werkzeuge, zweiter-rechner) die Modelle mit ID, Abschnitt, Pflicht/optional, Worker-Exe, Ordner im Baum `modelle\<Hersteller>\<Modell>`, Dateiliste (Größe aus dem echten Baum der .198, SHA-256 wo bekannt), Hugging-Face-Quelle, Lizenz (Name + Satz + „bestätigen“), GPU-Werte, „passt in 64 GB“, Summen je Baustein. |
| `katalog_bauen.py` | Erzeugt `katalog.json` neu aus dem Modellbaum der .198, dem H3-Manifest und `gufo-router-combo7.memory.json`. Die Zuordnung Modell → Dateien/Lizenz/Quelle steht als Python-Daten in dieser Datei; nur hier pflegen, dann neu laufen lassen. |
| `bausteine\*.ini` | Preset-Vorlagen je Baustein, aus dem Produktions-Preset `gufo-router-combo7.ini` übernommen, mit Platzhaltern. `router.ini` ist der Kopf (Startzeile, Umgebung), `zweiter-rechner.ini` die `[remote/…]`-Vorlage. |
| `einrichten.py` | Unbeaufsichtigte Einrichtung: liest `antworten.json`, prüft Dateien, schreibt `gufo-router.ini`, `start-router.cmd`, `einrichtung.json`, Bericht. Schalter `--plan`, `--schreiben`, `--export`, `--import`. |
| `antworten-beispiel.json` | Beispiel-Antworten (zweite EVO-X2, 128 GB, Modelle unter `C:\gufo-modelle`). |
| `vergleich_preset.py` | Vergleicht ein erzeugtes Preset mit dem Produktions-Preset je Abschnitt (Schlüssel und Werte). |

Alles Python 3.11, nur Standardbibliothek.

## Platzhalter in den Vorlagen

| Platzhalter | Bedeutung | Beispiel .198 | Beispiel .56 |
|---|---|---|---|
| `{modelle}` | Modellwurzel (`modelle\<Hersteller>\<Modell>`) | `D:\llm-modelle\gufo\modelle` | `C:\gufo-modelle` |
| `{app}` | App-Ordner der Version mit `gufo.exe` (Router + Standard-Worker, H3-Manifest) | `D:\llm-modelle\gufo\app\gufo-2.0.2-2026-11-13` | `C:\gufo\2.0.3\router` |
| `{app_35b}` `{app_3d}` `{app_moss}` `{app_d}` `{app_trellis2}` | Ordner der eigenen Worker-Builds (Qwen3.6-35B, TripoSG, MOSS-Music, D-Block/Seed-VC, TRELLIS.2) | `app\gufo-stockmind-35b-2026-10-26b` … | `C:\gufo\2.0.3\35b` … |
| `{daten}` | Wurzel für Ablage/Jobs/Caches/Stimmen: `jobs\`, `prompt-cache*\`, `video-jobs*\`, `voices\`, `expert-profiles\` | `D:\llm-modelle\gufo` | `C:\gufo` |
| `{pages}` `{preset}` | Seitenordner, Pfad des Presets (Standard: `{app}\..\pages`, `{daten}\gufo-router.ini`) | | |
| `{partner}` `{partner_ip}` | Adresse des zweiten Rechners (`http://IPv4:Port`, nur IPv4 – der Parser verlangt es) | `http://192.168.100.56:8081` | `http://192.168.100.198:8081` |
| `{port}` `{host}` | Port; `0.0.0.0` bei `erreichbarkeit = heimnetz`, sonst `127.0.0.1` | | |
| `{reserve}` `{idle_seconds}` `{webhook_allow}` | `GUFO_HOST_RESERVE_MIB`, `--sleep-idle-seconds`, `--jobs-webhook-allow` | 2048, 1800, 192.168.0.0/16 | |
| `{comfy_dir?}` `{comfy_ffmpeg?}` `{names_file?}` `{pool_wait?}` `{lora_qwen_image?}` | optional: ist der Wert leer, fällt die ganze Zeile weg | | |
| `{id}` | nur in `zweiter-rechner.ini`: Modell-ID je `[remote/…]`-Block | | |

Weitere Regeln der Vorlagen:

- **Marker:** jeder Abschnitt beginnt mit `# baustein: <name>` und `# modell: <id>`. Daran erkennt `einrichten.py` seine Abschnitte beim Nachrüsten (eigene Abschnitte eines Nutzers bleiben erhalten und wandern ans Ende unter „kept from the previous preset“). **Achtung:** der Marker ist `#`, nicht `;` – `preset.cpp` kennt nur `#` als Kommentar; ein `;`-Marker wäre „expected key = value“ und der Router startet nicht.
- **Schalter-Zeilen:** `#@<flag> key = value` wird nur übernommen, wenn das Flag aktiv ist. Aktiv sind `64`/`128` (RAM-Größe) und alle `schalter` aus den Antworten, die `true` sind. Heute genutzt: `#@stimmen_vorlage` für die `voice =`-Zeilen (Stimmen-Dateien unter `{daten}\voices\`).
- **64/128 GB:** Die einzige gemessene Variante ist die Modellwahl Flash-Next **Q3 (Vorgabe 64 GB)** gegen **Q4 (Vorgabe 128 GB)** (`vorgabe_ram` im Katalog). Die Speicherwerte `expert-cache-bytes = 24 GB` und `cache-ram-bytes = off` sind für beide Größen gleich: so läuft es auf der .198 (64 GB) und auf der .56 (128 GB, gleiches Preset über `preset_fuer_56.py`). Für 128 GB gibt es keine eigene Messung mit größeren Caches; das Paketbeispiel (`examples\router.ini`, upstream) sagt „leave out expert-cache-bytes on a 128 GB machine“, das ist nicht von uns gemessen und deshalb nicht in der Vorlage. Gleiches gilt für `cache-ram-bytes` des 27B (1 GiB wegen RAM auf der .198; 4 GiB ließen 64 GB ~0,1 GB Luft).
- **Router-Kopf:** Der Parser hat keinen `[router]`-Abschnitt (nur `[modalität/id]`), deshalb stehen Startzeile und Umgebung als `# start:` / `# start-opt:` / `# env:`-Kommentare in `router.ini`; `einrichten.py` baut daraus `start-router.cmd` (wie `start-router.ps1` der .198 bzw. `start-router-56.bat` aus `buendel_56.py`).

## antworten.json

Felder (siehe `antworten-beispiel.json`):

- `rechner`, `ram_gb` (64 oder 128), `bausteine` (Liste), `modelle_optional` (IDs aus `katalog.json`), `modelle_weglassen`.
- `pfade`: `modelle`, `app`, `daten` (Pflicht); `app_35b`, `app_3d`, `app_moss`, `app_d`, `app_trellis2`, `pages`, `preset`, `comfy_dir`, `comfy_ffmpeg`, `names_file` (optional).
- `router`: `port`, `erreichbarkeit` (`nur-dieser-rechner` | `heimnetz`), `adresse` (eigene IPv4, nötig für den Export), `idle_seconds`, `reserve_mib`, `webhook_allow`.
- `schalter`: `{"stimmen_vorlage": true}` = `voice`-Zeilen mit `{daten}\voices\erzaehler*.wav` schreiben (die Dateien von der .198 kopieren; sonst `false`).
- `partner`: `url`, `modelle` (Liste von IDs oder `"alle-lokalen"`), `pool_wait_seconds` (optional, 2.0.2).
- `lizenzen_bestaetigt`: Liste von Modell-IDs oder `["alle"]`. `--schreiben` verweigert, solange ein Modell mit `bestaetigen = true` nicht in der Liste steht; `--plan` listet sie.

Auswahl: Pflichtmodelle der gewählten Bausteine kommen immer, Modelle mit `vorgabe_ram` passend zur RAM-Größe automatisch (Flash-Next Q3/Q4), alles andere über `modelle_optional`. Abhängigkeiten (`benoetigt`, z. B. Uni3C braucht Wan 2.2 wegen T5; InfiniteTalk braucht Uni3C; RIFE-Worker braucht den H3-Ordner) werden mit aufgenommen und im Bericht genannt.

## Aufrufe

```
cd D:\llm-modelle\gufo\src\gufo-203\einrichtung
python einrichten.py --antworten antworten.json --plan
python einrichten.py --antworten antworten.json --schreiben --ziel C:\gufo\2.0.3
python einrichten.py --antworten antworten.json --export export-198.json
python einrichten.py --import export-198.json --antworten eigene.json --plan
python katalog_bauen.py
python vergleich_preset.py --erzeugt plan.ini --ignorieren external/qwen3.5-4b,external/qwen3.5-9b
```

- `--plan`: prüft und zeigt den Bericht (Bausteine, Modelle, GB; fehlende Dateien mit Größe und Quelle; fehlende Datendateien; fehlende Worker-Exes; Lizenzen). Nichts wird geschrieben. `--preset-ausgabe <datei>` legt das Preset zusätzlich für Tests ab.
- `--schreiben`: schreibt `gufo-router.ini` (Pfad aus `pfade.preset`, sonst `<ziel>\gufo-router.ini`; ein vorhandenes Preset wird als `.bak-<zeit>` gesichert, fremde Abschnitte bleiben am Ende erhalten), `start-router.cmd`, `einrichtung.json` (Antworten + Versionen + `gufo.exe --version` + Zeit + fehlende Dateien) und `einrichtung-bericht.md` in `--ziel`; legt die Datenordner unter `{daten}` an.
- `--export`: schreibt Antworten + die lokale Modell-Liste + eigene Adresse als `export.json` für den zweiten Rechner.
- `--import`: nimmt den Export, überschreibt `rechner`, `ram_gb`, `pfade`, `router`, `schalter`, `lizenzen_bestaetigt`, `bausteine`, `modelle_optional`, `modelle_weglassen` aus `--antworten` (eigene Angaben) und setzt `partner` = der exportierende Rechner mit seinen Modellen. Das ist `preset_fuer_56.py` als Knopf: dort wurden `D:\llm-modelle\gufo\modelle` → `C:\gufo-modelle`, `D:\llm-modelle\gufo\` → `C:\gufo\`, App-Ordner und `remote`-URLs getauscht – hier passiert dasselbe über die Platzhalter.

## Testlauf gegen die .198 (13.11.2026)

`--plan` mit allen acht Bausteinen und allen optionalen Modellen, Pfade der .198, Partner-Liste wie in der Produktion, dann `vergleich_preset.py` gegen `gufo-router-combo7.ini`:

- Produktion 57 Abschnitte, erzeugt 55, gemeinsam 55, **55 von 55 identisch** (Schlüssel und Werte).
- Nur in der Produktion: `[external/qwen3.5-4b]`, `[external/qwen3.5-9b]` – llama-server-Abschnitte für den Mailputzer (Pfad im WinGet-Ordner des Nutzers), absichtlich nicht im Katalog.
- Alle 367 Katalog-Dateien im Baum vorhanden, 0 Bytes fehlend.
- Export (.198) → Import mit `antworten-beispiel.json` (.56, 128 GB): Partner wird `http://192.168.100.198:8081`, 31 `[remote/…]`-Zwillinge, Flash-Next Q4 statt Q3 vorausgewählt, Worker-Pfade `C:\gufo\2.0.3\…`.
- `--schreiben` zweimal mit einem fremden Abschnitt dazwischen: Sicherung angelegt, fremder Abschnitt am Ende erhalten, 56 Marker = 56 Abschnitte.

## Bekannte Lücken (nichts geraten)

- **Keine `*.source-manifest.json` unter `modelle\`.** Es gibt nur das H3-Manifest im App-Ordner. SHA-256 sind daher nur bekannt für: MiniMax H3 (84 Dateien, Manifest), Qwen-Image-2.1 (LFS-Dateien aus `.tree.json`), IndexTTS `german\` (SHA256SUMS.txt). Alle anderen Dateien stehen mit `sha256_status = "fehlt"` im Katalog (33 Modelle ganz oder teilweise ohne SHA). Für den Download-Teil (Plan Teil 3) müssen die Prüfsummen erst erzeugt werden (Hugging-Face-API liefert LFS-SHA-256 je Datei).
- **Hugging-Face-Revision gepinnt** nur bei: Qwen3.8-27B (unsloth + z-lab), Flash-Next Q4, Qwen-Image-2.1 + Turbo, Qwen3-ASR, MiniMax H3 (aus `docs/models/*.md` und dem Manifest). Die übrigen Repos stammen aus Ordnernamen, Download-Logs oder Notizen (`angabe_von` im Katalog) und sind nicht gepinnt. **Kein HF-Repo** bekannt für lokal umgewandelte Ordner: ACE-Step `gufo\`, MOSS-Music `gufo`, TripoSG `gufo\`, TRELLIS.2 `gufo-trellis2`, Upscale/RIFE (GitHub bzw. Spiegel), IndexTTS-Bündel, beide LoRAs, voiceconv, Comfy-Org/Wan-2.1, H3-Vision-mmproj.
- **Lizenz „nicht geprüft“** (KB 1127 ungeprüft bzw. keine Lizenzdatei im Ordner): Qwen3.8-27B, Flash-Next Q3/Q4, Qwen3.6-35B (beide), LoRA multiple-angles, Uni3C-ControlNet, RIFE-Gewichte, Omni-Captioner, H3-Turbo-LoRA. Diese verlangen im Assistenten einen Haken wie die bekannten Sonderlizenzen (Fish < 500k US$, Qwen-Image nur Forschung, Hemmingway CC BY-NC, H3 Community License, DINOv3, bilibili, Seed-VC GPL).
- **GPU-Werte:** `gpu_gb_notiz` nur, wo ein Preset-Kommentar oder die Release-Notes einen gemessenen Wert nennen (12 Modelle); 19 Modelle ohne. Daneben `gpu_gb_router`/`host_gb_router` = vom Router gelernte Spitzen aus `gufo-router-combo7.memory.json` (gemessen, mit Datum) für alle bis auf Hemmingway und die LoRAs. Die `estimate_*`-Werte der memory.json werden nicht übernommen.
- **„passt in 64 GB“** folgt der Plan-Tabelle (Flash-Next Q4 und H3 „nur-128“). Beide laufen auf der .198 (64 GB) – Q4 mit 24-GB-Cache und ~8,5 GB freiem Host-RAM beim Laden, H3 nur allein; die Begrenzung der Kombination Flash-Next Q4 + H3 (Plan 7) ist noch nicht im Skript.
- **Nicht geprüft:** welche LM-Datei ACE-Step turbo ohne `--lm` lädt (beide aufgeführt); ob MOSS-Image die HF-Originale unter `text_encoder\` (16,8 GB) noch braucht; ob TTS-Worker ohne vorhandene `voice`-Datei starten (deshalb Schalter `stimmen_vorlage`).
- **Im Baum, aber nicht im Katalog** (nicht im Produktions-Preset): alibaba-pai ControlNet, bosonai Higgs, wan21-vace, TRELLIS-image-large, ACE-Step/MOSS/TripoSG/TRELLIS-Originalordner, Mailputzer-GGUFs (llama-server).
- **Noch nicht gebaut:** Router-Endpunkte `/router/setup/*`, Download mit Wiederaufnahme/SHA, Seite `/einrichtung/`, Nutzerverwaltung, Firewall-Regel/Autostart (Admin), Selbsttest (Plan Teile 2, 3, 4, 6). `start-router.cmd` wird geschrieben, aber nicht gestartet.
