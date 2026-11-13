# Unbeaufsichtigte Einrichtung fuer gufo-studio-windows (Teil 5 des Installer-Plans vom 13.11.2026).
# Liest antworten.json, prueft gegen katalog.json, welche Modelldateien im Zielbaum fehlen (nur pruefen, NICHT laden),
# und schreibt aus den Vorlagen in bausteine\ das Router-Preset gufo-router.ini, start-router.cmd, einrichtung.json
# und einen Bericht. Python 3.11, nur Standardbibliothek. KI-unterstuetzt (Claude Code), Baumeister .198.
#
#   python einrichten.py --antworten antworten.json --plan                 nur zeigen (Bericht auf die Konsole)
#   python einrichten.py --antworten antworten.json --schreiben [--ziel D]  Dateien schreiben (Standard: Ordner der Antworten)
#   python einrichten.py --antworten antworten.json --export export.json   Antworten fuer den zweiten Rechner ablegen
#   python einrichten.py --import export.json --antworten eigene.json --plan|--schreiben
#                                                                          Export einspielen: eigene Pfade/Rechner/Adresse
#                                                                          aus eigene.json, Partner = exportierender Rechner
import argparse, datetime, json, os, pathlib, re, subprocess, sys

HIER = pathlib.Path(__file__).resolve().parent
EINRICHTUNG_VERSION = "0.1 (2026-11-13, gufo 2.0.3)"
REIHENFOLGE = ["chat", "bilder", "stimme", "video", "musik", "3d", "werkzeuge", "zweiter-rechner"]
PFAD_SCHLUESSEL = ("modelle", "app", "daten", "pages", "preset", "app_35b", "app_3d", "app_moss", "app_d", "app_trellis2",
                   "comfy_dir", "comfy_ffmpeg", "names_file")
WORKER_PFAD = {"gufo.exe": "app", "router": "app", "35b": "app_35b", "gufo-3d": "app_3d", "gufo-moss": "app_moss", "gufo-d": "app_d", "trellis2": "app_trellis2"}

def lies_json(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig"))

def fehler(text):
    print("FEHLER:", text); sys.exit(2)

# ------------------------------------------------------------------------------------------------- Vorlagen
def lies_vorlage(name):
    """Bloecke einer Vorlage: [{'modell': id, 'zeilen': [...]}]; ein Block beginnt bei '# baustein:'."""
    text = (HIER / "bausteine" / f"{name}.ini").read_text(encoding="utf-8-sig")
    bloecke, akt = [], None
    for z in text.splitlines():
        if z.startswith("# baustein:"):
            akt = {"baustein": z.split(":", 1)[1].strip(), "modell": "", "zeilen": [z]}
            bloecke.append(akt); continue
        if akt is None: continue
        if z.startswith("# modell:"):
            akt["modell"] = z.split(":", 1)[1].strip()
        akt["zeilen"].append(z)
    for b in bloecke:
        while b["zeilen"] and not b["zeilen"][-1].strip(): b["zeilen"].pop()
    return bloecke

PLATZHALTER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(\?)?\}")

def fuelle(zeilen, werte, flags):
    """Platzhalter ersetzen; '{x?}' leer -> Zeile weg; '#@flag ' -> Zeile nur bei aktivem Flag."""
    out = []
    for z in zeilen:
        if z.startswith("#@"):
            flag, _, rest = z[2:].partition(" ")
            if flag not in flags: continue
            z = rest
        weg = False
        fehlend = []
        def ers(m):
            nonlocal weg
            name, opt = m.group(1), m.group(2)
            v = werte.get(name)
            if v is None or v == "":
                if opt: weg = True; return ""
                fehlend.append(name); return m.group(0)
            return str(v)
        z2 = PLATZHALTER.sub(ers, z)
        if fehlend: fehler(f"Platzhalter ohne Wert: {fehlend} in Zeile: {z}")
        if weg: continue
        out.append(z2)
    return out

# ------------------------------------------------------------------------------------------------- Antworten
def antworten_aus_import(export, eigene):
    """Export des ersten Rechners + eigene Angaben -> Antworten fuer diesen Rechner (Partner = der erste Rechner)."""
    a = json.loads(json.dumps(export["antworten"]))
    for k in ("rechner", "ram_gb", "pfade", "router", "schalter", "lizenzen_bestaetigt", "bausteine", "modelle_optional", "modelle_weglassen"):
        if k in eigene: a[k] = eigene[k]
    ex = export["exportiert_von"]
    a.setdefault("partner", {})
    a["partner"]["url"] = f"http://{ex['adresse']}:{ex['port']}"
    a["partner"]["modelle"] = ex["modelle"]
    if "zweiter-rechner" not in a.get("bausteine", []): a.setdefault("bausteine", []).append("zweiter-rechner")
    a["importiert_von"] = {"rechner": ex["rechner"], "zeit": export["zeit"]}
    return a

def pruefe_antworten(a):
    for k in ("rechner", "ram_gb", "bausteine", "pfade"):
        if k not in a: fehler(f"antworten.json: '{k}' fehlt")
    if a["ram_gb"] not in (64, 128): fehler("ram_gb muss 64 oder 128 sein")
    for b in a["bausteine"]:
        if b not in REIHENFOLGE: fehler(f"unbekannter Baustein '{b}' (erlaubt: {REIHENFOLGE})")
    for k in ("modelle", "app", "daten"):
        if not a["pfade"].get(k): fehler(f"pfade.{k} fehlt")
    r = a.setdefault("router", {})
    r.setdefault("port", 8081); r.setdefault("erreichbarkeit", "nur-dieser-rechner"); r.setdefault("idle_seconds", 1800)
    r.setdefault("reserve_mib", 2048); r.setdefault("webhook_allow", "127.0.0.1"); r.setdefault("adresse", "")
    a.setdefault("schalter", {}); a.setdefault("modelle_optional", []); a.setdefault("modelle_weglassen", [])
    a.setdefault("lizenzen_bestaetigt", []); a.setdefault("partner", {})

# ------------------------------------------------------------------------------------------------- Auswahl
def waehle_modelle(a, katalog):
    """Gewaehlte Modelle in Katalogreihenfolge: Pflicht + RAM-Vorgabe + modelle_optional + Abhaengigkeiten."""
    alle = {m["id"]: (b, m) for b in REIHENFOLGE if b in katalog["bausteine"] for m in katalog["bausteine"][b]["modelle"]}
    for mid in a["modelle_optional"] + a["modelle_weglassen"]:
        if mid not in alle: fehler(f"unbekanntes Modell '{mid}' (IDs siehe katalog.json)")
    gewaehlt = set()
    for b in a["bausteine"]:
        for m in katalog["bausteine"][b]["modelle"]:
            if m["pflicht"] or m.get("vorgabe_ram") == a["ram_gb"] or m["id"] in a["modelle_optional"]:
                gewaehlt.add(m["id"])
    for mid in a["modelle_optional"]:
        b = alle[mid][0]
        if b not in a["bausteine"]: fehler(f"Modell '{mid}' gehoert zum Baustein '{b}', der nicht gewaehlt ist")
    gewaehlt -= set(a["modelle_weglassen"])
    # Abhaengigkeiten (benoetigt) dazunehmen, Hinweis sammeln
    hinweise = []
    aenderung = True
    while aenderung:
        aenderung = False
        for mid in list(gewaehlt):
            for dep in alle[mid][1].get("benoetigt", []):
                if dep not in gewaehlt:
                    gewaehlt.add(dep); aenderung = True
                    hinweise.append(f"{mid} braucht {dep}: mit aufgenommen")
    reihe = [mid for mid in alle if mid in gewaehlt]
    return reihe, alle, hinweise

# ------------------------------------------------------------------------------------------------- Pruefung
def pruefe_dateien(a, reihe, alle):
    wurzel = pathlib.Path(a["pfade"]["modelle"]); daten = pathlib.Path(a["pfade"]["daten"])
    fehlend, falsch, da_bytes, fehl_bytes, geprueft = [], [], 0, 0, set()
    daten_fehlend = []
    for mid in reihe:
        b, m = alle[mid]
        for d in m["dateien"]:
            if d["pfad"] in geprueft: continue
            geprueft.add(d["pfad"])
            p = wurzel / d["pfad"]
            # Hardlinks (Turbo text_encoder, MOSS text_encoder.gguf) belegen keinen eigenen Platz
            soll = 0 if d.get("hardlink_auf") else (d["bytes"] or 0)
            if not p.is_file():
                fehlend.append((mid, d["pfad"], d["bytes"] or 0, m["quellen"])); fehl_bytes += soll
            elif d["bytes"] and p.stat().st_size != d["bytes"]:
                falsch.append((mid, d["pfad"], d["bytes"], p.stat().st_size)); fehl_bytes += soll
            else:
                da_bytes += soll
        for dd in m.get("daten_dateien", []):
            if not (daten / dd).is_file(): daten_fehlend.append((mid, dd))
    return {"fehlend": fehlend, "falsche_groesse": falsch, "bytes_da": da_bytes, "bytes_fehlend": fehl_bytes, "daten_fehlend": daten_fehlend}

def pruefe_worker(a, reihe, alle):
    fehlend = {}
    for mid in reihe:
        w = alle[mid][1]["worker"]
        k = WORKER_PFAD[w]
        pfad = a["pfade"].get(k)
        exe = pathlib.Path(pfad) / "gufo.exe" if pfad else None
        if exe is None or not exe.is_file():
            fehlend.setdefault(k, []).append(mid)
    return fehlend

def gufo_version(a):
    exe = pathlib.Path(a["pfade"]["app"]) / "gufo.exe"
    if not exe.is_file(): return ""
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=20, errors="replace")
        return ((r.stdout or r.stderr).strip().splitlines() or [""])[0]
    except Exception as e:
        return f"(nicht lesbar: {e})"

# ------------------------------------------------------------------------------------------------- Preset
def werte_fuer(a, reihe, alle):
    p = a["pfade"]; r = a["router"]
    w = {k: p.get(k, "") for k in PFAD_SCHLUESSEL}
    w["preset"] = w["preset"] or str(pathlib.Path(p["daten"]) / "gufo-router.ini")
    w["pages"] = w["pages"] or str(pathlib.Path(p["app"]).parent / "pages")
    w.update({"port": r["port"], "idle_seconds": r["idle_seconds"], "reserve": r["reserve_mib"], "webhook_allow": r["webhook_allow"],
              "host": "0.0.0.0" if r["erreichbarkeit"] == "heimnetz" else "127.0.0.1",
              "rechner": a["rechner"], "ram": a["ram_gb"], "zeit": datetime.datetime.now().isoformat(timespec="minutes"),
              "einrichtung_version": EINRICHTUNG_VERSION})
    partner = a["partner"].get("url", "") if "zweiter-rechner" in a["bausteine"] else ""
    w["partner"] = partner
    w["partner_ip"] = re.sub(r"^http://([0-9.]+):\d+/?$", r"\1", partner) if partner else ""
    w["pool_wait"] = a["partner"].get("pool_wait_seconds") or ""
    loras = []
    for mid in reihe:
        m = alle[mid][1]
        if m.get("art") == "lora" and m.get("lora_fuer") == "qwen-image-2.1":
            st = [d["pfad"] for d in m["dateien"] if d["pfad"].lower().endswith(".safetensors")]
            if st: loras.append(f"{m['lora_name']}={p['modelle']}\\{st[0]}")
    w["lora_qwen_image"] = ";".join(loras)
    return w

def flags_fuer(a):
    f = {str(a["ram_gb"])}
    f |= {k for k, v in a["schalter"].items() if v}
    f.add(a["router"]["erreichbarkeit"])   # heimnetz | nur-dieser-rechner: [router]-Abschnitt in router.ini
    return f

def partner_modelle(a, reihe, alle):
    if "zweiter-rechner" not in a["bausteine"]: return []
    pm = a["partner"].get("modelle", "alle-lokalen")
    if pm == "alle-lokalen":
        return [mid for mid in reihe if alle[mid][1].get("abschnitt") and alle[mid][1].get("art") != "lora"]
    return list(pm)

def baue_preset(a, reihe, alle, werte, flags):
    out = fuelle(lies_vorlage("router")[0]["zeilen"], werte, flags) + [""]
    vorlagen = {b: {bl["modell"]: bl for bl in lies_vorlage(b)} for b in REIHENFOLGE if b != "zweiter-rechner"}
    for mid in reihe:
        b, m = alle[mid]
        if m.get("art") == "lora": continue
        bl = vorlagen[b].get(mid)
        if bl is None: fehler(f"keine Vorlage fuer {mid} in bausteine\\{b}.ini")
        out += fuelle(bl["zeilen"], werte, flags) + [""]
    pm = partner_modelle(a, reihe, alle)
    if pm:
        bl = lies_vorlage("zweiter-rechner")[0]
        for mid in pm:
            # [remote/<id>] muss den lokalen Abschnitt treffen (moss-music-8b -> [external/moss-music]).
            sid = (alle[mid][1].get("abschnitt") or mid).split("/")[-1]
            out += fuelle(bl["zeilen"], dict(werte, id=sid), flags) + [""]
    return "\n".join(out) + "\n", pm

def fremde_bloecke(alt_text):
    """Alles aus einem vorhandenen Preset, das nicht zu einem '# baustein:'-Block gehoert."""
    fremd, eigen, kopf = [], False, False
    for z in alt_text.splitlines():
        if z.startswith("# baustein:"):
            eigen = True; kopf = False; continue
        if eigen:
            if z.startswith("["): kopf = True; continue
            if kopf and not z.strip(): eigen = False; continue
            continue
        fremd.append(z)
    while fremd and not fremd[-1].strip(): fremd.pop()
    return [z for z in fremd if not z.startswith("# --- kept from the previous preset")]

def start_cmd(werte, preset_text):
    start = opts = ""; env = []
    for z in preset_text.splitlines():
        if z.startswith("# start: "): start = z[len("# start: "):]
        elif z.startswith("# start-opt: "): opts += " " + z[len("# start-opt: "):]
        elif z.startswith("# env: "): env.append(z[len("# env: "):])
    zeilen = ["@echo off", f"rem gufo router, written by einrichtung\\einrichten.py {werte['einrichtung_version']} on {werte['zeit']} (KI-unterstuetzt).",
              "rem Logs: router.out.log / router.err.log next to the preset.", "chcp 65001 >nul",
              f'cd /d "{werte["app"]}"'] + [f"set {e}" for e in env] + [
              f'"{werte["app"]}\\gufo.exe" ' + start.split("gufo.exe router", 1)[1].strip().join(["router ", ""]) + opts
              + f' 1> "{pathlib.Path(werte["preset"]).parent}\\router.out.log" 2> "{pathlib.Path(werte["preset"]).parent}\\router.err.log"']
    return "\r\n".join(zeilen) + "\r\n"

# ------------------------------------------------------------------------------------------------- Bericht
def gb(n): return f"{n / 1e9:.2f} GB"

def bericht(a, katalog, reihe, alle, hinweise, pr, worker_fehlend, pm, lizenzen, ziel, geschrieben):
    z = [f"# Einrichtung {a['rechner']} ({a['ram_gb']} GB) – {datetime.datetime.now().isoformat(timespec='minutes')}", "",
         f"einrichten.py {EINRICHTUNG_VERSION}, Katalog {katalog['katalog_version']}. Modelle: {a['pfade']['modelle']}; App: {a['pfade']['app']}; Daten: {a['pfade']['daten']}.",
         f"Router: Port {a['router']['port']}, {a['router']['erreichbarkeit']}, Leerlauf {a['router']['idle_seconds']} s, GUFO_HOST_RESERVE_MIB={a['router']['reserve_mib']}.", "",
         "## Bausteine und Modelle", ""]
    for b in a["bausteine"]:
        ms = [mid for mid in reihe if alle[mid][0] == b]
        summe = 0; seen = set()
        for mid in ms:
            for d in alle[mid][1]["dateien"]:
                key = d.get("hardlink_auf", d["pfad"])
                if key not in seen and d["bytes"]: seen.add(key); summe += d["bytes"]
        z.append(f"- **{b}** ({katalog['bausteine'][b]['titel']}): {len(ms)} Modelle, {gb(summe)}")
        for mid in ms:
            m = alle[mid][1]
            art = "Pflicht" if m["pflicht"] else ("Vorgabe" if m.get("vorgabe_ram") == a["ram_gb"] else "gewaehlt")
            z.append(f"  - {mid} ({art}, {gb(m['bytes'])}, Worker {m['worker']}, passt in 64 GB: {m['passt_64gb']}) – {m['lizenz']['name']}")
    if pm: z.append(f"- **zweiter-rechner**: {len(pm)} [remote/...]-Zwillinge auf {a['partner'].get('url')}: {', '.join(pm)}")
    for h in hinweise: z.append(f"- Hinweis: {h}")
    z += ["", "## Dateien im Zielbaum", "", f"- vorhanden: {gb(pr['bytes_da'])}; fehlend oder falsche Groesse: {gb(pr['bytes_fehlend'])} (nur geprueft, nicht geladen)"]
    for mid, pfad, soll, quellen in pr["fehlend"]:
        passend = [x for x in quellen if pfad.lower().startswith(x["ordner"].lower() + "\\")] or quellen
        q = next((f"{x['hf_repo']}@{x['hf_revision'] or 'main (nicht gepinnt)'}" for x in passend if x["hf_repo"]), "Quelle nicht gepinnt: " + (passend[0]["hinweis"] or passend[0]["angabe_von"] if passend else "?"))
        z.append(f"  - FEHLT {pfad} ({gb(soll)}; {mid}; {q})")
    for mid, pfad, soll, ist in pr["falsche_groesse"]:
        z.append(f"  - GROESSE {pfad}: erwartet {soll} Bytes, vorhanden {ist} ({mid})")
    for mid, dd in pr["daten_fehlend"]:
        z.append(f"  - DATEN fehlt {a['pfade']['daten']}\\{dd} ({mid}; Datendatei, von der .198 kopieren oder Zeile entfernen)")
    if worker_fehlend:
        z += ["", "## Programme", ""]
        for k, ms in worker_fehlend.items():
            z.append(f"- pfade.{k}: gufo.exe nicht gefunden ({a['pfade'].get(k) or 'leer'}) – gebraucht von {', '.join(ms)}")
    z += ["", "## Lizenzen, die bestaetigt werden muessen", ""]
    for mid, l in lizenzen:
        ok = "bestaetigt" if (mid in a["lizenzen_bestaetigt"] or "alle" in a["lizenzen_bestaetigt"]) else "OFFEN"
        z.append(f"- [{ok}] {mid}: {l['name']} – {l['satz']}")
    if not lizenzen: z.append("- keine")
    z += ["", "## Ergebnis", ""]
    z += [f"- {g}" for g in geschrieben] if geschrieben else ["- nur Plan, nichts geschrieben"]
    return "\n".join(z) + "\n"

# ------------------------------------------------------------------------------------------------- Hauptlauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antworten", help="antworten.json (bei --import: eigene Angaben fuer diesen Rechner)")
    ap.add_argument("--katalog", default=str(HIER / "katalog.json"))
    ap.add_argument("--plan", action="store_true", help="nur pruefen und Bericht zeigen")
    ap.add_argument("--schreiben", action="store_true", help="gufo-router.ini, start-router.cmd, einrichtung.json, Bericht schreiben")
    ap.add_argument("--export", help="Antworten + Modellstand fuer den zweiten Rechner nach <datei> schreiben")
    ap.add_argument("--import", dest="import_", help="Export des ersten Rechners einspielen")
    ap.add_argument("--ziel", help="Ausgabeordner (Standard: Ordner von --antworten)")
    ap.add_argument("--preset-ausgabe", help="nur fuer Tests: Preset-Text zusaetzlich in diese Datei schreiben (auch bei --plan)")
    args = ap.parse_args()
    if not (args.plan or args.schreiben or args.export): ap.error("--plan, --schreiben oder --export angeben")

    katalog = lies_json(args.katalog)
    if args.import_:
        export = lies_json(args.import_)
        eigene = lies_json(args.antworten) if args.antworten else {}
        a = antworten_aus_import(export, eigene)
    else:
        if not args.antworten: ap.error("--antworten fehlt")
        a = lies_json(args.antworten)
    pruefe_antworten(a)
    ziel = pathlib.Path(args.ziel) if args.ziel else (pathlib.Path(args.antworten).resolve().parent if args.antworten else pathlib.Path.cwd())

    reihe, alle, hinweise = waehle_modelle(a, katalog)
    werte = werte_fuer(a, reihe, alle); flags = flags_fuer(a)
    pr = pruefe_dateien(a, reihe, alle)
    worker_fehlend = pruefe_worker(a, reihe, alle)
    preset_text, pm = baue_preset(a, reihe, alle, werte, flags)
    lizenzen = [(mid, alle[mid][1]["lizenz"]) for mid in reihe if alle[mid][1]["lizenz"].get("bestaetigen")]
    lizenzen += [(mid, alle[mid][1]["lizenz_zusatz"]) for mid in reihe if alle[mid][1].get("lizenz_zusatz", {}).get("bestaetigen")]
    offen = [mid for mid, _ in lizenzen if not (mid in a["lizenzen_bestaetigt"] or "alle" in a["lizenzen_bestaetigt"])]

    if args.preset_ausgabe:
        pathlib.Path(args.preset_ausgabe).write_text(preset_text, encoding="utf-8")

    geschrieben = []
    if args.schreiben:
        if offen: fehler(f"Lizenzen nicht bestaetigt (lizenzen_bestaetigt in antworten.json): {offen}")
        ziel.mkdir(parents=True, exist_ok=True)
        preset_pfad = pathlib.Path(werte["preset"])
        if a["pfade"].get("preset") is None: preset_pfad = ziel / "gufo-router.ini"; werte["preset"] = str(preset_pfad)
        text = preset_text
        if preset_pfad.exists():
            fremd = fremde_bloecke(preset_pfad.read_text(encoding="utf-8-sig"))
            if fremd:
                text += "\n# --- kept from the previous preset (not managed by einrichten.py) ---\n" + "\n".join(fremd) + "\n"
            sicherung = preset_pfad.with_name(preset_pfad.name + ".bak-" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            preset_pfad.replace(sicherung); geschrieben.append(f"Sicherung: {sicherung}")
        preset_pfad.parent.mkdir(parents=True, exist_ok=True)
        preset_pfad.write_text(text, encoding="utf-8"); geschrieben.append(f"Preset: {preset_pfad}")
        cmd = ziel / "start-router.cmd"; cmd.write_text(start_cmd(werte, text), encoding="utf-8"); geschrieben.append(f"Start: {cmd}")
        for d in ("jobs", "prompt-cache", "video-jobs", "video-jobs-wan", "video-jobs-camera", "video-jobs-talk", "video-jobs-rife", "jobs\\objekt", "jobs\\trellis2",
                  "prompt-cache-35b", "prompt-cache-flashnext", "prompt-cache-flashnext-q3", "prompt-cache-hemmingway", "voices", "expert-profiles"):
            (pathlib.Path(a["pfade"]["daten"]) / d).mkdir(parents=True, exist_ok=True)
        geschrieben.append(f"Datenordner unter {a['pfade']['daten']} angelegt (jobs, prompt-cache*, video-jobs*, voices, expert-profiles)")
        ein = {"einrichtung_version": EINRICHTUNG_VERSION, "katalog_version": katalog["katalog_version"], "zeit": datetime.datetime.now().isoformat(timespec="seconds"),
               "gufo_version": gufo_version(a), "antworten": a, "modelle": reihe, "partner_modelle": pm,
               "dateien_fehlend": [p for _, p, _, _ in pr["fehlend"]], "bytes_fehlend": pr["bytes_fehlend"]}
        (ziel / "einrichtung.json").write_text(json.dumps(ein, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"); geschrieben.append(f"Stand: {ziel / 'einrichtung.json'}")
    if args.export:
        ex = {"einrichtung_version": EINRICHTUNG_VERSION, "katalog_version": katalog["katalog_version"], "zeit": datetime.datetime.now().isoformat(timespec="seconds"),
              "exportiert_von": {"rechner": a["rechner"], "adresse": a["router"].get("adresse") or "ADRESSE-EINTRAGEN", "port": a["router"]["port"],
                                 "modelle": [mid for mid in reihe if alle[mid][1].get("abschnitt") and alle[mid][1].get("art") != "lora"]},
              "antworten": a}
        pathlib.Path(args.export).write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"); geschrieben.append(f"Export: {args.export}")
        if not a["router"].get("adresse"): print("HINWEIS: router.adresse ist leer; der zweite Rechner braucht sie als Partner-Adresse.")

    text = bericht(a, katalog, reihe, alle, hinweise, pr, worker_fehlend, pm, lizenzen, ziel, geschrieben)
    if args.schreiben:
        (ziel / "einrichtung-bericht.md").write_text(text, encoding="utf-8")
    print(text)
    if args.plan and offen: print(f"Vor --schreiben bestaetigen (lizenzen_bestaetigt): {offen}")

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
