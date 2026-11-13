# Vergleicht ein erzeugtes Preset (einrichten.py --plan --preset-ausgabe) mit dem Produktions-Preset:
# je Abschnitt die Schluessel und Werte (Kommentare und Leerzeilen zaehlen nicht). Nur lesen.
#   python vergleich_preset.py --produktion D:\llm-modelle\gufo\gufo-router-combo7.ini --erzeugt <plan.ini> [--ignorieren a,b]
import argparse, pathlib, re

def lies(p):
    abschnitte, akt = {}, None
    for z in pathlib.Path(p).read_text(encoding="utf-8-sig").splitlines():
        s = z.strip()
        if not s or s.startswith("#"): continue
        m = re.match(r"\[([^\]]+)\]", s)
        if m: akt = m.group(1).strip(); abschnitte.setdefault(akt, {}); continue
        k, _, v = s.partition("=")
        if akt is None: continue
        abschnitte[akt][k.strip()] = v.strip()
    return abschnitte

ap = argparse.ArgumentParser()
ap.add_argument("--produktion", default=r"D:\llm-modelle\gufo\gufo-router-combo7.ini")
ap.add_argument("--erzeugt", required=True)
ap.add_argument("--ignorieren", default="", help="Abschnitte, die im Produktions-Preset absichtlich fehlen duerfen (Komma)")
a = ap.parse_args()
P, E = lies(a.produktion), lies(a.erzeugt)
ign = {s.strip() for s in a.ignorieren.split(",") if s.strip()}
nur_p = [s for s in P if s not in E and s not in ign]
nur_e = [s for s in E if s not in P]
print(f"Produktion: {len(P)} Abschnitte, erzeugt: {len(E)} Abschnitte, gemeinsam: {len(set(P) & set(E))}")
print("nur in Produktion:", nur_p or "-")
print("nur erzeugt:", nur_e or "-")
gleich = 0
for s in P:
    if s not in E: continue
    p, e = P[s], E[s]
    fehlt = [k for k in p if k not in e]; extra = [k for k in e if k not in p]
    anders = [(k, p[k], e[k]) for k in p if k in e and p[k] != e[k]]
    if not fehlt and not extra and not anders: gleich += 1; continue
    print(f"[{s}]")
    for k in fehlt: print(f"   fehlt im erzeugten: {k} = {p[k]}")
    for k in extra: print(f"   zusaetzlich erzeugt: {k} = {e[k]}")
    for k, pv, ev in anders: print(f"   anders: {k}\n      prod: {pv}\n      neu:  {ev}")
print(f"identische Abschnitte (Schluessel und Werte): {gleich} von {len(set(P) & set(E))}")
