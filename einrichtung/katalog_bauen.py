# Baut einrichtung\katalog.json aus dem echten Modellbaum der .198 (nur lesen).
# Groessen kommen aus dem Dateisystem, SHA-256 nur aus vorhandenen Quellen (H3-Manifest, Qwen-Image .tree.json,
# IndexTTS SHA256SUMS.txt); alles andere bleibt leer und ist als "fehlt" markiert. GPU-Werte: aus den Kommentaren
# des Produktions-Presets (gpu_gb_notiz) und aus den vom Router gemessenen Spitzen in <preset>.memory.json
# (gpu_gb_router, learned_*). Nichts wird geschaetzt. KI-unterstuetzt (Claude Code), Baumeister .198.
#
#   python katalog_bauen.py [--modelle D:\llm-modelle\gufo\modelle] [--memory D:\llm-modelle\gufo\gufo-router-combo7.memory.json]
#                           [--h3-manifest <pfad>] [--ziel katalog.json]
import argparse, datetime, json, pathlib

HIER = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--modelle", default=r"D:\llm-modelle\gufo\modelle")
ap.add_argument("--memory", default=r"D:\llm-modelle\gufo\gufo-router-combo7.memory.json")
ap.add_argument("--h3-manifest", default=r"D:\llm-modelle\gufo\app\gufo-2.0.2-2026-11-13\MINIMAX_H3_FL2VA_BF16.source-manifest.json")
ap.add_argument("--ziel", default=str(HIER / "katalog.json"))
a = ap.parse_args()
WURZEL = pathlib.Path(a.modelle)

KATALOG_VERSION = "1.1 (2026-11-14)"
AUSSCHLUSS_NAMEN = {".gitattributes", ".files.txt", ".tree.json", "download.done"}
AUSSCHLUSS_ENDUNGEN = {".log", ".pyc", ".lock"}
AUSSCHLUSS_ORDNER = {".cache", "__pycache__", "assets", "examples", "figures"}

# ---------------------------------------------------------------------------------------------------------------
# Lizenzen (Name + ein Satz). Quelle jeweils genannt; "nicht geprueft" heisst: keine Lizenzdatei/-angabe gelesen.
L = {
    "apache": {"name": "Apache-2.0", "satz": "Frei, auch kommerziell; Lizenztext bei Weitergabe beilegen.", "bestaetigen": False},
    "mit": {"name": "MIT", "satz": "Frei, auch kommerziell; Hinweis beilegen.", "bestaetigen": False},
    "bsd3": {"name": "BSD-3-Clause", "satz": "Frei, auch kommerziell; Hinweis beilegen.", "bestaetigen": False},
    "ungeprueft": {"name": "nicht geprueft", "satz": "Keine Lizenzangabe im Ordner gelesen (KB 1127: ungeprueft); vor kommerzieller Nutzung pruefen.", "bestaetigen": True},
    "qwen-research": {"name": "Qwen Research License (20.09.2026)", "satz": "Qwen-Image 2.1: nur Forschung / nicht-kommerziell; kommerzielle Lizenz muss gesondert beantragt werden (Anfrage offen, KB 1127).", "bestaetigen": True},
    "cc-by-nc": {"name": "CC BY-NC 4.0", "satz": "Hemmingway-1.5: nur nicht-kommerziell; kommerziell nur mit Vertrag (KB 1127).", "bestaetigen": True},
    "fish": {"name": "Fish Audio Community License 1.0 (13.10.2026)", "satz": "Fish Drama-3: kommerziell nur unter 500.000 US$ Jahresumsatz; kein gehosteter Dienst fuer Fremde, Wasserzeichen behalten, Ausgaben als KI kennzeichnen (KB 1126).", "bestaetigen": True},
    "bilibili": {"name": "bilibili Model Use License", "satz": "IndexTTS-2.5-German: kommerziell frei bis 100 Mio. monatliche Nutzer / 1 Mrd. RMB Umsatz; Restrisiko HUI-Audio-Korpus beim deutschen Feinschliff (KB 1128).", "bestaetigen": True},
    "gpl3": {"name": "GPL-3.0 (Seed-VC)", "satz": "Nur intern; nie in oeffentliche gufo-Pakete (gufo ist MIT). Fabian 11.11.", "bestaetigen": True},
    "minimax": {"name": "MiniMax H3 Community License Agreement (02.08.2026)", "satz": "Zugang bei MiniMax beantragen (gated); KI-Kennzeichnung und \"MiniMax H3\" in der Beschreibung nennen (KB 1127). gufo verteilt keine Gewichte.", "bestaetigen": True},
    "trellis": {"name": "MIT + DINOv3 License (Meta, 19.08.2025)", "satz": "TRELLIS.2 ist MIT; der Bildencoder DINOv3 hat die Meta-DINOv3-Lizenz: \"Built with DINOv3\" nennen und die Lizenzkopie beilegen.", "bestaetigen": True},
    "moss-image": {"name": "Apache-2.0", "satz": "MOSS-Image SFT: Apache-2.0 laut LICENSE und Modellkarte. Achtung: rendert ohne Inhaltsfilter.", "bestaetigen": False},
    "moss-music": {"name": "Apache-2.0 (laut Modellkarte)", "satz": "MOSS-Music-8B-Instruct: Apache-2.0 laut Modellkarte (Preset-Kommentar 12.11.); Lizenztext liegt im HF-Repo D3rPapst/moss-music-8b-gufo bei (LICENSE).", "bestaetigen": False},
    "wan-apache": {"name": "Apache-2.0 (Wan 2.1/2.2, lightx2v-LoRA; Kijai-fp8 erbt)", "satz": "Wan-Gewichte Apache-2.0 (KB 1128); Kijai/WanVideo_comfy-Umwandlungen ohne eigene Lizenz, erben Apache-2.0.", "bestaetigen": False},
    "uni3c": {"name": "Apache-2.0 (Wan) + Uni3C nicht geprueft + MoGe-2 MIT", "satz": "Wan 2.1 I2V und lightx2v Apache-2.0 (KB 1128), MoGe-2 MIT (Modellkarte); Uni3C-ControlNet: Lizenz nicht geprueft.", "bestaetigen": True},
    "infinitetalk": {"name": "Apache-2.0 (Wan, InfiniteTalk, lightx2v) + MIT (wav2vec2)", "satz": "Alles kommerziell frei (KB 1128); Lizenztexte bei Weitergabe beilegen.", "bestaetigen": False},
    "rife": {"name": "MIT", "satz": "Practical-RIFE IFNet 4.7/4.9, MIT (hzwer); Quelle der .pth: Spiegel marduk191/rife ohne eigene Lizenzangabe (Lizenzpruefung 13.11.2026, hf-upload\\LIZENZPRUEFUNG-2026-11-13.md); Lizenztext im Repo D3rPapst/rife-ifnet-gufo.", "bestaetigen": False},
    "h3-turbo": {"name": "nicht geprueft (lightx2v-Repo)", "satz": "Turbo-LoRA: \"Model assets are governed by the repository license\" (README); Lizenz nicht gelesen.", "bestaetigen": True},
}

def lic(k, **extra):
    d = dict(L[k]); d.update(extra); return d

# ---------------------------------------------------------------------------------------------------------------
# Modelle je Baustein. Felder:
#   id          served-model-name = Modell-ID im Router
#   abschnitt   Preset-Abschnitt
#   pflicht     True = kommt mit dem Baustein; False = optional
#   worker      gufo.exe | router | 35b | gufo-3d | gufo-moss | gufo-d | trellis2  (siehe LIES-MICH)
#   ordner      Ordner im Baum modelle\<Hersteller>\<Modell>
#   dateien     Liste relativer Pfade (einzelne Dateien) und/oder Ordner mit "/" am Ende (alle Dateien darunter)
#   quellen     je Ordner: Hugging-Face-Repo + Revision + woher die Angabe stammt
#   gpu_gb_notiz  gemessener Wert aus Preset-Kommentar / Release-Notes (Quelle dabei), sonst None
#   passt_64gb  ja | nein | nur-128  (Plan Abschnitt 4 Schritt 2, mit Hinweis wo unsere Messung abweicht)
#   vorgabe_ram optional: 64 oder 128 = nur bei dieser RAM-Groesse vorausgewaehlt
Q = lambda ordner, repo="", rev="", von="", hinweis="": {"ordner": ordner, "hf_repo": repo, "hf_revision": rev, "angabe_von": von, "hinweis": hinweis}

# Unsere eigenen Repos (huggingface.co/D3rPapst, Upload 13.11.2026): Repo, Revision und die SHA-256 je Datei stehen in
# sha256-eigene-repos.json (Belege dort); die Q(...)-Eintraege unten nennen dieselben Repos fuer einrichten.py.
EIGENE = json.loads((HIER / "sha256-eigene-repos.json").read_text(encoding="utf-8"))
EIGENE_REPOS = EIGENE["repos"]; EIGENE_DATEIEN = EIGENE["dateien"]
def QE(ordner, hinweis):
    r = EIGENE_REPOS[ordner]
    return Q(ordner.replace("\\", "/"), r["hf_repo"], r["hf_revision"], "hf-upload\\upload-2026-11-13.log (" + r["commit_log"] + "), Karte", hinweis)
RIFE_Q = QE("rife", "rife49.safetensors aus unserem Repo (MIT); lokal unter rife\\weights\\")

BAUSTEINE = {
 "chat": {"titel": "Chat (Text)", "beschreibung": "Qwen3.8-27B mit DFlash2; optional Flash-Next (Q3 fuer 64 GB, Q4 fuer 128 GB), Qwen3.6-35B-A3B, Hemmingway-1.5.",
  "modelle": [
   {"id": "qwen3.8-27b", "abschnitt": "external/qwen3.8-27b", "name": "Qwen3.8-27B UD-Q4_K_XL + mmproj + DFlash2", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["unsloth/Qwen3.8-27B-GGUF", "z-lab/Qwen3.8-27B-DFlash2-GGUF"],
    "dateien": ["unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf", "unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf", "z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"],
    "quellen": [Q("unsloth/Qwen3.8-27B-GGUF", "unsloth/Qwen3.8-27B-GGUF", "4ca720788d1e01f1bff70c033e0d0028fd02e502", "docs/models/qwen3.8-27b.md"),
                Q("z-lab/Qwen3.8-27B-DFlash2-GGUF", "z-lab/Qwen3.8-27B-DFlash2-GGUF", "2d9571f8ce46e151f61c6499c99dee6079e1d610", "docs/models/qwen3.8-27b.md")],
    "lizenz": lic("ungeprueft", name="nicht geprueft (Qwen3.8: Apache-2.0 laut THIRD_PARTY_NOTICES fuer die Chat-Vorlage; KB 1127: ungeprueft)"),
    "gpu_gb_notiz": None, "passt_64gb": "ja", "daten_dateien": []},
   {"id": "qwen3.8-flash-next-q3", "abschnitt": "external/qwen3.8-flash-next-q3", "name": "Qwen3.8-Flash-Next UD-Q3_K_XL + MTP", "pflicht": False, "worker": "gufo.exe", "vorgabe_ram": 64,
    "ordner": ["unsloth/Qwen3.8-Flash-Next-GGUF"],
    "dateien": ["unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/", "unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"],
    "quellen": [Q("unsloth/Qwen3.8-Flash-Next-GGUF", "unsloth/Qwen3.8-Flash-Next-GGUF", "", "docs/models/qwen3.8-flash-next.md (Revision dort nur fuer UD-Q4_K_XL gepinnt)")],
    "lizenz": lic("ungeprueft"), "gpu_gb_notiz": None, "passt_64gb": "ja",
    "hinweis": "Gemessen auf der .198 (64 GB): 24,3-28,7 tok/s (Release-Notes 2.0.1/2.0.2) mit expert-cache-bytes 24 GB, cache-ram-bytes off, GUFO_HOST_RESERVE_MIB=2048."},
   {"id": "qwen3.8-flash-next", "abschnitt": "external/qwen3.8-flash-next", "name": "Qwen3.8-Flash-Next UD-Q4_K_XL + MTP", "pflicht": False, "worker": "gufo.exe", "vorgabe_ram": 128,
    "ordner": ["unsloth/Qwen3.8-Flash-Next-GGUF"],
    "dateien": ["unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/", "unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"],
    "quellen": [Q("unsloth/Qwen3.8-Flash-Next-GGUF", "unsloth/Qwen3.8-Flash-Next-GGUF", "d96201194f103c906994f283f504209dc8d2a10c", "docs/models/qwen3.8-flash-next.md")],
    "lizenz": lic("ungeprueft"), "gpu_gb_notiz": 31.7, "gpu_gb_notiz_quelle": "Preset-Kommentar 2026-10-26b (24 GB Cache, MTP): GPU 31.7 GB", "passt_64gb": "nur-128",
    "hinweis": "Plan Schritt 2: Q4 fuer 128 GB. Laeuft auf der .198 (64 GB) mit expert-cache-bytes 24 GB, cache-ram-bytes off und GUFO_HOST_RESERVE_MIB=2048 (14,2-15,7 tok/s), braucht aber ~8,5 GB freien Host-RAM zum Laden (Release-Notes 2.0.2). Vorgabe fuer 64 GB ist Q3.",
    "daten_dateien": ["expert-profiles/flash-next.prof"]},
   {"id": "qwen3.6-35b-a3b", "abschnitt": "external/qwen3.6-35b-a3b", "name": "Qwen3.6-35B-A3B Q4_K_M (eigener 35B-Worker)", "pflicht": False, "worker": "35b",
    "ordner": ["lmstudio-community/Qwen3.6-35B-A3B-GGUF"], "dateien": ["lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"],
    "quellen": [Q("lmstudio-community/Qwen3.6-35B-A3B-GGUF", "lmstudio-community/Qwen3.6-35B-A3B-GGUF", "", "Ordnername (nicht gepinnt)")],
    "lizenz": lic("ungeprueft"), "gpu_gb_notiz": 24.5, "gpu_gb_notiz_quelle": "Preset-Kommentar 31.10.2026: GPU peak 24.5 GiB (ohne DFlash2)", "passt_64gb": "ja"},
   {"id": "qwen3.6-35b-a3b-json", "abschnitt": "external/qwen3.6-35b-a3b-json", "name": "Qwen3.6-35B-A3B + DFlash2 (JSON/Film)", "pflicht": False, "worker": "35b",
    "ordner": ["lmstudio-community/Qwen3.6-35B-A3B-GGUF", "aminya/Qwen3.6-35B-A3B-DFlash2-GGUF"],
    "dateien": ["lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf", "aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf"],
    "quellen": [Q("lmstudio-community/Qwen3.6-35B-A3B-GGUF", "lmstudio-community/Qwen3.6-35B-A3B-GGUF", "", "Ordnername (nicht gepinnt)"),
                Q("aminya/Qwen3.6-35B-A3B-DFlash2-GGUF", "aminya/Qwen3.6-35B-A3B-DFlash2-GGUF", "", "Ordnername; README: base_model incoai/Qwen3.6-35B-A3B-DFlash2")],
    "lizenz": lic("ungeprueft", name="nicht geprueft (DFlash2-Draft: Apache-2.0 laut README)"), "gpu_gb_notiz": 25.4, "gpu_gb_notiz_quelle": "Preset-Kommentar 31.10.2026: GPU peak 25.4 GiB", "passt_64gb": "ja"},
   {"id": "hemmingway-1.5", "abschnitt": "external/hemmingway-1.5", "name": "Hemmingway-1.5 i1-Q4_K_M (Erzaehler, 27B-Feintuning) + DFlash2", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["mradermacher/Hemmingway-1.5-i1-GGUF", "z-lab/Qwen3.8-27B-DFlash2-GGUF"],
    "dateien": ["mradermacher/Hemmingway-1.5-i1-GGUF/Hemmingway-1.5.i1-Q4_K_M.gguf", "z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"],
    "quellen": [Q("mradermacher/Hemmingway-1.5-i1-GGUF", "mradermacher/Hemmingway-1.5-i1-GGUF", "", "Notiz gufo-modell-liste (nicht gepinnt)"),
                Q("z-lab/Qwen3.8-27B-DFlash2-GGUF", "z-lab/Qwen3.8-27B-DFlash2-GGUF", "2d9571f8ce46e151f61c6499c99dee6079e1d610", "docs/models/qwen3.8-27b.md")],
    "lizenz": lic("cc-by-nc"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
  ]},
 "bilder": {"titel": "Bilder", "beschreibung": "Qwen-Image 2.1 + Turbo; optional MOSS-Image und LoRAs.",
  "modelle": [
   {"id": "qwen-image-2.1", "abschnitt": "external/qwen-image-2.1", "name": "Qwen-Image-2.1 (BF16, Hugging-Face-Ordner)", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Qwen/Qwen-Image-2.1"], "dateien": ["Qwen/Qwen-Image-2.1/"],
    "quellen": [Q("Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1", "b3179ad355be050328e483a9dfdd9e60cd62adfa", "docs/models/qwen-image-2.1.md, THIRD_PARTY_NOTICES.md")],
    "lizenz": lic("qwen-research"), "gpu_gb_notiz": 30.2, "gpu_gb_notiz_quelle": "Preset-Kommentar 31.10.2026: GPU peak 30.0-30.2 GiB; 2.0.2 nach dem Job 22.0 GB (Release-Notes)", "passt_64gb": "ja"},
   {"id": "qwen-image-2.1-turbo", "abschnitt": "external/qwen-image-2.1-turbo", "name": "Qwen-Image-2.1 Turbo (8 Schritte)", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Qwen/Qwen-Image-2.1-Turbo"], "dateien": ["Qwen/Qwen-Image-2.1-Turbo/"],
    "quellen": [Q("Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1-Turbo", "fc13ba1986fcb905854dc8de2f52eee395f49b5b", "docs/models/qwen-image-2.1.md")],
    "lizenz": lic("qwen-research"), "gpu_gb_notiz": 30.2, "gpu_gb_notiz_quelle": "Preset-Kommentar 31.10.2026: same GPU peak as the base (30.0-30.2 GiB)", "passt_64gb": "ja",
    "hinweis": "text_encoder\\ ist identisch mit Qwen-Image-2.1 (auf der .198 Hardlinks; zaehlt in der Summe nur einmal)."},
   {"id": "lora-anime-consistency", "art": "lora", "lora_name": "Qwen2.1_Anime_consistency", "lora_fuer": "qwen-image-2.1", "name": "LoRA Qwen2.1 Anime consistency", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["loras/qwen21-anime-consistency"], "dateien": ["loras/qwen21-anime-consistency/"],
    "quellen": [Q("loras/qwen21-anime-consistency", "", "", "", "README: base_model Qwen/Qwen-Image-2.1, Repo nicht notiert")],
    "lizenz": lic("apache", satz="Apache-2.0 laut README der LoRA."), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "lora-multiple-angles", "art": "lora", "lora_name": "winkel", "lora_fuer": "qwen-image-2.1", "name": "LoRA Qwen2.1 multiple angles (Name 'winkel' wie auf der .56, Regie #56-709)", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["loras/qwen21-multiple-angles"], "dateien": ["loras/qwen21-multiple-angles/"],
    "quellen": [Q("loras/qwen21-multiple-angles", "", "", "", "kein README im Ordner")],
    "lizenz": lic("ungeprueft"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "moss-image", "abschnitt": "external/moss-image", "name": "MOSS-Image SFT (ohne Inhaltsfilter)", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["OpenMOSS-Team/MOSS-Image-SFT"], "dateien": ["OpenMOSS-Team/MOSS-Image-SFT/"],
    "quellen": [Q("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-SFT", "", "Ordnername + README (nicht gepinnt); text_encoder.gguf lokal aus text_encoder-bf16.gguf (Hardlink)")],
    "lizenz": lic("moss-image"), "gpu_gb_notiz": None, "passt_64gb": "ja",
    "hinweis": "2.0 liest <model>\\text_encoder.gguf (Hardlink auf text_encoder-bf16.gguf, zaehlt einmal). Ob der Worker die HF-Originale unter text_encoder\\ (16,8 GB) noch braucht, ist nicht geprueft."},
  ]},
 "stimme": {"titel": "Stimme", "beschreibung": "Qwen3-TTS (3 Varianten) + Qwen3-ASR; optional Fish Drama-3, IndexTTS-2.5 German, VoxCPM2, Ming; Seed-VC nur intern (GPL).",
  "modelle": [
   {"id": "qwen3-tts", "abschnitt": "external/qwen3-tts", "name": "Qwen3-TTS 1.7B CustomVoice", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], "dateien": ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/"],
    "quellen": [Q("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "", "docs/models/qwen3-tts.md (Base) + Ordnername")],
    "lizenz": lic("apache"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "qwen3-tts-voicedesign", "abschnitt": "external/qwen3-tts-voicedesign", "name": "Qwen3-TTS 1.7B VoiceDesign", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"], "dateien": ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"],
    "quellen": [Q("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "", "Ordnername")],
    "lizenz": lic("apache"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "qwen3-tts-clone", "abschnitt": "external/qwen3-tts-clone", "name": "Qwen3-TTS 1.7B Base (Stimmklon)", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"], "dateien": ["Qwen/Qwen3-TTS-12Hz-1.7B-Base/"],
    "quellen": [Q("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "", "docs/models/qwen3-tts.md")],
    "lizenz": lic("apache"), "gpu_gb_notiz": None, "passt_64gb": "ja", "daten_dateien": ["voices/erzaehler.wav", "voices/erzaehler.txt"]},
   {"id": "qwen3-asr-1.7b", "abschnitt": "external/qwen3-asr-1.7b", "name": "Qwen3-ASR 1.7B (Spracherkennung)", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Qwen/Qwen3-ASR-1.7B"], "dateien": ["Qwen/Qwen3-ASR-1.7B/"],
    "quellen": [Q("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5", "docs/models/qwen3-asr.md")],
    "lizenz": lic("apache"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "fish-drama-3", "abschnitt": "external/fish-drama-3", "name": "Fish Audio Drama-3 (35B-A3B)", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["fishaudio/drama-3"], "dateien": ["fishaudio/drama-3/"],
    "quellen": [Q("fishaudio/drama-3", "fishaudio/drama-3", "", "Notiz gufo-modell-liste + Ordnername (nicht gepinnt)")],
    "lizenz": lic("fish"), "gpu_gb_notiz": None, "passt_64gb": "ja",
    "hinweis": "Laeuft auf der .198 (64 GB) allein (Modell-Pruefung 2.0.0/2.0.2), aber nicht neben dem 27B (Preset-Kommentar)."},
   {"id": "indextts-2.5-german", "abschnitt": "external/indextts-2.5-german", "name": "IndexTTS-2.5 German (Buendel)", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["IndexTeam/IndexTTS-2.5-german-bundle"], "dateien": ["IndexTeam/IndexTTS-2.5-german-bundle/"],
    "quellen": [Q("IndexTeam/IndexTTS-2.5-german-bundle", "", "", "", "Buendel aus IndexTeam/IndexTTS-2.5 + deutschem GPT (sharrnah); Repos nicht gepinnt; german/SHA256SUMS.txt liegt bei")],
    "lizenz": lic("bilibili"), "gpu_gb_notiz": None, "passt_64gb": "ja", "daten_dateien": ["voices/erzaehler-9s.wav"]},
   {"id": "voxcpm2-tts", "abschnitt": "external/voxcpm2-tts", "name": "VoxCPM2 (48 kHz, Klon)", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["openbmb/VoxCPM2"], "dateien": ["openbmb/VoxCPM2/"],
    "quellen": [Q("openbmb/VoxCPM2", "openbmb/VoxCPM2", "", "Ordnername + Download-Log")],
    "lizenz": lic("apache"), "gpu_gb_notiz": None, "passt_64gb": "ja", "daten_dateien": ["voices/erzaehler-9s.wav"]},
   {"id": "ming-tts", "abschnitt": "external/ming-tts", "name": "Ming-3.0-tts 0.5B", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["inclusionAI/Ming-3.0-tts-0.5B"], "dateien": ["inclusionAI/Ming-3.0-tts-0.5B/"],
    "quellen": [Q("inclusionAI/Ming-3.0-tts-0.5B", "inclusionAI/Ming-3.0-tts-0.5B", "", "Ordnername + Download-Log")],
    "lizenz": lic("mit"), "gpu_gb_notiz": 1.9, "gpu_gb_notiz_quelle": "Preset-Kommentar 02.11.2026: ~1.9 GB VRAM", "passt_64gb": "ja", "daten_dateien": ["voices/erzaehler-9s.wav"]},
   {"id": "gufo-voice-convert", "abschnitt": "external/gufo-voice-convert", "name": "Seed-VC Stimm-Umwandlung (nur intern, GPL)", "pflicht": False, "worker": "gufo-d", "oeffentlich": False,
    "ordner": ["voiceconv"], "dateien": ["voiceconv/"],
    "quellen": [Q("voiceconv", "", "", "", "Seed-VC, HTDemucs, RMVPE, BigVGAN, Whisper, CAM++ aus verschiedenen Quellen; nicht gepinnt (RECHERCHE-SVC-2026-11-11.md)")],
    "lizenz": lic("gpl3"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
  ]},
 "video": {"titel": "Video", "beschreibung": "Wan 2.2 TI2V-5B; optional Wan 2.1 Kamera (Uni3C), InfiniteTalk, MiniMax H3 (nur 128 GB).",
  "modelle": [
   {"id": "wan2.2-ti2v-5b", "abschnitt": "external/wan2.2-ti2v-5b", "name": "Wan 2.2 TI2V-5B (Diffusers)", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "rife"], "dateien": ["Wan-AI/Wan2.2-TI2V-5B-Diffusers/", "rife/weights/rife49.safetensors"],
    "quellen": [Q("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "", "Ordnername + Download-Log"),
                RIFE_Q],
    "lizenz": lic("wan-apache", name="Apache-2.0"), "gpu_gb_notiz": 15.0, "gpu_gb_notiz_quelle": "Preset-Kommentar: GPU up to ~15 GiB", "passt_64gb": "ja"},
   {"id": "wan2.1-uni3c", "abschnitt": "external/wan2.1-uni3c", "name": "Wan 2.1 I2V 14B + Uni3C Kamerafahrten + MoGe-2", "pflicht": False, "worker": "gufo.exe", "benoetigt": ["wan2.2-ti2v-5b"],
    "ordner": ["wan21-camera"],
    "dateien": ["wan21-camera/gufo-camera.json", "wan21-camera/Wan2.1-I2V-14B-480P-Diffusers/", "wan21-camera/kijai/I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                "wan21-camera/kijai/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "wan21-camera/kijai/Wan21_Uni3C_controlnet_fp16.safetensors",
                "wan21-camera/moge-2-vitl-normal/moge2_vitl.safetensors", "wan21-camera/moge-2-vitl-normal/README.md", "rife/weights/rife49.safetensors"],
    "quellen": [Q("wan21-camera/Wan2.1-I2V-14B-480P-Diffusers", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "", "Ordnername (nur Teilordner: image_encoder, vae, Konfiguration)"),
                Q("wan21-camera/kijai", "Kijai/WanVideo_comfy", "", "KB 1128 (nicht gepinnt)"),
                Q("wan21-camera/moge-2-vitl-normal", "microsoft/moge-2-vitl-normal", "", "Ordnername; moge2_vitl.safetensors lokal aus model.pt umgewandelt"), RIFE_Q],
    "lizenz": lic("uni3c"), "gpu_gb_notiz": 27.0, "gpu_gb_notiz_quelle": "Preset-Kommentar: GPU ~27 GiB", "passt_64gb": "ja",
    "hinweis": "T5 kommt aus Wan2.2-TI2V-5B-Diffusers (gufo-camera.json), deshalb braucht dieses Modell wan2.2-ti2v-5b."},
   {"id": "wan2.1-infinitetalk-14b", "abschnitt": "external/wan2.1-infinitetalk-14b", "name": "Wan 2.1 InfiniteTalk 14B (Lippensynchron)", "pflicht": False, "worker": "gufo.exe", "benoetigt": ["wan2.2-ti2v-5b", "wan2.1-uni3c"],
    "ordner": ["wan21-talk", "Comfy-Org/Wan-2.1", "kijai/wav2vec2_safetensors"],
    "dateien": ["wan21-talk/gufo-talk.json", "Comfy-Org/Wan-2.1/model_patches/wan2.1_infiniteTalk_single_fp16.safetensors", "Comfy-Org/Wan-2.1/model_patches/wan2.1_infiniteTalk_multi_fp16.safetensors",
                "kijai/wav2vec2_safetensors/wav2vec2-chinese-base_fp16.safetensors", "rife/weights/rife49.safetensors"],
    "quellen": [Q("Comfy-Org/Wan-2.1", "", "", "", "Comfy-Org-Repackage (InfiniteTalk fp16), Repo nicht notiert"),
                Q("kijai/wav2vec2_safetensors", "Kijai/wav2vec2_safetensors", "", "Ordnername"), RIFE_Q],
    "lizenz": lic("infinitetalk"), "gpu_gb_notiz": None, "passt_64gb": "ja",
    "hinweis": "gufo-talk.json zeigt auf Wan2.1-I2V-14B-480P-Diffusers, den Kijai-DiT und die lightx2v-LoRA aus wan21-camera (deshalb braucht es wan2.1-uni3c) und auf T5 aus Wan 2.2."},
   {"id": "minimax-h3", "abschnitt": "external/minimax-h3", "name": "MiniMax H3 FL2VA (Video mit Ton) + Turbo-LoRA + Vision-Projektor", "pflicht": False, "worker": "gufo.exe",
    "ordner": ["MiniMaxAI/MiniMax-H3", "MiniMaxAI/MiniMax-H3-vision", "lightx2v/Minimax-h3-Turbo"],
    "dateien": ["MiniMaxAI/MiniMax-H3/", "MiniMaxAI/MiniMax-H3-vision/mmproj-BF16.gguf", "lightx2v/Minimax-h3-Turbo/", "rife/weights/rife49.safetensors"],
    "quellen": [Q("MiniMaxAI/MiniMax-H3", "MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe", "MINIMAX_H3_FL2VA_BF16.source-manifest.json (84 Dateien mit SHA-256)", "gated: Zugang bei MiniMax beantragen"),
                Q("MiniMaxAI/MiniMax-H3-vision", "", "", "", "mmproj-BF16.gguf lokal umgewandelt (Qwen3-VL-32B-Encoder), Quelle nicht notiert"),
                Q("lightx2v/Minimax-h3-Turbo", "lightx2v/Minimax-h3-Turbo", "", "Ordnername + Download-Log"), RIFE_Q],
    "lizenz": lic("minimax"), "lizenz_zusatz": lic("h3-turbo"), "gpu_gb_notiz": 47.0, "gpu_gb_notiz_quelle": "Preset-Kommentar: Peak GPU 31.5 GB dedicated + 15.5 GB shared; nothing else may run beside it", "passt_64gb": "nur-128",
    "hinweis": "Plan Schritt 2: H3 nur 128 GB. Laeuft auf der .198 (64 GB) allein (Host-RAM bis ~3 GB); der Plan begrenzt die Kombination Flash-Next Q4 + H3."},
  ]},
 "musik": {"titel": "Musik", "beschreibung": "ACE-Step 1.5 XL turbo; optional sft/base und MOSS-Music (Tonbewertung).",
  "modelle": [
   {"id": "ace-step-1.5-xl-turbo", "abschnitt": "external/ace-step-1.5-xl-turbo", "name": "ACE-Step 1.5 XL turbo", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["ACE-Step/gufo"],
    "dateien": ["ACE-Step/gufo/dit.safetensors", "ACE-Step/gufo/lm.safetensors", "ACE-Step/gufo/lm-0.6b.safetensors", "ACE-Step/gufo/text_encoder.safetensors", "ACE-Step/gufo/vae.safetensors",
                "ACE-Step/gufo/silence_latent.safetensors", "ACE-Step/gufo/tokenizer.json"],
    "quellen": [QE("ACE-Step\\gufo", "unsere Umwandlung; Quellen ACE-Step/acestep-v15-xl-turbo, acestep-5Hz-lm-0.6B, Comfy-Org/ACE-Step-1.5 (drei Dateien unveraendert), Qwen/Qwen3-Embedding-0.6B (tokenizer.json); Details auf der Karte")],
    "lizenz": lic("mit"), "gpu_gb_notiz": 15.0, "gpu_gb_notiz_quelle": "Preset-Kommentar: GPU 10.7 GB idle, ~15 GB peak", "passt_64gb": "ja",
    "hinweis": "Welche LM-Datei turbo ohne --lm nimmt (lm.safetensors oder lm-0.6b.safetensors), ist nicht geprueft; beide sind aufgefuehrt."},
   {"id": "ace-step-1.5-xl-sft", "abschnitt": "external/ace-step-1.5-xl-sft", "name": "ACE-Step 1.5 XL sft (50 Schritte, LM 1.7B)", "pflicht": False, "worker": "gufo.exe", "benoetigt": ["ace-step-1.5-xl-turbo"],
    "ordner": ["ACE-Step/gufo"], "dateien": ["ACE-Step/gufo/dit-sft.safetensors", "ACE-Step/gufo/lm-1.7b.safetensors"],
    "quellen": [QE("ACE-Step\\gufo", "unsere Umwandlung aus ACE-Step/acestep-v15-xl-sft (Konverter c641db8d/5d50ac23, 11.11.2026); Metadatum gufo.variant=base siehe Lizenzpruefung Zweifelsfall 3")],
    "lizenz": lic("mit"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "ace-step-1.5-xl-base", "abschnitt": "external/ace-step-1.5-xl-base", "name": "ACE-Step 1.5 XL base (extract/lego/complete)", "pflicht": False, "worker": "gufo.exe", "benoetigt": ["ace-step-1.5-xl-turbo"],
    "ordner": ["ACE-Step/gufo"], "dateien": ["ACE-Step/gufo/dit-base.safetensors", "ACE-Step/gufo/lm-1.7b.safetensors"],
    "quellen": [QE("ACE-Step\\gufo", "unsere Umwandlung aus ACE-Step/acestep-v15-xl-base (Konverter c641db8d/5d50ac23, 11.11.2026)")],
    "lizenz": lic("mit"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
   {"id": "moss-music-8b", "abschnitt": "external/moss-music", "name": "MOSS-Music-8B-Instruct (Tonbewertung, eigener Worker)", "pflicht": False, "worker": "gufo-moss",
    "ordner": ["OpenMOSS-Team/moss-music-gufo"], "dateien": ["OpenMOSS-Team/moss-music-gufo/moss-music-8b.llm-Q8_0.gguf", "OpenMOSS-Team/moss-music-gufo/moss-music-8b.audio.gguf"],
    "quellen": [QE("OpenMOSS-Team\\moss-music-gufo", "unsere Umwandlung (Audio-GGUF BF16 + LLM Q8_0, 12.11.2026) aus OpenMOSS-Team/MOSS-Music-8B-Instruct Revision fce7f830; Apache-2.0, Lizenztext im Repo. Die BF16-Originale (32,8 GB) liegen auf der .198 daneben und sind nicht aufgefuehrt.")],
    "lizenz": lic("moss-music"), "gpu_gb_notiz": 10.6, "gpu_gb_notiz_quelle": "Preset-Kommentar 12.11.2026: GPU ~10,6 GB bei Kontext 4096", "passt_64gb": "ja"},
  ]},
 "3d": {"titel": "3D", "beschreibung": "TripoSG (Bild -> GLB); optional TRELLIS.2 (mit Textur).",
  "modelle": [
   {"id": "triposg", "abschnitt": "external/triposg", "name": "TripoSG (eigener 3D-Worker)", "pflicht": True, "worker": "gufo-3d",
    "ordner": ["VAST-AI/TripoSG"], "dateien": ["VAST-AI/TripoSG/gufo/triposg.safetensors"],
    "quellen": [dict(QE("VAST-AI\\TripoSG\\gufo", "unsere Umwandlung (12.11.2026) aus VAST-AI/TripoSG Revision 2c1c516d (Original 7,9 GB, nicht aufgefuehrt); auf HF liegt triposg.safetensors im Wurzelverzeichnis, lokal unter VAST-AI\\TripoSG\\gufo\\"), ordner="VAST-AI/TripoSG")],
    "lizenz": lic("mit"), "gpu_gb_notiz": 6.7, "gpu_gb_notiz_quelle": "Preset-Kommentar 12.11.2026: Spitze 6,7 GB + 0,55 GB geteilt", "passt_64gb": "ja"},
   {"id": "trellis2", "abschnitt": "external/trellis2", "name": "TRELLIS.2-4B (eigener Worker, Built with DINOv3)", "pflicht": False, "worker": "trellis2",
    "ordner": ["microsoft/gufo-trellis2"], "dateien": ["microsoft/gufo-trellis2/"],
    "quellen": [QE("microsoft\\gufo-trellis2", "unsere Umwandlung (12.11.2026) aus microsoft/TRELLIS.2-4B Revision af44b45f (5 Teile) und microsoft/TRELLIS-image-large (ss_dec); trellis2-dinov3.safetensors NICHT im Repo: facebook/dinov3-vitl16-pretrain-lvd1689m (gated, DINOv3-Lizenz) selbst laden und mit tools/trellis2/convert_hf_to_gufo.py --parts dinov3 wandeln")],
    "lizenz": lic("trellis"), "gpu_gb_notiz": 19.3, "gpu_gb_notiz_quelle": "Preset-Kommentar 12.11.2026: GPU-Spitze 19,3 GB mit Textur", "passt_64gb": "ja"},
  ]},
 "werkzeuge": {"titel": "Werkzeuge", "beschreibung": "Real-ESRGAN-Hochskalierung; optional RIFE-Zwischenbilder (braucht den H3-Ordner) und Qwen3-Omni-Captioner (Audio beschreiben).",
  "modelle": [
   {"id": "gufo-upscale", "abschnitt": "external/gufo-upscale", "name": "Real-ESRGAN (general, wdn, anime)", "pflicht": True, "worker": "gufo.exe",
    "ordner": ["upscale"], "dateien": ["upscale/realesr-general-x4v3.safetensors", "upscale/realesr-general-wdn-x4v3.safetensors", "upscale/realesr-animevideov3.safetensors", "upscale/weights/Real-ESRGAN-LICENSE.txt"],
    "quellen": [QE("upscale", "unsere Umwandlung der Real-ESRGAN-Release-v0.2.5.0-.pth (SHA-256 der Quellen auf der Karte, docs/models/realesrgan); BSD-3-Clause, Lizenztext im Repo als LICENSE")],
    "lizenz": lic("bsd3"), "gpu_gb_notiz": 1.0, "gpu_gb_notiz_quelle": "Preset-Kommentar: Small model (~1 GB GPU)", "passt_64gb": "ja"},
   {"id": "gufo-rife", "abschnitt": "external/gufo-rife", "name": "RIFE 4.9 Zwischenbilder (Video-Worker mit H3-Modellpfad)", "pflicht": False, "worker": "gufo.exe", "benoetigt": ["minimax-h3"],
    "ordner": ["rife"], "dateien": ["rife/weights/rife49.safetensors"],
    "quellen": [QE("rife", "unsere Umwandlung von rife47.pth/rife49.pth (Spiegel marduk191/rife, SHA-256 der Quellen auf der Karte); Repo enthaelt rife49 + rife47 + *.convert.txt; lokal unter rife\\weights\\")],
    "lizenz": lic("rife"), "gpu_gb_notiz": None, "passt_64gb": "ja",
    "hinweis": "Der Produktions-Abschnitt zeigt mit model/manifest auf den MiniMax-H3-Ordner; ohne H3 ist dieser Abschnitt nicht startbar. RIFE selbst nutzen alle Video-Abschnitte ueber rife-model."},
   {"id": "qwen3-omni-captioner", "abschnitt": "omni/qwen3-omni-captioner", "name": "Qwen3-Omni-30B-A3B-Captioner Q4_K_M (laeuft im Router)", "pflicht": False, "worker": "router",
    "ordner": ["mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF"],
    "dateien": ["mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF/Qwen3-Omni-30B-A3B-Captioner.Q4_K_M.gguf", "mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF/Qwen3-Omni-30B-A3B-Captioner.mmproj-f16.gguf"],
    "quellen": [Q("mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF", "mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF", "", "Ordnername (nicht gepinnt)")],
    "lizenz": lic("ungeprueft"), "gpu_gb_notiz": None, "passt_64gb": "ja"},
  ]},
 "zweiter-rechner": {"titel": "Zweiter Rechner (Pool)", "beschreibung": "Keine Modelle: [remote/<id>]-Zwillinge fuer die Modelle, die der Partner hat (placement = pool), plus --remote-allow.",
  "modelle": []},
}

# ---------------------------------------------------------------------------------------------------------------
# SHA-256-Quellen
sha = {}      # relpfad (mit /) -> (sha256, quelle)
try:
    m = json.loads(pathlib.Path(a.h3_manifest).read_text(encoding="utf-8"))
    for f in m["files"]:
        sha["MiniMaxAI/MiniMax-H3/" + f["path"]] = (f["sha256"], "MINIMAX_H3_FL2VA_BF16.source-manifest.json")
except Exception as e:
    print("H3-Manifest nicht gelesen:", e)
try:
    for e in json.loads((WURZEL / "Qwen/Qwen-Image-2.1/.tree.json").read_text(encoding="utf-8")):
        if e.get("type") == "file" and e.get("lfs"):
            sha["Qwen/Qwen-Image-2.1/" + e["path"]] = (e["lfs"]["oid"], "Hugging-Face .tree.json (LFS-oid)")
except Exception as e:
    print(".tree.json nicht gelesen:", e)
try:
    for line in (WURZEL / "IndexTeam/IndexTTS-2.5-german-bundle/german/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            sha["IndexTeam/IndexTTS-2.5-german-bundle/german/" + parts[1]] = (parts[0], "german/SHA256SUMS.txt")
except Exception as e:
    print("SHA256SUMS nicht gelesen:", e)

# Unsere eigenen Repos: SHA-256 aus sha256-eigene-repos.json (hf-upload = Upload-Cache von hf upload-large-folder,
# lokal = sha256sum ueber die hochgeladene Datei); dazu je Datei hf_repo/hf_pfad/hf_revision fuer den Router-Download.
eigene_felder = {}   # relpfad (mit /) -> {hf_repo, hf_pfad, hf_revision}
for rel_win, e in EIGENE_DATEIEN.items():
    rel = rel_win.replace("\\", "/")
    if e.get("sha256"):
        sha[rel] = (e["sha256"], e["sha256_status"])
    ordner = next((o for o in EIGENE_REPOS if rel_win.lower().startswith(o.lower() + "\\")), None)
    if e.get("hf_pfad") is None:
        eigene_felder[rel] = {"hf_repo": None, "hf_pfad": None, "hf_revision": None}
    elif ordner:
        eigene_felder[rel] = {"hf_repo": EIGENE_REPOS[ordner]["hf_repo"], "hf_pfad": e["hf_pfad"], "hf_revision": EIGENE_REPOS[ordner]["hf_revision"]}

memory = {}
try:
    memory = json.loads(pathlib.Path(a.memory).read_text(encoding="utf-8"))["models"]
except Exception as e:
    print("memory.json nicht gelesen:", e)

# ---------------------------------------------------------------------------------------------------------------
def ausgeschlossen(rel):
    teile = rel.split("/")
    if any(t in AUSSCHLUSS_ORDNER for t in teile[:-1]): return True
    name = teile[-1]
    if name in AUSSCHLUSS_NAMEN: return True
    if pathlib.Path(name).suffix.lower() in AUSSCHLUSS_ENDUNGEN: return True
    if name.endswith(".convert.txt") or name.endswith(".pdf"): return True
    return False

def dateien_von(spec):
    """relative Pfade (mit /) fuer eine Dateiliste/Ordnerangabe"""
    out = []
    for s in spec:
        if s.endswith("/"):
            d = WURZEL / s
            if not d.is_dir():
                out.append((s, None)); continue
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(WURZEL).as_posix()
                    if not ausgeschlossen(rel): out.append((rel, p))
        else:
            p = WURZEL / s
            out.append((s, p if p.is_file() else None))
    return out

inodes = {}   # (dev, ino) -> erster Pfad
fehlend_im_baum = []
def datei_eintrag(rel, p):
    e = {"pfad": rel.replace("/", "\\"), "bytes": None, "sha256": "", "sha256_status": "fehlt"}
    if p is None:
        e["im_baum_198"] = False; fehlend_im_baum.append(rel); return e
    st = p.stat()
    e["bytes"] = st.st_size; e["im_baum_198"] = True
    key = (st.st_dev, st.st_ino)
    if st.st_nlink > 1 and key in inodes and inodes[key] != rel:
        e["hardlink_auf"] = inodes[key].replace("/", "\\")
    elif st.st_nlink > 1:
        inodes[key] = rel
    if rel in sha:
        e["sha256"], e["sha256_status"] = sha[rel]
    if rel in eigene_felder:
        e.update(eigene_felder[rel])
    return e

katalog = {"katalog_version": KATALOG_VERSION, "erzeugt": datetime.datetime.now().isoformat(timespec="seconds"),
           "erzeugt_aus": {"modellbaum": str(WURZEL), "memory_json": a.memory, "h3_manifest": a.h3_manifest},
           "hinweise": [
             "Groessen (bytes) kommen aus dem echten Baum der .198; Hardlinks (hardlink_auf) zaehlen in den Summen nur einmal.",
             "sha256_status: 'fehlt' = keine Pruefsumme bekannt; es gibt unter modelle\\ keine *.source-manifest.json. Vorhanden nur: H3-Manifest, Qwen-Image-2.1 .tree.json (LFS), IndexTTS german/SHA256SUMS.txt, dazu unsere eigenen Repos (sha256-eigene-repos.json).",
             "sha256_status 'hf-upload' = aus dem Upload-Cache von hf upload-large-folder (= LFS-oid auf Hugging Face), 'lokal' = sha256sum auf der .198 ueber die hochgeladene Datei (13.11.2026). hf_repo/hf_pfad/hf_revision je Datei = unsere umgewandelten Gewichte unter huggingface.co/D3rPapst (der Router laedt <hf_repo>/resolve/<hf_revision oder main>/<hf_pfad>); null = nicht auf HF (DINOv3 selbst wandeln). Dateien ohne diese Felder laedt der Router ueber die Quelle (quellen[]), deren Ordner der Pfad-Praefix ist.",
             "gpu_gb_notiz = gemessener Wert aus Preset-Kommentar/Release-Notes (Quelle dabei); gpu_gb_router/host_gb_router = vom Router gelernte Spitze (learned_* in gufo-router-combo7.memory.json, Datum dabei). Keine Schaetzwerte (estimate_* werden nicht uebernommen).",
             "passt_64gb folgt der Tabelle in INSTALLER-PLAN-2026-11-13.md Schritt 2; wo unsere Messung auf der .198 (64 GB) abweicht, steht es im Hinweis.",
             "Nicht im Katalog: [external/qwen3.5-4b] und [external/qwen3.5-9b] (llama-server fuer den Mailputzer, nicht gufo); alibaba-pai ControlNet, bosonai Higgs, wan21-vace, TRELLIS-image-large (im Baum, aber nicht im Produktions-Preset).",
           ],
           "worker": {"gufo.exe": "bin\\gufo.exe der Version (Router + alle Standard-Worker)", "router": "laeuft im Router-Prozess selbst (omni)",
                      "35b": "gufo-stockmind-35b (eigener Build, app\\gufo-stockmind-35b-2026-10-26b)", "gufo-3d": "TripoSG-Worker (feat/triposg, app\\gufo-3d-2026-11-12)",
                      "trellis2": "TRELLIS.2-Worker (feat/trellis2, app\\gufo-trellis2-2026-11-12)", "gufo-moss": "MOSS-Music-Worker (feat/moss-music, app\\gufo-moss-2026-11-12)",
                      "gufo-d": "D-Block-Worker mit Seed-VC (GPL, nicht veroeffentlicht, app\\gufo-d-2026-11-12)"},
           "bausteine": {}}

alle_pfade = {}
for bname, b in BAUSTEINE.items():
    out_modelle = []
    gesehen = set()   # dedupe innerhalb des Bausteins (Pfad + Hardlink-Ziel)
    summe = summe_pflicht = 0
    for mdl in b["modelle"]:
        eintraege = [datei_eintrag(rel, p) for rel, p in dateien_von(mdl["dateien"])]
        mbytes = sum(e["bytes"] or 0 for e in eintraege if "hardlink_auf" not in e)
        for e in eintraege:
            key = e.get("hardlink_auf", e["pfad"])
            if key not in gesehen and e["bytes"]:
                gesehen.add(key); summe += e["bytes"]
                if mdl["pflicht"]: summe_pflicht += e["bytes"]
            alle_pfade[e["pfad"]] = e
        mem = memory.get(mdl["id"], {})
        if mdl["id"] == "moss-music-8b": mem = memory.get("moss-music", {})
        gel = None
        if mem.get("learned_complete"):
            gel = {"gpu_gb": mem["learned_gpu_gb"], "host_gb": mem["learned_host_gb"], "gemessen_am": mem["learned_at"]}
        elif mem.get("learned_layouts"):
            l0 = mem["learned_layouts"].get("0", {})
            if l0.get("complete"):
                gel = {"gpu_gb": l0["gpu_gb"], "host_gb": l0["host_gb"], "gemessen_am": l0["at"] + " (learned_layouts[0])"}
        o = {
            "id": mdl["id"], "name": mdl["name"], "abschnitt": mdl.get("abschnitt", ""), "art": mdl.get("art", "modell"),
            "pflicht": mdl["pflicht"], "vorgabe_ram": mdl.get("vorgabe_ram"), "benoetigt": mdl.get("benoetigt", []), "worker": mdl["worker"],
            "oeffentlich": mdl.get("oeffentlich", True),
            "ordner": [x.replace("/", "\\") for x in mdl["ordner"]],
            "quellen": [dict(q, ordner=q["ordner"].replace("/", "\\")) for q in mdl["quellen"]],
            "lizenz": mdl["lizenz"],
            "gpu_gb_notiz": mdl.get("gpu_gb_notiz"), "gpu_gb_notiz_quelle": mdl.get("gpu_gb_notiz_quelle", ""),
            "gpu_gb_router": gel["gpu_gb"] if gel else None, "host_gb_router": gel["host_gb"] if gel else None, "router_gemessen_am": gel["gemessen_am"] if gel else "",
            "passt_64gb": mdl["passt_64gb"], "hinweis": mdl.get("hinweis", ""),
            "daten_dateien": [x.replace("/", "\\") for x in mdl.get("daten_dateien", [])],
            "bytes": mbytes, "gb": round(mbytes / 1e9, 2), "dateien": eintraege,
        }
        if "lizenz_zusatz" in mdl: o["lizenz_zusatz"] = mdl["lizenz_zusatz"]
        for k in ("lora_name", "lora_fuer"):
            if k in mdl: o[k] = mdl[k]
        out_modelle.append(o)
    katalog["bausteine"][bname] = {"titel": b["titel"], "beschreibung": b["beschreibung"], "modelle": out_modelle,
                                   "summe_bytes": summe, "summe_gb": round(summe / 1e9, 2),
                                   "summe_pflicht_bytes": summe_pflicht, "summe_pflicht_gb": round(summe_pflicht / 1e9, 2),
                                   "anzahl_modelle": len(out_modelle), "anzahl_dateien": sum(len(m["dateien"]) for m in out_modelle)}

# Gesamtsumme ueber alle Bausteine (Dateien, die in mehreren Bausteinen vorkommen, nur einmal)
ges = set(); gesamt = 0
for e in alle_pfade.values():
    key = e.get("hardlink_auf", e["pfad"])
    if e["bytes"] and key not in ges: ges.add(key); gesamt += e["bytes"]
katalog["gesamt_bytes"] = gesamt; katalog["gesamt_gb"] = round(gesamt / 1e9, 2)
katalog["anzahl_dateien"] = len(alle_pfade)
katalog["fehlend_im_baum_198"] = fehlend_im_baum
pathlib.Path(a.ziel).write_text(json.dumps(katalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"geschrieben: {a.ziel}")
for bname, b in katalog["bausteine"].items():
    print(f"  {bname:16} {b['anzahl_modelle']:2} Modelle {b['anzahl_dateien']:4} Dateien  Pflicht {b['summe_pflicht_gb']:8.2f} GB  alles {b['summe_gb']:8.2f} GB")
print(f"  gesamt {katalog['anzahl_dateien']} Dateien, {katalog['gesamt_gb']} GB; im Baum fehlend: {fehlend_im_baum}")
