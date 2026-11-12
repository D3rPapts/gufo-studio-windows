# TRELLIS.2-Referenz für den gufo-Port – Anleitung für die Regie (.56)

Stand 12.11.2026, KI-unterstützt (Claude Code, Baumeister). Zweck: PyTorch-Referenzwerte je Stufe,
gegen die der native gufo-Port (Branch `feat/trellis2`, Plan `TRELLIS2-PLAN.md` Abschnitt 6/7) abgenommen wird.

## Was hier liegt
- `testbilder/spielzeug.png`, `tasse.png`, `stuhl.png` – eigene Renderings (lizenzfrei, 768×768, RGBA mit
  Alphakanal → TRELLIS.2 nimmt den Alphapfad, **RMBG-2.0 (BRIA) wird nicht geladen**).
- `dump_referenz.py` – das Skript für die .56 (nur offizielles Repo + dessen Abhängigkeiten).
- `sparse_backend_cpu.py` – Ersatz für FlexGEMM/flash-attn, falls Triton auf ROCm-Windows nicht läuft
  (reines torch: Nachbartabelle + Gather-GEMM, dichte SDPA). **Ungetestet** (auf der .198 gibt es kein torch);
  Fehler bitte melden, dann passe ich es an.

## Ausführen auf der .56
1. Repo klonen: `git clone https://github.com/microsoft/TRELLIS.2` → Commit notieren. Umgebung nach dessen README
   (torch, transformers ≥ 4.56 mit `DINOv3ViTModel`, o-voxel; cumesh/flex_gemm/flash-attn nur wenn sie bauen).
2. Gewichte: liegen geprüft auf der .198 unter `D:\llm-modelle\microsoft\TRELLIS.2-4B` (16,2 GB) und
   `D:\llm-modelle\microsoft\TRELLIS-image-large\ckpts\ss_dec_conv3d_16l8_fp16.*` (die Pipeline erwartet den
   ss_dec unter `microsoft/TRELLIS-image-large/ckpts/…` – entweder HF-Download auf der .56 oder den Ordner
   daneben legen und `pipeline.json` anpassen). DINOv3: `D:\llm-modelle\facebook\dinov3-vitl16-pretrain-lvd1689m`
   (gated, Fabians Freigabe; Lizenzkopie liegt dabei) → auf der .56 als lokaler Pfad in `pipeline.json`
   (`image_cond_model.args.model_name`) eintragen oder mit demselben HF-Konto laden.
3. Dumps erzeugen (FP32; Grobstruktur ≈ 1 min, komplett mit Form/Textur 512 mehrere Minuten GPU):
   ```
   python dump_referenz.py --repo <TRELLIS.2-Klon> --weights <Snapshot TRELLIS.2-4B> --bilder testbilder --out dumps
   ```
   Falls flex_gemm/flash_attn nicht importierbar: `--cpu-backend` (dann `conv_probe` **nicht** als Beleg für die
   Tap-Reihenfolge werten, siehe unten). Erst nur die schnelle Stufe: `--nur-ss`.
4. Ordner `dumps\` zurück auf die .198 nach `D:\claude\gufo-arbeit\3d-port\referenz-trellis2\dumps\` (Direktkabel
   192.168.100.56 ↔ .198) und im Austausch-Chat melden – mit dem Inhalt von `umgebung.json` (Backends!).

## Was entsteht
`dumps/umgebung.json` (Versionen, Backends, Commit) und `dumps/conv_probe_*.npy` (einmal): eine `SparseConv3d`
(Form-Decoder `blocks.3.0.conv`, 128→128) auf 300 Zufallsvoxeln – `coords [N,3]`, `in [N,128]`, `out [N,128]`,
`weight` (Form verrät das Layout: FlexGEMM `[Co,3,3,3,Ci]` oder spconv `[Co,Ci,3,3,3]`), `bias`. Entscheidet, ob
Tap `t = kd·9 + kh·3 + kw` auf den Nachbarn `coord + (kd−1, kh−1, kw−1)` zeigt (Plan Abschnitt 8, Risiko 1) –
**nur aussagekräftig mit dem Original-Backend (flex_gemm/spconv), nicht mit `--cpu-backend`.**

Je Bild `dumps/<name>/`:
| Datei | Shape / Inhalt | Prüft in gufo |
|---|---|---|
| `preproc_crop.png` | Bild nach Alphapfad/Zuschnitt (RGB·Alpha, schwarzer Rand) | Vorverarbeitung |
| `preproc_512.npy` | [3,512,512] nach LANCZOS 512 + ImageNet-Normierung | Vorverarbeitung |
| `dino_tokens_512.npy` | [1029,1024] DINOv3-Token (CLS, 4 Register, 1024 Patches; Endnorm ohne Affin) | DINOv3 |
| `ss_noise.npy` | [8,16,16,16] Startrauschen (Seed 42) – gufo lädt es statt eigenem RNG | DiT-Eingang |
| `ss_block00.npy`, `ss_block29.npy` | [4096,1536] Blockausgaben bei Schritt 0, bedingt | DiT-Blöcke |
| `ss_v_step00_cond/uncond.npy` | [8,16,16,16] Geschwindigkeit Schritt 0 | DiT-Ausgang, CFG |
| `ss_latent_final.npy` | [8,16,16,16] nach 12 Schritten (g 7,5, Intervall, Rescale 0,7) | Sampler |
| `ss_logits64.npy`, `ss_occupancy64.npy`, `ss_coords32.npy` | [64,64,64] Logits/Belegung, [N,3] Token-Koordinaten | ss_dec |
| `shape_noise.npy`, `shape_block00/29.npy`, `shape_latent_final.npy`, `shape_coords.npy` | sparse Form-Fluss 512 | Form-DiT |
| `shape_dec_level{0..4}_{feats,coords}.npy`, `shape_subdiv_{0..3}.npy` | Zwischenstufen des Form-Decoders (nach jeder Stufe, Masken) | Sparse-Decoder, C2S |
| `mesh_vertices/faces.npy`, `mesh_ref.glb`, `mesh_stats.json` | Dual-Grid-Mesh (nach fill_holes, falls CuMesh läuft) | Mesh |
| `tex_noise.npy`, `tex_latent_final.npy`, `tex_dec_out.npy` [M,6], `tex_dec_coords.npy` | Textur-Fluss + -Decoder (·0,5+0,5) | Textur |

## Hinweise
- Alle Werte FP32 (`convert_to(float32)`, Decoder `convert_to_fp32`). Der Port rechnet die DiTs in BF16; Toleranzen im Plan.
- `preprocess_image` skaliert > 1024 px herunter und schneidet quadratisch um die Alpha-Box; die Testbilder sind 768 px.
- Das Repo ist Fremdcode und läuft nur in der Referenzumgebung auf der .56, nicht in gufo; aus dem Modellordner werden
  nur safetensors + JSON gelesen.
- DINOv3-Lizenz (Meta): Lizenzkopie liegt beim Snapshot; „Built with DINOv3“ ist in gufo zu nennen.
