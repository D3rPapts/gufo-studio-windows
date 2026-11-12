#!/usr/bin/env python3
"""Referenz-Dumps fuer den gufo-Port von TRELLIS.2 (laeuft auf der .56 mit torch).

Benutzt nur das offizielle Repo (microsoft/TRELLIS.2) und dessen Abhaengigkeiten.
Alle Testbilder haben einen Alphakanal -> kein RMBG-2.0 noetig (BRIA, nicht kommerziell;
das Modell wird nicht geladen). Alles FP32, Pipeline-Typ '512', fester Seed, das
Rauschen wird mitgespeichert. Die Stufen werden einzeln wie in
trellis2/pipelines/trellis2_image_to_3d.py aufgerufen, damit Zwischenwerte entstehen.

    python dump_referenz.py --repo C:\\pfad\\TRELLIS.2 --weights C:\\pfad\\microsoft\\TRELLIS.2-4B ^
        --bilder testbilder --out dumps [--device cuda] [--seed 42] [--cpu-backend] [--nur-ss]

--cpu-backend: ohne Triton/flash-attn/spconv (ROCm-Windows oder CPU): ATTN_BACKEND=sdpa,
  SPARSE_CONV_BACKEND=none und sparse_backend_cpu.install() (reines torch; langsam).
  Der Dump `conv_probe` ist dann KEIN unabhaengiger Nachweis der Tap-Reihenfolge -
  bitte im Protokoll vermerken, welches Backend lief.
--nur-ss: nur Vorverarbeitung, DINOv3, Grobstruktur-Fluss und -Decoder (schnell, ~1 min GPU).

Je Bild entsteht dumps/<name>/ (Dateien siehe README.md). KI-unterstuetzt (Claude Code), 12.11.2026.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Klon von microsoft/TRELLIS.2")
    ap.add_argument("--weights", required=True, help="HF-Snapshot microsoft/TRELLIS.2-4B (pipeline.json + ckpts/)")
    ap.add_argument("--bilder", required=True, help="Ordner mit spielzeug.png, tasse.png, stuhl.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu-backend", action="store_true")
    ap.add_argument("--nur-ss", action="store_true")
    args = ap.parse_args()

    if args.cpu_backend:
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_CONV_BACKEND"] = "none"
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    if args.cpu_backend:
        import sparse_backend_cpu
        sparse_backend_cpu.install()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules import sparse as sp
    from PIL import Image

    device = torch.device(args.device)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.weights)
    pipe.rembg_model = None            # nicht laden, nicht benutzen (Alphapfad)
    pipe.low_vram = True
    pipe.to(device)
    pipe._device = device
    # FP32 ueberall (die DiTs sind BF16, die Decoder FP16 gespeichert)
    for name, model in pipe.models.items():
        if hasattr(model, "convert_to"):
            model.convert_to(torch.float32)
        if hasattr(model, "convert_to_fp32"):
            model.convert_to_fp32()
        model.float()
    pipe.image_cond_model.model.float()

    env = {
        "torch": torch.__version__, "device": str(device), "seed": args.seed,
        "cpu_backend": args.cpu_backend, "repo_commit": _git_rev(args.repo),
        "ATTN_BACKEND": os.environ.get("ATTN_BACKEND", "flash_attn (default)"),
        "SPARSE_CONV_BACKEND": os.environ.get("SPARSE_CONV_BACKEND", "flex_gemm (default)"),
    }
    for mod in ("transformers", "numpy", "flex_gemm", "flash_attn", "spconv", "cumesh", "o_voxel"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "vorhanden")
        except Exception as e:  # noqa: BLE001
            env[mod] = f"fehlt ({type(e).__name__})"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "umgebung.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print("Umgebung:", env)

    # Tap-Reihenfolge der Sparse-Conv: eine Faltung des Form-Decoders auf Zufallsdaten (einmal).
    conv_probe(pipe, device, args.out, args.seed, sp)

    for name in ("spielzeug", "tasse", "stuhl"):
        path = os.path.join(args.bilder, name + ".png")
        if not os.path.isfile(path):
            print("fehlt:", path)
            continue
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        dump_one(pipe, path, out, args, device, sp, Image, torch)
        print(f"{name}: fertig in {time.time() - t0:.0f} s")


def _git_rev(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unbekannt"


def save(out, name, tensor):
    import torch
    arr = tensor.detach().float().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    np.save(os.path.join(out, name + ".npy"), arr)
    print(f"  {name}: {arr.shape} {arr.dtype}")


def conv_probe(pipe, device, out, seed, sp):
    """blocks.3.0.conv des Form-Decoders (128 -> 128, 3^3) auf 300 Zufallsvoxeln im 16^3-Wuerfel."""
    import torch
    dec = pipe.models["shape_slat_decoder"]
    conv = dec.blocks[3][0].conv
    g = torch.Generator().manual_seed(seed)
    coords = torch.unique(torch.randint(0, 16, (400, 3), generator=g), dim=0)[:300].int()
    coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=torch.int32), coords], dim=1).to(device)
    feats = torch.randn((coords.shape[0], conv.in_channels), generator=g).to(device)
    x = sp.SparseTensor(feats=feats, coords=coords)
    dec.to(device)
    with torch.no_grad():
        y = conv(x)
    dec.cpu()
    save(out, "conv_probe_coords", coords[:, 1:])
    save(out, "conv_probe_in", feats)
    save(out, "conv_probe_out", y.feats)
    save(out, "conv_probe_weight", conv.weight)   # [Co, kd, kh, kw, Ci] (FlexGEMM) oder [Co, Ci, kd, kh, kw]
    save(out, "conv_probe_bias", conv.bias)
    print("  conv_probe: weight shape", tuple(conv.weight.shape))


def dump_one(pipe, path, out, args, device, sp, Image, torch):
    with torch.no_grad():
        _dump_one(pipe, path, out, args, device, sp, Image, torch)


def _dump_one(pipe, path, out, args, device, sp, Image, torch):
    # 1. Vorverarbeitung (Alphapfad, kein RMBG)
    img = Image.open(path)
    pre = pipe.preprocess_image(img)
    pre.save(os.path.join(out, "preproc_crop.png"))
    # 2. DINOv3 bei 512 px (+ normierte Pixel)
    pipe.image_cond_model.image_size = 512
    im = pre.resize((512, 512), Image.LANCZOS)
    arr = np.array(im.convert("RGB")).astype(np.float32) / 255
    pix = torch.from_numpy(arr).permute(2, 0, 1).float()
    pix = pipe.image_cond_model.transform(pix)
    save(out, "preproc_512", pix)
    cond = pipe.get_cond([pre], 512)
    save(out, "dino_tokens_512", cond["cond"][0])      # [1029,1024]

    # 3. Grobstruktur-Fluss: eigenes Rauschen, Hooks auf Block 0/29, Schritt 0 bedingt/unbedingt
    flow = pipe.models["sparse_structure_flow_model"]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((1, flow.in_channels, 16, 16, 16), generator=gen, device=device)
    save(out, "ss_noise", noise[0])
    flow.to(device)
    captured = {}
    hooks = [flow.blocks[i].register_forward_hook(lambda m, inp, o, i=i: captured.__setitem__(i, o.detach().clone()))
             for i in (0, 29)]
    sampler = pipe.sparse_structure_sampler
    params = dict(pipe.sparse_structure_sampler_params)
    t0 = sampler_t0(params)
    t_model = torch.tensor([1000 * t0], device=device, dtype=torch.float32)
    v_cond = flow(noise, t_model, cond["cond"])
    save(out, "ss_block00", captured[0][0])            # [4096,1536]
    save(out, "ss_block29", captured[29][0])
    for h in hooks:
        h.remove()
    save(out, "ss_v_step00_cond", v_cond[0])           # [8,16,16,16]
    v_uncond = flow(noise, t_model, cond["neg_cond"])
    save(out, "ss_v_step00_uncond", v_uncond[0])
    z_s = sampler.sample(flow, noise.clone(), **cond, **params, verbose=False).samples
    save(out, "ss_latent_final", z_s[0])
    flow.cpu()
    dec = pipe.models["sparse_structure_decoder"]
    dec.to(device)
    logits = dec(z_s)
    dec.cpu()
    save(out, "ss_logits64", logits[0, 0])
    occ = logits > 0
    save(out, "ss_occupancy64", occ[0, 0].float())
    pooled = torch.nn.functional.max_pool3d(occ.float(), 2, 2, 0) > 0.5
    coords = torch.argwhere(pooled)[:, [0, 2, 3, 4]].int()
    save(out, "ss_coords32", coords[:, 1:])
    print(f"  Belegung: {int(occ.sum())} bei 64^3, {coords.shape[0]} Token bei 32^3")
    if args.nur_ss:
        return

    # 4. Form-Fluss 512 (Rauschen gespeichert), Form-Decoder mit Zwischenstufen
    flow_s = pipe.models["shape_slat_flow_model_512"]
    noise_s = torch.randn((coords.shape[0], flow_s.in_channels), generator=gen, device=device)
    save(out, "shape_noise", noise_s)
    flow_s.to(device)
    captured.clear()
    hooks = [flow_s.blocks[i].register_forward_hook(lambda m, inp, o, i=i: captured.__setitem__(i, o.feats.detach().clone()))
             for i in (0, 29)]
    x0 = sp.SparseTensor(feats=noise_s.clone(), coords=coords)
    params_s = dict(pipe.shape_slat_sampler_params)
    t_model = torch.tensor([1000 * sampler_t0(params_s)], device=device, dtype=torch.float32)
    _ = flow_s(x0, t_model, cond["cond"])
    save(out, "shape_block00", captured[0])
    save(out, "shape_block29", captured[29])
    for h in hooks:
        h.remove()
    slat = pipe.shape_slat_sampler.sample(flow_s, sp.SparseTensor(feats=noise_s.clone(), coords=coords), **cond, **params_s, verbose=False).samples
    std = torch.tensor(pipe.shape_slat_normalization["std"])[None].to(device)
    mean = torch.tensor(pipe.shape_slat_normalization["mean"])[None].to(device)
    slat = slat * std + mean
    save(out, "shape_latent_final", slat.feats)       # nach *std+mean
    save(out, "shape_coords", slat.coords[:, 1:])
    flow_s.cpu()

    sdec = pipe.models["shape_slat_decoder"]
    sdec.set_resolution(512)
    sdec.to(device)
    level_out = {}
    sub_out = {}
    for li, level in enumerate(sdec.blocks):
        last = level[-1]
        def hook(m, inp, o, li=li):
            if isinstance(o, tuple):
                level_out[li] = (o[0].feats.detach().clone(), o[0].coords.detach().clone())
                sub_out[li] = o[1].feats.detach().clone()
            else:
                level_out[li] = (o.feats.detach().clone(), o.coords.detach().clone())
        last.register_forward_hook(hook)
    meshes, subs = sdec(slat, return_subs=True)
    for li, (f, c) in level_out.items():
        save(out, f"shape_dec_level{li}_feats", f)
        save(out, f"shape_dec_level{li}_coords", c[:, 1:])
    for li, s in sub_out.items():
        save(out, f"shape_subdiv_{li}", (s > 0).float())
    mesh = meshes[0]
    verts, faces = mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy()
    boundary_before = None
    try:
        mesh.fill_holes()
    except Exception as e:  # noqa: BLE001
        print("  fill_holes nicht verfuegbar:", e)
    save(out, "mesh_vertices", mesh.vertices)
    save(out, "mesh_faces", mesh.faces)
    stats = {"vertices_raw": int(len(verts)), "faces_raw": int(len(faces)),
             "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]),
             "bbox_min": verts.min(0).tolist(), "bbox_max": verts.max(0).tolist()}
    try:
        import trimesh
        tm = trimesh.Trimesh(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy())
        tm.export(os.path.join(out, "mesh_ref.glb"))
        stats.update({"watertight": bool(tm.is_watertight), "volume": float(tm.volume) if tm.is_watertight else None})
    except Exception as e:  # noqa: BLE001
        print("  trimesh fehlt:", e)
    with open(os.path.join(out, "mesh_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("  mesh:", stats)
    sdec.cpu()

    # 5. Textur-Fluss + Textur-Decoder
    flow_t = pipe.models["tex_slat_flow_model_512"]
    flow_t.to(device)
    shape_norm = (slat - mean) / std
    noise_t = torch.randn((coords.shape[0], flow_t.in_channels - shape_norm.feats.shape[1]), generator=gen, device=device)
    save(out, "tex_noise", noise_t)
    tex = pipe.tex_slat_sampler.sample(flow_t, shape_norm.replace(noise_t.clone()), concat_cond=shape_norm, **cond,
                                       **pipe.tex_slat_sampler_params, verbose=False).samples
    tstd = torch.tensor(pipe.tex_slat_normalization["std"])[None].to(device)
    tmean = torch.tensor(pipe.tex_slat_normalization["mean"])[None].to(device)
    tex = tex * tstd + tmean
    save(out, "tex_latent_final", tex.feats)
    flow_t.cpu()
    tdec = pipe.models["tex_slat_decoder"]
    tdec.to(device)
    voxels = tdec(tex, guide_subs=subs) * 0.5 + 0.5
    save(out, "tex_dec_out", voxels.feats)
    save(out, "tex_dec_coords", voxels.coords[:, 1:])
    tdec.cpu()


def sampler_t0(params):
    r = params.get("rescale_t", 1.0)
    return r * 1.0 / (1 + (r - 1) * 1.0)   # t_0 = 1 bleibt 1


if __name__ == "__main__":
    main()
