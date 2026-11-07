"""Runs one stage of indextts_oracle.py in its own process (lean memory: stage weights only, read without mmap).

The oracle source is split at its "# === STAGE: <name> ===" markers. This process runs the prelude (imports, helpers,
text frontend; light), restores T/META from <out>/stages.pt (written by the earlier stages), sets the variables the
stage needs, runs the stage and saves stages.pt again (atomically). The vocoder stage writes oracle.safetensors/json.

Run: venv-indextts python -I indextts_oracle_staged.py <stage> <bundle_dir> <out_dir> <ref_code_dir>
Stages: front, w2v, emo, gpt, s2mel, vocoder. Written with AI assistance (Claude Code)."""
import os
import sys

STAGES = ["front", "w2v", "emo", "gpt", "s2mel", "vocoder"]
stage = sys.argv[1]
assert stage in STAGES, stage
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "indextts_oracle.py"), encoding="utf-8").read()
parts = src.split("# === STAGE: ")
prelude = parts[0]
segments = {}
for p in parts[1:]:
    name, body = p.split(" ===\n", 1)
    segments[name] = body
sys.argv = [os.path.join(here, "indextts_oracle.py")] + sys.argv[2:]
g = {"__name__": "__oracle__", "__file__": sys.argv[0]}
exec(compile(prelude, "indextts_oracle.py:prelude", "exec"), g)
torch = g["torch"]
OUT = g["OUT"]
ck = os.path.join(OUT, "stages.pt")
idx = STAGES.index(stage)
if idx > 0:
    state = torch.load(ck, weights_only=False)
    done = state["META"].get("_stages_done", [])
    missing = [s for s in STAGES[:idx] if s not in done]
    assert not missing, f"earlier stages missing: {missing}"
    g["T"].update(state["T"])
    g["META"].update(state["META"])
    V = g["T"]
    g.update(audio_16k=V["audio_16k"][None], emo_16k=V["emo_16k"][None], audio_22k=V["audio_22k"][None],
             ref_mel=V["ref_mel"][None], style=V["style"][None])
    if idx > STAGES.index("w2v"):
        g.update(spk_cond_emb=V["spk_cond_emb"][None], emo_cond_emb=V["emo_cond_emb"][None])
    if idx > STAGES.index("emo"):
        g["emovec"] = V["emovec"][None]
    if idx > STAGES.index("gpt"):
        g["codes_t"] = V["codes"][None]
    if idx > STAGES.index("s2mel"):
        g["vc_target"] = V["vc_target"][None]
exec(compile(segments[stage], f"indextts_oracle.py:{stage}", "exec"), g)
g["META"]["_stages_done"] = [s for s in STAGES if s in g["META"].get("_stages_done", [])] + [stage]
torch.save({"T": g["T"], "META": g["META"]}, ck + ".tmp")
os.replace(ck + ".tmp", ck)
g["log"](f"stage {stage} done")
