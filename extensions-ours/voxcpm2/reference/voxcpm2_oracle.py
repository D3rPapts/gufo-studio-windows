"""CPU fp32 reference oracle for the native gufo port of VoxCPM2 (OpenBMB, Apache-2.0, voxcpm 2.0.3).

Runs one short German "controllable cloning" request (reference voice only, emotion as a bracket instruction in front
of the text) on the CPU in float32 with a fixed seed and writes every intermediate the C++ port is compared against to
one safetensors file plus a JSON sidecar (token ids, strings, step counts).

Memory (<= ~2 GB): the model is built on the meta device. The small modules (local encoder, DiT, projections, heads)
are loaded once as BF16; the 28 + 8 transformer layers of the two LMs are streamed: each layer is read from the
safetensors file right before it runs and dropped back to meta afterwards (the per-layer code is a copy of
MiniCPMModel.forward / forward_step). Linear/Embedding layers widen the BF16 weights to FP32 per call, i.e. the math
is FP32 on BF16-valued weights (exactly what the C++ runtime does). The AR loop is capped at VOXCPM_ORACLE_STEPS
(default 8) teacher-free steps.

The AR loop is a line-by-line copy of VoxCPM2Model._inference (voxcpm 2.0.3) so intermediates can be captured; the
CFM sampler itself is the library's UnifiedCFM (its torch.randn draws are recorded).

Run: venv-voxcpm python -I voxcpm2_oracle.py [out_dir]   (CPU only, no CUDA/ROCm device is touched)
Env: VOXCPM_ORACLE_TEXT, VOXCPM_ORACLE_VOICE, VOXCPM_ORACLE_SEED, VOXCPM_ORACLE_THREADS
Written with AI assistance (Claude Code)."""
import json, math, os, struct, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"D:\llm-modelle\openbmb\VoxCPM2"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"D:\llm-modelle\gufo\ft\voxcpm-ref"
VOICE = os.environ.get("VOXCPM_ORACLE_VOICE", r"D:\llm-modelle\gufo\voices\erzaehler-9s.wav")
TEXT = os.environ.get("VOXCPM_ORACLE_TEXT", "(fröhlich und begeistert)Guten Morgen, heute scheint die Sonne.")
SEED = int(os.environ.get("VOXCPM_ORACLE_SEED", "1234"))
TIMESTEPS = 10
CFG = 2.0
torch.set_num_threads(int(os.environ.get("VOXCPM_ORACLE_THREADS", "8")))
torch.set_grad_enabled(False)

import librosa
from safetensors.torch import save_file
from transformers import LlamaTokenizerFast

from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
from voxcpm.model.utils import mask_multichar_chinese_tokens
from voxcpm.modules.audiovae import AudioVAEV2
from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE

os.makedirs(OUT, exist_ok=True)
T = {}
META = {"text": TEXT, "seed": SEED, "voice": VOICE, "torch": torch.__version__, "timesteps": TIMESTEPS, "cfg": CFG}


def put(name, tensor):
    t = tensor.detach().to("cpu")
    t = t.to(torch.float32) if t.is_floating_point() else t.to(torch.int64)
    T[name] = t.contiguous().clone()


# ---- FP32 math on BF16 weights ------------------------------------------------------------------------------------
def _linear(self, x):
    return F.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())


def _embedding(self, ids):
    return F.embedding(ids, self.weight).float()


nn.Linear.forward = _linear
nn.Embedding.forward = _embedding


# Plain reads (no mmap): file pages stay in the OS cache instead of this process' working set.
ST_PATH = os.path.join(MODEL, "model.safetensors")
with open(ST_PATH, "rb") as _f:
    _n = struct.unpack("<Q", _f.read(8))[0]
    ST_HEADER = {k: v for k, v in json.loads(_f.read(_n)).items() if k != "__metadata__"}
ST_BASE = 8 + _n
ST_KEYS = list(ST_HEADER.keys())
ST_FILE = open(ST_PATH, "rb")


def get_tensor(key):
    v = ST_HEADER[key]
    assert v["dtype"] == "BF16", key
    b, e = v["data_offsets"]
    ST_FILE.seek(ST_BASE + b)
    buf = bytearray(ST_FILE.read(e - b))
    return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(v["shape"])


def load_module(module, prefix):
    sd = {k[len(prefix):]: get_tensor(k) for k in ST_KEYS if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    return missing


def unload(module):
    module.to("meta")


def stream_forward(lm, prefix, inputs_embeds):
    """MiniCPMModel.forward(is_causal=True) + kv_cache.fill_caches, one layer resident at a time."""
    if lm.rope_emb is not None:
        position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long)
        position_emb = lm.rope_emb(position_ids)
    else:
        position_emb = None
    hidden_states = inputs_embeds
    n = inputs_embeds.size(1)
    trace = None
    for i, layer in enumerate(lm.layers):
        load_module(layer, f"{prefix}layers.{i}.")
        hidden_states, (k, v) = layer(hidden_states, position_emb, True)
        lm.kv_cache.kv_cache[0, i, :, :, :n, :] = k
        lm.kv_cache.kv_cache[1, i, :, :, :n, :] = v
        unload(layer)
        if i == 0:
            trace = hidden_states[0].clone()
    lm.kv_cache.current_length = n
    return lm.norm(hidden_states), trace


def stream_step(lm, prefix, inputs_embeds, position_id):
    """MiniCPMModel.forward_step, one layer resident at a time."""
    position_emb = lm.rope_emb(position_id) if lm.rope_emb is not None else None
    hidden_states = inputs_embeds
    for i, layer in enumerate(lm.layers):
        load_module(layer, f"{prefix}layers.{i}.")
        hidden_states = layer.forward_step(hidden_states, position_emb, position_id,
                                           lm.kv_cache.get_layer_cache(i))
        unload(layer)
    return lm.norm(hidden_states)


t0 = time.time()
with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as f:
    config = VoxCPMConfig.model_validate_json(f.read())
config.dtype = "float32"   # KV caches and audio_feat in fp32
config.max_length = 1024   # small static KV caches
config.device = "cpu"
tokenizer = LlamaTokenizerFast.from_pretrained(MODEL)

def load_vae():
    v = AudioVAEV2(config=config.audio_vae_config)
    sd = torch.load(os.path.join(MODEL, "audiovae.pth"), map_location="cpu", weights_only=True)
    v.load_state_dict(sd.get("state_dict", sd), strict=True)
    return v.float().eval()


vae = load_vae()


# ---- tokenizer cases ----------------------------------------------------------------------------------------------
tok = mask_multichar_chinese_tokens(tokenizer)
cases = [TEXT, "Guten Morgen, heute scheint die Sonne.", "Größe, Maß und Übermut – äöüß ÄÖÜ!",
         "(Sprich wütend und laut)Ihr könnt mich nicht aufhalten!", "Hello world. 12345 tokens?",
         "你好世界，今天天气很好。", "  doppelte  Leerzeichen\tund Tab ", "Emoji 😀 und € Zeichen…"]
META["tokenizer_cases"] = [{"text": c, "ids": [int(i) for i in tok(c)]} for c in cases]

# ---- reference audio -> VAE encoder -------------------------------------------------------------------------------
text = TEXT.replace("\n", " ")
import re
text = re.sub(r"\s+", " ", text)
audio, _ = librosa.load(VOICE, sr=16000, mono=True)
put("ref_wave16k_raw", torch.from_numpy(audio))
audio = torch.from_numpy(audio).unsqueeze(0)
patch_len = config.patch_size * vae.chunk_size
if audio.size(1) % patch_len != 0:
    audio = F.pad(audio, (0, patch_len - audio.size(1) % patch_len))
put("ref_wave16k", audio[0])
enc_trace = {}
hooks = [vae.encoder.block[i].register_forward_hook(lambda m, a, o, i=i: enc_trace.update({i: o}))
         for i in (0, 4)]
t1 = time.time()
mu = vae.encode(audio, 16000)  # [1, 64, frames]
for h in hooks:
    h.remove()
put("vae_enc_block0", enc_trace[0][0, :, :4000])   # first 4000 samples of the first conv
put("vae_enc_block4", enc_trace[4][0])
put("vae_enc_mu", mu[0])
ref_feat = mu.view(vae.latent_dim, -1, config.patch_size).permute(1, 2, 0)  # [T, P, D]
put("ref_feat", ref_feat)
META["vae_encode_s"] = round(time.time() - t1, 1)
latent_dim = vae.latent_dim
# The AudioVAE runs before the LM parts are loaded (peak RAM); the model only keeps a stub.
vae_stub = nn.Module()
for _name in ("chunk_size", "decode_chunk_size", "sample_rate", "out_sample_rate", "latent_dim"):
    setattr(vae_stub, _name, getattr(vae, _name))
del vae, enc_trace, mu, audio
import gc
gc.collect()
print("vae encode", META["vae_encode_s"], "s, ref patches", ref_feat.shape[0], flush=True)

with torch.device("meta"):
    model = VoxCPM2Model(config, tokenizer, vae_stub, device="cpu")
# setup_cache ran on "cpu" inside the meta context? make sure the caches are real tensors
for lm in (model.base_lm, model.residual_lm):
    lm.setup_cache(1, config.max_length, "cpu", torch.float32)
for prefix, module in [("feat_encoder.", model.feat_encoder), ("feat_decoder.", model.feat_decoder),
                       ("fsq_layer.", model.fsq_layer), ("enc_to_lm_proj.", model.enc_to_lm_proj),
                       ("lm_to_dit_proj.", model.lm_to_dit_proj), ("res_to_dit_proj.", model.res_to_dit_proj),
                       ("fusion_concat_proj.", model.fusion_concat_proj), ("stop_proj.", model.stop_proj),
                       ("stop_head.", model.stop_head)]:
    assert not load_module(module, prefix), prefix
for lm, prefix in ((model.base_lm, "base_lm."), (model.residual_lm, "residual_lm.")):
    load_module(lm.norm, prefix + "norm.")
MAX_STEPS = int(os.environ.get("VOXCPM_ORACLE_STEPS", "8"))
META["max_steps_cap"] = MAX_STEPS
# rebuild the non-persistent LongRoPE tables on the CPU (they were created on meta)
for module in model.modules():
    if isinstance(module, MiniCPMLongRoPE):
        module.inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float() / module.dim))
        module._set_cos_sin_cache(seq_len=module.max_position_embeddings, device="cpu", dtype=torch.float32)
model.eval()
META["load_s"] = round(time.time() - t0, 1)
print("loaded", META["load_s"], "s", flush=True)

# ---- inputs (VoxCPM2Model._generate_with_prompt_cache, mode "reference") ------------------------------------------
text_token = torch.LongTensor(tok(text))
text_token = torch.cat([text_token, torch.tensor([model.audio_start_token], dtype=torch.int32)], dim=-1)
text_length = text_token.shape[0]
target_text_length = len(tok(text))
ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat, text_token.device)
text_pad_feat = torch.zeros((text_length, model.patch_size, latent_dim), dtype=torch.float32)
text_token = torch.cat([ref_tokens, text_token])
audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32)])
audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32)])
put("input_ids", text_token)
put("text_mask", text_mask)
put("audio_mask", audio_mask)
META["input_ids"] = [int(i) for i in text_token]
META["target_text_length"] = target_text_length
max_len = min(int(target_text_length * 6.0 + 10), 4096)
META["max_len"] = max_len

text_token = text_token.unsqueeze(0)
text_mask = text_mask.unsqueeze(0)
feat = audio_feat.unsqueeze(0).float()
feat_mask = audio_mask.unsqueeze(0)

# ---- _inference, prefill --------------------------------------------------------------------------------------------
torch.manual_seed(SEED)
noises = []
_randn = torch.randn


def rec_randn(*a, **k):
    r = _randn(*a, **k)
    noises.append(r.clone())
    return r


torch.randn = rec_randn
est_calls = []
est_hook = model.feat_decoder.estimator.register_forward_hook(
    lambda m, a, o: est_calls.append([x.clone() for x in a] + [o.clone()]) if len(noises) == 1 else None)

t1 = time.time()
B, Tn, P, D = feat.shape
locenc_trace = {}
h1 = model.feat_encoder.encoder.layers[0].register_forward_hook(lambda m, a, o: None if "l0" in locenc_trace else locenc_trace.update(l0=o[0]))
feat_cls = model.feat_encoder(feat)
h1.remove()
put("locenc_l0_row0", locenc_trace["l0"][0])  # [5, 1024] of the first prefill sequence
put("locenc_out", feat_cls[0])
feat_embed = model.enc_to_lm_proj(feat_cls)
put("feat_embed", feat_embed[0])
load_module(model.base_lm.embed_tokens, "base_lm.embed_tokens.")
text_embed = model.base_lm.embed_tokens(text_token) * 1.0
unload(model.base_lm.embed_tokens)
combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
put("combined_embed", combined_embed[0])
prefix_feat_cond = feat[:, -1, ...]
pred_feat_seq = []
enc_outputs, base_l0 = stream_forward(model.base_lm, "base_lm.", combined_embed)
put("base_l0", base_l0)
put("base_out", enc_outputs[0])
enc_outputs = model.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
put("enc_outputs", enc_outputs[0])
lm_hidden = enc_outputs[:, -1, :]
residual_enc_inputs = model.fusion_concat_proj(torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1))
put("residual_inputs", residual_enc_inputs[0])
residual_enc_outputs, _ = stream_forward(model.residual_lm, "residual_lm.", residual_enc_inputs)
put("residual_out", residual_enc_outputs[0])
residual_hidden = residual_enc_outputs[:, -1, :]
META["prefill_s"] = round(time.time() - t1, 1)
print("prefill rows", combined_embed.shape[1], META["prefill_s"], "s", flush=True)

# ---- AR loop --------------------------------------------------------------------------------------------------------
steps = {k: [] for k in ("lm_hidden", "residual_hidden", "dit_hidden", "cond", "noise", "pred_feat", "curr_embed",
                         "stop_logits", "locenc_cls")}
t1 = time.time()
stopped = False
for i in range(min(max_len, MAX_STEPS)):
    steps["lm_hidden"].append(lm_hidden[0].clone())
    steps["residual_hidden"].append(residual_hidden[0].clone())
    dit_hidden_1 = model.lm_to_dit_proj(lm_hidden)
    dit_hidden_2 = model.res_to_dit_proj(residual_hidden)
    dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)
    steps["dit_hidden"].append(dit_hidden[0].clone())
    steps["cond"].append(prefix_feat_cond[0].clone())  # [P, D]
    n_before = len(noises)
    pred_feat = model.feat_decoder(mu=dit_hidden, patch_size=model.patch_size,
                                   cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                                   n_timesteps=TIMESTEPS, cfg_value=CFG).transpose(1, 2)  # [b, p, d]
    assert len(noises) == n_before + 1
    steps["noise"].append(noises[-1][0].transpose(0, 1).clone())  # [P, D]
    steps["pred_feat"].append(pred_feat[0].clone())
    cls = model.feat_encoder(pred_feat.unsqueeze(1))
    steps["locenc_cls"].append(cls[0, 0].clone())
    curr_embed = model.enc_to_lm_proj(cls)
    steps["curr_embed"].append(curr_embed[0, 0].clone())
    pred_feat_seq.append(pred_feat.unsqueeze(1))
    prefix_feat_cond = pred_feat
    stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
    steps["stop_logits"].append(stop_logits[0].clone())
    stop_flag = stop_logits.argmax(dim=-1)[0].item()
    print("step", i, "stop", stop_flag, "logits", [round(float(x), 4) for x in stop_logits[0]], flush=True)
    if i > 2 and stop_flag == 1:
        stopped = True
        break
    if i + 1 >= min(max_len, MAX_STEPS):
        break  # cap reached: the next hidden states are not needed
    lm_hidden = stream_step(model.base_lm, "base_lm.", curr_embed[:, 0, :],
                            torch.tensor([model.base_lm.kv_cache.step()])).clone()
    lm_hidden = model.fsq_layer(lm_hidden)
    curr_residual_input = model.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
    residual_hidden = stream_step(model.residual_lm, "residual_lm.", curr_residual_input,
                                  torch.tensor([model.residual_lm.kv_cache.step()])).clone()
torch.randn = _randn
est_hook.remove()
for k, v in steps.items():
    put("step_" + k, torch.stack(v))
META["steps"] = len(steps["pred_feat"])
META["stopped"] = stopped
META["ar_s"] = round(time.time() - t1, 1)
print("AR steps", META["steps"], "stopped", stopped, META["ar_s"], "s", flush=True)

# first step's estimator calls (9 CFG evaluations): x_in, mu_in, t_in, cond_in, dt_in -> velocity
put("est0_x", torch.stack([c[0] for c in est_calls]))
put("est0_mu", torch.stack([c[1] for c in est_calls]))
put("est0_t", torch.stack([c[2] for c in est_calls]))
put("est0_cond", torch.stack([c[3] for c in est_calls]))
put("est0_out", torch.stack([c[5] for c in est_calls]))
t_span = torch.linspace(1, 0, TIMESTEPS + 1, dtype=torch.float32)
t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
put("t_span", t_span)

# ---- VAE decode -----------------------------------------------------------------------------------------------------
pred_feat_seq = torch.cat(pred_feat_seq, dim=1)
del model
gc.collect()
vae = load_vae()
latent = pred_feat_seq.permute(0, 3, 1, 2).reshape(1, vae.latent_dim, -1)  # b d (t p)
put("latent", latent[0])
t1 = time.time()
dec_trace = {}
dh = [vae.decoder.model[i].register_forward_hook(lambda m, a, o, i=i: dec_trace.update({i: o})) for i in (1, 2, 7)]
wave = vae.decode(latent.float())
for h in dh:
    h.remove()
put("vae_dec_model1", dec_trace[1][0])
put("vae_dec_model2", dec_trace[2][0])
put("vae_dec_model7_head", dec_trace[7][0, :, :4000])
put("wave", wave[0, 0])
META["vae_decode_s"] = round(time.time() - t1, 1)
META["samples"] = int(wave.shape[-1])
print("vae decode", META["vae_decode_s"], "s samples", META["samples"], flush=True)

save_file(T, os.path.join(OUT, "oracle.safetensors"))
with open(os.path.join(OUT, "oracle.json"), "w", encoding="utf-8") as f:
    json.dump(META, f, ensure_ascii=False, indent=1)
import soundfile as sf
sf.write(os.path.join(OUT, "oracle.wav"), wave[0, 0].numpy(), 48000)
print("=== fertig", round(time.time() - t0, 1), "s", flush=True)
