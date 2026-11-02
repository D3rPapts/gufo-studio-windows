"""CPU fp32 oracle for Ming-3.0-tts prompt latents (audio-VAE encoder) and the ZERO_SHOT prompt layout.

Reference voice + transcript -> resample to 44.1 kHz, front padding, VAE encoder (Qwen2 24 layers + 4-layer
aggregator with CLS per 4-frame patch, fc3 -> mean/scale, softplus std, posterior sample), aggregator projection
of every prompt latent into the backbone input, prefill and the first AR steps (teacher forcing data).
Writes D:\\llm-modelle\\gufo\\ft\\ming-ref\\oracle_zs.safetensors + oracle_zs.json. CPU only.
Written with AI assistance (Claude Code)."""
import json, os, random, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
REPO = r"D:\llm-modelle\inclusionAI\src\Ming-3.0-tts"
sys.path.insert(0, REPO)
MODEL = r"D:\llm-modelle\inclusionAI\Ming-3.0-tts-0.5B"
OUT = r"D:\llm-modelle\gufo\ft\ming-ref"
VOICE = r"D:\claude\gufo-arbeit\chatterbox\stimmen\fabian_ref.wav"
PROMPT_TEXT = ("Hallo und herzlich willkommen! Heute lese ich ein paar Sätze vor, ganz ruhig und natürlich, damit eine "
               "künstliche Intelligenz lernen kann, wie meine Stimme klingt.")
TEXT = "Wir sehen uns morgen am Bahnhof."
SEED = 77
STEPS = 4
torch.set_num_threads(8)
torch.set_grad_enabled(False)

import soundfile as sf
import torchaudio
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import prompt_contract
import tts_tasks
from configuration_bailingmm import _relax_flash_attn_pins
from fm.dit import Aggregator
from fm.flowloss import FlowLoss
from audio_tokenizer.vae_modules import Encoder
from spkemb_extractor import SpkembExtractor

T, META = {}, {"text": TEXT, "prompt_text": PROMPT_TEXT, "seed": SEED}


def put(name, t):
    t = t.detach().cpu()
    T[name] = (t.float() if t.is_floating_point() else t.long()).contiguous().clone()


cfg = json.load(open(os.path.join(MODEL, "config.json"), encoding="utf-8"))
_relax_flash_attn_pins(cfg["audio_tokenizer_config"])
st = safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt", device="cpu")


def load(module, prefix):
    sd = {k[len(prefix):]: st.get_tensor(k).float() for k in st.keys() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("lm_head.weight")]
    assert not missing and not unexpected, (prefix, missing, unexpected)
    return module.float().eval()


t0 = time.time()
vcfg = cfg["audio_tokenizer_config"]
enc = load(Encoder(encoder_args=vcfg["enc_kwargs"]["backbone"], input_dim=vcfg["enc_kwargs"]["input_dim"],
                   hop_size=vcfg["enc_kwargs"]["hop_size"], latent_dim=vcfg["enc_kwargs"]["latent_dim"],
                   patch_size=vcfg["patch_size"]), "audio.encoder.")

data, sr = sf.read(VOICE, dtype="float32", always_2d=True)
wav = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
META["ref_sample_rate"] = sr
spk = SpkembExtractor(os.path.join(MODEL, "campplus.onnx"))(
    torchaudio.transforms.Resample(sr, 16000)(wav) if sr != 16000 else wav)
put("spk_emb", spk[0])
wav44 = torchaudio.transforms.Resample(sr, 44100)(wav) if sr != 44100 else wav
put("ref_wav44k", wav44[0])
align = enc.hop_size * max(1, enc.patch_size) * cfg["ditar_config"]["patch_size"]
rem = wav44.shape[-1] % align
if rem:
    padded = torch.zeros((1, wav44.shape[-1] + align - rem))
    padded[:, -wav44.shape[-1]:] = wav44
    wav44 = padded
put("prompt_wav", wav44[0])

# encoder internals
x = enc.get_frames(wav44)
x = enc.fc2(enc.fc1(x))
h = enc.encoder(inputs_embeds=x, output_hidden_states=True)
put("enc_hidden_l0", h.hidden_states[1][0])
put("enc_hidden_last", h.last_hidden_state[0])
hh = enc.pad_patch_insert_cls(h.last_hidden_state)
a = enc.aggregator(inputs_embeds=hh).last_hidden_state
bsz, _, dim = a.size()
a = a.reshape(-1, enc.patch_size + 1, dim)[:, -1:, :].reshape(bsz, -1, dim)
params = enc.fc3(a)  # [1, P, 128]
put("enc_params", params[0])
ref_h, _ = enc(wav44)
META["enc_split_max_diff"] = float((ref_h - params).abs().max())
mean, scale = params.transpose(1, 2).chunk(2, dim=1)
std = F.softplus(scale) + 1e-4
g = torch.Generator().manual_seed(SEED)
noise = torch.randn(mean.shape, generator=g)
latent = (mean + std * noise).transpose(1, 2)  # [1, P, 64]
put("enc_noise", noise[0].transpose(0, 1))
put("prompt_latent", latent[0])
del enc
print("encoder done", round(time.time() - t0, 1), "s, latents", latent.shape[1], flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
req = tts_tasks.build_task_generation_kwargs("TTS", text=TEXT, language_id="de", prompt_wav_path=VOICE,
                                             prompt_text=PROMPT_TEXT, use_spk_emb=True)
ids, spk_pos, lat_pos = prompt_contract.build_input_ids(tok, "dense", req["prompt"], TEXT, num_speakers=1,
                                                       instruction=req["instruction"], prompt_text=PROMPT_TEXT,
                                                       prompt_latent_steps=latent.shape[1])
META.update(input_ids=ids, speaker_positions=spk_pos, latent_position=lat_pos, prompt=req["prompt"])
put("input_ids", torch.tensor(ids))

llm = load(Qwen2ForCausalLM(Qwen2Config.from_dict(cfg["llm_config"])), "model.")
H = llm.config.hidden_size
spk_head = load(torch.nn.Linear(192, H), "spk_head.")
stop_head = load(torch.nn.Linear(H, 2), "stop_head.")
agg = load(Aggregator(in_channels=64, llm_input_dim=H, **cfg["aggregator_config"]), "linear_proj_audio.")
flow = load(FlowLoss(z_channels=64, llm_cond_dim=H, **cfg["ditar_config"]), "flowloss.")
proj_latent = agg(latent.reshape(-1, 1, 64)).reshape(1, -1, H)
put("prompt_latent_proj", proj_latent[0])
emb = llm.get_input_embeddings()(torch.tensor(ids).unsqueeze(0))
emb[0, spk_pos[0]] = spk_head(spk)[0]
emb[0, lat_pos:lat_pos + proj_latent.shape[1]] = proj_latent[0]
put("inputs_embeds", emb[0])

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
pos = torch.arange(emb.shape[1]).unsqueeze(0)
past, inp = None, emb
hist = latent[:, -1:, :].clone()
put("history0", hist[0])
conds, noises, lats, stops = [], [], [], []
for step in range(STEPS):
    out = llm(inputs_embeds=inp, position_ids=pos, use_cache=True, past_key_values=past, output_hidden_states=True)
    past = out.past_key_values
    z = out.hidden_states[-1][:, -1:, :]
    if step == 0:
        put("prefill_hidden_last", out.hidden_states[-1][0])
    conds.append(z[0, 0])
    state = torch.get_rng_state(); noises.append(torch.randn(1, 1, 64)[0]); torch.set_rng_state(state)
    s = flow.sample(z, hist, 2.0, 1, sigma=0.25, temperature=0.0)
    lats.append(s[0]); stops.append(stop_head(z)[0][0].softmax(-1))
    inp = agg(s); pos = pos[:, -1:] + 1; hist = s.clone()
put("cond", torch.stack(conds)); put("noise", torch.stack(noises)); put("latents", torch.cat(lats))
put("stop_prob", torch.stack(stops))
save_file(T, os.path.join(OUT, "oracle_zs.safetensors"))
META["seconds"] = round(time.time() - t0, 1)
json.dump(META, open(os.path.join(OUT, "oracle_zs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("=== fertig", META["seconds"], "s", flush=True)
