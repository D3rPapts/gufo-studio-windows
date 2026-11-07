"""CPU fp32 reference oracle for the native gufo port of Ming-3.0-tts-0.5B (inclusionAI, MIT).

Runs one short German EMOTION request (reference voice -> CAM++ speaker embedding, no prompt latents) on the CPU in
float32 with fixed seeds and writes every intermediate the C++ port is compared against to one safetensors file
(D:\\llm-modelle\\gufo\\ft\\ming-ref\\oracle.safetensors) plus a JSON sidecar (token ids, strings, step counts).

Only the parts the EMOTION path needs are built (Qwen2 backbone, spk/stop heads, aggregator, DiT flow head, VAE
decoder); the VAE encoder stays unloaded to keep the process below ~6 GB. The AR loop is a line-by-line copy of
BailingMMNativeForConditionalGeneration.sample (commit 9332fb94) so intermediate tensors can be captured.

Run: venv-ming python -I ming_oracle.py   (CPU only, no CUDA/ROCm device is touched)
Written with AI assistance (Claude Code)."""
import json, math, os, random, sys, time

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
TEXT = "Guten Morgen, heute scheint die Sonne."
EMOTION = "高兴"
SEED = 1234
MAX_STEPS = int(os.environ.get("MING_ORACLE_MAX_STEPS", "120"))

torch.set_num_threads(8)
torch.set_grad_enabled(False)
assert not torch.cuda.is_available() or os.environ["HIP_VISIBLE_DEVICES"] == ""

import soundfile as sf
import torchaudio
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import prompt_contract
import tts_tasks
from fm.dit import Aggregator
from fm.flowloss import FlowLoss
from fm.CFM import get_epss_timesteps
from audio_tokenizer.vae_modules import Decoder
from frontend.text_segment_cut import cut_text_by_semantic_length, get_semantic_length
from frontend.toolkit import preprosess_text
from spkemb_extractor import SpkembExtractor
import torchaudio.compliance.kaldi as kaldi

os.makedirs(OUT, exist_ok=True)
T = {}  # tensors to save (float32 / int64)
META = {"text": TEXT, "emotion": EMOTION, "seed": SEED, "voice": VOICE, "torch": torch.__version__}


def put(name, tensor):
    t = tensor.detach().to("cpu")
    t = t.to(torch.float32) if t.is_floating_point() else t.to(torch.int64)
    T[name] = t.contiguous().clone()


cfg = json.load(open(os.path.join(MODEL, "config.json"), encoding="utf-8"))
st = safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt", device="cpu")


def load(module, prefix):
    sd = {}
    for key in st.keys():
        if key.startswith(prefix):
            sd[key[len(prefix):]] = st.get_tensor(key).to(torch.float32)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("lm_head.weight")]
    assert not missing and not unexpected, (prefix, missing, unexpected)
    return module.float().eval()


t0 = time.time()
# ---- text frontend cases (segmentation contract) -------------------------------------------------------------------
seg_cases = [
    TEXT,
    "Das ist ein deutlich längerer Text, der in mehrere Teile zerlegt werden muss, weil er mehr als vierundzwanzig "
    "Wörter enthält. Danach folgt noch ein zweiter Satz! Und ein dritter?",
    "Kurz. Noch kürzer. Und dann ein Satz mit Zahlen wie 3,5 Millionen und 2026 – aber ohne Punkt am Ende",
    "Er flüstert: Komm her, schnell, wir haben keine Zeit mehr; sie kommen gleich zurück und dann ist alles vorbei.",
]
META["segments"] = [{"text": c, "pre": preprosess_text(c).strip(),
                     "fragments": cut_text_by_semantic_length(preprosess_text(c).strip(), 24)["fragments"],
                     "semantic_length": get_semantic_length(c)} for c in seg_cases]

# ---- prompt / tokens -----------------------------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompt_contract.validate_tokenizer(tok, "dense")
req = tts_tasks.build_task_generation_kwargs("EMOTION", text=TEXT, language_id="de", controls={"emotion": EMOTION},
                                             prompt_wav_path=VOICE, use_spk_emb=True)
META["prompt"] = req["prompt"]
META["instruction"] = req["instruction"]
ids, spk_pos, lat_pos = prompt_contract.build_input_ids(tok, "dense", req["prompt"], TEXT, num_speakers=1,
                                                       instruction=req["instruction"])
META["input_ids"] = ids
META["speaker_positions"] = spk_pos
put("input_ids", torch.tensor(ids))
# extra tokenizer cases (numbers, umlauts, punctuation, JSON instruction, freeform)
tok_cases = [TEXT, "Größe 1234 und 3,5 Millionen – „Zitat“ … ÄÖÜß!", req["instruction"], "  speaker_1:",
             tts_tasks.build_task_instruction("EMOTION", emotion="恐惧"), "Sprich leise und flüsternd, fast tonlos.",
             "Hallo\n\nWelt  zwei   Leerzeichen\t tab"]
META["tokenizer_cases"] = [{"text": c, "ids": tok.encode(c, add_special_tokens=False)} for c in tok_cases]
# all EMOTION instructions (the C++ side must reproduce the exact JSON)
META["emotion_instructions"] = {e: tts_tasks.build_task_instruction("EMOTION", emotion=e) for e in tts_tasks.EMOTION_VALUES}
META["freeform_example"] = tts_tasks.build_task_generation_kwargs(
    "FREEFORM", text=TEXT, language_id="de", instruction="Sprich leise flüsternd.", prompt_wav_path=VOICE,
    use_spk_emb=True)["instruction"]

# ---- speaker embedding (CAM++) -------------------------------------------------------------------------------------
data, sr = sf.read(VOICE, dtype="float32", always_2d=True)
wav = torch.from_numpy(data.T.copy())
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
put("ref_wav", wav[0])
META["ref_sample_rate"] = sr
wav16 = torchaudio.transforms.Resample(sr, 16000)(wav) if sr != 16000 else wav
put("ref_wav16k", wav16[0])
feat = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
put("fbank_raw", feat)
feat = feat - feat.mean(dim=0, keepdim=True)
put("fbank", feat)
ext = SpkembExtractor(os.path.join(MODEL, "campplus.onnx"))
spk = ext(wav16)  # (1,192)
put("spk_emb", spk[0])

# ---- modules -------------------------------------------------------------------------------------------------------
llm_cfg = Qwen2Config.from_dict(cfg["llm_config"])
llm = load(Qwen2ForCausalLM(llm_cfg), "model.")
META["llm_attn_impl"] = llm.config._attn_implementation
H = llm_cfg.hidden_size
spk_head = load(torch.nn.Linear(192, H, bias=True), "spk_head.")
stop_head = load(torch.nn.Linear(H, 2, bias=True), "stop_head.")
lat_dim = cfg["audio_tokenizer_config"]["enc_kwargs"]["latent_dim"]
agg = load(Aggregator(in_channels=lat_dim, llm_input_dim=H, **cfg["aggregator_config"]), "linear_proj_audio.")
flow = load(FlowLoss(z_channels=lat_dim, llm_cond_dim=H, **cfg["ditar_config"]), "flowloss.")
patch = cfg["ditar_config"]["patch_size"]
hist_patch = cfg["ditar_config"].get("history_patch_size", patch)
print("modules loaded", round(time.time() - t0, 1), "s", flush=True)

# ---- inputs_embeds -------------------------------------------------------------------------------------------------
proj_spk = spk_head(spk)
put("spk_proj", proj_spk[0])
input_ids = torch.tensor(ids).unsqueeze(0)
emb = llm.get_input_embeddings()(input_ids)
for p in spk_pos:
    emb[0, p] = proj_spk[0]
put("inputs_embeds", emb[0])

# ---- AR loop (copy of BailingMMNativeForConditionalGeneration.sample, dense, temperature 0) ------------------------
cfg_scale, sigma, temperature = 2.0, 0.25, 0.0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
attention_mask = torch.ones(input_ids.shape)
position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1).long()
past = None
inputs_embeds = emb
latents, conds, stops, noises, aggs = [], [], [], [], []
dit_trace = {}
orig_fwc = flow.cfm.model.forward_with_cfg
calls = {"n": 0}


def traced_fwc(x, t, c, latent_history, patch_size):
    out = orig_fwc(x=x, t=t, c=c, latent_history=latent_history, patch_size=patch_size)
    if calls["step"] == 0:
        k = calls["n"]
        dit_trace[f"dit0_x_{k}"] = x[0].clone()
        dit_trace[f"dit0_t_{k}"] = t.reshape(-1)[:1].clone()
        dit_trace[f"dit0_out_{k}"] = out.clone()  # (2, patch, 64): cond, uncond
        calls["n"] += 1
    return out


flow.cfm.model.forward_with_cfg = traced_fwc
for step in range(MAX_STEPS):
    calls["step"] = step
    out = llm(attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds,
              output_hidden_states=True, return_dict=True, use_cache=True, past_key_values=past)
    past = out.past_key_values
    z = out.hidden_states[-1][:, -1:, :]
    if step == 0:
        put("prefill_hidden_l0", out.hidden_states[1][0])
        put("prefill_hidden_l12", out.hidden_states[12][0])
        put("prefill_hidden_last", out.hidden_states[-1][0])
        latent_history = torch.zeros(1, hist_patch, lat_dim)
    conds.append(z[0, 0].clone())
    # flowloss.sample draws noise then sde noise (steps=10) from the global RNG, in this order
    state = torch.get_rng_state()
    noise_peek = torch.randn((1, patch, lat_dim))
    torch.set_rng_state(state)
    noises.append(noise_peek[0].clone())
    if step == 0:
        put("dit0_latent_history", latent_history[0])
    sampled = flow.sample(z, latent_history, cfg_scale, patch, sigma=sigma, temperature=temperature)
    latents.append(sampled[0].clone())
    prob = stop_head(z)[0][0].softmax(dim=-1)
    stops.append(prob.clone())
    if prob[1] > 0.5 and step > 3:
        break
    inputs_embeds = agg(sampled)
    aggs.append(inputs_embeds[0, 0].clone())
    position_ids = position_ids[:, -1:] + 1
    attention_mask = torch.ones(inputs_embeds.shape[0], 1)
    latent_history[:, :-patch, :] = latent_history[:, patch:, :].clone()
    latent_history[:, -patch:, :] = sampled[:]
    if step % 10 == 0:
        print("step", step, "stop", float(prob[1]), round(time.time() - t0, 1), "s", flush=True)

META["steps"] = len(latents)
META["stopped"] = bool(stops[-1][1] > 0.5)
put("cond", torch.stack(conds))            # (S, 896)   LLM final normed hidden per step
put("noise", torch.stack(noises))          # (S, 1, 64) initial flow noise per step
put("latents", torch.cat(latents, dim=0))  # (S, 64)     sampled latents
put("stop_prob", torch.stack(stops))       # (S, 2)
put("agg", torch.stack(aggs))              # (S-1, 896) aggregator output fed back
for k, v in dit_trace.items():
    put(k, v)
tt = get_epss_timesteps(10, device="cpu", dtype=torch.float32)
tt = tt + -1.0 * (torch.cos(torch.pi / 2 * tt) - 1 + tt)
put("dit_timesteps", tt)
print("AR done, steps", len(latents), "stopped", META["stopped"], round(time.time() - t0, 1), "s", flush=True)
del llm, past, agg, flow
save_file(T, os.path.join(OUT, "oracle_ar.safetensors"))
json.dump(META, open(os.path.join(OUT, "oracle_ar.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- VAE decode ----------------------------------------------------------------------------------------------------
vcfg = cfg["audio_tokenizer_config"]
from configuration_bailingmm import _relax_flash_attn_pins  # noqa: E402  (same downgrade MingTTS applies: SDPA)
_relax_flash_attn_pins(vcfg)
dec = Decoder(decoder_args=vcfg["dec_kwargs"]["backbone"], output_dim=vcfg["dec_kwargs"]["output_dim"],
              latent_dim=vcfg["dec_kwargs"]["latent_dim"], semantic_model=None, patch_size=vcfg["patch_size"])
dec = load(dec, "audio.decoder.")
META["vae_attn_impl"] = dec.decoder.config._attn_implementation
lat = torch.cat(latents, dim=0).unsqueeze(0)
x = dec.fc1(lat)
put("vae_fc1", x[0])
x = dec.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)
put("vae_upsampled", x[0])
o = dec.decoder(inputs_embeds=x, output_hidden_states=True)
put("vae_hidden_l0", o.hidden_states[1][0])
put("vae_hidden_last", o.last_hidden_state[0])
head_in = o.last_hidden_state
xp = dec.head.out(head_in)
put("vae_head_out", xp[0])
audio, _, _, _ = dec.head(head_in, streaming=False, last_chunk=True)
put("wav", audio[0, 0])
# same path through the module API, to make sure the manual split above matches low_level_reconstruct
ref_audio = dec.low_level_reconstruct(lat, past_key_values=None, use_cache=False,
                                      stream_state=(None, None, None), last_chunk=True)[0]
META["vae_split_matches"] = float((ref_audio[0, 0] - audio[0, 0]).abs().max())
# trailing-silence trim as in ming_tts.trim_trailing_silence
from ming_tts import trim_trailing_silence  # noqa: E402
trimmed = trim_trailing_silence(audio[0].float(), 44100)
put("wav_trimmed", trimmed[0])
META["wav_samples"] = int(audio.shape[-1])
META["wav_trimmed_samples"] = int(trimmed.shape[-1])
sf.write(os.path.join(OUT, "oracle.wav"), trimmed[0].numpy(), 44100)

# ---- VAE sliding-window probe: a 300-frame input with random latents to pin down the window width -----------------
g = torch.Generator().manual_seed(7)
probe = torch.randn(1, 75, lat_dim, generator=g) * 0.5
put("vae_probe_latents", probe[0])
pa = dec.low_level_reconstruct(probe, past_key_values=None, use_cache=False, stream_state=(None, None, None),
                               last_chunk=True)[0]
put("vae_probe_wav", pa[0, 0])

save_file(T, os.path.join(OUT, "oracle.safetensors"))
META["seconds"] = round(time.time() - t0, 1)
json.dump(META, open(os.path.join(OUT, "oracle.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("=== fertig", META["seconds"], "s; steps", META["steps"], "samples", META["wav_samples"], flush=True)
