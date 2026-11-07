"""CPU fp32 reference oracle for the native gufo port of IndexTTS-2.5 with the German GPT fine-tune
(IndexTeam/IndexTTS-2.5 + sharrnah/index-tts-2.5-german, bilibili Model Use License).

Runs one short German request with the settings of indextts_de.py (lang "de", prepare_german + text_normalization off,
8-value emotion vector, num_beams 1, temperature 0.7, top_p 0.8, top_k 30, repetition_penalty 10) on the CPU in float32
with a fixed seed and writes every intermediate the C++ port is compared against to oracle.safetensors + oracle.json.

The pipeline is a line-by-line copy of IndexTTS2.infer_generator (infer_v2_5.py, TTS-Audio-Suite d12a4c28) using the
original modules (UnifiedVoice, ConformerEncoder, PerceiverResampler, EnhancedCodec, MyModel/CFM/DiT, CAMPPlus,
BigVGAN, Wav2Vec2BertModel layers). Only two things differ, both for reproducibility in C++:
  * sampling: HF generate runs with do_sample=False; a last logits processor applies HF's own Temperature/TopK/TopP
    warpers to the repetition-penalised scores and draws the token by inverse CDF with a recorded uniform number
    (numpy RandomState(seed)), returning a one-hot score so greedy search takes that token;
  * memory (<= ~2 GB working set): checkpoints are mmap-loaded, the GPT is built on the meta device and assigned the
    mapped tensors, the 17 needed w2v-BERT layers are built and run one at a time.

Run: venv-indextts python -I indextts_oracle.py <bundle_dir> <out_dir> <ref_code_dir (…/engines/index_tts)>
Env: INDEXTTS_ORACLE_TEXT, INDEXTTS_ORACLE_VOICE, INDEXTTS_ORACLE_EMO (8 comma separated floats), INDEXTTS_ORACLE_SEED,
     INDEXTTS_ORACLE_THREADS, INDEXTTS_ORACLE_MAXTOK
CPU only; no CUDA/ROCm device is touched. Written with AI assistance (Claude Code)."""
import gc, importlib.util, json, math, os, re, sys, time, types, wave

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
BUNDLE, OUT, REF = sys.argv[1], sys.argv[2], sys.argv[3]
VOICE = os.environ.get("INDEXTTS_ORACLE_VOICE", r"D:\llm-modelle\gufo\voices\erzaehler-9s.wav")
TEXT = os.environ.get("INDEXTTS_ORACLE_TEXT", "Ich kann nicht mehr. Es tut so weh.")
EMO = [float(x) for x in os.environ.get("INDEXTTS_ORACLE_EMO", "0,0,0.6,0,0,0.2,0,0").split(",")]
SEED = int(os.environ.get("INDEXTTS_ORACLE_SEED", "1234"))
MAXTOK = int(os.environ.get("INDEXTTS_ORACLE_MAXTOK", "400"))
torch.set_num_threads(int(os.environ.get("INDEXTTS_ORACLE_THREADS", "8")))
torch.set_grad_enabled(False)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REF)

# indextts.utils.tokenizer imports whisper.tokenizer.Tokenizer only to subclass it; get_encoding does not need it.
_wh = types.ModuleType("whisper")
_wht = types.ModuleType("whisper.tokenizer")
_wht.Tokenizer = type("Tokenizer", (), {})
_wh.tokenizer = _wht
sys.modules["whisper"], sys.modules["whisper.tokenizer"] = _wh, _wht

import librosa
import torchaudio
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

T, META = {}, {"text": TEXT, "voice": VOICE, "emo_vector": EMO, "seed": SEED, "torch": torch.__version__}
t_start = time.time()


def put(name, tensor):
    t = torch.as_tensor(tensor).detach().to("cpu")
    t = t.to(torch.float32) if t.is_floating_point() else t.to(torch.int64)
    T[name] = t.contiguous().clone()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


import collections, collections.abc, pickle, struct, zipfile
_NP = {"FloatStorage": np.float32, "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16,
       "DoubleStorage": np.float64, "BoolStorage": np.bool_}


class _Ref:  # a tensor inside a torch zip checkpoint, read on access with plain file reads (no mmap / commit)
    def __init__(self, st, off, size, stride):
        self.st, self.off, self.size = st, off, tuple(size)

    def load(self, zf, prefix):
        dtype = _NP[self.st[0]]
        raw = bytearray(zf.read(prefix + "data/" + str(self.st[1])))
        arr = np.frombuffer(raw, dtype=dtype)
        n = int(np.prod(self.size)) if self.size else 1
        t = torch.from_numpy(arr[self.off:self.off + n].reshape(self.size))
        return t


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return lambda st, off, size, stride, *a: _Ref(st, off, size, stride)
        if name == "_rebuild_from_type_v2":
            return lambda func, typ, args, state: func(*args)
        if name == "_rebuild_parameter":
            return lambda t, *a: t
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if name.endswith("Storage"):
            return name
        if mod in ("builtins", "_codecs", "collections"):
            return super().find_class(mod, name)
        return lambda *a, **k: None

    def persistent_load(self, pid):
        return (pid[1], pid[2])


class LazyDict(collections.abc.Mapping):
    def __init__(self, d, zf, prefix):
        self.d, self.zf, self.prefix = d, zf, prefix

    def __getitem__(self, k):
        v = self.d[k]
        if isinstance(v, _Ref):
            return v.load(self.zf, self.prefix)
        if isinstance(v, dict):
            return LazyDict(v, self.zf, self.prefix)
        return v

    def __contains__(self, k):
        return k in self.d

    def __iter__(self):
        return iter(self.d)

    def __len__(self):
        return len(self.d)


def load_pth(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    root = _Unpickler(zf.open(pkl)).load()
    if isinstance(root, _Ref):
        return root.load(zf, pkl[:-8])
    return LazyDict(root, zf, pkl[:-8])


class LeanSafetensors:  # plain reads per tensor, no mmap
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = {k: v for k, v in json.loads(self.f.read(n)).items() if k != "__metadata__"}
        self.base = 8 + n

    def keys(self):
        return list(self.h)

    def get_tensor(self, k):
        m = self.h[k]
        assert m["dtype"] == "F32", m["dtype"]
        b, e = m["data_offsets"]
        self.f.seek(self.base + b)
        return torch.frombuffer(bytearray(self.f.read(e - b)), dtype=torch.float32).reshape(m["shape"])


def assign_checked(module, sd):
    module.load_state_dict(sd, strict=True, assign=True)
    left = [n for n, b in list(module.named_parameters()) + list(module.named_buffers()) if b.is_meta]
    assert not left, left[:10]
    return module.eval()


cfg = OmegaConf.load(os.path.join(BUNDLE, "config.yaml"))
GPT_PATH = os.path.join(BUNDLE, "german", "gpt.pth")
RESUME = os.environ.get("INDEXTTS_ORACLE_RESUME") == "1" and os.path.exists(os.path.join(OUT, "partial_s2mel.pt"))

def finish(vc_target):  # BigVGAN + write the oracle files
    global T, META
    # bigvgan/utils.py imports matplotlib only for plotting helpers -> stub it.
    _mpl = types.ModuleType("matplotlib")
    _mpl.use = lambda *a, **k: None
    _mpl.pylab = types.ModuleType("matplotlib.pylab")
    sys.modules.setdefault("matplotlib", _mpl)
    sys.modules.setdefault("matplotlib.pylab", _mpl.pylab)
    from indextts.s2mel.modules.bigvgan import bigvgan
    BV =os.path.join(BUNDLE, "bigvgan_v2_22khz_80band_256x")
    h = bigvgan.load_hparams_from_json(os.path.join(BV, "config.json"))
    with torch.device("meta"):
        voc = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    _g = load_pth(os.path.join(BV, "bigvgan_generator.pt"))["generator"]
    assign_checked(voc, {k: _g[k] for k in voc.state_dict()})
    voc.remove_weight_norm(); voc.eval()
    wav = voc(vc_target.float()).squeeze().unsqueeze(0)
    put("wav_float", wav[0])
    wav16 = torch.clamp(32767 * wav, -32767.0, 32767.0).type(torch.int16)
    put("wav_int16", wav16[0].to(torch.int64))
    with wave.open(os.path.join(OUT, "oracle.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
        w.writeframes(wav16[0].numpy().tobytes())
    META["wav_samples"] = int(wav.shape[-1])
    log(f"BigVGAN: {wav.shape[-1]} samples ({wav.shape[-1] / 22050:.2f} s)")

    save_file(T, os.path.join(OUT, "oracle.safetensors"))
    with open(os.path.join(OUT, "oracle.json"), "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=1)
    log("oracle written")


if RESUME:
    _p = torch.load(os.path.join(OUT, "partial_s2mel.pt"), weights_only=False)
    T.update(_p["T"]); META.update(_p["META"])
    log("resumed from partial_s2mel.pt")
    finish(T["vc_target"].unsqueeze(0))
    sys.exit(0)


# ------------------------------------------------------------------------------------------------ text frontend --
spec = importlib.util.spec_from_file_location("german_text", os.path.join(BUNDLE, "german", "german_text.py"))
german_text = importlib.util.module_from_spec(spec)
spec.loader.exec_module(german_text)
from indextts.utils.tokenizer import get_encoding, lang_to_token, LANGUAGE_DICT

enc = get_encoding(name="multilingual_zh_ja_yue_char_del", num_languages=99, model_dir=BUNDLE)
CHAR_REP = {"：": ",", "；": ",", ";": ",", "，": ",", "。": ".", "！": "!", "？": "?", "\n": " ", "·": "-", "、": ",",
            "...": "…", ",,,": "…", "，，，": "…", "……": "…", "“": "'", "”": "'", '"': "'", "‘": "'", "’": "'",
            "（": "'", "）": "'", "(": "'", ")": "'", "《": "'", "》": "'", "【": "'", "】": "'", "[": "'", "]": "'",
            "—": "-", "～": "-", "~": "-", "「": "'", "」": "'", ":": ","}  # TextNormalizer.char_rep_map (front.py)
CLEAN = re.compile("|".join(re.escape(p) for p in CHAR_REP))
PRON = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')
PROTECT = re.compile(r'<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>')
TEXT_POS_CAPACITY = cfg.gpt.max_text_tokens + 2


def is_kana(s):
    return bool(re.fullmatch(r'^[\u3040-\u309F]+$', s) or re.fullmatch(r'^[\u30A0-\u30FF]+$', s))


def apply_pron(text):  # infer_v2_5.apply_pronunciation_annotations
    def rep(m):
        word, pron = m.group(1), m.group(2).upper()
        if is_kana(pron):
            return f' {pron} '
        tok = 'SPECIAL_TOKEN_2' if re.search(r"[\u4e00-\u9fff]", word) else 'SPECIAL_TOKEN_1'
        return f'<|{tok}|>{pron}<|{tok}|>'
    return PRON.sub(rep, text)


def tok_len(s):
    return len(enc.encode(s, allowed_special='all'))


def split_text_by_tokens(text, max_tokens, lang_prefix):  # IndexTTS2.split_text_by_tokens
    budget = max(1, min(max_tokens, TEXT_POS_CAPACITY - 2) - tok_len(lang_prefix))
    if tok_len(text) <= budget:
        return [text]
    pieces, pos = [], 0
    for m in PROTECT.finditer(text):
        if m.start() > pos:
            pieces.append((text[pos:m.start()], False))
        pieces.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    chunks = []
    for piece, atomic in pieces:
        if atomic:
            chunks.append(piece)
            continue
        for part in re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', piece):
            if not part:
                continue
            if tok_len(part) <= budget:
                chunks.append(part)
                continue
            cur = ""
            for ch in part:
                if cur and tok_len(cur + ch) > budget:
                    chunks.append(cur)
                    cur = ch
                else:
                    cur += ch
            if cur:
                chunks.append(cur)
    segs, cur = [], ""
    for c in chunks:
        if cur and tok_len(cur + c) > budget:
            segs.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        segs.append(cur)
    return segs or [text]


def frontend(raw, lang="de", max_tokens=120):
    prepared = german_text.prepare_german(raw)          # indextts_de.py
    text = CLEAN.sub(lambda m: CHAR_REP[m.group()], prepared)  # infer_generator (normalization off, no lower for de)
    text = apply_pron(text)
    text = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)
    prefix = f'<|{lang.lower()}|> '
    segs = split_text_by_tokens(text, max_tokens, prefix)
    ids = [enc.encode(prefix + s, allowed_special='all') + [1] for s in segs]
    return {"input": raw, "prepared": prepared, "final": text, "segments": segs, "ids": ids}


CASES = [TEXT,
         "Am 3.10.2026 um 14:30 Uhr kostet das 12,50 € – ca. 15 % mehr als 1999.",
         "Größe: 1.234 km, z. B. Straße; „Anführung“ (Klammer) [eckig] … ja?! Nr. 7 und Dr. Müller.",
         "ÄÖÜ äöü ß — Gedankenstrich ~ Tilde; Zahlen 0 1 21 101 1000000 3,1415.",
         "Ein <Wort|W ER T> mit Aussprache und <|special_token_3|> Token.",
         "  Doppelte  Leerzeichen\tund Tabs\nund Zeilen... Ende,,, wirklich。",
         "Er rief: \"Halt!\" Dann - Stille. 8:05 Uhr, 23.59 Uhr, 5 kg, 3 m², 100 km/h, $5, 7 GB."]
META["text_cases"] = [frontend(c) for c in CASES]
META["lang_token"] = lang_to_token("de")
META["n_vocab_ranks"] = enc.n_vocab
META["special_tokens"] = {k: v for k, v in enc._special_tokens.items()}
main = META["text_cases"][0]
assert len(main["segments"]) == 1, main["segments"]
text_ids = main["ids"][0]
put("text_ids", torch.tensor(text_ids))
log(f"text: {main['final']!r} -> {len(text_ids)} ids")

# === STAGE: front ===
# ------------------------------------------------------------------------------------------------------- audio --
audio_np, sr = librosa.load(VOICE, sr=None, mono=True)       # _load_and_cut_audio(spk, 15)
audio = torch.tensor(audio_np).unsqueeze(0)[:, :int(15 * sr)]
audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
emo_np, _ = librosa.load(VOICE, sr=16000, mono=True)          # _load_and_cut_audio(emo, 15, sr=16000): soxr_hq
emo_16k = torch.tensor(emo_np).unsqueeze(0)[:, :int(15 * 16000)]
META["voice_rate"] = sr
put("audio", audio[0]); put("audio_22k", audio_22k[0]); put("audio_16k", audio_16k[0]); put("emo_16k", emo_16k[0])
from indextts.s2mel.modules.audio import mel_spectrogram
ref_mel = mel_spectrogram(audio_22k.float(), n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                          win_size=1024, fmin=0, fmax=None, center=False)
put("ref_mel", ref_mel[0])
feat = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
feat = feat - feat.mean(dim=0, keepdim=True)
put("campplus_fbank", feat)
log(f"audio {audio.shape[1]} @ {sr} Hz, ref_mel {tuple(ref_mel.shape)}, fbank {tuple(feat.shape)}")

# ------------------------------------------------------------------------------------------------------ CAM++ --
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
campplus = CAMPPlus(feat_dim=80, embedding_size=192)
campplus.load_state_dict(load_pth(os.path.join(BUNDLE, "campplus", "campplus_cn_common.bin")))
campplus.eval()
head = campplus.head(feat.unsqueeze(0).permute(0, 2, 1))
put("campplus_head", head[0])
style = campplus(feat.unsqueeze(0))
put("style", style[0])
del campplus; gc.collect()
log("CAM++ done")
if os.environ.get("INDEXTTS_ORACLE_LIGHT") == "1":  # text + audio front end + CAM++ only (no large weights)
    from transformers import SeamlessM4TFeatureExtractor
    _ex = SeamlessM4TFeatureExtractor.from_pretrained(os.path.join(BUNDLE, "w2v-bert-2.0"), local_files_only=True)
    for _tag, _wave in (("spk", audio_16k), ("emo", emo_16k)):
        _inp = _ex(_wave, sampling_rate=16000, return_tensors="pt")
        put(f"{_tag}_w2v_input", _inp["input_features"][0]); put(f"{_tag}_w2v_mask", _inp["attention_mask"][0])
    save_file(T, os.path.join(OUT, "oracle.safetensors"))
    with open(os.path.join(OUT, "oracle.json"), "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=1)
    log("light oracle written")
    sys.exit(0)

# === STAGE: w2v ===
# --------------------------------------------------------------------------------------------------- w2v-BERT --
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (Wav2Vec2BertEncoderLayer,
                                                                      Wav2Vec2BertFeatureProjection)
W2V = os.path.join(BUNDLE, "w2v-bert-2.0")
extractor = SeamlessM4TFeatureExtractor.from_pretrained(W2V, local_files_only=True)
w2v_cfg = Wav2Vec2BertConfig.from_pretrained(W2V, local_files_only=True)
stats = load_pth(os.path.join(BUNDLE, cfg.w2v_stat))
sem_mean, sem_std = stats["mean"], torch.sqrt(stats["var"])
ST = LeanSafetensors(os.path.join(W2V, "model.safetensors"))


def load_module(module_fn, prefix):
    with torch.device("meta"):
        module = module_fn()
    return assign_checked(module, {k[len(prefix):]: ST.get_tensor(k) for k in ST.keys() if k.startswith(prefix)})


def get_emb(wave16k, tag):  # IndexTTS2.get_emb == hidden_states[17] of Wav2Vec2BertModel, normalised
    inputs = extractor(wave16k, sampling_rate=16000, return_tensors="pt")
    x, mask = inputs["input_features"], inputs["attention_mask"]
    put(f"{tag}_w2v_input", x[0]); put(f"{tag}_w2v_mask", mask[0])
    proj = load_module(lambda: Wav2Vec2BertFeatureProjection(w2v_cfg), "feature_projection.")
    h, _ = proj(x)
    put(f"{tag}_w2v_proj", h[0])
    conv_mask = mask
    h = h.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
    ext = (1.0 - mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
    ext = ext.expand(ext.shape[0], 1, ext.shape[-1], ext.shape[-1])
    for i in range(17):
        layer = load_module(lambda: Wav2Vec2BertEncoderLayer(w2v_cfg), f"encoder.layers.{i}.")
        h = layer(h, attention_mask=ext, relative_position_embeddings=None, conv_attention_mask=conv_mask)[0]
        if i in (0, 8):
            put(f"{tag}_w2v_layer{i}", h[0])
        del layer
    gc.collect()
    put(f"{tag}_w2v_hidden17", h[0])
    emb = (h - sem_mean) / sem_std
    put(f"{tag}_cond_emb", emb[0])
    return emb


spk_cond_emb = get_emb(audio_16k, "spk")
emo_cond_emb = get_emb(emo_16k, "emo")
log(f"w2v-BERT done: spk {tuple(spk_cond_emb.shape)}, emo {tuple(emo_cond_emb.shape)}")

# === STAGE: emo ===
# ----------------------------------------------------------------------------------------- emotion conditioning --
GPT_SD = load_pth(GPT_PATH)["model"]
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler


def sub_sd(prefix):
    return {k[len(prefix):]: GPT_SD[k] for k in GPT_SD if k.startswith(prefix)}


emc = cfg.gpt.emo_condition_module
with torch.device("meta"):
    emo_enc = ConformerEncoder(input_size=1024, output_size=emc.output_size, linear_units=emc.linear_units,
                               attention_heads=emc.attention_heads, num_blocks=emc.num_blocks,
                               input_layer=emc.input_layer)
    emo_perc = PerceiverResampler(1024, dim_context=emc.output_size, ff_mult=emc.perceiver_mult,
                                  heads=emc.attention_heads, num_latents=1)
assign_checked(emo_enc, sub_sd("emo_conditioning_encoder."))
assign_checked(emo_perc, sub_sd("emo_perceiver_encoder."))
emovec_layer = torch.nn.Linear(1024, 1280); emovec_layer.load_state_dict(sub_sd("emovec_layer.")); emovec_layer.eval()
emo_layer = torch.nn.Linear(1280, 1280); emo_layer.load_state_dict(sub_sd("emo_layer.")); emo_layer.eval()
emo_cond_mask_pad = torch.nn.ConstantPad1d((1, 0), True)


def get_emovec(cond, tag):  # UnifiedVoice.get_emovec (lengths = feature dim, as in infer_v2_5)
    lengths = torch.tensor([cond.shape[-1]])
    xs = cond  # get_emo_conditioning(cond.transpose(1,2)) transposes back inside
    T_ = xs.size(1)
    from indextts.utils.common import make_pad_mask
    masks = ~make_pad_mask(lengths, T_).unsqueeze(1)
    e, pos_emb, m = emo_enc.embed(xs, masks)
    put(f"{tag}_emo_embed", e[0])
    for li, layer in enumerate(emo_enc.encoders):
        e, m, _, _ = layer(e, m, pos_emb, m)
        if li == 0:
            put(f"{tag}_emo_layer0", e[0])
    e = emo_enc.after_norm(e)
    put(f"{tag}_emo_encoded", e[0])
    full, full_mask = emo_enc(cond, lengths)  # the original module call, for the check below
    assert torch.equal(full, e)
    conds = emo_perc(e, emo_cond_mask_pad(full_mask.squeeze(1))).squeeze(1)
    put(f"{tag}_emo_perceiver", conds[0])
    v = emo_layer(emovec_layer(conds))
    put(f"{tag}_emovec", v[0])
    return v


emo_vec = get_emovec(emo_cond_emb, "emo")
base_vec = get_emovec(spk_cond_emb, "spk")
merged = base_vec + 1.0 * (emo_vec - base_vec)  # merge_emovec, alpha 1 (no external emotion audio)
put("emovec_merged", merged[0])
del emo_enc, emo_perc; gc.collect()


def normalize_emo_vec(v):  # IndexTTS2.normalize_emo_vec
    bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
    v = [a * b for a, b in zip(v, bias)]
    s = sum(v)
    if s > 0.8:
        v = [a * 0.8 / s for a in v]
    return v


emo_norm = normalize_emo_vec(EMO)
weight_vector = torch.tensor(emo_norm)
emo_matrix = torch.split(load_pth(os.path.join(BUNDLE, cfg.emo_matrix)), list(cfg.emo_num))
spk_matrix = torch.split(load_pth(os.path.join(BUNDLE, cfg.spk_matrix)), list(cfg.emo_num))
index = [int(torch.argmax(F.cosine_similarity(style.float(), m.float(), dim=1))) for m in spk_matrix]
emovec_mat = torch.sum(weight_vector.unsqueeze(1) * torch.cat([m[i].unsqueeze(0) for i, m in zip(index, emo_matrix)], 0),
                       0).unsqueeze(0)
emovec = emovec_mat + (1 - torch.sum(weight_vector)) * merged
META["emo_vector_normalized"] = emo_norm
META["emo_index"] = index
put("emovec_mat", emovec_mat[0]); put("emovec", emovec[0])
log(f"emotion: normalized {emo_norm}, matrix rows {index}")

# === STAGE: gpt ===
# ----------------------------------------------------------------------------------------------------- GPT AR --
# Same math as UnifiedVoice.inference_speech + GPT2InferenceModel + HF GPT2 (eager attention, gelu_new), written out
# so each of the 24 layers can be re-read from gpt.pth for every step (keeps the working set and commit low). The left
# pad row of prepare_gpt_inputs is attention-masked for every query, so it is left out (identical result).
from transformers.generation.logits_process import (RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                                                    TopKLogitsWarper, TopPLogitsWarper)
GPT_SD = load_pth(GPT_PATH)["model"]
G = lambda k: GPT_SD[k]
DIM, HEADS, LAYERS = cfg.gpt.model_dim, cfg.gpt.heads, cfg.gpt.layers
HD = DIM // HEADS
spk_latent = F.linear(style, G("spk_emb_proj.weight"), G("spk_emb_proj.bias"))
put("spk_latent", spk_latent[0])
ids = torch.tensor(text_ids)
text_in = torch.cat([torch.tensor([0]), ids[(ids != 1) & (ids != 0)], torch.tensor([1])])
_te = G("text_embedding.weight")
temb = _te[text_in] + G("text_pos_embedding.emb.weight")[: len(text_in)]
del _te
temb = temb + G("lang_embedding.weight")[lang_to_token("de")]
prefix = torch.cat([spk_latent + emovec, torch.zeros(2, DIM), temb], 0)
put("gpt_prefix_emb", prefix)
MEL = G("mel_embedding.weight"); MPOS = G("mel_pos_embedding.emb.weight")
LNF = (G("gpt.ln_f.weight"), G("gpt.ln_f.bias")); FN = (G("final_norm.weight"), G("final_norm.bias"))
HEAD = (G("mel_head.weight"), G("mel_head.bias"))
NAMES = ["ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight", "attn.c_proj.bias",
         "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"]
KC = [torch.zeros(0, DIM) for _ in range(LAYERS)]
VC = [torch.zeros(0, DIM) for _ in range(LAYERS)]


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gpt_forward(x):
    h = x
    for i in range(LAYERS):
        W = {n: G(f"gpt.h.{i}.{n}") for n in NAMES}
        a = F.layer_norm(h, (DIM,), W["ln_1.weight"], W["ln_1.bias"], 1e-5)
        q, k, v = torch.addmm(W["attn.c_attn.bias"], a, W["attn.c_attn.weight"]).split(DIM, dim=1)
        KC[i] = torch.cat([KC[i], k]); VC[i] = torch.cat([VC[i], v])
        nq, nk = q.shape[0], KC[i].shape[0]
        qh = q.view(nq, HEADS, HD).transpose(0, 1)
        kh = KC[i].view(nk, HEADS, HD).transpose(0, 1)
        vh = VC[i].view(nk, HEADS, HD).transpose(0, 1)
        w = torch.matmul(qh, kh.transpose(-1, -2)) / torch.full([], HD ** 0.5)
        causal = torch.ones(nq, nk, dtype=torch.bool).tril(nk - nq)
        w = torch.where(causal, w, torch.full([], torch.finfo(w.dtype).min))
        o = torch.matmul(torch.softmax(w, dim=-1), vh).transpose(0, 1).reshape(nq, DIM)
        h = h + torch.addmm(W["attn.c_proj.bias"], o, W["attn.c_proj.weight"])
        m = F.layer_norm(h, (DIM,), W["ln_2.weight"], W["ln_2.bias"], 1e-5)
        f = gelu_new(torch.addmm(W["mlp.c_fc.bias"], m, W["mlp.c_fc.weight"]))
        h = h + torch.addmm(W["mlp.c_proj.bias"], f, W["mlp.c_proj.weight"])
        del W
    z = F.layer_norm(F.layer_norm(h[-1:], (DIM,), LNF[0], LNF[1], 1e-5), (DIM,), FN[0], FN[1], 1e-5)
    return F.linear(z, HEAD[0], HEAD[1])


rng = np.random.RandomState(SEED)
WARPERS = [TemperatureLogitsWarper(0.7), TopKLogitsWarper(top_k=30), TopPLogitsWarper(top_p=0.8)]
PENALTY = RepetitionPenaltyLogitsProcessor(10.0)
REC = {"scores": [], "u": [], "tok": [], "kept": []}
stop = cfg.gpt.stop_mel_token
input_ids = [1] * (prefix.shape[0] + 1) + [cfg.gpt.start_mel_token]  # fake ids incl. the masked pad row
t0 = time.time()
logits = gpt_forward(torch.cat([prefix, (MEL[cfg.gpt.start_mel_token] + MPOS[0])[None]], 0))
for step in range(MAXTOK):
    scores = PENALTY(torch.tensor([input_ids]), logits.clone())
    REC["scores"].append(scores[0].clone())
    sc = scores
    for wp in WARPERS:
        sc = wp(torch.tensor([input_ids]), sc)
    probs = torch.softmax(sc[0].float(), dim=-1).double().numpy()
    u = float(rng.random_sample())
    cum = np.cumsum(probs)
    tok = min(int(np.searchsorted(cum, u * cum[-1], side="right")), len(probs) - 1)
    while probs[tok] == 0.0 and tok > 0:
        tok -= 1
    REC["u"].append(u); REC["tok"].append(tok); REC["kept"].append(int((probs > 0).sum()))
    input_ids.append(tok)
    if tok == stop:
        break
    logits = gpt_forward((MEL[tok] + MPOS[step + 2])[None])  # GPT2InferenceModel mel position quirk
    if step % 10 == 0:
        log(f"GPT step {step}")
META["gpt_seconds"] = time.time() - t0
codes = REC["tok"]
code_len = codes.index(stop) if stop in codes else len(codes)
META["codes_raw"] = codes
META["code_len"] = code_len
META["natural_stop"] = stop in codes
put("gpt_scores", torch.stack(REC["scores"]))
META["gpt_u"] = REC["u"]  # doubles, exact in JSON
put("gpt_tokens", torch.tensor(REC["tok"]))
META["gpt_kept"] = REC["kept"]
codes_t = torch.tensor(codes[:code_len]).unsqueeze(0)
put("codes", codes_t[0])
log(f"GPT: {len(codes)} steps ({code_len} codes, stop {stop in codes}) in {META['gpt_seconds']:.1f} s")
del KC, VC, GPT_SD; gc.collect()

# === STAGE: s2mel ===
# --------------------------------------------------------------------------------------- semantic codec / s2mel --
from indextts.codec.models import EnhancedCodec
import indextts.s2mel
# length_regulator imports indextts.s2mel.dac.nn.quantize; the dac package __init__ needs descript-audiotools,
# which is not used here -> register the two packages without running their __init__ files.
for _name in ("indextts.s2mel.dac", "indextts.s2mel.dac.nn"):
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [os.path.join(REF, *_name.split("."))]
    sys.modules[_name] = _pkg
from indextts.s2mel.modules.commons import MyModel

with torch.device("meta"):
    codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
saved = load_pth(os.path.join(BUNDLE, "codec.pth"))["model"]
assign_checked(codec, {k: saved[k] for k in codec.state_dict()})
q = codec.quantizer.vq2emb(codes_t.unsqueeze(0))
put("codec_vq2emb", q[0])
dec = codec.decoder(q)
put("codec_decoder", dec[0])
S_infer = codec.decode(codes_t)
put("S_infer", S_infer[0])
del codec, saved; gc.collect()

with torch.device("meta"):
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=False)
params = load_pth(os.path.join(BUNDLE, cfg.s2mel_checkpoint))["net"]
for key in s2mel.models:
    sub = params[key]
    names = {(k[len("module."):] if k.startswith("module.") else k): k for k in sub}
    msd = s2mel.models[key].state_dict()
    assign_checked(s2mel.models[key], {k: sub[names[k]] for k in msd})  # every module key is in the checkpoint
s2mel.eval()
s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
lr = s2mel.models['length_regulator']
ref_target_lengths = torch.LongTensor([ref_mel.size(2)])
prompt_condition = lr(spk_cond_emb, ylens=ref_target_lengths, n_quantizers=3, f0=None)[0]
put("prompt_condition", prompt_condition[0])
target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * 1.0)])
cond = lr(S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
put("cond", cond[0])
cat_condition = torch.cat([prompt_condition, cond], dim=1)
cfm = s2mel.models['cfm']
torch.manual_seed(SEED)
z = torch.randn([1, cfm.in_channels, cat_condition.size(1)]) * 1.0   # BASECFM.inference
put("cfm_noise", z[0])
t_span = torch.linspace(0, 1, 25 + 1)
ncall = [0]
_est = cfm.estimator.forward


def est_rec(*a, **k):
    out = _est(*a, **k)
    i = ncall[0]
    ncall[0] += 1
    if i == 0:
        put("cfm_call0_x", a[0]); put("cfm_call0_prompt_x", a[1]); put("cfm_call0_t", a[3])
        put("cfm_velocity_call0", out)
    if i == 24:
        put("cfm_call24_x", a[0]); put("cfm_velocity_call24", out)
    return out


cfm.estimator.forward = est_rec
t0 = time.time()
vc = cfm.solve_euler(z.clone(), torch.LongTensor([cat_condition.size(1)]), ref_mel, cat_condition, style, None,
                     t_span, inference_cfg_rate=0.7)
META["cfm_seconds"] = time.time() - t0
put("cfm_out", vc[0])
vc_target = vc[:, :, ref_mel.size(-1):]
put("vc_target", vc_target[0])
log(f"s2mel: prompt {ref_mel.size(-1)} + target {target_lengths.item()} frames, CFM {META['cfm_seconds']:.1f} s")
del s2mel, params; gc.collect()
torch.save({"T": T, "META": META}, os.path.join(OUT, "partial_s2mel.pt"))  # resume point (INDEXTTS_ORACLE_RESUME=1)

# === STAGE: vocoder ===
finish(vc_target)
