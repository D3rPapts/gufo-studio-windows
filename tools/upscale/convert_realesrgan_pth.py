#!/usr/bin/env python3
"""Converts a Real-ESRGAN SRVGGNetCompact checkpoint (realesr-animevideov3.pth,
realesr-general-x4v3.pth, realesr-general-wdn-x4v3.pth from the Real-ESRGAN
v0.2.5.0 release) into an FP32 safetensors file that `gufo video-upscale`
loads.

Standard library only; the pickle is read by the restricted unpickler of
tools/rife/convert_rife_pth.py (no code from the checkpoint is executed):

    python -I convert_realesrgan_pth.py realesr-animevideov3.pth \
        realesr-animevideov3.safetensors

The state_dict may be nested under "params_ema" or "params" (BasicSR). The
metadata records the architecture: number of body convolutions, features and
the upscale factor (from the last convolution's 3 * scale^2 rows).
"""

import importlib.util
import json
import pathlib
import struct
import sys
import zipfile

_RIFE = pathlib.Path(__file__).resolve().parent.parent / "rife" / "convert_rife_pth.py"
_spec = importlib.util.spec_from_file_location("convert_rife_pth", _RIFE)
_rife = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rife)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    with zipfile.ZipFile(argv[1]) as archive:
        pkl = [n for n in archive.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pkl) != 1:
            raise SystemExit("not a torch zip checkpoint")
        root = pkl[0][: -len("data.pkl")]
        state = _rife._Unpickler(archive.open(pkl[0])).load()
        for key in ("params_ema", "params"):
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state = state[key]
                break
        if not isinstance(state, dict):
            raise SystemExit("checkpoint is not a state_dict")
        tensors = []
        for key, value in state.items():
            if not isinstance(value, _rife._Tensor):
                continue
            name = key[len("module."):] if key.startswith("module.") else key
            raw = archive.read(f"{root}data/{value.storage.key}")
            tensors.append((name, list(value.size), _rife._contiguous_values(value, raw)))
    shapes = {name: size for name, size, _ in tensors}
    convs = sorted(int(n.split(".")[1]) for n, s in shapes.items()
                   if n.startswith("body.") and n.endswith(".weight") and len(s) == 4)
    if not convs or convs != list(range(0, 2 * len(convs) - 1, 2)):
        raise SystemExit("not an SRVGGNetCompact checkpoint (body.N convolutions)")
    first, last = shapes[f"body.{convs[0]}.weight"], shapes[f"body.{convs[-1]}.weight"]
    scale = int(round((last[0] / 3) ** 0.5))
    if first[1] != 3 or scale * scale * 3 != last[0]:
        raise SystemExit("unexpected SRVGGNetCompact shapes")
    for i in convs[:-1]:
        if shapes.get(f"body.{i + 1}.weight") != [first[0]]:
            raise SystemExit("expected a PReLU after every convolution but the last")
    header = {}
    offset = 0
    for name, size, data in tensors:
        header[name] = {"dtype": "F32", "shape": size,
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    header["__metadata__"] = {
        "source": argv[1].replace("\\", "/").split("/")[-1],
        "format": "realesrgan-srvgg",
        "num_conv": str(len(convs) - 2),
        "num_feat": str(first[0]),
        "upscale": str(scale),
    }
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(argv[2], "wb") as out:
        out.write(struct.pack("<Q", len(text)))
        out.write(text)
        for _, _, data in tensors:
            out.write(data)
    print(f"{len(tensors)} tensors, {offset / 1e6:.1f} MB, {len(convs) - 2} body convs, "
          f"{first[0]} features, x{scale} -> {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
