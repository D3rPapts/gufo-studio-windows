#!/usr/bin/env python3
"""Converts a RIFE 4.x IFNet checkpoint (rife47.pth / rife49.pth, the files
ComfyUI-Frame-Interpolation downloads) into an FP32 safetensors file that
`gufo video-interpolate` loads.

Standard library only, and the pickle is read by a restricted unpickler that
knows the handful of globals a plain state_dict uses, so no code from the
checkpoint is executed and PyTorch is not needed:

    python -I convert_rife_pth.py rife49.pth rife49.safetensors

A "module." prefix (DataParallel checkpoints) is stripped.
"""

import collections
import json
import pickle
import struct
import sys
import zipfile

_DTYPES = {
    "FloatStorage": ("F32", 4, "f"),
    "HalfStorage": ("F16", 2, "e"),
}


class _Storage:
    def __init__(self, kind, key):
        self.kind = kind
        self.key = key


class _StorageType:
    def __init__(self, name):
        self.name = name


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, *unused):
    return _Tensor(storage, offset, size, stride)


def _rebuild_parameter(data, *unused):
    return data


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 3 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        kind = pid[1]
        if not isinstance(kind, _StorageType):
            raise pickle.UnpicklingError("unexpected storage type")
        return _Storage(kind.name, str(pid[2]))


def _contiguous_values(tensor, raw):
    _, width, fmt = _DTYPES[tensor.storage.kind]
    count = 1
    for d in tensor.size:
        count *= d
    expected = []
    acc = 1
    for d in reversed(tensor.size):
        expected.insert(0, acc)
        acc *= d
    if tensor.size and tuple(expected) != tensor.stride:
        raise ValueError("non-contiguous tensor")
    begin = tensor.offset * width
    data = raw[begin:begin + count * width]
    if len(data) != count * width:
        raise ValueError("storage too short")
    if fmt == "f":
        return data
    values = struct.unpack(f"<{count}e", data)
    return struct.pack(f"<{count}f", *values)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    with zipfile.ZipFile(argv[1]) as archive:
        names = archive.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pkl) != 1:
            raise SystemExit("not a torch zip checkpoint")
        root = pkl[0][: -len("data.pkl")]
        state = _Unpickler(archive.open(pkl[0])).load()
        if not isinstance(state, dict):
            raise SystemExit("checkpoint is not a state_dict")
        tensors = []
        for key, value in state.items():
            if not isinstance(value, _Tensor):
                continue
            name = key[len("module."):] if key.startswith("module.") else key
            raw = archive.read(f"{root}data/{value.storage.key}")
            tensors.append((name, list(value.size), _contiguous_values(value, raw)))
    header = {}
    offset = 0
    for name, size, data in tensors:
        header[name] = {"dtype": "F32", "shape": size,
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    header["__metadata__"] = {"source": argv[1].replace("\\", "/").split("/")[-1],
                              "format": "rife-ifnet"}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(argv[2], "wb") as out:
        out.write(struct.pack("<Q", len(text)))
        out.write(text)
        for _, _, data in tensors:
            out.write(data)
    print(f"{len(tensors)} tensors, {offset / 1e6:.1f} MB -> {argv[2]}")
    for name, size, _ in tensors:
        print(name, size)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
