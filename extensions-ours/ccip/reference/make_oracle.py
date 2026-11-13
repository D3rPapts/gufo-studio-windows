# Oracle data for the native CCIP / face detector port: runs onnxruntime (CPU) on fixed inputs and saves the
# decoded RGB pixels, the model inputs, intermediate activations and outputs as safetensors files that the C++
# oracle test reads. Preprocessing is copied verbatim from ccip_ref.py (deepghs imgutils semantics).
# Written with AI assistance (Claude Code).
#
#   python -I make_oracle.py <model_feat.onnx> <face model.onnx> <out_dir> --ccip a.png b.png ... --faces k1.png k2.png
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from safetensors.numpy import save_file


def ccip_input(img):
    img = img.convert("RGB").resize((384, 384), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)[None]


def letterbox(img, size=640):
    img = img.convert("RGB")
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, dx, dy


def nms(boxes, scores, iou=0.7):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        u = area(boxes[i:i + 1]) + area(boxes[order[1:]]) - inter
        order = order[1:][inter / u < iou]
    return keep


def faces_from_raw(raw, img, r, dx, dy, conf=0.25):
    out = raw[0].T
    out = out[out[:, 4] >= conf]
    if not len(out):
        return []
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    b[:, [0, 2]] = (b[:, [0, 2]] - dx) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dy) / r
    k = nms(b, out[:, 4])
    W, H = img.size
    res = []
    for i in k:
        x0, y0, x1, y1 = b[i]
        res.append(([max(0, x0), max(0, y0), min(W, x1), min(H, y1)], float(out[i, 4])))
    return res


def with_outputs(path, names):
    m = onnx.load(path)
    have = {o.name for o in m.graph.output}
    for n in names:
        if n not in have:
            m.graph.output.append(onnx.ValueInfoProto(name=n))
    return m.SerializeToString()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat")
    ap.add_argument("face")
    ap.add_argument("out")
    ap.add_argument("--ccip", nargs="+", default=[])
    ap.add_argument("--faces", nargs="+", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # plain fp32 reference

    arch_depths = [3, 12, 18, 3]
    taps = {"stem": "/backbone/downsample_layers.0/post_norm/Mul_output_0"}
    for s, d in enumerate(arch_depths):
        taps[f"stage{s}"] = f"/backbone/stages.{s}/stages.{s}.{d - 1}/Add_1_output_0"
    taps["stage2_block0"] = "/backbone/stages.2/stages.2.0/Add_1_output_0"
    feat = ort.InferenceSession(with_outputs(a.feat, list(taps.values())), so, providers=["CPUExecutionProvider"])
    manifest = {"ccip": [], "faces": []}
    for p in a.ccip:
        name = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p)
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        x = ccip_input(img)
        outs = feat.run(["output"] + list(taps.values()), {"input": x})
        t = {"rgb": rgb, "input": x[0], "embedding": outs[0][0]}
        for (k, _), v in zip(taps.items(), outs[1:]):
            t[k] = v[0]  # NHWC
        fn = f"ccip_{len(manifest['ccip'])}.safetensors"
        save_file({k: np.ascontiguousarray(v) for k, v in t.items()}, os.path.join(a.out, fn))
        manifest["ccip"].append({"file": fn, "source": p, "size": [int(rgb.shape[1]), int(rgb.shape[0])]})
        print("ccip", name, rgb.shape, "norm", float(np.linalg.norm(outs[0][0])))

    face = ort.InferenceSession(a.face, so, providers=["CPUExecutionProvider"])
    for p in a.faces:
        img = Image.open(p)
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        x, r, dx, dy = letterbox(img)
        raw = face.run(None, {face.get_inputs()[0].name: x})[0]
        res = faces_from_raw(raw, img, r, dx, dy)
        boxes = np.array([b + [s] for b, s in res], dtype=np.float32).reshape(-1, 5)
        fn = f"faces_{len(manifest['faces'])}.safetensors"
        save_file({"rgb": rgb, "input": np.ascontiguousarray(x[0]), "raw": np.ascontiguousarray(raw[0]),
                   "boxes": boxes}, os.path.join(a.out, fn))
        manifest["faces"].append({"file": fn, "source": p, "size": [int(rgb.shape[1]), int(rgb.shape[0])],
                                  "letterbox": {"r": r, "dx": dx, "dy": dy}})
        print("faces", os.path.basename(p), rgb.shape, "raw", raw.shape, "boxes", len(res))
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
