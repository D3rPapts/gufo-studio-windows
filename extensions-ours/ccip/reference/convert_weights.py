# One-time converter: deepghs CCIP (CAFormer, model_feat.onnx) and anime face detector (YOLOv8n, model.onnx)
# ONNX initializers -> safetensors (F32, readable names) + JSON with the graph-derived architecture.
# Every architecture value is read off the ONNX graph (node attributes, constants, weight shapes); the script
# asserts the block patterns it relies on and fails on anything unexpected.
# Written with AI assistance (Claude Code).
#
# Usage (venv with onnx, numpy, safetensors; run with python -I):
#   python -I convert_weights.py ccip  <model_feat.onnx> <out_dir>
#   python -I convert_weights.py yolo  <model.onnx>      <out_dir>
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import save_file


class Graph:
    def __init__(self, path):
        self.model = onnx.load(path)
        g = self.model.graph
        self.inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        self.consts = {}
        self.nodes = []
        for n in g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.consts[n.output[0]] = numpy_helper.to_array(a.t)
                continue
            self.nodes.append(n)
        self.by_name = {n.name: n for n in self.nodes}

    def value(self, name):
        if name in self.inits:
            return self.inits[name]
        if name in self.consts:
            return self.consts[name]
        return None

    def attrs(self, node):
        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    def node(self, name):
        if name not in self.by_name:
            raise SystemExit(f"node not found: {name}")
        return self.by_name[name]


def scalar(a):
    a = np.asarray(a)
    assert a.size == 1, a.shape
    return float(a.reshape(-1)[0])


# ----------------------------------------------------------------------------- CCIP / CAFormer
def convert_ccip(path, out_dir):
    g = Graph(path)
    tensors = OrderedDict()

    def put(name, arr):
        assert name not in tensors, name
        tensors[name] = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))

    def ln(node_prefix, weight_name):
        # LayerNorm over the channel axis written as ReduceMean/Sub/Pow/ReduceMean/Add/Sqrt/Div/Mul (weight only).
        rm = g.node(node_prefix + "/ReduceMean")
        assert g.attrs(rm)["axes"] == [-1]
        add = g.node(node_prefix + "/Add")
        eps = scalar(g.value(add.input[1]))
        mul = g.node(node_prefix + "/Mul")
        assert mul.input[1] in g.inits
        put(weight_name, g.inits[mul.input[1]])
        assert f"{node_prefix}/Sub" in g.by_name and f"{node_prefix}/Sqrt" in g.by_name
        return eps

    def matmul_weight(node_name, out_name):
        n = g.node(node_name)
        assert n.op_type == "MatMul" and n.input[1] in g.inits, node_name
        w = g.inits[n.input[1]]  # [in, out]
        put(out_name, w)
        return w.shape

    def starrelu(prefix, out_prefix):
        # relu(x)^2 * scale + bias, scale/bias scalars
        assert g.node(prefix + "/relu/Relu").op_type == "Relu"
        pw = g.node(prefix + "/Pow")
        assert scalar(g.value(pw.input[1])) == 2.0
        mul = g.node(prefix + "/Mul")
        add = g.node(prefix + "/Add")
        s = [x for x in mul.input if x in g.inits][0]
        b = [x for x in add.input if x in g.inits][0]
        return {"scale": scalar(g.inits[s]), "bias": scalar(g.inits[b])}

    def conv(node_name, out_prefix, expect_bias):
        n = g.node(node_name)
        assert n.op_type == "Conv"
        a = g.attrs(n)
        put(out_prefix + ".weight", g.inits[n.input[1]])
        has_bias = len(n.input) > 2 and n.input[2] != ""
        assert has_bias == expect_bias, node_name
        if has_bias:
            put(out_prefix + ".bias", g.inits[n.input[2]])
        assert a.get("dilations", [1, 1]) == [1, 1]
        p = a["pads"]
        assert p[0] == p[2] and p[1] == p[3] and p[0] == p[1], p
        k = a["kernel_shape"]
        assert k[0] == k[1]
        s = a["strides"]
        assert s[0] == s[1]
        w = g.inits[n.input[1]]
        return {"in": int(w.shape[1] * a.get("group", 1)), "out": int(w.shape[0]), "kernel": int(k[0]),
                "stride": int(s[0]), "pad": int(p[0]), "groups": int(a.get("group", 1))}

    arch = OrderedDict(model="ccip-caformer-24-randaug-pruned", source=os.path.basename(path))
    arch["input"] = {"size": 384, "mean": 0.5, "std": 0.5}
    # Stem / downsample layers
    down = []
    eps_all = set()
    for i in range(4):
        p = f"/backbone/downsample_layers.{i}"
        d = OrderedDict()
        if i == 0:
            d["conv"] = conv(p + "/conv/Conv", "downsample.0.conv", True)
            eps_all.add(ln(p + "/post_norm", "downsample.0.post_norm.weight"))
            d["pre_norm"] = False
            d["post_norm"] = True
        else:
            eps_all.add(ln(p + "/pre_norm", f"downsample.{i}.pre_norm.weight"))
            d["conv"] = conv(p + "/conv/Conv", f"downsample.{i}.conv", True)
            d["pre_norm"] = True
            d["post_norm"] = False
        down.append(d)
    arch["downsample"] = down
    # Stages
    stages = []
    for s in range(4):
        blocks = []
        b = 0
        while f"/backbone/stages.{s}/stages.{s}.{b}/norm1/ReduceMean" in g.by_name:
            p = f"/backbone/stages.{s}/stages.{s}.{b}"
            o = f"stages.{s}.{b}"
            blk = OrderedDict()
            eps_all.add(ln(p + "/norm1", o + ".norm1.weight"))
            eps_all.add(ln(p + "/norm2", o + ".norm2.weight"))
            for rs in ("res_scale1", "res_scale2"):
                nn = p + f"/{rs}/Mul"
                if nn in g.by_name:
                    put(f"{o}.{rs}", g.inits[g.node(nn).input[1]])
                    blk[rs] = True
                else:
                    blk[rs] = False
            for ls in ("layer_scale1", "layer_scale2"):
                assert p + f"/{ls}/Mul" not in g.by_name, "layer scale not implemented"
            if p + "/token_mixer/qkv/MatMul" in g.by_name:
                blk["mixer"] = "attention"
                wshape = matmul_weight(p + "/token_mixer/qkv/MatMul", o + ".attn.qkv.weight")
                rs = g.node(p + "/token_mixer/Reshape")
                shape = [int(v) for v in g.value(rs.input[1]).ravel()]
                # qkv(x).reshape(B, N, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
                assert len(shape) == 5 and shape[2] == 3, shape
                assert g.attrs(g.node(p + "/token_mixer/Transpose"))["perm"] == [2, 0, 3, 1, 4]
                heads, head_dim = shape[3], shape[4]
                assert heads * head_dim * 3 == wshape[1]
                mul = g.node(p + "/token_mixer/Mul_1")
                blk["heads"] = heads
                blk["head_dim"] = head_dim
                blk["attn_scale"] = scalar(g.value(mul.input[1]))
                matmul_weight(p + "/token_mixer/proj/MatMul", o + ".attn.proj.weight")
                assert p + "/token_mixer/proj/Add" not in g.by_name, "proj bias not implemented"
            elif p + "/token_mixer/pwconv1/MatMul" in g.by_name:
                blk["mixer"] = "sepconv"
                matmul_weight(p + "/token_mixer/pwconv1/MatMul", o + ".sepconv.pwconv1.weight")
                blk["act1"] = starrelu(p + "/token_mixer/act1", o)
                c = conv(p + "/token_mixer/dwconv/Conv", o + ".sepconv.dwconv", False)
                assert c["groups"] == c["out"] == c["in"] and c["stride"] == 1
                blk["dw_kernel"] = c["kernel"]
                blk["dw_pad"] = c["pad"]
                matmul_weight(p + "/token_mixer/pwconv2/MatMul", o + ".sepconv.pwconv2.weight")
            else:
                raise SystemExit("unknown token mixer in " + p)
            w1 = matmul_weight(p + "/mlp/fc1/MatMul", o + ".mlp.fc1.weight")
            blk["mlp_hidden"] = int(w1[1])
            blk["mlp_act"] = starrelu(p + "/mlp/act", o)
            matmul_weight(p + "/mlp/fc2/MatMul", o + ".mlp.fc2.weight")
            assert p + "/mlp/fc1/Add" not in g.by_name and p + "/mlp/fc2/Add" not in g.by_name
            blocks.append(blk)
            b += 1
        stages.append(OrderedDict(dim=down[s]["conv"]["out"], depth=len(blocks), blocks=blocks))
    arch["stages"] = stages
    assert len(eps_all) == 1, eps_all
    arch["norm_eps"] = eps_all.pop()
    # Attention pool head (CLIP AttentionPool2d)
    ap = "/backbone/attnpool"
    pos = g.value(g.node(ap + "/Add").input[1])
    put("attnpool.positional_embedding", pos.reshape(pos.shape[0], pos.shape[-1]))
    q = g.node(ap + "/MatMul")
    assert q.input[0] == ap + "/Slice_1_output_0", q.input
    k = g.node(ap + "/MatMul_1")
    v = g.node(ap + "/MatMul_2")
    assert k.input[0] == ap + "/Add_output_0" and v.input[0] == ap + "/Add_output_0"
    for nm, mm, ad in (("q", q, "Add_1"), ("k", k, "Add_2"), ("v", v, "Add_3")):
        put(f"attnpool.{nm}_proj.weight", g.inits[mm.input[1]])
        a = g.node(ap + "/" + ad)
        put(f"attnpool.{nm}_proj.bias", g.inits[[x for x in a.input if x in g.inits][0]])
    heads = int(scalar(g.value(g.node(ap + "/Div").input[1])))
    qscale = scalar(g.value(g.node(ap + "/Div_1").input[1]))
    gemm = g.node(ap + "/Gemm")
    ga = g.attrs(gemm)
    assert ga.get("transB", 0) == 1 and ga.get("alpha", 1.0) == 1.0 and ga.get("beta", 1.0) == 1.0
    put("attnpool.c_proj.weight", g.inits[gemm.input[1]])  # [out, in]
    put("attnpool.c_proj.bias", g.inits[gemm.input[2]])
    arch["attnpool"] = OrderedDict(tokens=int(pos.shape[0]), dim=int(pos.shape[-1]), heads=heads,
                                   q_divisor=qscale, out_dim=int(g.inits[gemm.input[1]].shape[0]))
    arch["layout"] = ("linear weights are stored [in, out] (ONNX MatMul layout) except attnpool.c_proj.weight "
                      "[out, in]; convs are [out, in/groups, k, k]; norms are LayerNorm over channels without bias")
    # Every initializer must have been used (named check by value identity is too slow; compare counts/bytes).
    used = sum(t.size for t in tensors.values())
    total = sum(np.asarray(a).size for n, a in g.inits.items() if np.asarray(a).size > 8)
    big_used = sum(t.size for t in tensors.values() if t.size > 8)
    if big_used != total:
        raise SystemExit(f"unused large initializers: used {big_used} of {total} elements")
    write(out_dir, "ccip_caformer", tensors, arch)
    print("ccip tensors", len(tensors), "elements", used)


# ----------------------------------------------------------------------------- YOLOv8 face detector
def convert_yolo(path, out_dir):
    g = Graph(path)
    tensors = OrderedDict()
    convs = OrderedDict()
    for n in g.nodes:
        if n.op_type != "Conv":
            continue
        m = re.match(r"^/model\.(.+)/Conv$", n.name)
        assert m, n.name
        key = m.group(1).replace("/", ".")
        # "/model.22/cv2.0/cv2.0.0/conv/Conv" -> "22.cv2.0.cv2.0.0.conv"; shorten the doubled path segment
        key = re.sub(r"(cv\d)\.(\d)\.cv\d\.\d\.(\d)", r"\1.\2.\3", key)
        a = g.attrs(n)
        w = g.inits[n.input[1]]
        p = a["pads"]
        assert p[0] == p[2] == p[1] == p[3] and a["strides"][0] == a["strides"][1]
        assert a.get("group", 1) == 1 and a.get("dilations", [1, 1]) == [1, 1]
        has_bias = len(n.input) > 2
        # SiLU follows when the next node is Sigmoid + Mul on this output.
        out = n.output[0]
        cons = [c for c in g.nodes if out in c.input]
        silu = len(cons) == 2 and sorted(c.op_type for c in cons) == ["Mul", "Sigmoid"]
        convs[key] = OrderedDict(out=int(w.shape[0]), inp=int(w.shape[1]), kernel=int(w.shape[2]),
                                 stride=int(a["strides"][0]), pad=int(p[0]), bias=has_bias, silu=silu)
        tensors["model." + key + ".weight"] = np.ascontiguousarray(w.astype(np.float32))
        if has_bias:
            tensors["model." + key + ".bias"] = np.ascontiguousarray(g.inits[n.input[2]].astype(np.float32))
    pools = [g.attrs(n) for n in g.nodes if n.op_type == "MaxPool"]
    assert all(p["kernel_shape"] == [5, 5] and p["pads"] == [2, 2, 2, 2] and p["strides"] == [1, 1] for p in pools)
    resizes = [g.attrs(n) for n in g.nodes if n.op_type == "Resize"]
    assert all(r["mode"] == b"nearest" or r["mode"] == "nearest" for r in resizes)
    # C2f block sizes (number of bottlenecks, shortcut) per module index
    c2f = OrderedDict()
    for i in range(23):
        nb = 0
        while f"/model.{i}/m.{nb}/cv1/conv/Conv" in g.by_name:
            nb += 1
        if nb:
            c2f[str(i)] = OrderedDict(n=nb, shortcut=f"/model.{i}/m.0/Add" in g.by_name)
    dfl_w = tensors["model.22.dfl.conv.weight"].reshape(-1)
    assert np.allclose(dfl_w, np.arange(dfl_w.size)), dfl_w
    strides = [scalar(g.value(g.node(f"/model.22/{n}").input[1])) for n in ("Add_2", "Add_5", "Add_8")]
    arch = OrderedDict(model="face_detect_v1.4_n", source=os.path.basename(path), classes=["face"],
                       input={"size": 640, "pad_value": 114, "scale": 1.0 / 255.0},
                       strides=strides, reg_max=int(dfl_w.size), c2f=c2f, convs=convs,
                       sppf={"pool_kernel": 5, "pools": 3}, upsample="nearest x2",
                       output="[1, 4 + classes, anchors]: cx, cy, w, h in input pixels, class sigmoid")
    write(out_dir, "face_yolov8n", tensors, arch)
    print("yolo tensors", len(tensors))


def write(out_dir, stem, tensors, arch):
    os.makedirs(out_dir, exist_ok=True)
    save_file(dict(tensors), os.path.join(out_dir, stem + ".safetensors"),
              metadata={"format": "gufo-ccip", "arch": stem})
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=1)


if __name__ == "__main__":
    kind, src, dst = sys.argv[1:4]
    {"ccip": convert_ccip, "yolo": convert_yolo}[kind](src, dst)
