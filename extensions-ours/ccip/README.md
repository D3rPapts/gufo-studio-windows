# CCIP + anime face detector (gufo library extension)

Native C++ runtime (CPU fp32, OpenMP) for

- **CCIP** `ccip-caformer-24-randaug-pruned` (deepghs, OpenRAIL): CAFormer
  backbone (stages 64/128/320/512, depths 3/12/18/3, SepConv in stages 0-1,
  attention with res_scale in stages 2-3) + CLIP AttentionPool2d head, 768
  features. `CcipDiff` = clip((1 - cos) / 2, 0, 1); same character below
  0.1785 (deepghs metrics).
- **face_detect_v1.4_n** (deepghs, MIT): YOLOv8n, one class, letterbox 640,
  conf 0.25, NMS IoU 0.7.

No Python or ONNX at run time. Written with AI assistance (Claude Code).

## Why a library, not a provider

gufo capabilities are fixed by the core (text, embeddings, ASR, TTS, image,
video, decision). CCIP would fit `embeddings` with image parts, but an
out-of-tree extension cannot decode images (`DecodeImage` lives in
`gufo_core`, which extensions do not link), and the face detector fits no
capability at all. So this is `"kind": "library"`: the router's
`/v1/figuren` (in-tree, links `gufo_core`) decodes the images and calls
`FaceDetector::Detect`, `HeadCropBox`, `CcipModel::Embed` and `CcipDiff`
directly. An `embeddings` provider can be added later on top of the library.

## Weights

One-time conversion (venv with onnx, numpy, safetensors; `python -I`):

    python -I reference/convert_weights.py ccip <model_feat.onnx> <dir>
    python -I reference/convert_weights.py yolo <face model.onnx> <dir>

writes `ccip_caformer.safetensors/.json` and `face_yolov8n.safetensors/.json`.
Every architecture value in the JSON is read off the ONNX graph; the script
asserts the block patterns it relies on.

## Preprocessing

`image_ops.cpp` reproduces Pillow `Image.resize(BILINEAR)` for RGB8 bit for
bit (support-scaled triangle filter, 22-bit fixed point, uint8 after each
pass, Pillow's pass-order rule), so inputs equal the Python reference
(`ccip_ref.py` / deepghs imgutils) exactly.

## Test

`ccip.oracle` (`ccip_oracle_test`): oracle data from
`reference/make_oracle.py` (onnxruntime CPU). Env `GUFO_CCIP_MODEL`,
`GUFO_CCIP_FACE`, `GUFO_CCIP_ORACLE`, optional `GUFO_CCIP_THREADS`.
