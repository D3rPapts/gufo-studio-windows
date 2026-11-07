# ONNX Runtime C API header

`include/onnxruntime_c_api.h` and `include/onnxruntime_ep_c_api.h` (which the
first one includes) are copied unchanged from
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) tag `v1.23.2`
(`include/onnxruntime/core/session/`), MIT, see `LICENSE`. SHA-256:
`71125e66180a991d65c9bdbad4aa20daaa1f7a48c7a5c0fa5f18f250ac839a02` and
`b92778f50c36ecdf53d0344e0129a78b306afed653f81bba8a9597e3e6e9546f`. gufo does not link ONNX Runtime: `gufo serve upscale --device npu|cpu`
loads `onnxruntime.dll` (1.23 or newer) at run time, by default from the
Windows ML framework package. Nothing else in gufo loads it.
