# Ming-3.0-tts-0.5B (inclusionAI, MIT) — Qwen2.5 backbone + DiT flow head +
# audio-VAE decoder, CAM++ speaker encoder read from campplus.onnx.
# CPU reference runtime (FP32 math on BF16 weights); written with AI
# assistance (Claude Code).

find_package(OpenMP)

add_library(gufo_ming_tts STATIC
  config.cpp
  prompt.cpp
  safetensors.cpp
  tokenizer.cpp
)

target_include_directories(gufo_ming_tts PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(gufo_ming_tts PUBLIC
  gufo_platform
  ICU::uc
  ICU::i18n
  Threads::Threads)

if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_ming_tts PRIVATE OpenMP::OpenMP_CXX)
endif()
