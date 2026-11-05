# Ming-3.0-tts-0.5B (inclusionAI, MIT) — Qwen2.5 backbone + DiT flow head +
# audio-VAE decoder, CAM++ speaker encoder read from campplus.onnx.
# CPU reference runtime (FP32 math on BF16 weights); written with AI
# assistance (Claude Code).

find_package(OpenMP)

add_library(gufo_ming_tts STATIC
  config.cpp
  prompt.cpp
  safetensors.cpp
  speaker.cpp
  synthesis.cpp
  tokenizer.cpp
  cpu/fft.cpp
  cpu/model.cpp
  cpu/ops.cpp
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

# The CPU GEMV kernels use AVX2/FMA (every gfx1151 host CPU has them).
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  if (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set_source_files_properties(cpu/ops.cpp PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma;/clang:-mf16c")
  else()
    set_source_files_properties(cpu/ops.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
elseif (MSVC)
  set_source_files_properties(cpu/ops.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_ming_tts PRIVATE
    hip/model.cpp
    hip/ops.hip)
  target_compile_definitions(gufo_ming_tts PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_ming_tts PUBLIC
    gufo_fish_drama3
    hip::host
    roc::rocblas)
endif()
