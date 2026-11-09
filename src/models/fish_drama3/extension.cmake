# Fish Audio Drama-3 TTS (gufo-studio-windows): native FP8 (F8_E4M3 + 128x128
# scale_inv) MoE backbone, codebook tower and DAC codec on HIP. Ported from the
# 1.4.9 fork (src/models/fish_drama3); written with AI assistance (Claude Code).

gufo_extension_target(
  NAME gufo_fish_drama3
  SOURCES
    config.cpp
    loader.cpp
    prompt.cpp
    tensor.cpp
    tokenizer.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    gufo_qwen_text
    ICU::uc
    ICU::i18n
    Threads::Threads
)

if(TARGET gufo_fish_drama3)
  target_include_directories(gufo_fish_drama3 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

gufo_extension_target(
  NAME gufo_fish_drama3_hip
  SOURCES
    hip/backbone_runtime.cpp
    hip/codec_runtime.cpp
    hip/synthesis_runtime.cpp
    hip/tower_runtime.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_fish_drama3
    hip::host
    roc::hipblas
    roc::rocblas
)

if(TARGET gufo_fish_drama3_hip)
  gufo_extension_hip_sources(gufo_fish_drama3_hip
    SOURCES
      hip/backbone_ops.hip
      hip/codec_ops.hip
      hip/tower_ops.hip
  )
  target_compile_definitions(gufo_fish_drama3_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_fish_drama3_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
