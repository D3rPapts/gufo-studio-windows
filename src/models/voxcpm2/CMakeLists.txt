# VoxCPM2 (OpenBMB, Apache-2.0) - MiniCPM-4 base/residual LMs + local DiT
# flow head + local encoder, AudioVAE V2 (causal convs, 16 kHz in, 48 kHz out).
# CPU reference runtime (FP32 math on BF16 weights); written with AI
# assistance (Claude Code). Reuses the Ming-3.0-tts safetensors reader, CPU
# GEMV/attention kernels and resampler (gufo_ming_tts).

find_package(OpenMP)

add_library(gufo_voxcpm2 STATIC
  config.cpp
  synthesis.cpp
  tokenizer.cpp
  vae_weights.cpp
  cpu/audio_vae.cpp
  cpu/model.cpp
)

target_include_directories(gufo_voxcpm2 PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(gufo_voxcpm2 PUBLIC
  gufo_ming_tts
  gufo_platform
  Threads::Threads)

if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_voxcpm2 PRIVATE OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_voxcpm2 PRIVATE
    hip/model.cpp
    hip/ops.hip)
  target_compile_definitions(gufo_voxcpm2 PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_voxcpm2 PUBLIC
    gufo_fish_drama3
    hip::host)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  if (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set_source_files_properties(cpu/audio_vae.cpp PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma")
  else()
    set_source_files_properties(cpu/audio_vae.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
elseif (MSVC)
  set_source_files_properties(cpu/audio_vae.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()
