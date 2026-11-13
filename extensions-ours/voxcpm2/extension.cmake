# VoxCPM2 (OpenBMB, Apache-2.0) out-of-tree gufo extension: MiniCPM-4
# base/residual LMs + local DiT flow head + local encoder, AudioVAE V2 (causal
# convs, 16 kHz in, 48 kHz out). CPU reference runtime (FP32 math on BF16
# weights) plus an optional gfx1151 HIP runtime. Reuses the Ming-3.0-tts
# safetensors reader, CPU GEMV/attention kernels, resampler and HIP kernels,
# hence "depends": ["ming_tts"]. Ported from the gufo-studio-windows 1.x
# fork; written with AI assistance (Claude Code).
#
# Headers are included as "voxcpm2/..." and "ming_tts/...": the include root
# is the folder that holds both extension folders.

cmake_path(GET GUFO_EXTENSION_BUILD_ROOT PARENT_PATH _voxcpm2_include_root)

# See ming_tts/extension.cmake: plain driver flag, runtime library only.
find_package(OpenMP)
function(voxcpm2_use_openmp target)
  if(NOT TARGET ${target} OR NOT OpenMP_CXX_FOUND)
    return()
  endif()
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/clang:-fopenmp>)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
  endif()
  target_link_libraries(${target} PRIVATE ${OpenMP_CXX_LIBRARIES})
endfunction()

gufo_extension_target(
  NAME gufo_voxcpm2
  SOURCES
    config.cpp
    tokenizer.cpp
    vae_weights.cpp
    cpu/audio_vae.cpp
    cpu/model.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ming_tts
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_voxcpm2)
  target_include_directories(gufo_voxcpm2 PUBLIC
    "$<BUILD_INTERFACE:${_voxcpm2_include_root}>"
  )
  voxcpm2_use_openmp(gufo_voxcpm2)
  set(_voxcpm2_vae "${GUFO_EXTENSION_BUILD_ROOT}/cpu/audio_vae.cpp")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      set_source_files_properties("${_voxcpm2_vae}" PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma")
    else()
      set_source_files_properties("${_voxcpm2_vae}" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
  elseif(MSVC)
    set_source_files_properties("${_voxcpm2_vae}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
endif()

gufo_extension_target(
  NAME gufo_voxcpm2_hip
  SOURCES
    hip/audio_vae.cpp
    hip/model.cpp
  LINK_LIBRARIES
    gufo_voxcpm2
    gufo_ming_tts_hip
    gufo_ming_tts
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
)

if(TARGET gufo_voxcpm2_hip)
  gufo_extension_hip_sources(gufo_voxcpm2_hip
    SOURCES
      hip/ops.hip
  )
  target_compile_definitions(gufo_voxcpm2_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_voxcpm2_hip PUBLIC
    "$<BUILD_INTERFACE:${_voxcpm2_include_root}>"
  )
  target_compile_options(gufo_voxcpm2_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# Available on every build: the CPU reference runtime when HIP is absent.
gufo_extension_target(
  NAME gufo_voxcpm2_provider
  SOURCES
    ext_request.cpp
    synthesis.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_voxcpm2
    gufo_ming_tts
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_voxcpm2_provider)
  target_include_directories(gufo_voxcpm2_provider PUBLIC
    "$<BUILD_INTERFACE:${_voxcpm2_include_root}>"
  )
  voxcpm2_use_openmp(gufo_voxcpm2_provider)
  if(TARGET gufo_voxcpm2_hip)
    target_link_libraries(gufo_voxcpm2_provider PRIVATE gufo_voxcpm2_hip gufo_ext_hip)
    target_compile_definitions(gufo_voxcpm2_provider PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

gufo_extension_target(
  NAME voxcpm2_request_test
  SOURCES
    tests/request_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm2_provider
    gufo_voxcpm2
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET voxcpm2_request_test)
  add_test(NAME voxcpm2.request COMMAND voxcpm2_request_test)
  set_tests_properties(voxcpm2.request PROPERTIES LABELS "voxcpm2;cpu")
endif()

# Every stage against the Python CPU fp32 oracle (reference/voxcpm2_oracle.py).
# Needs GUFO_VOXCPM_MODEL and GUFO_VOXCPM_REF; skips (77) otherwise.
gufo_extension_target(
  NAME voxcpm2_oracle_test
  SOURCES
    tests/oracle_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm2_provider
    gufo_voxcpm2
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET voxcpm2_oracle_test)
  voxcpm2_use_openmp(voxcpm2_oracle_test)
  add_test(NAME voxcpm2.oracle COMMAND voxcpm2_oracle_test)
  set_tests_properties(voxcpm2.oracle PROPERTIES
    LABELS "voxcpm2;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)
endif()

# The provider's TtsRunner with a real checkpoint. Needs GUFO_VOXCPM_MODEL
# and GUFO_VOXCPM_VOICE.
gufo_extension_target(
  NAME voxcpm2_provider_test
  SOURCES
    tests/provider_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm2_provider
    gufo_voxcpm2
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET voxcpm2_provider_test)
  add_test(NAME voxcpm2.provider COMMAND voxcpm2_provider_test)
  set_tests_properties(voxcpm2.provider PROPERTIES
    LABELS "voxcpm2;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)
endif()

gufo_extension_target(
  NAME voxcpm2_hip_test
  SOURCES
    tests/hip_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm2_provider
    gufo_voxcpm2_hip
    gufo_voxcpm2
    gufo_ming_tts_hip
    gufo_ming_tts
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
)

if(TARGET voxcpm2_hip_test)
  target_compile_definitions(voxcpm2_hip_test PRIVATE ENGINE_ENABLE_HIP=1)
  add_test(NAME voxcpm2.hip COMMAND voxcpm2_hip_test)
  set_tests_properties(voxcpm2.hip PROPERTIES
    LABELS "voxcpm2;hip;gpu;gfx1151;external-model"
    SKIP_RETURN_CODE 77)
endif()
