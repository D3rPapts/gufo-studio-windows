# IndexTTS-2.5 (IndexTeam, bilibili Model Use License) with the German GPT
# fine-tune (sharrnah/index-tts-2.5-german) as an out-of-tree gufo extension:
# GPT-2 text->semantic AR model with CAM++ speaker and Conformer/Perceiver
# emotion conditioning, w2v-BERT 2.0 semantic features, semantic codec,
# DiT/CFM s2mel and BigVGAN v2 vocoder. CPU reference runtime (FP32, weights
# used in place from the mapped .pth files) plus an optional gfx1151 HIP
# runtime (GPT decoder, CFM/DiT loop with hipBLASLt BF16, BigVGAN). Reuses the
# Ming-3.0-tts safetensors reader, resampler, Kaldi fbank, FFT and HIP kernels
# and the VoxCPM2 HIP attention / skinny GEMM, hence "depends": ["ming_tts",
# "voxcpm2"]. Ported from the gufo-studio-windows 1.x fork; written with AI
# assistance (Claude Code).
#
# Headers are included as "indextts/...", "ming_tts/..." and "voxcpm2/...":
# the include root is the folder that holds the extension folders.

cmake_path(GET GUFO_EXTENSION_BUILD_ROOT PARENT_PATH _indextts_include_root)

# See ming_tts/extension.cmake: plain driver flag, runtime library only (the
# external-extension validator rejects the "-Xclang -fopenmp" of OpenMP_CXX).
find_package(OpenMP)
function(indextts_use_openmp target)
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

# CPU runtime and registration. python-soxr resampling (audio.cpp) needs
# libsoxr, the German text frontend (text.cpp) ICU.
gufo_extension_target(
  NAME gufo_indextts
  SOURCES
    audio.cpp
    bundle.cpp
    bigvgan.cpp
    campplus.cpp
    gpt.cpp
    pth.cpp
    s2mel.cpp
    text.cpp
    w2vbert.cpp
    cpu/conv.cpp
    cpu/ops.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ming_tts
    gufo_ext_runtime
    ICU::uc
    ICU::i18n
    PkgConfig::soxr
    Threads::Threads
)

if(TARGET gufo_indextts)
  target_include_directories(gufo_indextts PUBLIC
    "$<BUILD_INTERFACE:${_indextts_include_root}>"
  )
  indextts_use_openmp(gufo_indextts)
  set(_indextts_ops "${GUFO_EXTENSION_BUILD_ROOT}/cpu/ops.cpp")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      set_source_files_properties("${_indextts_ops}" PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma;/clang:-mf16c")
    else()
      set_source_files_properties("${_indextts_ops}" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    endif()
  elseif(MSVC)
    set_source_files_properties("${_indextts_ops}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
endif()

# gfx1151 runtime. blas.cpp: hipBLASLt BF16 projections + FP32 batched
# attention GEMMs of the DiT/CFM (hipBLAS GemmEx fallback). rocBLAS and
# hipBLASLt load their Tensile kernel libraries relative to their own DLL; the
# host stages the DLLs and the rocblas/ and hipblaslt/ data directories beside
# gufo.exe (gufo_stage_hip_runtime), the same as in 1.x.
gufo_extension_target(
  NAME gufo_indextts_hip
  SOURCES
    hip/blas.cpp
    hip/runtime.cpp
  LINK_LIBRARIES
    gufo_indextts
    gufo_voxcpm2_hip
    gufo_ming_tts_hip
    gufo_ming_tts
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_indextts_hip)
  gufo_extension_hip_sources(gufo_indextts_hip
    SOURCES
      hip/ops.hip
      hip/attention.hip
  )
  target_compile_definitions(gufo_indextts_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_indextts_hip PUBLIC
    "$<BUILD_INTERFACE:${_indextts_include_root}>"
  )
  target_compile_options(gufo_indextts_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# Available on every build. Without the HIP target, hip/runtime.cpp compiles
# to its "built without HIP" stub here and the engine stays on the CPU.
set(_indextts_provider_sources
  ext_request.cpp
  synthesis.cpp
  provider.cpp
)
if(NOT TARGET gufo_indextts_hip)
  list(APPEND _indextts_provider_sources hip/runtime.cpp)
endif()
gufo_extension_target(
  NAME gufo_indextts_provider
  SOURCES
    ${_indextts_provider_sources}
  LINK_LIBRARIES
    gufo_indextts
    gufo_ming_tts
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_indextts_provider)
  target_include_directories(gufo_indextts_provider PUBLIC
    "$<BUILD_INTERFACE:${_indextts_include_root}>"
  )
  indextts_use_openmp(gufo_indextts_provider)
  if(TARGET gufo_indextts_hip)
    target_link_libraries(gufo_indextts_provider PRIVATE gufo_indextts_hip gufo_ext_hip)
    target_compile_definitions(gufo_indextts_provider PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# Request mapping (emotion in instructions, language, sampling defaults,
# reference limits). CPU only, no checkpoint; the probe check runs when
# GUFO_INDEXTTS_MODEL is set (file checks only, no load).
gufo_extension_target(
  NAME indextts_request_test
  SOURCES
    tests/request_test.cpp
  LINK_LIBRARIES
    gufo_indextts_provider
    gufo_indextts
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET indextts_request_test)
  add_test(NAME indextts.request COMMAND indextts_request_test)
  set_tests_properties(indextts.request PROPERTIES LABELS "indextts;cpu")
endif()

# Every stage against the Python CPU fp32 oracle (reference/indextts_oracle.py).
# Needs GUFO_INDEXTTS_MODEL, GUFO_INDEXTTS_REF and GUFO_INDEXTTS_VOICE; skips
# (77) otherwise. GUFO_INDEXTTS_STAGES selects stages.
gufo_extension_target(
  NAME indextts_oracle_test
  SOURCES
    tests/oracle_test.cpp
  LINK_LIBRARIES
    gufo_indextts_provider
    gufo_indextts
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET indextts_oracle_test)
  indextts_use_openmp(indextts_oracle_test)
  add_test(NAME indextts.oracle COMMAND indextts_oracle_test)
  set_tests_properties(indextts.oracle PROPERTIES
    LABELS "indextts;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    TIMEOUT 3600)
endif()

# The provider's TtsRunner with the real bundle (inline reference, emotion in
# instructions, streaming). Needs GUFO_INDEXTTS_MODEL and GUFO_INDEXTTS_VOICE.
gufo_extension_target(
  NAME indextts_provider_test
  SOURCES
    tests/provider_test.cpp
  LINK_LIBRARIES
    gufo_indextts_provider
    gufo_indextts
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET indextts_provider_test)
  add_test(NAME indextts.provider COMMAND indextts_provider_test)
  set_tests_properties(indextts.provider PROPERTIES
    LABELS "indextts;quality;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    TIMEOUT 3600)
endif()

gufo_extension_target(
  NAME indextts_hip_test
  SOURCES
    tests/hip_test.cpp
  LINK_LIBRARIES
    gufo_indextts_provider
    gufo_indextts_hip
    gufo_indextts
    gufo_voxcpm2_hip
    gufo_ming_tts_hip
    gufo_ming_tts
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
)

if(TARGET indextts_hip_test)
  target_compile_definitions(indextts_hip_test PRIVATE ENGINE_ENABLE_HIP=1)
  # Test executables do not sit beside gufo.exe: stage the HIP runtime DLLs and
  # the rocBLAS/hipBLASLt kernel directories next to this one as well.
  if(COMMAND gufo_stage_hip_runtime)
    gufo_stage_hip_runtime(indextts_hip_test)
  endif()
  add_test(NAME indextts.hip COMMAND indextts_hip_test)
  set_tests_properties(indextts.hip PROPERTIES
    LABELS "indextts;hip;gpu;gfx1151;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE)
endif()
