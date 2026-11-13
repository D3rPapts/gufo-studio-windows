# Ming-3.0-tts-0.5B (inclusionAI, MIT) out-of-tree gufo extension:
# Qwen2.5 backbone + DiT flow head + audio-VAE decoder, CAM++ speaker encoder
# read from campplus.onnx. CPU reference runtime (FP32 math on BF16 weights)
# plus an optional gfx1151 HIP runtime for the AR loop and VAE decoder.
# Ported from the gufo-studio-windows 1.x fork; written with AI assistance
# (Claude Code).
#
# Headers are included as "ming_tts/..." (and "voxcpm2/..." by the dependent
# VoxCPM2 extension), so the include root is the folder that holds the
# extension folders, not the extension folder itself.

cmake_path(GET GUFO_EXTENSION_BUILD_ROOT PARENT_PATH _ming_tts_include_root)

# OpenMP: the imported OpenMP target carries "-Xclang -fopenmp" on clang-cl,
# which the external-extension validator rejects. Compile with the plain
# driver flag and link only the runtime library, as a plain path (the
# validator also inspects the compile options of a target wrapped in
# $<LINK_ONLY:...>).
find_package(OpenMP)
function(ming_tts_use_openmp target)
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

# AVX2/FMA/F16C for the CPU GEMV kernels (every gfx1151 host CPU has them).
function(ming_tts_use_avx2 source)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma;/clang:-mf16c")
    else()
      set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    endif()
  elseif(MSVC)
    set_source_files_properties("${source}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
endfunction()

gufo_extension_target(
  NAME gufo_ming_tts
  SOURCES
    config.cpp
    prompt.cpp
    safetensors.cpp
    speaker.cpp
    tokenizer.cpp
    cpu/fft.cpp
    cpu/model.cpp
    cpu/ops.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    ICU::uc
    ICU::i18n
    Threads::Threads
)

if(TARGET gufo_ming_tts)
  target_include_directories(gufo_ming_tts PUBLIC
    "$<BUILD_INTERFACE:${_ming_tts_include_root}>"
  )
  ming_tts_use_openmp(gufo_ming_tts)
  ming_tts_use_avx2("${GUFO_EXTENSION_BUILD_ROOT}/cpu/ops.cpp")
endif()

gufo_extension_target(
  NAME gufo_ming_tts_hip
  SOURCES
    hip/model.cpp
  LINK_LIBRARIES
    gufo_ming_tts
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
    roc::rocblas
)

if(TARGET gufo_ming_tts_hip)
  gufo_extension_hip_sources(gufo_ming_tts_hip
    SOURCES
      hip/ops.hip
  )
  target_compile_definitions(gufo_ming_tts_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_ming_tts_hip PUBLIC
    "$<BUILD_INTERFACE:${_ming_tts_include_root}>"
  )
  target_compile_options(gufo_ming_tts_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# The provider is available on every build: the engine falls back to the CPU
# reference runtime when the HIP target is absent.
gufo_extension_target(
  NAME gufo_ming_tts_provider
  SOURCES
    ext_request.cpp
    synthesis.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_ming_tts
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_ming_tts_provider)
  target_include_directories(gufo_ming_tts_provider PUBLIC
    "$<BUILD_INTERFACE:${_ming_tts_include_root}>"
  )
  ming_tts_use_openmp(gufo_ming_tts_provider)
  if(TARGET gufo_ming_tts_hip)
    target_link_libraries(gufo_ming_tts_provider PRIVATE gufo_ming_tts_hip gufo_ext_hip)
    target_compile_definitions(gufo_ming_tts_provider PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

gufo_extension_target(
  NAME ming_tts_controls_test
  SOURCES
    tests/controls_test.cpp
  LINK_LIBRARIES
    gufo_ming_tts_provider
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET ming_tts_controls_test)
  add_test(NAME ming_tts.controls COMMAND ming_tts_controls_test)
  set_tests_properties(ming_tts.controls PROPERTIES LABELS "ming-tts;cpu")
endif()

# Every stage against the Python CPU fp32 oracle (reference/ming_oracle.py).
# Needs GUFO_MING_MODEL and GUFO_MING_REF; skips (77) otherwise.
gufo_extension_target(
  NAME ming_tts_oracle_test
  SOURCES
    tests/oracle_test.cpp
  LINK_LIBRARIES
    gufo_ming_tts_provider
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET ming_tts_oracle_test)
  ming_tts_use_openmp(ming_tts_oracle_test)
  add_test(NAME ming_tts.oracle COMMAND ming_tts_oracle_test)
  set_tests_properties(ming_tts.oracle PROPERTIES
    LABELS "ming-tts;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)
endif()

# The provider's TtsRunner with a real checkpoint (presets, inline cloning,
# controls, streaming). Needs GUFO_MING_MODEL and GUFO_MING_VOICE.
gufo_extension_target(
  NAME ming_tts_provider_test
  SOURCES
    tests/provider_test.cpp
  LINK_LIBRARIES
    gufo_ming_tts_provider
    gufo_ming_tts
    gufo_ext_runtime
)

if(TARGET ming_tts_provider_test)
  add_test(NAME ming_tts.provider COMMAND ming_tts_provider_test)
  set_tests_properties(ming_tts.provider PROPERTIES
    LABELS "ming-tts;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)
endif()

gufo_extension_target(
  NAME ming_tts_bench
  SOURCES
    tests/bench_main.cpp
  LINK_LIBRARIES
    gufo_ming_tts_provider
    gufo_ming_tts
    gufo_ext_runtime
)

gufo_extension_target(
  NAME ming_tts_hip_test
  SOURCES
    tests/hip_test.cpp
  LINK_LIBRARIES
    gufo_ming_tts_provider
    gufo_ming_tts_hip
    gufo_ming_tts
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
)

if(TARGET ming_tts_hip_test)
  target_compile_definitions(ming_tts_hip_test PRIVATE ENGINE_ENABLE_HIP=1)
  add_test(NAME ming_tts.hip COMMAND ming_tts_hip_test)
  set_tests_properties(ming_tts.hip PROPERTIES
    LABELS "ming-tts;hip;gpu;gfx1151;external-model"
    SKIP_RETURN_CODE 77)
endif()
