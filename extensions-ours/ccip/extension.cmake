# CCIP character fingerprints (deepghs CAFormer) + anime face detector
# (YOLOv8n) as a gufo library extension: native CPU fp32 runtime, no Python
# or ONNX at run time. Callers (router /v1/figuren, CLI) link gufo_ccip and
# pass decoded RGB8 pixels. Written with AI assistance (Claude Code).
#
# Headers are included as "ccip/...": the include root is the folder that
# holds the extension folders.

cmake_path(GET GUFO_EXTENSION_BUILD_ROOT PARENT_PATH _ccip_include_root)

find_package(OpenMP)
function(ccip_use_openmp target)
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
  NAME gufo_ccip
  SOURCES
    tensors.cpp
    image_ops.cpp
    cpu/kernels.cpp
    cpu/caformer.cpp
    cpu/yolo.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_ccip)
  target_include_directories(gufo_ccip PUBLIC
    "$<BUILD_INTERFACE:${_ccip_include_root}>"
  )
  ccip_use_openmp(gufo_ccip)
  set(_ccip_simd "${GUFO_EXTENSION_BUILD_ROOT}/cpu/kernels.cpp")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      set_source_files_properties("${_ccip_simd}" PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma")
    else()
      set_source_files_properties("${_ccip_simd}" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
  elseif(MSVC)
    set_source_files_properties("${_ccip_simd}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
endif()

# Every stage against onnxruntime (reference/make_oracle.py). Needs
# GUFO_CCIP_MODEL (ccip_caformer.*), GUFO_CCIP_FACE (face_yolov8n.*) and
# GUFO_CCIP_ORACLE (manifest.json + *.safetensors); skips (77) otherwise.
gufo_extension_target(
  NAME ccip_oracle_test
  SOURCES
    tests/oracle_test.cpp
  LINK_LIBRARIES
    gufo_ccip
    gufo_ext_runtime
)

if(TARGET ccip_oracle_test)
  ccip_use_openmp(ccip_oracle_test)
  add_test(NAME ccip.oracle COMMAND ccip_oracle_test)
  set_tests_properties(ccip.oracle PROPERTIES
    LABELS "ccip;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 1200)
endif()
