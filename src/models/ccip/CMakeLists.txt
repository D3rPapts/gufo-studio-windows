# CCIP character fingerprints (deepghs ccip-caformer-24-randaug-pruned,
# OpenRAIL) + anime face detector (deepghs face_detect_v1.4_n, YOLOv8n, MIT):
# CPU fp32 runtime used by the router's POST /v1/figuren ("verfahren":
# "ccip"). Weights: reference/convert_weights.py (ONNX -> safetensors + JSON).
# Same code as the 2.0 library extension (branch v2/ccip). Written with AI
# assistance (Claude Code).

find_package(OpenMP)

add_library(gufo_ccip STATIC
  tensors.cpp
  image_ops.cpp
  cpu/kernels.cpp
  cpu/caformer.cpp
  cpu/yolo.cpp
)

target_include_directories(gufo_ccip PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_ccip PUBLIC cxx_std_20)
target_link_libraries(gufo_ccip PUBLIC Threads::Threads)

if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_ccip PRIVATE OpenMP::OpenMP_CXX)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  if (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set_source_files_properties(cpu/kernels.cpp PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma")
  else()
    set_source_files_properties(cpu/kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
elseif (MSVC)
  set_source_files_properties(cpu/kernels.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()
