# TripoSG (image -> 3D shape, VAST-AI, MIT): native port in progress.
# Stage 1 (CPU, builds without HIP): architecture constants (config.hpp,
# model_description.hpp), marching cubes and the GLB/OBJ writer.
# Stage 3: image preprocessing (preprocess.cpp), DINOv2-L weights + FP32 CPU
# reference (dinov2_cpu.cpp) and the HIP encoder (dinov2.hip), probe tool.
# AI-assisted (Claude Code), 2026-11-12.

add_library(gufo_triposg_cpu STATIC
  marching_cubes.cpp
  glb_writer.cpp
  preprocess.cpp
  dinov2_cpu.cpp)
target_include_directories(gufo_triposg_cpu PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_triposg_cpu PUBLIC cxx_std_20)
target_link_libraries(gufo_triposg_cpu PUBLIC gufo_platform)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_triposg_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
if (ENGINE_ENABLE_HIP)
  # The safetensors reader lives in gufo_wan22 in HIP builds.
  target_link_libraries(gufo_triposg_cpu PUBLIC gufo_wan22)
else()
  target_sources(gufo_triposg_cpu PRIVATE ${PROJECT_SOURCE_DIR}/src/models/wan22/tensor_file.cpp)
endif()

if (ENGINE_ENABLE_HIP)
  add_library(gufo_triposg STATIC EXCLUDE_FROM_ALL dinov2.hip)
  target_include_directories(gufo_triposg PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_triposg PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_compile_definitions(gufo_triposg PUBLIC GUFO_TRIPOSG_HIP=1)
  target_link_libraries(gufo_triposg PUBLIC gufo_triposg_cpu hip::host roc::rocblas)
endif()

# Stage probe: preprocessing + DINOv2 tokens as .npy for tools/triposg/vergleich.py.
add_executable(gufo-triposg-probe EXCLUDE_FROM_ALL probe_main.cpp)
set_target_properties(gufo-triposg-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(gufo-triposg-probe PRIVATE gufo_triposg_cpu gufo_qwen_image_21 gufo_core)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo-triposg-probe PRIVATE gufo_triposg)
  gufo_stage_hip_runtime(gufo-triposg-probe)
endif()

if (BUILD_TESTING)
  foreach(name marching_cubes glb_writer preprocess dinov2_cpu)
    add_executable(triposg_${name}_test
      ${PROJECT_SOURCE_DIR}/tests/models/triposg/${name}_test.cpp)
    target_link_libraries(triposg_${name}_test PRIVATE gufo_triposg_cpu)
    add_test(NAME triposg_${name}_test COMMAND triposg_${name}_test)
    set_tests_properties(triposg_${name}_test PROPERTIES LABELS "cpu;models;triposg")
  endforeach()
endif()
