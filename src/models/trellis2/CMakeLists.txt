# TRELLIS.2 (image -> 3D with PBR, Microsoft, MIT): native port in progress.
# Stage 1 (CPU, builds without HIP): architecture constants (config.hpp,
# model_description.hpp), sampler formulas (scheduler.hpp), sparse voxel engine
# reference (sparse_conv_cpu.cpp), DINOv3 encoder (dinov3_cpu.cpp), image
# preprocessing (preprocess.cpp), flow DiT reference (dit_cpu.cpp), sparse-structure
# decoder reference (ss_dec_cpu.cpp), PNG/npy helpers.
# Stage 5/6: sparse decoders (sparse_dec_cpu.cpp), dual-grid mesh, GLB writer, preview.
# HIP: dinov3.hip (FP32 rocBLAS), dit.hip (qwen_image_21 runtime, BF16), ss_dec.hip,
# sparse_dec.hip (sparse engine: device hashmap, neighbour table, gather + hipBLASLt).
# AI-assisted (Claude Code), 2026-11-12.

# xatlas (MIT, third_party/xatlas): UV unwrap for the texture baking; warnings of the vendored file are silenced.
add_library(gufo_xatlas STATIC ${PROJECT_SOURCE_DIR}/third_party/xatlas/xatlas.cpp)
target_include_directories(gufo_xatlas PUBLIC ${PROJECT_SOURCE_DIR}/third_party/xatlas)
target_compile_features(gufo_xatlas PUBLIC cxx_std_17)
if (MSVC)
  target_compile_options(gufo_xatlas PRIVATE /w)
else()
  target_compile_options(gufo_xatlas PRIVATE -w)
endif()
find_package(ZLIB REQUIRED)

add_library(gufo_trellis2_cpu STATIC
  texture_bake.cpp
  sparse_conv_cpu.cpp
  dinov3_cpu.cpp
  preprocess.cpp
  png.cpp
  dit_cpu.cpp
  ss_dec_cpu.cpp
  sparse_dec_cpu.cpp
  dual_grid_mesh.cpp
  glb_writer.cpp
  mesh_preview.cpp
  mesh_post.cpp)
target_include_directories(gufo_trellis2_cpu PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_trellis2_cpu PUBLIC cxx_std_20)
target_link_libraries(gufo_trellis2_cpu PUBLIC gufo_platform gufo_triposg_cpu PRIVATE gufo_xatlas ZLIB::ZLIB)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_trellis2_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
if (ENGINE_ENABLE_HIP)
  # The safetensors reader lives in gufo_wan22 in HIP builds.
  target_link_libraries(gufo_trellis2_cpu PUBLIC gufo_wan22)
else()
  target_sources(gufo_trellis2_cpu PRIVATE ${PROJECT_SOURCE_DIR}/src/models/wan22/tensor_file.cpp)
endif()

if (ENGINE_ENABLE_HIP)
  add_library(gufo_trellis2 STATIC EXCLUDE_FROM_ALL dinov3.hip dit.hip ss_dec.hip sparse_dec.hip pipeline.cpp runner.cpp)
  target_include_directories(gufo_trellis2 PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_trellis2 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_compile_definitions(gufo_trellis2 PUBLIC GUFO_TRELLIS2_HIP=1)
  target_link_libraries(gufo_trellis2 PUBLIC gufo_trellis2_cpu gufo_qwen_image_21 hip::host roc::rocblas)
endif()

# Server runner without a GPU: reports the missing HIP build.
add_library(gufo_trellis2_stub STATIC runner_stub.cpp)
target_include_directories(gufo_trellis2_stub PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_trellis2_stub PUBLIC cxx_std_20)

# Stage probe: DINOv3 tokens, sparse-structure flow and decoder as .npy dumps.
add_executable(gufo-trellis2-probe EXCLUDE_FROM_ALL probe_main.cpp)
set_target_properties(gufo-trellis2-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(gufo-trellis2-probe PRIVATE gufo_trellis2_cpu gufo_qwen_image_21 gufo_core)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo-trellis2-probe PRIVATE gufo_trellis2)
  gufo_stage_hip_runtime(gufo-trellis2-probe)
endif()

if (BUILD_TESTING)
  foreach(name sparse_conv scheduler model_description dinov3_cpu dit_cpu ss_dec_cpu preprocess sparse_dec_cpu dual_grid_mesh glb_writer mesh_post texture_bake)
    add_executable(trellis2_${name}_test
      ${PROJECT_SOURCE_DIR}/tests/models/trellis2/${name}_test.cpp)
    target_link_libraries(trellis2_${name}_test PRIVATE gufo_trellis2_cpu)
    add_test(NAME trellis2_${name}_test COMMAND trellis2_${name}_test)
    set_tests_properties(trellis2_${name}_test PROPERTIES LABELS "cpu;models;trellis2")
  endforeach()
endif()
