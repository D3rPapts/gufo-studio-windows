# TRELLIS.2 (image -> 3D with PBR, Microsoft, MIT): native port in progress.
# Stage 1 (CPU, builds without HIP): architecture constants (config.hpp,
# model_description.hpp), sampler formulas (scheduler.hpp) and the sparse voxel
# engine reference (sparse_conv_cpu.cpp: hashmap, neighbour table, submanifold
# conv as gather-GEMM, Channel2Spatial). HIP parts follow in later stages.
# AI-assisted (Claude Code), 2026-11-12.

add_library(gufo_trellis2_cpu STATIC
  sparse_conv_cpu.cpp)
target_include_directories(gufo_trellis2_cpu PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_trellis2_cpu PUBLIC cxx_std_20)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_trellis2_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

if (BUILD_TESTING)
  foreach(name sparse_conv scheduler model_description)
    add_executable(trellis2_${name}_test
      ${PROJECT_SOURCE_DIR}/tests/models/trellis2/${name}_test.cpp)
    target_link_libraries(trellis2_${name}_test PRIVATE gufo_trellis2_cpu)
    add_test(NAME trellis2_${name}_test COMMAND trellis2_${name}_test)
    set_tests_properties(trellis2_${name}_test PROPERTIES LABELS "cpu;models;trellis2")
  endforeach()
endif()
