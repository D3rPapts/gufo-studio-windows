# TripoSG (image -> 3D shape, VAST-AI, MIT): native port in progress.
# Stage 1 (CPU, builds without HIP): architecture constants (config.hpp,
# model_description.hpp), marching cubes and the GLB/OBJ writer with their unit
# tests. Later stages add dinov2.hip / dit.hip / vae.hip and `gufo serve objekt`.
# AI-assisted (Claude Code), 2026-11-12.

add_library(gufo_triposg_cpu STATIC
  marching_cubes.cpp
  glb_writer.cpp)
target_include_directories(gufo_triposg_cpu PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_triposg_cpu PUBLIC cxx_std_20)

if (BUILD_TESTING)
  add_executable(triposg_marching_cubes_test
    ${PROJECT_SOURCE_DIR}/tests/models/triposg/marching_cubes_test.cpp)
  target_link_libraries(triposg_marching_cubes_test PRIVATE gufo_triposg_cpu)
  add_test(NAME triposg_marching_cubes_test COMMAND triposg_marching_cubes_test)
  set_tests_properties(triposg_marching_cubes_test PROPERTIES
    LABELS "cpu;models;triposg")

  add_executable(triposg_glb_writer_test
    ${PROJECT_SOURCE_DIR}/tests/models/triposg/glb_writer_test.cpp)
  target_link_libraries(triposg_glb_writer_test PRIVATE gufo_triposg_cpu)
  add_test(NAME triposg_glb_writer_test COMMAND triposg_glb_writer_test)
  set_tests_properties(triposg_glb_writer_test PROPERTIES
    LABELS "cpu;models;triposg")
endif()
