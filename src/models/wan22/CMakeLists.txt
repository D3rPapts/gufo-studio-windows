# Wan 2.2 TI2V-5B: umT5-XXL tokenizer, text encoder, transformer (DiT), UniPC
# scheduler, VAE decoder and the text-to-video pipeline.
add_library(gufo_wan22 STATIC EXCLUDE_FROM_ALL tokenizer.cpp scheduler.cpp)
target_include_directories(gufo_wan22 PUBLIC ${PROJECT_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_wan22 PRIVATE text_encoder.hip transformer.hip
    attention.hip gemm.cpp vae.hip pipeline.cpp clip_vision.hip
    transformer21.hip uni3c.hip pipeline21.cpp
    moge.hip camera_path.cpp point_render.hip)
  target_compile_options(gufo_wan22 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_link_libraries(gufo_wan22 PUBLIC gufo_platform hip::host
    roc::rocblas roc::hipblaslt)

  foreach(tool t5 dit video check21)
    add_executable(gufo-wan-${tool} EXCLUDE_FROM_ALL ${tool}_main.cpp)
    set_target_properties(gufo-wan-${tool} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_link_libraries(gufo-wan-${tool} PRIVATE gufo_wan22)
    gufo_stage_hip_runtime(gufo-wan-${tool})
  endforeach()
  # --image reads PNG/JPEG/WebP through the core image decoder.
  target_link_libraries(gufo-wan-video PRIVATE gufo_core)
  target_link_libraries(gufo-wan-check21 PRIVATE gufo_core)
endif()

if (BUILD_TESTING)
  add_executable(wan22_tokenizer_test
    ${PROJECT_SOURCE_DIR}/tests/models/wan22/tokenizer_test.cpp)
  target_link_libraries(wan22_tokenizer_test PRIVATE gufo_wan22)
  add_test(NAME wan22_tokenizer_test COMMAND wan22_tokenizer_test)
  set_tests_properties(wan22_tokenizer_test PROPERTIES
    LABELS "cpu;models;wan22;external-model" SKIP_RETURN_CODE 77)
endif()
