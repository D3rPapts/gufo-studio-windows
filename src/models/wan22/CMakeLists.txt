# Wan 2.2 TI2V-5B: umT5-XXL tokenizer, text encoder and transformer (DiT).
add_library(gufo_wan22 STATIC EXCLUDE_FROM_ALL tokenizer.cpp)
target_include_directories(gufo_wan22 PUBLIC ${PROJECT_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_wan22 PRIVATE text_encoder.hip transformer.hip
    attention.hip gemm.cpp)
  target_compile_options(gufo_wan22 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_link_libraries(gufo_wan22 PUBLIC gufo_platform hip::host
    roc::rocblas roc::hipblaslt)

  add_executable(gufo-wan-t5 EXCLUDE_FROM_ALL t5_main.cpp)
  set_target_properties(gufo-wan-t5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-wan-t5 PRIVATE gufo_wan22)
  gufo_stage_hip_runtime(gufo-wan-t5)

  add_executable(gufo-wan-dit EXCLUDE_FROM_ALL dit_main.cpp)
  set_target_properties(gufo-wan-dit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-wan-dit PRIVATE gufo_wan22)
  gufo_stage_hip_runtime(gufo-wan-dit)
endif()

if (BUILD_TESTING)
  add_executable(wan22_tokenizer_test
    ${PROJECT_SOURCE_DIR}/tests/models/wan22/tokenizer_test.cpp)
  target_link_libraries(wan22_tokenizer_test PRIVATE gufo_wan22)
  add_test(NAME wan22_tokenizer_test COMMAND wan22_tokenizer_test)
  set_tests_properties(wan22_tokenizer_test PROPERTIES
    LABELS "cpu;models;wan22;external-model" SKIP_RETURN_CODE 77)
endif()
