# ACE-Step 1.5 XL turbo (music): Qwen3 tokenizer and 0.6B text encoder,
# condition encoder, DiT decoder, Oobleck VAE decoder and the text-to-music
# pipeline.
add_library(gufo_ace_step STATIC EXCLUDE_FROM_ALL tokenizer.cpp)
target_include_directories(gufo_ace_step PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ace_step PUBLIC gufo_core ICU::i18n)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_ace_step PRIVATE weights.hip kernels.hip attention.hip
    model.hip vae.hip gemm.cpp pipeline.cpp)
  target_compile_options(gufo_ace_step PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_link_libraries(gufo_ace_step PUBLIC gufo_platform hip::host
    roc::hipblaslt)

  add_executable(gufo-ace-check EXCLUDE_FROM_ALL check_main.cpp)
  set_target_properties(gufo-ace-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-ace-check PRIVATE gufo_ace_step)
  gufo_stage_hip_runtime(gufo-ace-check)

  add_executable(gufo-ace EXCLUDE_FROM_ALL music_main.cpp)
  set_target_properties(gufo-ace PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-ace PRIVATE gufo_ace_step)
  gufo_stage_hip_runtime(gufo-ace)
endif()
