# MMAudio large_44k_v2 (video / text to 44.1 kHz audio, "foley"): DFN5B CLIP,
# Synchformer, flow-matching network, VAE decoder, BigVGAN v2 vocoder, the
# pipeline behind `gufo serve foley` and the gufo-mmaudio / -check drivers.
# gufo-studio-windows only (no upstream capability); added from
# src/cli/serve/media/media.cmake.
# Written with AI assistance (Claude Code).
#
# CPU: tokenizer, frame selection / resizing, WAV and the video reader
# (FFmpeg through gufo_rife's video_io). The device stages need HIP.
add_library(gufo_mmaudio STATIC EXCLUDE_FROM_ALL tokenizer.cpp preprocess.cpp
  video_input.cpp)
target_include_directories(gufo_mmaudio PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_mmaudio PUBLIC gufo_core gufo_rife ICU::uc ICU::i18n)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_mmaudio PRIVATE kernels.hip common.cpp clip.cpp
    synchformer.cpp network.cpp vocoder.cpp pipeline.cpp)
  target_compile_options(gufo_mmaudio PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
  target_link_libraries(gufo_mmaudio PUBLIC gufo_platform hip::host
    roc::rocblas)

  # Spelled out (no foreach over the name) so the target ownership audit can
  # attribute both drivers to this directory.
  add_executable(gufo-mmaudio EXCLUDE_FROM_ALL foley_main.cpp)
  set_target_properties(gufo-mmaudio PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-mmaudio PRIVATE gufo_mmaudio)
  gufo_stage_hip_runtime(gufo-mmaudio)

  add_executable(gufo-mmaudio-check EXCLUDE_FROM_ALL check_main.cpp)
  set_target_properties(gufo-mmaudio-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-mmaudio-check PRIVATE gufo_mmaudio)
  gufo_stage_hip_runtime(gufo-mmaudio-check)
endif()

if (BUILD_TESTING)
  add_executable(mmaudio_preprocess_test
    ${PROJECT_SOURCE_DIR}/tests/models/mmaudio/preprocess_test.cpp)
  target_link_libraries(mmaudio_preprocess_test PRIVATE gufo_mmaudio)
  add_test(NAME mmaudio_preprocess_test COMMAND mmaudio_preprocess_test)
  set_tests_properties(mmaudio_preprocess_test PROPERTIES
    LABELS "cpu;models;mmaudio" SKIP_RETURN_CODE 77)
endif()
