# MOSS-Music-8B-Instruct (OpenMOSS): music captioning, lyrics transcription
# and musical question answering, ported natively (no Python worker).
# Stage 1: configuration, the weight table and the CPU shape tests.
# Stage 2 (in progress): the audio GGUF loader, the F32 CPU reference of the
# encoder/adapters and the gufo-moss-music dump driver. The HIP runtime, the
# Qwen3-8B LLM with DeepStack injection and the captioner follow; see
# tools/moss_music/convert_hf_to_gufo.py for the weight conversion.

add_library(gufo_moss_music STATIC EXCLUDE_FROM_ALL
  model.cpp
  reference.cpp)
target_include_directories(gufo_moss_music PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(gufo_moss_music PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
# gufo_qwen3_asr: WAV decoding, resampling and the Whisper log-mel frontend.
target_link_libraries(gufo_moss_music PUBLIC gufo_core gufo_qwen3_asr
  Threads::Threads)

add_executable(gufo-moss-music EXCLUDE_FROM_ALL main.cpp)
set_target_properties(gufo-moss-music PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(gufo-moss-music PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
target_link_libraries(gufo-moss-music PRIVATE gufo_moss_music)

if (BUILD_TESTING)
  # Pure CPU, no weights: Conv2d-stem shape math, chunking, time-marker
  # layout and the parameter count of the weight table (headers only).
  add_executable(moss_music_shape_test
    ${PROJECT_SOURCE_DIR}/tests/models/moss_music/shape_test.cpp)
  target_include_directories(moss_music_shape_test PRIVATE ${PROJECT_SOURCE_DIR})
  add_test(NAME moss_music_shape_test COMMAND moss_music_shape_test)
  set_tests_properties(moss_music_shape_test PROPERTIES
    LABELS "cpu;models;moss-music;quality")
endif()
