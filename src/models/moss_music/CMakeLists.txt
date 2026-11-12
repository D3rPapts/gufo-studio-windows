# MOSS-Music-8B-Instruct (OpenMOSS): music captioning, lyrics transcription
# and musical question answering, ported natively (no Python worker).
# Stage 1: configuration, the weight table and the CPU shape tests. The
# runtime (audio encoder on Qwen3-ASR's kernels, Qwen3-8B dense LLM with
# DeepStack injection, captioner) follows in later stages; see
# tools/moss_music/convert_hf_to_gufo.py for the weight conversion.

# Header-only until stage 2; it will link gufo_core (GGUF reader) and
# gufo_qwen3_asr (audio frontend, kernels) once the loader lands.
add_library(gufo_moss_music INTERFACE)
target_include_directories(gufo_moss_music INTERFACE ${PROJECT_SOURCE_DIR})

if (BUILD_TESTING)
  # Pure CPU, no weights: Conv2d-stem shape math, chunking, time-marker
  # layout and the parameter count of the weight table.
  add_executable(moss_music_shape_test
    ${PROJECT_SOURCE_DIR}/tests/models/moss_music/shape_test.cpp)
  target_link_libraries(moss_music_shape_test PRIVATE gufo_moss_music)
  add_test(NAME moss_music_shape_test COMMAND moss_music_shape_test)
  set_tests_properties(moss_music_shape_test PROPERTIES
    LABELS "cpu;models;moss-music;quality")
endif()
