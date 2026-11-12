# MOSS-Music-8B-Instruct (OpenMOSS): music captioning, lyrics transcription
# and musical question answering, ported natively (no Python worker).
# Stage 1: configuration, the weight table and the CPU shape tests.
# Stage 2: the audio GGUF loader, the F32 CPU reference of the encoder and
# adapters, the HIP audio encoder (400-frame chunks on Qwen3-ASR's kernels,
# DeepStack taps, adapters) and the gufo-moss-music dump/compare driver. The
# Qwen3-8B LLM with DeepStack injection and the captioner follow; see
# tools/moss_music/convert_hf_to_gufo.py for the weight conversion.

add_library(gufo_moss_music STATIC EXCLUDE_FROM_ALL
  model.cpp
  reference.cpp)
target_include_directories(gufo_moss_music PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(gufo_moss_music PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
# gufo_qwen3_asr: WAV decoding, resampling, the Whisper log-mel frontend and
# (with HIP) the audio kernels, BLAS wrappers and hipBLASLt plans.
target_link_libraries(gufo_moss_music PUBLIC gufo_core gufo_qwen3_asr
  Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_moss_music PRIVATE
    captioner.cpp
    hip/audio_encoder.cpp
    hip/llm.cpp
    hip/stem_ops.hip)
  set_source_files_properties(hip/stem_ops.hip PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gufo_moss_music PUBLIC ENGINE_ENABLE_HIP=1
    GUFO_MOSS_MUSIC_HIP=1)
  # The LLM's quantized matrices (Q8_0/Q4_K/Q6_K) run on the vendored
  # llama.cpp MMQ kernels of DeepSeek V4 Flash, linked read-only.
  target_link_libraries(gufo_moss_music PUBLIC hip::host roc::hipblas
    gufo_deepseek_v4_flash_mmq)
endif()

# The MMQ library asks its host for pre-quantized activations; the
# standalone driver answers like gufo-omni (ds4_hooks.cpp), the gufo binary
# through DeepSeek's definition.
add_executable(gufo-moss-music EXCLUDE_FROM_ALL main.cpp
  $<$<BOOL:${ENGINE_ENABLE_HIP}>:hip/ds4_hooks.cpp>)
set_target_properties(gufo-moss-music PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(gufo-moss-music PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
target_link_libraries(gufo-moss-music PRIVATE gufo_moss_music)

# `gufo serve moss-music`: the captioner inside the main binary (HIP only;
# the gufo binary carries DeepSeek's definition of the MMQ host hook).
option(GUFO_WITH_MOSS_MUSIC "Serve MOSS-Music-8B-Instruct from gufo" ON)
if (GUFO_WITH_MOSS_MUSIC AND ENGINE_ENABLE_HIP)
  target_link_libraries(gufo PRIVATE gufo_moss_music)
  target_compile_definitions(gufo PRIVATE GUFO_HAVE_MOSS_MUSIC=1)
endif()

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
