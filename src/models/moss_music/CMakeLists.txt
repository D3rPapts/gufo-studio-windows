# MOSS-Music-8B-Instruct (OpenMOSS): music captioning, lyrics transcription
# and musical question answering, ported natively (no Python worker).
# Stage 1: configuration, the weight table and the CPU shape tests.
# Stage 2: the audio GGUF loader, the F32 CPU reference of the encoder and
# adapters, the HIP audio encoder (400-frame chunks on Qwen3-ASR's kernels,
# DeepStack taps, adapters) and the gufo-moss-music dump/compare driver. The
# Qwen3-8B LLM with DeepStack injection and the captioner follow; see
# tools/moss_music/convert_hf_to_gufo.py for the weight conversion.
# gufo-studio-windows 2.1: added from src/cli/serve/media/media.cmake after
# the extensions (WAV decoding and the audio kernels come from the qwen3_asr
# extension targets gufo_qwen3_asr / gufo_qwen3_asr_hip); media.cmake links
# the captioner into gufo_media as `gufo serve moss-music` when
# GUFO_MOSS_MUSIC_HIP is on.

# `gufo serve moss-music`: the captioner inside the main binary (HIP only;
# the gufo binary carries DeepSeek's definition of the MMQ host hook).
option(GUFO_WITH_MOSS_MUSIC "Serve MOSS-Music-8B-Instruct from gufo" ON)
set(GUFO_MOSS_MUSIC_HIP OFF PARENT_SCOPE)

if (NOT TARGET gufo_qwen3_asr)
  message(STATUS "MOSS-Music off: needs the qwen3_asr extension")
  return()
endif()

# The HIP runtime needs Qwen3-ASR's kernels (audio_ops, blas, text_ops) and
# the vendored llama.cpp MMQ kernels of DeepSeek V4 Flash.
set(gufo_moss_music_hip OFF)
if (ENGINE_ENABLE_HIP)
  if (TARGET gufo_qwen3_asr_hip AND TARGET gufo_deepseek_v4_flash_mmq)
    set(gufo_moss_music_hip ON)
  else()
    message(STATUS "MOSS-Music captioner off: needs the qwen3_asr HIP and "
                   "deepseek_v4_flash extensions")
  endif()
endif()
set(GUFO_MOSS_MUSIC_HIP ${gufo_moss_music_hip} PARENT_SCOPE)

add_library(gufo_moss_music STATIC EXCLUDE_FROM_ALL
  model.cpp
  reference.cpp)
target_include_directories(gufo_moss_music PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(gufo_moss_music PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
# gufo_qwen3_asr: WAV decoding, resampling and the Whisper log-mel frontend.
target_link_libraries(gufo_moss_music PUBLIC gufo_core gufo_qwen3_asr
  Threads::Threads)

if (gufo_moss_music_hip)
  target_sources(gufo_moss_music PRIVATE
    captioner.cpp
    hip/audio_encoder.cpp
    hip/llm.cpp
    hip/stem_ops.hip)
  set_source_files_properties(hip/stem_ops.hip PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gufo_moss_music PUBLIC ENGINE_ENABLE_HIP=1
    GUFO_MOSS_MUSIC_HIP=1)
  # gufo_qwen3_asr_hip: the audio kernels, BLAS wrappers and hipBLASLt plans;
  # gufo_qwen_text: the Qwen tokenizer (src/families/qwen/text). The LLM's
  # quantized matrices (Q8_0/Q4_K/Q6_K) run on the vendored llama.cpp MMQ
  # kernels of DeepSeek V4 Flash, linked read-only.
  target_link_libraries(gufo_moss_music PUBLIC gufo_qwen3_asr_hip
    gufo_qwen_text hip::host roc::hipblas gufo_deepseek_v4_flash_mmq)
endif()

# The MMQ library asks its host for pre-quantized activations; the
# standalone driver answers like gufo-omni (ds4_hooks.cpp), the gufo binary
# through DeepSeek's definition.
add_executable(gufo-moss-music EXCLUDE_FROM_ALL main.cpp
  $<$<BOOL:${gufo_moss_music_hip}>:hip/ds4_hooks.cpp>)
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
