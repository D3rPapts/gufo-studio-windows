# Qwen3-Omni-30B-A3B-Captioner: audio in, caption out. GGUF loading, the F32
# CPU reference and the HIP runtime; gufo-omni is the driver for all of it.
# The audio tower reuses Qwen3-ASR's kernels and the thinker the vendored
# llama.cpp K-quant matmuls of DeepSeek V4 Flash, both linked read-only.
if (NOT ENGINE_ENABLE_HIP)
  return()
endif()

add_library(gufo_qwen3_omni STATIC EXCLUDE_FROM_ALL
  captioner.cpp
  model.cpp
  reference.cpp
  hip/audio_encoder.cpp
  hip/thinker.cpp
  hip/thinker_kernels.hip)
set_source_files_properties(hip/thinker_kernels.hip PROPERTIES LANGUAGE HIP)
target_include_directories(gufo_qwen3_omni PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(gufo_qwen3_omni PUBLIC ENGINE_ENABLE_HIP=1)
target_compile_options(gufo_qwen3_omni PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
target_link_libraries(gufo_qwen3_omni PUBLIC
  gufo_core gufo_qwen3_asr gufo_deepseek_v4_flash_mmq
  hip::host roc::hipblas Threads::Threads)

add_executable(gufo-omni EXCLUDE_FROM_ALL main.cpp ds4_hooks.cpp)
# Next to gufo.exe, where build.ps1 stages the ROCm and vcpkg DLLs.
set_target_properties(gufo-omni PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(gufo-omni PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
target_link_libraries(gufo-omni PRIVATE gufo_qwen3_omni)

# `gufo serve omni`: the captioner inside the main binary, which already
# carries DeepSeek's definition of the MMQ host hook.
option(GUFO_WITH_OMNI "Serve Qwen3-Omni-30B-A3B-Captioner from gufo" ON)
if (GUFO_WITH_OMNI)
  target_link_libraries(gufo PRIVATE gufo_qwen3_omni)
  target_compile_definitions(gufo PRIVATE GUFO_HAVE_QWEN3_OMNI=1)
endif()
