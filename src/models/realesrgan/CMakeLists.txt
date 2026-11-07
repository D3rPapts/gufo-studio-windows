# Real-ESRGAN SRVGGNetCompact upscaling (realesr-animevideov3,
# realesr-general-x4v3) of videos and images; FFmpeg I/O from gufo_rife.

# Host side of the NPU / CPU path (`gufo serve upscale --device npu|cpu`):
# tiling, pixel shuffle, residual and OpenCV-exact INTER_AREA fitting
# (cpu_post), and the run-time loaded ONNX Runtime sessions (ort_upscaler).
add_library(gufo_realesrgan_cpu STATIC cpu_post.cpp)
target_include_directories(gufo_realesrgan_cpu PUBLIC ${PROJECT_SOURCE_DIR})
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_realesrgan_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
if (GUFO_ENABLE_ONNXRUNTIME)
  target_sources(gufo_realesrgan_cpu PRIVATE ort_upscaler.cpp)
  target_include_directories(gufo_realesrgan_cpu PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include)
  target_compile_definitions(gufo_realesrgan_cpu PUBLIC GUFO_HAVE_ONNXRUNTIME=1)
  if (WIN32)
    target_link_libraries(gufo_realesrgan_cpu PRIVATE ole32)
  else()
    target_link_libraries(gufo_realesrgan_cpu PRIVATE ${CMAKE_DL_LIBS})
  endif()

  # NPU / CPU probe: one frame through a body ONNX, PPM out, optional
  # comparison with a reference PNG (Take One's output). Not named gufo-*.
  add_executable(npu-upscale-probe EXCLUDE_FROM_ALL npu_probe_main.cpp)
  set_target_properties(npu-upscale-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(npu-upscale-probe PRIVATE gufo_realesrgan_cpu
    gufo_core)
endif()

if (BUILD_TESTING)
  add_executable(realesrgan_cpu_post_test
    ${PROJECT_SOURCE_DIR}/tests/models/realesrgan/cpu_post_test.cpp)
  target_link_libraries(realesrgan_cpu_post_test PRIVATE gufo_realesrgan_cpu)
  add_test(NAME realesrgan_cpu_post_test COMMAND realesrgan_cpu_post_test)
  set_tests_properties(realesrgan_cpu_post_test PROPERTIES
    LABELS "cpu;models;realesrgan")
endif()

if (ENGINE_ENABLE_HIP)
  add_library(gufo_realesrgan STATIC EXCLUDE_FROM_ALL
    upscaler.hip
    upscale_video.cpp)
  target_include_directories(gufo_realesrgan PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_realesrgan PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
  target_link_libraries(gufo_realesrgan PUBLIC gufo_rife hip::host)

  # Standalone driver with the same options as `gufo video-upscale`.
  add_executable(gufo-upscale EXCLUDE_FROM_ALL upscale_main.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/video/video_upscale.cpp)
  set_target_properties(gufo-upscale PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-upscale PRIVATE gufo_realesrgan)
  gufo_stage_hip_runtime(gufo-upscale)
endif()
