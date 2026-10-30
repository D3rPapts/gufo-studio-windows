# Real-ESRGAN SRVGGNetCompact upscaling (realesr-animevideov3,
# realesr-general-x4v3) of videos and images; FFmpeg I/O from gufo_rife.
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
