# RIFE 4.7 video frame interpolation (rife47 / rife49 IFNet) with FFmpeg I/O.
add_library(gufo_rife STATIC EXCLUDE_FROM_ALL
  interpolate.cpp
  "$<$<PLATFORM_ID:Windows>:video_io_windows.cpp>"
  "$<$<NOT:$<PLATFORM_ID:Windows>>:video_io_posix.cpp>")
target_include_directories(gufo_rife PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(gufo_rife PRIVATE
  GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}"
  GUFO_FFPROBE_EXECUTABLE="${GUFO_FFPROBE_EXECUTABLE}")

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_rife PRIVATE ifnet.hip)
  target_compile_options(gufo_rife PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
  target_link_libraries(gufo_rife PUBLIC hip::host)
endif()

# Standalone driver with the same options as `gufo video-interpolate`.
if (ENGINE_ENABLE_HIP)
  add_executable(gufo-rife EXCLUDE_FROM_ALL rife_main.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/video/video_interpolate.cpp)
  set_target_properties(gufo-rife PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-rife PRIVATE gufo_rife)
  gufo_stage_hip_runtime(gufo-rife)
endif()
