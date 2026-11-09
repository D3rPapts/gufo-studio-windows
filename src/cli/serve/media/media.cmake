# gufo-studio-windows media parts without an upstream capability
# (proposal gufo-org/gufo#1066), kept together so each can become an
# extension once a capability exists:
#   - ACE-Step 1.5 XL turbo   `gufo serve music`   /v1/audio/music
#   - MMAudio large_44k_v2    `gufo serve foley`   /v1/audio/foley
#   - Real-ESRGAN             `gufo serve upscale` /v1/images/upscale and the
#                             "gufo.upscale" post-step of video jobs
#   - RIFE 4.7                the "gufo.interpolate" post-step of video jobs
#   - Qwen3-Omni Captioner    `gufo serve omni`    /v1/chat/completions with
#                             one input_audio part (no upstream capability
#                             takes audio into text generation yet)
# The post-step core (video_postprocess.cpp) is part of gufo_http, because
# video_jobs runs it after any provider. Included once from the top
# CMakeLists.txt after the `gufo` executable is defined.
# Written with AI assistance (Claude Code).

# `gufo serve upscale --device npu|cpu`: ONNX Runtime is loaded at run time
# (Windows ML on Windows); only its MIT C API header is vendored, so there is
# no link dependency. Off elsewhere until a Linux NPU path is tested.
if (WIN32)
  set(GUFO_ENABLE_ONNXRUNTIME_DEFAULT ON)
else()
  set(GUFO_ENABLE_ONNXRUNTIME_DEFAULT OFF)
endif()
option(GUFO_ENABLE_ONNXRUNTIME
  "Load ONNX Runtime at run time for the NPU / CPU upscaler"
  ${GUFO_ENABLE_ONNXRUNTIME_DEFAULT})

add_subdirectory(src/models/rife)
add_subdirectory(src/models/ace_step)
add_subdirectory(src/models/mmaudio)
add_subdirectory(src/models/realesrgan)
# Qwen3-Omni-30B-A3B-Captioner (`gufo serve omni`, HIP only).
add_subdirectory(src/models/qwen3_omni)

add_library(gufo_media STATIC
  src/cli/serve/media/media_serve.cpp
  src/cli/serve/media/music_api.cpp
  src/cli/serve/media/foley_api.cpp
  src/cli/serve/media/omni_api.cpp
  src/cli/serve/media/upscale_api.cpp
  src/cli/serve/media/video_rife.cpp
  src/cli/serve/media/video_upscale.cpp
)
target_link_libraries(gufo_media PUBLIC gufo_http gufo_core gufo_ace_step
  gufo_mmaudio gufo_rife gufo_realesrgan_cpu PNG::PNG)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_media PUBLIC gufo_realesrgan)
  # `gufo video-interpolate` / `gufo video-upscale` (HIP only).
  target_sources(gufo PRIVATE src/cli/video/video_interpolate.cpp
                              src/cli/video/video_upscale.cpp)
endif()
if (TARGET gufo_qwen3_omni)
  target_link_libraries(gufo_media PUBLIC gufo_qwen3_omni)
  target_compile_definitions(gufo_media PRIVATE GUFO_HAVE_QWEN3_OMNI=1)
endif()
target_link_libraries(gufo PRIVATE gufo_media)

if (BUILD_TESTING)
  # This file is included before the top-level `add_compile_options(-UNDEBUG)`
  # for tests, so the tests turn their assertions back on themselves.
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(gufo_media_test_undebug /UNDEBUG)
  else()
    set(gufo_media_test_undebug -UNDEBUG)
  endif()

  # Services without a model: request handling, PNG output, CPU-build
  # behaviour and the post-step command-line options.
  add_executable(media_services_test tests/cli/media_services_test.cpp)
  target_link_libraries(media_services_test PRIVATE gufo_media)
  target_compile_options(media_services_test PRIVATE ${gufo_media_test_undebug})
  add_test(NAME media_services_test COMMAND media_services_test)
  set_tests_properties(media_services_test PROPERTIES
    LABELS "cpu;server;media" TIMEOUT 60)
endif()
