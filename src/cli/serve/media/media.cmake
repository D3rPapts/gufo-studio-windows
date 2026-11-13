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
# TripoSG and TRELLIS.2 image -> 3D (`gufo serve objekt`, feat/trellis2):
# the GPU runners in HIP builds, stubs that report the missing GPU path
# otherwise.
add_subdirectory(src/models/triposg)
add_subdirectory(src/models/trellis2)

add_library(gufo_media STATIC
  src/cli/serve/media/media_serve.cpp
  src/cli/serve/media/music_api.cpp
  src/cli/serve/music_encode.cpp
  src/cli/serve/media/foley_api.cpp
  src/cli/serve/media/omni_api.cpp
  src/cli/serve/media/upscale_api.cpp
  src/cli/serve/media/objekt_service.cpp
  src/cli/serve/media/video_rife.cpp
  src/cli/serve/media/video_upscale.cpp
)
target_link_libraries(gufo_media PUBLIC gufo_http gufo_core gufo_ace_step
  gufo_mmaudio gufo_rife gufo_realesrgan_cpu gufo_qwen_image_21 PNG::PNG)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_media PUBLIC gufo_triposg gufo_trellis2)
else()
  target_link_libraries(gufo_media PUBLIC gufo_triposg_stub gufo_trellis2_stub)
endif()
# mp3 / m4a music results (src/cli/serve/music_encode.cpp) run this FFmpeg.
target_compile_definitions(gufo_media PRIVATE
  GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}")
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

  # gufo serve objekt with a fake runner: number range checks, DELETE while
  # a job runs (no model, no GPU). Built from objekt_service.cpp itself, so
  # the test needs no media / HIP libraries.
  add_executable(objekt_service_test tests/cli/objekt_service_test.cpp
    src/cli/serve/media/objekt_service.cpp)
  target_link_libraries(objekt_service_test PRIVATE gufo_qwen_image_21)
  target_compile_options(objekt_service_test PRIVATE ${gufo_media_test_undebug})
  add_test(NAME objekt_service_test COMMAND objekt_service_test)
  set_tests_properties(objekt_service_test PROPERTIES
    LABELS "cpu;server;media" TIMEOUT 60)

  # /v1/audio/music response_format / bitrate_kbps parsing, the FFmpeg
  # argument list and a real mp3 transcode of a tiny WAV (skips without
  # ffmpeg); no model.
  add_executable(music_encode_test tests/cli/music_encode_test.cpp
    src/cli/serve/music_encode.cpp)
  target_include_directories(music_encode_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_features(music_encode_test PRIVATE cxx_std_20)
  target_compile_definitions(music_encode_test PRIVATE
    GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}")
  add_test(NAME music_encode_test COMMAND music_encode_test)
  set_tests_properties(music_encode_test PROPERTIES
    LABELS "cpu;server;api;music" TIMEOUT 120 SKIP_RETURN_CODE 77)

  # Live music preview helpers (/v1/audio/music/<id>/preview): the clean
  # estimate x0 = x_t - t * v, the window around the middle of the song,
  # the step rule and the mono 24 kHz WAV; no model, no GPU.
  add_executable(ace_step_preview_test tests/models/ace_step_preview_test.cpp
    src/models/ace_step/preview.cpp)
  target_include_directories(ace_step_preview_test PRIVATE
    ${PROJECT_SOURCE_DIR})
  target_compile_features(ace_step_preview_test PRIVATE cxx_std_20)
  target_compile_options(ace_step_preview_test PRIVATE ${gufo_media_test_undebug})
  add_test(NAME ace_step_preview_test COMMAND ace_step_preview_test)
  set_tests_properties(ace_step_preview_test PROPERTIES
    LABELS "cpu;server;api;music;preview" TIMEOUT 30)

  # ACE-Step sampler mathematics (src/models/ace_step/sampler, audio_io):
  # timestep schedule, APG / CFG guidance, repaint masks, injection, blends,
  # waveform splice, task rules, WAV decoding and resampling; no model, no
  # GPU.
  add_executable(ace_step_sampler_test tests/models/ace_step_sampler_test.cpp
    src/models/ace_step/sampler.cpp src/models/ace_step/audio_io.cpp)
  target_include_directories(ace_step_sampler_test PRIVATE
    ${PROJECT_SOURCE_DIR})
  target_compile_features(ace_step_sampler_test PRIVATE cxx_std_20)
  target_compile_options(ace_step_sampler_test PRIVATE ${gufo_media_test_undebug})
  add_test(NAME ace_step_sampler_test COMMAND ace_step_sampler_test)
  set_tests_properties(ace_step_sampler_test PROPERTIES
    LABELS "cpu;server;api;music" TIMEOUT 60)
endif()
