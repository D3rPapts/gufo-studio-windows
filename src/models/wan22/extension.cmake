# Wan video family extension: Wan 2.2 TI2V 5B and Wan 2.1 I2V/T2V 14B with
# Uni3C camera control, VACE and InfiniteTalk (written with AI assistance).

# CPU: tokenizer, scheduler, VACE context, camera paths, request parsing and
# validation, speech decoding, provider registration.
gufo_extension_target(
  NAME gufo_wan22
  SOURCES
    tokenizer.cpp
    scheduler.cpp
    vace_context.cpp
    camera_path.cpp
    talk_audio.cpp
    request.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    PkgConfig::sndfile
    PkgConfig::soxr
)

if(TARGET gufo_wan22)
  target_include_directories(gufo_wan22 PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

# HIP: text encoder, DiTs, VAE, CLIP vision, Uni3C, MoGe-2, VACE,
# wav2vec2/InfiniteTalk, the pipelines, MP4 I/O and the runners.
if(WIN32)
  set(gufo_wan22_video_io video_io_windows.cpp)
else()
  set(gufo_wan22_video_io video_io_posix.cpp)
endif()
gufo_extension_target(
  NAME gufo_wan22_hip
  SOURCES
    gemm.cpp
    tensor_file.cpp
    parallel_read.cpp
    pipeline.cpp
    pipeline21.cpp
    pipeline_vace.cpp
    talk21.cpp
    ${gufo_wan22_video_io}
    runners.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_wan22
    gufo_platform
    hip::host
    roc::hipblaslt
    roc::rocblas
)

if(TARGET gufo_wan22_hip)
  gufo_extension_hip_sources(gufo_wan22_hip
    SOURCES
      text_encoder.hip
      transformer.hip
      transformer21.hip
      attention.hip
      vae.hip
      clip_vision.hip
      uni3c.hip
      vace.hip
      moge.hip
      point_render.hip
      wav2vec2.hip
      infinitetalk.hip
    OPTIONS
      -O3
      -ffp-contract=off
  )
  target_compile_definitions(gufo_wan22_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_wan22_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

# CPU tests: the Wan request formats through the core video API, the VACE
# context oracle and the umT5 tokenizer (skips without the model).
gufo_extension_target(
  NAME wan22_video_request_test
  SOURCES
    ${PROJECT_SOURCE_DIR}/tests/models/wan22/video_request_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/video_api.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/video_jobs.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/media/video_postprocess.cpp
  LINK_LIBRARIES
    gufo_wan22
    gufo_preview
    gufo_logging
    gufo_core
    gufo_cache
    gufo_ext_runtime
)
if(TARGET wan22_video_request_test)
  add_test(NAME wan22_video_request_test COMMAND wan22_video_request_test)
  set_tests_properties(wan22_video_request_test PROPERTIES
    LABELS "cpu;server;api;video;wan22")
endif()

gufo_extension_target(
  NAME wan22_vace_context_test
  SOURCES ${PROJECT_SOURCE_DIR}/tests/models/wan22/vace_context_test.cpp
  LINK_LIBRARIES gufo_wan22
)
if(TARGET wan22_vace_context_test)
  add_test(NAME wan22_vace_context_test COMMAND wan22_vace_context_test)
  set_tests_properties(wan22_vace_context_test PROPERTIES
    LABELS "cpu;models;wan22")
endif()

gufo_extension_target(
  NAME wan22_tokenizer_test
  SOURCES ${PROJECT_SOURCE_DIR}/tests/models/wan22/tokenizer_test.cpp
  LINK_LIBRARIES gufo_wan22
)
if(TARGET wan22_tokenizer_test)
  add_test(NAME wan22_tokenizer_test COMMAND wan22_tokenizer_test)
  set_tests_properties(wan22_tokenizer_test PROPERTIES
    LABELS "cpu;models;wan22;external-model" SKIP_RETURN_CODE 77)
endif()
