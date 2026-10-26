add_library(gufo_moss_image STATIC model.cpp)
target_include_directories(gufo_moss_image PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_moss_image PUBLIC gufo_qwen_image_21 gufo_core)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_moss_image PRIVATE hip/kernels.hip hip/pipeline.cpp)
  target_link_libraries(gufo_moss_image PUBLIC hip::host)
  # Preserve the BF16 operation boundaries of the PyTorch reference.
  target_compile_options(gufo_moss_image PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)
else()
  target_sources(gufo_moss_image PRIVATE model_stub.cpp)
endif()

if(ENGINE_ENABLE_HIP)
  add_executable(moss_image_probe EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/models/moss_image/probe.cpp)
  target_link_libraries(moss_image_probe PRIVATE gufo_moss_image)
endif()
