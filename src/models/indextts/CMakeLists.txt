# IndexTTS-2.5 (IndexTeam, bilibili Model Use License) with the German GPT
# fine-tune (sharrnah/index-tts-2.5-german): GPT-2 text->semantic AR model
# with CAM++ speaker and Conformer/Perceiver emotion conditioning, w2v-BERT 2.0
# semantic features, semantic codec, DiT/CFM s2mel and BigVGAN v2 vocoder.
# CPU reference runtime (FP32, weights used in place from the mapped .pth
# files); written with AI assistance (Claude Code). Reuses the Ming-3.0-tts
# safetensors reader, resampler, Kaldi fbank and FFT (gufo_ming_tts).

find_package(OpenMP)

add_library(gufo_indextts STATIC
  audio.cpp
  bigvgan.cpp
  campplus.cpp
  gpt.cpp
  pth.cpp
  s2mel.cpp
  synthesis.cpp
  text.cpp
  w2vbert.cpp
  cpu/conv.cpp
  cpu/ops.cpp
  hip/runtime.cpp
)

target_include_directories(gufo_indextts PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(gufo_indextts PUBLIC
  gufo_ming_tts
  gufo_platform
  ICU::uc
  ICU::i18n
  PkgConfig::SOXR
  Threads::Threads)

if (OpenMP_CXX_FOUND)
  target_link_libraries(gufo_indextts PRIVATE OpenMP::OpenMP_CXX)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  if (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set_source_files_properties(cpu/ops.cpp PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma;/clang:-mf16c")
  else()
    set_source_files_properties(cpu/ops.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
elseif (MSVC)
  set_source_files_properties(cpu/ops.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

if (ENGINE_ENABLE_HIP)
  # blas.cpp: hipBLASLt BF16 projections + FP32 batched attention GEMMs of the DiT/CFM.
  target_sources(gufo_indextts PRIVATE hip/ops.hip hip/attention.hip hip/blas.cpp)
  target_compile_definitions(gufo_indextts PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(gufo_indextts PUBLIC
    gufo_voxcpm2
    gufo_fish_drama3
    hip::host
    roc::hipblas
    roc::hipblaslt)
endif()