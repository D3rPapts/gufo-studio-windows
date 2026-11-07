# CCIP + anime face detector against onnxruntime (oracle data from
# src/models/ccip/reference/make_oracle.py). Needs GUFO_CCIP_MODEL,
# GUFO_CCIP_FACE and GUFO_CCIP_ORACLE; skips (77) otherwise.

find_package(OpenMP)

add_executable(ccip_oracle_test oracle_test.cpp)
target_link_libraries(ccip_oracle_test PRIVATE gufo_ccip)
if (OpenMP_CXX_FOUND)
  target_link_libraries(ccip_oracle_test PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME ccip_oracle_test COMMAND ccip_oracle_test)
set_tests_properties(ccip_oracle_test PROPERTIES
  LABELS "cpu;models;ccip;quality;external-model"
  SKIP_RETURN_CODE 77
  TIMEOUT 1200)
