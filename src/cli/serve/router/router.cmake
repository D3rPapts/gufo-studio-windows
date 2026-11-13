# gufo-studio-windows router (`gufo router`): preset-driven model switching,
# remote placement, memory guard, jobs and the ComfyUI adapter in front of
# `gufo serve` workers. Not part of upstream gufo; included once from the top
# CMakeLists.txt after the `gufo` executable is defined.

# 1 when the build carries the fork-only `gufo serve omni` (Qwen3-Omni
# captioner, src/cli/serve/media, HIP only); without it the router rejects
# local omni sections (a [remote/<id>] with modality = omni still works).
# music, foley and upscale are back since Wave C. A plain variable, not an
# option, so build trees that cached the old OFF default follow it too.
if (ENGINE_ENABLE_HIP)
  set(GUFO_ROUTER_FORK_SERVE ON)
else()
  set(GUFO_ROUTER_FORK_SERVE OFF)
endif()

add_library(gufo_router STATIC
  src/cli/serve/router/auth.cpp
  src/cli/serve/router/comfy_adapter.cpp
  src/cli/serve/router/comfy_graph.cpp
  src/cli/serve/router/einrichtung.cpp
  src/cli/serve/router/figuren.cpp
  src/cli/serve/router/figuren_ccip.cpp
  src/cli/serve/router/jobs.cpp
  src/cli/serve/router/names.cpp
  src/cli/serve/router/preset.cpp
  src/cli/serve/router/proxy.cpp
  src/cli/serve/router/registry.cpp
  src/cli/serve/router/remote.cpp
  src/cli/serve/router/memory.cpp
  src/cli/serve/router/residency.cpp
  src/cli/serve/router/rgba_image.cpp
  src/cli/serve/router/router.cpp
  src/cli/serve/router/ws_relay.cpp
  src/cli/serve/serve_options.cpp
)
# Router workers: fork/exec on Linux, CreateProcessW in a kill-on-close job on
# Windows.
if (WIN32)
  target_sources(gufo_router PRIVATE src/cli/serve/router/workers_win32.cpp
                                     src/cli/serve/router/memory_win32.cpp)
  target_link_libraries(gufo_router PRIVATE dxgi pdh)
else()
  target_sources(gufo_router PRIVATE src/cli/serve/router/workers.cpp
                                     src/cli/serve/router/memory_posix.cpp)
endif()
target_compile_definitions(gufo_router PUBLIC
  GUFO_ROUTER_FORK_SERVE=$<BOOL:${GUFO_ROUTER_FORK_SERVE}>)
target_link_libraries(gufo_router PUBLIC gufo_http gufo_core)
target_link_libraries(gufo PRIVATE gufo_router)

if (BUILD_TESTING)
  # This file is included before the top-level `add_compile_options(-UNDEBUG)`
  # for tests, so the router tests turn their assertions back on themselves.
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(gufo_router_test_undebug /UNDEBUG)
  else()
    set(gufo_router_test_undebug -UNDEBUG)
  endif()

  add_test(NAME gufo_router_help COMMAND gufo router --help)
  set_tests_properties(gufo_router_help PROPERTIES
    LABELS "cpu;server;cli"
    PASS_REGULAR_EXPRESSION "\\[llm/<model-id>\\]")

  # Portable router unit tests (no model, no GPU).
  foreach (router_test IN ITEMS
      router_preset_test router_registry_test router_memory_test
      router_names_test router_figuren_test router_proxy_remote_test
      router_jobs_test router_pool_test router_figuren_ccip_endpoint_test
      router_auth_test router_einrichtung_test)
    add_executable(${router_test} tests/cli/${router_test}.cpp)
    target_link_libraries(${router_test} PRIVATE gufo_router)
    target_compile_options(${router_test} PRIVATE ${gufo_router_test_undebug})
    add_test(NAME ${router_test} COMMAND ${router_test})
    set_tests_properties(${router_test} PROPERTIES LABELS "cpu;router"
      TIMEOUT 60)
  endforeach()
  # The setup assistant's test reads the real catalogue and templates and,
  # where python3 is found, compares against einrichtung/einrichten.py.
  target_compile_definitions(router_einrichtung_test PRIVATE
    GUFO_EINRICHTUNG_DIR="${PROJECT_SOURCE_DIR}/einrichtung")
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if (Python3_EXECUTABLE)
    target_compile_definitions(router_einrichtung_test PRIVATE
      GUFO_PYTHON3="${Python3_EXECUTABLE}")
  endif()

  # The ComfyUI graph translator alone: no server library needed.
  foreach (router_test IN ITEMS router_comfy_graph_test
      router_comfy_control_test)
    add_executable(${router_test} tests/cli/${router_test}.cpp
      src/cli/serve/router/comfy_graph.cpp)
    target_include_directories(${router_test} PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(${router_test} PRIVATE cxx_std_20)
    target_compile_options(${router_test} PRIVATE ${gufo_router_test_undebug})
    add_test(NAME ${router_test} COMMAND ${router_test})
    set_tests_properties(${router_test} PROPERTIES LABELS "cpu;router"
      TIMEOUT 30)
  endforeach()

  # POSIX sockets, fork and signals: Linux only.
  if (NOT WIN32)
    foreach (router_test IN ITEMS router_proxy_test router_workers_test
        router_ws_relay_test)
      add_executable(${router_test} tests/cli/${router_test}.cpp)
      target_link_libraries(${router_test} PRIVATE gufo_router)
      target_compile_options(${router_test} PRIVATE ${gufo_router_test_undebug})
    add_test(NAME ${router_test} COMMAND ${router_test})
      set_tests_properties(${router_test} PROPERTIES LABELS "cpu;router"
        TIMEOUT 30)
    endforeach()
  endif()
endif()

# CCIP figure check ("verfahren": "ccip" in POST /v1/figuren), CPU only; the
# library is the extensions-ours/ccip extension (gufo_ccip).
if (TARGET gufo_ccip)
  target_link_libraries(gufo_router PUBLIC gufo_ccip)
else()
  message(FATAL_ERROR "gufo router needs the ccip extension (GUFO_EXTENSION_DIRS with extensions-ours)")
endif()
if (BUILD_TESTING)
  add_executable(router_figuren_ccip_test tests/cli/router_figuren_ccip_test.cpp
    src/cli/serve/router/figuren_ccip.cpp)
  target_include_directories(router_figuren_ccip_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_features(router_figuren_ccip_test PRIVATE cxx_std_20)
  add_test(NAME router_figuren_ccip_test COMMAND router_figuren_ccip_test)
  set_tests_properties(router_figuren_ccip_test PROPERTIES LABELS "cpu;router"
    TIMEOUT 30)
endif()
