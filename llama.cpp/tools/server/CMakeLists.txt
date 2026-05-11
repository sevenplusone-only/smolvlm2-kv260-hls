set(TARGET llama-server)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

option(LLAMA_SERVER_SMOLVLM2_FPGA "Enable SmolVLM2 FPGA session integration for llama-server" OFF)

if (MINGW)
    # fix: https://github.com/ggml-org/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

set(TARGET_SRCS
    server.cpp
    utils.hpp
)

if (LLAMA_SERVER_SMOLVLM2_FPGA)
    list(APPEND TARGET_SRCS
        ${CMAKE_SOURCE_DIR}/../smolvlm2_kv260/llamacpp_fpga/fpga_backend.cpp
        ${CMAKE_SOURCE_DIR}/../smolvlm2_kv260/llamacpp_fpga/smolvlm2_xrt_runner.cpp
        ${CMAKE_SOURCE_DIR}/../smolvlm2_kv260/llamacpp_fpga/fpga_session.cpp
        ${CMAKE_SOURCE_DIR}/../smolvlm2_kv260/llamacpp_fpga/fpga_c_api.cpp
    )
endif()
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

if (LLAMA_SERVER_SMOLVLM2_FPGA)
    set(SMOLVLM2_FPGA_XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation prefix for SmolVLM2 FPGA server path")
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/../smolvlm2_kv260/llamacpp_fpga)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
    target_include_directories(${TARGET} PRIVATE ${SMOLVLM2_FPGA_XRT_ROOT}/include)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_SMOLVLM2_FPGA=1)
endif()

target_link_libraries(${TARGET} PRIVATE common mtmd ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SMOLVLM2_FPGA)
    target_link_directories(${TARGET} PRIVATE ${SMOLVLM2_FPGA_XRT_ROOT}/lib)
    target_link_libraries(${TARGET} PRIVATE xrt_coreutil xrt_core)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
