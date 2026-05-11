cmake_minimum_required(VERSION 3.16)
project(smolvlm2_llamacpp_fpga CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_ROOT "" CACHE PATH "Path to the official llama.cpp checkout")
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation prefix")

add_library(smolvlm2_fpga_backend
    fpga_backend.cpp
    fpga_c_api.cpp
    fpga_session.cpp
    smolvlm2_xrt_runner.cpp
)

target_include_directories(smolvlm2_fpga_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XRT_ROOT}/include
)

if (LLAMA_CPP_ROOT)
    target_include_directories(smolvlm2_fpga_backend PUBLIC
        ${LLAMA_CPP_ROOT}/include
        ${LLAMA_CPP_ROOT}/common
        ${LLAMA_CPP_ROOT}/src
    )
endif()

target_link_directories(smolvlm2_fpga_backend PUBLIC ${XRT_ROOT}/lib)
target_link_libraries(smolvlm2_fpga_backend PUBLIC xrt_coreutil xrt_core)
