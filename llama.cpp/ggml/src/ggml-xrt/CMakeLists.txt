# ggml-xrt backend — Xilinx XRT for KV260 FPGA (Phase 1)
# Builds only on Linux / aarch64 with XRT installed at /usr/{include,lib}

find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h
    HINTS /usr/include/xrt /opt/xilinx/xrt/include
)

find_library(XRT_CORE_UTIL
    NAMES xrt_coreutil
    HINTS /usr/lib /usr/lib/aarch64-linux-gnu /opt/xilinx/xrt/lib
)

if (NOT XRT_INCLUDE_DIR OR NOT XRT_CORE_UTIL)
    message(FATAL_ERROR "ggml-xrt: XRT not found. Install xrt-dev and re-configure (-DGGML_XRT=ON).")
endif()

message(STATUS "XRT found: include=${XRT_INCLUDE_DIR} lib=${XRT_CORE_UTIL}")

ggml_add_backend_library(ggml-xrt
    ggml-xrt.cpp
    xrt-runtime.cpp
    weight-pack.cpp
)

target_include_directories(ggml-xrt PRIVATE ${XRT_INCLUDE_DIR})
target_include_directories(ggml-xrt PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../vendor)
target_link_libraries(ggml-xrt PRIVATE ${XRT_CORE_UTIL})
target_compile_features(ggml-xrt PRIVATE cxx_std_17)
