#pragma once

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>
#include <memory>
#include <string>
#include "weight-pack.h"

namespace ggml_xrt {

struct runtime_state {
    xrt::device device;
    xrt::uuid   xclbin_uuid;
    xrt::kernel kernel;
    xrt::kernel vit_kernel;
    xrt::kernel connector_kernel;
    xrt::kernel decoder_kernel;
    std::string xclbin_path;
    std::string manifest_path;
    sidecar_manifest sidecar;
    bool        ready = false;
};

runtime_state * get_runtime();

} // namespace ggml_xrt
