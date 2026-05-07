#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace smolvlm2 {

struct FpgaBuffer {
    xrt::bo bo;
    size_t  bytes = 0;
};

class FpgaBackend {
public:
    explicit FpgaBackend(const std::string &xclbin_path, unsigned device_index = 0);

    FpgaBuffer alloc(size_t bytes, const xrt::kernel &kernel, int group_id);
    void sync_to_device(FpgaBuffer &buffer);
    void sync_from_device(FpgaBuffer &buffer);

    xrt::kernel &decoder_kernel() { return decoder_kernel_; }
    xrt::kernel &vit_kernel() { return vit_kernel_; }
    xrt::kernel &connector_kernel() { return connector_kernel_; }

private:
    xrt::device device_;
    xrt::uuid   uuid_;
    xrt::kernel decoder_kernel_;
    xrt::kernel vit_kernel_;
    xrt::kernel connector_kernel_;
};

} // namespace smolvlm2
