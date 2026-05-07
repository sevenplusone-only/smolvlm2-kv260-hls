#include "fpga_backend.h"

namespace smolvlm2 {

FpgaBackend::FpgaBackend(const std::string &xclbin_path, unsigned device_index)
    : device_(device_index)
{
    uuid_ = device_.load_xclbin(xclbin_path);
    decoder_kernel_ = xrt::kernel(device_, uuid_, "smolvlm2_decoder_layer",
                                  xrt::kernel::cu_access_mode::exclusive);
    vit_kernel_ = xrt::kernel(device_, uuid_, "smolvlm2_vit_prefill_kernel",
                              xrt::kernel::cu_access_mode::exclusive);
    connector_kernel_ = xrt::kernel(device_, uuid_, "smolvlm2_connector_kernel",
                                    xrt::kernel::cu_access_mode::exclusive);
}

FpgaBuffer FpgaBackend::alloc(size_t bytes, const xrt::kernel &kernel, int group_id) {
    FpgaBuffer buffer;
    buffer.bytes = bytes;
    buffer.bo = xrt::bo(device_, bytes, XCL_BO_FLAGS_NONE, kernel.group_id(group_id));
    return buffer;
}

void FpgaBackend::sync_to_device(FpgaBuffer &buffer) {
    buffer.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void FpgaBackend::sync_from_device(FpgaBuffer &buffer) {
    buffer.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

} // namespace smolvlm2
