#include "xrt-runtime.h"
#include "ggml-impl.h"
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace ggml_xrt {

static runtime_state g_state;
static std::once_flag g_once;

static std::optional<xrt::kernel> try_open_kernel(
        const xrt::device & device,
        const xrt::uuid & uuid,
        const char * name) {
    try {
        return xrt::kernel(device, uuid, name);
    } catch (...) {
        return std::nullopt;
    }
}

static void init_once() {
    const char * path = std::getenv("GGML_XRT_XCLBIN");
    if (!path) {
        GGML_LOG_WARN("ggml-xrt: GGML_XRT_XCLBIN not set - XRT backend disabled\n");
        return;
    }
    try {
        g_state.xclbin_path = path;
        g_state.device      = xrt::device(0);
        g_state.xclbin_uuid = g_state.device.load_xclbin(path);

        if (auto k = try_open_kernel(g_state.device, g_state.xclbin_uuid, "fpga_gemm_kernel")) {
            g_state.kernel = std::move(*k);
        } else {
            throw std::runtime_error("no compatible XRT GEMM kernel found");
        }

        if (auto k = try_open_kernel(g_state.device, g_state.xclbin_uuid, "smolvlm2_vit_prefill_kernel")) {
            g_state.vit_kernel = std::move(*k);
        }
        if (auto k = try_open_kernel(g_state.device, g_state.xclbin_uuid, "smolvlm2_connector_kernel")) {
            g_state.connector_kernel = std::move(*k);
        }
        if (auto k = try_open_kernel(g_state.device, g_state.xclbin_uuid, "smolvlm2_decoder_layer")) {
            g_state.decoder_kernel = std::move(*k);
        }

        const char * manifest = std::getenv("GGML_XRT_MANIFEST");
        if (manifest && manifest[0] != '\0') {
            g_state.manifest_path = manifest;
            g_state.sidecar = load_sidecar_manifest(g_state.manifest_path);
            if (g_state.sidecar.empty()) {
                GGML_LOG_WARN("ggml-xrt: failed to load sidecar manifest %s, keeping gguf fallback\n", manifest);
            } else {
                GGML_LOG_INFO("ggml-xrt: loaded sidecar manifest %s (%zu tensors)\n",
                              manifest, g_state.sidecar.tensors.size());
            }
        }

        g_state.ready = true;
        GGML_LOG_INFO("ggml-xrt: loaded %s\n", path);
    } catch (const std::exception & e) {
        GGML_LOG_WARN("ggml-xrt: init failed (%s) - falling back to CPU\n", e.what());
        g_state.ready = false;
    }
}

runtime_state * get_runtime() {
    std::call_once(g_once, init_once);
    return &g_state;
}

} // namespace ggml_xrt
