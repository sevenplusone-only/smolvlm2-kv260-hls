#include "ggml-impl.h"
#include "ggml-xrt.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "xrt-runtime.h"
#include "weight-pack.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct ggml_backend_xrt_context {
    ggml_xrt::runtime_state * rt = nullptr;
};

static const char * ggml_backend_xrt_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XRT";
}

static void ggml_backend_xrt_free(ggml_backend_t backend) {
    delete (ggml_backend_xrt_context *)backend->context;
    delete backend;
}

// ------------ Per-weight cached BOs (attached to tensor->extra) ------------
struct xrt_weight_cache {
    xrt::bo w_bo;    // [N][K] int8
    xrt::bo s_bo;    // [N][K/32] fp16
    bool from_sidecar = false;
    ~xrt_weight_cache() = default;
};

// Scratch buffers per backend instance (lazy-allocated, reused across ops)
struct xrt_scratch {
    xrt::bo              act_bo;      // [M][K] int8
    xrt::bo              act_s_bo;    // [M][K/32] fp16
    xrt::bo              out_bo;      // [M][N] float (written by kernel)
    std::vector<uint8_t> q8_workbuf;  // for F32->Q8_0 per-row quantize
    std::vector<float>   out_host;    // padded host readback
    size_t M = 0, K = 0, N = 0;
};

static void ggml_backend_xrt_mul_mat(ggml_backend_xrt_context * ctx, struct ggml_tensor * dst) {
    auto * rt = ctx->rt;
    GGML_ASSERT(rt && rt->ready);
    ggml_tensor * src0 = dst->src[0]; // weight Q8_0
    ggml_tensor * src1 = dst->src[1]; // act F32

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    static constexpr int64_t kTileM = 64;
    const int64_t M_pad = ((M + kTileM - 1) / kTileM) * kTileM;
    const size_t K_G = K / 32;
    // supports_op (Task 2a whitelist) has already filtered to valid (K, N) and M gate;
    // this assert is a dispatch-side tripwire for a broken whitelist.
    GGML_ASSERT(K > 0 && N > 0 && M > 0 && (K % 32 == 0));

    // --- Lazy scratch (re)allocation keyed on (M, K, N) triple.
    // Phase 1.5 Task 2a: host may now call with mixed ViT shapes — (1024,768,3072)
    // ffn_up vs (1024,768,768) attn. M/K constant within Task 2a, N varies, so
    // out_bo size differs and must be resized. Keyed match avoids under-allocation
    // silent-truncation of out_bo. Scratch thrashing across layers is accepted
    // for step-1 (single rekey per shape switch; Task 2b adds LLM shapes).
    static xrt_scratch scratch;
    if (scratch.M != (size_t)M_pad || scratch.K != (size_t)K || scratch.N != (size_t)N) {
        scratch.act_bo   = xrt::bo(rt->device, M_pad*K,     rt->kernel.group_id(0));
        scratch.act_s_bo = xrt::bo(rt->device, M_pad*K_G*2, rt->kernel.group_id(2));
        scratch.out_bo   = xrt::bo(rt->device, M_pad*N*4,   rt->kernel.group_id(4));
        scratch.q8_workbuf.resize(M_pad * K_G * sizeof(block_q8_0));
        scratch.out_host.resize(M_pad * N);
        scratch.M = M_pad; scratch.K = K; scratch.N = N;
    }

    // --- Weight cache: first encounter of this tensor -> unpack + sync to device ---
    // NOTE: weight_cache leaks on model reload (Phase 1 simplification). The BOs stay
    // allocated until process exit. For Phase 2, attach a destructor via a custom
    // ggml_backend_buffer_t that frees extras on buffer_free().
    if (src0->extra == nullptr) {
        auto * wc = new xrt_weight_cache();
        wc->w_bo = xrt::bo(rt->device, N*K,     rt->kernel.group_id(1));
        wc->s_bo = xrt::bo(rt->device, N*K_G*2, rt->kernel.group_id(3));
        const char * tensor_name = ggml_get_name(src0);
        const auto * sidecar_tensor = rt->sidecar.find_alias(tensor_name);
        if (sidecar_tensor != nullptr &&
            sidecar_tensor->quant == "q8_0" &&
            sidecar_tensor->K == (size_t) K &&
            sidecar_tensor->N == (size_t) N) {
            const uint8_t * sidecar_src = rt->sidecar.weights_blob.data() + sidecar_tensor->offset_bytes;
            ggml_xrt::unpack_q8_0_blob_to_split(sidecar_src,
                                                wc->w_bo.map<int8_t *>(),
                                                wc->s_bo.map<uint16_t *>(),
                                                (size_t)N, (size_t)K);
            wc->from_sidecar = true;
        } else {
            ggml_xrt::unpack_gguf_q8_to_split(src0->data,
                                              wc->w_bo.map<int8_t *>(),
                                              wc->s_bo.map<uint16_t *>(),
                                              (size_t)N, (size_t)K);
        }
        wc->w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wc->s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        src0->extra = wc;
    }
    auto * wc = (xrt_weight_cache *)src0->extra;

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    // --- Quantize activation (F32 -> block_q8_0) per-row into q8_workbuf ---
    const float * act_f32 = (const float *)src1->data;
    auto * blocks = (block_q8_0 *)scratch.q8_workbuf.data();
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_0_ref(&act_f32[m*K], &blocks[m*K_G], K);
    }
    for (int64_t m = M; m < M_pad; ++m) {
        std::memset(&blocks[m*K_G], 0, K_G * sizeof(block_q8_0));
    }

    // --- De-interleave Q8_0 blocks into scratch.act_bo (int8) + scratch.act_s_bo (fp16) ---
    int8_t  * act_i8 = scratch.act_bo.map<int8_t *>();
    uint16_t * act_s = scratch.act_s_bo.map<uint16_t *>();
    for (int64_t m = 0; m < M_pad; ++m) {
        for (size_t kg = 0; kg < K_G; ++kg) {
            const block_q8_0 & b = blocks[m*K_G + kg];
            act_s[m*K_G + kg] = *(const uint16_t *)&b.d;
            std::memcpy(&act_i8[m*K + kg*32], b.qs, 32);
        }
    }
    auto t1 = clk::now();

    scratch.act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    scratch.act_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t2 = clk::now();

    // --- Kernel launch ---
    auto run = rt->kernel(
        scratch.act_bo,
        wc->w_bo,
        scratch.act_s_bo,
        wc->s_bo,
        scratch.out_bo,
        (int)M_pad, (int)K, (int)N);
    run.wait();
    auto t3 = clk::now();

    // --- Read output back ---
    scratch.out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float * out_ptr = scratch.out_bo.map<const float *>();
    std::memcpy(scratch.out_host.data(), out_ptr, M_pad * N * sizeof(float));
    for (int64_t m = 0; m < M; ++m) {
        std::memcpy((float *)dst->data + m * N, scratch.out_host.data() + m * N, N * sizeof(float));
    }
    auto t4 = clk::now();

    auto us = [](clk::time_point a, clk::time_point b) {
        return (long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    static int call_no = 0;
    ++call_no;
    fprintf(stderr, "ggml-xrt-timing: call=%d pack_us=%ld sync_to_us=%ld kernel_us=%ld sync_from_us=%ld total_us=%ld\n",
            call_no, us(t0,t1), us(t1,t2), us(t2,t3), us(t3,t4), us(t0,t4));
}

static enum ggml_status ggml_backend_xrt_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_xrt_context *)backend->context;
    // XRT C++ API throws on BO alloc / sync / kernel launch failure. Contain it
    // here so device faults surface as GGML_STATUS_FAILED to the scheduler
    // instead of unwinding out of a C-linkage callback and killing the process.
    try {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * node = cgraph->nodes[i];
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    ggml_backend_xrt_mul_mat(ctx, node);
                    break;
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    break;
                default:
                    GGML_ABORT("ggml-xrt: unsupported op %s (supports_op should have rejected)",
                              ggml_op_desc(node));
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "ggml-xrt: graph_compute failed: %s\n", e.what());
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i xrt_backend_i = {
    /* .get_name                = */ ggml_backend_xrt_get_name,
    /* .free                    = */ ggml_backend_xrt_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_xrt_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_xrt_guid(void) {
    static ggml_guid guid = { 0x78, 0x52, 0x54, 0x20, 0x67, 0x67, 0x6d, 0x6c, 0x2d, 0x78, 0x72, 0x74, 0x2d, 0x70, 0x31, 0x00 };
    return &guid;
}

static const char * ggml_backend_xrt_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev); return "XRT";
}
static const char * ggml_backend_xrt_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev); return "Xilinx KV260 (FPGA, XRT)";
}
static void ggml_backend_xrt_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev); *free = 0; *total = 0;
}
static enum ggml_backend_dev_type ggml_backend_xrt_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev); return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static void ggml_backend_xrt_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xrt_device_get_name(dev);
    props->description = ggml_backend_xrt_device_get_description(dev);
    props->type        = ggml_backend_xrt_device_get_type(dev);
    ggml_backend_xrt_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /*.async=*/false, /*.host_buffer=*/false, /*.buffer_from_host_ptr=*/true, /*.events=*/false };
}

static ggml_backend_t ggml_backend_xrt_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * ctx = new ggml_backend_xrt_context();
    ctx->rt = ggml_xrt::get_runtime();
    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_xrt_guid(),
        /* .iface   = */ xrt_backend_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_xrt_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xrt_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev); GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_xrt_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    // Runtime not ready (xclbin missing, XRT load failed) → reject everything, scheduler falls back to CPU.
    auto * rt = ggml_xrt::get_runtime();
    if (!rt->ready) return false;

    // Diagnostic: GGML_XRT_DISABLE_DISPATCH forces every op to be rejected while
    // the XRT runtime stays loaded. Used in Step 0D to isolate "XRT init/threading
    // overhead" from "FPGA kernel slower than CPU" in prefill throughput analysis.
    static const bool disable_dispatch = []{
        const char * env = std::getenv("GGML_XRT_DISABLE_DISPATCH");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    if (disable_dispatch) return false;

    // v3.0 overlay (SHIP_K=960, 2026-04-17): MUL_MAT only, Q8_0×F32.
    // kSupported whitelist gates (spec_K, spec_N); spec_M runtime-variable,
    // 64-aligned, ≤ kKernelMaxM.
    //   ViT  (K=768): ffn_up (768,3072), attn_q/k/v/out (768,768)
    //   Text (K=960): attn_q/o (960,960), attn_k/v (960,320), ffn_gate/up (960,2560)
    // Not yet supported (v3.1+):
    //   ffn_down (2560,960)   — needs K-panel splitting (K>SHIP_K)
    //   lm_head  (960,49280)  — needs N-panel splitting (N>GEMM_MAX_N=3072)
    if (op->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * src0 = op->src[0]; // weight
    const ggml_tensor * src1 = op->src[1]; // activation
    if (!src0 || !src1)                   return false;
    if (src1->type != GGML_TYPE_F32)      return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    // ggml MUL_MAT: src0 shape = (k, spec_N), src1 shape = (k, spec_M), dst shape = (spec_N, spec_M)
    const int64_t K      = src0->ne[0];
    const int64_t spec_N = src0->ne[1];
    const int64_t spec_M = src1->ne[1];
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;

    static constexpr int64_t kKernelMaxM  = 1024;  // gemm_dims_valid cap
    const auto * sidecar_tensor = rt->sidecar.find_alias(ggml_get_name(src0));
    const bool has_sidecar_q8 =
        sidecar_tensor != nullptr &&
        sidecar_tensor->quant == "q8_0" &&
        sidecar_tensor->K == (size_t) K &&
        sidecar_tensor->N == (size_t) spec_N;
    if (src0->type != GGML_TYPE_Q8_0 && !has_sidecar_q8) return false;
    const bool shape_ok =
        (K == 768 && (spec_N == 768 || spec_N == 3072)) ||
        (K == 960 && (spec_N == 320 || spec_N == 960 || spec_N == 2560));
    const bool m_ok = (spec_M > 0) && (spec_M <= kKernelMaxM);

    static const bool trace = []{
        const char * env = std::getenv("GGML_XRT_TRACE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    if (trace) {
        fprintf(stderr, "[xrt_trace] K=%ld N=%ld M=%ld shape_ok=%d m_ok=%d dispatch=%d\n",
                (long)K, (long)spec_N, (long)spec_M, (int)shape_ok, (int)m_ok,
                (int)(shape_ok && m_ok));
    }

    if (!shape_ok) return false;
    if (!m_ok)     return false;
    return true;
}

static bool ggml_backend_xrt_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_xrt_device_i = {
    /* .get_name             = */ ggml_backend_xrt_device_get_name,
    /* .get_description      = */ ggml_backend_xrt_device_get_description,
    /* .get_memory           = */ ggml_backend_xrt_device_get_memory,
    /* .get_type             = */ ggml_backend_xrt_device_get_type,
    /* .get_props            = */ ggml_backend_xrt_device_get_props,
    /* .init_backend         = */ ggml_backend_xrt_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xrt_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_xrt_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xrt_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xrt_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_xrt_reg_get_name(ggml_backend_reg_t reg) { GGML_UNUSED(reg); return "XRT"; }
static size_t ggml_backend_xrt_reg_get_device_count(ggml_backend_reg_t reg) { GGML_UNUSED(reg); return 1; }
static ggml_backend_dev_t ggml_backend_xrt_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_xrt_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &dev;
}
static void * ggml_backend_xrt_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg); GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_xrt_reg_i = {
    /* .get_name         = */ ggml_backend_xrt_reg_get_name,
    /* .get_device_count = */ ggml_backend_xrt_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xrt_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xrt_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xrt_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xrt_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

ggml_backend_t ggml_backend_xrt_init(void) {
    auto * dev = ggml_backend_reg_dev_get(ggml_backend_xrt_reg(), 0);
    return ggml_backend_xrt_device_init_backend(dev, nullptr);
}

bool ggml_backend_is_xrt(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_xrt_guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_xrt_reg)
