#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-xrt.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Phase 1 L2 gate: XRT MUL_MAT numerical parity with CPU on ViT shapes.
// Step-1 (v2.1 overlay, SHIP_K=768): ViT ffn_up + attn only.
// Step-2 (v2.2 overlay, SHIP_K=960): re-enable LLM K=960 shapes.
// Skips when FPGA prerequisites are absent so ctest can run without the lab.

int main() {
    if (!getenv("GGML_XRT_XCLBIN")) {
        printf("SKIP: GGML_XRT_XCLBIN not set (no FPGA lab prerequisites)\n");
        return 0;
    }

    struct shape_case { int64_t M, K, N; const char * label; };
    const shape_case cases[] = {
        {1024, 768, 3072, "ViT ffn_up"},
        {1024, 768,  768, "ViT attn_q/k/v/out"},
        // TODO(Task 2b / v2.2 SHIP_K=960): re-enable after xclbin rebuild
        // { 512, 960, 2560, "LLM ffn_gate/up (M=512)"},
        // {1024, 960, 2560, "LLM ffn_gate/up (M=1024, at cap)"},
    };
    const float TOL = 2e-3f;

    ggml_backend_t xrt_bk = ggml_backend_xrt_init();
    ggml_backend_t cpu_bk = ggml_backend_cpu_init();
    if (!xrt_bk || !cpu_bk) { printf("FAIL: backend init\n"); return 1; }

    int pass_count = 0;
    for (const auto & sc : cases) {
        const int64_t M = sc.M, K = sc.K, N = sc.N;
        printf("--- shape %s: M=%lld K=%lld N=%lld ---\n",
               sc.label, (long long)M, (long long)K, (long long)N);

        ggml_init_params p = { /*.mem_size=*/ 64*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
        ggml_context * ctx = ggml_init(p);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);
        ggml_tensor * out = ggml_mul_mat(ctx, w, a);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        if (!buf) { printf("FAIL: backend_alloc_ctx_tensors\n"); return 1; }

        std::mt19937 rng((uint32_t)(0xC0FFEE ^ (M * 131 + K * 17 + N)));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> a_f32((size_t)M * K);
        for (auto & x : a_f32) x = dist(rng);
        std::vector<float> w_f32((size_t)N * K);
        for (auto & x : w_f32) x = dist(rng);
        std::memcpy(a->data, a_f32.data(), a_f32.size() * sizeof(float));

        const size_t row_bytes = (size_t)(K / 32) * sizeof(block_q8_0);
        for (int64_t n = 0; n < N; ++n) {
            quantize_row_q8_0_ref(&w_f32[(size_t)n * K],
                                  (block_q8_0 *)((uint8_t *)w->data + (size_t)n * row_bytes),
                                  K);
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        const size_t out_bytes = (size_t)M * N * sizeof(float);

        std::memset(out->data, 0xAA, out_bytes);
        if (ggml_backend_graph_compute(xrt_bk, gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: XRT graph_compute for %s\n", sc.label); return 1;
        }
        std::vector<float> out_xrt((size_t)M * N);
        std::memcpy(out_xrt.data(), out->data, out_bytes);

        std::memset(out->data, 0xAA, out_bytes);
        if (ggml_backend_graph_compute(cpu_bk, gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: CPU graph_compute for %s\n", sc.label); return 1;
        }
        std::vector<float> out_cpu((size_t)M * N);
        std::memcpy(out_cpu.data(), out->data, out_bytes);

        float max_abs_cpu = 0.0f, max_abs_err = 0.0f;
        size_t argmax_err = 0;
        for (size_t i = 0; i < out_cpu.size(); ++i) {
            float a_cpu = std::fabs(out_cpu[i]);
            float err   = std::fabs(out_xrt[i] - out_cpu[i]);
            if (a_cpu > max_abs_cpu) max_abs_cpu = a_cpu;
            if (err   > max_abs_err) { max_abs_err = err; argmax_err = i; }
        }
        float rel = max_abs_err / (max_abs_cpu > 0.0f ? max_abs_cpu : 1.0f);
        printf("max_abs_cpu=%g max_abs_err=%g rel=%g at idx=%zu (xrt=%g cpu=%g)\n",
               max_abs_cpu, max_abs_err, rel, argmax_err, out_xrt[argmax_err], out_cpu[argmax_err]);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        if (rel > TOL) { printf("FAIL: %s rel %g > tol %g\n", sc.label, rel, TOL); return 1; }
        ++pass_count;
        printf("PASS: %s\n", sc.label);
    }

    ggml_backend_free(cpu_bk);
    ggml_backend_free(xrt_bk);

    const int total = (int)(sizeof(cases) / sizeof(cases[0]));
    printf("ALL PASS (%d/%d shapes)\n", pass_count, total);
    return 0;
}
