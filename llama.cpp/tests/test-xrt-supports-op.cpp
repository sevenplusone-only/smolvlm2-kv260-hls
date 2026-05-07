#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xrt.h"
#include <cstdio>
#include <cstdlib>

// ggml mul_mat convention: a shape (k, m_outer), b shape (k, n_outer), out shape (m_outer, n_outer).
// In spec terms: m_outer = spec_N (weight rows), n_outer = spec_M (act rows), k = spec_K.
// So for Phase 1 target (spec_M=1024, spec_K=768, spec_N=3072), pass (m=3072, n=1024, k=768).
static bool supports(ggml_backend_dev_t dev, ggml_type ta, ggml_type tb, int64_t m, int64_t n, int64_t k) {
    ggml_init_params p = { 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, ta, k, m);  // weight (k, spec_N)
    ggml_tensor * b = ggml_new_tensor_2d(ctx, tb, k, n);  // act    (k, spec_M)
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    bool r = ggml_backend_dev_supports_op(dev, c);
    ggml_free(ctx);
    return r;
}

int main() {
    // Skip when XRT runtime prerequisites are absent so this test can run under
    // plain ctest outside the FPGA lab. supports_op returns false unconditionally
    // when rt->ready is false, so a positive-case check cannot distinguish
    // "shape rejected" from "runtime unavailable" — gate on the env var instead.
    if (!getenv("GGML_XRT_XCLBIN")) {
        printf("SKIP: GGML_XRT_XCLBIN not set (no FPGA lab prerequisites)\n");
        return 0;
    }

    auto * reg = ggml_backend_xrt_reg();
    auto * dev = ggml_backend_reg_dev_get(reg, 0);

    // Positive case 1 (existing): ViT ffn_up shape (M=1024, K=768, N=3072) → (m=3072, n=1024, k=768)
    if (!supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 3072, 1024, 768)) {
        printf("FAIL: ViT shape (K=768,N=3072,M=1024) rejected (xclbin load failed?)\n"); return 1;
    }

    // Positive case 1b (NEW — Task 2a): ViT attn_q/k/v/out (M=1024, K=768, N=768)
    if (!supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 768, 1024, 768)) {
        printf("FAIL: ViT attn shape (K=768,N=768,M=1024) rejected\n"); return 1;
    }

    // TODO(Task 2b / v2.2 SHIP_K=960): re-enable after xclbin rebuild
    // // Positive case 2: LLM ffn_gate/up (M=512, K=960, N=2560)
    // if (!supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 2560, 512, 960)) {
    //     printf("FAIL: LLM ffn_gate/up shape (K=960,N=2560,M=512) rejected\n"); return 1;
    // }
    // // Positive case 3: LLM at M=1024
    // if (!supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 2560, 1024, 960)) {
    //     printf("FAIL: LLM ffn_gate/up shape (K=960,N=2560,M=1024) rejected\n"); return 1;
    // }

    // Negative: swapped M/N
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 1024, 3072, 768)) {
        printf("FAIL: swapped M/N accepted\n"); return 1;
    }
    // Negative: F32 weight
    if (supports(dev, GGML_TYPE_F32,  GGML_TYPE_F32, 3072, 1024, 768)) {
        printf("FAIL: F32xF32 accepted (should be Q8_0xF32 only)\n"); return 1;
    }
    // Negative: projector shape (not in whitelist)
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 960, 64, 12288)) {
        printf("FAIL: projector shape accepted\n"); return 1;
    }
    // Negative (NEW — step-1): LLM shape but M=100 not 64-aligned
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 2560, 100, 960)) {
        printf("FAIL: LLM shape with unaligned M=100 accepted (M%%64 gate broken)\n"); return 1;
    }
    // Negative (NEW — step-1): ViT shape but M=100 not 64-aligned
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 3072, 100, 768)) {
        printf("FAIL: ViT shape with unaligned M=100 accepted\n"); return 1;
    }
    // Negative (NEW — C-2): LLM shape with M=2048 exceeds kernel M≤1024 cap
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 2560, 2048, 960)) {
        printf("FAIL: LLM shape with M=2048 accepted (exceeds kernel M<=1024 cap)\n"); return 1;
    }
    // Negative (NEW — C-2): ViT shape with M=2048 exceeds kernel M≤1024 cap
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 3072, 2048, 768)) {
        printf("FAIL: ViT shape with M=2048 accepted (exceeds kernel cap)\n"); return 1;
    }
    // Negative (NEW — C-2): LLM at exactly M=1088 (one tile over, 64-aligned)
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 2560, 1088, 960)) {
        printf("FAIL: LLM shape with M=1088 accepted (exceeds cap though aligned)\n"); return 1;
    }
    // Negative (NEW — step-1): attn_q shape (K=960, N=960) NOT in step-1 whitelist
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 960, 512, 960)) {
        printf("FAIL: attn_q shape (K=960,N=960) accepted (not in step-1 whitelist)\n"); return 1;
    }
    // Negative (NEW — step-1): ffn_down shape (K=2560, N=960) NOT in step-1 whitelist
    if (supports(dev, GGML_TYPE_Q8_0, GGML_TYPE_F32, 960, 512, 2560)) {
        printf("FAIL: ffn_down shape (K=2560,N=960) accepted (not in step-1 whitelist)\n"); return 1;
    }

    printf("PASS\n");
    return 0;
}
