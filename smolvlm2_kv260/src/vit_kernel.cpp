// =============================================================================
// vit_kernel.cpp  —  ViT + connector kernels matching config.json dimensions
// =============================================================================
#include "vit_kernel.h"
#include "unified_w4a8_gemm.h"
#include "approx_math.h"

static AXI256 connector_shuffle_buf[(IMAGE_TOKENS * CONNECTOR_IN) / 32];
#pragma HLS BIND_STORAGE variable=connector_shuffle_buf type=ram_t2p impl=uram

static inline INT32 axi_read_i32(const AXI256 *base, int idx) {
#pragma HLS INLINE
    AXI256 word = base[idx >> 3];
    int lane = idx & 7;
    return (INT32)word.range(lane * 32 + 31, lane * 32);
}

static inline void axi_write_i32(AXI256 *base, int idx, INT32 value) {
#pragma HLS INLINE
    int word_idx = idx >> 3;
    int lane = idx & 7;
    AXI256 word = base[word_idx];
    word.range(lane * 32 + 31, lane * 32) = (ap_uint<32>)value;
    base[word_idx] = word;
}

static inline INT16 vit_exp_approx(INT32 x) {
#pragma HLS INLINE
    return approx_exp_q15(x);
}

static void vit_attention_full(
    const AXI256 *qkv,
          AXI256 *out
) {
#pragma HLS INLINE off
    static INT32 scores[VIT_TOKENS];
#pragma HLS BIND_STORAGE variable=scores type=ram_t2p impl=bram

    for (int t = 0; t < VIT_TOKENS; ++t) {
        for (int h = 0; h < VIT_HEADS; ++h) {
            INT32 max_score = -2147483647;
            for (int s = 0; s < VIT_TOKENS; ++s) {
#pragma HLS PIPELINE II=1
                INT43 dot = 0;
                for (int d = 0; d < VIT_HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=8
                    int q_idx = t * VIT_QKV + h * VIT_HEAD_DIM + d;
                    int k_idx = s * VIT_QKV + VIT_C + h * VIT_HEAD_DIM + d;
                    dot += (INT43)axi_read_i32(qkv, q_idx) * axi_read_i32(qkv, k_idx);
                }
                INT32 score = (INT32)(dot >> 3); // 1/sqrt(64)
                scores[s] = score;
                if (score > max_score) max_score = score;
            }

            INT32 denom = 0;
            for (int s = 0; s < VIT_TOKENS; ++s) {
#pragma HLS PIPELINE II=1
                denom += vit_exp_approx(scores[s] - max_score);
            }
            if (denom == 0) denom = 1;

            for (int d = 0; d < VIT_HEAD_DIM; ++d) {
                INT43 acc = 0;
                for (int s = 0; s < VIT_TOKENS; ++s) {
#pragma HLS PIPELINE II=1
                    INT16 p = vit_exp_approx(scores[s] - max_score);
                    int v_idx = s * VIT_QKV + 2 * VIT_C + h * VIT_HEAD_DIM + d;
                    acc += (INT43)p * axi_read_i32(qkv, v_idx);
                }
                int out_idx = t * VIT_C + h * VIT_HEAD_DIM + d;
                axi_write_i32(out, out_idx, (INT32)(acc / denom));
            }
        }
    }
}

static void pixel_shuffle_4x(
    const AXI256 *vit_tokens,
          AXI256 *shuffled
) {
#pragma HLS INLINE off
    // Input token order: 32x32 patches, each 768 channels.
    // Output token order: 8x8 image tokens, each 4x4 neighboring patches
    // concatenated into 12288 channels.
    for (int oy = 0; oy < VIT_PATCH_GRID / PIXEL_SHUFFLE_FACTOR; ++oy) {
        for (int ox = 0; ox < VIT_PATCH_GRID / PIXEL_SHUFFLE_FACTOR; ++ox) {
            int out_tok = oy * (VIT_PATCH_GRID / PIXEL_SHUFFLE_FACTOR) + ox;
            for (int py = 0; py < PIXEL_SHUFFLE_FACTOR; ++py) {
                for (int px = 0; px < PIXEL_SHUFFLE_FACTOR; ++px) {
                    int in_y = oy * PIXEL_SHUFFLE_FACTOR + py;
                    int in_x = ox * PIXEL_SHUFFLE_FACTOR + px;
                    int in_tok = in_y * VIT_PATCH_GRID + in_x;
                    int patch_slot = (py * PIXEL_SHUFFLE_FACTOR + px) * VIT_C;
                    for (int c = 0; c < VIT_C; c += 32) {
#pragma HLS PIPELINE II=1
                        AXI256 word = 0;
                        for (int b = 0; b < 32; ++b) {
#pragma HLS UNROLL
                            INT32 src = axi_read_i32(vit_tokens, in_tok * VIT_C + c + b);
                            INT8 q = (INT8)(src >> 8);
                            word.range(b * 8 + 7, b * 8) = (ap_uint<8>)q;
                        }
                        int out_word = (out_tok * CONNECTOR_IN + patch_slot + c) / 32;
                        shuffled[out_word] = word;
                    }
                }
            }
        }
    }
}

extern "C" void smolvlm2_vit_prefill_kernel(
    const AXI256 *act_in,
    const AXI256 *wgt,
    const AXI256 *meta,
          AXI256 *act_out,
    int           mode,
    int           M,
    int           K,
    int           N,
    int           wgt_offset,
    int           meta_offset
) {
#pragma HLS INTERFACE m_axi port=act_in  bundle=gmem_vact offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=wgt     bundle=gmem_vw   offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=meta    bundle=gmem_vm   offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=act_out bundle=gmem_vout offset=slave max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=mode    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=M       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=K       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=N       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=wgt_offset  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=meta_offset bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return  bundle=ctrl

    const AXI256 *wgt_base = wgt + wgt_offset;
    const AXI256 *meta_base = meta + meta_offset;

    const bool supported_vit_attn =
        mode == MODE_VIT_ATTN && M == VIT_TOKENS && K == VIT_QKV && N == VIT_C;

    const bool supported_mode_shape =
        (mode == MODE_VIT_PATCH && M == VIT_TOKENS && K == VIT_C && N == VIT_C) ||
        (mode == MODE_VIT_QKV   && M == VIT_TOKENS && K == VIT_C && N == VIT_QKV) ||
        (mode == MODE_VIT_FFN   && M == VIT_TOKENS && K == VIT_C && N == VIT_FFN_DIM) ||
        (mode == MODE_VIT_FFN   && M == VIT_TOKENS && K == VIT_FFN_DIM && N == VIT_C);

    const bool supported_generic_gemm =
        mode == MODE_GENERIC_GEMM && is_supported_match_shape(M, K, N);

    if (supported_vit_attn) {
        vit_attention_full(act_in, act_out);
    } else if (supported_mode_shape || supported_generic_gemm) {
        unified_gemm_w4a8_int8_in_int32_out(act_in, wgt_base, meta_base, act_out, M, K, N);
    }
}

extern "C" void smolvlm2_connector_kernel(
    const AXI256 *vit_tokens,
    const AXI256 *projector_wgt,
    const AXI256 *projector_meta,
          AXI256 *image_tokens,
    int           run_projector,
    int           wgt_offset,
    int           meta_offset
) {
#pragma HLS INTERFACE m_axi port=vit_tokens    bundle=gmem_cact offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=projector_wgt bundle=gmem_cw   offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=projector_meta bundle=gmem_cm  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=image_tokens  bundle=gmem_cout offset=slave max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=run_projector bundle=ctrl
#pragma HLS INTERFACE s_axilite port=wgt_offset  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=meta_offset bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    const AXI256 *wgt_base = projector_wgt + wgt_offset;
    const AXI256 *meta_base = projector_meta + meta_offset;

    pixel_shuffle_4x(vit_tokens, connector_shuffle_buf);
    if (run_projector) {
        unified_gemm_w4a8_int8_in_int32_out(
            connector_shuffle_buf,
            wgt_base,
            meta_base,
            image_tokens,
            IMAGE_TOKENS,
            CONNECTOR_IN,
            CONNECTOR_OUT
        );
    } else {
        for (int i = 0; i < (IMAGE_TOKENS * CONNECTOR_OUT) / 32; ++i) {
#pragma HLS PIPELINE II=1
            image_tokens[i] = connector_shuffle_buf[i];
        }
    }
}
