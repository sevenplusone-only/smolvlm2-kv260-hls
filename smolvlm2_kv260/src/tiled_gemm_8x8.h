// =============================================================================
// tiled_gemm_8x8.h  —  Clean-room 8x8x8 W4A8 GEMM primitive
// =============================================================================
// This file intentionally does not reuse the SEC source layout.  It captures the
// useful hardware idea only: a fixed 8x8x8 micro-array with accumulation cycles
// selected by runtime K, and two INT8 multiplies packed into one DSP where the
// tool can infer it.
// =============================================================================
#pragma once
#include "common.h"

static inline INT8 axi_read_i8(const AXI256 *base, int idx) {
#pragma HLS INLINE
    AXI256 word = base[idx >> 5];
    int lane = idx & 31;
    return (INT8)word.range(lane * 8 + 7, lane * 8);
}

static inline INT8 dequant_w4_at(const AXI256 *wgt, const AXI256 *meta, int out_col, int k, int K) {
#pragma HLS INLINE
    int grp = k / GRP;
    int nib_idx = out_col * K + k;
    AXI256 raw_w = wgt[nib_idx >> 6]; // 64 nibbles per AXI256
    int nib_lane = nib_idx & 63;
    ap_uint<4> nib = raw_w.range(nib_lane * 4 + 3, nib_lane * 4);

    int meta_idx = out_col * (K / GRP) + grp;
    AXI256 raw_m = meta[meta_idx >> 4]; // 16 groups per AXI256, 2B/group
    int shift = (meta_idx & 15) * 16;
    INT8 scale = (INT8)raw_m.range(shift + 7, shift);
    INT8 zp    = (INT8)raw_m.range(shift + 15, shift + 8);
    ap_int<9> centered = (ap_int<9>)nib - (ap_int<9>)zp;
    return (INT8)((centered * (ap_int<16>)scale) >> 4);
}

static inline void pack_mul_2int8_array(
    INT8 w0, INT8 w1, INT8 act, INT16 &r0, INT16 &r1
) {
#pragma HLS INLINE
    ap_int<27> packed = (ap_int<27>)(((ap_int<18>)w1 << 9) | (ap_int<9>)w0);
    ap_int<35> product;
#pragma HLS bind_op variable=product op=mul impl=dsp
    product = (ap_int<35>)(packed * (ap_int<8>)act);
    r0 = (INT16)(product.range(8, 0).to_int());
    r1 = (INT16)(product.range(26, 9).to_int());
}

static inline void gemm_micro_8x8x8(
    INT8  act[ARRAY_M][ARRAY_K],
    INT8  wgt[ARRAY_N][ARRAY_K],
    INT32 psum[ARRAY_M][ARRAY_N]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=act complete dim=0
#pragma HLS ARRAY_PARTITION variable=wgt complete dim=0
#pragma HLS ARRAY_PARTITION variable=psum complete dim=0

    for (int m = 0; m < ARRAY_M; ++m) {
#pragma HLS UNROLL
        for (int n = 0; n < ARRAY_N; n += 2) {
#pragma HLS UNROLL
            for (int k = 0; k < ARRAY_K; ++k) {
#pragma HLS UNROLL
                INT16 r0, r1;
                pack_mul_2int8_array(wgt[n][k], wgt[n + 1][k], act[m][k], r0, r1);
                psum[m][n]     += (INT32)r0;
                psum[m][n + 1] += (INT32)r1;
            }
        }
    }
}

static void tiled_gemm_8x8_w4a8(
    const AXI256 *act,
    const AXI256 *wgt,
    const AXI256 *meta,
          AXI256 *out,
    int M,
    int K,
    int N
) {
#pragma HLS INLINE off
    for (int mt = 0; mt < M; mt += ARRAY_M) {
        HLS_LOOP_TRIPCOUNT(1, 128);
        for (int nt = 0; nt < N; nt += ARRAY_N) {
            HLS_LOOP_TRIPCOUNT(1, 6160);

            INT32 psum[ARRAY_M][ARRAY_N];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=0
            for (int m = 0; m < ARRAY_M; ++m) {
#pragma HLS UNROLL
                for (int n = 0; n < ARRAY_N; ++n) {
#pragma HLS UNROLL
                    psum[m][n] = 0;
                }
            }

            for (int kt = 0; kt < K; kt += ARRAY_K) {
#pragma HLS PIPELINE II=1
                HLS_LOOP_TRIPCOUNT(96, 1536);
                INT8 act_tile[ARRAY_M][ARRAY_K];
                INT8 wgt_tile[ARRAY_N][ARRAY_K];
#pragma HLS ARRAY_PARTITION variable=act_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=wgt_tile complete dim=0

                for (int m = 0; m < ARRAY_M; ++m) {
#pragma HLS UNROLL
                    for (int k = 0; k < ARRAY_K; ++k) {
#pragma HLS UNROLL
                        const bool row_valid = (mt + m) < M;
                        const bool k_valid = (kt + k) < K;
                        act_tile[m][k] = (row_valid && k_valid)
                            ? axi_read_i8(act, (mt + m) * K + kt + k)
                            : (INT8)0;
                    }
                }
                for (int n = 0; n < ARRAY_N; ++n) {
#pragma HLS UNROLL
                    for (int k = 0; k < ARRAY_K; ++k) {
#pragma HLS UNROLL
                        const bool col_valid = (nt + n) < N;
                        const bool k_valid = (kt + k) < K;
                        wgt_tile[n][k] = (col_valid && k_valid)
                            ? dequant_w4_at(wgt, meta, nt + n, kt + k, K)
                            : (INT8)0;
                    }
                }

                gemm_micro_8x8x8(act_tile, wgt_tile, psum);
            }

            for (int m = 0; m < ARRAY_M; ++m) {
#pragma HLS PIPELINE II=1
                const bool row_valid = (mt + m) < M;
                AXI256 word = 0;
                for (int n = 0; n < ARRAY_N; ++n) {
#pragma HLS UNROLL
                    word.range(n * 32 + 31, n * 32) = row_valid ? (ap_uint<32>)psum[m][n] : (ap_uint<32>)0;
                }
                if (row_valid) {
                    out[((mt + m) * N + nt) >> 3] = word;
                }
            }
        }
    }
}
