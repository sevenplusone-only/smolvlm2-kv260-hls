// =============================================================================
// unified_w4a8_gemm.h  —  Shared W4A8 GEMM primitive for ViT / connector / decoder
// =============================================================================
#pragma once
#include "common.h"

static inline INT8 axi_read_i8_linear(const AXI256 *base, int idx) {
#pragma HLS INLINE
    AXI256 word = base[idx >> 5];
    int lane = idx & 31;
    return (INT8)word.range(lane * 8 + 7, lane * 8);
}

static inline INT32 axi_read_i32_linear(const AXI256 *base, int idx) {
#pragma HLS INLINE
    AXI256 word = base[idx >> 3];
    int lane = idx & 7;
    return (INT32)word.range(lane * 32 + 31, lane * 32);
}

static inline void axi_write_i32_linear(AXI256 *base, int idx, INT32 value) {
#pragma HLS INLINE
    const int word_idx = idx >> 3;
    const int lane = idx & 7;
    AXI256 word = base[word_idx];
    word.range(lane * 32 + 31, lane * 32) = (ap_uint<32>)value;
    base[word_idx] = word;
}

static inline void pack_mul_2int8_shared(
    INT8  w_lo,
    INT8  w_hi,
    INT8  act,
    INT16 &res_lo,
    INT16 &res_hi
) {
#pragma HLS INLINE
    ap_int<27> packed = (ap_int<27>)(((ap_int<18>)w_hi << 9) | (ap_int<9>)w_lo);
    ap_int<35> product;
#pragma HLS bind_op variable=product op=mul impl=dsp
    product = (ap_int<35>)(packed * (ap_int<8>)act);
    res_lo  = (INT16)(product.range(8, 0).to_int());
    res_hi  = (INT16)(product.range(26, 9).to_int());
}

static inline void dequant_w4_group_shared(
    WgtPack  wgt_pack,
    INT8     scale,
    INT8     zp,
    INT8     out[K_UNROLL]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete
    for (int i = 0; i < K_UNROLL; ++i) {
#pragma HLS UNROLL
        INT4p nibble = wgt_pack.range(i * 4 + 3, i * 4);
        ap_int<9> tmp = (ap_int<9>)nibble - (ap_int<9>)zp;
        out[i] = (INT8)((tmp * (ap_int<16>)scale) >> 4);
    }
}

static inline void mac_tile_shared(
    INT8  act[TILE_M][K_UNROLL],
    INT8  wgt[TILE_N][K_UNROLL],
    INT32 psum[TILE_M][TILE_N]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=act complete dim=0
#pragma HLS ARRAY_PARTITION variable=wgt complete dim=0
#pragma HLS ARRAY_PARTITION variable=psum complete dim=0

    for (int m = 0; m < TILE_M; ++m) {
#pragma HLS UNROLL
        for (int n = 0; n < TILE_N; n += 2) {
#pragma HLS UNROLL
            for (int k = 0; k < K_UNROLL; ++k) {
#pragma HLS UNROLL
                INT16 r0, r1;
                pack_mul_2int8_shared(wgt[n][k], wgt[n + 1][k], act[m][k], r0, r1);
                psum[m][n]     += (INT32)r0;
                psum[m][n + 1] += (INT32)r1;
            }
        }
    }
}

static void unified_gemm_w4a8_int8_in_int32_out(
    const AXI256 *act,
    const AXI256 *wgt,
    const AXI256 *meta,
          AXI256 *out,
    int M,
    int K,
    int N
) {
#pragma HLS INLINE off

    const int k_grps = K / GRP;

    for (int mt = 0; mt < M; mt += TILE_M) {
        HLS_LOOP_TRIPCOUNT(1, 256);
        for (int nt = 0; nt < N; nt += TILE_N) {
            HLS_LOOP_TRIPCOUNT(1, 1024);

            INT32 psum[TILE_M][TILE_N];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=0
            for (int m = 0; m < TILE_M; ++m) {
#pragma HLS UNROLL
                for (int n = 0; n < TILE_N; ++n) {
#pragma HLS UNROLL
                    psum[m][n] = 0;
                }
            }

            for (int kg = 0; kg < k_grps; ++kg) {
#pragma HLS PIPELINE II=1
                HLS_LOOP_TRIPCOUNT(1, 384);

                INT8 act_tile[TILE_M][K_UNROLL];
                INT8 wgt_tile[TILE_N][K_UNROLL];
#pragma HLS ARRAY_PARTITION variable=act_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=wgt_tile complete dim=0

                for (int m = 0; m < TILE_M; ++m) {
#pragma HLS UNROLL
                    for (int k = 0; k < K_UNROLL; ++k) {
#pragma HLS UNROLL
                        const bool row_valid = (mt + m) < M;
                        act_tile[m][k] = row_valid
                            ? axi_read_i8_linear(act, (mt + m) * K + kg * GRP + k)
                            : (INT8)0;
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
#pragma HLS UNROLL factor=2
                    const bool col_valid = (nt + n) < N;
                    if (!col_valid) {
                        for (int k = 0; k < K_UNROLL; ++k) {
#pragma HLS UNROLL
                            wgt_tile[n][k] = 0;
                        }
                        continue;
                    }

                    const int gn = nt + n;
                    const int wgt_off = gn * k_grps + kg;
                    const int meta_off = wgt_off;
                    AXI256 raw_w = wgt[wgt_off >> 1];
                    AXI256 raw_m = meta[meta_off >> 4];
                    WgtPack wpack = ((wgt_off & 1) == 0)
                        ? (WgtPack)raw_w.range(127, 0)
                        : (WgtPack)raw_w.range(255, 128);

                    const int meta_shift = (meta_off & 15) * 16;
                    INT8 sc = (INT8)raw_m.range(meta_shift + 7, meta_shift);
                    INT8 zp = (INT8)raw_m.range(meta_shift + 15, meta_shift + 8);
                    dequant_w4_group_shared(wpack, sc, zp, wgt_tile[n]);
                }

                mac_tile_shared(act_tile, wgt_tile, psum);
            }

            for (int m = 0; m < TILE_M; ++m) {
#pragma HLS PIPELINE II=1
                if ((mt + m) >= M) {
                    continue;
                }
                for (int n = 0; n < TILE_N && (nt + n) < N; ++n) {
                    axi_write_i32_linear(out, (mt + m) * N + nt + n, psum[m][n]);
                }
            }
        }
    }
}
