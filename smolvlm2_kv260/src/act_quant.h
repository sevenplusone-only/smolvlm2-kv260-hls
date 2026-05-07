// =============================================================================
// act_quant.h  —  Activation-only dynamic quantization for projection inputs
// =============================================================================
#pragma once
#include "common.h"

static void act_quant(
    hls::stream<INT32> &in_s,
    hls::stream<INT8>  &out_s,
    hls::stream<INT8>  &scale_s,
    int                 rows,
    int                 cols
) {
#pragma HLS INLINE off
    static INT32 row_buf[MAX_DEC_GEMM_K];
#pragma HLS BIND_STORAGE variable=row_buf type=ram_t2p impl=uram
#pragma HLS ARRAY_PARTITION variable=row_buf cyclic factor=32

    for (int r = 0; r < rows; ++r) {
        INT32 absmax = 0;
        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            HLS_LOOP_TRIPCOUNT(960, 2560);
            INT32 v = in_s.read();
            row_buf[c] = v;
            INT32 a = v < 0 ? -v : v;
            if (a > absmax) absmax = a;
        }

        INT8 scale = (INT8)(absmax / 127 + 1);
        scale_s.write(scale);

        for (int c = 0; c < cols; ++c) {
#pragma HLS PIPELINE II=1
            HLS_LOOP_TRIPCOUNT(960, 2560);
            out_s.write((INT8)(row_buf[c] / (INT32)scale));
        }
    }
}
