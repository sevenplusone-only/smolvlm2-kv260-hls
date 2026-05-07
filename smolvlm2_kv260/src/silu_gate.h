// =============================================================================
// silu_gate.h  —  SiLU 激活 + Gate×Up 融合 + 在线量化（流内完成）
// =============================================================================
// 设计原则：
//   ① FFN gate/up GEMM 输出为 N=5120 的拼接流（前 2560 = gate，后 2560 = up）
//   ② SiLU(gate) × up，结果即 FFN 中间激活
//   ③ 为 FFN down GEMM 准备：对结果做 per-token absmax 量化（INT8）
//   ④ 全流水：每 token 扫描两次（gate 缓冲 + up 流式乘），避免 double buffer
//   注意：SiLU(x) = x / (1 + exp(-x))，用分段 LUT 近似（[-8,8] 范围 256 条目）
// =============================================================================
#pragma once
#include "common.h"
#include "approx_math.h"

// ---------------------------------------------------------------------------
// SiLU LUT（分段线性近似，256 条目，INT16 输出，Q1.15 定点）
// ---------------------------------------------------------------------------
static constexpr int SILU_FRAC_BITS = 8;
static constexpr INT32 SILU_Q_ONE   = 1 << SILU_FRAC_BITS;

// 输入 x 为 INT32（GEMM 累加结果），映射到 LUT 索引
static inline INT16 silu_approx(INT32 x) {
#pragma HLS INLINE
    // Input stream is carried in Q8.8-ish fixed point; compress to x/64 domain
    // for the LUT and return Q8.8 output.
    INT32 x_scaled = x >> 2;
    return approx_silu_q8_8(x_scaled);
}

// ---------------------------------------------------------------------------
// silu_gate_fuse_quant
//   输入：gate_up_stream（INT32，N=5120 per token，前 FFN_DIM=gate，后 FFN_DIM=up）
//   输出：quant_stream（INT8，N=2560 per token），scale_stream（INT8 per token）
// ---------------------------------------------------------------------------
void silu_gate_fuse_quant(
    hls::stream<INT32> &gate_up_s,   // FFN gate/up GEMM 输出（5120 per token）
    hls::stream<INT8>  &ffn_act_s,   // 量化后 FFN 激活（2560 per token）
    hls::stream<INT8>  &ffn_scale_s, // per-token scale
    int                 n_tokens
) {
#pragma HLS INLINE off

    // gate 缓冲（INT16，节省 BRAM；gate 先到，up 后到）
    static INT16 gate_buf[FFN_DIM];
#pragma HLS BIND_STORAGE variable=gate_buf type=ram_1p impl=bram
#pragma HLS ARRAY_PARTITION variable=gate_buf cyclic factor=8

    for (int t = 0; t < n_tokens; ++t) {
        HLS_LOOP_TRIPCOUNT(1, 512);

        // Pass 1：读 gate（FFN_DIM=2560），应用 SiLU，存入 gate_buf
    GATE_PASS: for (int n = 0; n < FFN_DIM; ++n) {
#pragma HLS PIPELINE II=1
            INT32 g     = gate_up_s.read();
            INT16 sg    = silu_approx(g);   // SiLU(gate)，Q1.15
            gate_buf[n] = sg;
        }

        // Pass 2：读 up，乘以 gate_buf，同时计算 absmax
        INT32 absmax = 0;
        static INT32 product_buf[FFN_DIM];
#pragma HLS BIND_STORAGE variable=product_buf type=ram_1p impl=bram
#pragma HLS ARRAY_PARTITION variable=product_buf cyclic factor=8

    UP_PASS: for (int n = 0; n < FFN_DIM; ++n) {
#pragma HLS PIPELINE II=1
            INT32 u   = gate_up_s.read();
            // gate_buf[n] is Q8.8, u stays in the INT32 accumulation domain.
            INT32 p   = (INT32)((INT43)gate_buf[n] * u >> SILU_FRAC_BITS);
            product_buf[n] = p;
            INT32 abp = p < 0 ? -p : p;
            if (abp > absmax) absmax = abp;
        }

        // per-token scale
        INT8 tok_sc = (INT8)(absmax / 127 + 1);
        ffn_scale_s.write(tok_sc);

        // Pass 3：量化输出
    QUANT_PASS: for (int n = 0; n < FFN_DIM; ++n) {
#pragma HLS PIPELINE II=1
            INT8 q = (INT8)(product_buf[n] / (INT32)tok_sc);
            ffn_act_s.write(q);
        }
    }
}

// ---------------------------------------------------------------------------
// residual_add
//   将 GEMM 输出（INT32 stream）与原始激活（INT32 URAM 缓冲）相加
//   结果写回 URAM（供下一 layer 使用）
//   "顺带做"：在 Accumulator 输出口后直接 inline，不单独成模块
// ---------------------------------------------------------------------------
void residual_add(
    hls::stream<INT32> &delta_s,   // GEMM 输出（残差增量）
    INT32               res[MAX_L][C], // 原始激活（URAM）
    int                 n_tokens
) {
#pragma HLS INLINE off
    for (int t = 0; t < n_tokens; ++t) {
        HLS_LOOP_TRIPCOUNT(1, 512);
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            INT32 d   = delta_s.read();
            res[t][c] = res[t][c] + d;   // 直接累加
        }
    }
}
