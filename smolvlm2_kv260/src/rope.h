// =============================================================================
// rope.h  —  RoPE 位置编码（LUT 近似）
// =============================================================================
#pragma once
#include "common.h"
#include "approx_math.h"

// ---------------------------------------------------------------------------
// 旋转单对维度（x0, x1） → (x0*cos - x1*sin, x0*sin + x1*cos)
// RoPE 角度和三角函数在 FPGA 侧在线计算
// ---------------------------------------------------------------------------
static inline void rope_rotate_pair(
    INT32   x0, INT32 x1,          // 输入（来自 GEMM 输出，INT32 精度）
    INT16   cos_v, INT16 sin_v,    // Q1.15
    INT32  &y0, INT32 &y1          // 旋转后输出
) {
#pragma HLS INLINE
    INT43 r0 = (INT43)x0 * cos_v - (INT43)x1 * sin_v;
    INT43 r1 = (INT43)x0 * sin_v + (INT43)x1 * cos_v;
    y0 = (INT32)(r0 >> 15);
    y1 = (INT32)(r1 >> 15);
}

// ---------------------------------------------------------------------------
// rope_apply_stream
//   对一个 token 的 Q 或 K 向量应用 RoPE
//   in_s:  来自 GEMM 的 INT32 流（head_dim 宽，一次处理一个 head）
//   out_s: 旋转后 INT32 流
//   pos:   当前 token 位置（用于 LUT 索引）
//   n_heads: Q 为 H_Q=15，K 为 H_KV=5
//   n_tokens: 序列长度
// ---------------------------------------------------------------------------
void rope_apply_stream(
    hls::stream<INT32> &in_s,
    hls::stream<INT32> &out_s,
    int                 pos_start,   // 序列起始位置（Prefill 通常=0，Decode=cur_len）
    int                 n_heads,
    int                 n_tokens
) {
#pragma HLS INLINE off

    for (int t = 0; t < n_tokens; ++t) {
        HLS_LOOP_TRIPCOUNT(1, 512);
        int pos = pos_start + t;

        for (int h = 0; h < n_heads; ++h) {
            HLS_LOOP_TRIPCOUNT(5, 15);

            // 读入一个 head 的 D_HEAD=64 个元素
            INT32 vec[D_HEAD];
#pragma HLS ARRAY_PARTITION variable=vec cyclic factor=4
        LOAD: for (int d = 0; d < D_HEAD; ++d) {
#pragma HLS PIPELINE II=1
                vec[d] = in_s.read();
            }

            // 应用旋转：对 32 对 (d, d+32) 做旋转
        ROTATE: for (int d = 0; d < D_HEAD / 2; ++d) {
#pragma HLS PIPELINE II=1
                INT16 cv, sv;
                approx_rope_sincos_q15(d, pos, cv, sv);
                INT32 y0, y1;
                rope_rotate_pair(vec[d], vec[d + D_HEAD/2], cv, sv, y0, y1);
                vec[d]           = y0;
                vec[d + D_HEAD/2] = y1;
            }

            // 写出
        STORE: for (int d = 0; d < D_HEAD; ++d) {
#pragma HLS PIPELINE II=1
                out_s.write(vec[d]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rope_split_qk
//   将 QKV 拼接流拆分为 Q-stream（旋转后）、K-stream（旋转后）、V-stream（不旋转）
//   输入流：Q[H_Q*D_HEAD] + K[H_KV*D_HEAD] + V[H_KV*D_HEAD] 按 token 排列
// ---------------------------------------------------------------------------
void rope_split_qk(
    hls::stream<INT32> &qkv_in,      // 来自 QKV GEMM 输出
    hls::stream<INT32> &q_rot_out,   // 旋转后 Q
    hls::stream<INT32> &k_rot_out,   // 旋转后 K（用于 attention + KV cache 写入）
    hls::stream<INT32> &v_out,        // V 原样转发（用于 RV BMM）
    int                 pos_start,
    int                 n_tokens
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // 临时 stream：分离 QKV
    static hls::stream<INT32> q_pre("q_pre");
    static hls::stream<INT32> k_pre("k_pre");
#pragma HLS STREAM variable=q_pre depth=FIFO_D
#pragma HLS STREAM variable=k_pre depth=FIFO_D

    // Split loop：从 qkv_in 分发到 q_pre / k_pre / v_out
SPLIT: for (int t = 0; t < n_tokens; ++t) {
        HLS_LOOP_TRIPCOUNT(1, 512);
        // Q 部分（H_Q × D_HEAD）
        for (int i = 0; i < H_Q * D_HEAD; ++i) {
#pragma HLS PIPELINE II=1
            q_pre.write(qkv_in.read());
        }
        // K 部分（H_KV × D_HEAD）
        for (int i = 0; i < H_KV * D_HEAD; ++i) {
#pragma HLS PIPELINE II=1
            k_pre.write(qkv_in.read());
        }
        // V 部分（H_KV × D_HEAD）直通
        for (int i = 0; i < H_KV * D_HEAD; ++i) {
#pragma HLS PIPELINE II=1
            v_out.write(qkv_in.read());
        }
    }

    // 对 Q/K 分别应用 RoPE
    rope_apply_stream(q_pre, q_rot_out, pos_start, H_Q,  n_tokens);
    rope_apply_stream(k_pre, k_rot_out, pos_start, H_KV, n_tokens);
}
