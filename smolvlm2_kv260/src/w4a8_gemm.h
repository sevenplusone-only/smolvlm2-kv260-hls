// =============================================================================
// w4a8_gemm.h  —  W4A8 GEMM 引擎（独立设计，不照抄参考代码）
// =============================================================================
// 核心思路：
//   ① 权重以 W4 存储（128-bit/32 nibble per AXI 读），在线解包为 INT8
//   ② pack_mul_2int8：一个 DSP48E2 同时完成 2×INT8 MAC
//      原理：将 w1,w2 打包为 27-bit (w2<<9 | w1)，与 act_val 相乘，
//             低位取 w1*act，高位右移取 w2*act，结果一拍出
//   ③ 外层 M/N tile 循环，内层 K=32 完全展开（GRP=32 = 一组 scale）
//   ④ GroupMeta 每组 1 拍读取，与权重加载 overlap（scales 独立 AXI bundle）
// =============================================================================
#pragma once
#include "common.h"
#include "unified_w4a8_gemm.h"
#include "approx_math.h"

// ---------------------------------------------------------------------------
// pack_mul_2int8
//   将 (w_hi, w_lo) 两个 INT8 权重打包进一个 27-bit 大操作数，
//   与 act (INT8) 共享一次 DSP48E2 乘法，输出两个乘积
// ---------------------------------------------------------------------------
static inline void pack_mul_2int8(
    INT8     w_lo,   // 权重 0
    INT8     w_hi,   // 权重 1
    INT8     act,    // 共享激活值
    INT16   &res_lo, // 输出 w_lo × act
    INT16   &res_hi  // 输出 w_hi × act
) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    pack_mul_2int8_shared(w_lo, w_hi, act, res_lo, res_hi);
}

// ---------------------------------------------------------------------------
// dequant_w4_group
//   将 128-bit 权重包（32 个 INT4）解包为 INT8，应用 per-group scale/zero
//   完全组合逻辑（LUTRAM 实现），与权重加载路径 overlap
// ---------------------------------------------------------------------------
static inline void dequant_w4_group(
    WgtPack  wgt_pack,          // 128-bit，含 32 个 INT4
    INT8     scale,             // per-group scale
    INT8     zp,                // per-group zero_point
    INT8     out[K_UNROLL]      // 解包输出 32 个 INT8
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete
    dequant_w4_group_shared(wgt_pack, scale, zp, out);
}

// ---------------------------------------------------------------------------
// w4a8_mac_unit
//   处理一个 K_UNROLL=32 大小的内层 K-tile
//   输入：激活向量 act[32]，解包权重 wgt[32]（已 dequant）
//   输出：累积到 psum[TM][TN] 中（部分和）
// ---------------------------------------------------------------------------
static inline void w4a8_mac_unit(
    INT8     act[TILE_M][K_UNROLL],   // M-tile 激活
    INT8     wgt[TILE_N][K_UNROLL],   // N-tile 权重（转置后布局）
    INT32    psum[TILE_M][TILE_N]     // 累加目标
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=act  complete dim=0
#pragma HLS ARRAY_PARTITION variable=wgt  complete dim=0
#pragma HLS ARRAY_PARTITION variable=psum complete dim=0

    for (int m = 0; m < TILE_M; ++m) {
#pragma HLS UNROLL
        for (int n = 0; n < TILE_N; n += 2) {
#pragma HLS UNROLL
            for (int k = 0; k < K_UNROLL; ++k) {
#pragma HLS UNROLL
                INT16 r0, r1;
                pack_mul_2int8_shared(wgt[n][k], wgt[n+1][k], act[m][k], r0, r1);
                psum[m][n]   += (INT32)r0;
                psum[m][n+1] += (INT32)r1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// §1  RMSNorm + 激活量化（流水级）
//   输入：raw activation stream（INT32，定点）
//   输出：量化后 INT8 stream + per-token scale（INT8）
//   设计：三阶段流水：①累计平方和 ②sqrt+归一化 ③absmax量化
//   全部在 PL 内，不出片，与下游 GEMM DATAFLOW 连接
// ---------------------------------------------------------------------------
static void rmsnorm_quant(
    hls::stream<INT32> &in_s,          // 原始激活（hidden_size 宽）
    const INT8          gamma[C],       // RMSNorm 可学习参数（W8 量化）
    hls::stream<INT8>  &out_s,          // 量化输出
    hls::stream<INT8>  &scale_s,        // per-token scale（1 个值/token）
    int                 seq_len
) {
#pragma HLS INLINE off
    static INT32 buf[C];
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=8

    for (int t = 0; t < seq_len; ++t) {
#pragma HLS PIPELINE off   // 外层 token 循环不 pipeline

        // Pass 1：读入并累计平方和
        INT43 sq_sum = 0;
    PASS1: for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            INT32 x = in_s.read();
            buf[c]  = x;
            sq_sum += (INT43)((ap_int<32>)x * x);
        }

        // Pass 2：计算 rms_inv（Q1.15），完全基于整数规格化 + LUT。
        // x_real = x_int / 256，因此 mean(x_real^2) = rms2_q16 / 2^16。
        // 1 / sqrt(mean(x_real^2)) = 2^8 / sqrt(rms2_q16)。
        ap_uint<32> rms2_q16 = (ap_uint<32>)((sq_sum + (INT43)(C / 2)) / (INT43)C);
        if (rms2_q16 == 0) {
            rms2_q16 = 1;
        }
        INT32 rms_inv_q23 = (INT32)approx_rsqrt_q15(rms2_q16) << 8;

        // Pass 3：归一化 + 乘 gamma + absmax 量化
        INT32 absmax = 0;
        static INT32 norm_buf[C];
#pragma HLS ARRAY_PARTITION variable=norm_buf cyclic factor=8
    PASS3: for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            INT32 normed    = (INT32)(((INT43)buf[c] * rms_inv_q23) >> 15);
            // gamma is exported as Q6.0 with unit value ≈ 64.
            INT32 scaled    = (INT32)((ap_int<32>)normed * gamma[c] >> 6);
            norm_buf[c]     = scaled;
            INT32 abval     = scaled < 0 ? -scaled : scaled;
            if (abval > absmax) absmax = abval;
        }

        // 计算 per-token scale（absmax / 127，结果作为 INT8 scale factor）
        INT8 tok_scale = (INT8)(absmax / 127 + 1);
        scale_s.write(tok_scale);

        // Pass 4：量化输出
    PASS4: for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            INT8 q = (INT8)(norm_buf[c] / (INT32)tok_scale);
            out_s.write(q);
        }
    }
}

// ---------------------------------------------------------------------------
// §2  W4A8 GEMM 核心
//   支持任意 M×K×N，以 TILE_M × TILE_N 分块处理
//   权重从 DDR（ap_uint<256>*，双半端口）读取
//   激活从 stream（已经由 RMSNorm+Quant 发出）
//   输出到 stream（INT32 psum，等待 Accumulator/scale 归一化）
// ---------------------------------------------------------------------------
void w4a8_gemm(
    hls::stream<INT8>   &act_s,         // 来自 rmsnorm_quant
    hls::stream<INT8>   &act_scale_s,   // per-token scale
    const AXI256        *wgt_half0,     // W4 权重前半（HPC0）
    const AXI256        *wgt_half1,     // W4 权重后半（HPC1）
    const AXI256        *meta_half0,    // GroupMeta 前半（HPC1）
    const AXI256        *meta_half1,    // GroupMeta 后半（HPC1）
    hls::stream<INT32>  &out_s,         // 输出到 Accumulator / 残差加
    int M, int K, int N                 // 运行时形状（K 必须为 GRP 整数倍）
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE m_axi port=wgt_half0 bundle=gmem_w0 max_read_burst_length=16 offset=slave
#pragma HLS INTERFACE m_axi port=wgt_half1 bundle=gmem_w1 max_read_burst_length=16 offset=slave
#pragma HLS INTERFACE m_axi port=meta_half0 bundle=gmem_m0 max_read_burst_length=16 offset=slave
#pragma HLS INTERFACE m_axi port=meta_half1 bundle=gmem_m1 max_read_burst_length=16 offset=slave

    // -----------------------------------------------------------------------
    // 片上激活缓冲（URAM，双端口，256-bit 宽，不跨 URAM 端口）
    // -----------------------------------------------------------------------
    static INT8 act_buf[MAX_L][MAX_DEC_GEMM_K];
#pragma HLS BIND_STORAGE variable=act_buf type=ram_t2p impl=uram
#pragma HLS ARRAY_PARTITION variable=act_buf cyclic factor=32 dim=2

    // 读入激活到片上缓冲（stream → URAM）
    for (int t = 0; t < M; ++t) {
        for (int c = 0; c < K; ++c) {
#pragma HLS PIPELINE II=1
            HLS_LOOP_TRIPCOUNT(960, 2560);
            act_buf[t][c] = act_s.read();
        }
    }

    // per-token scale 存入寄存器数组
    INT8 tok_scales[MAX_L];
    for (int t = 0; t < M; ++t) {
#pragma HLS PIPELINE II=1
        tok_scales[t] = act_scale_s.read();
    }

    // -----------------------------------------------------------------------
    // 外层 N-tile 循环（列分块）
    // -----------------------------------------------------------------------
    int n_tiles = (N + TILE_N - 1) / TILE_N;
    int k_grps  = K / GRP;            // K 必须是 GRP 整数倍

    for (int nt = 0; nt < n_tiles; ++nt) {
        HLS_LOOP_TRIPCOUNT(1, 80);    // N_QKV/TILE_N=25

        int n_base = nt * TILE_N;
        int n_len  = (n_base + TILE_N <= N) ? TILE_N : (N - n_base);

        // -----------------------------------------------------------------------
        // 外层 M-tile 循环（行分块，适配 URAM 访问）
        // -----------------------------------------------------------------------
        for (int mt = 0; mt < M; mt += TILE_M) {
            HLS_LOOP_TRIPCOUNT(1, 128);

            int m_len = (mt + TILE_M <= M) ? TILE_M : (M - mt);

            // 初始化部分和
            INT32 psum[TILE_M][TILE_N];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=0
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n) {
#pragma HLS UNROLL
                    psum[m][n] = 0;
                }

            // -------------------------------------------------------------------
            // 内层 K-group 循环
            // -------------------------------------------------------------------
            for (int kg = 0; kg < k_grps; ++kg) {
#pragma HLS PIPELINE II=1
                HLS_LOOP_TRIPCOUNT(30, 320);

                int k_base = kg * GRP;

                // 读取权重 nibble pack（128-bit per N-column per group）
                // 布局：wgt[n][k_group] → 一次 128-bit 读 = 32 nibble
                // 双半端口交错：n 偶数 → half0，n 奇数 → half1
                INT8 wgt_dq[TILE_N][K_UNROLL];
#pragma HLS ARRAY_PARTITION variable=wgt_dq complete dim=0

                for (int n = 0; n < n_len; ++n) {
#pragma HLS UNROLL factor=2
                    int   gn      = n_base + n;
                    // 权重在 DDR 的偏移：按 (n_col, k_group) 排列
                    // 每组 = 128-bit = 16 byte = 1 个 AXI256 的低半
                    int   wgt_off = (gn * k_grps + kg);
                    // meta 偏移：每组 2 byte（scale + zp）
                    int   meta_off = wgt_off;

                    AXI256 raw_w, raw_m;
                    if ((gn & 1) == 0) {
                        raw_w = wgt_half0[wgt_off >> 1];  // 256-bit 读，取低/高 128
                        raw_m = meta_half0[meta_off >> 4]; // 256-bit 含 16 组 meta
                    } else {
                        raw_w = wgt_half1[wgt_off >> 1];
                        raw_m = meta_half1[meta_off >> 4];
                    }

                    // 取 128-bit 低/高
                    WgtPack wpack;
                    if ((wgt_off & 1) == 0)
                        wpack = raw_w.range(127, 0);
                    else
                        wpack = raw_w.range(255, 128);

                    // GroupMeta：每 256-bit AXI 含 16 组（每组 2 byte）
                    int meta_shift = (meta_off % 16) * 16;
                    INT8 sc = (INT8)raw_m.range(meta_shift+7,  meta_shift);
                    INT8 zp = (INT8)raw_m.range(meta_shift+15, meta_shift+8);

                    // W4 → INT8 dequant（组合逻辑，LUTRAM）
                    dequant_w4_group(wpack, sc, zp, wgt_dq[n]);
                }

                // M-tile 激活加载（从 URAM）
                INT8 act_tile[TILE_M][K_UNROLL];
#pragma HLS ARRAY_PARTITION variable=act_tile complete dim=0
                for (int m = 0; m < m_len; ++m) {
#pragma HLS UNROLL
                    for (int k = 0; k < K_UNROLL; ++k) {
#pragma HLS UNROLL
                        act_tile[m][k] = act_buf[mt + m][k_base + k];
                    }
                }

                // MAC
                w4a8_mac_unit(act_tile, wgt_dq, psum);
            } // kg

            // -------------------------------------------------------------------
            // 输出：psum 做 scale 归一化（INT32 → 输出 stream）
            // per-element 反量化：out = psum * w_scale * a_scale / 2^shift
            // 这里简化：直接把 psum 写出，Accumulator 层统一处理 scale
            // -------------------------------------------------------------------
            for (int m = 0; m < m_len; ++m) {
                for (int n = 0; n < n_len; ++n) {
#pragma HLS PIPELINE II=1
                    out_s.write((INT32)psum[m][n] * (INT32)tok_scales[mt + m]);
                }
            }
        } // mt
    } // nt
}
