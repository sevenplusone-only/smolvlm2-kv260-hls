// =============================================================================
// decoder_kernel.cpp  —  SmolVLM2-500M Decoder Layer 顶层 HLS Kernel
// =============================================================================
// 结构总览：
//   extern "C" void smolvlm2_decoder_layer(...)
//     #pragma HLS DATAFLOW
//     §1  rmsnorm_quant        ← attention pre-norm + 激活量化
//     §2  w4a8_gemm (QKV)     ← Q[960]+K[320]+V[320] = N=1600，一次扫描
//     §3  rope_split_qk        ← Q/K 旋转，V 直通
//     §4  multi_head_attention ← Flash Attn + KV Cache Ping-Pong
//     §5  w4a8_gemm (O proj)  ← N=960，O 投影
//     §6  residual_add (1)    ← O proj + 原始激活残差
//     §7  rmsnorm_quant (FFN) ← FFN pre-norm
//     §8  w4a8_gemm (gate/up) ← N=5120，gate/up 拼接权重
//     §9  silu_gate_fuse_quant ← SiLU(gate)×up + 量化
//     §10 w4a8_gemm (down)    ← N=960
//     §11 residual_add (2)    ← down proj + §6 输出残差
//     [optional] §12 lm_head  ← N=49280，M=1
// =============================================================================
#include "decoder_kernel.h"
#include "w4a8_gemm.h"
#include "rope.h"
#include "attention.h"
#include "silu_gate.h"
#include "act_quant.h"

// =============================================================================
// 内部辅助：激活 DDR↔片上 URAM 的加载 / 存储
// =============================================================================

// URAM 激活主缓冲（两层：res_buf_0 存前层残差，res_buf_1 存 FFN 前残差）
static INT32 res_buf_0[MAX_L][C];  // attention 残差基础
static INT32 res_buf_1[MAX_L][C];  // FFN 残差基础
#pragma HLS BIND_STORAGE variable=res_buf_0 type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=res_buf_1 type=ram_t2p impl=uram
#pragma HLS ARRAY_PARTITION variable=res_buf_0 cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=res_buf_1 cyclic factor=4 dim=2

// gamma 片上缓冲（BRAM，C=960 × INT8 = 1KB，1 BRAM-36K 即可）
static INT8 gamma_attn_buf[C];
static INT8 gamma_ffn_buf[C];
#pragma HLS BIND_STORAGE variable=gamma_attn_buf type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=gamma_ffn_buf  type=rom_1p impl=bram

// ---------------------------------------------------------------------------
// load_act：DDR → URAM（Ping-Pong 选择）
// ---------------------------------------------------------------------------
static void load_act(
    const AXI256 *act_ping,
    const AXI256 *act_pong,
    bool          use_ping,
    int           seq_len,
    int           token_offset,
    int           token_count
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE m_axi port=act_ping bundle=gmem_ap max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=act_pong bundle=gmem_ap max_read_burst_length=16

    const AXI256 *src = use_ping ? act_ping : act_pong;

    (void)seq_len;
    for (int t = 0; t < token_count; ++t) {
        for (int c = 0; c < C; c += 32) {
#pragma HLS PIPELINE II=1
            int src_word = ((token_offset + t) * C + c) / 32;
            AXI256 word = src[src_word];
            for (int b = 0; b < 32; ++b) {
#pragma HLS UNROLL
                // Keep residual buffers in Q8.8 so RMSNorm / residual paths share
                // one internal numeric convention.
                res_buf_0[token_offset + t][c + b] = (INT32)((INT8)word.range(b*8+7, b*8)) << 8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// load_gamma：DDR → BRAM（C=960 INT8，960/32=30 个 AXI256 读）
// ---------------------------------------------------------------------------
static void load_gamma(
    const AXI256 *gamma_ddr,
    INT8          gamma_buf[C]
) {
#pragma HLS INLINE off
    for (int v = 0; v < C / 32; ++v) {
#pragma HLS PIPELINE II=1
        AXI256 word = gamma_ddr[v];
        for (int b = 0; b < 32; ++b) {
#pragma HLS UNROLL
            gamma_buf[v * 32 + b] = (INT8)word.range(b*8+7, b*8);
        }
    }
}

// ---------------------------------------------------------------------------
// store_act：URAM → DDR（输出到 act_out）
// ---------------------------------------------------------------------------
static void store_act(
    AXI256       *act_out,
    int           seq_len,
    int           token_offset,
    int           token_count
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE m_axi port=act_out bundle=gmem_ao max_write_burst_length=16

    (void)seq_len;
    for (int t = 0; t < token_count; ++t) {
        for (int c = 0; c < C; c += 32) {
#pragma HLS PIPELINE II=1
            AXI256 word = 0;
            for (int b = 0; b < 32; ++b) {
#pragma HLS UNROLL
                INT8 val = (INT8)(res_buf_1[token_offset + t][c + b] >> 8); // 截断高精度
                word.range(b*8+7, b*8) = (ap_uint<8>)val;
            }
            int dst_word = ((token_offset + t) * C + c) / 32;
            act_out[dst_word] = word;
        }
    }
}

// ---------------------------------------------------------------------------
// res_to_stream：将 URAM 残差缓冲转为 hls::stream（供 rmsnorm_quant 输入）
// ---------------------------------------------------------------------------
static void res_to_stream(
    INT32              src[MAX_L][C],
    hls::stream<INT32> &out_s,
    int                 seq_len,
    int                 token_offset,
    int                 token_count
) {
#pragma HLS INLINE off
    (void)seq_len;
    for (int t = 0; t < token_count; ++t) {
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            out_s.write(src[token_offset + t][c]);
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_out_to_res：将 GEMM 输出流（INT32）写回 URAM 残差缓冲
// ---------------------------------------------------------------------------
static void stream_to_res(
    hls::stream<INT32> &in_s,
    INT32               dst[MAX_L][C],
    int                 seq_len
) {
#pragma HLS INLINE off
    for (int t = 0; t < seq_len; ++t) {
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            dst[t][c] = in_s.read();
        }
    }
}

static void residual_add_range(
    hls::stream<INT32> &delta_s,
    INT32               res[MAX_L][C],
    int                 token_offset,
    int                 token_count
) {
#pragma HLS INLINE off
    for (int t = 0; t < token_count; ++t) {
        HLS_LOOP_TRIPCOUNT(1, 512);
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            INT32 d = delta_s.read();
            res[token_offset + t][c] = res[token_offset + t][c] + d;
        }
    }
}

// ---------------------------------------------------------------------------
// o_proj_scatter：将 attention 输出（H_Q × D_HEAD per token）
//                 重排为 C=960 维连续流（供 O proj GEMM 输入）
// ---------------------------------------------------------------------------
static void o_scatter(
    hls::stream<INT32> &o_head_s,   // flash_attention 输出（head 顺序）
    hls::stream<INT32> &o_flat_s,   // 重排后（token 顺序，C=960）
    int                 seq_len
) {
#pragma HLS INLINE off
    // 简化版：attention 输出已经是 token-major 顺序，直接转发
    // 实际需处理 head-interleave → token-major 重排
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < H_Q * D_HEAD; ++i) {
#pragma HLS PIPELINE II=1
            o_flat_s.write(o_head_s.read());
        }
    }
}

// =============================================================================
// 顶层 Kernel
// =============================================================================
extern "C" void smolvlm2_decoder_layer(
    const AXI256  *act_ping,
    const AXI256  *act_pong,
          AXI256  *act_out,
    const AXI256  *wgt_half0,
    const AXI256  *wgt_half1,
    const AXI256  *meta_half0,
    const AXI256  *meta_half1,
          AXI256  *k_cache_ddr,
          AXI256  *v_cache_ddr,
    const AXI256  *gamma_attn,
    const AXI256  *gamma_ffn,
    const AXI256  *lmhead_wgt,
    const AXI256  *lmhead_meta,
          AXI256  *lmhead_out,
    int            seq_len,
    int            kv_len,
    int            pos_start,
    int            exec_mode,
    int            token_tile_offset,
    int            token_tile_size,
    int            run_lmhead,
    int            layer_id,
    int            use_ping,
    int            wgt_offset,
    int            meta_offset,
    int            causal,
    int            num_heads,
    int            num_kv_heads
) {
// ---------------------------------------------------------------------------
// AXI 接口声明（避免 II 崩塌的关键：必须 ap_uint<256>*，不能 int8_t*）
// ---------------------------------------------------------------------------
#pragma HLS INTERFACE m_axi port=act_ping    bundle=gmem_ap  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=act_pong    bundle=gmem_ap  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=act_out     bundle=gmem_ao  offset=slave max_write_burst_length=16
#pragma HLS INTERFACE m_axi port=wgt_half0   bundle=gmem_w0  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=wgt_half1   bundle=gmem_w1  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=meta_half0  bundle=gmem_m0  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=meta_half1  bundle=gmem_m1  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=k_cache_ddr bundle=gmem_kc  offset=slave max_read_burst_length=16 max_write_burst_length=8
#pragma HLS INTERFACE m_axi port=v_cache_ddr bundle=gmem_vc  offset=slave max_read_burst_length=16 max_write_burst_length=8
#pragma HLS INTERFACE m_axi port=gamma_attn  bundle=gmem_g0  offset=slave max_read_burst_length=4
#pragma HLS INTERFACE m_axi port=gamma_ffn   bundle=gmem_g1  offset=slave max_read_burst_length=4
#pragma HLS INTERFACE m_axi port=lmhead_wgt  bundle=gmem_lw  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=lmhead_meta bundle=gmem_lm  offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=lmhead_out  bundle=gmem_lo  offset=slave max_write_burst_length=16

// AXI Lite 控制接口（PS 轮询 ap_done/ap_idle）
#pragma HLS INTERFACE s_axilite port=seq_len    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=kv_len     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=pos_start  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=exec_mode bundle=ctrl
#pragma HLS INTERFACE s_axilite port=token_tile_offset bundle=ctrl
#pragma HLS INTERFACE s_axilite port=token_tile_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=run_lmhead bundle=ctrl
#pragma HLS INTERFACE s_axilite port=layer_id   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=use_ping   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=wgt_offset bundle=ctrl
#pragma HLS INTERFACE s_axilite port=meta_offset bundle=ctrl
#pragma HLS INTERFACE s_axilite port=causal     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_heads  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_kv_heads bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return     bundle=ctrl

// ---------------------------------------------------------------------------
// DATAFLOW 区域
// ---------------------------------------------------------------------------
#pragma HLS DATAFLOW

    // -----------------------------------------------------------------------
    // 内部 hls::stream 连接管道（DATAFLOW 节点间通信）
    // -----------------------------------------------------------------------

    // §1 RMSNorm → §2 QKV GEMM
    static hls::stream<INT32> s_raw_act("s_raw_act");
    static hls::stream<INT8>  s_norm1_q("s_norm1_q");
    static hls::stream<INT8>  s_norm1_sc("s_norm1_sc");
#pragma HLS STREAM variable=s_raw_act  depth=FIFO_D
#pragma HLS STREAM variable=s_norm1_q  depth=FIFO_D
#pragma HLS STREAM variable=s_norm1_sc depth=FIFO_D

    // §2 QKV GEMM → §3 RoPE
    static hls::stream<INT32> s_qkv_out("s_qkv_out");
#pragma HLS STREAM variable=s_qkv_out depth=FIFO_ATTN

    // §3 RoPE → §4 Attention
    static hls::stream<INT32> s_q_rot("s_q_rot");
    static hls::stream<INT32> s_k_rot("s_k_rot");
    static hls::stream<INT32> s_v_pass("s_v_pass");
#pragma HLS STREAM variable=s_q_rot  depth=FIFO_ATTN
#pragma HLS STREAM variable=s_k_rot  depth=FIFO_D
#pragma HLS STREAM variable=s_v_pass depth=FIFO_D

    // §4 Attention → §5 O proj
    static hls::stream<INT32> s_attn_o("s_attn_o");
    static hls::stream<INT32> s_o_flat("s_o_flat");
    static hls::stream<INT8>  s_o_q("s_o_q");
    static hls::stream<INT8>  s_o_sc("s_o_sc");
#pragma HLS STREAM variable=s_attn_o depth=FIFO_ATTN
#pragma HLS STREAM variable=s_o_flat depth=FIFO_D
#pragma HLS STREAM variable=s_o_q    depth=FIFO_D
#pragma HLS STREAM variable=s_o_sc   depth=FIFO_D

    // §5 O proj → §6 残差（写回 URAM，不走 stream）
    static hls::stream<INT32> s_oproj_out("s_oproj_out");
#pragma HLS STREAM variable=s_oproj_out depth=FIFO_D

    // §7 FFN RMSNorm → §8 gate/up GEMM
    static hls::stream<INT32> s_raw_act2("s_raw_act2");
    static hls::stream<INT8>  s_norm2_q("s_norm2_q");
    static hls::stream<INT8>  s_norm2_sc("s_norm2_sc");
#pragma HLS STREAM variable=s_raw_act2 depth=FIFO_D
#pragma HLS STREAM variable=s_norm2_q  depth=FIFO_D
#pragma HLS STREAM variable=s_norm2_sc depth=FIFO_D

    // §8 gate/up → §9 SiLU fuse
    static hls::stream<INT32> s_gateup_out("s_gateup_out");
#pragma HLS STREAM variable=s_gateup_out depth=FIFO_D

    // §9 SiLU → §10 FFN down GEMM
    static hls::stream<INT8>  s_ffn_act("s_ffn_act");
    static hls::stream<INT8>  s_ffn_sc("s_ffn_sc");
#pragma HLS STREAM variable=s_ffn_act depth=FIFO_D
#pragma HLS STREAM variable=s_ffn_sc  depth=FIFO_D

    // §10 FFN down → §11 残差（写回 URAM）
    static hls::stream<INT32> s_down_out("s_down_out");
#pragma HLS STREAM variable=s_down_out depth=FIFO_D

    // -----------------------------------------------------------------------
    // 预处理：加载激活和 gamma 到片上（并发在 DATAFLOW 前）
    // -----------------------------------------------------------------------
    (void)layer_id;
    (void)causal;
    (void)num_heads;
    (void)num_kv_heads;

    const bool throughput_prefill = exec_mode == 1;
    const int work_token_offset = throughput_prefill ? token_tile_offset : 0;
    const int work_token_count  = throughput_prefill ? token_tile_size : seq_len;

    const AXI256 *layer_wgt0  = wgt_half0  + wgt_offset;
    const AXI256 *layer_wgt1  = wgt_half1  + wgt_offset;
    const AXI256 *layer_meta0 = meta_half0 + meta_offset;
    const AXI256 *layer_meta1 = meta_half1 + meta_offset;

    load_act(act_ping, act_pong, use_ping != 0, seq_len, work_token_offset, work_token_count);
    load_gamma(gamma_attn, gamma_attn_buf);
    load_gamma(gamma_ffn,  gamma_ffn_buf);

    // -----------------------------------------------------------------------
    // §1  RMSNorm + 激活量化（attention pre-norm）
    // -----------------------------------------------------------------------
    res_to_stream(res_buf_0, s_raw_act, seq_len, work_token_offset, work_token_count);
    rmsnorm_quant(s_raw_act, gamma_attn_buf, s_norm1_q, s_norm1_sc, work_token_count);

    // -----------------------------------------------------------------------
    // §2  QKV 融合 GEMM（N=1600：Q960+K320+V320）
    //     wgt_half0/1：QKV 拼接权重（offline 预处理，列方向拼接）
    //     meta_half0/1：对应 GroupMeta
    // -----------------------------------------------------------------------
    w4a8_gemm(s_norm1_q, s_norm1_sc,
              layer_wgt0, layer_wgt1,
              layer_meta0, layer_meta1,
              s_qkv_out,
              work_token_count, C, N_QKV);

    // -----------------------------------------------------------------------
    // §3  RoPE（Q 和 K 旋转，V 直通）
    // -----------------------------------------------------------------------
    rope_split_qk(s_qkv_out, s_q_rot, s_k_rot, s_v_pass, pos_start + work_token_offset, work_token_count);

    // -----------------------------------------------------------------------
    // §4  Flash Attention + KV Cache Ping-Pong
    // -----------------------------------------------------------------------
    multi_head_attention(
        s_q_rot, s_k_rot, s_v_pass,
        k_cache_ddr, v_cache_ddr,
        s_attn_o,
        pos_start + work_token_offset, kv_len, work_token_count, layer_id
    );

    // §4 → §5：重排 head-major → flat token-major
    o_scatter(s_attn_o, s_o_flat, work_token_count);

    // -----------------------------------------------------------------------
    // §5  O 投影 GEMM（N=960）
    //     使用专门的 O proj 权重（wgt_half 地址偏移由 PS 提前计算好）
    //     此处复用 w4a8_gemm，传入的 wgt 指针是 QKV 权重之后的 O proj 偏移
    //     注：需要第二套 gamma stream——用 s_o_q/s_o_sc 作为激活量化后的输入
    // -----------------------------------------------------------------------
    act_quant(s_o_flat, s_o_q, s_o_sc, work_token_count, C);

    // O proj：wgt_half 指针地址偏移由调用方（PS）设置好
    // 工程中将 wgt_half0 + QKV_WGT_OFFSET 作为 O proj 权重起始地址
    // 此处直接使用同一指针（csim 验证时由 testbench 保证偏移）
    w4a8_gemm(s_o_q, s_o_sc,
              layer_wgt0 + (C * N_QKV / (8 * 32)),   // QKV 权重之后
              layer_wgt1 + (C * N_QKV / (8 * 32)),
              layer_meta0 + (C * N_QKV / (GRP * 16)),
              layer_meta1 + (C * N_QKV / (GRP * 16)),
              s_oproj_out,
              work_token_count, C, C);

    // -----------------------------------------------------------------------
    // §6  残差加 1：O proj 输出 + res_buf_0 → res_buf_1（attention 残差）
    // -----------------------------------------------------------------------
    residual_add_range(s_oproj_out, res_buf_0, work_token_offset, work_token_count);
    // 将 res_buf_0（已更新）复制到 res_buf_1 作为 FFN 残差基准
    // 并行发射给 §7 RMSNorm
    for (int t = 0; t < work_token_count; ++t) {
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            res_buf_1[work_token_offset + t][c] = res_buf_0[work_token_offset + t][c];
        }
    }

    // -----------------------------------------------------------------------
    // §7  RMSNorm（FFN pre-norm）
    // -----------------------------------------------------------------------
    res_to_stream(res_buf_1, s_raw_act2, seq_len, work_token_offset, work_token_count);
    rmsnorm_quant(s_raw_act2, gamma_ffn_buf, s_norm2_q, s_norm2_sc, work_token_count);

    // -----------------------------------------------------------------------
    // §8  FFN gate/up 拼接 GEMM（N=5120）
    //     权重地址偏移：O proj 之后
    // -----------------------------------------------------------------------
    const int QKV_WGT_ELEMS   = C * N_QKV;        // 960×1600 个 INT4
    const int OPROJ_WGT_ELEMS = C * C;             // 960×960
    const int FFN_WGT_OFFSET  = (QKV_WGT_ELEMS + OPROJ_WGT_ELEMS) / (8 * 32);

    w4a8_gemm(s_norm2_q, s_norm2_sc,
              layer_wgt0 + FFN_WGT_OFFSET,
              layer_wgt1 + FFN_WGT_OFFSET,
              layer_meta0 + (FFN_WGT_OFFSET * GRP / 32),
              layer_meta1 + (FFN_WGT_OFFSET * GRP / 32),
              s_gateup_out,
              work_token_count, C, N_FFN_GATEUP);

    // -----------------------------------------------------------------------
    // §9  SiLU(gate) × up + per-token 量化（流内融合）
    // -----------------------------------------------------------------------
    silu_gate_fuse_quant(s_gateup_out, s_ffn_act, s_ffn_sc, work_token_count);

    // -----------------------------------------------------------------------
    // §10 FFN down GEMM（N=960，K=2560=FFN_DIM）
    // -----------------------------------------------------------------------
    const int DOWN_WGT_OFFSET = FFN_WGT_OFFSET + (C * N_FFN_GATEUP) / (8 * 32);

    w4a8_gemm(s_ffn_act, s_ffn_sc,
              layer_wgt0 + DOWN_WGT_OFFSET,
              layer_wgt1 + DOWN_WGT_OFFSET,
              layer_meta0 + (DOWN_WGT_OFFSET * GRP / 32),
              layer_meta1 + (DOWN_WGT_OFFSET * GRP / 32),
              s_down_out,
              work_token_count, FFN_DIM, C);

    // -----------------------------------------------------------------------
    // §11 残差加 2：FFN down + res_buf_1 → res_buf_1（最终输出）
    // -----------------------------------------------------------------------
    residual_add_range(s_down_out, res_buf_1, work_token_offset, work_token_count);

    // -----------------------------------------------------------------------
    // 写回 DDR（act_out，供下一层或 lm_head 使用）
    // -----------------------------------------------------------------------
    store_act(act_out, seq_len, work_token_offset, work_token_count);

    // -----------------------------------------------------------------------
    // §12 lm_head（可选，M=1，K=960，N=49280）
    //     仅在 run_lmhead=1 时执行（Decode 阶段最后一个 layer 之后）
    // -----------------------------------------------------------------------
    if (run_lmhead) {
        static hls::stream<INT8>  s_lm_q("s_lm_q");
        static hls::stream<INT8>  s_lm_sc("s_lm_sc");
        static hls::stream<INT32> s_lm_out("s_lm_out");
#pragma HLS STREAM variable=s_lm_q   depth=FIFO_D
#pragma HLS STREAM variable=s_lm_sc  depth=FIFO_D
#pragma HLS STREAM variable=s_lm_out depth=64

        // 最后 token 的激活（已在 res_buf_1[seq_len-1] 中）
        static hls::stream<INT32> s_lm_raw("s_lm_raw");
#pragma HLS STREAM variable=s_lm_raw depth=FIFO_D

        // 仅发射最后一个 token
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            s_lm_raw.write(res_buf_1[seq_len - 1][c]);
        }

        // RMSNorm（lm_head 前的 final norm）
        rmsnorm_quant(s_lm_raw, gamma_ffn_buf, s_lm_q, s_lm_sc, 1);

        // lm_head GEMM（M=1，K=960，N=49280）
        w4a8_gemm(s_lm_q, s_lm_sc,
                  lmhead_wgt, lmhead_wgt + (VOCAB / 2) / 32,
                  lmhead_meta, lmhead_meta + VOCAB / (GRP * 16),
                  s_lm_out,
                  1, C, VOCAB);

        // 写出 logits（INT32，49280 维，PS 端做 Top-K Sampling）
        int lm_vecs = VOCAB / 8;  // 49280/(256/32)=49280/8=6160 个 AXI256
        for (int v = 0; v < lm_vecs; ++v) {
#pragma HLS PIPELINE II=1
            AXI256 word = 0;
            for (int b = 0; b < 8; ++b) {
#pragma HLS UNROLL
                INT32 val = s_lm_out.read();
                word.range(b*32+31, b*32) = (ap_uint<32>)val;
            }
            lmhead_out[v] = word;
        }
    }
}
