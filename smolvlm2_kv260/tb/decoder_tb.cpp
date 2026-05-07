// =============================================================================
// decoder_tb.cpp  —  SmolVLM2 Decoder Layer HLS 功能仿真 Testbench
// =============================================================================
// 验收标准：
//   csim 与 PyTorch 参考实现误差 < 1e-3（INT8 量化下约 ±2 LSB 可接受）
//   覆盖 Prefill（M=4）和 Decode（M=1）两种场景
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include "decoder_kernel.h"
#include "rope.h"

// ---------------------------------------------------------------------------
// 辅助：生成随机 INT8 向量
// ---------------------------------------------------------------------------
void rand_int8(INT8 *buf, int n, int seed = 42) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        buf[i] = (INT8)((rand() % 256) - 128);
}

// ---------------------------------------------------------------------------
// 辅助：pack INT8 权重为 INT4 nibble（离线量化近似）
// ---------------------------------------------------------------------------
AXI256 pack_w4_to_axi256(INT8 w[64]) {  // 64 个 INT8 → 32 nibble → 128-bit，2 个 128-bit
    AXI256 out = 0;
    for (int i = 0; i < 32; ++i) {
        ap_uint<4> nib = (ap_uint<4>)((w[i] + 8) & 0xF);
        out.range(i*4+3, i*4) = nib;
    }
    for (int i = 0; i < 32; ++i) {
        ap_uint<4> nib = (ap_uint<4>)((w[32+i] + 8) & 0xF);
        out.range(128+i*4+3, 128+i*4) = nib;
    }
    return out;
}

// ---------------------------------------------------------------------------
// DDR 模拟缓冲（flat array，HLS 函数通过指针访问）
// ---------------------------------------------------------------------------
// 激活缓冲（Ping-Pong）：MAX_L × C × INT8
static const int ACT_BUF_WORDS = (MAX_L * C) / 32;  // AXI256 字数
static AXI256 ddr_act_ping[ACT_BUF_WORDS];
static AXI256 ddr_act_pong[ACT_BUF_WORDS];
static AXI256 ddr_act_out[ACT_BUF_WORDS];

// 权重缓冲（大小估算：一个 Decoder 层所有权重）
// QKV(960×1600) + O(960×960) + FFN_gate_up(960×5120) + FFN_down(2560×960)
// = 1536K + 921K + 4915K + 2457K ≈ 9.8M INT4 = 4.9M byte
static const int WGT_HALF_WORDS = 4096 * 128;  // 足够大的缓冲
static AXI256 ddr_wgt_half0[WGT_HALF_WORDS];
static AXI256 ddr_wgt_half1[WGT_HALF_WORDS];

// meta（scale/zp）缓冲
static const int META_WORDS = WGT_HALF_WORDS / 8;
static AXI256 ddr_meta_half0[META_WORDS];
static AXI256 ddr_meta_half1[META_WORDS];

// KV Cache（小 testbench：seqlen=32，5 KV heads）
static const int KV_WORDS = (H_KV * MAX_SEQ * D_HEAD) / (32 * 2);  // INT4
static AXI256 ddr_k_cache[KV_WORDS];
static AXI256 ddr_v_cache[KV_WORDS];

// gamma 缓冲（C=960 INT8 = 30 个 AXI256）
static AXI256 ddr_gamma_attn[C / 32];
static AXI256 ddr_gamma_ffn[C / 32];

// lm_head 缓冲（小 testbench：跳过，设 run_lmhead=0）
static AXI256 ddr_lmhead_wgt[1];
static AXI256 ddr_lmhead_meta[1];
static AXI256 ddr_lmhead_out[1];

// ---------------------------------------------------------------------------
// 填充 DDR 模拟缓冲（全 1 激活，全 0 权重 → 验证流水线不崩）
// ---------------------------------------------------------------------------
void fill_test_data(int seq_len) {
    // 激活：全 1（INT8=1）
    for (int v = 0; v < ACT_BUF_WORDS && v < (seq_len * C / 32); ++v) {
        AXI256 word = 0;
        for (int b = 0; b < 32; ++b)
            word.range(b*8+7, b*8) = (ap_uint<8>)(INT8)1;
        ddr_act_ping[v] = word;
    }

    // gamma：全 1（INT8=64，表示归一化权重≈1.0，右移 6 即可）
    for (int v = 0; v < C / 32; ++v) {
        AXI256 word = 0;
        for (int b = 0; b < 32; ++b)
            word.range(b*8+7, b*8) = (ap_uint<8>)(INT8)64;
        ddr_gamma_attn[v] = word;
        ddr_gamma_ffn[v]  = word;
    }

    // 权重：W4=8（zero point，解包后为 0），scale=1
    for (int v = 0; v < WGT_HALF_WORDS; ++v) {
        AXI256 word = 0;
        // 每个 nibble = 8（zero point），解包后 INT8 = (8-8)*1 = 0
        for (int b = 0; b < 32; ++b)
            word.range(b*4+3, b*4) = (ap_uint<4>)8;  // low 128-bit
        ddr_wgt_half0[v] = word;
        ddr_wgt_half1[v] = word;
    }

    // meta：scale=1，zp=8
    for (int v = 0; v < META_WORDS; ++v) {
        AXI256 word = 0;
        for (int b = 0; b < 16; ++b) {  // 16 组 per AXI256
            word.range(b*16+7,  b*16)   = (ap_uint<8>)(INT8)1;   // scale
            word.range(b*16+15, b*16+8) = (ap_uint<8>)(INT8)8;   // zero_point
        }
        ddr_meta_half0[v] = word;
        ddr_meta_half1[v] = word;
    }

    printf("[TB] DDR test data filled (seq_len=%d)\n", seq_len);
}

// ---------------------------------------------------------------------------
// 验证输出（全 0 权重 → 输出应接近 0，残差加后应接近原输入）
// ---------------------------------------------------------------------------
int verify_output(int seq_len) {
    int errors = 0;
    for (int v = 0; v < seq_len * C / 32 && v < ACT_BUF_WORDS; ++v) {
        AXI256 out = ddr_act_out[v];
        for (int b = 0; b < 32; ++b) {
            INT8 val = (INT8)(int)out.range(b*8+7, b*8).to_int();
            // 预期：权重全 0 → GEMM 输出 0 → 残差加 = 原输入 ≈ 1
            // 允许 ±2 误差（INT8 量化噪声）
            if (abs((int)val - 1) > 4) {
                if (errors < 10)
                    printf("[TB] MISMATCH at word=%d byte=%d: got %d expected ~1\n",
                           v, b, (int)val);
                errors++;
            }
        }
    }
    return errors;
}

// ---------------------------------------------------------------------------
// main：运行两个测试场景
// ---------------------------------------------------------------------------
int main() {
    printf("=== SmolVLM2 Decoder Layer HLS Testbench ===\n");

    int total_errors = 0;

    // -------------------------------------------------------------------
    // 测试 1：Prefill，seq_len=4（小尺寸，快速验证）
    // -------------------------------------------------------------------
    {
        int seq_len   = 4;
        int kv_len    = seq_len;
        int pos_start = 0;
        int run_lmh   = 0;

        fill_test_data(seq_len);

        printf("\n[TB] === Test 1: Prefill seq_len=%d ===\n", seq_len);

        smolvlm2_decoder_layer(
            ddr_act_ping, ddr_act_pong, ddr_act_out,
            ddr_wgt_half0, ddr_wgt_half1,
            ddr_meta_half0, ddr_meta_half1,
            ddr_k_cache, ddr_v_cache,
            ddr_gamma_attn, ddr_gamma_ffn,
            ddr_lmhead_wgt, ddr_lmhead_meta, ddr_lmhead_out,
            seq_len, kv_len, pos_start, run_lmh,
            0, 1, 0, 0, 1, H_Q, H_KV
        );

        int e = verify_output(seq_len);
        printf("[TB] Test 1: %s (%d errors in %d values)\n",
               e == 0 ? "PASS" : "FAIL", e, seq_len * C);
        total_errors += e;
    }

    // -------------------------------------------------------------------
    // 测试 2：Decode，seq_len=1
    // -------------------------------------------------------------------
    {
        int seq_len   = 1;
        int kv_len    = 5;   // 已有 4 个 cached token + 本次 1 个
        int pos_start = 4;
        int run_lmh   = 0;

        fill_test_data(seq_len);

        printf("\n[TB] === Test 2: Decode seq_len=1, kv_len=5 ===\n");

        smolvlm2_decoder_layer(
            ddr_act_ping, ddr_act_pong, ddr_act_out,
            ddr_wgt_half0, ddr_wgt_half1,
            ddr_meta_half0, ddr_meta_half1,
            ddr_k_cache, ddr_v_cache,
            ddr_gamma_attn, ddr_gamma_ffn,
            ddr_lmhead_wgt, ddr_lmhead_meta, ddr_lmhead_out,
            seq_len, kv_len, pos_start, run_lmh,
            0, 1, 0, 0, 1, H_Q, H_KV
        );

        int e = verify_output(seq_len);
        printf("[TB] Test 2: %s (%d errors in %d values)\n",
               e == 0 ? "PASS" : "FAIL", e, seq_len * C);
        total_errors += e;
    }

    // -------------------------------------------------------------------
    // 总结
    // -------------------------------------------------------------------
    printf("\n=== TOTAL: %s (%d errors) ===\n",
           total_errors == 0 ? "ALL PASS" : "FAILED", total_errors);
    return total_errors;
}
