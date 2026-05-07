// =============================================================================
// decoder_kernel.h  —  顶层 Decoder Layer Kernel 声明
// =============================================================================
#pragma once
#include "common.h"

// 顶层 kernel 参数结构（运行时配置）
struct DecoderConfig {
    int seq_len;       // 当前 token 数（Prefill: L≤512，Decode: 1）
    int kv_len;        // KV Cache 当前有效长度（含本次输出后）
    int pos_start;     // RoPE 位置编码起始
    int layer_id;      // 当前层 id（0~31），用于 KV Cache DDR 偏移计算
    int run_lmhead;    // 是否执行 lm_head（1=最后 token 的最后 layer 后执行）
    int is_prefill;    // 1=Prefill，0=Decode
    int use_ping;      // 1=act_ping 输入，0=act_pong 输入
    int causal;        // 1=LLM causal attention，0=ViT/full attention
    int num_heads;     // 当前 kernel 配置的 Q heads
    int num_kv_heads;  // 当前 kernel 配置的 KV heads
    int wgt_offset;    // AXI256 word offset into packed weights
    int meta_offset;   // AXI256 word offset into packed meta
};

// 顶层 kernel 接口声明
extern "C" void smolvlm2_decoder_layer(
    // 激活 Ping-Pong（HPC0）
    const AXI256  *act_ping,
    const AXI256  *act_pong,
          AXI256  *act_out,

    // 权重双半端口（HPC0/HPC1 各一半，降低 AXI 争用）
    const AXI256  *wgt_half0,    // HPC0
    const AXI256  *wgt_half1,    // HPC1

    // GroupMeta（scale/zero_point）双半端口
    const AXI256  *meta_half0,   // HPC1
    const AXI256  *meta_half1,   // HPC1

    // KV Cache DDR（HPC1）
          AXI256  *k_cache_ddr,
          AXI256  *v_cache_ddr,

    // RMSNorm gamma（两个 RMSNorm 的 gamma，离线量化为 INT8）
    const AXI256  *gamma_attn,   // attention pre-norm gamma（C=960）
    const AXI256  *gamma_ffn,    // FFN pre-norm gamma

    // lm_head 权重（可选，仅最后层最后 token 使用）
    const AXI256  *lmhead_wgt,
    const AXI256  *lmhead_meta,
          AXI256  *lmhead_out,   // 输出 logits（INT32，49280 维）

    // 运行时配置（AXI Lite）
    int            seq_len,
    int            kv_len,
    int            pos_start,
    int            run_lmhead,
    int            layer_id,
    int            use_ping,
    int            wgt_offset,
    int            meta_offset,
    int            causal,
    int            num_heads,
    int            num_kv_heads
);
