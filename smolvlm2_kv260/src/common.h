// =============================================================================
// common.h  —  SmolVLM2-500M × KV260  公共类型 / 模型常数
// =============================================================================
// 设计原则：
//   ① 所有位宽用 ap_int/ap_uint，不用 float/double
//   ② AXI 读端口必须用 ap_uint<256>*，保证 Load II=1
//   ③ group_size=32 对齐 128-bit AXI burst（32×INT4=16 byte），dequant II=1
// =============================================================================
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_fixed.h>

// ---------------------------------------------------------------------------
// §0  模型维度常数  (来自 config.json)
// ---------------------------------------------------------------------------
static constexpr int C          = 960;    // hidden_size
static constexpr int H_Q        = 15;     // num_attention_heads
static constexpr int H_KV       = 5;      // num_key_value_heads  (GQA ratio=3)
static constexpr int D_HEAD     = 64;     // head_dim
static constexpr int FFN_DIM    = 2560;   // intermediate_size
static constexpr int N_QKV      = C + H_KV*D_HEAD*2; // 960+320+320=1600
static constexpr int N_FFN_GATEUP = FFN_DIM * 2;     // 5120，离线拼接
static constexpr int VOCAB      = 49280;

// KV Cache 尺寸上限
static constexpr int MAX_SEQ    = 2048;
static constexpr int MAX_L      = 512;    // Prefill 最大序列长度

// ---------------------------------------------------------------------------
// §0.1 Vision / connector dimensions from config.json
// ---------------------------------------------------------------------------
static constexpr int IMAGE_SIZE          = 512;
static constexpr int PATCH_SIZE          = 16;
static constexpr int VIT_PATCH_GRID      = IMAGE_SIZE / PATCH_SIZE;  // 32
static constexpr int VIT_TOKENS          = VIT_PATCH_GRID * VIT_PATCH_GRID; // 1024
static constexpr int VIT_C               = 768;
static constexpr int VIT_HEADS           = 12;
static constexpr int VIT_HEAD_DIM        = 64;
static constexpr int VIT_QKV             = VIT_C * 3;     // 2304
static constexpr int VIT_FFN_DIM         = VIT_C * 4;     // 3072
static constexpr int PIXEL_SHUFFLE_FACTOR = 4;
static constexpr int IMAGE_TOKENS        = VIT_TOKENS / (PIXEL_SHUFFLE_FACTOR * PIXEL_SHUFFLE_FACTOR); // 64
static constexpr int CONNECTOR_IN        = VIT_C * PIXEL_SHUFFLE_FACTOR * PIXEL_SHUFFLE_FACTOR;        // 12288
static constexpr int CONNECTOR_OUT       = C;             // 960
static constexpr int IMAGE_TOKEN_ID      = 49190;

// ---------------------------------------------------------------------------
// §1  量化参数
// ---------------------------------------------------------------------------
static constexpr int GRP        = 32;     // group_size（W4 量化组）
static constexpr int TILE_K     = GRP;    // 内层 K-unroll = group_size，每拍切 1 组 scale

// ---------------------------------------------------------------------------
// §2  Tensor Core 阵列参数
//     TM=2, TN=2, K_UNROLL=32, pack_mul×2 → 等效 256 MAC/cycle
//     DSP 估算：(TM×TN×K_UNROLL)/2 * factor ≈ 64 DSP，8×8 实例 = 512 DSP
// ---------------------------------------------------------------------------
static constexpr int TM         = 2;
static constexpr int TN         = 2;
static constexpr int K_UNROLL   = GRP;   // 32
static constexpr int TILE_M     = 4;     // 外层 M-tile（适配 URAM 行宽）
static constexpr int TILE_N     = 64;    // 外层 N-tile（BRAM 行宽）

// Clean-room SEC-inspired micro-array shape.  This is the reusable matmul
// primitive used by the new ViT/connector path; decoder GEMM can migrate to it
// incrementally after shape validation.
static constexpr int ARRAY_M     = 8;
static constexpr int ARRAY_N     = 8;
static constexpr int ARRAY_K     = 8;

static constexpr int MAX_DEC_GEMM_K = FFN_DIM;       // decoder K: 960 or 2560
static constexpr int MAX_ANY_GEMM_K = CONNECTOR_IN;  // largest planned K: 12288

// ---------------------------------------------------------------------------
// §3  Flash Attention Tiling
// ---------------------------------------------------------------------------
static constexpr int QT         = 32;    // Q-tile 大小（token）
static constexpr int KVT        = 64;    // KV-tile 大小

// ---------------------------------------------------------------------------
// §4  数据类型别名
// ---------------------------------------------------------------------------
using INT4p  = ap_uint<4>;   // W4 原始权重（单个 nibble）
using INT8   = ap_int<8>;    // 激活 / 反量化权重
using INT16  = ap_int<16>;   // 中间累加
using INT32  = ap_int<32>;   // GEMM 部分和
using INT43  = ap_int<43>;   // 累加器最宽路径（文档 DW_R=43）

// AXI 总线宽度
using AXI256 = ap_uint<256>; // 32 byte/拍，Load II=1 的关键

// scale/zero-point 存为 INT8，乘法后升为 INT16
using SCALE_T  = ap_int<8>;
using SCALE16  = ap_int<16>;

// 激活向量（每拍处理 K_UNROLL=32 个 INT8）
struct ActVec {
    INT8 v[K_UNROLL]; // 32×INT8 = 256-bit，与 AXI256 对齐
};

// 权重 nibble 打包向量（128-bit → 32 个 INT4）
using WgtPack = ap_uint<128>; // 32 INT4 = 16 byte

// 量化组元数据
struct GroupMeta {
    INT8  scale;       // W4 per-group scale（INT8 表示）
    INT8  zero_point;  // W4 per-group zero_point
};

// GEMM 输出（单个 token × N 维，累加器输出，INT32）
using GemmOut = ap_int<32>;

// Softmax 精度（DW_SOFTMAX=21）
using SOFTMAX_T = ap_int<21>;

// exp 近似精度（DW_EXP=9）
using EXP_T = ap_uint<9>;

enum KernelMode {
    MODE_LLM_DECODER = 0,
    MODE_VIT_PATCH   = 1,
    MODE_VIT_QKV     = 2,
    MODE_VIT_ATTN    = 3,
    MODE_VIT_FFN     = 4,
    MODE_CONNECTOR   = 5,
    MODE_LM_HEAD     = 6,
    MODE_GENERIC_GEMM = 7
};

struct GemmShape {
    int M;
    int K;
    int N;
};

static inline bool is_supported_match_shape(int M, int K, int N) {
#pragma HLS INLINE
    return
        (M == VIT_TOKENS   && K == VIT_C        && N == VIT_QKV) ||
        (M == VIT_TOKENS   && K == VIT_C        && N == VIT_FFN_DIM) ||
        (M == VIT_TOKENS   && K == VIT_FFN_DIM  && N == VIT_C) ||
        (M == IMAGE_TOKENS && K == CONNECTOR_IN && N == CONNECTOR_OUT) ||
        (K == C            && N == N_QKV) ||
        (K == C            && N == C) ||
        (K == C            && N == N_FFN_GATEUP) ||
        (K == FFN_DIM      && N == C) ||
        (M == 1            && K == C            && N == VOCAB);
}

// ---------------------------------------------------------------------------
// §5  hls::stream FIFO 深度（DATAFLOW 下 stream 即对应 Ping-Pong 缓冲）
// ---------------------------------------------------------------------------
static constexpr int FIFO_D     = 16;   // 普通 stream depth
static constexpr int FIFO_ATTN  = 64;   // attention score stream（宽，需大一点）

// ---------------------------------------------------------------------------
// §7  LUT 精度常量
// ---------------------------------------------------------------------------
static constexpr int RMSNORM_RSQRT_LUT_SIZE = 64;
static constexpr int EXP_LUT_SIZE           = 512;
static constexpr int SILU_LUT_SIZE          = 385;
static constexpr INT32 Q15_ONE              = 32768;
static constexpr INT32 Q15_INV_SQRT2        = 23170;  // round(2^-0.5 * 2^15)
static constexpr INT32 Q15_SQRT2            = 46341;  // round(2^0.5 * 2^15)

// ---------------------------------------------------------------------------
// §6  辅助宏：确保循环被 HLS 看到 trip count 上界
// ---------------------------------------------------------------------------
#define HLS_LOOP_TRIPCOUNT(min_v, max_v) \
    _Pragma("HLS LOOP_TRIPCOUNT min=" #min_v " max=" #max_v)
