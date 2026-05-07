# SmolVLM2-500M × KV260 HLS 工程

## 工程概述

单 Kernel DATAFLOW 方案：将 SmolVLM2-500M 的 Decoder Layer（含 QKV 融合、GQA Flash Attention、FFN）完整封装进一个 `extern "C" void smolvlm2_decoder_layer(...)` kernel，顶层一个 `#pragma HLS DATAFLOW`，所有子模块通过 `hls::stream` 连接。

当前工程同时补入了 config.json 对齐的 ViT/connector 路径：
- ViT：`512x512` 图像、`16x16` patch、`1024` patch token、hidden `768`、`12` heads、QKV `768x2304`、FFN `768->3072->768`
- Connector：`use_resampler=false`，走 `pixel_shuffle_factor=4`，`1024x768 -> 64x12288 -> 64x960`
- LLM：hidden `960`、QKV `960x1600`、FFN `960->5120->2560->960`、lm_head `960x49280`

ViT/connector 使用 clean-room 的 `8x8x8` W4A8 GEMM primitive；SEC 工程只作为阵列思想参考，不复用 APOT 或源码结构。

当前仓库包含两条并行推进的推理栈路径：
- `native llama.cpp + ggml-xrt`：以 `ggml` backend 方式接入 KV260，优先把 `mul_mat` 与 `--mmproj` 的主干工程化打通，并保留 CPU fallback。
- `monolithic HLS kernels`：`smolvlm2_decoder_layer`、`smolvlm2_vit_prefill_kernel`、`smolvlm2_connector_kernel` 继续承担更完整的 FPGA 主路径验证。

**核心设计取舍：**
- 比 Odysseia 更完整（含 lm_head、RMSNorm 量化融合、完整 KV Cache ping-pong）
- 比 SEC 更聚合（单 kernel，无跨 kernel 通信开销，DATAFLOW 内自动并发）

---

## 工程文件结构

```
smolvlm2_kv260/
├── src/
│   ├── common.h           # 模型常数 / 数据类型 / 精度位宽定义
│   ├── w4a8_gemm.h        # W4A8 GEMM 引擎（pack_mul_2int8 + dequant）
│   ├── rope.h             # RoPE（CORDIC LUT，Q1.15 精度）
│   ├── attention.h        # Flash Attention + KV Cache Ping-Pong
│   ├── silu_gate.h        # SiLU(gate)×up 融合 + 在线量化
│   ├── tiled_gemm_8x8.h   # 8×8×8 W4A8 通用矩阵阵列
│   ├── vit_kernel.h/.cpp  # ViT / pixel shuffle / connector HLS kernels
│   ├── decoder_kernel.h   # 顶层 kernel 接口声明
│   └── decoder_kernel.cpp # 顶层 kernel 实现（含 DATAFLOW）
├── tb/
│   └── decoder_tb.cpp     # HLS csim testbench
├── scripts/
│   ├── hls_synth.tcl      # Vitis HLS 综合脚本
│   ├── hls_synth_vit.tcl  # ViT kernel 综合脚本
│   ├── hls_synth_connector.tcl # Connector kernel 综合脚本
│   ├── build.sh           # 完整构建流程（HLS → v++ link）
│   ├── host_driver.cpp    # PS 端 XRT 调度框架
│   └── quantize_weights.py# W4 离线量化工具
├── llamacpp_fpga/         # 官方 llama.cpp 的 XRT adapter/runner
├── cfg/
│   └── kv260_link.cfg     # v++ 链接配置（端口 → DDR bank 绑定）
└── Makefile
```

---

## 计算图（DATAFLOW 流水）

```
DDR(act_ping/pong)
    │ load_act
    ▼
res_buf_0 [URAM]
    │ res_to_stream
    ▼
§1  rmsnorm_quant ────────── gamma_attn_buf [BRAM]
    │ s_norm1_q / s_norm1_sc
    ▼
§2  w4a8_gemm (QKV, N=1600) ── wgt_half0/1 [DDR HPC0/HPC1]
    │ s_qkv_out              ── meta_half0/1 [DDR HPC1]
    ▼
§3  rope_split_qk
    ├─ s_q_rot (Q, 旋转)
    ├─ s_k_rot (K, 旋转)
    └─ s_v_pass (V, 直通)
         │
         ▼
§4  multi_head_attention ── k_cache_ddr / v_cache_ddr [DDR HPC1]
    │ s_attn_o             （KV Cache INT4 Ping-Pong，head 级别）
    ▼
§5  w4a8_gemm (O proj, N=960) + rmsnorm_quant（O 激活量化）
    │ s_oproj_out
    ▼
§6  residual_add(1) → res_buf_0 → copy → res_buf_1
    │
    ▼
§7  rmsnorm_quant (FFN) ── gamma_ffn_buf [BRAM]
    │ s_norm2_q
    ▼
§8  w4a8_gemm (gate/up, N=5120)
    │ s_gateup_out
    ▼
§9  silu_gate_fuse_quant  (SiLU LUT + absmax quant，流内三遍扫描)
    │ s_ffn_act / s_ffn_sc
    ▼
§10 w4a8_gemm (FFN down, N=960, K=2560)
    │ s_down_out
    ▼
§11 residual_add(2) → res_buf_1
    │ store_act
    ▼
DDR(act_out)
    │
    └─ [可选] §12 lm_head (M=1, K=960, N=49280)
                  → lmhead_out (logits INT32)
```

---

## 关键设计决策

### 1. AXI 接口（防 II 崩塌）

```cpp
// 必须用 ap_uint<256>*，一拍读 32 字节，Load II=1
// 绝对不能用 int8_t* 配合 UNROLL（HLS 产生多端口冲突，DSP 从 680+ 跌至 113）
#pragma HLS INTERFACE m_axi port=wgt_half0 bundle=gmem_w0 max_read_burst_length=16
```

K=960 时的对齐要求：`960 / 32 = 30`（整除），burst 边界保持对齐，trip count 可静态推断。

### 2. pack_mul_2int8（DSP 利用率翻倍）

一个 DSP48E2 同时做 2 个 INT8 乘法：
- 打包：`packed = (w_hi << 9) | w_lo`（27-bit）
- 一次 27×8 乘法出 35-bit 结果
- 低 9 位 = `w_lo × act`，高 17 位右移 = `w_hi × act`
- DSP 消耗从 `TM×TN×K_UNROLL` 降至一半

### 3. W4 Dequant 与权重加载 Overlap

```cpp
// dequant_w4_group 纯组合逻辑（LUTRAM），与 AXI burst 读取 overlap
// 不打断流水线，DSP 消耗为零
static inline void dequant_w4_group(WgtPack wgt, INT8 sc, INT8 zp, INT8 out[32]) {
#pragma HLS INLINE
    ...
}
```

### 4. Flash Attention（在线 Softmax）

Q-tile(32) × KV-tile(64) 双重循环，片上 BRAM 消耗与序列长度无关：
```
O_tile：32×64×2B = 4KB → 2 BRAM-36K（固定）
m/l 状态：32×4B = 128B → 寄存器
```

### 5. KV Cache INT4 Ping-Pong

head 级别预取：head-i Attention 计算期间，DMA 预取 head-(i+1) KV 数据：
```
DMA 时间（seqlen=512）：16KB / 8GB/s ≈ 2μs
Attention 计算：约 5~10μs
→ DMA 完全被 PL 计算掩盖
```

---

## 资源估算

| 资源  | 预算（留 20% margin） | 估算使用 | 余量  |
|-------|----------------------|---------|-------|
| DSP   | 998                  | ~512    | 48%   |
| BRAM  | 115                  | ~25     | 78%   |
| URAM  | 51                   | ~16     | 69%   |
| LUT   | 94K                  | ~25K    | 73%   |

---

## 构建流程

### 平台区分

- macOS:
  - 可以做代码修改、维度检查、量化脚本准备、`llama.cpp` 非 XRT 构建。
  - 不能本地完成 XRT/Vitis HLS 编译，也不能运行 KV260 overlay。
- Linux / KV260:
  - 负责 `vitis_hls`、`v++`、XRT runtime、板端运行与 profiling。

### 快速 csim（功能验证，~5分钟）
```bash
cd scripts
vitis_hls -f hls_synth.tcl
# 或
make csim
```

### HLS 综合（~15分钟）
```bash
make synth
# 检查 DSP > 600，WNS > 0 @ 200MHz
```

### 完整构建（~90分钟）
```bash
export PLATFORM=xilinx_kv260_smartcamera_202320_1
make link
```

### 权重量化
```bash
pip install torch transformers
make quant
# 输出到 quantized_weights/
```

量化脚本现在会导出两类资产：
- HLS 主链路资产：
  - `weights_half0.bin`
  - `weights_half1.bin`
  - `meta_half0.bin`
  - `meta_half1.bin`
  - `gamma_attn.bin`
  - `gamma_ffn.bin`
  - `vision_wgt.bin`
  - `vision_meta.bin`
  - `connector_wgt.bin`
  - `connector_meta.bin`
- native `ggml-xrt` sidecar 资产：
  - `manifest.json`
  - `native_q8_0_weights.bin`

`manifest.json` 中的 `native_backend.tensors[*].name` 使用 `llama.cpp`/GGUF canonical tensor name，例如：
- `blk.0.attn_q.weight`
- `blk.0.ffn_gate.weight`
- `output.weight`
- `v.blk.0.attn_q.weight`
- `mm.model.fc.weight`

这样 `ggml-xrt` 可以直接按 tensor name 命中 sidecar 资产，不需要依赖执行顺序猜测。

### native llama.cpp + XRT

Linux / KV260 上可开启 native backend：

```bash
cmake -S llama.cpp -B build-kv260 -DGGML_XRT=ON
cmake --build build-kv260 -j
```

运行前至少需要：

```bash
export GGML_XRT_XCLBIN=/path/to/fpga_gemm_or_kv260_overlay.xclbin
export GGML_XRT_MANIFEST=/path/to/quantized_weights/manifest.json
```

说明：
- `GGML_XRT_XCLBIN` 仍对应 native `mul_mat` 所需的 XRT compute unit。
- `GGML_XRT_MANIFEST` 用于加载 sidecar 权重资产；缺失时 backend 会退回旧的 GGUF Q8_0 路径或 CPU fallback。
- 当前 `ggml-xrt` 已识别 sidecar manifest，但 monolithic decoder / vision kernels 仍处于并行集成阶段。

### 多 kernel overlay

显式启用多 kernel overlay 时，`scripts/build.sh` 会综合并链接：
- `smolvlm2_decoder_layer`
- `smolvlm2_vit_prefill_kernel`
- `smolvlm2_connector_kernel`

对应命令：

```bash
make link_multi
```

如需仅保留 decoder kernel：

```bash
ENABLE_MULTI_KERNEL=0 bash scripts/build.sh
```

`make link` 仍保持原来的 decoder-only 行为；
`make link_multi` 才会显式开启 decoder + vision + connector 多 kernel 链接。

### PS 端驱动编译（在 KV260 上）
```bash
make host
./build/smolvlm2_infer smolvlm2_decoder.xclbin quantized_weights/weights_half0.bin
```

---

## 主要风险与对策

| 风险 | 对策 |
|------|------|
| SHIP_K=960 → II 崩塌 | 检查 `ap_uint<256>*` 对齐，K=960/32=30 整除，强制 `LOOP_TRIPCOUNT` |
| Accumulator 累加链时序（43-bit）| 关键路径插 `PIPELINE rewind`，拆成两拍寄存器打断 |
| BRAM 端口冲突 | 严格 `BIND_STORAGE type=ram_t2p impl=bram`，激活用 URAM |
| Flash Attention exp 溢出 | `m_old - m_new` 差值大时 exp→0，LUT 直接截断为 0 处理 |

---

## 精度说明

- W4A8，group_size=32，RTN 量化
- KV Cache INT4，per-group absmax 量化
- RoPE cos/sin：Q1.15（INT16）
- exp 近似：9-bit LUT，覆盖 [-8, 0]
- SiLU 近似：256 条目分段线性 LUT
- 预期 PPL 损失：+2.05（baseline 18.35 → ~20.4，ΔPPL ratio 1.112×）
