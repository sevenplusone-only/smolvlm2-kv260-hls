# SmolVLM2-500M × KV260 技术方案审计

## 审计前提

本审计基于你原先的技术方案文档和当前仓库代码，采用以下口径：

- 当前工程只能判定为`代码原型 / 架构原型`
- 你明确说明`还没有真实上 HLS`
- 因此所有依赖 `csynth`、时序、资源、`xclbin`、板端 profiling 的指标，默认都`不判定为完成`
- README 中的资源、时延、PPL、带宽推导，默认视为`设计目标或估算`，不是完成事实

### 状态定义

- `已落代码`：仓库中有真实实现，对应方案中的模块和数据流
- `部分完成`：有实现，但存在简化版、占位、关键路径未闭环，或缺测试/HLS证据
- `未完成`：只有设计说明、接口、注释目标，或者核心步骤仍未实现
- `不判定为完成`：凡是需要 HLS / 板端结果支撑的指标，在当前前提下统一不判完成

---

## 总评

当前工程已经把方案中的大部分核心模块搭成了代码原型，包括 decoder 单 kernel、W4A8 GEMM、RoPE、Flash Attention、KV Cache INT4 路径、ViT/connector 路径、host 调度框架和 testbench 骨架。

最近一轮主干落地补充了三件对“完整工程”最关键的事情：

- `llama.cpp` 已在工作区补齐完整源码，并保留原生 `ggml-xrt` backend 接入点。
- `smolvlm2_kv260/scripts/quantize_weights.py` 现在同时导出：
  - HLS decoder/vision/connector 所需的 fused W4A8 资产；
  - native `ggml-xrt` 可直接按 tensor name 命中的 sidecar manifest + `native_q8_0_weights.bin`。
- 构建脚本已经区分：
  - macOS 负责代码/脚本/非 XRT 构建与检查；
  - Linux/KV260 负责 Vitis HLS、v++、XRT、板端运行。

但它还不能被表述成“已完成 HLS 实现”或“已达成 KV260 指标”。原因很直接：一方面没有真实 HLS 综合/时序/资源/板端结果；另一方面若只看源码，仍有多处`简化版`、`placeholder`、`可选路径`和`未闭环实现`。

一句话总结：

`方案原型已搭建，硬件实现指标尚未完成验证。`

---

## 第一部分：逐指标审计表

### 1. 架构层

| 方案指标 | 当前状态 | 判断依据 | 证据文件 |
|---|---|---|---|
| 单 kernel DATAFLOW 的 decoder layer | 已落代码 | 顶层 `smolvlm2_decoder_layer(...)` 已存在，且显式使用 `#pragma HLS DATAFLOW`，QKV/Attn/FFN/lm_head 路径已串起 | `smolvlm2_kv260/src/decoder_kernel.cpp:174`, `smolvlm2_kv260/src/decoder_kernel.cpp:236` |
| Decoder 主数据流完整出现 | 已落代码 | 已包含 load_act、RMSNorm、QKV GEMM、RoPE、Attention、O proj、FFN、残差、store_act | `smolvlm2_kv260/src/decoder_kernel.cpp:48`, `smolvlm2_kv260/src/decoder_kernel.cpp:297`, `smolvlm2_kv260/src/decoder_kernel.cpp:353`, `smolvlm2_kv260/src/decoder_kernel.cpp:404` |
| ViT / connector 路径已在工程中出现 | 已落代码 | ViT kernel、connector kernel、pixel shuffle 路径都已编码 | `smolvlm2_kv260/src/vit_kernel.cpp:117`, `smolvlm2_kv260/src/vit_kernel.cpp:151` |
| lm_head 放 PL 端 | 部分完成 | decoder 中已有可选 lm_head 路径，但仅在 `run_lmhead` 时触发，且未见板端闭环证据 | `smolvlm2_kv260/src/decoder_kernel.cpp:433`, `smolvlm2_kv260/src/decoder_kernel.cpp:460` |

### 2. 算法与算子层

| 方案指标 | 当前状态 | 判断依据 | 证据文件 |
|---|---|---|---|
| W4A8 权重路径 | 已落代码 | 已有 W4 解包、在线 dequant、INT8 MAC 路径 | `smolvlm2_kv260/src/w4a8_gemm.h:45`, `smolvlm2_kv260/src/w4a8_gemm.h:166` |
| `pack_mul_2int8` DSP 复用设计 | 已落代码 | 双 INT8 乘法打包逻辑已实现，并绑定 DSP | `smolvlm2_kv260/src/w4a8_gemm.h:20`, `smolvlm2_kv260/src/w4a8_gemm.h:32` |
| RMSNorm + 激活量化融合 | 已落代码 | `rmsnorm_quant(...)` 已实现并在 decoder 主流中使用 | `smolvlm2_kv260/src/w4a8_gemm.h:104`, `smolvlm2_kv260/src/decoder_kernel.cpp:297` |
| QKV 拼接 GEMM | 已落代码 | `N=1600` 的 QKV 融合 GEMM 已在主 kernel 中调用 | `smolvlm2_kv260/src/decoder_kernel.cpp:305` |
| FFN gate/up 拼接 GEMM | 已落代码 | `N=5120` 的 gate/up 融合 GEMM 已存在 | `smolvlm2_kv260/src/decoder_kernel.cpp:369` |
| RoPE | 已落代码 | Q/K 分流与旋转逻辑已实现 | `smolvlm2_kv260/src/rope.h:52`, `smolvlm2_kv260/src/rope.h:105` |
| Flash Attention tiled online softmax | 部分完成 | Attention 主体已编码，但实现中多处标注为`简化`；KV 反量化、Q tile 读取、DDR 写回都不是完整闭环 | `smolvlm2_kv260/src/attention.h:124`, `smolvlm2_kv260/src/attention.h:154`, `smolvlm2_kv260/src/attention.h:183`, `smolvlm2_kv260/src/attention.h:205` |
| KV Cache INT4 存储 | 部分完成 | INT4 量化和片上 ping-pong buffer 已实现，但 DDR 写回处明确写着“实际需要 read-modify-write，此处用简化版” | `smolvlm2_kv260/src/attention.h:51`, `smolvlm2_kv260/src/attention.h:111` |
| O proj 输入重排 | 部分完成 | `o_scatter(...)` 已存在，但明确标注“简化版”，实际 head-interleave 到 token-major 重排未完成 | `smolvlm2_kv260/src/decoder_kernel.cpp:155`, `smolvlm2_kv260/src/decoder_kernel.cpp:161` |
| GEMM 输出 scale 合并 / accumulator 闭环 | 部分完成 | `psum` 已输出，但代码里明确写明“这里简化：直接把 psum 写出，Accumulator 层统一处理 scale” | `smolvlm2_kv260/src/w4a8_gemm.h:298` |
| ViT attention | 部分完成 | 已有 `vit_attention_full(...)`，但 exp 近似是第一版简化实现，不应宣称已验证精度/性能 | `smolvlm2_kv260/src/vit_kernel.cpp:26`, `smolvlm2_kv260/src/vit_kernel.cpp:35` |

### 3. 工程层

| 方案指标 | 当前状态 | 判断依据 | 证据文件 |
|---|---|---|---|
| HLS testbench | 部分完成 | testbench 已存在，但主要验证“全 0 权重下流水线不崩”，不是 PyTorch 对齐级精度验证 | `smolvlm2_kv260/tb/decoder_tb.cpp:120`, `smolvlm2_kv260/tb/decoder_tb.cpp:168` |
| HLS 综合脚本 | 已落代码 | `hls_synth.tcl` 已存在，包含 `csim_design` / `csynth_design` / `export_design` | `smolvlm2_kv260/scripts/hls_synth.tcl:21`, `smolvlm2_kv260/scripts/hls_synth.tcl:60`, `smolvlm2_kv260/scripts/hls_synth.tcl:65`, `smolvlm2_kv260/scripts/hls_synth.tcl:92` |
| Makefile / build 流程 | 已落代码 | `make csim`、`make synth`、`make link`、`make host`、`make quant` 已定义 | `smolvlm2_kv260/Makefile:41`, `smolvlm2_kv260/Makefile:56`, `smolvlm2_kv260/Makefile:87`, `smolvlm2_kv260/Makefile:98`, `smolvlm2_kv260/Makefile:110` |
| Host 调度框架 | 部分完成 | XRT BO 分配、layer loop、decode loop 已存在；但权重/meta/gamma/embedding/tokenizer 仍是简化版或占位 | `smolvlm2_kv260/scripts/host_driver.cpp:190`, `smolvlm2_kv260/scripts/host_driver.cpp:215`, `smolvlm2_kv260/scripts/host_driver.cpp:227`, `smolvlm2_kv260/scripts/host_driver.cpp:341` |
| 权重量化脚本 | 已落代码 | 仓库中存在离线量化脚本与构建入口 | `smolvlm2_kv260/scripts/quantize_weights.py`, `smolvlm2_kv260/Makefile:110` |

### 4. 指标层

| 方案指标 | 当前状态 | 判断依据 | 证据文件 |
|---|---|---|---|
| `DSP > 600` | 不判定为完成 | 需要 `csynth.rpt` 或综合日志支撑，当前没有实际结果 | 无 |
| `Load II = 1 / compute loop II = 1` | 不判定为完成 | 代码里有 `#pragma HLS PIPELINE II=1`，但 pragma 不是结果，必须看 HLS report | `smolvlm2_kv260/src/decoder_kernel.cpp:63`, `smolvlm2_kv260/src/w4a8_gemm.h:238` |
| `WNS > 0 @ 200MHz` | 不判定为完成 | `hls_synth.tcl` 设置了 200MHz，但没有时序结果 | `smolvlm2_kv260/scripts/hls_synth.tcl:13`, `smolvlm2_kv260/scripts/hls_synth.tcl:39` |
| `BRAM/URAM/LUT 在预算内` | 不判定为完成 | README 中只有估算，没有综合报告 | 无 |
| `PS-PL 调度开销 < 5%` | 不判定为完成 | 需要板端 profiling，当前 host 代码不能替代实测 | 无 |
| `端到端单层/单 token 时延` | 不判定为完成 | 需要 HLS/板端运行结果，当前没有 | 无 |
| `PPL 损失 +2.05` | 不判定为完成 | 文档中是目标/估算，仓库内没有精度对齐产物或评测结果 | 无 |

### 5. P0-P5 验收项

| 优先级 | 方案验收项 | 当前状态 | 判断依据 |
|---|---|---|---|
| P0 | 解决 `SHIP_K=768→960` 的 II 崩塌，恢复 DSP=680+，Load II=1 | 不判定为完成 | 没有真实综合结果 |
| P1 | W4 反量化嵌入 `load_wgt` 路径，csim 误差 < 1e-3 | 部分完成 | 反量化路径已编码，但没有 PyTorch 对齐级验证证据 |
| P2 | `pack_mul_2int8` 嵌入并验证 DSP48E2 利用 | 部分完成 | 代码已实现，但无综合报告证明 DSP 推断结果 |
| P3 | Flash Attention tile 数值稳定性验证 | 部分完成 | 主体已实现，但存在简化版且无参考误差报告 |
| P4 | PS-PL 集成与 layer 级流水 profiling | 未完成 | host 有框架，但无真实板端 profiling |
| P5 | Vivado 时序修复，WNS > 0 @ 200MHz | 未完成 | 没有实现结果或 timing report |

---

## 第二部分：已经做出的“加速设计”

下面这些可以表述为“已经在代码中体现出的结构性加速设计”，但不能表述为“已经实测证明达成加速”。

### 1. 权重压缩和权重加载优化

- 采用 W4A8 路径，权重按 INT4 存储，降低权重带宽压力
- `dequant_w4_group(...)` 把 W4 在线展开到 INT8，避免离线展开成更大权重
- `meta` 与权重分开读取，为 group 级 dequant 做准备

证据：

- `smolvlm2_kv260/src/w4a8_gemm.h:45`
- `smolvlm2_kv260/src/w4a8_gemm.h:166`

### 2. DSP 复用设计

- `pack_mul_2int8(...)` 试图用一个 DSP 完成两个 INT8 乘法
- 这是典型的算力密度优化设计，目标是降低 DSP 消耗或提升等效 MAC 密度

证据：

- `smolvlm2_kv260/src/w4a8_gemm.h:20`

### 3. 激活复用与算子融合

- QKV 融合 GEMM：一次读激活，产出 Q/K/V
- FFN gate/up 融合 GEMM：一次读激活，产出两支路
- RMSNorm 与激活量化直接连到下游 GEMM
- SiLU(gate) × up 与量化融合，减少中间写回

证据：

- `smolvlm2_kv260/src/decoder_kernel.cpp:305`
- `smolvlm2_kv260/src/decoder_kernel.cpp:369`
- `smolvlm2_kv260/src/w4a8_gemm.h:104`
- `smolvlm2_kv260/src/silu_gate.h:35`

### 4. DATAFLOW 和单 kernel 化

- decoder 主路径采用单 kernel + stream + DATAFLOW
- 这是减少 PS-PL 往返和中间落 DDR 的结构性优化方向

证据：

- `smolvlm2_kv260/src/decoder_kernel.cpp:236`

### 5. KV Cache 压缩与 head 级 ping-pong 设计

- KV Cache 采用 INT4 存储设计
- attention 中加入双 bank `kv_buf[2]`，目标是在 head 级别隐藏缓存搬运

证据：

- `smolvlm2_kv260/src/attention.h:37`
- `smolvlm2_kv260/src/attention.h:43`
- `smolvlm2_kv260/src/attention.h:276`

### 6. Vision 侧预处理压缩

- 已实现 `pixel_shuffle_4x`，把 `1024 × 768` patch token 重排到 `64 × 12288`
- connector projector 路径已接到独立 kernel 中

证据：

- `smolvlm2_kv260/src/vit_kernel.cpp:82`
- `smolvlm2_kv260/src/vit_kernel.cpp:151`

---

## 第三部分：目前不能宣称完成的内容

以下内容在答辩或汇报中不应表述成“已完成”：

### 1. 所有 HLS 结果类指标

- `DSP / BRAM / URAM / LUT` 实际占用
- `II=1`
- `WNS > 0 @ 200MHz`
- `Load II=1`
- `DSP > 600`

原因：没有真实 `csynth` / timing 报告。

### 2. 所有板端结果类指标

- KV260 上真实 token 延迟
- 单层时延
- Prefill / Decode 吞吐
- `PS-PL 调度开销 < 5%`
- DDR 带宽是否被完全掩盖

原因：没有真实 `xclbin` 运行和 profiling。

### 3. 几个关键路径还只是原型

- `o_scatter(...)` 还是简化版
- KV Cache DDR 写回没有完整 read-modify-write
- Flash Attention 中多处是简化 dequant / 简化 tile 处理
- GEMM 输出 scale 合并还没有完整闭环
- host 端 embedding、tokenizer、完整权重/meta/gamma 加载仍未完成

证据：

- `smolvlm2_kv260/src/decoder_kernel.cpp:161`
- `smolvlm2_kv260/src/attention.h:113`
- `smolvlm2_kv260/src/attention.h:154`
- `smolvlm2_kv260/src/attention.h:205`
- `smolvlm2_kv260/src/w4a8_gemm.h:301`
- `smolvlm2_kv260/scripts/host_driver.cpp:191`
- `smolvlm2_kv260/scripts/host_driver.cpp:217`
- `smolvlm2_kv260/scripts/host_driver.cpp:341`

---

## 第四部分：对外表述建议

如果你需要答辩或写说明，当前比较稳妥的说法是：

`我们已经完成了 SmolVLM2-500M 在 KV260 上的 HLS 架构原型与关键算子代码实现，包含 W4A8 GEMM、RoPE、Flash Attention、KV Cache INT4、单 kernel DATAFLOW、ViT/connector 路径和 PS 端调度框架；但当前尚未完成真实 HLS 综合和上板验证，因此资源、时序、吞吐、时延等硬件指标暂不能宣称已达成。`

不建议当前使用的说法：

- `已经完成 KV260 部署`
- `已经达到 200MHz`
- `已经实现 II=1`
- `已经验证 DSP/BRAM 在预算内`
- `已经证明 token 延迟达到目标`

---

## 第五部分：最简结论

从方案完成度看：

- `架构设计与代码原型：大部分已落代码`
- `功能闭环：部分完成`
- `HLS 指标与硬件结果：尚未完成验证`

从“加速了哪些部分”看：

- 已经在代码层面加速设计了`权重带宽`、`DSP 复用`、`激活复用`、`算子融合`、`单 kernel DATAFLOW`、`KV Cache 压缩与预取`、`ViT token 压缩`
- 但这些目前仍应表述为`加速设计已实现`，不是`加速结果已验证`
