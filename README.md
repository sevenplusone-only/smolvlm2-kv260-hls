# HLS_1 Handoff Notes

这个仓库当前包含两条并行但已经收敛中的主线：

- `smolvlm2_kv260/`
  - KV260 上的 HLS/XRT 主工程
  - 当前量化主线是 `W4A8 + AWQ + group_size=32`
  - 目标是让 `ViT -> connector -> bridge -> prefill -> decode` 尽可能统一挂到 FPGA
- `llama.cpp/`
  - PS 端软件栈与 server
  - 当前已经接入 `FpgaSession`/XRT runtime，用于把多模态请求显式导向 FPGA 主线

## 当前状态

- 量化主线：
  - 当前硬件/导出主线按 `W4A8 + AWQ + group_size=32` 对齐
  - `manifest` / `host_driver` / `llamacpp_fpga` / `llama-server` 已基本对齐
- FPGA 主路径：
  - `prefill` 和 `decode` 的主计算路径已经挂到 FPGA
  - `vision` 路径支持 PS 端轻预处理后，把 patch activation 写入 FPGA，再在 FPGA 上完成 `ViT + connector + bridge`
- 仍由 PS 负责的部分：
  - tokenizer
  - 图像 decode / resize / normalize
  - token embedding lookup 与当前的 int8 bridge
  - 采样（当前先走 greedy top1）

## 最重要的工程事实

decode 阶段当前最大的瓶颈不是乘法器，也不是“没有使用 KV cache”。

真实情况是：
- attention 已经会从 `KV cache` 读取历史 K/V
- 但每个 decode token 仍会把 32 层 decoder 的 `weight/meta` 从 DDR 重读
- `lm_head` 仍然是显著大读

所以真正的下一阶段优化方向不是“改成直接从 KV cache 读权重”，因为 KV cache 里存的是历史 K/V，不是 decoder layer 权重。

真正该做的是：
- layer-resident / token-window 调度
- 让同一层 weight/meta 服务多个 token
- weight/meta 预取和当前层计算 overlap

注意：对单个 exact 自回归请求，`token-window > 1` 不能凭空成立，因为 token `t + 1` 必须等待 token `t` 的 logits 和采样结果。`decode_window > 1` 更适合 batched decode、speculative decode accepted tokens、replay 或已知 token window。

## 当前可接受的结论

- 这个仓库已经适合上传到 GitHub 并转到 Win/Linux 继续开发
- 但它还不是“已经在板上完全验证通过的最终版”
- Mac 上已经能完成：
  - export/manifest 一致性检查
  - transfer budget 分析
  - dataflow 静态审计
  - XRT stub 下的 C++ 结构检查

## 推荐下一步

1. 先上传一个“干净工程版”到 GitHub，不带本地量化大目录和虚拟环境。
2. 在 Win/Linux 上继续做：
   - Vitis HLS 综合
   - v++ link
   - XRT 真链接
   - 板端最小路径验证
3. 板端验证通过后，再主攻 decode 的 weight/meta 复用和 overlap。

更详细的 decode 数据搬运优化边界见：

```text
smolvlm2_kv260/DECODE_TRANSFER_OPTIMIZATION.md
```
