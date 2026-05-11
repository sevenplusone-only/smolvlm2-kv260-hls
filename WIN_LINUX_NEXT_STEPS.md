# Win/Linux Next Steps

## 1. 在 GitHub 拉取干净仓库

先确认仓库中不包含：
- `.venv`
- `quant_experiments*`
- 本地 `hardware_export` 大目录
- `.DS_Store`

## 2. 在 Win 机器上做的事

Win 更适合：
- 整理工程
- 编辑代码
- 管理 GitHub 仓库
- 准备 Python 量化环境

Win 上不建议作为最终板端编译环境，除非你已经完整配置好：
- Vitis / Vitis HLS
- XRT
- KV260 对应 platform

## 3. 更推荐的板端主环境

更推荐在 Linux / KV260 对应环境做：

- `vitis_hls`
- `v++`
- XRT 真链接
- host/runtime 真编译
- 板端运行与 profiling

## 4. Linux 上优先顺序

### 第一步：结构验证

```bash
python3 smolvlm2_kv260/scripts/audit_dataflow_static.py
bash smolvlm2_kv260/scripts/mac_preflight.sh
```

虽然脚本名叫 `mac_preflight`，但它本质上也能做结构检查。

### 第二步：量化导出一致性

如果要重新导出：

```bash
python3 smolvlm2_kv260/scripts/check_hardware_export.py <hardware_export_dir>
python3 smolvlm2_kv260/scripts/replay_hardware_export.py --export-dir <hardware_export_dir> --layers all --check-lmhead
```

### 第三步：HLS 综合

```bash
cd smolvlm2_kv260
make synth
```

关注：
- Fmax
- DSP / BRAM / URAM
- `DATAFLOW` 是否退化

### 第四步：多 kernel 链接

```bash
cd smolvlm2_kv260
make link_multi
```

### 第五步：server / XRT 真编译

在安装好 XRT 的 Linux 环境：

```bash
cmake -S llama.cpp -B build-kv260 -DLLAMA_SERVER_SMOLVLM2_FPGA=ON
cmake --build build-kv260 -j
```

如果你走 native backend，也可以额外尝试：

```bash
cmake -S llama.cpp -B build-kv260 -DGGML_XRT=ON
cmake --build build-kv260 -j
```

## 5. 板端最小验证路径

不要一上来就跑完整多轮生成，建议按下面顺序：

1. `vision -> connector -> bridge`
2. `prefill first layer`
3. `full prefill`
4. `decode first token`
5. `multi-token decode`
6. `llama-server multimodal request`

## 6. 下一阶段真正该优化什么

不是去“从 KV cache 读取权重”。

因为：
- `KV cache` 里存的是历史 K/V
- decoder `weight/meta` 仍然是独立的层参数

下一阶段真正该做的是：
- `decode window > 1`
- layer-resident 调度
- weight/meta 预取和当前层计算 overlap
- 更强的 connector -> decoder 设备侧直通

注意：`decode window > 1` 不是 exact 单请求自回归生成的默认加速项。它需要 batched decode、speculative accepted tokens、replay、或其他已知 token window。单请求 exact decode 的第一优先级仍然是稳定 `window=1`，再用 profiling 判断是否值得引入 speculative 或多请求调度。
