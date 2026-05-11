# Upload Checklist

## 建议上传的内容

- `llama.cpp/`
- `smolvlm2_kv260/src/`
- `smolvlm2_kv260/scripts/`
- `smolvlm2_kv260/llamacpp_fpga/`
- `smolvlm2_kv260/cfg/`
- `smolvlm2_kv260/Makefile`
- `smolvlm2_kv260/README.md`
- 顶层 `README.md`
- 顶层 `.gitignore`

## 不建议上传的内容

- `.DS_Store`
- `.venv/`
- `aicas_semi/`
- `smolvlm2_kv260/quant_experiments*/`
- `smolvlm2_kv260/quant_pipeline/`
- 本地导出的 `hardware_export` 大包
- 本地 build/cache 文件
- 本地 `.xclbin` / `.xo` / `.jou` / `.log`

## 上传前自检

在当前 Mac 上至少跑：

```bash
python3 HLS_1/smolvlm2_kv260/scripts/audit_dataflow_static.py
bash HLS_1/smolvlm2_kv260/scripts/mac_preflight.sh
```

预期：
- `Static dataflow audit PASS`
- `Mac preflight PASS`

## Git 状态建议

上传前建议确认：

- 只提交代码、脚本、文档
- 不提交量化实验大目录
- 不提交本地虚拟环境
- 不提交 macOS 垃圾文件

