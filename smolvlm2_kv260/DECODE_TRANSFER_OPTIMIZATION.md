# Decode Transfer Optimization Notes

## Current Ground Truth

The current quantization and export path is:

- `W4A8`
- `AWQ`
- `group_size = 32`
- host-folded AWQ export for KV260

The local export bundle exists at:

```text
smolvlm2_kv260/quant_experiments_awq_repair_20260508_010000/w4a8_awq_g32/hardware_export
```

It contains the expected runtime assets:

- `weights_half0.bin`
- `weights_half1.bin`
- `meta_half0.bin`
- `meta_half1.bin`
- `vision_wgt.bin`
- `vision_meta.bin`
- `connector_wgt.bin`
- `connector_meta.bin`
- `lmhead_wgt_half0.bin`
- `lmhead_wgt_half1.bin`
- `lmhead_meta_half0.bin`
- `lmhead_meta_half1.bin`
- `hardware_export_manifest.repaired.json`

## What KV Cache Does And Does Not Solve

KV cache is already part of the attention path.

In `src/attention.h`, K/V cache is addressed by:

- `layer_id`
- `head_id`
- `pos`
- `group`

This means historical K/V reuse is structurally present.

However, KV cache stores historical attention state, not layer weights. It does not remove decoder `weight/meta` reads.

The dominant decode transfer pressure is still:

- decoder layer `weight/meta` reread for every generated token
- `lm_head` read on every generated token

## Exact Decode Limitation

For one exact autoregressive request, token `t + 1` is not known until token `t` has produced logits and the PS/runtime has sampled the next token.

So a simple `decode_window = 2` is only valid when the next token activation is already available. That can happen in:

- batched multi-request decode
- accepted speculative decode tokens
- teacher-forced replay / known-token validation
- prompt or prefill-style token blocks

It is not directly available for one exact greedy/sampling request without changing the generation algorithm.

## What Is Already In The Code

The current runtime has a throughput decode skeleton:

- `SmolVlm2XrtRunner::run_decode_throughput_window`
- `smolvlm2_fpga_run_decode_window`
- decoder kernel arguments:
  - `exec_mode`
  - `token_tile_offset`
  - `token_tile_size`

The llama-server path intentionally calls decode with `window_tokens = 1` for exact generation.

## KV260-Conservative Optimization Direction

Because KV260 resources are limited, avoid trying to keep all 32 layers of decoder weights on chip. The decoder weights are far too large for BRAM/URAM.

The next practical optimization stages are:

1. Keep exact decode stable with `window_tokens = 1`.
2. Use `decode_window > 1` only for batched requests, speculative accepted tokens, or replay.
3. Preserve layer-wise scheduling so one layer launch can serve multiple known tokens when available.
4. Use XRT events / multiple BOs later to overlap next-layer weight/meta movement with current-layer compute.
5. Keep connector -> decoder bridge device-side; future v2 can write connector output directly into decoder activation layout.

## Why Not Full Weight Residency

Approximate decoder weight/meta size is well above what KV260 can keep on chip. Full residency would push the design into resource and timing failure.

The realistic KV260 target is not full residency. It is:

- reduce repeated DDR round trips when a token window exists
- improve overlap
- keep exact single-request decode correct
- use board profiling to find whether DDR, lm_head, or kernel launch overhead dominates first

## Commands To Recheck

```bash
python3 scripts/analyze_transfer_budget.py --manifest-name hardware_export_manifest.repaired.json
python3 scripts/estimate_kv260_feasibility.py --manifest-name hardware_export_manifest.repaired.json
python3 scripts/audit_dataflow_static.py
```

