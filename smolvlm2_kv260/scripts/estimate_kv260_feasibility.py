#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_transfer_budget import build_rows, build_savings, mib, MODEL_MAX_SEQ, HW_MAX_SEQ, seq_cap_mode


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_export = root / "quant_experiments_awq_repair_20260508_010000" / "w4a8_awq_g32" / "hardware_export"
    parser = argparse.ArgumentParser(description="Rough KV260 feasibility model from transfer budget.")
    parser.add_argument("export_dir", nargs="?", type=Path, default=default_export)
    parser.add_argument("--manifest-name", default="hardware_export_manifest.repaired.json")
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--effective-bandwidth-gbps", type=float, default=4.0)
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    parser.add_argument("--macs-per-cycle", type=float, default=256.0)
    parser.add_argument("--decode-window", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.export_dir / args.manifest_name).read_text(encoding="utf-8"))
    rows = build_rows(manifest, args.prefill_len, args.decode_steps)
    savings = {item.name: item.bytes for item in build_savings()}

    totals: dict[str, int] = {}
    for row in rows:
        totals[row.phase] = totals.get(row.phase, 0) + row.total_bytes

    window_saving_name = "decode_window2_batch_or_speculative_weight_meta_saved_bytes"
    bw_bytes_s = args.effective_bandwidth_gbps * 1e9 / 8.0
    prefill_bytes = totals.get("prefill", 0) + totals.get("vision", 0) + totals.get("connector", 0)
    decode_bytes = totals.get("decode", 0)
    exact_decode_bytes_v2 = max(0, decode_bytes - savings.get("exact_decode_weight_meta_saved_bytes", 0) * args.decode_steps)
    window_decode_bytes_v2 = max(0, decode_bytes - savings.get(window_saving_name, 0) * args.decode_steps)
    prefill_time_s = prefill_bytes / bw_bytes_s
    decode_time_s = decode_bytes / bw_bytes_s
    exact_decode_time_s_v2 = exact_decode_bytes_v2 / bw_bytes_s if exact_decode_bytes_v2 > 0 else 0.0
    window_decode_time_s_v2 = window_decode_bytes_v2 / bw_bytes_s if window_decode_bytes_v2 > 0 else 0.0
    decode_tok_s = args.decode_steps / decode_time_s if decode_time_s > 0 else 0.0
    exact_decode_tok_s_v2 = args.decode_steps / exact_decode_time_s_v2 if exact_decode_time_s_v2 > 0 else 0.0
    window_decode_tok_s_v2 = args.decode_steps / window_decode_time_s_v2 if window_decode_time_s_v2 > 0 else 0.0

    peak_macs_s = args.macs_per_cycle * args.clock_mhz * 1e6
    # Decoder matrix MACs per generated token, rough only: 32 layers plus lm_head.
    c = 960
    h_kv = 5
    d_head = 64
    ffn = 2560
    vocab = 49280
    per_layer_macs = (
        c * (c + h_kv * d_head * 2)
        + c * c
        + c * ffn * 2
        + ffn * c
    )
    decode_macs_per_token = per_layer_macs * 32 + c * vocab
    compute_decode_tok_s = peak_macs_s / decode_macs_per_token

    print("=== KV260 Feasibility Estimate ===")
    print(f"effective_bandwidth = {args.effective_bandwidth_gbps:.2f} Gbit/s")
    print(f"clock = {args.clock_mhz:.1f} MHz, macs/cycle = {args.macs_per_cycle:.1f}")
    print(f"model_max_seq = {MODEL_MAX_SEQ}, assumed_hw_max_seq = {HW_MAX_SEQ}, seq_cap_mode = {seq_cap_mode()}")
    print(f"decode_window = {args.decode_window}")
    print()
    print("Bandwidth-limited lower bound:")
    print(f"  vision+connector+prefill bytes: {mib(prefill_bytes):.2f} MiB")
    print(f"  prefill transfer time: {prefill_time_s * 1000:.1f} ms")
    print(f"  v1 decode bytes for {args.decode_steps} tokens: {mib(decode_bytes):.2f} MiB")
    print(f"  v1 decode transfer time: {decode_time_s:.2f} s")
    print(f"  v1 decode tokens/s from bandwidth only: {decode_tok_s:.3f}")
    print(f"  exact single-request v2 decode bytes: {mib(exact_decode_bytes_v2):.2f} MiB")
    print(f"  exact single-request v2 transfer time: {exact_decode_time_s_v2:.2f} s")
    print(f"  exact single-request v2 tokens/s: {exact_decode_tok_s_v2:.3f}")
    print(f"  window=2 batch/speculative upper-bound bytes: {mib(window_decode_bytes_v2):.2f} MiB")
    print(f"  window=2 batch/speculative transfer time: {window_decode_time_s_v2:.2f} s")
    print(f"  window=2 batch/speculative tokens/s: {window_decode_tok_s_v2:.3f}")
    print()
    print("Compute-only optimistic bound:")
    print(f"  decoder+lmhead MACs/token: {decode_macs_per_token / 1e9:.3f} GMAC")
    print(f"  peak MAC/s: {peak_macs_s / 1e9:.3f} GMAC/s")
    print(f"  decode tokens/s from compute only: {compute_decode_tok_s:.3f}")
    print()
    if window_decode_tok_s_v2 < compute_decode_tok_s:
        print("Conclusion: bandwidth is the first-order bottleneck under this model.")
    else:
        print("Conclusion: compute may become visible after transfer reduction.")
    print("Feasibility signal: exact single-request decode cannot reuse future-token weights without speculative or batched scheduling; window benefit is an upper-bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
