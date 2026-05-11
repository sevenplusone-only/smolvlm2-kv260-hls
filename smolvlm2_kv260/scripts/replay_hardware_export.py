#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_pipeline.quant_ops import np_quantize_w4_rtn


DECODER_LAYER_KEYS = [
    ("qkv", ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]),
    ("attn_out", ["self_attn.o_proj.weight"]),
    ("gate_up", ["mlp.gate_proj.weight", "mlp.up_proj.weight"]),
    ("down", ["mlp.down_proj.weight"]),
]


def parse_args() -> argparse.Namespace:
    default_export = (
        ROOT
        / "quant_experiments_awq_repair_20260508_010000"
        / "w4a8_awq_g32"
        / "hardware_export"
    )
    default_checkpoint = default_export.parent / "checkpoint" / "student_state.pt"
    parser = argparse.ArgumentParser(
        description="Replay exported W4A8 layer binaries against the saved PyTorch checkpoint."
    )
    parser.add_argument("--export-dir", type=Path, default=default_export)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--layers", default="0", help="comma list or 'all'")
    parser.add_argument("--check-vision", action="store_true")
    parser.add_argument("--check-connector", action="store_true")
    parser.add_argument("--check-lmhead", action="store_true")
    parser.add_argument(
        "--raw-state-dict-ok",
        action="store_true",
        help="Acknowledge that raw checkpoint tensors may not include QuantizedLinear export folding.",
    )
    return parser.parse_args()


def load_torch_state(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for replay. Run this in the same env used for quant export, "
            "for example: conda run -n robot python scripts/replay_hardware_export.py ..."
        ) from exc
    return torch.load(path, map_location="cpu")


def tensor_to_numpy(t: Any) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def find_key(state: dict[str, Any], suffix: str) -> str:
    matches = [key for key in state.keys() if key.endswith(suffix)]
    if len(matches) != 1:
        preview = ", ".join(matches[:6])
        raise KeyError(f"expected one key ending with {suffix!r}, found {len(matches)}: {preview}")
    return matches[0]


def find_text_key(state: dict[str, Any], layer: int, suffix: str) -> str:
    tails = [
        f"text_model.layers.{layer}.{suffix}",
        f"language_model.model.layers.{layer}.{suffix}",
        f"model.text_model.layers.{layer}.{suffix}",
    ]
    matches = [key for key in state.keys() if any(key.endswith(tail) for tail in tails)]
    if len(matches) != 1:
        preview = ", ".join(matches[:8])
        raise KeyError(f"expected one text layer key for layer {layer} suffix {suffix!r}, found {len(matches)}: {preview}")
    return matches[0]


def layer_weight(state: dict[str, Any], layer: int, suffix: str) -> np.ndarray:
    return tensor_to_numpy(state[find_text_key(state, layer, suffix)])


def quantize_matrix(matrix: np.ndarray, group_size: int) -> tuple[bytes, bytes]:
    packed, scales, zeros = np_quantize_w4_rtn(matrix, group_size)
    meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
    return packed.tobytes(), meta.tobytes()


def quantize_decoder_layer(state: dict[str, Any], layer: int, group_size: int) -> tuple[bytes, bytes, bytes, bytes]:
    wh0 = bytearray()
    wh1 = bytearray()
    mh0 = bytearray()
    mh1 = bytearray()
    for _name, suffixes in DECODER_LAYER_KEYS:
        matrix = np.concatenate([layer_weight(state, layer, suffix) for suffix in suffixes], axis=0)
        packed, scales, zeros = np_quantize_w4_rtn(matrix, group_size)
        half = packed.shape[0] // 2
        meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
        wh0.extend(packed[:half].tobytes())
        wh1.extend(packed[half:].tobytes())
        mh0.extend(meta[:half].tobytes())
        mh1.extend(meta[half:].tobytes())
    return bytes(wh0), bytes(wh1), bytes(mh0), bytes(mh1)


def compare_slice(label: str, expected: bytes, blob: bytes, offset: int) -> list[str]:
    actual = blob[offset : offset + len(expected)]
    if actual == expected:
        return []
    first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    return [f"{label} mismatch at exported offset {offset}+{first}; span={len(expected)}"]


def parse_layers(spec: str, total: int) -> list[int]:
    if spec == "all":
        return list(range(total))
    return [int(part) for part in spec.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    if not args.raw_state_dict_ok:
        print("note: replay uses raw checkpoint tensors.")
        print("note: final AWQ/LoRA parity should be run in the full export environment with model wrappers restored.")
        print("note: pass --raw-state-dict-ok to keep this reminder quiet.")
    manifest_path = args.export_dir / "hardware_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_size = int(manifest["scheme"]["group_size"])
    layers = parse_layers(args.layers, len(manifest.get("decoder_layers", [])))

    state = load_torch_state(args.checkpoint)
    w0 = (args.export_dir / "weights_half0.bin").read_bytes()
    w1 = (args.export_dir / "weights_half1.bin").read_bytes()
    m0 = (args.export_dir / "meta_half0.bin").read_bytes()
    m1 = (args.export_dir / "meta_half1.bin").read_bytes()

    errors: list[str] = []
    for layer in layers:
        entry = manifest["decoder_layers"][layer]
        if "w_offset_bytes" not in entry or "meta_offset_bytes" not in entry:
            errors.append(f"decoder layer {layer} missing byte offsets")
            continue
        exp_w0, exp_w1, exp_m0, exp_m1 = quantize_decoder_layer(state, layer, group_size)
        errors.extend(compare_slice(f"layer {layer} weights_half0", exp_w0, w0, entry["w_offset_bytes"]))
        errors.extend(compare_slice(f"layer {layer} weights_half1", exp_w1, w1, entry["w_offset_bytes"]))
        errors.extend(compare_slice(f"layer {layer} meta_half0", exp_m0, m0, entry["meta_offset_bytes"]))
        errors.extend(compare_slice(f"layer {layer} meta_half1", exp_m1, m1, entry["meta_offset_bytes"]))

    if args.check_lmhead:
        lm_key = find_key(state, "lm_head.weight")
        packed, meta = quantize_matrix(tensor_to_numpy(state[lm_key]), group_size)
        half_w = len(packed) // 2
        half_m = len(meta) // 2
        errors.extend(compare_slice("lmhead_wgt_half0", packed[:half_w], (args.export_dir / "lmhead_wgt_half0.bin").read_bytes(), 0))
        errors.extend(compare_slice("lmhead_wgt_half1", packed[half_w:], (args.export_dir / "lmhead_wgt_half1.bin").read_bytes(), 0))
        errors.extend(compare_slice("lmhead_meta_half0", meta[:half_m], (args.export_dir / "lmhead_meta_half0.bin").read_bytes(), 0))
        errors.extend(compare_slice("lmhead_meta_half1", meta[half_m:], (args.export_dir / "lmhead_meta_half1.bin").read_bytes(), 0))

    if args.check_vision or args.check_connector:
        print("vision/connector replay is scaffolded but model key aliases vary by Transformers version.")
        print("decoder/lmhead replay is the critical offset and half-split check for this export.")

    if errors:
        print("hardware export replay FAILED")
        for err in errors:
            print(" -", err)
        return 1

    print("hardware export replay PASS")
    print(f" export_dir = {args.export_dir}")
    print(f" layers = {layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
