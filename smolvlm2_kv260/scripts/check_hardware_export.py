#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path


AXI256_BYTES = 32

C = 960
H_KV = 5
D_HEAD = 64
FFN_DIM = 2560
NUM_LAYERS = 32
VOCAB = 49280

VIT_TOKENS = 1024
VIT_C = 768
VIT_QKV = 2304
VIT_FFN = 3072
IMAGE_TOKENS = 64
CONNECTOR_IN = 12288
CONNECTOR_OUT = 960

DECODER_MATRICES = [
    ("qkv", C, C + H_KV * D_HEAD * 2),
    ("attn_out", C, C),
    ("gate_up", C, FFN_DIM * 2),
    ("down", FFN_DIM, C),
]

VISION_EXPECTED_PREFIX = [
    ("patch_embed", VIT_C, VIT_C),
]


def packed_w4_bytes(k: int, n: int) -> int:
    return k * n // 2


def meta_bytes(k: int, n: int) -> int:
    return n * (k // 32) * 2


def expect_eq(actual: int, expected: int, msg: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{msg}: got {actual}, expected {expected}")


def size(path: Path) -> int:
    return path.stat().st_size


def expect(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate KV260 hardware_export sizes, offsets, and shapes.")
    parser.add_argument("export_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--manifest-name", default="hardware_export_manifest.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.export_dir is not None:
        export_dir = args.export_dir.expanduser().resolve()
    else:
        export_dir = root / "quant_experiments_awq_repair_20260508_010000" / "w4a8_awq_g32" / "hardware_export"
    manifest_path = export_dir / args.manifest_name
    if not manifest_path.exists():
        print(f"hardware export check FAILED\n - missing manifest: {manifest_path}")
        return 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    expect(payload.get("kv260_status") == "native_w4a8_ready_host_folded_awq",
           "kv260_status is not native_w4a8_ready_host_folded_awq", errors)

    files = {name: export_dir / name for name in payload.get("artifacts", [])}
    for name, path in files.items():
        expect(path.exists(), f"missing artifact: {name}", errors)

    required_artifacts = [
        "vision_wgt.bin", "vision_meta.bin",
        "connector_wgt.bin", "connector_meta.bin",
        "weights_half0.bin", "weights_half1.bin",
        "meta_half0.bin", "meta_half1.bin",
        "gamma_attn.bin", "gamma_ffn.bin",
        "lmhead_wgt_half0.bin", "lmhead_wgt_half1.bin",
        "lmhead_meta_half0.bin", "lmhead_meta_half1.bin",
    ]
    for name in required_artifacts:
        expect((export_dir / name).exists(), f"missing required artifact: {name}", errors)

    decoder_layers = payload.get("decoder_layers", [])
    expect(len(decoder_layers) == NUM_LAYERS, f"decoder layer count {len(decoder_layers)} != {NUM_LAYERS}", errors)

    decoder_full_wgt_bytes = sum(packed_w4_bytes(k, n) for _, k, n in DECODER_MATRICES)
    decoder_half_wgt_bytes = decoder_full_wgt_bytes // 2
    decoder_full_meta_bytes = sum(meta_bytes(k, n) for _, k, n in DECODER_MATRICES)
    decoder_half_meta_bytes = decoder_full_meta_bytes // 2

    if decoder_layers:
        for idx, layer in enumerate(decoder_layers):
            expect(layer["layer"] == idx, f"decoder layer index mismatch at {idx}", errors)
            expect("w_offset_axi256" in layer and "meta_offset_axi256" in layer,
                   f"decoder layer {idx} missing axi offsets", errors)
            if "w_offset_axi256" in layer and "meta_offset_axi256" in layer:
                expect(layer["w_offset_bytes"] == layer["w_offset_axi256"] * AXI256_BYTES,
                       f"decoder layer {idx} weight offset bytes/axi mismatch", errors)
                expect(layer["meta_offset_bytes"] == layer["meta_offset_axi256"] * AXI256_BYTES,
                       f"decoder layer {idx} meta offset bytes/axi mismatch", errors)
            if idx + 1 < len(decoder_layers) and "w_offset_bytes" in layer and "meta_offset_bytes" in layer:
                nxt = decoder_layers[idx + 1]
                expect_eq(nxt["w_offset_bytes"] - layer["w_offset_bytes"], decoder_half_wgt_bytes,
                          f"decoder layer {idx} packed half span mismatch", errors)
                expect_eq(nxt["meta_offset_bytes"] - layer["meta_offset_bytes"], decoder_half_meta_bytes,
                          f"decoder layer {idx} meta half span mismatch", errors)

        if "w_offset_bytes" in decoder_layers[-1] and "meta_offset_bytes" in decoder_layers[-1]:
            expect_eq(decoder_layers[-1]["w_offset_bytes"] + decoder_half_wgt_bytes,
                      size(export_dir / "weights_half0.bin"),
                      "decoder packed half file final span mismatch", errors)
            expect_eq(decoder_layers[-1]["meta_offset_bytes"] + decoder_half_meta_bytes,
                      size(export_dir / "meta_half0.bin"),
                      "decoder meta half file final span mismatch", errors)

    expect(size(export_dir / "weights_half0.bin") == size(export_dir / "weights_half1.bin"),
           "decoder weight halves size mismatch", errors)
    expect(size(export_dir / "meta_half0.bin") == size(export_dir / "meta_half1.bin"),
           "decoder meta halves size mismatch", errors)
    expect_eq(size(export_dir / "weights_half0.bin"), decoder_half_wgt_bytes * NUM_LAYERS,
              "decoder weight half total size mismatch", errors)
    expect_eq(size(export_dir / "meta_half0.bin"), decoder_half_meta_bytes * NUM_LAYERS,
              "decoder meta half total size mismatch", errors)
    expect(size(export_dir / "gamma_attn.bin") == NUM_LAYERS * C,
           "gamma_attn.bin size mismatch", errors)
    expect(size(export_dir / "gamma_ffn.bin") == NUM_LAYERS * C,
           "gamma_ffn.bin size mismatch", errors)

    connector = payload.get("connector", {})
    expect(connector.get("M") == IMAGE_TOKENS, "connector M mismatch", errors)
    expect(connector.get("K") == CONNECTOR_IN, "connector K mismatch", errors)
    expect(connector.get("N") == CONNECTOR_OUT, "connector N mismatch", errors)
    expect(connector.get("w_offset_axi256") == 0, "connector weight offset expected 0", errors)
    expect(connector.get("meta_offset_axi256") == 0, "connector meta offset expected 0", errors)
    expect_eq(size(export_dir / "connector_wgt.bin"), packed_w4_bytes(CONNECTOR_IN, CONNECTOR_OUT),
              "connector_wgt.bin size mismatch", errors)
    expect_eq(size(export_dir / "connector_meta.bin"), meta_bytes(CONNECTOR_IN, CONNECTOR_OUT),
              "connector_meta.bin size mismatch", errors)

    vision_layers = payload.get("vision_layers", [])
    expect(len(vision_layers) >= 1 + 12 * 4, "vision layer manifest looks incomplete", errors)
    if vision_layers:
        patch = vision_layers[0]
        expect(patch.get("name") == "patch_embed", "vision first entry is not patch_embed", errors)
        expect(patch.get("M") == VIT_TOKENS and patch.get("K") == VIT_C and patch.get("N") == VIT_C,
               "patch_embed shape mismatch", errors)
        for idx, layer in enumerate(vision_layers):
            for key in ("name", "w_offset_axi256", "meta_offset_axi256", "M", "K", "N"):
                expect(key in layer, f"vision layer {idx} missing {key}", errors)
            if not all(key in layer for key in ("w_offset_axi256", "meta_offset_axi256", "K", "N")):
                continue
            expect(layer["w_offset_axi256"] * AXI256_BYTES >= 0, f"vision layer {idx} negative weight offset", errors)
            expect(layer["meta_offset_axi256"] * AXI256_BYTES >= 0, f"vision layer {idx} negative meta offset", errors)
            if idx + 1 < len(vision_layers):
                nxt = vision_layers[idx + 1]
                expected_w = packed_w4_bytes(layer["K"], layer["N"])
                expected_m = meta_bytes(layer["K"], layer["N"])
                expect_eq(nxt["w_offset_axi256"] * AXI256_BYTES - layer["w_offset_axi256"] * AXI256_BYTES,
                          expected_w, f"vision layer {idx} packed span mismatch", errors)
                expect_eq(nxt["meta_offset_axi256"] * AXI256_BYTES - layer["meta_offset_axi256"] * AXI256_BYTES,
                          expected_m, f"vision layer {idx} meta span mismatch", errors)

        vision_w_bytes = size(export_dir / "vision_wgt.bin")
        vision_m_bytes = size(export_dir / "vision_meta.bin")
        for idx, layer in enumerate(vision_layers):
            cur_w = layer["w_offset_axi256"] * AXI256_BYTES
            cur_m = layer["meta_offset_axi256"] * AXI256_BYTES
            next_w = vision_w_bytes if idx + 1 == len(vision_layers) else vision_layers[idx + 1]["w_offset_axi256"] * AXI256_BYTES
            next_m = vision_m_bytes if idx + 1 == len(vision_layers) else vision_layers[idx + 1]["meta_offset_axi256"] * AXI256_BYTES
            expect_eq(next_w - cur_w, packed_w4_bytes(layer["K"], layer["N"]),
                      f"vision layer {idx} packed file span mismatch", errors)
            expect_eq(next_m - cur_m, meta_bytes(layer["K"], layer["N"]),
                      f"vision layer {idx} meta file span mismatch", errors)

        for idx in range(0, min(len(vision_layers) - 1, 1 + 12 * 4 - 1), 4):
            group = vision_layers[1 + idx:1 + idx + 4]
            if len(group) < 4:
                break
            expect(group[0]["name"].endswith(".qkv"), f"vision block group {idx//4} missing qkv", errors)
            expect(group[1]["name"].endswith(".attn_out"), f"vision block group {idx//4} missing attn_out", errors)
            expect(group[2]["name"].endswith(".ffn_up"), f"vision block group {idx//4} missing ffn_up", errors)
            expect(group[3]["name"].endswith(".ffn_down"), f"vision block group {idx//4} missing ffn_down", errors)
            expect(group[0]["K"] == VIT_C and group[0]["N"] == VIT_QKV, f"vision block group {idx//4} qkv shape mismatch", errors)
            expect(group[1]["K"] == VIT_C and group[1]["N"] == VIT_C, f"vision block group {idx//4} attn_out shape mismatch", errors)
            expect(group[2]["K"] == VIT_C and group[2]["N"] == VIT_FFN, f"vision block group {idx//4} ffn_up shape mismatch", errors)
            expect(group[3]["K"] == VIT_FFN and group[3]["N"] == VIT_C, f"vision block group {idx//4} ffn_down shape mismatch", errors)

    expect(size(export_dir / "vision_wgt.bin") > 0, "vision_wgt.bin empty", errors)
    expect(size(export_dir / "vision_meta.bin") > 0, "vision_meta.bin empty", errors)
    expect(size(export_dir / "connector_wgt.bin") > 0, "connector_wgt.bin empty", errors)
    expect(size(export_dir / "connector_meta.bin") > 0, "connector_meta.bin empty", errors)
    expect(size(export_dir / "lmhead_wgt_half0.bin") == size(export_dir / "lmhead_wgt_half1.bin"),
           "lmhead weight halves size mismatch", errors)
    expect(size(export_dir / "lmhead_meta_half0.bin") == size(export_dir / "lmhead_meta_half1.bin"),
           "lmhead meta halves size mismatch", errors)
    expect(size(export_dir / "lmhead_wgt_half0.bin") * 2 == (C * VOCAB) // 2,
           "lmhead packed weight total size mismatch", errors)
    expect(size(export_dir / "lmhead_meta_half0.bin") * 2 == (C * VOCAB) // 32 * 2,
           "lmhead packed meta total size mismatch", errors)

    if errors:
        print("hardware export check FAILED")
        for err in errors:
            print(" -", err)
        if any("missing axi offsets" in err for err in errors):
            print("hint: regenerate hardware_export with the updated export.py so decoder_layers include w_offset_axi256/meta_offset_axi256")
        return 1

    print("hardware export check PASS")
    print(f" export_dir = {export_dir}")
    print(f" decoder_layers = {len(decoder_layers)}")
    print(f" vision_entries = {len(vision_layers)}")
    print(f" connector = {connector}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
