#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


AXI256_BYTES = 32


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_export = root / "quant_experiments_awq_repair_20260508_010000" / "w4a8_awq_g32" / "hardware_export"
    parser = argparse.ArgumentParser(description="Add missing AXI256 offsets to an existing hardware_export manifest.")
    parser.add_argument("export_dir", nargs="?", type=Path, default=default_export)
    parser.add_argument("--in-place", action="store_true", help="Overwrite manifest instead of writing .repaired.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.export_dir / "hardware_export_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = 0
    for layer in payload.get("decoder_layers", []):
        if "w_offset_axi256" not in layer:
            if layer["w_offset_bytes"] % AXI256_BYTES != 0:
                raise SystemExit(f"layer {layer.get('layer')} weight offset is not AXI256 aligned")
            layer["w_offset_axi256"] = layer["w_offset_bytes"] // AXI256_BYTES
            changed += 1
        if "meta_offset_axi256" not in layer:
            if layer["meta_offset_bytes"] % AXI256_BYTES != 0:
                raise SystemExit(f"layer {layer.get('layer')} meta offset is not AXI256 aligned")
            layer["meta_offset_axi256"] = layer["meta_offset_bytes"] // AXI256_BYTES
            changed += 1

    out_path = manifest_path if args.in_place else manifest_path.with_suffix(".repaired.json")
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"fields_added = {changed}")
    if not args.in_place:
        print("run with --in-place only after you accept that this repairs metadata, not binary contents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
