#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.json"

EXPECTED = {
    "image_token_id": 49190,
    "scale_factor": 4,
    "text.hidden_size": 960,
    "text.intermediate_size": 2560,
    "text.num_attention_heads": 15,
    "text.num_key_value_heads": 5,
    "text.num_hidden_layers": 32,
    "text.vocab_size": 49280,
    "text.pixel_shuffle_factor": 4,
    "text.use_resampler": False,
    "vision.hidden_size": 768,
    "vision.image_size": 512,
    "vision.patch_size": 16,
    "vision.num_attention_heads": 12,
}

DERIVED = {
    "vision.patch_tokens": 1024,
    "vision.qkv": 2304,
    "vision.ffn": 3072,
    "connector.tokens": 64,
    "connector.in": 12288,
    "connector.out": 960,
    "llm.qkv": 1600,
    "llm.gate_up": 5120,
}

def main():
    cfg = json.loads(CONFIG.read_text())
    text = cfg["text_config"]
    vision = cfg["vision_config"]
    got = {
        "image_token_id": cfg["image_token_id"],
        "scale_factor": cfg["scale_factor"],
        "text.hidden_size": text["hidden_size"],
        "text.intermediate_size": text["intermediate_size"],
        "text.num_attention_heads": text["num_attention_heads"],
        "text.num_key_value_heads": text["num_key_value_heads"],
        "text.num_hidden_layers": text["num_hidden_layers"],
        "text.vocab_size": text["vocab_size"],
        "text.pixel_shuffle_factor": text["pixel_shuffle_factor"],
        "text.use_resampler": text["use_resampler"],
        "vision.hidden_size": vision["hidden_size"],
        "vision.image_size": vision["image_size"],
        "vision.patch_size": vision["patch_size"],
        "vision.num_attention_heads": vision["num_attention_heads"],
    }
    got.update({
        "vision.patch_tokens": (vision["image_size"] // vision["patch_size"]) ** 2,
        "vision.qkv": vision["hidden_size"] * 3,
        "vision.ffn": vision["hidden_size"] * 4,
        "connector.tokens": (vision["image_size"] // vision["patch_size"]) ** 2 // (text["pixel_shuffle_factor"] ** 2),
        "connector.in": vision["hidden_size"] * text["pixel_shuffle_factor"] ** 2,
        "connector.out": text["hidden_size"],
        "llm.qkv": text["hidden_size"] + text["num_key_value_heads"] * text["head_dim"] * 2,
        "llm.gate_up": text["intermediate_size"] * 2,
    })

    expected = {**EXPECTED, **DERIVED}
    errors = []
    for key, want in expected.items():
        if got[key] != want:
            errors.append(f"{key}: got {got[key]} expected {want}")

    if errors:
        print("Dimension check FAILED")
        for err in errors:
            print("  " + err)
        return 1

    print("Dimension check PASS")
    for key in sorted(expected):
        print(f"  {key} = {got[key]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
