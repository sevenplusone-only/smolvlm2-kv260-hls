from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .modeling import QuantizedLinear
from .quant_ops import np_quantize_w4_rtn
from .schemes import QuantSchemeSpec

VIT_TOKENS = 1024
VIT_C = 768
VIT_QKV = 2304
VIT_FFN_DIM = 3072
IMAGE_TOKENS = 64
CONNECTOR_IN = 12288
CONNECTOR_OUT = 960


def save_local_checkpoint(
    model: Any,
    scheme: QuantSchemeSpec,
    output_dir: Path,
    extra_payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scheme": scheme.to_manifest(),
        "state_dict_path": "student_state.pt",
        "extra": extra_payload,
    }
    torch.save(model.state_dict(), output_dir / "student_state.pt")
    (output_dir / "scheme_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def _extract_weight(module: Any) -> np.ndarray:
    if isinstance(module, QuantizedLinear):
        return module.export_fp_weight().numpy()
    return module.weight.detach().float().cpu().numpy()


def _resolve_attr(root: Any, *candidates: str) -> Any:
    for path in candidates:
        cur = root
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise AttributeError(f"Unable to resolve any of: {candidates}")


def _quantize_layer_bundle(layer: Any, group_size: int) -> tuple[bytes, bytes, bytes, bytes, np.ndarray, np.ndarray]:
    attn = layer.self_attn
    mlp = layer.mlp
    norm1 = _resolve_attr(layer, "input_layernorm")
    norm2 = _resolve_attr(layer, "post_feedforward_layernorm", "post_attention_layernorm")

    matrices = [
        np.concatenate([_extract_weight(attn.q_proj), _extract_weight(attn.k_proj), _extract_weight(attn.v_proj)], axis=0),
        _extract_weight(attn.o_proj),
        np.concatenate([_extract_weight(mlp.gate_proj), _extract_weight(mlp.up_proj)], axis=0),
        _extract_weight(mlp.down_proj),
    ]

    weight_half0 = bytearray()
    weight_half1 = bytearray()
    meta_half0 = bytearray()
    meta_half1 = bytearray()

    for matrix in matrices:
        packed, scales, zeros = np_quantize_w4_rtn(matrix, group_size)
        half = packed.shape[0] // 2
        weight_half0.extend(packed[:half].tobytes())
        weight_half1.extend(packed[half:].tobytes())
        meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
        meta_half0.extend(meta[:half].tobytes())
        meta_half1.extend(meta[half:].tobytes())

    gamma_attn = np.clip(np.round(norm1.weight.detach().float().cpu().numpy() * 64.0), -127, 127).astype(np.int8)
    gamma_ffn = np.clip(np.round(norm2.weight.detach().float().cpu().numpy() * 64.0), -127, 127).astype(np.int8)
    return bytes(weight_half0), bytes(weight_half1), bytes(meta_half0), bytes(meta_half1), gamma_attn, gamma_ffn


def export_kv260_w4a8_bundle(model: Any, scheme: QuantSchemeSpec, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = _resolve_attr(model, "model.text_model", "language_model.model", "text_model")
    vision = _resolve_attr(model, "model.vision_model", "vision_model")
    connector = _resolve_attr(model, "model.connector", "connector")

    all_wh0 = bytearray()
    all_wh1 = bytearray()
    all_mh0 = bytearray()
    all_mh1 = bytearray()
    all_ga = bytearray()
    all_gf = bytearray()
    layer_manifest: list[dict[str, Any]] = []

    for index, layer in enumerate(llm.layers):
        offset_w = len(all_wh0)
        offset_m = len(all_mh0)
        wh0, wh1, mh0, mh1, ga, gf = _quantize_layer_bundle(layer, scheme.group_size)
        all_wh0.extend(wh0)
        all_wh1.extend(wh1)
        all_mh0.extend(mh0)
        all_mh1.extend(mh1)
        all_ga.extend(ga.tobytes())
        all_gf.extend(gf.tobytes())
        layer_manifest.append(
            {
                "layer": index,
                "w_offset_bytes": offset_w,
                "meta_offset_bytes": offset_m,
                "w_offset_axi256": offset_w // 32,
                "meta_offset_axi256": offset_m // 32,
                "quant": scheme.weight_format,
            }
        )

    (output_dir / "weights_half0.bin").write_bytes(bytes(all_wh0))
    (output_dir / "weights_half1.bin").write_bytes(bytes(all_wh1))
    (output_dir / "meta_half0.bin").write_bytes(bytes(all_mh0))
    (output_dir / "meta_half1.bin").write_bytes(bytes(all_mh1))
    (output_dir / "gamma_attn.bin").write_bytes(bytes(all_ga))
    (output_dir / "gamma_ffn.bin").write_bytes(bytes(all_gf))

    vision_w = bytearray()
    vision_m = bytearray()
    vision_manifest: list[dict[str, Any]] = []

    patch_embed = _resolve_attr(vision, "embeddings.patch_embedding", "patch_embedding")
    patch_weight = _extract_weight(patch_embed)
    if patch_weight.ndim == 4:
        patch_weight = patch_weight.reshape(patch_weight.shape[0], -1)
    packed, scales, zeros = np_quantize_w4_rtn(patch_weight, scheme.group_size)
    patch_meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
    vision_manifest.append(
        {
            "name": "patch_embed",
            "w_offset_axi256": len(vision_w) // 32,
            "meta_offset_axi256": len(vision_m) // 32,
            "M": VIT_TOKENS,
            "K": VIT_C,
            "N": VIT_C,
        }
    )
    vision_w.extend(packed.tobytes())
    vision_m.extend(patch_meta.tobytes())

    for index, layer in enumerate(vision.encoder.layers):
        matrices = [
            ("qkv", np.concatenate([
                _extract_weight(layer.self_attn.q_proj),
                _extract_weight(layer.self_attn.k_proj),
                _extract_weight(layer.self_attn.v_proj),
            ], axis=0), VIT_C, VIT_QKV),
            ("attn_out", _extract_weight(layer.self_attn.out_proj), VIT_C, VIT_C),
            ("ffn_up", _extract_weight(layer.mlp.fc1), VIT_C, VIT_FFN_DIM),
            ("ffn_down", _extract_weight(layer.mlp.fc2), VIT_FFN_DIM, VIT_C),
        ]
        for name, matrix, k_dim, n_dim in matrices:
            packed, scales, zeros = np_quantize_w4_rtn(matrix, scheme.group_size)
            vision_meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
            vision_manifest.append(
                {
                    "name": f"blk.{index}.{name}",
                    "w_offset_axi256": len(vision_w) // 32,
                    "meta_offset_axi256": len(vision_m) // 32,
                    "M": VIT_TOKENS,
                    "K": k_dim,
                    "N": n_dim,
                }
            )
            vision_w.extend(packed.tobytes())
            vision_m.extend(vision_meta.tobytes())

    connector_proj = _resolve_attr(
        connector,
        "modality_projection.proj",
        "proj",
    )
    connector_w, connector_scales, connector_zeros = np_quantize_w4_rtn(
        _extract_weight(connector_proj),
        scheme.group_size,
    )
    connector_meta = np.stack([connector_scales, connector_zeros], axis=-1).reshape(connector_w.shape[0], -1)

    (output_dir / "vision_wgt.bin").write_bytes(bytes(vision_w))
    (output_dir / "vision_meta.bin").write_bytes(bytes(vision_m))
    (output_dir / "connector_wgt.bin").write_bytes(connector_w.tobytes())
    (output_dir / "connector_meta.bin").write_bytes(connector_meta.tobytes())

    lm_head = _extract_weight(model.lm_head)
    packed, scales, zeros = np_quantize_w4_rtn(lm_head, scheme.group_size)
    half = packed.shape[0] // 2
    (output_dir / "lmhead_wgt_half0.bin").write_bytes(packed[:half].tobytes())
    (output_dir / "lmhead_wgt_half1.bin").write_bytes(packed[half:].tobytes())
    lm_meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
    (output_dir / "lmhead_meta_half0.bin").write_bytes(lm_meta[:half].tobytes())
    (output_dir / "lmhead_meta_half1.bin").write_bytes(lm_meta[half:].tobytes())

    awq_ready = "native_w4a8_ready_host_folded_awq" if scheme.awq_enabled else "native_w4a8_ready"

    connector_manifest = {
        "name": "connector.projector",
        "w_offset_axi256": 0,
        "meta_offset_axi256": 0,
        "M": IMAGE_TOKENS,
        "K": CONNECTOR_IN,
        "N": CONNECTOR_OUT,
    }

    export_manifest = {
        "scheme": scheme.to_manifest(),
        "kv260_status": awq_ready,
        "decoder_layers": layer_manifest,
        "vision_layers": vision_manifest,
        "connector": connector_manifest,
        "artifacts": [
            "vision_wgt.bin",
            "vision_meta.bin",
            "connector_wgt.bin",
            "connector_meta.bin",
            "weights_half0.bin",
            "weights_half1.bin",
            "meta_half0.bin",
            "meta_half1.bin",
            "gamma_attn.bin",
            "gamma_ffn.bin",
            "lmhead_wgt_half0.bin",
            "lmhead_wgt_half1.bin",
            "lmhead_meta_half0.bin",
            "lmhead_meta_half1.bin",
        ],
    }
    (output_dir / "hardware_export_manifest.json").write_text(
        json.dumps(export_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return export_manifest


def export_hardware_bundle(model: Any, scheme: QuantSchemeSpec, output_dir: Path) -> dict[str, Any]:
    if scheme.export_mode.startswith("kv260_w4a8"):
        return export_kv260_w4a8_bundle(model, scheme, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scheme": scheme.to_manifest(),
        "kv260_status": "replay_required",
        "next_steps": [
            "run layer replay against PyTorch reference",
            "run kernel C-sim and resource check on Linux/Vitis host",
            "integrate with board path only after replay parity",
        ],
    }
    (output_dir / "hardware_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
