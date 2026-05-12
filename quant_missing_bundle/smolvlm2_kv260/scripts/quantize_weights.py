#!/usr/bin/env python3
# =============================================================================
# quantize_weights.py  —  SmolVLM2-500M W4 离线量化工具
# =============================================================================
# 功能：
#   ① 从 HuggingFace 加载 SmolVLM2-500M fp16 权重
#   ② 对所有线性层做 RTN W4 量化，group_size=32
#   ③ 导出二进制格式：
#      - weights.bin：W4 权重（nibble 打包，层序 + 矩阵序排列）
#      - meta.bin：   GroupMeta（scale INT8 + zero_point INT8）
#      - gamma.bin：  RMSNorm gamma（INT8）
#      - lmhead.bin： lm_head 权重（W4，同格式）
# 依赖：pip install torch transformers
# =============================================================================
import os
import sys
import struct
import json
import re
import numpy as np
import torch
from pathlib import Path

MODEL_ID   = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GROUP_SIZE = 32
OUTPUT_DIR = Path("./quantized_weights")
VISION_HIDDEN = 768
VISION_TOKENS = 1024
VISION_QKV = 2304
VISION_FFN = 3072
CONNECTOR_IN = 12288
CONNECTOR_OUT = 960

# ---------------------------------------------------------------------------
# RTN（Round-To-Nearest）W4 量化，per-group，group_size=32
# ---------------------------------------------------------------------------
def quantize_w4_rtn(weight_fp32: np.ndarray, group_size: int = 32):
    """
    Args:
        weight_fp32: shape [out_features, in_features]，float32
        group_size:  量化组大小（沿 in_features 方向分组）
    Returns:
        w_q:    uint8 array，shape [out_features, in_features//2]（nibble 打包）
        scales: int8 array，shape [out_features, n_groups]
        zeros:  int8 array，shape [out_features, n_groups]（zero_point）
    """
    out_f, in_f = weight_fp32.shape
    assert in_f % group_size == 0, f"in_features {in_f} must be divisible by {group_size}"
    n_groups = in_f // group_size

    w = weight_fp32.reshape(out_f, n_groups, group_size)  # [out, n_grp, grp]

    # per-group absmax
    w_max  = np.abs(w).max(axis=-1, keepdims=True)       # [out, n_grp, 1]
    w_max  = np.clip(w_max, 1e-8, None)

    # 量化到 [0, 15]（无符号 INT4）
    scale   = w_max / 7.5                                 # 使得 absmax → ±7.5（居中）
    zero_pt = 8.0                                          # 对称量化的零点偏移

    w_q_float = np.round(w / scale + zero_pt)
    w_q_float = np.clip(w_q_float, 0, 15)
    w_q       = w_q_float.astype(np.uint8)                # [out, n_grp, grp]

    # 压缩 scale 到 INT8（scale_fp32 / 2^shift → INT8）
    # 简化：将 scale 归一化到 [-127, 127] 范围，记录移位量
    scale_sq = scale.squeeze(-1)                           # [out, n_grp]
    scale_max = np.abs(scale_sq).max()
    scale_shift = max(0, int(np.ceil(np.log2(scale_max + 1e-8))) - 6)
    scale_int8 = np.clip(np.round(scale_sq / (2**scale_shift)), -127, 127).astype(np.int8)
    zero_int8  = np.full_like(scale_int8, 8, dtype=np.int8)  # zero_point=8（固定）

    # nibble 打包：将 [grp_size] 的 uint4 → [grp_size/2] 的 uint8
    w_q_flat = w_q.reshape(out_f, n_groups * group_size)  # [out, in]
    w_packed = np.zeros((out_f, in_f // 2), dtype=np.uint8)
    w_packed = (w_q_flat[:, 0::2] & 0xF) | ((w_q_flat[:, 1::2] & 0xF) << 4)

    return w_packed, scale_int8, zero_int8, scale_shift

# ---------------------------------------------------------------------------
# 矩阵列拼接（QKV 融合、FFN gate/up 融合）
# ---------------------------------------------------------------------------
def concat_cols(*matrices):
    """沿列方向（out_features 方向）拼接权重矩阵"""
    return np.concatenate(matrices, axis=0)

def append_packed_matrix(blob_w, blob_m, manifest, name, matrix, group_size=GROUP_SIZE):
    packed, scales, zeros, shift = quantize_w4_rtn(matrix, group_size)
    w_offset = len(blob_w)
    m_offset = len(blob_m)
    meta = np.stack([scales, zeros], axis=-1).reshape(packed.shape[0], -1)
    blob_w += packed.tobytes()
    blob_m += meta.tobytes()
    manifest.append({
        "name": name,
        "M": None,
        "K": int(matrix.shape[1]),
        "N": int(matrix.shape[0]),
        "w_offset_bytes": w_offset,
        "meta_offset_bytes": m_offset,
        "w_offset_axi256": w_offset // 32,
        "meta_offset_axi256": m_offset // 32,
        "scale_shift": int(shift),
        "quant": "rtn_w4a8_g32"
    })
    return blob_w, blob_m

def quantize_q8_0_rtn(weight_fp32: np.ndarray, block_size: int = 32) -> bytes:
    """Pack weights into GGML Q8_0 block layout for native ggml-xrt fallback."""
    out_f, in_f = weight_fp32.shape
    assert in_f % block_size == 0, f"in_features {in_f} must be divisible by {block_size}"
    n_blocks = in_f // block_size

    w = weight_fp32.reshape(out_f, n_blocks, block_size)
    w_absmax = np.clip(np.abs(w).max(axis=-1, keepdims=True), 1e-8, None)
    scale = (w_absmax / 127.0).astype(np.float32)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)

    blob = bytearray()
    for r in range(out_f):
        for b in range(n_blocks):
            blob.extend(np.array(scale[r, b, 0], dtype=np.float16).tobytes())
            blob.extend(q[r, b].tobytes())

    return bytes(blob)

def append_native_q8_tensor(blob, tensors, domain, name, matrix):
    packed = quantize_q8_0_rtn(matrix)
    offset = len(blob)
    blob += packed
    entry = {
        "domain": domain,
        "name": name,
        "K": int(matrix.shape[1]),
        "N": int(matrix.shape[0]),
        "offset_bytes": offset,
        "nbytes": len(packed),
        "quant": "q8_0",
    }
    tensors.append(entry)
    return blob

def append_native_q8_tensor_aliases(blob, tensors, domain, names, matrix):
    names = [n for n in names if n]
    if not names:
        return blob
    blob = append_native_q8_tensor(blob, tensors, domain, names[0], matrix)
    base = tensors[-1]
    for alias in names[1:]:
        tensors.append({
            "domain": domain,
            "name": alias,
            "K": base["K"],
            "N": base["N"],
            "offset_bytes": base["offset_bytes"],
            "nbytes": base["nbytes"],
            "quant": base["quant"],
        })
    return blob

def canonical_decoder_name(layer_idx, kind):
    mapping = {
        "q": f"blk.{layer_idx}.attn_q.weight",
        "k": f"blk.{layer_idx}.attn_k.weight",
        "v": f"blk.{layer_idx}.attn_v.weight",
        "o": f"blk.{layer_idx}.attn_output.weight",
        "gate": f"blk.{layer_idx}.ffn_gate.weight",
        "up": f"blk.{layer_idx}.ffn_up.weight",
        "down": f"blk.{layer_idx}.ffn_down.weight",
    }
    return mapping[kind]

def canonical_multimodal_name(hf_name):
    patterns = [
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj", lambda m: ["v.blk.%s.attn_q.weight" % m.group(1)]),
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj", lambda m: ["v.blk.%s.attn_k.weight" % m.group(1)]),
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj", lambda m: ["v.blk.%s.attn_v.weight" % m.group(1)]),
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.self_attn\.o_proj", lambda m: [f"v.blk.{m.group(1)}.attn_out.weight", f"v.blk.{m.group(1)}.attn_output.weight"]),
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1", lambda m: [f"v.blk.{m.group(1)}.ffn_up.weight"]),
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2", lambda m: [f"v.blk.{m.group(1)}.ffn_down.weight"]),
        (r"(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.mlp\.gate_proj", lambda m: [f"v.blk.{m.group(1)}.ffn_gate.weight"]),
        (r"(?:model\.)?connector\.modality_projection\.proj", lambda m: ["mm.model.fc.weight", "mm.0.weight", "mm.1.weight"]),
        (r"(?:model\.)?multi_modal_projector\.(?:linear_1|0)", lambda m: ["mm.0.weight"]),
        (r"(?:model\.)?multi_modal_projector\.(?:linear_2|2)", lambda m: ["mm.1.weight"]),
        (r"(?:model\.)?vision_model\.post_layernorm", lambda m: ["v.post_ln.weight"]),
        (r"(?:model\.)?vision_model\.embeddings\.patch_embedding", lambda m: ["v.patch_embd.weight", "vision.patch_embed"]),
    ]
    for pattern, builder in patterns:
        match = re.search(pattern, hf_name)
        if match:
            return builder(match)
    return None

def export_shape_matched_vision(model, out_dir, manifest):
    """Best-effort clean-room exporter for config-matched ViT/connector matrices.

    The exact HF module names vary between SmolVLM releases, so this scans
    Linear/Conv2d weights by shape and writes the matrices needed by the HLS
    runner.  Names in manifest remain descriptive and independent of any
    reference project layout.
    """
    vision_w = b""
    vision_m = b""
    connector_w = b""
    connector_m = b""

    linears = []
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linears.append((name, module.weight.detach().float().cpu().numpy()))
        elif isinstance(module, torch.nn.Conv2d):
            convs.append((name, module.weight.detach().float().cpu().numpy()))

    # Patch embedding: Conv2d [768, 3, 16, 16] flattens to [768, 768].
    for name, weight in convs:
        if weight.shape[0] == VISION_HIDDEN and int(np.prod(weight.shape[1:])) == VISION_HIDDEN:
            mat = weight.reshape(VISION_HIDDEN, VISION_HIDDEN)
            vision_w, vision_m = append_packed_matrix(
                vision_w, vision_m, manifest["vision"], f"vision.patch_embed:{name}", mat)
            break

    for name, weight in linears:
        out_f, in_f = weight.shape
        if (out_f, in_f) in {
            (VISION_QKV, VISION_HIDDEN),
            (VISION_HIDDEN, VISION_HIDDEN),
            (VISION_FFN, VISION_HIDDEN),
            (VISION_HIDDEN, VISION_FFN),
        }:
            vision_w, vision_m = append_packed_matrix(
                vision_w, vision_m, manifest["vision"], f"vision.linear:{name}", weight)
        elif (out_f, in_f) == (CONNECTOR_OUT, CONNECTOR_IN):
            connector_w, connector_m = append_packed_matrix(
                connector_w, connector_m, manifest["connector"], f"connector.projector:{name}", weight)

    (out_dir / "vision_wgt.bin").write_bytes(vision_w)
    (out_dir / "vision_meta.bin").write_bytes(vision_m)
    (out_dir / "connector_wgt.bin").write_bytes(connector_w)
    (out_dir / "connector_meta.bin").write_bytes(connector_m)

# ---------------------------------------------------------------------------
# 导出单层权重
# ---------------------------------------------------------------------------
def export_layer(layer_idx, layer, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取各线性层权重（HuggingFace 格式：weight.shape = [out, in]）
    # SmolVLM2-500M LLM decoder 层路径：
    # model.language_model.model.layers[i].self_attn.{q,k,v,o}_proj.weight
    # model.language_model.model.layers[i].mlp.{gate,up,down}_proj.weight

    attn    = layer.self_attn
    mlp     = layer.mlp
    norm1   = layer.input_layernorm
    norm2   = layer.post_feedforward_layernorm

    # Q, K, V 权重（列拼接：QKV 融合）
    w_q = attn.q_proj.weight.float().numpy()   # [960, 960]
    w_k = attn.k_proj.weight.float().numpy()   # [320, 960]
    w_v = attn.v_proj.weight.float().numpy()   # [320, 960]
    w_qkv = concat_cols(w_q, w_k, w_v)         # [1600, 960]

    # O proj
    w_o = attn.o_proj.weight.float().numpy()   # [960, 960]

    # FFN gate/up（列拼接）
    w_gate = mlp.gate_proj.weight.float().numpy()  # [2560, 960]
    w_up   = mlp.up_proj.weight.float().numpy()    # [2560, 960]
    w_gateup = concat_cols(w_gate, w_up)            # [5120, 960]

    # FFN down
    w_down = mlp.down_proj.weight.float().numpy()  # [960, 2560]

    print(f"  Layer {layer_idx}: QKV{w_qkv.shape} O{w_o.shape} "
          f"GateUp{w_gateup.shape} Down{w_down.shape}")

    # 量化并打包
    results = {}
    for name, mat in [("qkv", w_qkv), ("o", w_o), ("gateup", w_gateup), ("down", w_down)]:
        packed, scales, zeros, shift = quantize_w4_rtn(mat, GROUP_SIZE)
        results[name] = (packed, scales, zeros)
        print(f"    {name}: packed{packed.shape} scale{scales.shape} shift={shift}")

    # 写二进制：权重（前半 / 后半 交错，适配双端口）
    # 格式：[QKV_packed | O_packed | GateUp_packed | Down_packed]
    # 前半 = 每个矩阵的前 out_features/2 行
    # 后半 = 后 out_features/2 行
    layer_wgt_half0  = b""
    layer_wgt_half1  = b""
    layer_meta_half0 = b""
    layer_meta_half1 = b""

    for name in ["qkv", "o", "gateup", "down"]:
        packed, scales, zeros = results[name]
        n_rows = packed.shape[0]
        half   = n_rows // 2

        layer_wgt_half0  += packed[:half].tobytes()
        layer_wgt_half1  += packed[half:].tobytes()

        # meta：(scale, zero) 交错存储，按列顺序
        meta = np.stack([scales, zeros], axis=-1)  # [out, n_grp, 2]
        meta_flat = meta.reshape(n_rows, -1)        # [out, n_grp*2]
        layer_meta_half0 += meta_flat[:half].tobytes()
        layer_meta_half1 += meta_flat[half:].tobytes()

    # RMSNorm gamma（INT8，C=960）
    g1 = norm1.weight.float().numpy()
    g2 = norm2.weight.float().numpy()
    # 简化归一化：将 fp32 gamma 映射到 INT8（× 64，右移 6 即可）
    g1_int8 = np.clip(np.round(g1 * 64), -127, 127).astype(np.int8)
    g2_int8 = np.clip(np.round(g2 * 64), -127, 127).astype(np.int8)

    return (layer_wgt_half0, layer_wgt_half1,
            layer_meta_half0, layer_meta_half1,
            g1_int8, g2_int8)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import transformers
    print(f"Loading SmolVLM2-500M from HuggingFace: {MODEL_ID}")
    print("(This may take several minutes and ~2GB download)")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    # 获取 LLM decoder 层（SmolVLM2 语言模型部分）
    llm    = model.language_model.model
    layers = llm.layers
    print(f"Found {len(layers)} decoder layers")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_model": MODEL_ID,
        "version": 2,
        "config_match": {
            "vision_tokens": VISION_TOKENS,
            "vision_hidden": VISION_HIDDEN,
            "vision_qkv": VISION_QKV,
            "vision_ffn": VISION_FFN,
            "pixel_shuffle_factor": 4,
            "connector_in": CONNECTOR_IN,
            "connector_out": CONNECTOR_OUT,
            "llm_hidden": 960,
            "llm_qkv": 1600,
            "llm_ffn_gateup": 5120,
            "vocab": 49280,
            "image_token_id": 49190,
            "use_resampler": False,
        },
        "native_backend": {
            "weights_file": "native_q8_0_weights.bin",
            "tensor_quant": "q8_0",
            "tensors": [],
        },
        "decoder": [],
        "vision": [],
        "connector": [],
        "lmhead": []
    }

    # 合并所有层的权重到两个大文件
    all_wgt_half0  = b""
    all_wgt_half1  = b""
    all_meta_half0 = b""
    all_meta_half1 = b""
    all_gamma_attn = b""
    all_gamma_ffn  = b""
    native_blob = b""

    for i, layer in enumerate(layers):
        print(f"\nQuantizing layer {i}/{len(layers)-1}...")
        layer_wgt_offset = len(all_wgt_half0)
        layer_meta_offset = len(all_meta_half0)
        (wh0, wh1, mh0, mh1, ga, gf) = export_layer(i, layer, OUTPUT_DIR)
        all_wgt_half0  += wh0
        all_wgt_half1  += wh1
        all_meta_half0 += mh0
        all_meta_half1 += mh1
        all_gamma_attn += ga.tobytes()
        all_gamma_ffn  += gf.tobytes()
        manifest["decoder"].append({
            "layer": i,
            "w_offset_bytes": layer_wgt_offset,
            "meta_offset_bytes": layer_meta_offset,
            "w_offset_axi256": layer_wgt_offset // 32,
            "meta_offset_axi256": layer_meta_offset // 32,
            "matrices": [
                {"name": "qkv", "M": None, "K": 960, "N": 1600},
                {"name": "o_proj", "M": None, "K": 960, "N": 960},
                {"name": "gate_up", "M": None, "K": 960, "N": 5120},
                {"name": "down", "M": None, "K": 2560, "N": 960},
            ]
        })

        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "q")], layer.self_attn.q_proj.weight.float().numpy())
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "k")], layer.self_attn.k_proj.weight.float().numpy())
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "v")], layer.self_attn.v_proj.weight.float().numpy())
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "o")], layer.self_attn.o_proj.weight.float().numpy())
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "gate")], layer.mlp.gate_proj.weight.float().numpy())
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "up")], layer.mlp.up_proj.weight.float().numpy())
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "text",
            [canonical_decoder_name(i, "down")], layer.mlp.down_proj.weight.float().numpy())

    # 写文件
    (OUTPUT_DIR / "weights_half0.bin").write_bytes(all_wgt_half0)
    (OUTPUT_DIR / "weights_half1.bin").write_bytes(all_wgt_half1)
    (OUTPUT_DIR / "meta_half0.bin").write_bytes(all_meta_half0)
    (OUTPUT_DIR / "meta_half1.bin").write_bytes(all_meta_half1)
    (OUTPUT_DIR / "gamma_attn.bin").write_bytes(all_gamma_attn)
    (OUTPUT_DIR / "gamma_ffn.bin").write_bytes(all_gamma_ffn)
    (OUTPUT_DIR / "native_q8_0_weights.bin").write_bytes(native_blob)

    # lm_head
    print("\nQuantizing lm_head...")
    lm_weight = model.language_model.lm_head.weight.float().numpy()  # [49280, 960]
    lm_packed, lm_scales, lm_zeros, _ = quantize_w4_rtn(lm_weight, GROUP_SIZE)
    half = lm_packed.shape[0] // 2
    (OUTPUT_DIR / "lmhead_wgt_half0.bin").write_bytes(lm_packed[:half].tobytes())
    (OUTPUT_DIR / "lmhead_wgt_half1.bin").write_bytes(lm_packed[half:].tobytes())
    lm_meta = np.stack([lm_scales, lm_zeros], axis=-1)
    lm_meta_flat = lm_meta.reshape(lm_packed.shape[0], -1)
    (OUTPUT_DIR / "lmhead_meta_half0.bin").write_bytes(lm_meta_flat[:half].tobytes())
    (OUTPUT_DIR / "lmhead_meta_half1.bin").write_bytes(lm_meta_flat[half:].tobytes())
    manifest["lmhead"].append({
        "name": "lm_head",
        "M": 1,
        "K": 960,
        "N": 49280,
        "files": ["lmhead_wgt_half0.bin", "lmhead_wgt_half1.bin",
                  "lmhead_meta_half0.bin", "lmhead_meta_half1.bin"],
        "quant": "rtn_w4a8_g32"
    })
    native_blob = append_native_q8_tensor_aliases(
        native_blob, manifest["native_backend"]["tensors"], "text", ["output.weight", "output"], lm_weight)
    (OUTPUT_DIR / "native_q8_0_weights.bin").write_bytes(native_blob)

    print("\nExporting shape-matched ViT/connector matrices...")
    export_shape_matched_vision(model, OUTPUT_DIR, manifest)

    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        canon = canonical_multimodal_name(name)
        if canon is None:
            continue
        weight = module.weight.detach().float().cpu().numpy()
        if weight.ndim == 4:
            out_f = weight.shape[0]
            in_f = int(np.prod(weight.shape[1:]))
            if in_f % GROUP_SIZE != 0:
                continue
            weight = weight.reshape(out_f, in_f)
        if weight.shape[1] % GROUP_SIZE != 0:
            continue
        native_blob = append_native_q8_tensor_aliases(
            native_blob, manifest["native_backend"]["tensors"], "vision", canon, weight)

    (OUTPUT_DIR / "native_q8_0_weights.bin").write_bytes(native_blob)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 统计
    total_bytes = (len(all_wgt_half0) + len(all_wgt_half1) +
                   len(all_meta_half0) + len(all_meta_half1))
    print(f"\n=== Quantization Complete ===")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Manifest: manifest.json")
    print(f"Total weight bytes: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"(fp16 original: ~{total_bytes * 4 / 1024 / 1024:.1f} MB)")
    print(f"Compression ratio: 4x")

if __name__ == "__main__":
    main()
