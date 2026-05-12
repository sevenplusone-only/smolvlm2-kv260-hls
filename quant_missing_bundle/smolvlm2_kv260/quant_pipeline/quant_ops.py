from __future__ import annotations

import math
from itertools import combinations_with_replacement, product
from typing import Iterable

import numpy as np
import torch


def _safe_absmax(tensor: torch.Tensor, dim: int | tuple[int, ...], keepdim: bool = True) -> torch.Tensor:
    return tensor.abs().amax(dim=dim, keepdim=keepdim).clamp(min=1e-8)


def w4_fake_quant(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    orig_dtype = weight.dtype
    wf = weight.float()
    out_features, in_features = wf.shape
    pad = (group_size - in_features % group_size) % group_size
    if pad:
        wf = torch.cat([wf, torch.zeros(out_features, pad, device=wf.device)], dim=1)
    grouped = wf.reshape(out_features, -1, group_size)
    scale = _safe_absmax(grouped, dim=-1, keepdim=True) / 7.0
    quant = (grouped / scale).round().clamp(-8, 7)
    dequant = (quant * scale).reshape(out_features, -1)
    if pad:
        dequant = dequant[:, :in_features]
    return dequant.to(orig_dtype)


def a8_per_token_fake_quant(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    scale = _safe_absmax(xf, dim=-1, keepdim=True) / 127.0
    quant = (xf / scale).round().clamp(-128, 127)
    return (quant * scale).to(orig_dtype)


def bfp_fake_quant(x: torch.Tensor, block_size: int, mantissa_levels: int = 127) -> torch.Tensor:
    orig_dtype = x.dtype
    flat = x.float().reshape(-1)
    total = flat.numel()
    pad = (block_size - total % block_size) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    blocks = flat.reshape(-1, block_size)
    max_abs = blocks.abs().amax(dim=1, keepdim=True).clamp(min=2 ** -16)
    shared_exp = torch.floor(torch.log2(max_abs))
    scale = 2.0 ** shared_exp
    normalized = blocks / scale
    quant = (normalized * mantissa_levels).round().clamp(-mantissa_levels, mantissa_levels) / mantissa_levels
    out = (quant * scale).reshape(-1)
    if pad:
        out = out[:total]
    return out.reshape_as(x).to(orig_dtype)


def build_apot_codebook(bits: int, *, max_terms: int = 2, max_shift: int = 7) -> torch.Tensor:
    magnitudes = [0.0]
    shifts = list(range(max_shift + 1))
    for terms in range(1, max_terms + 1):
        for combo in combinations_with_replacement(shifts, terms):
            for signs in product((-1.0, 1.0), repeat=terms):
                value = 0.0
                for sign, shift in zip(signs, combo):
                    value += sign * (2.0 ** (-shift))
                magnitudes.append(value)
    codebook = sorted(set(magnitudes))
    target_size = 2 ** bits
    if len(codebook) > target_size:
        idx = np.linspace(0, len(codebook) - 1, num=target_size, dtype=int)
        codebook = [codebook[i] for i in idx]
    return torch.tensor(codebook, dtype=torch.float32)


def apot_fake_quant(x: torch.Tensor, bits: int, *, max_terms: int = 2, max_shift: int = 7) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    if xf.ndim == 1:
        xf = xf.unsqueeze(0)
    codebook = build_apot_codebook(bits, max_terms=max_terms, max_shift=max_shift).to(xf.device)
    scale = _safe_absmax(xf, dim=-1, keepdim=True)
    norm = xf / scale
    diff = (norm.unsqueeze(-1) - codebook).abs()
    indices = diff.argmin(dim=-1)
    out = codebook[indices] * scale
    return out.reshape_as(x).to(orig_dtype)


def quantize_tensor_by_format(
    tensor: torch.Tensor,
    fmt: str,
    *,
    group_size: int,
    block_size: int,
    bits: int,
) -> torch.Tensor:
    if fmt in {"w4_uniform_rtn", "w4_awq_rtn"}:
        return w4_fake_quant(tensor, group_size)
    if fmt == "a8_per_token":
        return a8_per_token_fake_quant(tensor)
    if fmt == "bfp8":
        return bfp_fake_quant(tensor, block_size=block_size)
    if fmt == "apot4":
        return apot_fake_quant(tensor, bits=bits, max_terms=2, max_shift=6)
    if fmt == "apot8":
        return apot_fake_quant(tensor, bits=bits, max_terms=2, max_shift=7)
    raise ValueError(f"Unsupported quant format: {fmt}")


def np_quantize_w4_rtn(weight_fp32: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_features, in_features = weight_fp32.shape
    assert in_features % group_size == 0, f"in_features={in_features} must divide group_size={group_size}"
    grouped = weight_fp32.reshape(out_features, -1, group_size)
    max_abs = np.clip(np.abs(grouped).max(axis=-1, keepdims=True), 1e-8, None)
    scale = max_abs / 7.0
    quant = np.round(grouped / scale).clip(-8, 7).astype(np.int8)
    stored = (quant + 8).astype(np.uint8)
    flat = stored.reshape(out_features, in_features)
    packed = (flat[:, 0::2] & 0x0F) | ((flat[:, 1::2] & 0x0F) << 4)
    scales = np.clip(np.round(scale.squeeze(-1) * 64.0), -127, 127).astype(np.int8)
    zeros = np.full_like(scales, 8, dtype=np.int8)
    return packed, scales, zeros


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def prediction_matches_any(prediction: str, answers: Iterable[str]) -> bool:
    norm_pred = normalize_text(prediction)
    return any(normalize_text(answer) in norm_pred for answer in answers if answer)


def stable_softmax_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_student = student_logits.float() / temperature
    scaled_teacher = teacher_logits.float() / temperature
    log_probs = torch.log_softmax(scaled_student, dim=-1)
    targets = torch.softmax(scaled_teacher, dim=-1)
    loss = torch.nn.functional.kl_div(log_probs, targets, reduction="batchmean")
    return loss * (temperature ** 2)


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(loss)) if math.isfinite(loss) else float("inf")
