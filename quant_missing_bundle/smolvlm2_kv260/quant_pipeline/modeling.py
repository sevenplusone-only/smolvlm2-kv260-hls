from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_ops import quantize_tensor_by_format
from .schemes import QuantSchemeSpec


@dataclass
class ZeroSkipStat:
    module_name: str
    zero_count: int = 0
    total_count: int = 0

    def ratio(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.zero_count / self.total_count


class ZeroSkipCollector:
    def __init__(self) -> None:
        self.stats: dict[str, ZeroSkipStat] = {}

    def record(self, module_name: str, tensor: torch.Tensor) -> None:
        stat = self.stats.setdefault(module_name, ZeroSkipStat(module_name=module_name))
        stat.zero_count += int((tensor == 0).sum().item())
        stat.total_count += int(tensor.numel())

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "zero_count": stat.zero_count,
                "total_count": stat.total_count,
                "zero_ratio": stat.ratio(),
            }
            for name, stat in self.stats.items()
        }


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        name: str,
        linear: nn.Linear,
        scheme: QuantSchemeSpec,
        *,
        collector: ZeroSkipCollector | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.scheme = scheme
        self.collector = collector
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        rank = scheme.lora_rank
        if rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, dtype=self.weight.dtype))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=self.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        self.register_buffer("awq_input_scale", torch.ones(self.in_features), persistent=True)
        self.register_buffer("has_awq_scale", torch.tensor(False), persistent=True)

    def set_awq_scale(self, scale: torch.Tensor) -> None:
        self.awq_input_scale = scale.detach().to(self.weight.device, dtype=self.weight.dtype)
        self.has_awq_scale = torch.tensor(True, device=self.weight.device)

    def _effective_weight(self) -> torch.Tensor:
        weight = self.weight
        if self.lora_a is not None and self.lora_b is not None:
            delta = (self.lora_b @ self.lora_a).to(weight.dtype)
            weight = weight + delta
        return weight

    def export_fp_weight(self) -> torch.Tensor:
        weight = self._effective_weight().detach().float().cpu()
        if bool(self.has_awq_scale.item()):
            weight = weight * self.awq_input_scale.detach().float().cpu().unsqueeze(0)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work_x = x
        weight = self._effective_weight()
        if bool(self.has_awq_scale.item()):
            scale = self.awq_input_scale.view(*([1] * (x.ndim - 1)), -1)
            work_x = work_x / scale
            weight = weight * self.awq_input_scale.unsqueeze(0)

        q_x = quantize_tensor_by_format(
            work_x,
            self.scheme.act_format,
            group_size=max(self.scheme.group_size, 1),
            block_size=max(self.scheme.block_size, 1),
            bits=self.scheme.act_bits,
        )
        if self.collector is not None and self.scheme.zero_skip_stats:
            self.collector.record(self.name, q_x.detach())

        q_w = quantize_tensor_by_format(
            weight,
            self.scheme.weight_format,
            group_size=max(self.scheme.group_size, 1),
            block_size=max(self.scheme.block_size, 1),
            bits=self.scheme.weight_bits,
        )
        bias = self.bias
        if bias is not None and bias.dtype != q_x.dtype:
            bias = bias.to(q_x.dtype)
        if q_w.dtype != q_x.dtype:
            q_w = q_w.to(q_x.dtype)
        return F.linear(q_x, q_w, bias)


@dataclass
class LmHeadPruneStat:
    ratio_requested: float
    ratio_applied: float
    pruned_count: int
    total_count: int
    threshold: float
    zero_count_after: int
    zero_ratio_after: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ratio_requested": self.ratio_requested,
            "ratio_applied": self.ratio_applied,
            "pruned_count": self.pruned_count,
            "total_count": self.total_count,
            "threshold": self.threshold,
            "zero_count_after": self.zero_count_after,
            "zero_ratio_after": self.zero_ratio_after,
        }


def _resolve_weight_parameter(module: nn.Module) -> nn.Parameter:
    if isinstance(module, QuantizedLinear):
        return module.weight
    if isinstance(module, nn.Linear):
        return module.weight
    raise TypeError(f"Unsupported lm_head module type: {module.__class__.__name__}")


def prune_lm_head_magnitude(module: nn.Module, ratio: float) -> LmHeadPruneStat:
    ratio = float(max(0.0, min(1.0, ratio)))
    weight = _resolve_weight_parameter(module)
    total_count = int(weight.numel())
    if total_count == 0 or ratio <= 0.0:
        zero_count_after = int((weight == 0).sum().item())
        return LmHeadPruneStat(
            ratio_requested=ratio,
            ratio_applied=0.0,
            pruned_count=0,
            total_count=total_count,
            threshold=0.0,
            zero_count_after=zero_count_after,
            zero_ratio_after=(zero_count_after / total_count) if total_count else 0.0,
        )

    prune_count = min(total_count - 1, int(total_count * ratio))
    if prune_count <= 0:
        zero_count_after = int((weight == 0).sum().item())
        return LmHeadPruneStat(
            ratio_requested=ratio,
            ratio_applied=0.0,
            pruned_count=0,
            total_count=total_count,
            threshold=0.0,
            zero_count_after=zero_count_after,
            zero_ratio_after=(zero_count_after / total_count) if total_count else 0.0,
        )

    with torch.no_grad():
        flat_abs = weight.detach().abs().reshape(-1)
        threshold = float(torch.kthvalue(flat_abs, prune_count).values.item())
        mask = weight.detach().abs() > threshold
        kept = int(mask.sum().item())
        if kept == total_count:
            topk_indices = torch.topk(flat_abs, k=total_count - prune_count, largest=True).indices
            flat_mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            flat_mask[topk_indices] = True
            mask = flat_mask.view_as(weight)
            kept = int(mask.sum().item())
        weight.mul_(mask.to(weight.dtype))
        zero_count_after = int((weight == 0).sum().item())

    pruned_count = total_count - kept
    return LmHeadPruneStat(
        ratio_requested=ratio,
        ratio_applied=(pruned_count / total_count) if total_count else 0.0,
        pruned_count=pruned_count,
        total_count=total_count,
        threshold=threshold,
        zero_count_after=zero_count_after,
        zero_ratio_after=(zero_count_after / total_count) if total_count else 0.0,
    )


def _replace_module(parent: nn.Module, attr_name: str, child: nn.Module) -> None:
    setattr(parent, attr_name, child)


def wrap_linear_modules(
    model: nn.Module,
    scheme: QuantSchemeSpec,
    *,
    collector: ZeroSkipCollector | None = None,
    skip_lm_head: bool = False,
) -> list[str]:
    replaced: list[str] = []
    module_lookup = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if skip_lm_head and name.endswith("lm_head"):
            continue
        if "." not in name:
            continue
        parent_name, attr_name = name.rsplit(".", 1)
        parent = module_lookup[parent_name]
        _replace_module(parent, attr_name, QuantizedLinear(name, module, scheme, collector=collector))
        replaced.append(name)
    return replaced


def iter_quantized_linear_modules(model: nn.Module) -> list[tuple[str, QuantizedLinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantizedLinear)]
