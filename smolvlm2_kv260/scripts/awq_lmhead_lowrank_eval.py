#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.data import load_dataset_manifest
from quant_pipeline.eval import evaluate_multimodal_records, prepare_inputs
from quant_pipeline.modeling import QuantizedLinear
from quant_pipeline.runner import ExperimentRunner, RunnerConfig
from quant_pipeline.schemes import SCHEME_REGISTRY


@dataclass
class CandidateResult:
    rank: int
    compression_ratio: float
    frozen_accuracy: float
    frozen_correct: int
    frozen_total: int
    calib_accuracy: float
    calib_correct: int
    calib_total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LowRankLinear(nn.Module):
    def __init__(self, left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.left = nn.Parameter(left.detach().clone(), requires_grad=False)
        self.right = nn.Parameter(right.detach().clone(), requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.in_features = right.shape[1]
        self.out_features = left.shape[0]
        self.rank = left.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.matmul(x, self.right.transpose(0, 1))
        return torch.matmul(hidden, self.left.transpose(0, 1)) + (self.bias if self.bias is not None else 0.0)


def _load_awq_model(
    model_path: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[Any, Any]:
    runner = ExperimentRunner(
        RunnerConfig(
            model_path=model_path,
            output_dir=PROJECT_ROOT / "tmp_awq_lmhead_lowrank",
            device=device,
        )
    )
    model, processor = runner.load_model_and_processor()
    scheme = SCHEME_REGISTRY["w4a8_awq_g32"]
    wrapped = runner.prepare_manifest  # silence lint-like reasoning; runner instance is otherwise used for device
    del wrapped

    from quant_pipeline.modeling import wrap_linear_modules

    wrap_linear_modules(model, scheme, collector=None, skip_lm_head=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device(device))
    model.eval()
    return model, processor


def _extract_awq_weight_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(module, QuantizedLinear):
        weight = module.export_fp_weight().to(dtype=torch.float32)
        bias = module.bias.detach().float().cpu() if module.bias is not None else None
        return weight, bias
    if isinstance(module, nn.Linear):
        weight = module.weight.detach().float().cpu()
        bias = module.bias.detach().float().cpu() if module.bias is not None else None
        return weight, bias
    raise TypeError(f"Unsupported lm_head module type: {type(module).__name__}")


def _build_low_rank_module(weight: torch.Tensor, bias: torch.Tensor | None, rank: int) -> LowRankLinear:
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    rank = min(rank, s.numel())
    left = u[:, :rank] * s[:rank].unsqueeze(0)
    right = vh[:rank, :]
    return LowRankLinear(left.contiguous(), right.contiguous(), bias)


def _compression_ratio(rows: int, cols: int, rank: int, has_bias: bool) -> float:
    original = rows * cols + (rows if has_bias else 0)
    compressed = rank * (rows + cols) + (rows if has_bias else 0)
    return compressed / original


def _calib_token_accuracy(
    model: Any,
    processor: Any,
    records: list[Any],
    device: torch.device,
    limit: int,
) -> tuple[int, int, float]:
    model.eval()
    correct = 0
    total = 0
    for record in records[:limit]:
        inputs = prepare_inputs(processor, record, device, include_answer=True)
        labels = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        target = labels[:, 1:]
        pred = logits.argmax(dim=-1)
        mask = target != processor.tokenizer.pad_token_id
        if mask.any():
            correct += int(((pred == target) & mask).sum().item())
            total += int(mask.sum().item())
    acc = (correct / total) * 100.0 if total else 0.0
    return correct, total, acc


def main() -> None:
    model_path = Path("/Users/miracle/smolvlm_local")
    base_dir = PROJECT_ROOT / "quant_experiments_awq_repair_20260508_010000"
    checkpoint = base_dir / "w4a8_awq_g32" / "checkpoint" / "student_state.pt"
    manifest_path = PROJECT_ROOT / "quant_experiments_lmhead_prune_final" / "dataset_manifest.json"
    output_dir = PROJECT_ROOT / "quant_experiments_awq_lmhead_lowrank"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    device = "cpu"
    dataset = load_dataset_manifest(manifest_path)
    model, processor = _load_awq_model(model_path, checkpoint, device)

    base_lm_head = model.lm_head
    weight, bias = _extract_awq_weight_bias(base_lm_head)
    rows, cols = weight.shape
    has_bias = bias is not None

    baseline_eval = evaluate_multimodal_records(
        model,
        processor,
        dataset.splits["frozen_eval"],
        torch.device(device),
        max_new_tokens=48,
        progress_label="awq-baseline/frozen40",
    )
    baseline_payload = {
        "frozen_accuracy": baseline_eval.accuracy,
        "frozen_correct": baseline_eval.correct,
        "frozen_total": baseline_eval.total,
    }

    calib_records = dataset.splits["ptq_multimodal_calib"]
    rank_candidates = [896, 832, 768, 704, 640]
    candidate_payloads: list[CandidateResult] = []
    best_candidate: CandidateResult | None = None

    for rank in rank_candidates:
        model.lm_head = _build_low_rank_module(weight, bias, rank).to(torch.device(device))
        calib_correct, calib_total, calib_acc = _calib_token_accuracy(
            model,
            processor,
            calib_records,
            torch.device(device),
            limit=min(8, len(calib_records)),
        )
        frozen_eval = evaluate_multimodal_records(
            model,
            processor,
            dataset.splits["frozen_eval"],
            torch.device(device),
            max_new_tokens=48,
            progress_label=f"awq-lowrank-r{rank}/frozen40",
        )
        result = CandidateResult(
            rank=rank,
            compression_ratio=_compression_ratio(rows, cols, rank, has_bias),
            frozen_accuracy=frozen_eval.accuracy,
            frozen_correct=frozen_eval.correct,
            frozen_total=frozen_eval.total,
            calib_accuracy=calib_acc,
            calib_correct=calib_correct,
            calib_total=calib_total,
        )
        candidate_payloads.append(result)
        if frozen_eval.accuracy >= 40.0:
            if best_candidate is None or result.compression_ratio < best_candidate.compression_ratio:
                best_candidate = result

    output = {
        "method": "awq_w4a8_g32_lmhead_lowrank_only",
        "checkpoint": str(checkpoint),
        "dataset_manifest": str(manifest_path),
        "baseline": baseline_payload,
        "lm_head_shape": [rows, cols],
        "candidates": [item.to_dict() for item in candidate_payloads],
        "selected": best_candidate.to_dict() if best_candidate is not None else None,
    }
    (output_dir / "result.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


