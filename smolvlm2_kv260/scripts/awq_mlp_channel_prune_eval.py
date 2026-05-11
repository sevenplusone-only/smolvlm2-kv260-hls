#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import traceback
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
from quant_pipeline.modeling import QuantizedLinear, wrap_linear_modules
from quant_pipeline.quant_ops import stable_softmax_kl
from quant_pipeline.runner import ExperimentRunner, RunnerConfig
from quant_pipeline.schemes import SCHEME_REGISTRY


FFN_DIM = 2560
GROUP_SIZE = 32
TILE_N = 64
ALIGNMENT = TILE_N
DEFAULT_TARGET_HIDDEN = 2432


@dataclass
class EvalSummary:
    accuracy: float
    correct: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageResult:
    stage: str
    frozen: EvalSummary
    heldout: EvalSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "frozen": self.frozen.to_dict(),
            "heldout": self.heldout.to_dict(),
        }


def _round_hidden(target_hidden: int, hidden_size: int) -> int:
    target_hidden = max(ALIGNMENT, min(hidden_size, target_hidden))
    rounded = int(round(target_hidden / ALIGNMENT) * ALIGNMENT)
    return max(ALIGNMENT, min(hidden_size, rounded))


def _load_awq_model(model_path: Path, checkpoint_path: Path, device: str) -> tuple[Any, Any]:
    runner = ExperimentRunner(
        RunnerConfig(
            model_path=model_path,
            output_dir=PROJECT_ROOT / "tmp_awq_mlp_prune",
            device=device,
        )
    )
    model, processor = runner.load_model_and_processor()
    wrap_linear_modules(model, SCHEME_REGISTRY["w4a8_awq_g32"], collector=None, skip_lm_head=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device(device))
    model.eval()
    return model, processor


def _slice_quantized_linear(module: QuantizedLinear, keep_idx: torch.Tensor, *, slice_rows: bool) -> QuantizedLinear:
    cloned = copy.deepcopy(module).to("cpu")
    keep_idx = keep_idx.cpu().long()

    with torch.no_grad():
        if slice_rows:
            cloned.weight = nn.Parameter(cloned.weight.detach()[keep_idx, :].clone(), requires_grad=False)
            if cloned.bias is not None:
                cloned.bias = nn.Parameter(cloned.bias.detach()[keep_idx].clone(), requires_grad=False)
            if cloned.lora_b is not None:
                cloned.lora_b = nn.Parameter(cloned.lora_b.detach()[keep_idx, :].clone())
            cloned.out_features = int(keep_idx.numel())
        else:
            cloned.weight = nn.Parameter(cloned.weight.detach()[:, keep_idx].clone(), requires_grad=False)
            if cloned.lora_a is not None:
                cloned.lora_a = nn.Parameter(cloned.lora_a.detach()[:, keep_idx].clone())
            if bool(cloned.has_awq_scale.item()):
                cloned.awq_input_scale = cloned.awq_input_scale.detach()[keep_idx].clone()
            cloned.in_features = int(keep_idx.numel())
    return cloned


def _compute_keep_idx(mlp: nn.Module, target_hidden: int) -> tuple[torch.Tensor, dict[str, Any]]:
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj
    if not all(isinstance(mod, QuantizedLinear) for mod in (gate_proj, up_proj, down_proj)):
        raise TypeError("Expected QuantizedLinear modules for gate/up/down projections")

    gate_w = gate_proj.export_fp_weight().float().cpu()
    up_w = up_proj.export_fp_weight().float().cpu()
    down_w = down_proj.export_fp_weight().float().cpu()

    score = gate_w.abs().mean(dim=1) + up_w.abs().mean(dim=1) + down_w.abs().mean(dim=0)
    hidden_size = int(score.numel())
    keep_hidden = _round_hidden(target_hidden, hidden_size)
    keep_idx = torch.topk(score, k=keep_hidden, largest=True).indices.sort().values

    gateup_n_tiles_before = (hidden_size * 2) // TILE_N
    gateup_n_tiles_after = (keep_hidden * 2) // TILE_N
    down_k_groups_before = hidden_size // GROUP_SIZE
    down_k_groups_after = keep_hidden // GROUP_SIZE
    compression = (keep_hidden * gate_w.shape[1] + keep_hidden * up_w.shape[1] + down_w.shape[0] * keep_hidden) / (
        gate_w.numel() + up_w.numel() + down_w.numel()
    )

    return keep_idx, {
        "hidden_size": hidden_size,
        "keep_hidden": keep_hidden,
        "pruned_hidden": hidden_size - keep_hidden,
        "compression_ratio": compression,
        "gateup_n_tiles_before": gateup_n_tiles_before,
        "gateup_n_tiles_after": gateup_n_tiles_after,
        "down_k_groups_before": down_k_groups_before,
        "down_k_groups_after": down_k_groups_after,
        "ffn_theoretical_speedup": hidden_size / keep_hidden,
    }


def _apply_mlp_pruning(model: Any, target_hidden: int) -> dict[str, Any]:
    # Use one global channel mask derived from layer-0, then apply it to all layers.
    # This keeps tensor subspace alignment consistent across the 32-layer stack.
    first_layer = model.model.text_model.layers[0]
    global_keep_idx, shared_stat = _compute_keep_idx(first_layer.mlp, target_hidden)
    for layer in model.model.text_model.layers:
        layer.mlp.gate_proj = _slice_quantized_linear(layer.mlp.gate_proj, global_keep_idx, slice_rows=True)
        layer.mlp.up_proj = _slice_quantized_linear(layer.mlp.up_proj, global_keep_idx, slice_rows=True)
        layer.mlp.down_proj = _slice_quantized_linear(layer.mlp.down_proj, global_keep_idx, slice_rows=False)
    model.to(next(model.parameters()).device)
    model.eval()
    return shared_stat


def _run_awq_calibration(model: Any, processor: Any, records: list[Any], device: torch.device) -> dict[str, Any]:
    module_stats: dict[str, torch.Tensor] = {}
    module_steps: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].detach().float()
            mean_abs = x.abs().reshape(-1, x.shape[-1]).mean(dim=0).cpu()
            module_stats[name] = module_stats.get(name, torch.zeros_like(mean_abs)) + mean_abs
            module_steps[name] = module_steps.get(name, 0) + 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    for record in records:
        inputs = prepare_inputs(processor, record, device, include_answer=False)
        with torch.no_grad():
            model(**inputs)

    for hook in hooks:
        hook.remove()

    applied: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        if name not in module_stats:
            continue
        mean_abs = module_stats[name] / max(module_steps[name], 1)
        scale = mean_abs.pow(0.5)
        scale = (scale / scale.mean().clamp(min=1e-6)).clamp(0.25, 4.0)
        module.set_awq_scale(scale.to(module.weight.device, dtype=module.weight.dtype))
        applied[name] = {
            "min": float(scale.min().item()),
            "max": float(scale.max().item()),
            "mean": float(scale.mean().item()),
        }
    return {"calibrated_modules": len(applied), "records_used": len(records)}


def _freeze_non_lora_params(model: Any) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        requires = ("lora_a" in name) or ("lora_b" in name)
        param.requires_grad = requires
        if requires:
            trainable += param.numel()
    return trainable


def _run_light_distill(
    student: Any,
    teacher: Any,
    processor: Any,
    train_records: list[Any],
    val_records: list[Any],
    device: torch.device,
    *,
    max_steps: int,
    learning_rate: float,
    eval_max_new_tokens: int,
) -> dict[str, Any]:
    trainable = [param for param in student.parameters() if param.requires_grad]
    if not trainable:
        return {"status": "skipped", "reason": "no trainable parameters"}

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    teacher.eval()
    student.train()

    best_val = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    effective_train_records = train_records or val_records
    val_slice = val_records[: min(len(val_records), 8)]

    for step, record in enumerate(effective_train_records[:max_steps], start=1):
        optimizer.zero_grad(set_to_none=True)
        inputs = prepare_inputs(processor, record, device, include_answer=True)
        labels = inputs["input_ids"].clone()

        student_out = student(**inputs, labels=labels)
        with torch.no_grad():
            teacher_out = teacher(**inputs)

        ce_loss = student_out.loss
        kl_loss = stable_softmax_kl(student_out.logits, teacher_out.logits, temperature=2.0)
        loss = ce_loss + 0.25 * kl_loss
        loss.backward()
        optimizer.step()

        should_validate = step == 1 or step == max_steps or (step % 10 == 0)
        if should_validate and val_slice:
            val_eval = evaluate_multimodal_records(
                student,
                processor,
                val_slice,
                device,
                max_new_tokens=eval_max_new_tokens,
                progress_label="prune-distill/val",
            )
            history.append({"step": step, "loss": float(loss.item()), "val_accuracy": val_eval.accuracy})
            if val_eval.accuracy > best_val:
                best_val = val_eval.accuracy
                best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

    if best_state is not None:
        student.load_state_dict(best_state, strict=False)
    student.eval()
    return {"status": "finished", "history": history, "best_val_accuracy": best_val}


def _evaluate(model: Any, processor: Any, dataset: Any, device: torch.device, stage: str) -> StageResult:
    frozen = evaluate_multimodal_records(
        model,
        processor,
        dataset.splits["frozen_eval"],
        device,
        max_new_tokens=48,
        progress_label=f"{stage}/frozen40",
    )
    heldout = evaluate_multimodal_records(
        model,
        processor,
        dataset.splits["heldout_eval"],
        device,
        max_new_tokens=48,
        progress_label=f"{stage}/heldout20",
    )
    return StageResult(
        stage=stage,
        frozen=EvalSummary(frozen.accuracy, frozen.correct, frozen.total),
        heldout=EvalSummary(heldout.accuracy, heldout.correct, heldout.total),
    )


def main() -> None:
    model_path = Path("/Users/miracle/smolvlm_local")
    checkpoint = PROJECT_ROOT / "quant_experiments_awq_repair_20260508_010000" / "w4a8_awq_g32" / "checkpoint" / "student_state.pt"
    manifest_path = PROJECT_ROOT / "quant_experiments_lmhead_prune_final" / "dataset_manifest.json"
    output_dir = PROJECT_ROOT / "quant_experiments_awq_mlp_prune"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model path: {model_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    dataset = load_dataset_manifest(manifest_path)
    device = torch.device("cpu")
    result_path = output_dir / "result.json"
    payload: dict[str, Any] = {
        "method": "awq_w4a8_g32_text_mlp_structured_channel_prune",
        "checkpoint": str(checkpoint),
        "dataset_manifest": str(manifest_path),
        "target_hidden": DEFAULT_TARGET_HIDDEN,
        "alignment": {
            "group_size": GROUP_SIZE,
            "tile_n": TILE_N,
            "channel_alignment": ALIGNMENT,
        },
        "stages": [],
    }

    def _flush() -> None:
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        teacher, processor = _load_awq_model(model_path, checkpoint, "cpu")
        baseline = _evaluate(teacher, processor, dataset, device, "awq-baseline")
        payload["stages"].append(baseline.to_dict())
        _flush()

        student, _ = _load_awq_model(model_path, checkpoint, "cpu")
        prune_stat = _apply_mlp_pruning(student, DEFAULT_TARGET_HIDDEN)
        payload["prune_stat"] = prune_stat
        _flush()

        pruned_eval = _evaluate(student, processor, dataset, device, f"awq-pruned-h{DEFAULT_TARGET_HIDDEN}")
        payload["stages"].append(pruned_eval.to_dict())
        _flush()

        calib_info = _run_awq_calibration(student, processor, dataset.splits["ptq_multimodal_calib"], device)
        payload["calibration"] = calib_info
        _flush()

        calibrated_eval = _evaluate(student, processor, dataset, device, f"awq-pruned-calib-h{DEFAULT_TARGET_HIDDEN}")
        payload["stages"].append(calibrated_eval.to_dict())
        _flush()

        trainable_params = _freeze_non_lora_params(student)
        payload["trainable_lora_params"] = trainable_params
        _flush()

        distill_info = _run_light_distill(
            student,
            teacher,
            processor,
            dataset.splits["qat_train"],
            dataset.splits["qat_val"],
            device,
            max_steps=40,
            learning_rate=1e-4,
            eval_max_new_tokens=48,
        )
        payload["distillation"] = distill_info
        _flush()

        distilled_eval = _evaluate(student, processor, dataset, device, f"awq-pruned-distill-h{DEFAULT_TARGET_HIDDEN}")
        payload["stages"].append(distilled_eval.to_dict())
        _flush()
    except Exception as exc:
        payload["failure"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _flush()
        raise

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
