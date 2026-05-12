from __future__ import annotations

import copy
import gc
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .constants import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DEVICE_PRIORITY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    RESULT_SCHEMA_PATH,
)
from .data import DatasetManifest, SampleRecord, build_dataset_manifest, load_dataset_manifest, shrink_manifest_for_quick_mode
from .eval import evaluate_generation_sanity, evaluate_multimodal_records, evaluate_text_perplexity, prepare_inputs
from .export import export_hardware_bundle, save_local_checkpoint
from .modeling import ZeroSkipCollector, iter_quantized_linear_modules, prune_lm_head_magnitude, wrap_linear_modules
from .quant_ops import stable_softmax_kl
from .results import evaluate_viability, rank_scheme_results, write_json
from .schemes import SCHEME_REGISTRY, QuantSchemeSpec


@dataclass
class RunnerConfig:
    model_path: Path
    output_dir: Path
    cache_root: Path = DEFAULT_CACHE_ROOT
    seed: int = DEFAULT_SEED
    allow_hf_download: bool = False
    extra_mm_manifest: Path | None = None
    data_manifest: Path | None = None
    device: str | None = None
    max_qat_steps: int = 120
    eval_max_new_tokens: int = 48
    learning_rate: float = 1e-4
    topk_board_candidates: int = 2
    lm_head_prune_ratio: float = 0.0
    quick_mode: bool = False
    quick_frozen_eval_limit: int = 12
    quick_heldout_eval_limit: int = 8
    quick_ptq_mm_limit: int = 8
    quick_ptq_text_limit: int = 64
    quick_qat_val_limit: int = 8


class ExperimentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(explicit: str | None) -> torch.device:
        if explicit:
            return torch.device(explicit)
        for candidate in DEFAULT_DEVICE_PRIORITY:
            if candidate == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if candidate == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    def prepare_manifest(self) -> DatasetManifest:
        if self.config.data_manifest and self.config.data_manifest.exists():
            manifest = load_dataset_manifest(self.config.data_manifest)
            return self._maybe_quick_manifest(manifest)
        manifest_path = self.output_dir / "dataset_manifest.json"
        if manifest_path.exists():
            manifest = load_dataset_manifest(manifest_path)
            return self._maybe_quick_manifest(manifest)
        manifest = build_dataset_manifest(
            model_path=str(self.config.model_path),
            output_path=manifest_path,
            cache_root=self.config.cache_root,
            seed=self.config.seed,
            allow_hf_download=self.config.allow_hf_download,
            extra_mm_manifest=self.config.extra_mm_manifest,
        )
        return self._maybe_quick_manifest(manifest)

    def _maybe_quick_manifest(self, manifest: DatasetManifest) -> DatasetManifest:
        if not self.config.quick_mode:
            return manifest
        return shrink_manifest_for_quick_mode(
            manifest,
            frozen_eval_limit=self.config.quick_frozen_eval_limit,
            heldout_eval_limit=self.config.quick_heldout_eval_limit,
            ptq_mm_limit=self.config.quick_ptq_mm_limit,
            ptq_text_limit=self.config.quick_ptq_text_limit,
            qat_val_limit=self.config.quick_qat_val_limit,
        )

    def _scheme_output_dir(self, scheme_name: str) -> Path:
        ratio = self.config.lm_head_prune_ratio
        if ratio <= 0.0:
            return self.output_dir / scheme_name
        suffix = f"lmhead_prune_{int(round(ratio * 1000)):03d}"
        return self.output_dir / f"{scheme_name}_{suffix}"

    def run_prepare_only(self) -> dict[str, Any]:
        manifest = self.prepare_manifest()
        payload = {
            "stage": "prepare",
            "model_path": str(self.config.model_path),
            "dataset_manifest": str(self.output_dir / "dataset_manifest.json"),
            "diagnostics": manifest.diagnostics,
            "split_sizes": {
                split: len(records)
                for split, records in manifest.splits.items()
            },
        }
        write_json(self.output_dir / "prepare_summary.json", payload)
        return payload

    def load_model_and_processor(self) -> tuple[Any, Any]:
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.config.model_path)
        dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
        model = SmolVLMForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        model.eval()
        return model, processor

    def _load_baseline_metrics(self, model: Any, processor: Any, manifest: DatasetManifest) -> dict[str, Any]:
        baseline_path = self.output_dir / "baseline_metrics.json"
        if baseline_path.exists():
            return json.loads(baseline_path.read_text(encoding="utf-8"))

        text_metrics = evaluate_text_perplexity(model, processor.tokenizer, manifest.splits["ptq_text_calib"], self.device)
        frozen_eval = evaluate_multimodal_records(
            model,
            processor,
            manifest.splits["frozen_eval"],
            self.device,
            max_new_tokens=self.config.eval_max_new_tokens,
            progress_label="baseline/frozen",
        )
        heldout_eval = evaluate_multimodal_records(
            model,
            processor,
            manifest.splits["heldout_eval"],
            self.device,
            max_new_tokens=self.config.eval_max_new_tokens,
            progress_label="baseline/heldout",
        )
        generation_sanity = evaluate_generation_sanity(model, processor, self.device, manifest.splits["heldout_eval"][:2])
        payload = {
            "scheme": "baseline_fp",
            "text_ppl": text_metrics,
            "frozen_eval": {
                "accuracy": frozen_eval.accuracy,
                "correct": frozen_eval.correct,
                "total": frozen_eval.total,
                "samples": frozen_eval.samples,
            },
            "heldout_eval": {
                "accuracy": heldout_eval.accuracy,
                "correct": heldout_eval.correct,
                "total": heldout_eval.total,
                "samples": heldout_eval.samples,
            },
            "generation_sanity": generation_sanity,
        }
        baseline_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _run_awq_calibration(self, model: Any, processor: Any, records: list[SampleRecord]) -> dict[str, Any]:
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

        for name, module in iter_quantized_linear_modules(model):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

        model.eval()
        for record in records:
            inputs = prepare_inputs(processor, record, self.device, include_answer=False)
            with torch.no_grad():
                model(**inputs)

        for hook in hooks:
            hook.remove()

        applied: dict[str, Any] = {}
        for name, module in iter_quantized_linear_modules(model):
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
        return {"calibrated_modules": applied, "records_used": len(records)}

    def _load_existing_checkpoint(
        self,
        student: Any,
        scheme: QuantSchemeSpec,
    ) -> tuple[bool, dict[str, Any]]:
        scheme_dir = self._scheme_output_dir(scheme.name)
        checkpoint_dir = scheme_dir / "checkpoint"
        state_path = checkpoint_dir / "student_state.pt"
        manifest_path = checkpoint_dir / "scheme_manifest.json"
        if not state_path.exists():
            return False, {}

        state_dict = torch.load(state_path, map_location="cpu")
        student.load_state_dict(state_dict, strict=False)

        prior_payload: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                extra_payload = manifest_payload.get("extra", {})
                if isinstance(extra_payload, dict):
                    prior_payload = extra_payload
            except json.JSONDecodeError:
                prior_payload = {}
        return True, prior_payload

    def _run_qat_training(
        self,
        student: Any,
        teacher: Any,
        processor: Any,
        train_records: list[SampleRecord],
        val_records: list[SampleRecord],
        scheme: QuantSchemeSpec,
    ) -> dict[str, Any]:
        trainable = [param for param in student.parameters() if param.requires_grad]
        if not trainable:
            return {"status": "skipped", "reason": "no trainable parameters"}

        optimizer = torch.optim.AdamW(trainable, lr=self.config.learning_rate)
        teacher.eval()
        student.train()
        best_val = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float | int]] = []

        effective_train_records = train_records
        if not effective_train_records:
            effective_train_records = val_records

        val_slice = val_records[: min(len(val_records), 8 if self.config.quick_mode else 32)]
        for step, record in enumerate(effective_train_records[: self.config.max_qat_steps], start=1):
            optimizer.zero_grad(set_to_none=True)
            inputs = prepare_inputs(processor, record, self.device, include_answer=True)
            labels = inputs["input_ids"].clone()
            student_out = student(**inputs, labels=labels)
            with torch.no_grad():
                teacher_out = teacher(**inputs)
            ce_loss = student_out.loss
            kl_loss = stable_softmax_kl(student_out.logits, teacher_out.logits, temperature=2.0)
            loss = ce_loss + 0.25 * kl_loss
            loss.backward()
            optimizer.step()

            should_validate = False
            if self.config.quick_mode:
                should_validate = step == 1 or step == self.config.max_qat_steps
            else:
                should_validate = step % 10 == 0 or step == 1

            if should_validate:
                val_eval = evaluate_multimodal_records(
                    student,
                    processor,
                    val_slice,
                    self.device,
                    max_new_tokens=self.config.eval_max_new_tokens,
                    progress_label="train/val",
                )
                history.append({"step": step, "loss": float(loss.item()), "val_accuracy": val_eval.accuracy})
                if val_eval.accuracy > best_val:
                    best_val = val_eval.accuracy
                    best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

        if best_state is not None:
            student.load_state_dict(best_state, strict=False)
        student.eval()
        return {"status": "finished", "history": history, "best_val_accuracy": best_val}

    def _evaluate_scheme(
        self,
        model: Any,
        processor: Any,
        manifest: DatasetManifest,
        scheme: QuantSchemeSpec,
        collector: ZeroSkipCollector | None,
    ) -> dict[str, Any]:
        text_metrics = evaluate_text_perplexity(model, processor.tokenizer, manifest.splits["ptq_text_calib"], self.device)
        frozen_eval = evaluate_multimodal_records(
            model,
            processor,
            manifest.splits["frozen_eval"],
            self.device,
            max_new_tokens=self.config.eval_max_new_tokens,
            progress_label=f"{scheme.name}/frozen",
        )
        heldout_eval = evaluate_multimodal_records(
            model,
            processor,
            manifest.splits["heldout_eval"],
            self.device,
            max_new_tokens=self.config.eval_max_new_tokens,
            progress_label=f"{scheme.name}/heldout",
        )
        generation_sanity = evaluate_generation_sanity(model, processor, self.device, manifest.splits["heldout_eval"][:2])
        return {
            "scheme": scheme.name,
            "text_ppl": text_metrics,
            "frozen_eval": {
                "accuracy": frozen_eval.accuracy,
                "correct": frozen_eval.correct,
                "total": frozen_eval.total,
                "samples": frozen_eval.samples,
            },
            "heldout_eval": {
                "accuracy": heldout_eval.accuracy,
                "correct": heldout_eval.correct,
                "total": heldout_eval.total,
                "samples": heldout_eval.samples,
            },
            "generation_sanity": generation_sanity,
            "zero_skip": collector.to_dict() if collector is not None else {},
        }

    def run_scheme(self, scheme_name: str, stage: str = "all") -> dict[str, Any]:
        scheme = SCHEME_REGISTRY[scheme_name]
        manifest = self.prepare_manifest()
        base_model, processor = self.load_model_and_processor()
        baseline = self._load_baseline_metrics(base_model, processor, manifest)

        collector = ZeroSkipCollector() if scheme.zero_skip_stats else None
        student = copy.deepcopy(base_model).to(self.device)
        for parameter in student.parameters():
            parameter.requires_grad = False
        teacher = base_model
        replaced = wrap_linear_modules(student, scheme, collector=collector, skip_lm_head=False)

        # On CPU-only hosts, avoid carrying extra model copies longer than needed.
        if not scheme.requires_qat:
            del base_model
            gc.collect()

        restored, prior_payload = self._load_existing_checkpoint(student, scheme)
        stage_info: dict[str, Any] = {
            "wrapped_modules": len(replaced),
            "checkpoint_restored": restored,
        }
        prune_stat = prune_lm_head_magnitude(student.lm_head, self.config.lm_head_prune_ratio)
        stage_info["lm_head_pruning"] = prune_stat.to_dict()
        if prior_payload:
            stage_info["prior_stage_payload"] = prior_payload

        scheme_dir = self._scheme_output_dir(scheme.name)
        scheme_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "scheme": scheme.name,
            "scheme_manifest": scheme.to_manifest(),
            "baseline": baseline,
            "stage_info": stage_info,
            "results": {},
            "runtime": {
                "device": str(self.device),
                "timestamp": int(time.time()),
                "quick_mode": self.config.quick_mode,
                "lm_head_prune_ratio": self.config.lm_head_prune_ratio,
            },
        }
        write_json(scheme_dir / "result.json", payload)

        try:
            if stage in {"calibrate", "all"} and scheme.awq_enabled:
                stage_info["awq_calibration"] = self._run_awq_calibration(student, processor, manifest.splits["ptq_multimodal_calib"])
                payload["runtime"]["timestamp"] = int(time.time())
                write_json(scheme_dir / "result.json", payload)

            if stage in {"train", "all"} and scheme.requires_qat:
                stage_info["training"] = self._run_qat_training(
                    student,
                    teacher,
                    processor,
                    manifest.splits["qat_train"],
                    manifest.splits["qat_val"],
                    scheme,
                )
                payload["runtime"]["timestamp"] = int(time.time())
                checkpoint_dir = scheme_dir / "checkpoint"
                save_local_checkpoint(student, scheme, checkpoint_dir, extra_payload=stage_info)
                write_json(scheme_dir / "result.json", payload)
                gc.collect()

            if stage in {"eval", "all"}:
                payload["results"] = self._evaluate_scheme(student, processor, manifest, scheme, collector)
                payload["results"]["verdict"] = evaluate_viability(baseline, payload["results"]).to_dict()
                payload["runtime"]["timestamp"] = int(time.time())
                write_json(scheme_dir / "result.json", payload)

            if stage in {"export", "all"}:
                checkpoint_dir = scheme_dir / "checkpoint"
                export_dir = scheme_dir / "hardware_export"
                save_local_checkpoint(student, scheme, checkpoint_dir, extra_payload=payload["results"] or stage_info)
                if not self.config.quick_mode:
                    stage_info["hardware_export"] = export_hardware_bundle(student, scheme, export_dir)
                else:
                    stage_info["hardware_export"] = {"skipped": True, "reason": "quick_mode"}
                payload["runtime"]["timestamp"] = int(time.time())
                write_json(scheme_dir / "result.json", payload)
        except Exception as exc:
            stage_info["failure"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            payload["runtime"]["timestamp"] = int(time.time())
            write_json(scheme_dir / "result.json", payload)
            checkpoint_dir = scheme_dir / "checkpoint"
            save_local_checkpoint(student, scheme, checkpoint_dir, extra_payload=stage_info)
            raise

        if stage in {"calibrate", "train"}:
            checkpoint_dir = scheme_dir / "checkpoint"
            save_local_checkpoint(student, scheme, checkpoint_dir, extra_payload=stage_info)
        return payload

    def run_many(self, scheme_names: list[str], stage: str = "all") -> dict[str, Any]:
        results = [self.run_scheme(name, stage=stage) for name in scheme_names]
        eval_results = [item["results"] for item in results if item.get("results")]
        ranking = rank_scheme_results(eval_results) if eval_results else []
        top_candidates = [item["scheme"] for item in ranking[: self.config.topk_board_candidates]]
        summary = {
            "schema_path": str(RESULT_SCHEMA_PATH),
            "schemes": results,
            "ranking": ranking,
            "top_board_candidates": top_candidates,
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary
