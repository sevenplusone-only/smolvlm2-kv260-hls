from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_PTQ_MM_CALIB_SAMPLES,
    DEFAULT_PTQ_TEXT_CALIB_SAMPLES,
    DEFAULT_QAT_TRAIN_SAMPLES,
    DEFAULT_QAT_VAL_SAMPLES,
    FROZEN_EVAL_IMAGES,
    FROZEN_EVAL_JSON,
    LOCAL_WIKITEXT_CACHE,
)


@dataclass
class SampleRecord:
    sample_id: str
    source: str
    split: str
    prompt: str
    answers: list[str]
    modality: str
    image_path: str | None = None
    text: str | None = None
    dataset_name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    version: int
    seed: int
    model_path: str
    splits: dict[str, list[SampleRecord]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "model_path": self.model_path,
            "splits": {
                name: [item.to_dict() for item in records]
                for name, records in self.splits.items()
            },
            "diagnostics": self.diagnostics,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_frozen_eval_records() -> list[SampleRecord]:
    raw_items = json.loads(FROZEN_EVAL_JSON.read_text(encoding="utf-8"))
    records: list[SampleRecord] = []
    for item in raw_items:
        image_path = FROZEN_EVAL_IMAGES / item["image_path"]
        answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
        records.append(
            SampleRecord(
                sample_id=f"frozen::{item['dataset_name']}::{item['id']}",
                source="versavlm_frozen_100",
                split="frozen_eval",
                prompt=item["question"],
                answers=answers,
                modality="image_text",
                image_path=str(image_path),
                dataset_name=item.get("dataset_name"),
                meta={"type": item.get("type"), "orig_id": item.get("id")},
            )
        )
    return records


def _stratified_local_multimodal_fallback(
    records: list[SampleRecord],
) -> dict[str, list[SampleRecord]]:
    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        bucket = str(record.meta.get("type", record.dataset_name or "unknown"))
        grouped.setdefault(bucket, []).append(record)

    splits = {
        "frozen_eval": [],
        "heldout_eval": [],
        "ptq_multimodal_calib": [],
        "qat_val": [],
        "qat_train": [],
    }

    for bucket_records in grouped.values():
        count = len(bucket_records)
        if count <= 1:
            splits["frozen_eval"].extend(bucket_records)
            continue

        n_frozen = max(1, round(count * 0.4))
        n_heldout = max(1, round(count * 0.2))
        n_ptq = max(1, round(count * 0.2))
        remaining = count - n_frozen - n_heldout - n_ptq
        n_qat_val = max(0, remaining)

        cursor = 0
        splits["frozen_eval"].extend(bucket_records[cursor: cursor + n_frozen])
        cursor += n_frozen
        splits["heldout_eval"].extend(bucket_records[cursor: cursor + n_heldout])
        cursor += n_heldout
        splits["ptq_multimodal_calib"].extend(bucket_records[cursor: cursor + n_ptq])
        cursor += n_ptq
        splits["qat_val"].extend(bucket_records[cursor: cursor + n_qat_val])

    return splits


def _materialize_hf_image(image_obj: Any, image_path: Path) -> str:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image_obj, "save"):
        image_obj.save(image_path)
    elif isinstance(image_obj, dict) and "bytes" in image_obj:
        image_path.write_bytes(image_obj["bytes"])
    else:
        raise ValueError("Unsupported image object for materialization")
    return str(image_path)


def _load_hf_textvqa_records(cache_root: Path, sample_limit: int | None = None) -> list[SampleRecord]:
    import os
    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf_datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf_hub"))
    ds = load_dataset("textvqa", split="validation", cache_dir=str(cache_root / "hf_datasets"))
    if sample_limit is not None:
        ds = ds.select(range(min(sample_limit, len(ds))))

    records: list[SampleRecord] = []
    for idx, item in enumerate(ds):
        answer_list = item.get("answers") or item.get("answer") or []
        if isinstance(answer_list, str):
            answer_list = [answer_list]
        image_path = cache_root / "textvqa_images" / f"{idx:06d}.png"
        mat_path = _materialize_hf_image(item["image"], image_path)
        records.append(
            SampleRecord(
                sample_id=f"textvqa::{idx}",
                source="hf_textvqa_validation",
                split="multimodal_pool",
                prompt=item["question"],
                answers=[str(answer) for answer in answer_list if str(answer).strip()],
                modality="image_text",
                image_path=mat_path,
                dataset_name="textVQA",
                meta={"question_id": item.get("question_id")},
            )
        )
    return records


def _load_extra_manifest_records(manifest_path: Path) -> list[SampleRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[SampleRecord] = []
    for idx, item in enumerate(payload):
        answers = item.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        records.append(
            SampleRecord(
                sample_id=item.get("sample_id", f"extra::{idx}"),
                source=item.get("source", "extra_manifest"),
                split=item.get("split", "multimodal_pool"),
                prompt=item["prompt"],
                answers=answers,
                modality=item.get("modality", "image_text"),
                image_path=item.get("image_path"),
                text=item.get("text"),
                dataset_name=item.get("dataset_name"),
                meta=item.get("meta", {}),
            )
        )
    return records


def _load_wikitext_sentences(cache_root: Path, limit: int | None = None) -> list[SampleRecord]:
    if LOCAL_WIKITEXT_CACHE.exists():
        lines = [line.strip() for line in LOCAL_WIKITEXT_CACHE.read_text(encoding="utf-8").splitlines()]
        sentences: list[SampleRecord] = []
        for idx, text in enumerate(lines):
            if len(text) < 40:
                continue
            sentences.append(
                SampleRecord(
                    sample_id=f"wikitext_cache::{idx}",
                    source="wikitext_local_cache",
                    split="text_pool",
                    prompt=text,
                    answers=[],
                    modality="text",
                    text=text,
                )
            )
            if limit is not None and len(sentences) >= limit:
                break
        if sentences:
            return sentences

    import os
    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf_datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf_hub"))
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=str(cache_root / "hf_datasets"))
    sentences: list[SampleRecord] = []
    for idx, entry in enumerate(ds["text"]):
        text = str(entry).strip()
        if len(text) < 40:
            continue
        sentences.append(
            SampleRecord(
                sample_id=f"wikitext::{idx}",
                source="wikitext_test",
                split="text_pool",
                prompt=text,
                answers=[],
                modality="text",
                text=text,
            )
        )
        if limit is not None and len(sentences) >= limit:
            break

    LOCAL_WIKITEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_WIKITEXT_CACHE.write_text(
        "\n".join(record.text or record.prompt for record in sentences),
        encoding="utf-8",
    )
    return sentences


def build_dataset_manifest(
    *,
    model_path: str,
    output_path: Path,
    cache_root: Path,
    seed: int,
    allow_hf_download: bool,
    extra_mm_manifest: Path | None = None,
    ptq_mm_samples: int = DEFAULT_PTQ_MM_CALIB_SAMPLES,
    ptq_text_samples: int = DEFAULT_PTQ_TEXT_CALIB_SAMPLES,
    qat_train_samples: int = DEFAULT_QAT_TRAIN_SAMPLES,
    qat_val_samples: int = DEFAULT_QAT_VAL_SAMPLES,
) -> DatasetManifest:
    rng = random.Random(seed)

    base_frozen_records = _load_frozen_eval_records()
    multimodal_pool: list[SampleRecord] = []
    diagnostics: dict[str, Any] = {
        "requested_counts": {
            "ptq_multimodal_calib": ptq_mm_samples,
            "ptq_text_calib": ptq_text_samples,
            "qat_train": qat_train_samples,
            "qat_val": qat_val_samples,
        }
    }

    if extra_mm_manifest is not None and extra_mm_manifest.exists():
        multimodal_pool.extend(_load_extra_manifest_records(extra_mm_manifest))

    if allow_hf_download:
        try:
            multimodal_pool.extend(_load_hf_textvqa_records(cache_root, sample_limit=ptq_mm_samples + qat_train_samples + qat_val_samples + 200))
            diagnostics["hf_download"] = "textvqa_validation"
        except Exception as exc:  # pragma: no cover - runtime availability only
            diagnostics["hf_download_error"] = str(exc)

    rng.shuffle(multimodal_pool)
    text_pool = _load_wikitext_sentences(cache_root, limit=max(ptq_text_samples, 1024))
    rng.shuffle(text_pool)

    if multimodal_pool:
        frozen_eval = base_frozen_records
        heldout_eval = multimodal_pool[:200]
        remainder = multimodal_pool[200:]
        ptq_mm = remainder[:ptq_mm_samples]
        remainder = remainder[ptq_mm_samples:]
        qat_val = remainder[:qat_val_samples]
        remainder = remainder[qat_val_samples:]
        qat_train = remainder[:qat_train_samples]
        diagnostics["multimodal_source"] = "external_or_downloaded"
    else:
        fallback_splits = _stratified_local_multimodal_fallback(list(base_frozen_records))
        frozen_eval = fallback_splits["frozen_eval"]
        heldout_eval = fallback_splits["heldout_eval"]
        ptq_mm = fallback_splits["ptq_multimodal_calib"]
        qat_val = fallback_splits["qat_val"]
        qat_train = fallback_splits["qat_train"]
        diagnostics["multimodal_source"] = "local_fallback_split_from_100"
        diagnostics["fallback_note"] = (
            "External multimodal pool unavailable; created disjoint software-only splits "
            "from the local 100 image-text samples."
        )

    ptq_text = text_pool[:ptq_text_samples]

    diagnostics["actual_counts"] = {
        "frozen_eval": len(frozen_eval),
        "heldout_eval": len(heldout_eval),
        "ptq_multimodal_calib": len(ptq_mm),
        "ptq_text_calib": len(ptq_text),
        "qat_train": len(qat_train),
        "qat_val": len(qat_val),
    }
    diagnostics["data_shortfall"] = {
        key: diagnostics["requested_counts"][key] - diagnostics["actual_counts"].get(key, 0)
        for key in ("ptq_multimodal_calib", "ptq_text_calib", "qat_train", "qat_val")
    }

    manifest = DatasetManifest(
        version=1,
        seed=seed,
        model_path=model_path,
        splits={
            "frozen_eval": frozen_eval,
            "heldout_eval": heldout_eval,
            "ptq_multimodal_calib": ptq_mm,
            "ptq_text_calib": ptq_text,
            "qat_train": qat_train,
            "qat_val": qat_val,
        },
        diagnostics=diagnostics,
    )
    manifest.write_json(output_path)
    return manifest


def shrink_manifest_for_quick_mode(
    manifest: DatasetManifest,
    *,
    frozen_eval_limit: int,
    heldout_eval_limit: int,
    ptq_mm_limit: int,
    ptq_text_limit: int,
    qat_val_limit: int,
) -> DatasetManifest:
    quick_splits = {
        "frozen_eval": manifest.splits["frozen_eval"][:frozen_eval_limit],
        "heldout_eval": manifest.splits["heldout_eval"][:heldout_eval_limit],
        "ptq_multimodal_calib": manifest.splits["ptq_multimodal_calib"][:ptq_mm_limit],
        "ptq_text_calib": manifest.splits["ptq_text_calib"][:ptq_text_limit],
        "qat_train": list(manifest.splits["ptq_multimodal_calib"][:ptq_mm_limit]),
        "qat_val": manifest.splits["qat_val"][:qat_val_limit] or manifest.splits["heldout_eval"][:qat_val_limit],
    }
    diagnostics = dict(manifest.diagnostics)
    diagnostics["quick_mode"] = {
        "enabled": True,
        "limits": {
            "frozen_eval": frozen_eval_limit,
            "heldout_eval": heldout_eval_limit,
            "ptq_multimodal_calib": ptq_mm_limit,
            "ptq_text_calib": ptq_text_limit,
            "qat_val": qat_val_limit,
            "qat_train_source": "copied_from_ptq_multimodal_calib",
        },
        "actual_counts": {name: len(records) for name, records in quick_splits.items()},
    }
    return DatasetManifest(
        version=manifest.version,
        seed=manifest.seed,
        model_path=manifest.model_path,
        splits=quick_splits,
        diagnostics=diagnostics,
    )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = {
        split: [SampleRecord(**record) for record in records]
        for split, records in payload["splits"].items()
    }
    return DatasetManifest(
        version=payload["version"],
        seed=payload["seed"],
        model_path=payload["model_path"],
        splits=splits,
        diagnostics=payload.get("diagnostics", {}),
    )
