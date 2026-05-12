from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .constants import TEXT_ONLY_SANITY_PROMPTS
from .data import SampleRecord
from .quant_ops import perplexity_from_loss, prediction_matches_any


@dataclass
class MultimodalEvalResult:
    accuracy: float
    correct: int
    total: int
    samples: list[dict[str, Any]]


def build_prompt_messages(record: SampleRecord, *, include_answer: bool = False) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    if record.modality == "image_text" and record.image_path:
        user_content.append({"type": "image", "path": record.image_path})
    if record.text and record.modality == "text":
        user_content.append({"type": "text", "text": record.text})
    else:
        user_content.append({"type": "text", "text": record.prompt})

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if include_answer and record.answers:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record.answers[0]}],
            }
        )
    return messages


def prepare_inputs(processor: Any, record: SampleRecord, device: torch.device, *, include_answer: bool = False) -> dict[str, torch.Tensor]:
    messages = build_prompt_messages(record, include_answer=include_answer)
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "add_generation_prompt": not include_answer,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }

    if record.modality == "image_text" and record.image_path:
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            _ = image.size
    inputs = processor.apply_chat_template(**kwargs)
    moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    if device.type == "cuda" and "pixel_values" in moved and hasattr(moved["pixel_values"], "dtype"):
        if moved["pixel_values"].dtype.is_floating_point:
            moved["pixel_values"] = moved["pixel_values"].to(torch.float16)
    return moved


def evaluate_text_perplexity(
    model: Any,
    tokenizer: Any,
    text_records: list[SampleRecord],
    device: torch.device,
    *,
    max_tokens: int = 512,
) -> dict[str, float]:
    text = "\n".join(record.text or record.prompt for record in text_records)
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    encodings = {key: value.to(device) for key, value in encodings.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=encodings["input_ids"], attention_mask=encodings.get("attention_mask"), labels=encodings["input_ids"])
    loss = float(outputs.loss.item())
    return {"loss": loss, "ppl": perplexity_from_loss(loss)}


def _decode_prediction(processor: Any, generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor | None = None) -> str:
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    text = decoded[0] if decoded else ""
    if prompt_input_ids is not None:
        prompt_decoded = processor.batch_decode(prompt_input_ids, skip_special_tokens=True)[0]
        if text.startswith(prompt_decoded):
            text = text[len(prompt_decoded):]
    return text.strip()


def evaluate_multimodal_records(
    model: Any,
    processor: Any,
    records: list[SampleRecord],
    device: torch.device,
    *,
    max_new_tokens: int = 48,
    progress_label: str | None = None,
) -> MultimodalEvalResult:
    model.eval()
    results: list[dict[str, Any]] = []
    correct = 0

    total = len(records)
    label = progress_label or "eval"
    for idx, record in enumerate(records, start=1):
        inputs = prepare_inputs(processor, record, device, include_answer=False)
        with torch.no_grad():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        prediction = _decode_prediction(processor, generated, inputs.get("input_ids"))
        is_correct = prediction_matches_any(prediction, record.answers)
        correct += int(is_correct)
        running_acc = (correct / idx) * 100.0 if idx else 0.0
        print(f"[{label}] {idx}/{total} correct={correct} running_acc={running_acc:.2f}% sample={record.sample_id}", flush=True)
        results.append(
            {
                "sample_id": record.sample_id,
                "prediction": prediction,
                "answers": record.answers,
                "correct": is_correct,
                "dataset_name": record.dataset_name,
            }
        )

    accuracy = (correct / total) * 100.0 if total else 0.0
    return MultimodalEvalResult(accuracy=accuracy, correct=correct, total=total, samples=results)


def evaluate_generation_sanity(
    model: Any,
    processor: Any,
    device: torch.device,
    sanity_records: list[SampleRecord],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    model.eval()

    records = list(sanity_records[:6])
    for prompt in TEXT_ONLY_SANITY_PROMPTS:
        record = SampleRecord(
            sample_id=f"sanity::{len(failures)}",
            source="builtin_sanity",
            split="sanity",
            prompt=prompt,
            answers=[],
            modality="text",
            text=prompt,
        )
        records.append(record)

    for record in records:
        inputs = prepare_inputs(processor, record, device, include_answer=False)
        with torch.no_grad():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=32)
        text = _decode_prediction(processor, generated, inputs.get("input_ids"))
        bad = (not text) or ("nan" in text.lower())
        if bad:
            failures.append({"sample_id": record.sample_id, "prediction": text})

    return {"ok": not failures, "failures": failures}
