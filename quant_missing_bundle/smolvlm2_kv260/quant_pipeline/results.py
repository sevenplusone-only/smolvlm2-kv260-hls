from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import MULTIMODAL_MAX_DROP, TEXT_PPL_MAX_RATIO


@dataclass
class SchemeVerdict:
    name: str
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "verdict": self.verdict, "reasons": self.reasons}


def evaluate_viability(baseline: dict[str, Any], candidate: dict[str, Any]) -> SchemeVerdict:
    reasons: list[str] = []
    verdict = "可行"

    base_acc = float(baseline["frozen_eval"]["accuracy"])
    cand_acc = float(candidate["frozen_eval"]["accuracy"])
    if base_acc - cand_acc > MULTIMODAL_MAX_DROP:
        verdict = "不可行"
        reasons.append(f"frozen_eval accuracy drop {base_acc - cand_acc:.2f} > {MULTIMODAL_MAX_DROP:.2f}")

    base_ppl = float(baseline["text_ppl"]["ppl"])
    cand_ppl = float(candidate["text_ppl"]["ppl"])
    if base_ppl > 0 and cand_ppl / base_ppl > TEXT_PPL_MAX_RATIO:
        verdict = "不可行"
        reasons.append(f"text PPL ratio {cand_ppl / base_ppl:.3f} > {TEXT_PPL_MAX_RATIO:.3f}")

    if not candidate.get("generation_sanity", {}).get("ok", False):
        verdict = "不可行"
        reasons.append("generation sanity failed")

    if not reasons:
        reasons.append("meets local threshold")
    return SchemeVerdict(name=candidate["scheme"], verdict=verdict, reasons=reasons)


def rank_scheme_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda item: (
            -float(item["frozen_eval"]["accuracy"]),
            -float(item["heldout_eval"]["accuracy"]),
            float(item["text_ppl"]["ppl"]),
        ),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
