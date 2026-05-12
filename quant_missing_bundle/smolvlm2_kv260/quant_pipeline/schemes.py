from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QuantSchemeSpec:
    name: str
    weight_format: str
    act_format: str
    group_size: int
    block_size: int
    calibration_mode: str
    train_mode: str
    checkpoint_path: str
    export_mode: str
    requires_qat: bool
    awq_enabled: bool
    zero_skip_stats: bool
    lora_rank: int
    weight_bits: int
    act_bits: int
    description: str

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


SCHEME_REGISTRY: dict[str, QuantSchemeSpec] = {
    "w4a8_g32_zero": QuantSchemeSpec(
        name="w4a8_g32_zero",
        weight_format="w4_uniform_rtn",
        act_format="a8_per_token",
        group_size=32,
        block_size=1,
        calibration_mode="multimodal_ptq",
        train_mode="none",
        checkpoint_path="checkpoints/w4a8_g32_zero",
        export_mode="kv260_w4a8",
        requires_qat=False,
        awq_enabled=False,
        zero_skip_stats=True,
        lora_rank=0,
        weight_bits=4,
        act_bits=8,
        description="Uniform W4A8 baseline with zero-skip statistics only.",
    ),
    "w4a8_awq_g32": QuantSchemeSpec(
        name="w4a8_awq_g32",
        weight_format="w4_awq_rtn",
        act_format="a8_per_token",
        group_size=32,
        block_size=1,
        calibration_mode="multimodal_awq",
        train_mode="lora_awq_repair",
        checkpoint_path="checkpoints/w4a8_awq_g32",
        export_mode="kv260_w4a8_awq",
        requires_qat=True,
        awq_enabled=True,
        zero_skip_stats=False,
        lora_rank=8,
        weight_bits=4,
        act_bits=8,
        description="Multimodal AWQ-calibrated W4A8 with short LoRA repair training.",
    ),
    "w4_apot_a8_bfp": QuantSchemeSpec(
        name="w4_apot_a8_bfp",
        weight_format="apot4",
        act_format="bfp8",
        group_size=32,
        block_size=16,
        calibration_mode="qat_teacher_distill",
        train_mode="lora_qat",
        checkpoint_path="checkpoints/w4_apot_a8_bfp",
        export_mode="replay_then_hw",
        requires_qat=True,
        awq_enabled=False,
        zero_skip_stats=False,
        lora_rank=8,
        weight_bits=4,
        act_bits=8,
        description="APoT-4 weights with BFP8 activations and short QAT.",
    ),
    "full_bfp": QuantSchemeSpec(
        name="full_bfp",
        weight_format="bfp8",
        act_format="bfp8",
        group_size=1,
        block_size=16,
        calibration_mode="qat_teacher_distill",
        train_mode="lora_qat",
        checkpoint_path="checkpoints/full_bfp",
        export_mode="replay_then_hw",
        requires_qat=True,
        awq_enabled=False,
        zero_skip_stats=False,
        lora_rank=8,
        weight_bits=8,
        act_bits=8,
        description="End-to-end BFP8 weights and activations with short QAT.",
    ),
    "full_apot": QuantSchemeSpec(
        name="full_apot",
        weight_format="apot4",
        act_format="apot8",
        group_size=1,
        block_size=16,
        calibration_mode="qat_teacher_distill",
        train_mode="lora_qat",
        checkpoint_path="checkpoints/full_apot",
        export_mode="replay_then_hw",
        requires_qat=True,
        awq_enabled=False,
        zero_skip_stats=False,
        lora_rank=8,
        weight_bits=4,
        act_bits=8,
        description="End-to-end APoT weights and activations with short QAT.",
    ),
}
