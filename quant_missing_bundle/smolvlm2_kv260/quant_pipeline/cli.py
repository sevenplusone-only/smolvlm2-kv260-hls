from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_OUTPUT_ROOT, SMOLVLM_LOCAL_DEFAULT
from .runner import ExperimentRunner, RunnerConfig
from .schemes import SCHEME_REGISTRY


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified SmolVLM2 quantization experiment runner")
    parser.add_argument("--model-path", type=Path, default=SMOLVLM_LOCAL_DEFAULT)
    parser.add_argument("--scheme", choices=["all", *SCHEME_REGISTRY.keys()], default="all")
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--extra-mm-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=["prepare", "calibrate", "train", "eval", "export", "all"], default="all")
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-qat-steps", type=int, default=120)
    parser.add_argument("--eval-max-new-tokens", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--topk-board-candidates", type=int, default=2)
    parser.add_argument("--lm-head-prune-ratio", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-frozen-eval", type=int, default=12)
    parser.add_argument("--quick-heldout-eval", type=int, default=8)
    parser.add_argument("--quick-ptq-mm", type=int, default=8)
    parser.add_argument("--quick-ptq-text", type=int, default=64)
    parser.add_argument("--quick-qat-val", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runner = ExperimentRunner(
        RunnerConfig(
            model_path=args.model_path,
            output_dir=args.output_dir,
            seed=args.seed,
            allow_hf_download=args.allow_hf_download,
            extra_mm_manifest=args.extra_mm_manifest,
            data_manifest=args.data_manifest,
            device=args.device,
            max_qat_steps=args.max_qat_steps,
            eval_max_new_tokens=args.eval_max_new_tokens,
            learning_rate=args.learning_rate,
            topk_board_candidates=args.topk_board_candidates,
            lm_head_prune_ratio=args.lm_head_prune_ratio,
            quick_mode=args.quick,
            quick_frozen_eval_limit=args.quick_frozen_eval,
            quick_heldout_eval_limit=args.quick_heldout_eval,
            quick_ptq_mm_limit=args.quick_ptq_mm,
            quick_ptq_text_limit=args.quick_ptq_text,
            quick_qat_val_limit=args.quick_qat_val,
        )
    )
    if args.stage == "prepare":
        runner.run_prepare_only()
        return
    if args.scheme == "all":
        runner.run_many(list(SCHEME_REGISTRY.keys()), stage=args.stage)
    else:
        runner.run_scheme(args.scheme, stage=args.stage)


if __name__ == "__main__":
    main()
