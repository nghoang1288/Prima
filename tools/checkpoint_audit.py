#!/usr/bin/env python3
"""Inspect the official full PRIMA checkpoint entirely on CPU.

This intentionally uses the repository compatibility loader because the official
checkpoint is a pickled full model object. Only run it on a checkpoint you trust.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from tools.models import ModelLoader


def params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def gib(numel: int, bytes_per_param: float) -> float:
    return numel * bytes_per_param / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = ModelLoader.load_full_prima_model(
        {"full_model_ckpt": str(args.checkpoint)},
        device="cpu",
        low_vram=False,
    )

    diagnosis = getattr(model, "diagnosisheads", {})
    referral = getattr(model, "referralheads", {})

    diag_counts = {name: params(item[0]) for name, item in diagnosis.items()}
    ref_counts = {name: params(item[0]) for name, item in referral.items()}
    visual_count = params(model.clipvisualmodel)
    priority_count = params(model.priorityhead)

    report: Dict[str, Any] = {
        "visual_params": visual_count,
        "diagnosis_head_count": len(diag_counts),
        "diagnosis_params_total": sum(diag_counts.values()),
        "referral_head_count": len(ref_counts),
        "referral_params_total": sum(ref_counts.values()),
        "priority_params": priority_count,
        "estimated_memory_gib": {
            "visual_fp16": gib(visual_count, 2),
            "visual_fp32": gib(visual_count, 4),
            "diagnosis_fp32": gib(sum(diag_counts.values()), 4),
            "diagnosis_int8_weights_only_lower_bound": gib(
                sum(diag_counts.values()), 1
            ),
            "referral_fp32": gib(sum(ref_counts.values()), 4),
            "priority_fp32": gib(priority_count, 4),
        },
        "diagnosis_params_by_head": diag_counts,
        "referral_params_by_head": ref_counts,
    }

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
