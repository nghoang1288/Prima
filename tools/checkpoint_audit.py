#!/usr/bin/env python3
"""Inspect the official full PRIMA checkpoint entirely on CPU.

Only run this against a checkpoint you trust: the official full checkpoint is a
pickled Python object and therefore requires the compatibility loader.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from tools.models import ModelLoader


def params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def gib(numel: int, bytes_per_param: float) -> float:
    return numel * bytes_per_param / (1024**3)


def audit_heads(heads: Dict[str, Tuple[torch.nn.Module, int]]) -> Dict[str, Any]:
    logical_params = {}
    aliases = defaultdict(list)
    unique_modules = {}

    for name, item in heads.items():
        module = item[0]
        count = params(module)
        logical_params[name] = count
        object_id = id(module)
        aliases[object_id].append(name)
        unique_modules.setdefault(object_id, (module, count))

    unique_param_total = sum(count for _, count in unique_modules.values())
    alias_groups = [
        names
        for names in aliases.values()
        if len(names) > 1
    ]

    return {
        "logical_head_count": len(heads),
        "unique_module_count": len(unique_modules),
        "logical_param_total_double_counting_aliases": sum(
            logical_params.values()
        ),
        "unique_param_total": unique_param_total,
        "alias_group_count": len(alias_groups),
        "alias_groups": alias_groups,
        "params_by_logical_head": logical_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = ModelLoader.load_full_prima_model(
        {"full_model_ckpt": str(args.checkpoint)},
        device="cpu",
        low_vram=False,
        prune_inference_only=True,
    )

    diagnosis = getattr(model, "diagnosisheads", {})
    referral = getattr(model, "referralheads", {})

    diag = audit_heads(diagnosis)
    ref = audit_heads(referral)
    visual_count = params(model.clipvisualmodel)
    priority_count = params(model.priorityhead)

    # A diagnosis module and referral module could theoretically be the same
    # Python object. Compute a global unique task-head total as well.
    task_modules = {}
    for collection in (diagnosis, referral):
        for item in collection.values():
            module = item[0]
            task_modules.setdefault(id(module), module)
    task_modules.setdefault(id(model.priorityhead), model.priorityhead)
    unique_task_params = sum(params(module) for module in task_modules.values())

    report: Dict[str, Any] = {
        "visual_params": visual_count,
        "diagnosis": diag,
        "referral": ref,
        "priority_params": priority_count,
        "unique_task_module_count_including_priority": len(task_modules),
        "unique_task_params_including_priority": unique_task_params,
        "estimated_memory_gib": {
            "visual_fp16": gib(visual_count, 2),
            "visual_fp32": gib(visual_count, 4),
            "unique_task_fp32": gib(unique_task_params, 4),
            "unique_task_int8_weights_only_lower_bound": gib(
                unique_task_params, 1
            ),
            "diagnosis_unique_fp32": gib(diag["unique_param_total"], 4),
            "referral_unique_fp32": gib(ref["unique_param_total"], 4),
            "priority_fp32": gib(priority_count, 4),
        },
    }

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
