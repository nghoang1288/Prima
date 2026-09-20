#!/usr/bin/env python3
"""Compare PRIMA prediction JSONs from baseline and optimized runtimes."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like output, got shape {arr.shape}")
    return float(arr[0])


def compare_group(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> Tuple[Dict[str, Any], int]:
    names = sorted(set(baseline) | set(candidate))
    rows = {}
    sign_flips = 0
    for name in names:
        if name not in baseline or name not in candidate:
            rows[name] = {"missing": True}
            continue
        a = scalar(baseline[name])
        b = scalar(candidate[name])
        flip = (a >= 0) != (b >= 0)
        sign_flips += int(flip)
        rows[name] = {
            "baseline": a,
            "candidate": b,
            "abs_diff": abs(a - b),
            "sign_flip_at_zero": flip,
        }
    return rows, sign_flips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    a = json.loads(args.baseline.read_text())
    b = json.loads(args.candidate.read_text())

    report: Dict[str, Any] = {"groups": {}}
    all_diffs = []
    total_flips = 0

    for group in ("diagnosis", "referral", "priority"):
        rows, flips = compare_group(a.get(group, {}), b.get(group, {}))
        report["groups"][group] = rows
        total_flips += flips
        all_diffs.extend(
            item["abs_diff"]
            for item in rows.values()
            if "abs_diff" in item
        )

    report["summary"] = {
        "max_abs_diff": max(all_diffs) if all_diffs else 0.0,
        "mean_abs_diff": float(np.mean(all_diffs)) if all_diffs else 0.0,
        "sign_flips_at_zero": total_flips,
    }

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
