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
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    track_zero_flip: bool = True,
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
        flip = (a >= 0) != (b >= 0) if track_zero_flip else None
        sign_flips += int(bool(flip)) if track_zero_flip else 0
        rows[name] = {
            "baseline": a,
            "candidate": b,
            "abs_diff": abs(a - b),
        }
        if track_zero_flip:
            rows[name]["sign_flip_at_zero"] = flip
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
    missing_output_count = 0

    for group in ("diagnosis", "referral"):
        rows, flips = compare_group(
            a.get(group, {}),
            b.get(group, {}),
            track_zero_flip=True,
        )
        report["groups"][group] = rows
        missing_output_count += sum(
            1 for item in rows.values() if item.get("missing")
        )
        total_flips += flips
        all_diffs.extend(
            item["abs_diff"]
            for item in rows.values()
            if "abs_diff" in item
        )

    priority_rows, _ = compare_group(
        a.get("priority", {}),
        b.get("priority", {}),
        track_zero_flip=False,
    )
    report["groups"]["priority"] = priority_rows
    missing_output_count += sum(
        1 for item in priority_rows.values() if item.get("missing")
    )
    all_diffs.extend(
        item["abs_diff"]
        for item in priority_rows.values()
        if "abs_diff" in item
    )

    baseline_priority_labels = set(a.get("priority", {}))
    candidate_priority_labels = set(b.get("priority", {}))
    same_priority_labels = baseline_priority_labels == candidate_priority_labels
    if same_priority_labels and baseline_priority_labels:
        priority_labels = sorted(baseline_priority_labels)
        baseline_priority = {
            label: scalar(a["priority"][label]) for label in priority_labels
        }
        candidate_priority = {
            label: scalar(b["priority"][label]) for label in priority_labels
        }
        baseline_label = max(baseline_priority, key=baseline_priority.get)
        candidate_label = max(candidate_priority, key=candidate_priority.get)
        report["priority_decision"] = {
            "comparable": True,
            "baseline": baseline_label,
            "candidate": candidate_label,
            "changed": baseline_label != candidate_label,
        }
    else:
        report["priority_decision"] = {
            "comparable": False,
            "baseline_labels": sorted(baseline_priority_labels),
            "candidate_labels": sorted(candidate_priority_labels),
            "changed": None,
        }

    report["summary"] = {
        "max_abs_diff": max(all_diffs) if all_diffs else 0.0,
        "mean_abs_diff": float(np.mean(all_diffs)) if all_diffs else 0.0,
        "threshold_sign_flips_at_zero": total_flips,
        "missing_output_count": missing_output_count,
    }

    if "clip_emb" in a and "clip_emb" in b:
        emb_a = np.asarray(a["clip_emb"], dtype=np.float64).reshape(-1)
        emb_b = np.asarray(b["clip_emb"], dtype=np.float64).reshape(-1)
        if emb_a.shape != emb_b.shape:
            report["clip_embedding"] = {
                "shape_match": False,
                "baseline_shape": list(emb_a.shape),
                "candidate_shape": list(emb_b.shape),
            }
        else:
            denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
            cosine = (
                float(np.dot(emb_a, emb_b) / denom)
                if denom > 0
                else float("nan")
            )
            report["clip_embedding"] = {
                "shape_match": True,
                "cosine_similarity": cosine,
                "max_abs_diff": float(np.max(np.abs(emb_a - emb_b))),
                "mean_abs_diff": float(np.mean(np.abs(emb_a - emb_b))),
            }

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
