#!/usr/bin/env python3
"""Small CUDA smoke test for PRIMA's attention runtime.

Run before loading official checkpoints. It verifies CUDA visibility and compares
attention_backend=auto against the portable SDPA reference on the same FP16
weights/input.
"""

from __future__ import annotations

import json
import os

import torch

from Prima_training_and_evaluation.model_parts import Attention


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    return float(torch.dot(af, bf) / denom)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    module = Attention(
        dim=128,
        heads=4,
        dim_head=32,
        dropout=0.0,
        causal=False,
    ).to(device=device, dtype=torch.float16).eval()

    lengths = [11, 7, 19]
    cu = torch.tensor(
        [0, lengths[0], lengths[0] + lengths[1], sum(lengths)],
        device=device,
        dtype=torch.int32,
    )
    x = torch.randn(
        sum(lengths),
        128,
        device=device,
        dtype=torch.float16,
    )

    previous = os.environ.get("PRIMA_ATTENTION_BACKEND")
    try:
        with torch.inference_mode():
            os.environ["PRIMA_ATTENTION_BACKEND"] = "auto"
            auto = module(x, cu, torch.tensor(max(lengths), device=device))

            os.environ["PRIMA_ATTENTION_BACKEND"] = "sdpa"
            sdpa = module(x, cu, torch.tensor(max(lengths), device=device))
    finally:
        if previous is None:
            os.environ.pop("PRIMA_ATTENTION_BACKEND", None)
        else:
            os.environ["PRIMA_ATTENTION_BACKEND"] = previous

    diff = (auto.float() - sdpa.float()).abs()
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "max_abs_diff_auto_vs_sdpa": float(diff.max()),
        "mean_abs_diff_auto_vs_sdpa": float(diff.mean()),
        "cosine_similarity_auto_vs_sdpa": cosine(auto, sdpa),
        "finite_auto": bool(torch.isfinite(auto).all()),
        "finite_sdpa": bool(torch.isfinite(sdpa).all()),
    }
    print(json.dumps(report, indent=2))

    if not report["finite_auto"] or not report["finite_sdpa"]:
        raise RuntimeError("Non-finite values in attention smoke test")
    if report["cosine_similarity_auto_vs_sdpa"] < 0.999:
        raise RuntimeError(
            "Auto attention differs unexpectedly from SDPA reference"
        )
    if report["max_abs_diff_auto_vs_sdpa"] > 0.05:
        raise RuntimeError(
            "Auto attention max absolute drift exceeds smoke-test tolerance"
        )


if __name__ == "__main__":
    main()
