#!/usr/bin/env python3
"""Report the runtime environment before attempting PRIMA inference."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def is_wsl() -> bool:
    try:
        text = Path("/proc/version").read_text().lower()
    except Exception:
        return False
    return "microsoft" in text or "wsl" in text


def nvidia_smi() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def main() -> None:
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "monai": package_version("monai"),
        "torchao": package_version("torchao"),
        "simpleitk": package_version("SimpleITK"),
        "pydicom": package_version("pydicom"),
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": nvidia_smi(),
        "warnings": [],
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        report["gpu"] = {
            "name": props.name,
            "total_vram_gib": props.total_memory / (1024**3),
            "compute_capability": list(torch.cuda.get_device_capability(dev)),
        }
        if props.total_memory / (1024**3) <= 9.0:
            report["warnings"].append(
                "8 GB-class GPU detected: use low_vram=true and keep compile_visual=false for the first run."
            )

        smi = report.get("nvidia_smi")
        if smi:
            try:
                first_line = smi.splitlines()[0]
                fields = [field.strip() for field in first_line.split(",")]
                driver_major = int(fields[1].split(".")[0])
                report["nvidia_driver_major"] = driver_major
                if torch.version.cuda and torch.version.cuda.startswith("13.") and driver_major < 580:
                    report["warnings"].append(
                        "CUDA 13.x runtime detected with NVIDIA driver <580. "
                        "Update the Windows NVIDIA driver before running PRIMA."
                    )
            except (ValueError, IndexError):
                report["warnings"].append(
                    "Could not parse NVIDIA driver version from nvidia-smi."
                )
    else:
        report["warnings"].append(
            "CUDA is not available to PyTorch. On Windows, run PRIMA inside WSL2 with a current NVIDIA Windows driver."
        )

    if not str(report["torch"]).startswith("2.14.0"):
        report["warnings"].append(
            "This branch is CI-tested on PyTorch 2.14.0."
        )
    if report["monai"] != "1.6.0":
        report["warnings"].append(
            "This branch is CI-tested on MONAI 1.6.0."
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
