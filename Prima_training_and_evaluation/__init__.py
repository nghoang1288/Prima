"""PRIMA training/evaluation package with lazy public imports.

Historically this module eagerly imported the training dataset, which pulled
training-only dependencies (for example pathos) into inference-only processes.
PEP 562 lazy attribute loading keeps the public API compatible without those
startup side effects.
"""

from importlib import import_module
from typing import Dict, Tuple

__all__ = [
    "CLIP",
    "SerieCLIP",
    "ViT",
    "GPTWrapper",
    "HierViT",
    "SerieTransformerEncoder",
    "Transformer",
    "PreNorm",
    "FeedForward",
    "Attention",
    "MrDataset",
    "MedicalImagePatchifier",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "CLIP": (".model", "CLIP"),
    "SerieCLIP": (".model", "SerieCLIP"),
    "ViT": (".model_parts", "ViT"),
    "GPTWrapper": (".model_parts", "GPTWrapper"),
    "HierViT": (".model_parts", "HierViT"),
    "SerieTransformerEncoder": (".model_parts", "SerieTransformerEncoder"),
    "Transformer": (".model_parts", "Transformer"),
    "PreNorm": (".model_parts", "PreNorm"),
    "FeedForward": (".model_parts", "FeedForward"),
    "Attention": (".model_parts", "Attention"),
    "MrDataset": (".dataset", "MrDataset"),
    "MedicalImagePatchifier": (".patchify", "MedicalImagePatchifier"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
