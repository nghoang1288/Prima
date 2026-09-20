import sys
import types

import pytest
import torch
import torch.nn as nn

from tools.models import FullMRIModel, ModelLoader


def make_tiny_full_model():
    model = FullMRIModel.__new__(FullMRIModel)
    nn.Module.__init__(model)

    visual = nn.Linear(4, 4)
    clip = nn.Module()
    clip.visual_model = visual

    model.clipmodel = clip
    model.clipvisualmodel = visual
    model.diagnosisheads = {}
    model.referralheads = {}
    model.m1 = nn.ModuleList()
    model.m2 = nn.ModuleList()
    model.priorityhead = nn.Linear(4, 3)
    return model


@pytest.mark.parametrize(
    "historical_module",
    ["complete_visual_model", "full_model"],
)
def test_full_checkpoint_historical_module_alias_roundtrip(
    tmp_path, historical_module
):
    model = make_tiny_full_model()
    path = tmp_path / "full_model.pt"

    original_module = FullMRIModel.__module__
    fake_module = types.ModuleType(historical_module)
    fake_module.FullMRIModel = FullMRIModel
    sys.modules[historical_module] = fake_module
    FullMRIModel.__module__ = historical_module
    try:
        torch.save(model, path)
    finally:
        FullMRIModel.__module__ = original_module
        sys.modules.pop(historical_module, None)

    restored = ModelLoader.load_full_prima_model(
        {"full_model_ckpt": str(path)},
        device="cpu",
        low_vram=False,
        prune_inference_only=True,
    )

    assert isinstance(restored, FullMRIModel)
    assert hasattr(restored, "clipvisualmodel")
    assert not hasattr(restored, "clipmodel")
    assert restored.clipvisualmodel.weight.shape == (4, 4)
