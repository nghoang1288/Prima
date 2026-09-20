import torch
import torch.nn as nn

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)


def test_torchao_018_int8_dynamic_cpu_smoke():
    torch.manual_seed(17)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 8),
    ).eval()

    x = torch.randn(4, 64)
    with torch.inference_mode():
        reference = model(x)

    quantize_(
        model,
        Int8DynamicActivationInt8WeightConfig(version=2),
    )
    with torch.inference_mode():
        actual = model(x)

    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()
