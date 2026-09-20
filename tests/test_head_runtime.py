import torch
import torch.nn as nn

from tools.models import FullMRIModel


class FakeVisual(nn.Module):
    def forward(self, x, retpool=False):
        assert retpool
        return torch.tensor([[1.0, 2.0, 3.0, 4.0]])


class CountingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.thresh = 0.0
        self.calls = 0
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


def test_shared_task_head_is_evaluated_once():
    model = FullMRIModel.__new__(FullMRIModel)
    nn.Module.__init__(model)

    model.clipvisualmodel = FakeVisual()
    shared = CountingHead()
    model.diagnosisheads = {
        "diag_a": [shared, 0],
        "diag_b": [shared, 1],
    }
    model.referralheads = {
        "ref_a": [shared, 2],
    }
    model.priorityhead = nn.Linear(4, 3)

    out = model({}, heads_on_cpu=True)

    assert shared.calls == 1
    assert out["diagnosis"]["diag_a"].item() == 1.0
    assert out["diagnosis"]["diag_b"].item() == 2.0
    assert out["referral"]["ref_a"].item() == 3.0


def test_pytorch_dynamic_int8_cpu_linear_smoke():
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 8),
    ).eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )

    x = torch.randn(4, 64)
    with torch.inference_mode():
        out = quantized(x)

    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()
