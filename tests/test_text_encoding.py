import torch

from tools.utilities import chartovec


def test_chartovec_runtime_limit_preserves_end_token():
    text = "a" * 500
    encoded = chartovec(text, max_length=200)

    assert len(encoded) == 200
    assert encoded[-1].item() == 46
    assert torch.all(encoded[:-1] == 1)


def test_chartovec_historical_behavior_unchanged_without_limit():
    encoded = chartovec("abc")
    assert encoded.tolist() == [1, 2, 3, 46]
