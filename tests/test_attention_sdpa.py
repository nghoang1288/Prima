import torch

from Prima_training_and_evaluation.model_parts import sdpa_varlen


def manual_attention(qkv: torch.Tensor, culen: torch.Tensor) -> torch.Tensor:
    q, k, v = qkv.unbind(dim=1)
    out = torch.empty_like(q)
    bounds = culen.tolist()
    scale = q.shape[-1] ** -0.5
    for start, stop in zip(bounds[:-1], bounds[1:]):
        qs = q[start:stop].transpose(0, 1)
        ks = k[start:stop].transpose(0, 1)
        vs = v[start:stop].transpose(0, 1)
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
        probs = scores.softmax(dim=-1)
        out[start:stop] = torch.matmul(probs, vs).transpose(0, 1)
    return out


def test_sdpa_varlen_matches_historical_math():
    torch.manual_seed(3)
    qkv = torch.randn(11, 3, 2, 8)
    culen = torch.tensor([0, 4, 11], dtype=torch.int32)

    expected = manual_attention(qkv, culen)
    actual = sdpa_varlen(qkv, culen)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
