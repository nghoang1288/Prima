import torch
from positional_encodings.torch_encodings import PositionalEncoding3D

from Prima_training_and_evaluation.patchify import (
    MedicalImagePatchifier,
    _sinusoidal_3d_at_coords,
)


def test_on_demand_3d_encoding_matches_original_grid():
    d = 30
    reference = PositionalEncoding3D(d)(
        torch.zeros(1, 7, 8, 9, d)
    )[0]

    coords = torch.tensor(
        [[0, 0, 0], [1, 2, 3], [6, 7, 8], [4, 1, 7]],
        dtype=torch.long,
    )
    actual = _sinusoidal_3d_at_coords(
        coords, d, dtype=torch.float32, device=torch.device("cpu")
    )
    expected = reference[
        coords[:, 0], coords[:, 1], coords[:, 2]
    ]

    assert torch.equal(actual, expected)


def test_patchifier_preserves_input_dtype_and_avoids_full_grid():
    patchifier = MedicalImagePatchifier(in_dim=256, d=30)
    assert patchifier.p_enc is None

    x = torch.randn(8, 2, 8, 8, 2, dtype=torch.float16)
    coords = [
        torch.tensor(
            [[z * 4, y * 32, x_ * 32]
             for z in range(2)
             for y in range(2)
             for x_ in range(2)],
            dtype=torch.long,
        )
    ]
    out = patchifier([x], coords)

    assert len(out) == 1
    assert out[0].dtype == torch.float16
    assert out[0].shape == (8, 289)
