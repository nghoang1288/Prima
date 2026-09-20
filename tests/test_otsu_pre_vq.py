import torch

from tools.mrcommondataset import generate_otsu_thresholds
from tools.utilities import filtercoords, select_otsu_indices


def test_prefilter_then_encode_matches_encode_then_filter():
    # Build metadata with enough foreground patches to select threshold 5.
    coords = []
    values = []
    for z in range(8):
        for y in range(8):
            for x in range(2):
                coords.append((z * 4, y * 32, x * 32))
                # Mix bins while keeping >25 patches above the starting threshold.
                values.append(float((z + y + x) % 20))

    meta = {
        "OtsuThresholds": generate_otsu_thresholds(
            coords,
            values,
            vol_shape=(32, 256, 64),
            patch_shape=[4, 32, 32],
            find_holes=True,
        ),
        "emb_index": {idx: coord for idx, coord in enumerate(coords)},
    }

    tokens = torch.arange(len(coords) * 6, dtype=torch.float32).view(
        len(coords), 6
    )

    # A per-patch deterministic encoder stands in for the eval-mode VQ encoder.
    def encode(x):
        return x * 1.25 + 3.0

    all_embeddings = encode(tokens)
    useids, positions, percent = select_otsu_indices(
        meta,
        len(tokens),
        start_percentage=5,
        min_count=25,
    )
    old_embeddings, old_positions, old_ids = filtercoords(
        meta,
        percent,
        all_embeddings,
    )

    new_embeddings = encode(tokens[useids])

    assert torch.equal(useids, old_ids)
    assert torch.equal(positions, old_positions)
    assert torch.equal(new_embeddings, old_embeddings)
