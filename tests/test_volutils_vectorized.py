import torch

from tools.VolUtils import (
    adjusted_patch_shape,
    pad_volume_for_patches,
    percentile_mask,
    scale,
    tokenize_volume,
)


def legacy_reference(volume: torch.Tensor, mask_perc: int = 50):
    patch_size, _ = adjusted_patch_shape(tuple(volume.shape))
    padded = pad_volume_for_patches(volume.float(), patch_size)
    mask = percentile_mask(padded, mask_perc)
    scaled = scale(padded)

    patches, coords, values = [], [], []
    for z in range(0, padded.shape[0], patch_size[0]):
        for y in range(0, padded.shape[1], patch_size[1]):
            for x in range(0, padded.shape[2], patch_size[2]):
                sl = (
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                patches.append(scaled[sl])
                coords.append((z, y, x))
                values.append(float(mask[sl].float().mean() * 100.0))
    return patches, coords, values


def test_vectorized_patchification_matches_legacy_order_and_values():
    torch.manual_seed(7)
    volume = torch.rand(9, 64, 64) * 100.0

    new_patches, new_coords, new_values, *_ = tokenize_volume(volume)
    old_patches, old_coords, old_values = legacy_reference(volume)

    assert new_coords == old_coords
    assert len(new_patches) == len(old_patches)
    assert torch.allclose(torch.tensor(new_values), torch.tensor(old_values))
    for new, old in zip(new_patches, old_patches):
        assert torch.allclose(new, old)


def test_exact_256_cube_defaults_to_sitk_array_z_axis():
    patch_shape, z_idx = adjusted_patch_shape((256, 256, 256))
    assert z_idx == 0
    assert patch_shape == [4, 32, 32]


def test_dataset_metadata_keeps_original_patch_shape():
    import numpy as np
    from tools.mrcommondataset import MrVoxelDataset

    volume = np.ones((8, 64, 64), dtype=np.float32)
    dataset = MrVoxelDataset([volume])
    tokens, meta = dataset[0]

    assert tokens.numel() > 0
    assert meta["PatchShape"] == [4, 32, 32]
    assert tuple(tokens.shape[1:]) == (8, 32, 32)
