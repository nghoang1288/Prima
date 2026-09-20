import numpy as np
import scipy.ndimage as ndi

from tools.mrcommondataset import generate_otsu_thresholds


def legacy_generate(coordinates, otsu, vol_shape, patch_shape, threshold=10):
    threshold_coords = [
        (idx, coordinates[idx])
        for idx, val in enumerate(otsu)
        if val >= threshold and val < threshold + 1
    ]
    highlight = [coord for _, coord in threshold_coords]

    mask = np.zeros(vol_shape, dtype=np.bool_)
    pz, py, px = patch_shape
    for z, y, x in highlight:
        mask[z:z + pz, y:y + py, x:x + px] = True

    filled = ndi.binary_fill_holes(mask)
    infill = []
    for z in range(0, vol_shape[0], pz):
        for y in range(0, vol_shape[1], py):
            for x in range(0, vol_shape[2], px):
                patch = filled[z:z + pz, y:y + py, x:x + px]
                if np.average(patch) == 1 and (z, y, x) not in highlight:
                    infill.append((z, y, x))
    return threshold_coords, infill


def test_patch_grid_hole_fill_matches_voxel_legacy():
    patch = [4, 8, 8]
    grid = (3, 3, 3)
    shape = tuple(g * p for g, p in zip(grid, patch))

    coordinates = []
    otsu = []
    # A closed 3x3x3 shell in one threshold bin; legacy fill should fill center.
    for gz in range(grid[0]):
        for gy in range(grid[1]):
            for gx in range(grid[2]):
                if (gz, gy, gx) == (1, 1, 1):
                    continue
                coordinates.append((gz * patch[0], gy * patch[1], gx * patch[2]))
                otsu.append(10.4)

    expected_out, expected_in = legacy_generate(
        coordinates, otsu, shape, patch, threshold=10
    )
    actual = generate_otsu_thresholds(
        coordinates, otsu, shape, patch, find_holes=True
    )[10]

    assert actual["OutfillCoords"] == expected_out
    assert sorted(actual["InfillCoords"]) == sorted(expected_in)
    assert actual["InfillCoords"] == [(4, 8, 8)]
