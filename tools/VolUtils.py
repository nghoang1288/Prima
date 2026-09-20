import os
import json
import time
import logging
import concurrent.futures
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import pydicom as pyd
import SimpleITK as sitk
from glob import glob
from monai.transforms import (
    Compose,
    Spacingd,
    EnsureChannelFirstd,
    Resized,
    ToTensord,
    LoadImage,
    Resize,
)
import monai.transforms as montransform
import nibabel as nib

# Constants
MAX_SLICETHICKNESS_THRESHOLD = 4
DEFAULT_PATCH_SHAPE = [32, 32, 32]


def load_series_sitk(series_path: str) -> np.ndarray:
    """
    Load a series using SimpleITK and convert to numpy array.
    
    Args:
        series_path: Path to the series file
        
    Returns:
        Numpy array containing the image data
    """
    try:
        image = sitk.ReadImage(series_path)
        return sitk.GetArrayFromImage(image)
    except Exception as e:
        raise RuntimeError(f"Failed to load series from {series_path}: {str(e)}")


def percentile_mask(image: Union[np.ndarray, torch.Tensor], mask_threshold: int = 50) -> Union[np.ndarray, torch.Tensor]:
    """
    Create a binary mask based on percentile threshold.
    
    Args:
        image: Input image as numpy array or torch tensor
        mask_threshold: Threshold value for masking
        
    Returns:
        Binary mask of same type as input
    """
    try:
        # Handle both numpy arrays and torch tensors
        if isinstance(image, torch.Tensor):
            if image.max() < 5:
                mask = image > (mask_threshold / 100)
            else:
                mask = image > mask_threshold
        else:
            if image.max() < 5:
                mask = image > (mask_threshold / 100)
            else:
                mask = image > mask_threshold
        return mask
    except Exception as e:
        raise RuntimeError(f"Failed to create percentile mask: {str(e)}")


def adjusted_patch_shape(
    image_shape: Tuple[int, int, int],
    patch_shape: Optional[List[int]] = None,
    z_val: int = 4,
) -> Tuple[List[int], Optional[int]]:
    """
    Adjust the patch shape based on the image shape.
    
    Args:
        image_shape: Shape of the input image
        patch_shape: Optional initial patch shape
        z_val: Value to use for z dimension
        
    Returns:
        Tuple of (adjusted patch shape, z dimension index)
    """
    try:
        if patch_shape is None:
            patch_shape = DEFAULT_PATCH_SHAPE.copy()

        z_idx = None

        for idx, dim_size in enumerate(image_shape):
            if dim_size != 256:
                z_idx = idx
                patch_shape[z_idx] = z_val
                break

        return patch_shape, z_idx
    except Exception as e:
        raise RuntimeError(f"Failed to adjust patch shape: {str(e)}")


def pad_volume_for_patches(volume: Union[np.ndarray, torch.Tensor],
                           patch_size: List[int]) -> torch.Tensor:
    """
    Pad the volume so that it can be evenly divided into patches.
    
    Args:
        volume: Input volume as numpy array or torch tensor
        patch_size: Size of patches to create
        
    Returns:
        Padded volume as torch tensor
    """
    try:
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume.astype(np.float32))

        pad_sizes = [(ps - s % ps) % ps for s, ps in zip(volume.shape, patch_size)]
        pad = []
        for p in pad_sizes[::-1]:
            pad.extend([p // 2, p - p // 2])

        padded_volume = F.pad(volume, pad)
        return padded_volume
    except Exception as e:
        raise RuntimeError(f"Failed to pad volume: {str(e)}")


def scale(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Scale input to [0, 1] range.
    
    Args:
        x: Input array or tensor
        
    Returns:
        Scaled array or tensor
    """
    try:
        max_x = x.max().item() if isinstance(x, torch.Tensor) else x.max()
        if max_x > 0:
            return x / max_x
        return x
    except Exception as e:
        raise RuntimeError(f"Failed to scale input: {str(e)}")


def tokenize_volume(
    volume: Union[np.ndarray, torch.Tensor],
    mask_perc: int = 50
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]], List[float], Tuple[int, int, int], List[int], Optional[int]]:
    """Vectorized volume patchification with legacy-compatible ordering.

    Patches are emitted in z/y/x-major order, matching the historical nested
    Python loops, but extraction and mask statistics are computed with tensor
    unfold operations. Input is normalized to float32 to avoid the old
    float64 RAM amplification.
    """
    try:
        start = time.time()
        if isinstance(volume, np.ndarray):
            img = torch.from_numpy(np.asarray(volume, dtype=np.float32))
        else:
            img = volume.detach().to(dtype=torch.float32, device='cpu')

        patch_size, z_idx = adjusted_patch_shape(tuple(img.shape))
        logging.info(f"Patch shape is {patch_size}")

        padded_volume = pad_volume_for_patches(img, patch_size).contiguous()
        mask_ = percentile_mask(padded_volume, mask_perc)
        scaled_padded_vol = scale(padded_volume)

        p0, p1, p2 = patch_size
        patch_view = (
            scaled_padded_vol
            .unfold(0, p0, p0)
            .unfold(1, p1, p1)
            .unfold(2, p2, p2)
        )
        mask_view = (
            mask_
            .unfold(0, p0, p0)
            .unfold(1, p1, p1)
            .unfold(2, p2, p2)
        )

        # Contiguous reshape preserves the legacy z -> y -> x iteration order.
        patches_tensor = patch_view.contiguous().view(-1, p0, p1, p2)
        values_tensor = (
            mask_view.to(torch.float32)
            .mean(dim=(-1, -2, -3))
            .reshape(-1)
            .mul(100.0)
        )

        z_starts = torch.arange(0, padded_volume.shape[0], p0, dtype=torch.long)
        y_starts = torch.arange(0, padded_volume.shape[1], p1, dtype=torch.long)
        x_starts = torch.arange(0, padded_volume.shape[2], p2, dtype=torch.long)
        zz, yy, xx = torch.meshgrid(z_starts, y_starts, x_starts, indexing='ij')
        coords_tensor = torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)

        patches = list(patches_tensor.unbind(0))
        coordinates = [tuple(map(int, row)) for row in coords_tensor.tolist()]
        values_ = values_tensor.tolist()

        elapsed_time = time.time() - start
        logging.info(
            'Finished vectorized volume patchification: %d patches in %.2f seconds',
            len(patches), elapsed_time,
        )

        return patches, coordinates, values_, tuple(padded_volume.shape), patch_size, z_idx
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize volume: {str(e)}")


def resize_tokens_batch(tensor_list: List[torch.Tensor], patch_shape: List[int]) -> List[torch.Tensor]:
    """
    Resize a batch of tokens to a target shape.
    
    Args:
        tensor_list: List of input tensors
        patch_shape: Target shape for resizing
        
    Returns:
        List of resized tensors
    """
    try:
        resize = Resize(spatial_size=patch_shape)
        batch_tensor = np.stack(tensor_list)  # Stack tensors to create a batch
        resized_batch = resize(batch_tensor)
        return list(resized_batch)
    except Exception as e:
        raise RuntimeError(f"Failed to resize tokens batch: {str(e)}")
