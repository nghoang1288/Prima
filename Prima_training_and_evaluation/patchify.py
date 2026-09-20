import math
from typing import List, Optional

import torch


def _sinusoidal_3d_at_coords(
    coords: torch.Tensor,
    channels: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the same 3D sinusoidal encoding as positional-encodings 6.x.

    Unlike the historical implementation, this computes only the coordinates
    requested by the current MRI series instead of materializing a fixed
    100x100x100 grid (~120 MB for d=30).
    """
    coords = coords.to(device=device, dtype=torch.float32)
    internal_channels = int(math.ceil(channels / 6) * 2)
    if internal_channels % 2:
        internal_channels += 1

    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(
                0,
                internal_channels,
                2,
                device=device,
                dtype=torch.float32,
            )
            / internal_channels
        )
    )

    def axis_encoding(pos: torch.Tensor) -> torch.Tensor:
        phase = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)

    enc = torch.cat(
        (
            axis_encoding(coords[:, 0]),
            axis_encoding(coords[:, 1]),
            axis_encoding(coords[:, 2]),
        ),
        dim=1,
    )
    return enc[:, :channels].to(dtype=dtype)


class MedicalImagePatchifier(torch.nn.Module):
    """Add positional and orientation features to VQ-VAE visual tokens."""

    def __init__(self, in_dim: int = 1024, d: int = 30):
        super().__init__()
        self.out_dim = in_dim + d + 3
        if d % 3 != 0:
            raise ValueError("Positional encoding dimension must be divisible by 3")
        self.d = d
        # Keep the historical attribute for pickle compatibility, but do not
        # allocate the one-million-position table for new runtime instances.
        self.p_enc = None

    def forward(
        self,
        xs: List[torch.Tensor],
        coords: Optional[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        processed_tokens: List[torch.Tensor] = []

        for i, x in enumerate(xs):
            shapes = x.size()
            orientation = x.new_zeros(3)

            if shapes[2] == 2:
                orientation[0] = 1
                div1, div2, div3 = 4, 32, 32
            elif shapes[3] == 2:
                orientation[1] = 1
                div1, div2, div3 = 32, 4, 32
                x = x.transpose(2, 3)
            else:
                if shapes[4] != 2:
                    raise AssertionError(
                        f"Expected one VQ-VAE spatial axis of size 2, got {shapes}"
                    )
                orientation[2] = 1
                div1, div2, div3 = 32, 32, 4
                x = x.transpose(2, 4)

            if coords is None:
                div1, div2, div3 = 1, 1, 1
                xc, yc, zc = 8, 8, 8
                if orientation[0].item() == 1:
                    xc = len(x) // 64
                if orientation[1].item() == 1:
                    yc = len(x) // 64
                if orientation[2].item() == 1:
                    zc = len(x) // 64
                curr_coords = coordinate_tensor(
                    xc, yc, zc, device=x.device, dtype=torch.long
                ).view(-1, 3)
            else:
                curr_coords = coords[i].to(device=x.device, dtype=torch.long)

            div_tensor = torch.tensor(
                [div1, div2, div3],
                dtype=torch.long,
                device=curr_coords.device,
            ).unsqueeze(0)
            div_coords = curr_coords // div_tensor

            pos_encodings = _sinusoidal_3d_at_coords(
                div_coords,
                self.d,
                dtype=x.dtype,
                device=x.device,
            )
            orientation_info = orientation.unsqueeze(0).expand(shapes[0], -1)

            processed_tokens.append(
                torch.cat(
                    (
                        x.flatten(start_dim=1),
                        pos_encodings,
                        orientation_info,
                    ),
                    dim=1,
                )
            )

        return processed_tokens


def coordinate_tensor(
    x: int,
    y: int,
    z: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    xs = torch.arange(x, device=device, dtype=dtype)
    ys = torch.arange(y, device=device, dtype=dtype)
    zs = torch.arange(z, device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)


# Alias for checkpoints saved when this class was named RachelDatasetPatchifier.
RachelDatasetPatchifier = MedicalImagePatchifier
