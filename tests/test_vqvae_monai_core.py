import torch
from monai.networks.nets import VQVAE


def test_monai_core_vqvae_supports_prima_style_3d_encode():
    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_layers=1,
        downsample_parameters=((2, 4, 1, 1), (2, 2, 1, 0)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 2, 1, 0, 0)),
        num_channels=(16, 16),
        num_res_channels=(16, 16),
        num_embeddings=32,
        embedding_dim=2,
    ).eval()

    x = torch.randn(2, 1, 8, 32, 32)
    with torch.inference_mode():
        encoded = model.encode(x)

    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape[0] == 2
    assert encoded.shape[1] == 16
    assert encoded.ndim == 5
