import torch

from tools.models import ModelLoader


def test_monai_core_vqvae_accepts_published_prima_config_schema():
    # Deliberately use the legacy "num_channels" key from Prima's published
    # sample config. ModelLoader must adapt it to MONAI core's "channels".
    config = {
        "vqvae_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_layers": 1,
            "downsample_parameters": [[2, 4, 1, 1], [2, 2, 1, 0]],
            "upsample_parameters": [[2, 4, 1, 1, 0], [2, 2, 1, 0, 0]],
            "num_channels": [16, 16],
            "num_res_channels": [16, 16],
            "num_embeddings": 32,
            "embedding_dim": 2,
        }
    }

    model = ModelLoader.load_vqvae_model(config).eval()
    x = torch.randn(2, 1, 8, 32, 32)
    with torch.inference_mode():
        encoded = model.encode(x)

    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape[0] == 2
    assert encoded.ndim == 5
