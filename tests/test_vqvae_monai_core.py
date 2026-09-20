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
        direct_encoder = model.encoder(x)

    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape[0] == 2
    assert encoded.ndim == 5
    assert torch.equal(encoded, direct_encoder)


def test_vqvae_checkpoint_mmap_meta_assign_roundtrip(tmp_path):
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

    original = ModelLoader.load_vqvae_model(config).eval()
    checkpoint = tmp_path / "vqvae_state.pt"
    torch.save(original.state_dict(), checkpoint)

    load_config = {
        "vqvae_config": {
            **config["vqvae_config"],
            "ckpt_path": str(checkpoint),
        }
    }
    restored = ModelLoader.load_vqvae_model(load_config).eval()

    original_state = original.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], restored_state[key])
