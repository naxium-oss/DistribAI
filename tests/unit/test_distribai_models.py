"""
Unit tests for native DistribAI model profiles and custom builders
"""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_distribai_wrapper_import():
    from worker.src.compute.distribai_models import (
        CustomModelBuilder,
        DistribAIModelWrapper,
        get_model,
    )

    assert DistribAIModelWrapper is not None
    assert CustomModelBuilder is not None
    assert get_model is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_custom_model_builder_tiny():
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_tiny_model(vocab_size=256)
    assert model is not None
    batch_size, seq_len = 2, 32
    x = torch.randint(0, 256, (batch_size, seq_len))
    with torch.no_grad():
        output = model(x)
    assert output is not None
    assert output.shape[0] == batch_size


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_custom_model_builder_small():
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_small_model(vocab_size=256)
    assert model is not None
    params = model.param_count()
    assert params > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_custom_model_builder_medium():
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_medium_model(vocab_size=256)
    assert model is not None
    params = model.param_count()
    assert params > 1000000


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_custom_model_creation():
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_custom_model(
        dim=128,
        n_layers=6,
        n_heads=8,
        vocab_size=256,
        seq_len=512,
    )
    assert model is not None
    assert model.config.dim == 128
    assert model.config.n_unique_layers == 6


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_model_save_load(tmp_path):
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_tiny_model(vocab_size=256)
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    model.save_checkpoint(str(checkpoint_path), step=100, loss=0.5)
    assert checkpoint_path.exists()
    new_model = CustomModelBuilder.create_tiny_model(vocab_size=256)
    info = new_model.load_checkpoint(str(checkpoint_path))
    assert not any(key.startswith("unique_layers.") for key in info["model_state"])
    assert info["step"] == 100
    assert info["loss"] == 0.5


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_model_gradient_norm():
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_tiny_model(vocab_size=256)
    x = torch.randint(0, 256, (2, 32))
    output = model(x)
    loss = output.mean()
    loss.backward()
    norm = model.get_gradient_norm()
    assert norm > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_model_gradient_clipping():
    from worker.src.compute.distribai_models import CustomModelBuilder

    model = CustomModelBuilder.create_tiny_model(vocab_size=256)
    x = torch.randint(0, 256, (2, 32))
    output = model(x)
    loss = output.mean()
    loss.backward()
    model.clip_gradients(max_norm=1.0)
    norm = model.get_gradient_norm()
    assert norm <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_get_model_function():
    from worker.src.compute.distribai_models import get_model

    tiny = get_model("tiny", vocab_size=256)
    assert tiny is not None
    small = get_model("small", vocab_size=256)
    assert small is not None
    custom = get_model("custom", vocab_size=256)
    assert custom is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_model_configs_dict():
    from worker.src.compute.distribai_models import DistribAIModelWrapper

    expected_profiles = {
        "distribai-tiny",
        "distribai-small",
        "distribai-base",
        "distribai-medium",
        "distribai-large",
        "distribai-xl",
        "distribai-lstm-small",
        "distribai-resnet-tiny",
        "distribai-hybrid-small",
        "distribai-dense-tiny",
    }
    assert set(DistribAIModelWrapper.MODEL_CONFIGS) == expected_profiles
    small_config = DistribAIModelWrapper.MODEL_CONFIGS["distribai-small"]
    assert "dim" in small_config
    assert "n_unique_layers" in small_config
    assert DistribAIModelWrapper.MODEL_CONFIGS["distribai-lstm-small"]["architecture"] == "lstm"
    assert DistribAIModelWrapper.MODEL_CONFIGS["distribai-resnet-tiny"]["architecture"] == "resnet_lm"
    assert (
        DistribAIModelWrapper.MODEL_CONFIGS["distribai-hybrid-small"]["architecture"]
        == "hybrid_attn_rnn"
    )
    assert DistribAIModelWrapper.MODEL_CONFIGS["distribai-dense-tiny"]["architecture"] == "dense_ffn"
    transformer_profiles = {
        name
        for name, config in DistribAIModelWrapper.MODEL_CONFIGS.items()
        if config["architecture"] == "decoder_transformer"
    }
    assert transformer_profiles == {
        "distribai-tiny",
        "distribai-small",
        "distribai-base",
        "distribai-medium",
        "distribai-large",
        "distribai-xl",
    }


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_model_invalid_name():
    from worker.src.compute.distribai_models import get_model

    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_native_decoder_is_causal_and_uses_configured_depth():
    from worker.src.compute.distribai_models import (
        DistribAIModelWrapper,
        DistribAITinyLanguageModel,
    )

    model = DistribAIModelWrapper("distribai-tiny", vocab_size=32, seq_len=16).eval()
    assert len(model.unique_layers) == model.config.n_unique_layers
    assert len(model.model.logical_layers) == model.config.n_logical_layers
    for index, layer in enumerate(model.model.logical_layers):
        assert layer is model.model.layers[index % model.config.n_unique_layers]
    assert not any(key.startswith("unique_layers.") for key in model.state_dict())

    prefix = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline = model(prefix)
        changed_future = model(torch.tensor([[1, 2, 3, 5]]))
    torch.testing.assert_close(baseline[:, :3], changed_future[:, :3])

    decoder = DistribAITinyLanguageModel(vocab_size=32, seq_len=16).eval()
    with pytest.raises(ValueError, match="exceeds configured limit"):
        decoder(torch.ones((1, 17), dtype=torch.long))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.parametrize(
    "family,config",
    [
        (
            "decoder_transformer",
            {"dim": 32, "n_unique_layers": 1, "n_logical_layers": 2, "n_heads": 4, "ffn_dim": 64},
        ),
        ("gru", {"dim": 32, "gru_layers": 1}),
        ("gated_conv", {"dim": 32, "n_logical_layers": 2, "conv_kernel": 3}),
        (
            "moe_decoder",
            {"dim": 32, "n_logical_layers": 2, "ffn_dim": 64, "num_experts": 2, "top_k": 1},
        ),
        ("lstm", {"dim": 32, "gru_layers": 1}),
        ("resnet_lm", {"dim": 32, "n_logical_layers": 2, "conv_kernel": 3}),
        (
            "hybrid_attn_rnn",
            {"dim": 32, "n_logical_layers": 2, "n_heads": 4, "ffn_dim": 64},
        ),
        ("dense_ffn", {"dim": 32, "n_logical_layers": 2, "ffn_dim": 64}),
    ],
)
def test_declarative_architecture_families_forward(family, config):
    from worker.src.compute.distribai_models import DistribAIModelWrapper

    model = DistribAIModelWrapper(
        "uploaded-architecture",
        vocab_size=16,
        custom_config={"family": family, "seq_len": 9, **config},
    ).eval()
    with torch.no_grad():
        output = model(torch.randint(0, 16, (2, 8)))

    assert output.shape == (2, 8, 16)
    assert model.config.family == family
    assert model.config.seq_len == 9
