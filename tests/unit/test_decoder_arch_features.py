"""Native decoder features: GQA, QK-norm, head gating, looped layers, residuals."""

from __future__ import annotations

import pytest
import torch

from services_python.architecture_config import validate_architecture_config
from worker.src.compute.distribai_models import (
    DistribAITinyLanguageModel,
    get_model,
)


@pytest.mark.unit
def test_validate_accepts_attention_feature_flags():
    config = validate_architecture_config(
        {
            "family": "decoder_transformer",
            "dim": 64,
            "n_heads": 8,
            "n_kv_heads": 2,
            "n_unique_layers": 2,
            "n_logical_layers": 4,
            "ffn_dim": 128,
            "qk_norm": True,
            "use_head_gating": True,
            "embedding_scale": True,
            "attn_res_block_size": 2,
        }
    )
    assert config["n_kv_heads"] == 2
    assert config["qk_norm"] is True
    assert config["use_head_gating"] is True
    assert config["embedding_scale"] is True
    assert config["attn_res_block_size"] == 2


@pytest.mark.unit
def test_validate_rejects_indivisible_kv_heads():
    with pytest.raises(ValueError, match="n_kv_heads"):
        validate_architecture_config(
            {
                "family": "decoder_transformer",
                "dim": 64,
                "n_heads": 8,
                "n_kv_heads": 3,
            }
        )


@pytest.mark.unit
def test_gqa_qk_norm_head_gating_forward_and_backward():
    model = DistribAITinyLanguageModel(
        vocab_size=64,
        dim=64,
        n_unique_layers=2,
        n_logical_layers=4,
        n_heads=8,
        n_kv_heads=2,
        ffn_dim=128,
        seq_len=32,
        qk_norm=True,
        use_head_gating=True,
        embedding_scale=True,
        attn_res_block_size=2,
    )
    tokens = torch.randint(0, 64, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, 64)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.unit
def test_looped_layers_share_unique_weights():
    model = DistribAITinyLanguageModel(
        vocab_size=32,
        dim=32,
        n_unique_layers=2,
        n_logical_layers=6,
        n_heads=4,
        n_kv_heads=4,
        ffn_dim=64,
        seq_len=16,
    )
    assert len(model.layers) == 2
    assert len(model.logical_layers) == 6
    assert model.logical_layers[0] is model.layers[0]
    assert model.logical_layers[2] is model.layers[0]
    assert model.logical_layers[1] is model.layers[1]


@pytest.mark.unit
def test_get_model_builds_featureful_decoder_from_config():
    wrapper = get_model(
        "uploaded-architecture",
        vocab_size=48,
        architecture_config={
            "family": "decoder_transformer",
            "dim": 48,
            "n_heads": 6,
            "n_kv_heads": 2,
            "n_unique_layers": 2,
            "n_logical_layers": 4,
            "ffn_dim": 96,
            "seq_len": 24,
            "qk_norm": True,
            "use_head_gating": True,
            "embedding_scale": True,
            "attn_res_block_size": 2,
        },
    )
    tokens = torch.randint(0, 48, (1, 8))
    assert wrapper(tokens).shape == (1, 8, 48)
