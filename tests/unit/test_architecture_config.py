"""Tests for the declarative, uploadable architecture configuration contract."""

import json

import pytest

from services_python.architecture_config import (
    MAX_ARCHITECTURE_CONFIG_BYTES,
    architecture_config_from_json,
    validate_architecture_config,
)
from services_python.schemas import validate_job_create


@pytest.mark.parametrize(
    "family",
    [
        "decoder_transformer",
        "gru",
        "gated_conv",
        "moe_decoder",
        "lstm",
        "resnet_lm",
        "hybrid_attn_rnn",
        "dense_ffn",
    ],
)
def test_supported_family_is_normalized(family):
    config = validate_architecture_config({"family": family, "dim": 128})

    assert config["family"] == family
    assert config["architecture"] == family
    assert config["version"] == 1


def test_lstm_reuses_gru_layers_key():
    config = validate_architecture_config({"family": "lstm", "dim": 128, "gru_layers": 3})
    assert config["gru_layers"] == 3


def test_hybrid_attn_rnn_requires_divisible_heads():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        validate_architecture_config(
            {"family": "hybrid_attn_rnn", "dim": 130, "n_heads": 8}
        )


def test_resnet_and_dense_ffn_accept_logical_depth():
    resnet = validate_architecture_config(
        {"family": "resnet_lm", "dim": 64, "n_logical_layers": 4, "conv_kernel": 3}
    )
    dense = validate_architecture_config(
        {"family": "dense_ffn", "dim": 64, "n_logical_layers": 6, "ffn_dim": 256}
    )
    assert resnet["family"] == "resnet_lm"
    assert dense["family"] == "dense_ffn"


def test_json_parser_rejects_non_json_architecture():
    with pytest.raises(ValueError, match="invalid architecture JSON"):
        architecture_config_from_json("not json")


def test_version_must_be_a_strict_integer():
    for version in (1.5, True, None):
        with pytest.raises(ValueError, match="version"):
            validate_architecture_config({"family": "gru", "version": version})


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unsupported architecture_config keys"):
        validate_architecture_config({"family": "gru", "python_class": "Bad"})


def test_family_aliases_must_agree():
    with pytest.raises(ValueError, match="must agree"):
        validate_architecture_config(
            {"family": "gru", "architecture": "decoder_transformer"}
        )


def test_bounds_and_cross_field_rules_are_enforced():
    with pytest.raises(ValueError, match="between 16 and 4096"):
        validate_architecture_config({"family": "gru", "dim": 8})
    with pytest.raises(ValueError, match="divisible by n_heads"):
        validate_architecture_config(
            {"family": "decoder_transformer", "dim": 130, "n_heads": 8}
        )
    with pytest.raises(ValueError, match="top_k cannot exceed"):
        validate_architecture_config(
            {"family": "moe_decoder", "dim": 128, "num_experts": 2, "top_k": 3}
        )
    with pytest.raises(ValueError, match="n_logical_layers"):
        validate_architecture_config(
            {
                "family": "decoder_transformer",
                "dim": 128,
                "n_unique_layers": 4,
                "n_logical_layers": 2,
            }
        )


def test_non_transformer_families_do_not_use_transformer_layer_rule():
    config = validate_architecture_config(
        {"family": "gated_conv", "dim": 128, "n_logical_layers": 2}
    )
    assert config["n_logical_layers"] == 2


def test_config_size_is_bounded():
    oversized = {"family": "gru", "description": "x" * MAX_ARCHITECTURE_CONFIG_BYTES}
    with pytest.raises(ValueError, match="exceeds 64 KiB"):
        validate_architecture_config(oversized)


def test_deep_config_is_rejected_without_recursion_error():
    nested = []
    for _ in range(1100):
        nested = [nested]
    with pytest.raises(ValueError, match="nested too deeply"):
        validate_architecture_config({"family": "gru", "nested": nested})


def test_job_schema_accepts_architecture_and_legacy_hparams_alias_is_normalizable():
    valid, error, request = validate_job_create(
        {
            "base_model": "uploaded-architecture",
            "dataset_ref": "https://example.test/data.json",
            "steps": 4,
            "architecture_config": {"family": "gru", "dim": 128, "gru_layers": 1},
            "hparams": {"lr": 0.01},
        }
    )

    assert valid, error
    assert request.architecture_config["family"] == "gru"
    assert request.hparams["lr"] == 0.01


def test_architecture_json_round_trip_is_finite():
    config = architecture_config_from_json(
        json.dumps({"family": "moe_decoder", "dim": 128, "num_experts": 4, "top_k": 2})
    )
    assert json.loads(json.dumps(config, allow_nan=False))["family"] == "moe_decoder"
