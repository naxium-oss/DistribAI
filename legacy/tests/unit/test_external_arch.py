"""Unit tests for arbitrary external architecture loading (generic refs only)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from worker.src.compute.external_arch import (
    external_arch_allowed,
    load_external_architecture,
    looks_like_external_model_ref,
    normalize_model_ref,
)


@pytest.mark.unit
def test_looks_like_external_model_ref_accepts_hub_and_urls():
    assert looks_like_external_model_ref("org/custom-decoder")
    assert looks_like_external_model_ref("hf://org/custom-decoder")
    assert looks_like_external_model_ref("https://example.test/models/x")
    assert not looks_like_external_model_ref("tiny")
    assert not looks_like_external_model_ref("distribai-base")
    assert not looks_like_external_model_ref("no slash name")


@pytest.mark.unit
def test_normalize_model_ref_strips_hf_prefix():
    assert normalize_model_ref("hf://org/name") == "org/name"
    assert normalize_model_ref("  org/name  ") == "org/name"


@pytest.mark.unit
def test_external_arch_allowed_honors_env_and_explicit(monkeypatch):
    monkeypatch.delenv("DISTRIBAI_ALLOW_EXTERNAL_ARCH", raising=False)
    assert external_arch_allowed() is False
    assert external_arch_allowed(True) is True
    assert external_arch_allowed(False) is False
    monkeypatch.setenv("DISTRIBAI_ALLOW_EXTERNAL_ARCH", "1")
    assert external_arch_allowed() is True


@pytest.mark.unit
def test_load_external_architecture_requires_allow():
    with pytest.raises(PermissionError, match="External architectures are disabled"):
        load_external_architecture("org/any-model", allow=False)


def _install_fake_transformers(monkeypatch, *, causal_side_effect=None, causal_return=None):
    config_mod = MagicMock()
    causal_mod = MagicMock()
    base_mod = MagicMock()
    if causal_side_effect is not None:
        causal_mod.from_pretrained.side_effect = causal_side_effect
    else:
        causal_mod.from_pretrained.return_value = causal_return or MagicMock(name="causal")
    base_mod.from_pretrained.return_value = MagicMock(name="base")
    fake = types.ModuleType("transformers")
    fake.AutoConfig = config_mod
    fake.AutoModelForCausalLM = causal_mod
    fake.AutoModel = base_mod
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return config_mod, causal_mod, base_mod


@pytest.mark.unit
def test_load_external_architecture_uses_causal_lm(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOW_EXTERNAL_ARCH", "1")
    fake_model = MagicMock(name="causal")
    config_mod, causal_mod, base_mod = _install_fake_transformers(
        monkeypatch, causal_return=fake_model
    )
    model = load_external_architecture("org/custom-arch", allow=True)
    assert model is fake_model
    config_mod.from_pretrained.assert_called_once()
    causal_mod.from_pretrained.assert_called_once()
    base_mod.from_pretrained.assert_not_called()


@pytest.mark.unit
def test_load_external_architecture_falls_back_to_automodel(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOW_EXTERNAL_ARCH", "1")
    config_mod, causal_mod, base_mod = _install_fake_transformers(
        monkeypatch, causal_side_effect=ValueError("not a causal lm")
    )
    model = load_external_architecture("org/encoder-only", allow=True)
    assert model is base_mod.from_pretrained.return_value
    config_mod.from_pretrained.assert_called_once()
    causal_mod.from_pretrained.assert_called_once()
    base_mod.from_pretrained.assert_called_once()


@pytest.mark.unit
def test_load_external_architecture_from_scratch_applies_overrides_and_skips_download(monkeypatch):
    """from_scratch must build via from_config (no weight download) after
    applying config_overrides, so training-from-scratch jobs never require
    pulling a large/gated pretrained checkpoint just to get the arch shape."""
    monkeypatch.setenv("DISTRIBAI_ALLOW_EXTERNAL_ARCH", "1")
    fake_config = MagicMock(hidden_size=4096)
    causal_mod = MagicMock()
    causal_mod.from_config.return_value = MagicMock(name="from_scratch_model")
    config_mod = MagicMock()
    config_mod.from_pretrained.return_value = fake_config
    base_mod = MagicMock()
    fake = types.ModuleType("transformers")
    fake.AutoConfig = config_mod
    fake.AutoModelForCausalLM = causal_mod
    fake.AutoModel = base_mod
    monkeypatch.setitem(sys.modules, "transformers", fake)

    model = load_external_architecture(
        "org/huge-gated-checkpoint",
        allow=True,
        from_scratch=True,
        config_overrides={"hidden_size": 8, "num_hidden_layers": 1},
    )

    assert model is causal_mod.from_config.return_value
    assert fake_config.hidden_size == 8
    assert fake_config.num_hidden_layers == 1
    causal_mod.from_config.assert_called_once()
    causal_mod.from_pretrained.assert_not_called()
    base_mod.from_pretrained.assert_not_called()
