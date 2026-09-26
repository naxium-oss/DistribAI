"""Unit tests for scripts.cli.identity (mirrors client/lib/identityStore.js)."""

from __future__ import annotations

import pytest

from scripts.cli.identity import ensure_identity, new_org_id, normalize_node_id


@pytest.mark.unit
def test_new_org_id_format():
    org_id = new_org_id()
    assert org_id.startswith("org-")
    assert len(org_id) == len("org-") + 16  # 8 bytes -> 16 hex chars, token_hex(8)


@pytest.mark.unit
def test_new_org_id_is_unique():
    assert new_org_id() != new_org_id()


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("My Node", "my-node"),
        ("  spaced   out  ", "spaced-out"),
        ("Already-Lower", "already-lower"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_node_id(raw, expected):
    assert normalize_node_id(raw) == expected


@pytest.mark.unit
def test_ensure_identity_fills_all_missing_fields():
    config, changed = ensure_identity({}, username="tester")
    assert changed is True
    assert config["org_id"].startswith("org-")
    assert config["node_name"] == "tester"
    assert config["node_id"] == "tester"


@pytest.mark.unit
def test_ensure_identity_is_noop_when_fully_populated():
    existing = {"org_id": "org-existing", "node_name": "existing-name", "node_id": "existing-name"}
    config, changed = ensure_identity(existing, username="ignored")
    assert changed is False
    assert config == existing


@pytest.mark.unit
def test_ensure_identity_preserves_existing_org_id():
    existing = {"org_id": "org-keep-me"}
    config, changed = ensure_identity(existing, username="tester")
    assert changed is True
    assert config["org_id"] == "org-keep-me"


@pytest.mark.unit
def test_ensure_identity_derives_node_id_from_existing_node_name():
    existing = {"org_id": "org-x", "node_name": "Gaming Rig 01"}
    config, changed = ensure_identity(existing, username="ignored")
    assert changed is True
    assert config["node_id"] == "gaming-rig-01"
    assert config["node_name"] == "Gaming Rig 01"


@pytest.mark.unit
def test_ensure_identity_falls_back_to_generated_node_id_when_name_blank(monkeypatch):
    def fake_token_hex(n: int) -> str:
        return "a" * (n * 2)

    monkeypatch.setattr("scripts.cli.identity.secrets.token_hex", fake_token_hex)
    config, changed = ensure_identity({"org_id": "org-x", "node_name": "   "})
    assert changed is True
    assert config["node_id"] == "node-aaaaaaaa"


@pytest.mark.unit
def test_ensure_identity_username_lookup_failure_is_non_fatal(monkeypatch):
    def raise_oserror():
        raise OSError("no such user")

    monkeypatch.setattr("scripts.cli.identity.getpass.getuser", raise_oserror)
    config, changed = ensure_identity({"org_id": "org-x"})
    assert changed is True
    assert "node_name" not in config
    # No node_name and no username -> base falls back to the literal "node".
    assert config["node_id"] == "node"


@pytest.mark.unit
def test_ensure_identity_does_not_mutate_input_dict():
    original = {"org_id": "org-x"}
    ensure_identity(original)
    assert original == {"org_id": "org-x"}
