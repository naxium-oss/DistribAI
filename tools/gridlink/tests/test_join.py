from __future__ import annotations

import json

import pytest
from gridlink.cli import main
from gridlink.join import WORKER_BINARY, JoinPlan, plan_join
from gridlink.link import GridLink

TUNNEL_LINK = (
    "distribai://join?v=1&engine=native&provider=cloudflare&host=abc.trycloudflare.com"
    "&port=443&tls=1&invite=team-alpha&name=demo-grid"
)
LEGACY_LINK = (
    "distribai://join?v=1&engine=legacy&provider=cloudflare&host=abc.trycloudflare.com"
    "&port=443&tls=1&invite=team-alpha"
)


def test_plan_join_from_a_bare_host_uses_the_native_worker():
    plan = plan_join("10.0.0.5:50051")
    assert plan.engine == "native"
    assert plan.env["ORCHESTRATOR_URL"] == "http://10.0.0.5:50051"
    assert "GRPC_USE_TLS" not in plan.env
    assert plan.argv[0] == WORKER_BINARY
    assert plan.argv[plan.argv.index("--orchestrator") + 1] == "http://10.0.0.5:50051"


def test_plan_join_from_a_tunnel_link_keeps_invite_and_notes_tls():
    plan = plan_join(TUNNEL_LINK)
    assert plan.env["ORCHESTRATOR_URL"] == "https://abc.trycloudflare.com:443"
    assert plan.env["DISTRIBAI_INVITE_CODE"] == "team-alpha"
    assert "--invite" in plan.argv and "team-alpha" in plan.argv
    assert any("public certificate" in note for note in plan.notes)


def test_plan_join_legacy_engine_builds_the_python_command():
    plan = plan_join(LEGACY_LINK)
    assert plan.engine == "legacy"
    assert plan.env["ORCHESTRATOR_URL"] == "abc.trycloudflare.com:443"
    assert plan.env["GRPC_USE_TLS"] == "true"
    assert plan.argv[1:3] == ["-m", "worker.src.daemon.run"]


def test_engine_override_wins_over_the_link():
    plan = plan_join(TUNNEL_LINK, engine="legacy")
    assert plan.engine == "legacy"
    assert "-m" in plan.argv


def test_plan_join_ephemeral_and_state_dir_and_node_id():
    plan = plan_join(
        TUNNEL_LINK,
        node_id="colab-1",
        state_dir="/tmp/distribai",
        ephemeral=True,
        admin_url="https://abc.trycloudflare.com",
    )
    assert plan.env["DISTRIBAI_EPHEMERAL"] == "1"
    assert plan.env["STATE_DIR"] == "/tmp/distribai"
    assert plan.env["ADMIN_URL"] == "https://abc.trycloudflare.com"
    assert plan.argv[plan.argv.index("--node-id") + 1] == "colab-1"
    assert plan.argv[plan.argv.index("--work-dir") + 1] == "/tmp/distribai"


def test_plan_join_uses_a_custom_worker_binary():
    plan = plan_join(TUNNEL_LINK, worker_bin="./bin/distribai_worker")
    assert plan.argv[0] == "./bin/distribai_worker"


def test_plan_join_accepts_a_gridlink_object():
    plan = plan_join(GridLink.parse(TUNNEL_LINK))
    assert isinstance(plan, JoinPlan)
    assert plan.to_dict()["tls"] is True
    assert plan.to_dict()["engine"] == "native"


def test_plan_join_warns_about_plaintext_off_loopback():
    plan = plan_join("10.0.0.5:50061")
    assert any("plaintext" in note for note in plan.notes)
    assert any("LAN address" in note for note in plan.notes)


def test_shell_renders_env_and_command():
    plan = plan_join("host:50061", ephemeral=True)
    shell = plan.shell()
    assert "ORCHESTRATOR_URL=http://host:50061" in shell
    assert "distribai_worker" in shell


def test_render_is_human_readable():
    text = plan_join(TUNNEL_LINK).render()
    assert "engine       : native" in text
    assert "orchestrator : https://abc.trycloudflare.com:443" in text
    assert "notes        :" in text


# --------------------------------------------------------------------- cli ----


def test_cli_providers(capsys):
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "cloudflare" in out
    assert "Local bind" in out or "local" in out


def test_cli_link_json(capsys):
    assert (
        main(["link", "--host", "abc.trycloudflare.com", "--port", "443", "--tls", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["host"] == "abc.trycloudflare.com"
    assert payload["tls"] is True
    assert payload["engine"] == "native"
    assert payload["link"].startswith("distribai://join?")


def test_cli_link_json_legacy_engine(capsys):
    assert main(["link", "--host", "grid.example.com", "--engine", "legacy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "legacy"
    assert payload["port"] == 50051
    assert payload["orchestrator_url"] == "grid.example.com:50051"


def test_cli_join_json(capsys):
    assert main(["join", "10.0.0.5:50051", "--json", "--ephemeral"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "native"
    assert payload["orchestrator_url"] == "http://10.0.0.5:50051"
    assert payload["env"]["DISTRIBAI_EPHEMERAL"] == "1"


def test_cli_join_json_legacy_engine(capsys):
    assert main(["join", "10.0.0.5:50051", "--engine", "legacy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "legacy"
    assert payload["env"]["GRPC_USE_TLS"] == "false"


def test_cli_join_requires_target(capsys):
    assert main(["join"]) == 2
    assert "provide a join link" in capsys.readouterr().err


def test_cli_expose_local_json(capsys):
    assert main(["expose", "--provider", "local", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "local"
    assert payload["public_url"] is None
    assert payload["engine"] == "native"
    assert payload["link"].startswith("distribai://join?")


def test_cli_expose_legacy_default_port(capsys):
    assert main(["expose", "--provider", "local", "--engine", "legacy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["orchestrator_url"] == "127.0.0.1:50051"


def test_cli_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        main(["expose", "--provider", "wireguard"])
