import subprocess
import sys
from pathlib import Path


def test_mini_smoke_module_has_health_and_admin_checks():
    source = Path("scripts/dev/mini_smoke.py").read_text(encoding="utf-8")
    assert "/admin/health" in source
    assert "/admin/nodes" in source
    assert "/admin/jobs" in source
    assert "DISTRIBAI_ADMIN_SECRET" in source


def test_simulate_grid_cli_uses_real_entrypoints():
    import scripts.dev.simulate_grid_cli as simulate_grid_cli

    parser = simulate_grid_cli.build_parser()
    args = parser.parse_args(["--workers", "3", "--grpc-port", "19001", "--admin-port", "19002"])

    assert args.workers == 3
    source = Path("scripts/dev/simulate_grid_cli.py").read_text(encoding="utf-8")
    assert "-m" in source
    assert "services_python.orchestrator_grpc" in source
    assert "worker.src.daemon.run" in source


def _client_dashboard_sources() -> str:
    root = Path("client")
    parts = [root / "server.js"]
    for subdir in ("routes", "lib"):
        d = root / subdir
        if d.is_dir():
            parts.extend(sorted(d.glob("*.js")))
    return "".join(p.read_text(encoding="utf-8") for p in parts if p.is_file())


def test_dashboard_static_references_backed_routes():
    html = Path("worker/src/dashboard/static/node/index.html").read_text(encoding="utf-8")
    jobs = Path("worker/src/dashboard/static/node/jobs.html").read_text(encoding="utf-8")
    server = _client_dashboard_sources()
    preview = Path("worker/src/dashboard/static/node/index-preview.js").read_text(encoding="utf-8")

    for source in (html, jobs):
        assert source.count('id="architectureConfig"') == 1
        assert source.count('id="architectureConfigFile"') == 1
        assert source.count('id="createJobDataset"') == 1
        assert 'type="file"' in source
    assert '<select id="createJobModel"' not in html
    assert '<select id="createJobModel"' not in jobs
    assert "base_model: 'uploaded-architecture'" in preview
    assert "architecture_config: architectureConfig" in preview
    assert "steps," in preview

    for route in [
        "/api/admin/distribai/registry/sync",
        "/api/admin/public-release/publish",
        "/api/docs/list",
        "/api/docs/read",
        "/api/settings/reset-node",
        "/api/settings/unlink-node",
        "/api/worker/stream",
        "/api/admin/votes",
        "/api/admin/votes/",
        "/api/worker/jobs/",
    ]:
        assert route in html or route in server
        assert route in server

    assert "/api/admin/import/" not in preview
    assert "/api/admin/import/" not in server


def test_sdk_python_modules_compile():
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "sdk/python/distribai"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_playwright_config_uses_single_worker_for_shared_webserver():
    config = Path("playwright.config.js").read_text(encoding="utf-8")
    assert "workers: 1" in config


def test_bundle_script_writes_manifest_and_targets():
    bundle_src = Path("scripts/packaging/bundle.py").read_text(encoding="utf-8")
    assert "grid-manifest.json" in bundle_src
    assert "pyinstaller" in bundle_src.lower()
    assert "orchestrator_grpc.py" in bundle_src
    assert "daemon/run.py" in bundle_src
