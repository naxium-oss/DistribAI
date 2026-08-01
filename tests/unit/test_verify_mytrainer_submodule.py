import subprocess
import sys
from pathlib import Path


def test_verify_mytrainer_submodule_skips_when_missing(tmp_path, monkeypatch):
    script = Path("scripts/ci/verify_mytrainer_submodule.py")
    fake_root = tmp_path / "repo"
    (fake_root / "scripts" / "ci").mkdir(parents=True)
    (fake_root / "scripts" / "ci" / "verify_mytrainer_submodule.py").write_text(
        script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(fake_root / "scripts" / "ci" / "verify_mytrainer_submodule.py")],
        cwd=fake_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "SKIP" in result.stdout or "not found" in result.stdout.lower()


def test_verify_mytrainer_submodule_ok_with_marker(tmp_path):
    mytrainer = tmp_path / "external" / "mytrainer"
    (mytrainer / "configs").mkdir(parents=True)
    (mytrainer / "configs" / "grid_architectures.json").write_text("{}", encoding="utf-8")
    script = Path("scripts/ci/verify_mytrainer_submodule.py").read_text(encoding="utf-8")
    patched = script.replace(
        'REPO_ROOT = Path(__file__).resolve().parents[2]',
        f'REPO_ROOT = Path(r"{tmp_path}")',
    )
    runner = tmp_path / "run_verify.py"
    runner.write_text(patched, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
