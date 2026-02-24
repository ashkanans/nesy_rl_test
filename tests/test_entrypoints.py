import subprocess
import sys
from pathlib import Path


def test_train_cb_compat_entrypoint_help():
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "train_cb.py"), "--help"]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--env" in result.stdout
    assert "--smoke" in result.stdout


def test_new_train_entrypoint_help():
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "scripts" / "train.py"), "--help"]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--env" in result.stdout
