import json
import subprocess
import sys
from pathlib import Path


def test_dt_train_eval_smoke_frozenlake(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "dt_train_fl"
    eval_dir = tmp_path / "dt_eval_fl"

    train_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train_dt.py"),
        "--env",
        "frozenlake",
        "--smoke",
        "--run_dir",
        str(train_dir),
    ]
    subprocess.run(train_cmd, check=True, cwd=repo_root)

    ckpt = train_dir / "dt_state_0.pt"
    metrics_train = train_dir / "metrics.json"
    assert ckpt.exists()
    assert metrics_train.exists()

    eval_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_dt.py"),
        "--env",
        "frozenlake",
        "--smoke",
        "--checkpoint",
        str(ckpt),
        "--run_dir",
        str(eval_dir),
    ]
    subprocess.run(eval_cmd, check=True, cwd=repo_root)

    metrics_eval = eval_dir / "metrics.json"
    assert metrics_eval.exists()
    payload = json.loads(metrics_eval.read_text())
    assert payload["model_type"] == "dt"
    assert "random_goal_rate" in payload
    assert "better_than_random" in payload


def test_dt_antmaze_smoke_graceful_skip(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "dt_antmaze_skip"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train_dt.py"),
        "--env",
        "antmaze",
        "--smoke",
        "--run_dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    metrics = run_dir / "metrics.json"
    assert metrics.exists()
    payload = json.loads(metrics.read_text())
    assert payload.get("skipped") is True
    assert "skip_reason" in payload
