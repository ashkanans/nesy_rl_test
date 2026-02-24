import json
import subprocess
import sys
from pathlib import Path


def _train_smoke_ckpt(repo_root: Path, env_name: str, run_dir: Path, spec: str):
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--env",
        env_name,
        "--smoke",
        "--spec",
        spec,
        "--use_safe_dfa",
        "--no_eval_after_train",
        "--run_dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    return run_dir / "cb_state_0.pt"


def _run_suite(repo_root: Path, env_name: str, checkpoint: Path, suite_dir: Path):
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_suite.py"),
        "--env",
        env_name,
        "--suite",
        "v1",
        "--smoke",
        "--max_suite_items",
        "1",
        "--checkpoint",
        str(checkpoint),
        "--suite_run_dir",
        str(suite_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    json_path = suite_dir / "suite_metrics.json"
    csv_path = suite_dir / "suite_metrics.csv"
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["env"] == env_name
    assert payload["num_rows"] >= 1


def _run_suite_dt(repo_root: Path, env_name: str, checkpoint: Path, suite_dir: Path):
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_suite.py"),
        "--env",
        env_name,
        "--model_type",
        "dt",
        "--suite",
        "v1",
        "--smoke",
        "--max_suite_items",
        "1",
        "--checkpoint",
        str(checkpoint),
        "--suite_run_dir",
        str(suite_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    json_path = suite_dir / "suite_metrics.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["env"] == env_name
    assert payload["num_rows"] >= 1
    assert payload["rows"][0]["checkpoint_path"] == str(checkpoint)


def test_eval_suite_smoke_for_frozenlake_and_cb(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    fl_train = tmp_path / "fl_train"
    fl_ckpt = _train_smoke_ckpt(
        repo_root=repo_root,
        env_name="frozenlake",
        run_dir=fl_train,
        spec="reach_goal_while_avoid_holes",
    )
    _run_suite(
        repo_root=repo_root,
        env_name="frozenlake",
        checkpoint=fl_ckpt,
        suite_dir=tmp_path / "fl_suite",
    )

    cb_train = tmp_path / "cb_train"
    cb_ckpt = _train_smoke_ckpt(
        repo_root=repo_root,
        env_name="cb",
        run_dir=cb_train,
        spec="avoid_single_bomb_22",
    )
    _run_suite(
        repo_root=repo_root,
        env_name="cb",
        checkpoint=cb_ckpt,
        suite_dir=tmp_path / "cb_suite",
    )


def test_eval_suite_smoke_dt_frozenlake(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "dt_train"
    ckpt = train_dir / "dt_state_0.pt"
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
    assert ckpt.exists()
    _run_suite_dt(
        repo_root=repo_root,
        env_name="frozenlake",
        checkpoint=ckpt,
        suite_dir=tmp_path / "dt_suite",
    )
