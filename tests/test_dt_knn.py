import json
import subprocess
import sys
from pathlib import Path

import torch

from models.dt_model import DecisionTransformerDiscrete


def _make_ckpt(path: Path, num_states: int, num_actions: int = 4):
    model = DecisionTransformerDiscrete(
        num_states=num_states,
        num_actions=num_actions,
        context_len=20,
        n_embd=32,
        n_layer=1,
        n_head=1,
        dropout=0.0,
        max_timestep=256,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head.bias.zero_()
        model.head.bias[1] = 5.0
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "env": "frozenlake",
            "seed": 0,
            "context_len": 20,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 32,
            "dropout": 0.0,
            "max_timestep": 256,
            "num_states": int(num_states),
            "num_actions": int(num_actions),
            "rtg_target": 1.0,
        },
    }
    torch.save(payload, path)


def _run_eval_knn(repo_root: Path, env_name: str, checkpoint: Path, run_dir: Path, spec: str):
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_dt.py"),
        "--env",
        env_name,
        "--checkpoint",
        str(checkpoint),
        "--run_dir",
        str(run_dir),
        "--dt_mode",
        "knn",
        "--knn_k",
        "8",
        "--num_episodes",
        "64",
        "--max_steps",
        "30",
        "--eval_num_episodes",
        "16",
        "--spec",
        spec,
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert payload["decoding_mode"] == "knn"


def test_eval_dt_knn_mode_runs_for_frozenlake_and_cb(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fl_ckpt = tmp_path / "fl.pt"
    cb_ckpt = tmp_path / "cb.pt"
    _make_ckpt(fl_ckpt, num_states=16)
    _make_ckpt(cb_ckpt, num_states=81)

    _run_eval_knn(
        repo_root=repo_root,
        env_name="frozenlake",
        checkpoint=fl_ckpt,
        run_dir=tmp_path / "fl_knn",
        spec="avoid_holes",
    )
    _run_eval_knn(
        repo_root=repo_root,
        env_name="cb",
        checkpoint=cb_ckpt,
        run_dir=tmp_path / "cb_knn",
        spec="avoid_single_bomb_22",
    )


def test_compare_dt_planners_writes_comparison_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    ckpt = tmp_path / "fl.pt"
    _make_ckpt(ckpt, num_states=16)
    run_dir = tmp_path / "cmp"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "compare_dt_planners.py"),
        "--env",
        "frozenlake",
        "--checkpoint",
        str(ckpt),
        "--spec",
        "avoid_holes",
        "--run_dir",
        str(run_dir),
        "--num_episodes",
        "64",
        "--max_steps",
        "30",
        "--eval_num_episodes",
        "32",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    json_path = run_dir / "comparison.json"
    csv_path = run_dir / "comparison.csv"
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text())
    assert "baseline_metrics" in payload
    assert "knn_metrics" in payload
    assert isinstance(payload.get("improved"), bool)
