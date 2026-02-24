import json
import subprocess
import sys
from pathlib import Path

import torch

from models.dt_model import DecisionTransformerDiscrete


def _make_down_biased_dt_checkpoint(path: Path):
    model = DecisionTransformerDiscrete(
        num_states=16,
        num_actions=4,
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
        # FrozenLake action id 1 = DOWN.
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
            "num_states": 16,
            "num_actions": 4,
            "rtg_target": 1.0,
        },
    }
    torch.save(payload, path)


def _run_eval(repo_root: Path, run_dir: Path, ckpt_path: Path, mode: str):
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_dt.py"),
        "--env",
        "frozenlake",
        "--checkpoint",
        str(ckpt_path),
        "--run_dir",
        str(run_dir),
        "--seed",
        "123",
        "--eval_num_episodes",
        "100",
        "--eval_max_steps",
        "30",
        "--num_episodes",
        "64",
        "--max_steps",
        "30",
        "--spec",
        "avoid_holes",
    ]
    if mode == "constrained":
        cmd.extend(
            [
                "--dt_mode",
                "constrained",
                "--num_action_candidates",
                "4",
                "--lookahead_horizon",
                "2",
                "--hard_prune_reject_sink",
                "--candidate_sampling",
                "topk",
            ]
        )
    subprocess.run(cmd, check=True, cwd=repo_root)
    return json.loads((run_dir / "metrics.json").read_text())


def test_dt_constrained_reduces_hazard_rate_on_frozenlake(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = tmp_path / "down_biased_dt.pt"
    _make_down_biased_dt_checkpoint(ckpt_path)

    greedy_metrics = _run_eval(repo_root, tmp_path / "greedy", ckpt_path, mode="greedy")
    constrained_metrics = _run_eval(
        repo_root, tmp_path / "constrained", ckpt_path, mode="constrained"
    )

    assert greedy_metrics["decoding_mode"] == "greedy"
    assert constrained_metrics["decoding_mode"] == "constrained"
    assert constrained_metrics["hazard_hit_rate"] < greedy_metrics["hazard_hit_rate"]
