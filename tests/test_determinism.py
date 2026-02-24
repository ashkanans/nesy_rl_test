import json
import os
import subprocess
import sys
from pathlib import Path

from tools.metrics_compare import compare_metrics


def _run(cmd, repo_root: Path, env):
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _pipeline_tt_cb(repo_root: Path, tmp_path: Path, seed: int):
    run_a = tmp_path / "tt_cb_a"
    run_b = tmp_path / "tt_cb_b"
    cmds = [
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--env",
            "cb",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "avoid_single_bomb_22",
            "--use_safe_dfa",
            "--no_eval_after_train",
            "--run_dir",
            str(run_a / "train"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate.py"),
            "--env",
            "cb",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "avoid_single_bomb_22",
            "--use_safe_dfa",
            "--checkpoint",
            str(run_a / "train" / "cb_state_0.pt"),
            "--run_dir",
            str(run_a / "eval"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--env",
            "cb",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "avoid_single_bomb_22",
            "--use_safe_dfa",
            "--no_eval_after_train",
            "--run_dir",
            str(run_b / "train"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate.py"),
            "--env",
            "cb",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "avoid_single_bomb_22",
            "--use_safe_dfa",
            "--checkpoint",
            str(run_b / "train" / "cb_state_0.pt"),
            "--run_dir",
            str(run_b / "eval"),
        ],
    ]
    return cmds, run_a / "eval" / "metrics.json", run_b / "eval" / "metrics.json"


def _pipeline_tt_frozenlake(repo_root: Path, tmp_path: Path, seed: int):
    run_a = tmp_path / "tt_fl_a"
    run_b = tmp_path / "tt_fl_b"
    cmds = [
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "reach_goal_while_avoid_holes",
            "--use_safe_dfa",
            "--no_eval_after_train",
            "--run_dir",
            str(run_a / "train"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "reach_goal_while_avoid_holes",
            "--use_safe_dfa",
            "--checkpoint",
            str(run_a / "train" / "cb_state_0.pt"),
            "--run_dir",
            str(run_a / "eval"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "reach_goal_while_avoid_holes",
            "--use_safe_dfa",
            "--no_eval_after_train",
            "--run_dir",
            str(run_b / "train"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--spec",
            "reach_goal_while_avoid_holes",
            "--use_safe_dfa",
            "--checkpoint",
            str(run_b / "train" / "cb_state_0.pt"),
            "--run_dir",
            str(run_b / "eval"),
        ],
    ]
    return cmds, run_a / "eval" / "metrics.json", run_b / "eval" / "metrics.json"


def _pipeline_dt_frozenlake(repo_root: Path, tmp_path: Path, seed: int):
    run_a = tmp_path / "dt_fl_a"
    run_b = tmp_path / "dt_fl_b"
    cmds = [
        [
            sys.executable,
            str(repo_root / "scripts" / "train_dt.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--no_eval_after_train",
            "--run_dir",
            str(run_a / "train"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_dt.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--checkpoint",
            str(run_a / "train" / "dt_state_0.pt"),
            "--run_dir",
            str(run_a / "eval"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "train_dt.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--no_eval_after_train",
            "--run_dir",
            str(run_b / "train"),
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_dt.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--seed",
            str(seed),
            "--checkpoint",
            str(run_b / "train" / "dt_state_0.pt"),
            "--run_dir",
            str(run_b / "eval"),
        ],
    ]
    return cmds, run_a / "eval" / "metrics.json", run_b / "eval" / "metrics.json"


def test_seeded_smoke_runs_are_deterministic_tt_and_dt(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    seed = 123
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONHASHSEED"] = str(seed)

    pipelines = [
        _pipeline_tt_cb,
        _pipeline_tt_frozenlake,
        _pipeline_dt_frozenlake,
    ]

    for builder in pipelines:
        cmds, metrics_a_path, metrics_b_path = builder(repo_root, tmp_path, seed)
        for cmd in cmds:
            _run(cmd, repo_root, env)
        lhs = _load_json(metrics_a_path)
        rhs = _load_json(metrics_b_path)
        mismatches = compare_metrics(lhs, rhs, float_atol=1e-8)
        assert not mismatches, f"{builder.__name__} mismatches: {mismatches}"
