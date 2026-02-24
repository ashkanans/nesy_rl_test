import json
import subprocess
import sys
from pathlib import Path


CORE_SCHEMA_KEYS = {
    "return_mean",
    "violation_rate",
    "satisfaction_rate",
    "runtime_sec",
    "env",
    "spec",
    "seed",
}

EXTENDED_SCHEMA_KEYS = {
    "num_episodes",
    "violation_rate_episode",
    "violation_rate_step",
    "goal_rate",
    "hazard_hit_rate",
    "decoding_mode",
    "beam_width",
    "model_type",
    "checkpoint_path",
    "run_id",
    "timestamp_utc",
}


def _run(cmd, repo_root: Path):
    subprocess.run(cmd, check=True, cwd=repo_root)


def test_tt_dt_eval_schema_compatibility_frozenlake_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    spec = "reach_goal_while_avoid_holes"

    tt_train_dir = tmp_path / "tt_train"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--spec",
            spec,
            "--no_eval_after_train",
            "--run_dir",
            str(tt_train_dir),
        ],
        repo_root,
    )
    tt_ckpt = tt_train_dir / "cb_state_0.pt"
    assert tt_ckpt.exists()

    tt_eval_dir = tmp_path / "tt_eval"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--spec",
            spec,
            "--checkpoint",
            str(tt_ckpt),
            "--run_dir",
            str(tt_eval_dir),
        ],
        repo_root,
    )

    dt_train_dir = tmp_path / "dt_train"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_dt.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--run_dir",
            str(dt_train_dir),
        ],
        repo_root,
    )
    dt_ckpt = dt_train_dir / "dt_state_0.pt"
    assert dt_ckpt.exists()

    dt_eval_dir = tmp_path / "dt_eval"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "eval_dt.py"),
            "--env",
            "frozenlake",
            "--smoke",
            "--spec",
            spec,
            "--checkpoint",
            str(dt_ckpt),
            "--run_dir",
            str(dt_eval_dir),
        ],
        repo_root,
    )

    tt_metrics = json.loads((tt_eval_dir / "metrics.json").read_text())
    dt_metrics = json.loads((dt_eval_dir / "metrics.json").read_text())

    for key in CORE_SCHEMA_KEYS | EXTENDED_SCHEMA_KEYS:
        assert key in tt_metrics
        assert key in dt_metrics

    assert tt_metrics["spec"] == spec
    assert dt_metrics["spec"] == spec
    assert tt_metrics["env"] == "frozenlake"
    assert dt_metrics["env"] == "frozenlake"
    assert tt_metrics["model_type"] == "tt"
    assert dt_metrics["model_type"] == "dt"

    assert (tt_eval_dir / "dfa_summary.json").exists()
    assert (dt_eval_dir / "dfa_summary.json").exists()
    assert (tt_eval_dir / "automaton_rollout_stats.json").exists()
    assert (dt_eval_dir / "automaton_rollout_stats.json").exists()
