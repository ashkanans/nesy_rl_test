from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from planning.eval_runtime import write_json
from tools.metrics_compare import compare_metrics


def _run(cmd, env):
    subprocess.run(shlex.split(cmd), check=True, env=env)


def _load_metrics(path: Path):
    import json

    return json.loads(path.read_text())


def _pipeline_commands(pipeline: str, root: Path, seed: int, run_a: Path, run_b: Path):
    py = shlex.quote(sys.executable)
    if pipeline == "tt_cb":
        train_a = f"{py} scripts/train.py --env cb --smoke --seed {seed} --spec avoid_single_bomb_22 --no_eval_after_train --run_dir {run_a / 'train'}"
        train_b = f"{py} scripts/train.py --env cb --smoke --seed {seed} --spec avoid_single_bomb_22 --no_eval_after_train --run_dir {run_b / 'train'}"
        eval_a = f"{py} scripts/evaluate.py --env cb --smoke --seed {seed} --spec avoid_single_bomb_22 --checkpoint {run_a / 'train' / 'cb_state_0.pt'} --run_dir {run_a / 'eval'}"
        eval_b = f"{py} scripts/evaluate.py --env cb --smoke --seed {seed} --spec avoid_single_bomb_22 --checkpoint {run_b / 'train' / 'cb_state_0.pt'} --run_dir {run_b / 'eval'}"
        return [train_a, eval_a, train_b, eval_b], run_a / "eval" / "metrics.json", run_b / "eval" / "metrics.json"
    if pipeline == "tt_frozenlake":
        train_a = f"{py} scripts/train.py --env frozenlake --smoke --seed {seed} --spec reach_goal_while_avoid_holes --no_eval_after_train --run_dir {run_a / 'train'}"
        train_b = f"{py} scripts/train.py --env frozenlake --smoke --seed {seed} --spec reach_goal_while_avoid_holes --no_eval_after_train --run_dir {run_b / 'train'}"
        eval_a = f"{py} scripts/evaluate.py --env frozenlake --smoke --seed {seed} --spec reach_goal_while_avoid_holes --checkpoint {run_a / 'train' / 'cb_state_0.pt'} --run_dir {run_a / 'eval'}"
        eval_b = f"{py} scripts/evaluate.py --env frozenlake --smoke --seed {seed} --spec reach_goal_while_avoid_holes --checkpoint {run_b / 'train' / 'cb_state_0.pt'} --run_dir {run_b / 'eval'}"
        return [train_a, eval_a, train_b, eval_b], run_a / "eval" / "metrics.json", run_b / "eval" / "metrics.json"
    if pipeline == "dt_frozenlake":
        train_a = f"{py} scripts/train_dt.py --env frozenlake --smoke --seed {seed} --no_eval_after_train --run_dir {run_a / 'train'}"
        train_b = f"{py} scripts/train_dt.py --env frozenlake --smoke --seed {seed} --no_eval_after_train --run_dir {run_b / 'train'}"
        eval_a = f"{py} scripts/eval_dt.py --env frozenlake --smoke --seed {seed} --checkpoint {run_a / 'train' / 'dt_state_0.pt'} --run_dir {run_a / 'eval'}"
        eval_b = f"{py} scripts/eval_dt.py --env frozenlake --smoke --seed {seed} --checkpoint {run_b / 'train' / 'dt_state_0.pt'} --run_dir {run_b / 'eval'}"
        return [train_a, eval_a, train_b, eval_b], run_a / "eval" / "metrics.json", run_b / "eval" / "metrics.json"
    raise ValueError(f"Unsupported pipeline '{pipeline}'")


def main():
    parser = argparse.ArgumentParser(description="Manual determinism checker for smoke pipelines.")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["tt_cb", "tt_frozenlake", "dt_frozenlake"],
        default="tt_cb",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--work_dir", type=str, default="runs/determinism_check")
    parser.add_argument("--float_atol", type=float, default=1e-8)
    args = parser.parse_args()

    root = Path.cwd()
    base = Path(args.work_dir) / args.pipeline
    run_a = base / "run_a"
    run_b = base / "run_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("PYTHONHASHSEED", str(args.seed))

    commands, metrics_a_path, metrics_b_path = _pipeline_commands(
        args.pipeline, root, args.seed, run_a, run_b
    )
    for cmd in commands:
        _run(cmd, env=env)

    metrics_a = _load_metrics(metrics_a_path)
    metrics_b = _load_metrics(metrics_b_path)
    mismatches = compare_metrics(metrics_a, metrics_b, float_atol=args.float_atol)

    report = {
        "pipeline": args.pipeline,
        "seed": int(args.seed),
        "metrics_a_path": str(metrics_a_path),
        "metrics_b_path": str(metrics_b_path),
        "deterministic": len(mismatches) == 0,
        "mismatches": mismatches,
    }
    out = base / "determinism_report.json"
    write_json(str(out), report)
    print(f"Determinism report saved to {out}")
    if mismatches:
        raise SystemExit("\n".join(["Determinism check failed:"] + mismatches))


if __name__ == "__main__":
    main()
