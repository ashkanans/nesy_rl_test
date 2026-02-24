from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.eval_runtime import ensure_run_dir, write_json


def _parse_args():
    p = argparse.ArgumentParser(description="Compare DT greedy vs kNN suffix planners.")
    p.add_argument("--env", type=str, choices=["cb", "frozenlake"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_num_episodes", type=int, default=100)
    p.add_argument("--eval_max_steps", type=int, default=None)
    p.add_argument("--num_episodes", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--base_runs_dir", type=str, default="runs")
    p.add_argument("--frozenlake_map_size", type=str, choices=["4x4", "8x8"], default="4x4")
    p.add_argument("--frozenlake_is_slippery", action="store_true")
    p.add_argument("--policy_mix", type=float, default=0.0)
    p.add_argument("--knn_k", type=int, default=16)
    p.add_argument("--knn_return_weight", type=float, default=1.0)
    p.add_argument("--knn_satisfaction_weight", type=float, default=2.0)
    return p.parse_args()


def _run_eval(args, mode: str, out_dir: str):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_dt.py"),
        "--env",
        args.env,
        "--checkpoint",
        args.checkpoint,
        "--spec",
        args.spec,
        "--seed",
        str(args.seed),
        "--eval_num_episodes",
        str(args.eval_num_episodes),
        "--num_episodes",
        str(args.num_episodes),
        "--max_steps",
        str(args.max_steps),
        "--run_dir",
        out_dir,
        "--dt_mode",
        mode,
    ]
    if args.eval_max_steps is not None:
        cmd.extend(["--eval_max_steps", str(args.eval_max_steps)])
    if args.env == "frozenlake":
        cmd.extend(["--frozenlake_map_size", args.frozenlake_map_size])
        if args.frozenlake_is_slippery:
            cmd.append("--frozenlake_is_slippery")
        cmd.extend(["--policy_mix", str(args.policy_mix)])
    if mode == "knn":
        cmd.extend(
            [
                "--knn_k",
                str(args.knn_k),
                "--knn_return_weight",
                str(args.knn_return_weight),
                "--knn_satisfaction_weight",
                str(args.knn_satisfaction_weight),
            ]
        )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    with open(os.path.join(out_dir, "metrics.json"), "r") as f:
        return json.load(f)


def _write_csv(path: str, baseline: dict, knn: dict, improved: bool):
    rows = [
        {"planner": "greedy", **baseline},
        {"planner": "knn", **knn},
    ]
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["improved_vs_greedy"])
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["improved_vs_greedy"] = bool(improved and row["planner"] == "knn")
            writer.writerow(row)


def main():
    args = _parse_args()
    if args.run_dir is None:
        run_dir, _, _ = ensure_run_dir(args.env, base_dir=args.base_runs_dir)
        run_dir = os.path.join(run_dir, "dt_knn_compare")
    else:
        run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    greedy_dir = os.path.join(run_dir, "greedy")
    knn_dir = os.path.join(run_dir, "knn")
    os.makedirs(greedy_dir, exist_ok=True)
    os.makedirs(knn_dir, exist_ok=True)

    greedy_metrics = _run_eval(args, mode="greedy", out_dir=greedy_dir)
    knn_metrics = _run_eval(args, mode="knn", out_dir=knn_dir)

    greedy_sat = float(greedy_metrics.get("satisfaction_rate") or 0.0)
    knn_sat = float(knn_metrics.get("satisfaction_rate") or 0.0)
    improved = bool(knn_sat > greedy_sat)

    payload = {
        "env": args.env,
        "spec": args.spec,
        "seed": int(args.seed),
        "baseline_metrics": greedy_metrics,
        "knn_metrics": knn_metrics,
        "improved": improved,
        "improvement_criterion": "knn.satisfaction_rate > greedy.satisfaction_rate",
    }
    json_path = os.path.join(run_dir, "comparison.json")
    csv_path = os.path.join(run_dir, "comparison.csv")
    write_json(json_path, payload)
    _write_csv(csv_path, greedy_metrics, knn_metrics, improved)
    print(f"Comparison written to {run_dir}")
    print(f"improved={improved}")


if __name__ == "__main__":
    main()
