from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class DecodeMode:
    name: str
    decoding_mode: str
    beam_width: int
    plan_horizon: int
    sat_rerank_weight: float
    hard_prune_reject_sink: bool


def parse_args():
    p = argparse.ArgumentParser(
        description="FL-02: TT decoding comparison on FrozenLake 4x4 non-slippery (avoid_holes)."
    )
    p.add_argument("--output_root", type=str, default="runs/frozenlake/fl02_tt_decoding_compare")

    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    p.add_argument("--episodes_per_eval_seed", type=int, default=10)

    p.add_argument("--train_num_episodes", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--policy_mix", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.0)

    p.add_argument("--spec", type=str, default="avoid_holes")
    p.add_argument("--use_safe_dfa", action="store_true")
    p.add_argument("--frozenlake_map_size", type=str, default="4x4", choices=["4x4", "8x8"])
    p.add_argument("--frozenlake_is_slippery", action="store_true")
    p.add_argument("--beam_width", type=int, default=8)
    p.add_argument("--plan_horizon", type=int, default=16)
    p.add_argument("--sat_rerank_weight", type=float, default=1.0)

    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def _run(cmd: list[str]):
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _train_checkpoint(args, train_dir: Path) -> Path:
    if args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if args.skip_train:
        raise ValueError("--skip_train requires --checkpoint to be set.")

    cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--env",
        "frozenlake",
        "--spec",
        args.spec,
        "--seed",
        str(args.train_seed),
        "--num_episodes",
        str(args.train_num_episodes),
        "--max_steps",
        str(args.max_steps),
        "--epochs",
        str(args.train_epochs),
        "--batch_size",
        str(args.batch_size),
        "--block_size",
        str(args.block_size),
        "--n_layer",
        str(args.n_layer),
        "--n_head",
        str(args.n_head),
        "--n_embd",
        str(args.n_embd),
        "--policy_mix",
        str(args.policy_mix),
        "--alpha",
        str(args.alpha),
        "--frozenlake_map_size",
        args.frozenlake_map_size,
        "--run_dir",
        str(train_dir),
        "--no_eval_after_train",
    ]
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.frozenlake_is_slippery:
        cmd.append("--frozenlake_is_slippery")

    _run(cmd)

    checkpoints = sorted(train_dir.glob("cb_state_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint generated under {train_dir}")
    return checkpoints[-1]


def _decode_modes(args) -> list[DecodeMode]:
    return [
        DecodeMode(
            name="greedy",
            decoding_mode="greedy",
            beam_width=int(args.beam_width),
            plan_horizon=int(args.plan_horizon),
            sat_rerank_weight=float(args.sat_rerank_weight),
            hard_prune_reject_sink=False,
        ),
        DecodeMode(
            name="beam",
            decoding_mode="beam",
            beam_width=int(args.beam_width),
            plan_horizon=int(args.plan_horizon),
            sat_rerank_weight=float(args.sat_rerank_weight),
            hard_prune_reject_sink=False,
        ),
        DecodeMode(
            name="constrained_beam_sat_rerank",
            decoding_mode="constrained_beam",
            beam_width=int(args.beam_width),
            plan_horizon=int(args.plan_horizon),
            sat_rerank_weight=float(args.sat_rerank_weight),
            hard_prune_reject_sink=False,
        ),
        DecodeMode(
            name="constrained_beam_hard_prune",
            decoding_mode="constrained_beam",
            beam_width=int(args.beam_width),
            plan_horizon=int(args.plan_horizon),
            sat_rerank_weight=float(args.sat_rerank_weight),
            hard_prune_reject_sink=True,
        ),
    ]


def _run_eval(args, ckpt: Path, mode: DecodeMode, eval_seed: int, eval_dir: Path):
    cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--env",
        "frozenlake",
        "--checkpoint",
        str(ckpt),
        "--spec",
        args.spec,
        "--alpha",
        str(args.alpha),
        "--seed",
        str(eval_seed),
        "--num_episodes",
        "32",
        "--max_steps",
        str(args.max_steps),
        "--block_size",
        str(args.block_size),
        "--policy_mix",
        str(args.policy_mix),
        "--frozenlake_map_size",
        args.frozenlake_map_size,
        "--eval_num_episodes",
        str(args.episodes_per_eval_seed),
        "--eval_max_steps",
        str(args.max_steps),
        "--decoding_mode",
        mode.decoding_mode,
        "--beam_width",
        str(mode.beam_width),
        "--plan_horizon",
        str(mode.plan_horizon),
        "--sat_rerank_weight",
        str(mode.sat_rerank_weight),
        "--run_dir",
        str(eval_dir),
    ]
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.frozenlake_is_slippery:
        cmd.append("--frozenlake_is_slippery")
    if mode.hard_prune_reject_sink:
        cmd.append("--hard_prune_reject_sink")
    if args.save_plots:
        cmd.append("--save_plots")

    _run(cmd)


def _aggregate_mode(mode_name: str, rows: list[dict], stats_rows: list[dict]) -> dict:
    returns = []
    sats = []
    hazards = []
    goals = []
    lengths = []
    runtimes = []
    fallback_decodes = 0
    reject_sink_entries = 0
    total_episodes = 0

    for metrics, stats in zip(rows, stats_rows):
        runtimes.append(float(metrics.get("runtime_sec") or 0.0))
        fallback_decodes += int(stats.get("fallback_decodes") or 0)
        reject_sink_entries += int(stats.get("reject_sink_entries") or 0)

        ep_returns = list(stats.get("episode_returns") or [])
        ep_sats = list(stats.get("episode_satisfaction") or [])
        ep_haz = list(stats.get("episode_hazard_hits") or [])
        ep_goal = list(stats.get("episode_goal_hits") or [])
        ep_len = list(stats.get("episode_lengths") or [])

        returns.extend(float(x) for x in ep_returns)
        sats.extend(float(x) for x in ep_sats)
        hazards.extend(float(x) for x in ep_haz)
        goals.extend(float(x) for x in ep_goal)
        lengths.extend(float(x) for x in ep_len)
        total_episodes += int(stats.get("num_episodes") or len(ep_returns))

    return {
        "decoding_label": mode_name,
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "satisfaction_rate": float(np.mean(sats)) if sats else None,
        "hazard_hit_rate": float(np.mean(hazards)) if hazards else None,
        "goal_rate": float(np.mean(goals)) if goals else None,
        "episode_length_mean": float(np.mean(lengths)) if lengths else None,
        "runtime_sec_mean": float(np.mean(runtimes)) if runtimes else None,
        "runtime_sec_total": float(np.sum(runtimes)) if runtimes else None,
        "reject_sink_entries_total": int(reject_sink_entries),
        "fallback_decodes_total": int(fallback_decodes),
        "num_eval_runs": int(len(rows)),
        "num_episodes_total": int(total_episodes),
    }


def _render_charts(charts_dir: Path, mode_rows: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    charts_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["decoding_label"] for row in mode_rows]

    # decoding_metrics_grouped.png
    metric_keys = ["hazard_hit_rate", "satisfaction_rate", "return_mean"]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.25
    plt.figure(figsize=(10, 4))
    for i, key in enumerate(metric_keys):
        vals = [float(row.get(key) or 0.0) for row in mode_rows]
        plt.bar(x + (i - 1) * width, vals, width=width, label=key)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("value")
    plt.title("FL-02 decoding comparison metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "decoding_metrics_grouped.png")
    plt.close()

    # decoding_runtime_bar.png
    runtime_vals = [float(row.get("runtime_sec_mean") or 0.0) for row in mode_rows]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, runtime_vals)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("mean runtime_sec per eval run")
    plt.title("FL-02 decoding runtime")
    plt.tight_layout()
    plt.savefig(charts_dir / "decoding_runtime_bar.png")
    plt.close()

    # decoding_pareto_return_vs_hazard.png
    xs = [float(row.get("hazard_hit_rate") or 0.0) for row in mode_rows]
    ys = [float(row.get("return_mean") or 0.0) for row in mode_rows]
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys)
    for row, x_val, y_val in zip(mode_rows, xs, ys):
        plt.annotate(row["decoding_label"], (x_val, y_val), textcoords="offset points", xytext=(4, 4))
    plt.xlabel("hazard_hit_rate (lower is better)")
    plt.ylabel("return_mean (higher is better)")
    plt.title("FL-02 return vs hazard Pareto view")
    plt.tight_layout()
    plt.savefig(charts_dir / "decoding_pareto_return_vs_hazard.png")
    plt.close()


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    evals_dir = output_root / "evals"
    aggregate_dir = output_root / "aggregate"
    charts_dir = output_root / "charts"

    for d in [train_dir, evals_dir, aggregate_dir, charts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _train_checkpoint(args, train_dir)

    modes = _decode_modes(args)
    per_seed_rows = []
    mode_payload = {}

    for mode in modes:
        mode_metrics = []
        mode_stats = []
        for seed in args.eval_seeds:
            eval_dir = evals_dir / mode.name / f"seed_{seed}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            _run_eval(args, checkpoint_path, mode, int(seed), eval_dir)

            metrics = _read_json(eval_dir / "metrics.json")
            stats = _read_json(eval_dir / "automaton_rollout_stats.json")
            mode_metrics.append(metrics)
            mode_stats.append(stats)
            per_seed_rows.append(
                {
                    "decoding_label": mode.name,
                    "eval_seed": int(seed),
                    "return_mean": metrics.get("return_mean"),
                    "satisfaction_rate": metrics.get("satisfaction_rate"),
                    "hazard_hit_rate": metrics.get("hazard_hit_rate"),
                    "runtime_sec": metrics.get("runtime_sec"),
                    "goal_rate": metrics.get("goal_rate"),
                    "reject_sink_entries": stats.get("reject_sink_entries"),
                    "fallback_decodes": stats.get("fallback_decodes"),
                    "num_episodes": stats.get("num_episodes"),
                    "run_dir": str(eval_dir),
                }
            )

        mode_payload[mode.name] = {
            "mode": {
                "decoding_mode": mode.decoding_mode,
                "beam_width": mode.beam_width,
                "plan_horizon": mode.plan_horizon,
                "sat_rerank_weight": mode.sat_rerank_weight,
                "hard_prune_reject_sink": mode.hard_prune_reject_sink,
            },
            "metrics_files": [str(evals_dir / mode.name / f"seed_{seed}" / "metrics.json") for seed in args.eval_seeds],
            "rollout_stats_files": [
                str(evals_dir / mode.name / f"seed_{seed}" / "automaton_rollout_stats.json")
                for seed in args.eval_seeds
            ],
        }

    mode_rows = []
    for mode in modes:
        rows = [r for r in per_seed_rows if r["decoding_label"] == mode.name]
        metrics_rows = [_read_json(Path(r["run_dir"]) / "metrics.json") for r in rows]
        stats_rows = [_read_json(Path(r["run_dir"]) / "automaton_rollout_stats.json") for r in rows]
        mode_rows.append(_aggregate_mode(mode.name, metrics_rows, stats_rows))

    mode_rows.sort(key=lambda r: r["decoding_label"])

    _write_csv(aggregate_dir / "per_seed_metrics.csv", per_seed_rows)
    _write_json(aggregate_dir / "per_seed_metrics.json", {"rows": per_seed_rows})
    _write_csv(aggregate_dir / "comparison.csv", mode_rows)
    _write_json(aggregate_dir / "comparison.json", {"rows": mode_rows})

    rollout_diag = {
        "per_mode": mode_payload,
        "aggregated": {
            row["decoding_label"]: {
                "reject_sink_entries_total": row["reject_sink_entries_total"],
                "fallback_decodes_total": row["fallback_decodes_total"],
                "num_episodes_total": row["num_episodes_total"],
            }
            for row in mode_rows
        },
    }
    _write_json(aggregate_dir / "rollout_diagnostics.json", rollout_diag)

    _render_charts(charts_dir, mode_rows)

    summary = {
        "job": "FL-02",
        "env": "frozenlake",
        "map_size": args.frozenlake_map_size,
        "is_slippery": bool(args.frozenlake_is_slippery),
        "spec": args.spec,
        "use_safe_dfa": bool(args.use_safe_dfa),
        "alpha": float(args.alpha),
        "beam_width": int(args.beam_width),
        "plan_horizon": int(args.plan_horizon),
        "sat_rerank_weight": float(args.sat_rerank_weight),
        "train_seed": int(args.train_seed),
        "eval_seeds": [int(s) for s in args.eval_seeds],
        "episodes_per_eval_seed": int(args.episodes_per_eval_seed),
        "total_eval_episodes_per_mode": int(len(args.eval_seeds) * args.episodes_per_eval_seed),
        "checkpoint_path": str(checkpoint_path),
        "output_root": str(output_root),
        "artifacts": {
            "comparison_csv": str(aggregate_dir / "comparison.csv"),
            "comparison_json": str(aggregate_dir / "comparison.json"),
            "per_seed_csv": str(aggregate_dir / "per_seed_metrics.csv"),
            "rollout_diagnostics_json": str(aggregate_dir / "rollout_diagnostics.json"),
            "charts": [
                str(charts_dir / "decoding_metrics_grouped.png"),
                str(charts_dir / "decoding_runtime_bar.png"),
                str(charts_dir / "decoding_pareto_return_vs_hazard.png"),
            ],
        },
    }
    _write_json(output_root / "fl02_summary.json", summary)

    print(f"FL-02 package saved to: {output_root}")


if __name__ == "__main__":
    main()
