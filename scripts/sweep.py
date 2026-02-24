from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.eval_runtime import ensure_run_dir, write_json
from train_cb import get_arg_parser, train


def _parse_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="TT sweep runner with Pareto aggregation.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--discounts", type=float, nargs="+", default=None)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--num_samples_grid", type=int, nargs="+", default=None)
    parser.add_argument("--max_combinations", type=int, default=None)
    parser.add_argument(
        "--sweep_run_dir",
        type=str,
        default=None,
        help="Explicit sweep output directory. If unset, uses runs/<env>/<timestamp>/sweep.",
    )
    return parser.parse_args()


def _grid_values(args):
    alphas = args.alphas if args.alphas is not None else [float(args.alpha)]
    discounts = args.discounts if args.discounts is not None else [float(args.discount)]
    temperatures = (
        args.temperatures if args.temperatures is not None else [float(args.temperature)]
    )
    num_samples_grid = (
        args.num_samples_grid if args.num_samples_grid is not None else [int(args.num_samples)]
    )

    if args.smoke:
        # Keep sweep smoke runtime bounded.
        alphas = alphas[:2]
        discounts = discounts[:2]
        temperatures = temperatures[:2]
        num_samples_grid = num_samples_grid[:2]

    combos = list(product(alphas, discounts, temperatures, num_samples_grid))
    if args.max_combinations is not None:
        combos = combos[: int(args.max_combinations)]
    return combos


def _load_metrics(run_dir: str) -> dict:
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "r") as f:
        return json.load(f)


def _is_dominated(a: dict, b: dict) -> bool:
    # objectives: maximize return/satisfaction, minimize violation
    ar, asat, av = a["return_mean"], a["satisfaction_rate"], a["violation_rate"]
    br, bsat, bv = b["return_mean"], b["satisfaction_rate"], b["violation_rate"]
    if any(x is None for x in [ar, asat, av, br, bsat, bv]):
        return False
    no_worse = (br >= ar) and (bsat >= asat) and (bv <= av)
    strictly_better = (br > ar) or (bsat > asat) or (bv < av)
    return bool(no_worse and strictly_better)


def _pareto(rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if _is_dominated(row, other):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_plots(sweep_dir: str, rows: list[dict], pareto_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _vals(key):
        xs = []
        ys = []
        for row in rows:
            if row.get(key) is None or row.get("return_mean") is None:
                continue
            xs.append(float(row[key]))
            ys.append(float(row["return_mean"]))
        return xs, ys

    x_v, y_v = _vals("violation_rate")
    if x_v:
        plt.figure(figsize=(5, 4))
        plt.scatter(x_v, y_v, label="all")
        px = [float(r["violation_rate"]) for r in pareto_rows if r.get("violation_rate") is not None and r.get("return_mean") is not None]
        py = [float(r["return_mean"]) for r in pareto_rows if r.get("violation_rate") is not None and r.get("return_mean") is not None]
        if px:
            plt.scatter(px, py, marker="x", s=70, label="pareto")
        plt.xlabel("violation_rate (lower is better)")
        plt.ylabel("return_mean (higher is better)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "pareto_return_vs_violation.png"))
        plt.close()

    x_s, y_s = _vals("satisfaction_rate")
    if x_s:
        plt.figure(figsize=(5, 4))
        plt.scatter(x_s, y_s, label="all")
        px = [float(r["satisfaction_rate"]) for r in pareto_rows if r.get("satisfaction_rate") is not None and r.get("return_mean") is not None]
        py = [float(r["return_mean"]) for r in pareto_rows if r.get("satisfaction_rate") is not None and r.get("return_mean") is not None]
        if px:
            plt.scatter(px, py, marker="x", s=70, label="pareto")
        plt.xlabel("satisfaction_rate (higher is better)")
        plt.ylabel("return_mean (higher is better)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "pareto_return_vs_satisfaction.png"))
        plt.close()

    # Optional grid heatmap on alpha x temperature with mean satisfaction.
    unique_alpha = sorted({float(r["alpha"]) for r in rows if r.get("alpha") is not None})
    unique_temp = sorted({float(r["temperature"]) for r in rows if r.get("temperature") is not None})
    if len(unique_alpha) >= 2 and len(unique_temp) >= 2:
        mat = np.full((len(unique_alpha), len(unique_temp)), np.nan, dtype=np.float32)
        for i, a in enumerate(unique_alpha):
            for j, t in enumerate(unique_temp):
                vals = [
                    float(r["satisfaction_rate"])
                    for r in rows
                    if r.get("alpha") == a
                    and r.get("temperature") == t
                    and r.get("satisfaction_rate") is not None
                ]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        if np.isfinite(mat).any():
            plt.figure(figsize=(6, 4))
            plt.imshow(mat, aspect="auto", origin="lower")
            plt.colorbar(label="mean satisfaction_rate")
            plt.xticks(range(len(unique_temp)), [f"{t:.3g}" for t in unique_temp])
            plt.yticks(range(len(unique_alpha)), [f"{a:.3g}" for a in unique_alpha])
            plt.xlabel("temperature")
            plt.ylabel("alpha")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "grid_heatmap_satisfaction.png"))
            plt.close()


def main():
    args = _parse_args()
    combos = _grid_values(args)
    if not combos:
        raise ValueError("Sweep has no parameter combinations.")

    if args.sweep_run_dir is None:
        root, _, _ = ensure_run_dir(args.env, run_dir=None, base_dir=args.base_runs_dir)
        sweep_dir = os.path.join(root, "sweep")
    else:
        sweep_dir = args.sweep_run_dir
    os.makedirs(sweep_dir, exist_ok=True)

    rows = []
    for idx, (alpha, discount, temperature, num_samples) in enumerate(combos):
        run_args = copy.deepcopy(args)
        run_args.alpha = float(alpha)
        run_args.discount = float(discount)
        run_args.temperature = float(temperature)
        run_args.num_samples = int(num_samples)
        run_args.run_dir = os.path.join(sweep_dir, "runs", f"combo_{idx:03d}")
        run_args.save_path = run_args.run_dir
        train(run_args)
        metrics = _load_metrics(run_args.run_dir)
        row = {
            "combo_id": int(idx),
            "alpha": float(alpha),
            "discount": float(discount),
            "temperature": float(temperature),
            "num_samples": int(num_samples),
        }
        row.update(metrics)
        rows.append(row)

    pareto_rows = _pareto(rows)
    summary = {
        "env": args.env,
        "num_combinations": int(len(rows)),
        "rows": rows,
        "pareto_count": int(len(pareto_rows)),
    }
    write_json(os.path.join(sweep_dir, "summary.json"), summary)
    _write_csv(os.path.join(sweep_dir, "summary.csv"), rows)
    write_json(os.path.join(sweep_dir, "pareto_points.json"), pareto_rows)
    _save_plots(sweep_dir, rows, pareto_rows)
    print(f"Sweep artifacts saved to {sweep_dir}")


if __name__ == "__main__":
    main()
