#!/usr/bin/env python3
"""
Recreate Figure 5-style ColourBomb spec-comparison bar chart from aggregated CSV.

Default output is in repo root: cb_spec_comparison_goal_vs_safety.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _default_csv() -> Path:
    return Path(__file__).resolve().parent / "paper" / "tables" / "colourbomb" / "cb_alpha_agg_metrics.csv"


def _default_out() -> Path:
    return Path(__file__).resolve().parent / "cb_spec_comparison_goal_vs_safety.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate ColourBomb best-satisfaction spec-comparison figure."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_default_csv(),
        help="Input aggregated CSV (default: paper/tables/colourbomb/cb_alpha_agg_metrics.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_out(),
        help="Output PNG path (default: repo root cb_spec_comparison_goal_vs_safety.png)",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open()))

    best = {}
    for spec in ("avoid_bombs", "reach_goal_while_safe"):
        candidates = [r for r in rows if r["spec_group"] == spec]
        best_row = max(candidates, key=lambda r: float(r["satisfaction_rate_mean"]))
        best[spec] = best_row

    labels = ["avoid_bombs", "reach_goal_while_safe"]
    sat = [float(best[s]["satisfaction_rate_mean"]) for s in labels]
    goal = [float(best[s]["goal_rate_mean"]) for s in labels]
    bomb = [float(best[s]["bomb_hit_rate_mean"]) for s in labels]

    x = [0, 1]
    w = 0.22
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.bar([i - w for i in x], sat, width=w, label="satisfaction_rate")
    ax.bar(x, goal, width=w, label="goal_rate")
    ax.bar([i + w for i in x], bomb, width=w, label="bomb_hit_rate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean over seeds")
    ax.set_title("Best-satisfaction alpha per spec (greedy decoding)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3, loc="upper center")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
