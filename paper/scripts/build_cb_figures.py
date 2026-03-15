#!/usr/bin/env python3
"""
Build ColourBomb paper figures/tables from completed CB run artifacts.

Expected inputs:
  runs/cb/alpha_sens_avoid_bombs_greedy_s*/baseline_metrics.csv
  runs/cb/alpha_sens_reach_goal_while_safe_greedy_s*/baseline_metrics.csv
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs" / "cb"
FIG_ROOT = REPO_ROOT / "paper" / "figures" / "colourbomb"
TAB_ROOT = REPO_ROOT / "paper" / "tables" / "colourbomb"

METRICS = [
    "return_mean",
    "satisfaction_rate",
    "violation_rate",
    "goal_rate",
    "bomb_hit_rate",
    "runtime_sec",
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _baseline_to_alpha(name: str) -> float:
    if name == "vanilla":
        return 0.0
    if name.startswith("logic_alpha"):
        return float(name.replace("logic_alpha", ""))
    raise ValueError(f"Unknown baseline format: {name}")


def _collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = "alpha_sens_*_greedy_s*/baseline_metrics.csv"
    for csv_path in sorted(RUNS_ROOT.glob(pattern)):
        run_name = csv_path.parent.name
        seed_match = re.search(r"_s(\d+)$", run_name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        spec_group = (
            "avoid_bombs"
            if "avoid_bombs" in run_name
            else "reach_goal_while_safe"
        )
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                baseline = row["baseline"]
                alpha = _baseline_to_alpha(baseline)
                out: dict[str, object] = {
                    "run_name": run_name,
                    "seed": seed,
                    "spec_group": spec_group,
                    "baseline": baseline,
                    "alpha": alpha,
                }
                for m in METRICS:
                    out[m] = float(row[m])
                rows.append(out)
    return rows


def _write_seed_table(rows: list[dict[str, object]]) -> None:
    TAB_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = TAB_ROOT / "cb_alpha_seed_metrics.csv"
    fields = ["run_name", "seed", "spec_group", "baseline", "alpha"] + METRICS
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote: {out_path}")


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bucket: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        key = (str(r["spec_group"]), str(r["baseline"]), float(r["alpha"]))
        bucket[key].append(r)

    out: list[dict[str, object]] = []
    for (spec_group, baseline, alpha), rs in sorted(
        bucket.items(), key=lambda x: (x[0][0], x[0][2])
    ):
        row: dict[str, object] = {
            "spec_group": spec_group,
            "baseline": baseline,
            "alpha": alpha,
            "n_seeds": len(rs),
        }
        for m in METRICS:
            xs = [float(r[m]) for r in rs]
            row[f"{m}_mean"] = _mean(xs)
            row[f"{m}_std"] = _std(xs)
        out.append(row)
    return out


def _write_agg_table(rows: list[dict[str, object]]) -> None:
    TAB_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = TAB_ROOT / "cb_alpha_agg_metrics.csv"
    fields = ["spec_group", "baseline", "alpha", "n_seeds"] + [
        f"{m}_{s}" for m in METRICS for s in ("mean", "std")
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote: {out_path}")


def _series(
    agg_rows: list[dict[str, object]], spec_group: str, metric: str
) -> tuple[list[float], list[float], list[float]]:
    filtered = [r for r in agg_rows if str(r["spec_group"]) == spec_group]
    filtered.sort(key=lambda r: float(r["alpha"]))
    x = [float(r["alpha"]) for r in filtered]
    y = [float(r[f"{metric}_mean"]) for r in filtered]
    e = [float(r[f"{metric}_std"]) for r in filtered]
    return x, y, e


def _plot_avoid_bombs(agg_rows: list[dict[str, object]]) -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / "cb_alpha_sweep_avoid_bombs.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    titles = [
        ("satisfaction_rate", "Satisfaction"),
        ("bomb_hit_rate", "Bomb Hit Rate"),
        ("return_mean", "Return"),
    ]
    for ax, (metric, title) in zip(axes, titles):
        x, y, e = _series(agg_rows, "avoid_bombs", metric)
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean over seeds")
    fig.suptitle("ColourBomb (spec=avoid_bombs, greedy decoding)")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def _plot_tradeoff(agg_rows: list[dict[str, object]]) -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / "cb_alpha_sweep_reach_goal_while_safe.png"

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    items = [
        ("satisfaction_rate", "Satisfaction"),
        ("goal_rate", "Goal Rate"),
        ("bomb_hit_rate", "Bomb Hit Rate"),
        ("return_mean", "Return"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), items):
        x, y, e = _series(agg_rows, "reach_goal_while_safe", metric)
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Mean over seeds")
    axes[1, 0].set_ylabel("Mean over seeds")
    fig.suptitle("ColourBomb (spec=reach_goal_while_safe, greedy decoding)")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def _plot_spec_comparison(agg_rows: list[dict[str, object]]) -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / "cb_spec_comparison_goal_vs_safety.png"

    # Compare best-alpha point per spec by satisfaction.
    best = {}
    for spec in ("avoid_bombs", "reach_goal_while_safe"):
        candidates = [r for r in agg_rows if str(r["spec_group"]) == spec]
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
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def _plot_metrics_bar_example(agg_rows: list[dict[str, object]]) -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / "cb_metrics_bar_example.png"

    specs = ["avoid_bombs", "reach_goal_while_safe"]
    metrics = ["satisfaction_rate", "goal_rate", "bomb_hit_rate", "return_mean"]
    metric_titles = {
        "satisfaction_rate": "Satisfaction",
        "goal_rate": "Goal Rate",
        "bomb_hit_rate": "Bomb Hit Rate",
        "return_mean": "Return",
    }

    fig, axes = plt.subplots(len(specs), len(metrics), figsize=(14.5, 6.6))
    if len(specs) == 1:
        axes = [axes]  # pragma: no cover

    for i, spec in enumerate(specs):
        rs = [r for r in agg_rows if str(r["spec_group"]) == spec]
        rs = sorted(rs, key=lambda r: float(r["alpha"]))
        labels = [
            "vanilla" if float(r["alpha"]) == 0.0 else f"a={float(r['alpha']):.2g}"
            for r in rs
        ]
        x = list(range(len(rs)))
        colors = [
            "#4C78A8" if float(r["alpha"]) == 0.0 else "#F58518"
            for r in rs
        ]
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            vals = [float(r[f"{metric}_mean"]) for r in rs]
            errs = [float(r[f"{metric}_std"]) for r in rs]
            ax.bar(x, vals, yerr=errs, color=colors, capsize=3)
            ax.set_title(metric_titles[metric], fontsize=10)
            ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel(spec.replace("_", " "))

    legend_handles = [
        Patch(facecolor="#4C78A8", label="vanilla (alpha=0)"),
        Patch(facecolor="#F58518", label="logic-regularized (alpha>0)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "ColourBomb: Aggregated Metrics-Bar View (mean ± std over seeds)",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.25, hspace=0.45)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def _plot_cb_layout() -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIG_ROOT / "cb_environment_layout.png"

    grid = [
        [".", ".", ".", ".", ".", ".", ".", "P", "P"],
        ["BLU", "BLU", "#", "#", "#", "#", "#", "P", "P"],
        ["BLU", "BLU", ".", ".", "B", ".", ".", ".", "."],
        ["B", ".", "#", "#", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "B", "."],
        [".", ".", ".", ".", ".", "#", ".", "#", "#"],
        [".", "#", "#", "#", ".", "#", ".", ".", "."],
        [".", "#", "G", "#", ".", ".", "#", "Y", "."],
        [".", ".", "S", ".", ".", ".", "B", "Y", "."],
    ]

    color_map = {
        ".": "#F6F6F6",
        "#": "#4D4D4D",
        "B": "#E45756",
        "S": "#54A24B",
        "G": "#72B7B2",
        "P": "#ECA82C",
        "Y": "#EECA3B",
        "BLU": "#4C78A8",
    }
    label_map = {
        ".": "",
        "#": "W",
        "B": "B",
        "S": "S",
        "G": "G",
        "P": "P",
        "Y": "Y",
        "BLU": "U",
    }

    n_rows = len(grid)
    n_cols = len(grid[0])
    fig, ax = plt.subplots(figsize=(7.2, 6.6), constrained_layout=True)

    for r in range(n_rows):
        for c in range(n_cols):
            sym = grid[r][c]
            y = n_rows - 1 - r
            rect = Rectangle(
                (c, y), 1, 1, facecolor=color_map[sym], edgecolor="#DDDDDD", linewidth=1.0
            )
            ax.add_patch(rect)
            txt = label_map[sym]
            if txt:
                ax.text(
                    c + 0.5,
                    y + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=("white" if sym in {"#", "B", "BLU"} else "black"),
                    fontweight="bold",
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(range(n_cols + 1))
    ax.set_yticks(range(n_rows + 1))
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("ColourBomb Environment Layout (9x9)")

    legend = [
        Patch(facecolor=color_map["S"], label="S: start"),
        Patch(facecolor=color_map["B"], label="B: bomb (hazard)"),
        Patch(facecolor=color_map["#"], label="W: wall (blocked)"),
        Patch(facecolor=color_map["P"], label="P/Y/U: goal terminals"),
        Patch(facecolor=color_map["."], edgecolor="#CCCCCC", label="empty cell"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def _write_brief_text(agg_rows: list[dict[str, object]]) -> None:
    out = TAB_ROOT / "cb_result_snippets.txt"
    lines = []
    for spec in ("avoid_bombs", "reach_goal_while_safe"):
        rs = [r for r in agg_rows if str(r["spec_group"]) == spec]
        rs = sorted(rs, key=lambda r: float(r["alpha"]))
        best_sat = max(rs, key=lambda r: float(r["satisfaction_rate_mean"]))
        best_goal = max(rs, key=lambda r: float(r["goal_rate_mean"]))
        lines.append(
            f"[{spec}] best satisfaction alpha={best_sat['alpha']}: "
            f"sat={best_sat['satisfaction_rate_mean']:.3f}, "
            f"goal={best_sat['goal_rate_mean']:.3f}, "
            f"bomb={best_sat['bomb_hit_rate_mean']:.3f}, "
            f"ret={best_sat['return_mean_mean']:.3f}"
        )
        lines.append(
            f"[{spec}] best goal alpha={best_goal['alpha']}: "
            f"sat={best_goal['satisfaction_rate_mean']:.3f}, "
            f"goal={best_goal['goal_rate_mean']:.3f}, "
            f"bomb={best_goal['bomb_hit_rate_mean']:.3f}, "
            f"ret={best_goal['return_mean_mean']:.3f}"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out}")


def main() -> None:
    rows = _collect_rows()
    if not rows:
        raise RuntimeError("No CB alpha-sensitivity run CSVs found under runs/cb.")
    _write_seed_table(rows)
    agg = _aggregate(rows)
    _write_agg_table(agg)
    _plot_avoid_bombs(agg)
    _plot_tradeoff(agg)
    _plot_spec_comparison(agg)
    _plot_metrics_bar_example(agg)
    _plot_cb_layout()
    _write_brief_text(agg)


if __name__ == "__main__":
    main()
