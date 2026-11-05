"""
Alpha-sweep figures for the ltlgrid runs of scripts/ltlgrid_train_all.py (mean +- std over seeds).

One figure per (reward, dataset, task) and one per (reward, dataset) averaged over tasks, each with four
panels against the logic-loss weight alpha: success rate, failure rate, return, final imitation loss.
alpha = 0 is the no-logic baseline.

  python scripts/ltlgrid_alpha_figures.py --results runs/sweep_s0 runs/sweep_s1 --out_dir runs/figures

--results takes several out_dirs (e.g. one per seed group); their results.csv files are concatenated.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.ltlgrid_dfa import LTLGRID_FORMULAS  # noqa: E402

PANELS = [
    ("success_rate", "Success Rate"),
    ("failure_rate", "Failure Rate"),
    ("return_mean", "Return"),
    ("final_imitation_loss", "Imitation Loss"),
]


def load_rows(result_dirs):
    rows = []
    for d in result_dirs:
        with open(Path(d) / "results.csv", newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "reward": r["reward"], "task": int(r["task"]), "dataset": r["dataset"],
                    "seed": int(r["seed"]), "alpha": float(r["alpha"]),
                    **{k: float(r[k]) for k, _ in PANELS},
                })
    return rows


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def draw(series, title, out_path):
    """series: {alpha: [value per seed]} per metric -> {metric: {alpha: [...]}}."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), PANELS):
        alphas = sorted(series[key])
        stats = [mean_std(series[key][a]) for a in alphas]
        ax.errorbar(alphas, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o", capsize=3, linewidth=1.8)
        ax.set_title(label)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Mean over seeds")
    axes[1, 0].set_ylabel("Mean over seeds")
    n_seeds = max(len(v) for v in series["success_rate"].values())
    fig.suptitle(f"{title} ({n_seeds} seeds, greedy decoding)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="out_dirs holding results.csv")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args(argv)

    rows = load_rows(args.results)
    out = Path(args.out_dir)
    groups = sorted({(r["reward"], r["dataset"]) for r in rows})
    written = 0
    for reward, dataset in groups:
        sub = [r for r in rows if r["reward"] == reward and r["dataset"] == dataset]
        by_task = defaultdict(lambda: {k: defaultdict(list) for k, _ in PANELS})
        by_seed = defaultdict(lambda: {k: defaultdict(list) for k, _ in PANELS})  # (alpha, seed) -> values over tasks
        for r in sub:
            for k, _ in PANELS:
                by_task[r["task"]][k][r["alpha"]].append(r[k])
                by_seed[(r["alpha"], r["seed"])][k]["v"].append(r[k])
        for task, series in sorted(by_task.items()):
            _, _, name = LTLGRID_FORMULAS[task - 1]
            draw(series, f"LTLGrid {name} (reward={reward}, data={dataset})",
                 out / reward / dataset / f"task{task}_alpha_sweep.png")
            written += 1
        # average over tasks first, so that the error bars are over seeds
        pooled = {k: defaultdict(list) for k, _ in PANELS}
        for (alpha, _seed), vals in by_seed.items():
            for k, _ in PANELS:
                pooled[k][alpha].append(float(np.mean(vals[k]["v"])))
        draw(pooled, f"LTLGrid, mean over tasks (reward={reward}, data={dataset})",
             out / reward / dataset / "all_tasks_alpha_sweep.png")
        written += 1
    print(f"wrote {written} figures under {out}")


if __name__ == "__main__":
    main()
