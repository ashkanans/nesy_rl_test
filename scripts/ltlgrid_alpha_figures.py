"""
Alpha-sweep figures for the ltlgrid runs of scripts/ltlgrid_train_all.py (mean +- std over seeds).

One figure per (reward, dataset, task) and one per (reward, dataset) averaged over tasks, each with four
panels against the logic-loss weight alpha: success rate, failure rate, return, final imitation loss.
alpha = 0 is the no-logic baseline and has no grounder, so it is the first point of every line. When the runs
compare grounders (oracle, supervised, label_free), each grounder is one line; results.csv files from before
the comparison hold one supervised grounder and give one line.

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
GROUNDER_ORDER = ["oracle", "supervised", "label_free"]
GROUNDER_STYLE = {  # colour, marker
    "oracle": ("#2ca02c", "s"),
    "supervised": ("#1f77b4", "o"),
    "label_free": ("#d62728", "^"),
}


def load_rows(result_dirs):
    rows = []
    for d in result_dirs:
        with open(Path(d) / "results.csv", newline="") as f:
            for r in csv.DictReader(f):
                alpha = float(r["alpha"])
                # no grounder column (older files) means the supervised grounder; alpha 0 has none
                grounder = "none" if alpha == 0 else (r.get("grounder") or "supervised")
                rows.append({
                    "reward": r["reward"], "task": int(r["task"]), "dataset": r["dataset"],
                    "seed": int(r["seed"]), "alpha": alpha, "grounder": grounder,
                    **{k: float(r[k]) for k, _ in PANELS},
                })
    return rows


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def split_by_grounder(rows):
    """[(grounder, rows)]: the alpha > 0 rows of each grounder, each with the shared alpha = 0 baseline rows."""
    base = [r for r in rows if r["alpha"] == 0]
    names = sorted({r["grounder"] for r in rows if r["alpha"] > 0},
                   key=lambda g: GROUNDER_ORDER.index(g) if g in GROUNDER_ORDER else len(GROUNDER_ORDER))
    if not names:
        return [("baseline", base)]
    return [(g, base + [r for r in rows if r["grounder"] == g]) for g in names]


def collect(rows):
    """{metric: {alpha: [value per run]}}"""
    out = {k: defaultdict(list) for k, _ in PANELS}
    for r in rows:
        for k, _ in PANELS:
            out[k][r["alpha"]].append(r[k])
    return out


def draw(lines, title, out_path):
    """lines: [(grounder, {metric: {alpha: [values per seed]}})]. One errorbar line per grounder."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    n_seeds = 0
    for ax, (key, label) in zip(axes.ravel(), PANELS):
        for j, (g, series) in enumerate(lines):
            alphas = sorted(series[key])
            stats = [mean_std(series[key][a]) for a in alphas]
            n_seeds = max(n_seeds, max(len(series[key][a]) for a in alphas))
            color, marker = GROUNDER_STYLE.get(g, (None, "o"))
            shift = (j - (len(lines) - 1) / 2) * 0.006  # keep the error bars of the grounders apart
            ax.errorbar([a + shift for a in alphas], [m for m, _ in stats], yerr=[s for _, s in stats],
                        marker=marker, capsize=3, linewidth=1.6, color=color, label=g.replace("_", " "))
        ax.set_title(label)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Mean over seeds")
    axes[1, 0].set_ylabel("Mean over seeds")
    if len(lines) > 1:
        axes[0, 0].legend(title="grounder", fontsize=8)
    fig.suptitle(f"{title} ({n_seeds} seeds, greedy decoding)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="out_dirs holding results.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", type=int, nargs="+", help="only draw these tasks (default: all found)")
    args = p.parse_args(argv)

    rows = load_rows(args.results)
    if args.tasks:
        rows = [r for r in rows if r["task"] in args.tasks]
    out = Path(args.out_dir)
    written = 0
    for reward, dataset in sorted({(r["reward"], r["dataset"]) for r in rows}):
        sub = [r for r in rows if r["reward"] == reward and r["dataset"] == dataset]
        for task in sorted({r["task"] for r in sub}):
            _, _, name = LTLGRID_FORMULAS[task - 1]
            lines = [(g, collect([r for r in rs if r["task"] == task])) for g, rs in split_by_grounder(sub)]
            draw(lines, f"LTLGrid {name} (reward={reward}, data={dataset})",
                 out / reward / dataset / f"task{task}_alpha_sweep.png")
            written += 1
        # average over tasks first, so that the error bars are over seeds
        lines = []
        for g, rs in split_by_grounder(sub):
            per = defaultdict(lambda: defaultdict(list))  # (alpha, seed) -> metric -> values over tasks
            for r in rs:
                for k, _ in PANELS:
                    per[(r["alpha"], r["seed"])][k].append(r[k])
            pooled = {k: defaultdict(list) for k, _ in PANELS}
            for (alpha, _seed), vals in per.items():
                for k, _ in PANELS:
                    pooled[k][alpha].append(float(np.mean(vals[k])))
            lines.append((g, pooled))
        draw(lines, f"LTLGrid, mean over tasks (reward={reward}, data={dataset})",
             out / reward / dataset / "all_tasks_alpha_sweep.png")
        written += 1
    print(f"wrote {written} figures under {out}")


if __name__ == "__main__":
    main()
