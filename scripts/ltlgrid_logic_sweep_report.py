"""
Summarise a scripts/ltlgrid_logic_sweep.py run: paired effect of each logic-loss arm against the
baseline, per data regime and per task group, with bootstrap confidence intervals.

Pairing: an arm run is compared with the baseline run of the same (regime, task, seed), which shares
the training subset, initialisation seed and evaluation episodes. Primary metric: goal rate.

Uncertainty: hierarchical bootstrap. Tasks are resampled with replacement, then seeds within each
resampled task, and the statistic is the mean over tasks of the per-task mean difference. The sign
test counts tasks whose mean difference is positive vs negative (ties dropped).

Task groups are fixed by formula type, not by results: sequential = tasks whose formula contains an
ordered visit F(c0 & F c1) (3, 4, 5, 6, 9, 10), non-sequential = the others (1, 2, 7, 8).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = {
    "all tasks": list(range(1, 11)),
    "sequential (3,4,5,6,9,10)": [3, 4, 5, 6, 9, 10],
    "non-sequential (1,2,7,8)": [1, 2, 7, 8],
}
ARM_LABELS = {
    "oracle_a1": "oracle grounder, alpha=1",
    "sup_a1": "supervised grounder, alpha=1",
    "sup_a5": "supervised grounder, alpha=5",
    "weak_a1": "weak grounder, alpha=1",
}


def load_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "error" not in row:
            rows[row["id"]] = row
    return list(rows.values())


def paired_diffs(rows, arm, metric):
    """{regime: {task: [diff per seed]}} for arm - base over complete (regime, task, seed) pairs."""
    index = {(r["regime"], r["task"], r["seed"], r["arm"]): r for r in rows}
    out = defaultdict(lambda: defaultdict(list))
    for (regime, task, seed, a), r in index.items():
        if a != arm:
            continue
        base = index.get((regime, task, seed, "base"))
        if base is not None and metric in r and metric in base:
            out[regime][task].append(r[metric] - base[metric])
    return out


def bootstrap_ci(per_task, tasks, n_boot=10000, seed=0):
    """Hierarchical bootstrap CI of the mean over tasks of the per-task mean difference."""
    data = [np.asarray(per_task[t], dtype=float) for t in tasks if len(per_task.get(t, [])) > 0]
    if not data:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    point = float(np.mean([d.mean() for d in data]))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(len(data), size=len(data))
        stats[b] = np.mean([rng.choice(data[i], size=len(data[i]), replace=True).mean() for i in pick])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def sign_test(per_task, tasks):
    means = [float(np.mean(per_task[t])) for t in tasks if len(per_task.get(t, [])) > 0]
    wins = sum(m > 1e-9 for m in means)
    losses = sum(m < -1e-9 for m in means)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return wins, losses, p


def fmt_pp(x):
    return "n/a" if x != x else f"{100 * x:+.1f}"


def effect_table(rows, regimes, arms, metric, groups, n_boot):
    lines = []
    results = {}
    for regime in regimes:
        lines.append(f"\n#### Regime `{regime}`\n")
        lines.append("| arm | task group | pairs | mean diff (pp) | 95% CI | tasks up / down | sign-test p | pair wins / losses / ties |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for arm in arms:
            per_task = paired_diffs(rows, arm, metric).get(regime, {})
            for gname, tasks in groups.items():
                pt = {t: per_task[t] for t in tasks if t in per_task}
                if not pt:
                    continue
                point, lo, hi = bootstrap_ci(pt, tasks, n_boot=n_boot)
                up, down, p = sign_test(pt, tasks)
                flat = [d for t in tasks for d in pt.get(t, [])]
                w = sum(d > 1e-9 for d in flat)
                l = sum(d < -1e-9 for d in flat)
                results[(regime, arm, gname)] = (point, lo, hi)
                lines.append(
                    f"| {ARM_LABELS.get(arm, arm)} | {gname} | {len(flat)} | {fmt_pp(point)} | "
                    f"[{fmt_pp(lo)}, {fmt_pp(hi)}] | {up} / {down} | {p:.2f} | {w} / {l} / {len(flat) - w - l} |"
                )
    return "\n".join(lines), results


def baseline_table(rows, regimes):
    lines = ["| regime | " + " | ".join(f"T{t}" for t in range(1, 11)) + " | mean |", "|---|" + "---|" * 11]
    for regime in regimes:
        cells, means = [], []
        for t in range(1, 11):
            v = [r["goal_rate"] for r in rows if r["regime"] == regime and r["task"] == t and r["arm"] == "base"]
            if v:
                cells.append(f"{100 * np.mean(v):.0f}±{100 * np.std(v):.0f}")
                means.append(np.mean(v))
            else:
                cells.append("–")
        lines.append(f"| `{regime}` | " + " | ".join(cells) + f" | {100 * np.mean(means):.0f} |" if means else "")
    return "\n".join(lines)


def grounder_table(rows, regimes, arms):
    keys = [
        ("grounder_dataset_acc", "label acc (data)"),
        ("grounder_reward_replay_episode_acc", "reward replay (episodes)"),
        ("grounder_grid_acc", "label acc (all 64 pairs)"),
        ("grounder_unseen_pair_acc", "label acc (unseen pairs)"),
    ]
    lines = ["| regime | arm | " + " | ".join(k[1] for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for regime in regimes:
        for arm in arms:
            if arm == "oracle_a1":
                continue
            sub = [r for r in rows if r["regime"] == regime and r["arm"] == arm]
            if not sub:
                continue
            cells = []
            for k, _ in keys:
                v = [r[k] for r in sub if k in r and r[k] is not None]
                cells.append(f"{np.mean(v):.2f}" if v else "–")
            lines.append(f"| `{regime}` | {ARM_LABELS.get(arm, arm)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def forest_figure(results, regimes, arms, groups, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, mark, zero = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#8a8984"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        len(regimes), len(groups), figsize=(4.2 * len(groups), 0.45 * len(arms) * 1.0 * len(regimes) + 1.6 * len(regimes) + 0.8),
        sharex=True, squeeze=False,
    )
    fig.patch.set_facecolor(surface)
    all_vals = [v for k, vs in results.items() for v in vs if v == v]
    lim = max(0.05, max(abs(v) for v in all_vals) * 1.15) if all_vals else 0.1
    for i, regime in enumerate(regimes):
        for j, gname in enumerate(groups):
            ax = axes[i][j]
            ax.set_facecolor(surface)
            ys = np.arange(len(arms))[::-1]
            for y, arm in zip(ys, arms):
                key = (regime, arm, gname)
                if key not in results:
                    continue
                point, lo, hi = results[key]
                ax.plot([100 * lo, 100 * hi], [y, y], color=mark, lw=2, solid_capstyle="round", zorder=2)
                ax.plot([100 * point], [y], "o", ms=7, mfc=mark, mec=surface, mew=2, zorder=3)
                ax.text(100 * lim * 0.98, y, f"{100 * point:+.1f}", ha="right", va="center", color=ink2, fontsize=9)
            ax.axvline(0, color=zero, lw=1, zorder=1)
            ax.set_xlim(-100 * lim, 100 * lim)
            ax.set_ylim(-0.6, len(arms) - 0.4)
            ax.set_yticks(ys)
            ax.set_yticklabels([ARM_LABELS.get(a, a) if j == 0 else "" for a in arms], color=ink2)
            ax.tick_params(axis="x", colors=ink2, length=0)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.grid(True, color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(gname, color=ink, fontsize=10.5, loc="left", pad=8)
            if j == 0:
                ax.annotate(f"regime {regime}", xy=(0, 1.02), xycoords="axes fraction", color=ink2, fontsize=9, va="bottom") if i > 0 else None
            if i == len(regimes) - 1:
                ax.set_xlabel("goal rate vs. no-logic baseline (percentage points, paired)", color=ink2)
    fig.suptitle("Effect of the symbol logic loss on DT success", color=ink, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=surface)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--n_boot", type=int, default=10000)
    args = p.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    rows = load_rows(sweep_dir / "results.jsonl")
    regimes = sorted({r["regime"] for r in rows})
    arms = [a for a in ARM_LABELS if any(r["arm"] == a for r in rows)]
    n_seeds = len({r["seed"] for r in rows})

    out = [f"# ltlgrid logic-loss sweep\n", f"{len(rows)} completed runs, {n_seeds} seeds, regimes {regimes}.\n"]
    out.append("## Baseline goal rate (%, mean±sd over seeds)\n")
    out.append(baseline_table(rows, regimes))

    for metric, title in [("goal_rate", "Goal rate (primary)"), ("return_mean", "Mean return (secondary)"),
                          ("hazard_rate", "Failure rate (secondary)"), ("sup_loss", "Final imitation loss (secondary)")]:
        table, results = effect_table(rows, regimes, arms, metric, GROUPS if metric == "goal_rate" else {"all tasks": GROUPS["all tasks"]}, args.n_boot)
        out.append(f"\n## {title}: arm minus baseline\n")
        if metric == "sup_loss":
            out.append("Units are raw loss, shown x100 in the tables below (not percentage points).\n")
        out.append(table)
        if metric == "goal_rate":
            forest_figure(results, regimes, arms, GROUPS, sweep_dir / "forest_goal_rate.png")

    out.append("\n## Grounder quality (mean over runs)\n")
    out.append(grounder_table(rows, regimes, arms))
    out.append("\n![effect plot](forest_goal_rate.png)\n")
    (sweep_dir / "report.md").write_text("\n".join(out) + "\n")
    print(f"wrote {sweep_dir / 'report.md'} and forest_goal_rate.png")


if __name__ == "__main__":
    main()
