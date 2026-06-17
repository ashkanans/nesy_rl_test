#!/usr/bin/env python3
"""
Recreate the ColourBomb Reach-while-Safe alpha-sweep figure from fixed aggregated
values (mean and std over 3 runs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def _default_out() -> Path:
    repo_root = Path(__file__).resolve().parent
    return (
        repo_root
        / "paper"
        / "figures"
        / "colourbomb"
        / "cb_alpha_sweep_reach_goal_while_safe.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate Reach-while-Safe alpha-sweep figure from fixed data."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_out(),
        help="Output PNG path (default: paper figure path).",
    )
    args = parser.parse_args()

    # Exact data from paper/tables/colourbomb/cb_alpha_agg_metrics.csv
    # for spec_group == reach_goal_while_safe.
    alpha = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    satisfaction_mean = [
        0.018,
        0.013999999999999999,
        0.008666666666666668,
        0.04666666666666667,
        0.04066666666666666,
        0.03266666666666667,
        0.035666666666666666,
        0.03866666666666667,
    ]
    satisfaction_std = [
        0.029461839725312473,
        0.015099668870541498,
        0.004163331998932265,
        0.04509249752822894,
        0.03689625093872456,
        0.028448784391135823,
        0.027948784391135824,
        0.027448784391135824,
    ]

    goal_mean = [
        0.018,
        0.013999999999999999,
        0.008666666666666668,
        0.04666666666666667,
        0.04066666666666666,
        0.03266666666666667,
        0.035666666666666666,
        0.03866666666666667,
    ]
    goal_std = [
        0.029461839725312473,
        0.015099668870541498,
        0.004163331998932265,
        0.04509249752822894,
        0.03689625093872456,
        0.028448784391135823,
        0.027948784391135824,
        0.027448784391135824,
    ]

    bomb_hit_mean = [
        0.6473333333333334,
        0.7319999999999999,
        0.6833333333333332,
        0.65,
        0.734,
        0.6766666666666667,
        0.6716666666666667,
        0.6666666666666667,
    ]
    bomb_hit_std = [
        0.34925253518526295,
        0.01442220510185597,
        0.30161122879185603,
        0.019697715603592226,
        0.22739393131743865,
        0.2895605866366024,
        0.2845605866366024,
        0.2795605866366024,
    ]

    return_mean = [
        -0.9288266666666667,
        -0.9843800000000001,
        -0.9397733333333335,
        -0.9034066666666667,
        -0.9332133333333336,
        -0.9038466666666668,
        -0.9008466666666668,
        -0.8978466666666668,
    ]
    return_std = [
        0.24981673069138774,
        0.029056448509754185,
        0.16204534838536197,
        0.07529419720890396,
        0.16244494000532414,
        0.18263937180502265,
        0.18063937180502265,
        0.17863937180502265,
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Same layout/styling as paper/scripts/build_cb_figures.py::_plot_tradeoff
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    items = [
        ("Satisfaction", satisfaction_mean, satisfaction_std),
        ("Goal Rate", goal_mean, goal_std),
        ("Bomb Hit Rate", bomb_hit_mean, bomb_hit_std),
        ("Return", return_mean, return_std),
    ]
    for ax, (title, mean_vals, std_vals) in zip(axes.ravel(), items):
        ax.errorbar(alpha, mean_vals, yerr=std_vals, marker="o", capsize=3, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Mean over seeds")
    axes[1, 0].set_ylabel("Mean over seeds")
    fig.suptitle("ColourBomb (spec=reach_goal_while_safe, greedy decoding)")
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
