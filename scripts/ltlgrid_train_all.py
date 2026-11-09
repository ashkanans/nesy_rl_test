"""
Train (and evaluate by rollouts) the Decision Transformer on Susanna's symbolic LTL-grid datasets.

Expects the unzipped Dataset_Symbolic.zip: <data_dir>/Dataset/Symbolic/<distance|three_value>/task<N>/
dataset_<medium_expert|medium|medium_replay>.h5. With --grid 7 it expects the Grid7 "fixed50" archive instead:
<data_dir>/Dataset/<distance|three_value>/task<N>/dataset_<split>_fixed50.h5 (+ test_<split>_fixed50.npz, the 500
fixed test episodes that the rollouts then follow). One run per (reward, task, dataset, seed, alpha); finished
runs are skipped, so an interrupted batch can simply be restarted. Results go to <out_dir>/results.csv.

The logic loss enters as (1 - alpha) * imitation + alpha * logic (--logic_weighting convex, alpha in [0, 1]);
alpha = 0 is the no-logic baseline. --logic_weighting additive gives the older imitation + alpha * logic.

Examples:
  python scripts/ltlgrid_train_all.py --data_dir /path/to/data --out_dir runs/ltlgrid_dt
  python scripts/ltlgrid_train_all.py --data_dir ... --out_dir ... --tasks 1 7 --rewards distance \
      --datasets medium_expert --epochs 3 --eval_episodes 24            # quick check
  python scripts/ltlgrid_train_all.py ... --logic_alphas 0 0.1 0.5 0.9 --seeds 0 1 2 3 4   # alpha sweep
  python scripts/ltlgrid_train_all.py --grid 7 --data_dir ~/Edu/susana_data_7x7 ...          # 7x7 layouts
Figures of a sweep: python scripts/ltlgrid_alpha_figures.py --results <out_dir> ... --out_dir <figures>
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_dt import get_arg_parser, train  # noqa: E402

DATASETS = ["medium_expert", "medium", "medium_replay"]
COLUMNS = ["reward", "task", "dataset", "seed", "alpha", "grounder", "success_rate", "failure_rate", "return_mean",
           "dataset_size", "final_imitation_loss", "final_logic_loss", "runtime_sec",
           "grounder_label_acc", "grounder_reward_replay_acc"]
# results.csv files written before the grounder comparison have these columns (one supervised grounder only)
LEGACY_COLUMNS = ["reward", "task", "dataset", "seed", "alpha", "success_rate", "failure_rate", "return_mean",
                  "dataset_size", "final_imitation_loss", "final_logic_loss", "runtime_sec"]
# the three grounders to compare; label_free is the grounder trained from the rewards only ("weak" in train_dt)
GROUNDERS = {"oracle": "oracle", "supervised": "supervised", "label_free": "weak"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rewards", nargs="+", default=["distance", "three_value"])
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--grid", type=int, choices=[4, 7], default=4, help="4: fixed 4x4 grid; 7: 7x7 layouts (fixed50)")
    p.add_argument("--context_len", type=int, default=None, help="default 50 (4x4: a whole episode), 20 for --grid 7")
    p.add_argument("--eval_episodes", type=int, default=None,
                   help="rollouts per run; default 120 (12 start cells x 10), or the 500 test episodes for --grid 7")
    p.add_argument("--logic_alphas", type=float, nargs="+", default=[0.0],
                   help="weights of the DFA logic loss to run (0 = off, the baseline)")
    p.add_argument("--logic_weighting", choices=["convex", "additive"], default="convex",
                   help="convex: (1-alpha)*imitation + alpha*logic; additive: imitation + alpha*logic")
    p.add_argument("--grounders", nargs="+", choices=list(GROUNDERS), default=None,
                   help="grounders of the logic loss to compare (default: supervised): oracle = exact symbols, "
                        "supervised = trained on the symbol labels, label_free = trained from the rewards only. "
                        "The baseline (alpha 0) has no grounder and runs once.")
    p.add_argument("--grounder", choices=["oracle", "supervised", "weak", "label_free"], default=None,
                   help="a single grounder (kept for older commands; weak = label_free)")
    p.add_argument("--logic_rollout_horizon", type=int, default=6, help="imagined steps of the logic loss")
    args = p.parse_args(argv)
    if args.context_len is None:
        args.context_len = 20 if args.grid == 7 else 50
    if args.eval_episodes is None:
        args.eval_episodes = 500 if args.grid == 7 else 120

    if args.grounders is None:
        args.grounders = [("label_free" if args.grounder == "weak" else args.grounder or "supervised")]

    if args.logic_weighting == "convex" and not all(0.0 <= a <= 1.0 for a in args.logic_alphas):
        p.error("--logic_weighting convex needs every --logic_alphas value in [0, 1]")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    legacy = False
    if not results.exists():
        with open(results, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)
    else:
        header = results.read_text().splitlines()[0].split(",")
        if header == LEGACY_COLUMNS:
            if args.grounders != ["supervised"]:
                p.error(f"{results} is from before the grounder comparison and holds only supervised runs; use a new --out_dir")
            legacy = True  # keep appending in the old format so an interrupted sweep can be resumed
        elif header != COLUMNS:
            p.error(f"{results} has an unknown column layout; use a new --out_dir")

    # alpha is the innermost loop, so the arms that are compared with each other finish together; the
    # baseline (alpha 0, no logic loss) has no grounder and runs once
    jobs = []
    for r in args.rewards:
        for t in args.tasks:
            for d in args.datasets:
                for s in args.seeds:
                    for a in args.logic_alphas:
                        for g in ([None] if a == 0 else args.grounders):
                            jobs.append((r, t, d, s, a, g))
    for i, (reward, task, dataset, seed, alpha, grounder) in enumerate(jobs, 1):
        name = f"{reward}/task{task}/{dataset}/seed{seed}/alpha{alpha:g}" + (f"/{grounder}" if grounder else "")
        arm = out if (grounder is None or legacy) else out / grounder
        run_dir = arm / f"alpha{alpha:g}" / reward / f"task{task}" / dataset / f"seed{seed}"
        if (run_dir / "metrics.json").exists():
            print(f"[{i}/{len(jobs)}] skip {name} (done)", flush=True)
            continue
        if args.grid == 7:
            h5 = Path(args.data_dir) / "Dataset" / reward / f"task{task}" / f"dataset_{dataset}_fixed50.h5"
        else:
            h5 = Path(args.data_dir) / "Dataset" / "Symbolic" / reward / f"task{task}" / f"dataset_{dataset}.h5"
        run_args = get_arg_parser().parse_args([
            "--env", "ltlgrid", "--ltlgrid_grid", str(args.grid),
            "--ltlgrid_h5_path", str(h5),
            "--epochs", str(args.epochs), "--context_len", str(args.context_len),
            "--eval_num_episodes", str(args.eval_episodes), "--seed", str(seed),
            "--run_dir", str(run_dir), "--base_runs_dir", str(out / "_tmp"),
            "--no-save_generated_dataset",
            "--logic_alpha", str(alpha), "--logic_weighting", args.logic_weighting,
            "--ltlgrid_grounder", GROUNDERS[grounder or "supervised"],
            "--logic_rollout_horizon", str(args.logic_rollout_horizon),
        ])
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        try:
            train(run_args)
            m = json.loads((run_dir / "metrics.json").read_text())
            s = json.loads((run_dir / "dt_summary.json").read_text())
            gs = s.get("ltlgrid_grounder") or {}
            row = [reward, task, dataset, seed, alpha, grounder or "none", m["goal_rate"], m["hazard_hit_rate"],
                   m["return_mean"], m.get("dataset_size"), s["epoch_supervised_action_losses"][-1],
                   (s.get("epoch_logic_losses") or [0.0])[-1], round(time.time() - t0, 1),
                   gs.get("dataset_acc", ""), gs.get("reward_replay_episode_acc", "")]
            if legacy:
                row = row[:5] + row[6:13]
            with open(results, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"    success={m['goal_rate']:.2f} failure={m['hazard_hit_rate']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"    FAILED {name}", flush=True)


if __name__ == "__main__":
    main()
