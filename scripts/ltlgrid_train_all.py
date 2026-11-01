"""
Train (and evaluate by rollouts) the Decision Transformer on Susanna's symbolic LTL-grid datasets.

Expects the unzipped Dataset_Symbolic.zip: <data_dir>/Dataset/Symbolic/<distance|three_value>/task<N>/
dataset_<medium_expert|medium|medium_replay>.h5. One run per (reward, task, dataset, seed); finished runs
are skipped, so an interrupted batch can simply be restarted. Results go to <out_dir>/results.csv.

Examples:
  python scripts/ltlgrid_train_all.py --data_dir /path/to/data --out_dir runs/ltlgrid_dt
  python scripts/ltlgrid_train_all.py --data_dir ... --out_dir ... --tasks 1 7 --rewards distance \
      --datasets medium_expert --epochs 3 --eval_episodes 24            # quick check
  python scripts/ltlgrid_train_all.py ... --logic_alpha 10 --grounder supervised   # with logic loss (horizon 6)
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_dt import get_arg_parser, train  # noqa: E402

DATASETS = ["medium_expert", "medium", "medium_replay"]
COLUMNS = ["reward", "task", "dataset", "seed", "success_rate", "failure_rate", "return_mean",
           "dataset_size", "final_imitation_loss", "final_logic_loss", "runtime_sec"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rewards", nargs="+", default=["distance", "three_value"])
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--context_len", type=int, default=50, help="50 = a whole episode")
    p.add_argument("--eval_episodes", type=int, default=120, help="rollouts per run (12 start cells x 10)")
    p.add_argument("--logic_alpha", type=float, default=0.0, help="weight of the DFA logic loss (0 = off)")
    p.add_argument("--grounder", choices=["oracle", "supervised", "weak"], default="supervised")
    p.add_argument("--logic_rollout_horizon", type=int, default=6, help="imagined steps of the logic loss")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    if not results.exists():
        with open(results, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

    jobs = [(r, t, d, s) for r in args.rewards for t in args.tasks for d in args.datasets for s in args.seeds]
    for i, (reward, task, dataset, seed) in enumerate(jobs, 1):
        run_dir = out / reward / f"task{task}" / dataset / f"seed{seed}"
        name = f"{reward}/task{task}/{dataset}/seed{seed}"
        if (run_dir / "metrics.json").exists():
            print(f"[{i}/{len(jobs)}] skip {name} (done)", flush=True)
            continue
        h5 = Path(args.data_dir) / "Dataset" / "Symbolic" / reward / f"task{task}" / f"dataset_{dataset}.h5"
        run_args = get_arg_parser().parse_args([
            "--env", "ltlgrid", "--ltlgrid_h5_path", str(h5),
            "--epochs", str(args.epochs), "--context_len", str(args.context_len),
            "--eval_num_episodes", str(args.eval_episodes), "--seed", str(seed),
            "--run_dir", str(run_dir), "--base_runs_dir", str(out / "_tmp"),
            "--no-save_generated_dataset",
            "--logic_alpha", str(args.logic_alpha), "--ltlgrid_grounder", args.grounder,
            "--logic_rollout_horizon", str(args.logic_rollout_horizon),
        ])
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        try:
            train(run_args)
            m = json.loads((run_dir / "metrics.json").read_text())
            s = json.loads((run_dir / "dt_summary.json").read_text())
            row = [reward, task, dataset, seed, m["goal_rate"], m["hazard_hit_rate"], m["return_mean"],
                   m.get("dataset_size"), s["epoch_supervised_action_losses"][-1],
                   (s.get("epoch_logic_losses") or [0.0])[-1], round(time.time() - t0, 1)]
            with open(results, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"    success={m['goal_rate']:.2f} failure={m['hazard_hit_rate']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"    FAILED {name}", flush=True)


if __name__ == "__main__":
    main()
