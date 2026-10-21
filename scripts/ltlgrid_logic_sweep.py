"""
Multi-task, multi-seed sweep of the ltlgrid symbol logic loss.

For every (data regime, task, seed) it trains a DT once per arm (logic off, or on with a given
grounder / weight) and evaluates it with rollouts in the env. Within a (regime, task, seed) all arms
share the data subset, initialisation seed and evaluation episodes, so arms are compared pairwise.

Results are appended to <out_dir>/results.jsonl (one row per run); an interrupted sweep resumes
where it stopped. Use scripts/ltlgrid_logic_sweep_report.py to summarise.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# arm name -> (logic_alpha, grounder); alpha 0 is the baseline without logic loss
ARMS = {
    "base": (0.0, None),
    "oracle_a1": (1.0, "oracle"),
    "sup_a1": (1.0, "supervised"),
    "sup_a5": (5.0, "supervised"),
    "weak_a1": (1.0, "weak"),
}
# Data regimes are strings: "n<N>" = N random training episodes, "s<K>" = all episodes from K
# random start cells, "x<K>" = only the noise-free optimal episodes from K random start cells
# (evaluation always starts from all 12 cells).
DEFAULT_EPOCHS = {"n15": 200, "n50": 120, "s2": 40, "s4": 25, "x2": 60, "x4": 40, "x12": 25}


def parse_regime(regime):
    """'x2' -> ('x', 2, None); 'x2e15' -> ('x', 2, 15) (optional fixed number of epochs)."""
    m = re.fullmatch(r"([nsx])(\d+)(?:e(\d+))?", regime)
    if m is None:
        raise ValueError(f"unknown regime '{regime}' (use n<N>, s<K> or x<K>, optionally e<epochs>)")
    return m.group(1), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def build_argv(job, args):
    alpha, grounder = ARMS[job["arm"]]
    kind, size, epochs_override = parse_regime(job["regime"])
    epochs = epochs_override or args.epochs.get(f"{kind}{size}", 40)
    h5 = Path(args.data_dir) / f"Results/Symbolic/distance/task{job['task']}/datasets/dataset_medium_expert.h5"
    run_dir = Path(args.out_dir) / "runs" / job["id"]
    argv = [
        "--env", "ltlgrid",
        "--ltlgrid_h5_path", str(h5),
        "--epochs", str(epochs),
        "--context_len", "10",
        "--eval_num_episodes", str(args.eval_episodes),
        "--seed", str(job["seed"]),
        "--run_dir", str(run_dir),
        "--base_runs_dir", str(Path(args.out_dir) / "runs"),
        "--no-save_generated_dataset",
        "--logic_alpha", str(alpha),
        "--logic_rollout_horizon", str(args.horizon),
    ]
    if kind == "n":
        argv += ["--ltlgrid_limit_episodes", "--num_episodes", str(size)]
    elif kind == "s":
        argv += ["--ltlgrid_num_starts", str(size)]
    else:
        argv += ["--ltlgrid_expert_only", "--ltlgrid_num_starts", str(size)]
    if grounder is not None:
        argv += ["--ltlgrid_grounder", grounder]
    return argv, run_dir


def run_job(payload):
    job, argv, run_dir = payload
    sys.path.insert(0, str(REPO_ROOT))
    import torch

    torch.set_num_threads(1)
    row = dict(job)
    t0 = time.time()
    try:
        import contextlib

        from scripts.train_dt import get_arg_parser, train

        os.makedirs(run_dir, exist_ok=True)
        with open(Path(run_dir) / "train.log", "w") as log, contextlib.redirect_stdout(log):
            train(get_arg_parser().parse_args(argv))
        metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
        summary = json.loads((Path(run_dir) / "dt_summary.json").read_text())
        row.update(
            goal_rate=metrics["goal_rate"],
            hazard_rate=metrics["hazard_hit_rate"],
            return_mean=metrics["return_mean"],
            sup_loss=summary["epoch_supervised_action_losses"][-1],
            logic_loss=(summary.get("epoch_logic_losses") or [0.0])[-1],
            dataset_size=metrics.get("dataset_size"),
        )
        for k, v in (summary.get("ltlgrid_grounder") or {}).items():
            if isinstance(v, (int, float)):
                row[f"grounder_{k}"] = v
    except Exception as exc:  # keep the sweep going; the failure is recorded in the row
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_sec"] = time.time() - t0
    return row


def make_jobs(args):
    jobs = []
    for regime in args.regimes:
        for task in args.tasks:
            for seed in args.seeds:
                for arm in args.arms:
                    jobs.append(
                        {
                            "id": f"{regime}_t{task}_s{seed}_{arm}",
                            "regime": regime,
                            "task": task,
                            "seed": seed,
                            "arm": arm,
                        }
                    )
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="directory containing Results/Symbolic/distance/task*/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(1, 11)), help="1-based task numbers")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--regimes", nargs="+", default=["n15", "s2"], help="n<N> random episodes / s<K> start cells / x<K> expert-only start cells")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--epochs", type=str, default=None, help="e.g. 'n15:200,s2:40' (regime:epochs)")
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--horizon", type=int, default=6, help="logic rollout horizon")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--max_jobs", type=int, default=None, help="stop after this many new jobs (for pilots)")
    args = p.parse_args(argv)

    if args.epochs:
        args.epochs = {k: int(v) for k, v in (kv.split(":") for kv in args.epochs.split(","))}
    else:
        args.epochs = dict(DEFAULT_EPOCHS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add(row["id"])

    jobs = [j for j in make_jobs(args) if j["id"] not in done]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    print(f"{len(done)} jobs already done, {len(jobs)} to run on {args.workers} workers", flush=True)
    if not jobs:
        return

    payloads = []
    for job in jobs:
        argv_, run_dir = build_argv(job, args)
        payloads.append((job, argv_, run_dir))

    t0 = time.time()
    finished = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_job, pl) for pl in payloads]
        with open(results_path, "a", encoding="utf-8") as out:
            for fut in as_completed(futures):
                row = fut.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                finished += 1
                elapsed = time.time() - t0
                eta = elapsed / finished * (len(jobs) - finished)
                status = row.get("error") or f"goal={row['goal_rate']:.2f}"
                print(
                    f"[{finished}/{len(jobs)}] {row['id']} {status} "
                    f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
