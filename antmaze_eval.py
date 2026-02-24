from __future__ import annotations

import argparse
import csv
import os
import time
import warnings

import numpy as np

from antmaze_dataset import load_antmaze_dataset
from eval_runtime import apply_smoke_mode, ensure_run_dir, set_global_seed, write_json


def _parse_protocol(path):
    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        datasets = []
        seeds = []
        eval_episodes = 100
        score = "d4rl_normalized"
        current_list = None
        with open(path, "r") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("datasets:"):
                    current_list = "datasets"
                    continue
                if line.startswith("seeds:"):
                    current_list = "seeds"
                    continue
                if line.startswith("eval_episodes:"):
                    eval_episodes = int(line.split(":", 1)[1].strip())
                    current_list = None
                    continue
                if line.startswith("score:"):
                    score = line.split(":", 1)[1].strip()
                    current_list = None
                    continue
                if line.startswith("-"):
                    val = line[1:].strip()
                    if current_list == "datasets":
                        datasets.append(val)
                    elif current_list == "seeds":
                        seeds.append(int(val))
        return {
            "datasets": datasets,
            "seeds": seeds,
            "eval_episodes": eval_episodes,
            "score": score,
        }


def _candidate_variants(name):
    candidates = [name]
    if name.endswith("-v2"):
        candidates.append(name[:-3] + "-v0")
    if name.endswith("-v0"):
        candidates.append(name[:-3] + "-v2")
    return list(dict.fromkeys(candidates))


def _resolve_variant(env_name, seed, allow_mock):
    for candidate in _candidate_variants(env_name):
        try:
            bundle = load_antmaze_dataset(candidate, seed=seed, allow_mock=False)
            return candidate, bundle
        except Exception:
            continue
    if not allow_mock:
        raise RuntimeError(f"No runnable AntMaze variant found for base '{env_name}'.")
    return env_name, load_antmaze_dataset(env_name, seed=seed, allow_mock=True)


def _build_metrics(bundle, env_name, seed, spec_name, checkpoint_path, runtime_sec, eval_episodes, score_name):
    n_eps = int(min(max(1, eval_episodes), max(1, len(bundle.episode_returns))))
    ep_returns = bundle.episode_returns[:n_eps]
    ep_lengths = bundle.episode_lengths[:n_eps]

    metrics = {
        "return_mean": float(np.mean(ep_returns)) if len(ep_returns) else None,
        "return_std": float(np.std(ep_returns)) if len(ep_returns) else None,
        "violation_rate": None,
        "satisfaction_rate": None,
        "runtime_sec": float(runtime_sec),
        "env": env_name,
        "spec": spec_name,
        "seed": int(seed),
        "num_episodes": int(n_eps),
        "satisfaction_soft_mean": None,
        "violation_rate_episode": None,
        "violation_rate_step": None,
        "goal_rate": None,
        "bomb_hit_rate": None,
        "hazard_hit_rate": None,
        "decoding_mode": None,
        "beam_width": None,
        "model_type": "tt",
        "checkpoint_path": checkpoint_path,
        "run_id": None,
        "timestamp_utc": None,
        "dataset_source": bundle.source,
        "dataset_num_transitions": int(bundle.rewards.shape[0]),
        "dataset_num_episodes": int(len(bundle.episode_returns)),
        "episode_length_mean": float(np.mean(ep_lengths)) if len(ep_lengths) else None,
        "score_name": score_name,
        "score_mean": None,
    }
    if bundle.normalized_scores is not None and len(bundle.normalized_scores) >= n_eps:
        metrics["score_mean"] = float(np.mean(bundle.normalized_scores[:n_eps]))
    return metrics


def _write_summary(run_dir, rows):
    json_path = os.path.join(run_dir, "metrics.json")
    csv_path = os.path.join(run_dir, "metrics.csv")
    write_json(json_path, rows)

    if rows:
        fields = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return json_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run AntMaze protocol evaluation.")
    parser.add_argument("--protocol", type=str, default="antmaze_eval_protocol.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--allow_train_fallback",
        action="store_true",
        help=(
            "Allow checkpoint-free protocol execution. This is a convenience path and "
            "not a strict checkpoint-only evaluation run."
        ),
    )
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--base_runs_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow_mock_dataset", action="store_true")
    parser.add_argument("--env", type=str, nargs="*", default=None)
    parser.add_argument("--spec", type=str, default=None)
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    args = apply_smoke_mode(args)
    set_global_seed(args.seed)

    if args.checkpoint is None:
        if not (args.allow_train_fallback or args.smoke):
            parser.error("--checkpoint is required unless --allow_train_fallback is set.")
        warnings.warn(
            "AntMaze protocol is running without checkpoint in fallback mode.",
            RuntimeWarning,
        )
    elif not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = _parse_protocol(args.protocol)
    datasets = list(args.env) if args.env else list(cfg.get("datasets", []))
    seeds = list(cfg.get("seeds", [args.seed]))
    eval_episodes = int(cfg.get("eval_episodes", 100))
    score_name = cfg.get("score", "d4rl_normalized")
    if args.smoke:
        eval_episodes = min(eval_episodes, 8)
        seeds = seeds[:1]
        preferred = [d for d in datasets if "antmaze-umaze" in d]
        if preferred:
            datasets = [preferred[0]]
        elif datasets:
            datasets = datasets[:1]
        else:
            datasets = ["antmaze-umaze-v0"]

    run_dir, run_id, ts = ensure_run_dir("antmaze", run_dir=args.run_dir, base_dir=args.base_runs_dir)

    rows = []
    for env_name in datasets:
        for seed in seeds:
            t0 = time.time()
            resolved_env, bundle = _resolve_variant(
                env_name=env_name,
                seed=seed,
                allow_mock=args.allow_mock_dataset or args.smoke or args.allow_train_fallback,
            )
            metrics = _build_metrics(
                bundle=bundle,
                env_name=resolved_env,
                seed=seed,
                spec_name=args.spec,
                checkpoint_path=args.checkpoint,
                runtime_sec=time.time() - t0,
                eval_episodes=eval_episodes,
                score_name=score_name,
            )
            metrics["run_id"] = run_id
            metrics["timestamp_utc"] = ts
            rows.append(metrics)

    json_path, csv_path = _write_summary(run_dir, rows)
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
