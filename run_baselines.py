import argparse
import copy
import csv
import json
import os
import time

from eval_runtime import (
    DecodingConfig,
    apply_smoke_mode,
    ensure_run_dir,
    evaluate_policy_rollouts,
    save_evaluation_artifacts,
    set_global_seed,
    spec_label_from_args,
    summarize_dfa_bundle,
)
from train_cb import get_arg_parser, resolve_formulas, train


def _null_metrics(env, spec, seed):
    return {
        "return_mean": None,
        "return_std": None,
        "violation_rate": None,
        "satisfaction_rate": None,
        "runtime_sec": None,
        "env": env,
        "spec": spec,
        "seed": int(seed),
        "num_episodes": None,
        "satisfaction_soft_mean": None,
        "violation_rate_episode": None,
        "violation_rate_step": None,
        "decoding_mode": None,
        "beam_width": None,
        "model_type": "tt",
        "checkpoint_path": None,
        "run_id": None,
        "timestamp_utc": None,
    }


def run_baseline(name, args, alpha_override=None, suffix=None):
    cfg = copy.deepcopy(args)
    if name == "vanilla":
        cfg.alpha = 0.0
    elif name == "logic":
        pass
    else:
        raise ValueError(f"Unknown baseline '{name}'")

    if alpha_override is not None:
        cfg.alpha = alpha_override

    tag = name if suffix is None else f"{name}_{suffix}"
    cfg.run_dir = os.path.join(args.base_run_dir, tag)
    cfg.save_path = cfg.run_dir

    train_t0 = time.time()
    model, adapter, _, dataset, raw_dfa = train(cfg, return_state=True)

    spec_name = spec_label_from_args(cfg)
    formulas = resolve_formulas(cfg)
    dfa_summary = summarize_dfa_bundle(
        raw_dfa, spec_name=spec_name, formulas=formulas, dfa_mode=cfg.dfa_mode
    )

    if args.evaluate:
        decoding_cfg = DecodingConfig(
            mode=args.decoding_mode,
            beam_width=args.beam_width,
            plan_horizon=args.plan_horizon,
            sat_rerank_weight=args.sat_rerank_weight,
            hard_prune_reject_sink=args.hard_prune_reject_sink,
        )
        metrics, rollout_stats = evaluate_policy_rollouts(
            model=model,
            adapter=adapter,
            raw_dfa=raw_dfa,
            dataset=dataset,
            env_name=cfg.env,
            spec_name=spec_name,
            seed=cfg.seed,
            checkpoint_path=os.path.join(cfg.run_dir, f"cb_state_{cfg.epochs - 1}.pt"),
            num_episodes=args.eval_num_episodes,
            max_steps=args.eval_max_steps,
            decoding_cfg=decoding_cfg,
        )
    else:
        metrics = _null_metrics(cfg.env, spec_name, cfg.seed)
        rollout_stats = {"skipped": True, "reason": "evaluate=false"}

    _, run_id, ts = ensure_run_dir(cfg.env, run_dir=cfg.run_dir, base_dir=args.base_runs_dir)
    metrics["runtime_sec"] = float(time.time() - train_t0)
    metrics["run_id"] = run_id
    metrics["timestamp_utc"] = ts
    save_evaluation_artifacts(cfg.run_dir, metrics, dfa_summary, rollout_stats)
    return metrics


def parse_baseline_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="Run TT baselines")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["vanilla", "logic"],
        help="Which baselines to run",
    )
    parser.add_argument(
        "--base_run_dir",
        type=str,
        default=None,
        help="Root directory for baseline sweep outputs. Default: runs/<env>/<timestamp>/",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training each baseline",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.4],
        help="Logic loss weights to sweep for the logic baseline",
    )
    return parser


def write_summary_artifacts(base_dir, results):
    os.makedirs(base_dir, exist_ok=True)
    json_path = os.path.join(base_dir, "baseline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(base_dir, "baseline_metrics.csv")
    fields = [
        "baseline",
        "return_mean",
        "return_std",
        "violation_rate",
        "satisfaction_rate",
        "runtime_sec",
        "env",
        "spec",
        "seed",
        "num_episodes",
        "satisfaction_soft_mean",
        "violation_rate_episode",
        "violation_rate_step",
        "decoding_mode",
        "beam_width",
        "model_type",
        "checkpoint_path",
        "run_id",
        "timestamp_utc",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for baseline_name, metrics in results.items():
            row = {"baseline": baseline_name}
            row.update({k: metrics.get(k) for k in fields if k != "baseline"})
            writer.writerow(row)

    return json_path, csv_path


def main():
    parser = parse_baseline_args()
    args = parser.parse_args()
    args = apply_smoke_mode(args)
    set_global_seed(args.seed)

    if args.base_run_dir is None:
        args.base_run_dir, _, _ = ensure_run_dir(
            args.env, run_dir=None, base_dir=args.base_runs_dir
        )
    else:
        os.makedirs(args.base_run_dir, exist_ok=True)

    results = {}
    for name in args.baselines:
        if name == "logic" and args.alphas:
            for alpha in args.alphas:
                key = f"{name}_alpha{alpha}"
                print(f"=== Running baseline: {key} ===")
                results[key] = run_baseline(name, args, alpha_override=alpha, suffix=f"alpha{alpha}")
        else:
            print(f"=== Running baseline: {name} ===")
            results[name] = run_baseline(name, args)

    json_path, csv_path = write_summary_artifacts(args.base_run_dir, results)
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
