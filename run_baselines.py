import argparse
import copy
import json
import os

import torch

from train_cb import (
    get_arg_parser,
    train,
    evaluate_model,
    rollout_nrm_nav_policy,
)


def run_baseline(name, args, alpha_override=None, suffix=None):
    """Train a single baseline and optionally evaluate it."""
    cfg = copy.deepcopy(args)
    if name == "vanilla":
        cfg.alpha = 0.0
    elif name == "logic":
        # use cfg.alpha as provided
        pass
    else:
        raise ValueError(f"Unknown baseline '{name}'")

    if alpha_override is not None:
        cfg.alpha = alpha_override
    tag = name if suffix is None else f"{name}_{suffix}"
    cfg.save_path = os.path.join(args.base_save_path, tag)

    # Train and keep the objects for evaluation
    model, adapter, deep_dfa, dataset, raw_dfa = train(cfg, return_state=True)

    metrics = {}
    if args.evaluate:
        metrics = evaluate_model(
            model,
            adapter,
            raw_dfa,
            dataset,
            batch_size=args.eval_batch_size,
            append_end_token=getattr(args, "append_end_token_to_dfa", False),
        )
        # optional rollout evaluation for nrm_nav
        if args.eval_rollouts and args.env == "nrm_nav":
            rollout_metrics = rollout_nrm_nav_policy(
                model,
                adapter,
                raw_dfa,
                dataset.env.cfg,
                num_episodes=args.rollout_episodes,
                max_steps=args.rollout_max_steps,
                greedy=True,
                append_end_token=getattr(args, "append_end_token_to_dfa", False),
            )
            # prefix rollout metrics to avoid collisions
            for k, v in rollout_metrics.items():
                metrics[f"rollout_{k}"] = v

        os.makedirs(cfg.save_path, exist_ok=True)
        metrics_path = os.path.join(cfg.save_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def parse_baseline_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="Run baselines")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["vanilla", "logic"],
        help="Which baselines to run",
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="cb_runs",
        help="Root directory to save checkpoints/metrics",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--eval_rollouts",
        action="store_true",
        help="For nrm_nav: also evaluate policy rollouts in the environment",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.4],
        help="Logic loss weights to sweep for logic baseline",
    )
    parser.add_argument(
        "--rollout_episodes",
        type=int,
        default=100,
        help="Number of rollout episodes per model for nrm_nav policy evaluation",
    )
    parser.add_argument(
        "--rollout_max_steps",
        type=int,
        default=None,
        help="Maximum steps per rollout episode (defaults to env max_steps)",
    )
    return parser


def main():
    parser = parse_baseline_args()
    args = parser.parse_args()

    results = {}
    for name in args.baselines:
        if name == "logic" and args.alphas:
            for a in args.alphas:
                print(f"=== Running baseline: {name} alpha={a} ===")
                metrics = run_baseline(name, args, alpha_override=a, suffix=f"alpha{a}")
                results[f"{name}_alpha{a}"] = metrics
        else:
            print(f"=== Running baseline: {name} ===")
            metrics = run_baseline(name, args)
            results[name] = metrics

    if results:
        summary_path = os.path.join(args.base_save_path, "baseline_metrics.json")
        os.makedirs(args.base_save_path, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
