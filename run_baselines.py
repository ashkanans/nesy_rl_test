import argparse
import copy
import json
import os

import torch

from train_cb import (
    get_arg_parser,
    train,
    evaluate_model,
)


def run_baseline(name, args):
    """Train a single baseline and optionally evaluate it."""
    cfg = copy.deepcopy(args)
    if name == "vanilla":
        cfg.alpha = 0.0
    elif name == "logic":
        # use cfg.alpha as provided
        pass
    else:
        raise ValueError(f"Unknown baseline '{name}'")

    cfg.save_path = os.path.join(args.base_save_path, name)

    # Train and keep the objects for evaluation
    model, adapter, dfa, dataset = train(cfg, return_state=True)

    metrics = {}
    if args.evaluate:
        metrics = evaluate_model(
            model, adapter, dfa, dataset, batch_size=args.eval_batch_size
        )
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
        "--eval_batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    return parser


def main():
    parser = parse_baseline_args()
    args = parser.parse_args()

    results = {}
    for name in args.baselines:
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
