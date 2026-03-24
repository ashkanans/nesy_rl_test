import argparse
import copy
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.eval_runtime import (
    DecodingConfig,
    apply_smoke_mode,
    ensure_run_dir,
    evaluate_policy_rollouts,
    save_evaluation_artifacts,
    set_global_seed,
    spec_label_from_args,
    summarize_dfa_bundle,
)
from train_cb import (
    analyze_dataset,
    build_adapter_and_dfa,
    build_dataset,
    get_arg_parser,
    resolve_formulas,
    save_dataset_artifact,
    train,
)


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
        "goal_rate": None,
        "bomb_hit_rate": None,
        "hazard_hit_rate": None,
        "target_bomb22_hit_rate": None,
        "target_hazard_hit_rate": None,
        "decoding_mode": None,
        "beam_width": None,
        "model_type": "tt",
        "checkpoint_path": None,
        "run_id": None,
        "timestamp_utc": None,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _save_args_snapshot(args, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {k: _to_jsonable(v) for k, v in vars(args).items()}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _run_dataset_overview(args):
    print("=== Dataset overview (pre-run) ===")
    inspect_args = copy.deepcopy(args)
    inspect_args.save_path = args.base_run_dir
    inspect_args.run_dir = args.base_run_dir
    dataset = build_dataset(inspect_args)
    info = save_dataset_artifact(inspect_args, dataset, artifact_tag="dataset_overview")
    if info is not None:
        print(f"Dataset artifact saved to: {info['npz_path']}")
        print(f"Dataset metadata saved to: {info['meta_path']}")
    adapter, _, raw_dfa = build_adapter_and_dfa(inspect_args, dataset)
    analyze_dataset(inspect_args, dataset, adapter, raw_dfa)


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
    formulas = resolve_formulas(cfg, dataset=dataset)
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
    save_evaluation_artifacts(
        cfg.run_dir,
        metrics,
        dfa_summary,
        rollout_stats,
        save_plots=getattr(args, "save_plots", False),
    )
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
    parser.add_argument(
        "--skip_dataset_analysis",
        action="store_true",
        help="Skip pre-run dataset overview analysis and plots.",
    )
    parser.add_argument(
        "--cb_policy_mix_specs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional set of ColourBomb policy mix specs to sweep in one invocation. "
            "When set, each mix runs under <base_run_dir>/mix_<slug>/."
        ),
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
        "goal_rate",
        "bomb_hit_rate",
        "hazard_hit_rate",
        "target_bomb22_hit_rate",
        "target_hazard_hit_rate",
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


def _save_summary_plots(base_dir, results):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    labels = list(results.keys())
    if not labels:
        return

    def _vals(key):
        out = []
        for label in labels:
            val = results[label].get(key)
            out.append(np.nan if val is None else float(val))
        return np.asarray(out, dtype=np.float32)

    width = 0.2
    x = np.arange(len(labels))
    keys = ["goal_rate", "bomb_hit_rate", "satisfaction_rate", "return_mean"]
    vals = [_vals(k) for k in keys]

    plt.figure(figsize=(8, 4))
    for i, (k, v) in enumerate(zip(keys, vals)):
        plt.bar(x + (i - 1.5) * width, np.nan_to_num(v, nan=0.0), width=width, label=k)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_bar.png"))
    plt.close()

    sats = _vals("satisfaction_rate")
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(labels)), np.nan_to_num(sats, nan=0.0), marker="o")
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "satisfaction_trend.png"))
    plt.close()

    rets = _vals("return_mean")
    plt.figure(figsize=(5, 4))
    plt.scatter(np.nan_to_num(rets, nan=0.0), np.nan_to_num(sats, nan=0.0))
    for i, label in enumerate(labels):
        plt.annotate(label, (np.nan_to_num(rets[i], nan=0.0), np.nan_to_num(sats[i], nan=0.0)))
    plt.xlabel("return_mean")
    plt.ylabel("satisfaction_rate")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "return_vs_satisfaction.png"))
    plt.close()


def main():
    parser = parse_baseline_args()
    args = parser.parse_args()
    args = apply_smoke_mode(args)
    if args.evaluate and not getattr(args, "save_plots", False):
        args.save_plots = True
    set_global_seed(args.seed)

    if args.base_run_dir is None:
        base_root, _, _ = ensure_run_dir(args.env, run_dir=None, base_dir=args.base_runs_dir)
    else:
        base_root = args.base_run_dir
    os.makedirs(base_root, exist_ok=True)

    mix_specs = args.cb_policy_mix_specs
    if mix_specs is not None and args.env != "cb":
        raise ValueError("--cb_policy_mix_specs is only supported when --env cb.")
    if mix_specs is None:
        mix_specs = [args.cb_policy_mix_spec]

    all_mix_results = {}
    csv_rows = []

    for mix_idx, mix_spec in enumerate(mix_specs):
        run_args = copy.deepcopy(args)
        run_args.cb_policy_mix_spec = str(mix_spec)
        if len(mix_specs) == 1 and args.cb_policy_mix_specs is None:
            run_args.base_run_dir = base_root
            mix_key = "default"
        else:
            mix_slug = _slug(mix_spec)
            run_args.base_run_dir = os.path.join(base_root, f"mix_{mix_slug}")
            mix_key = mix_spec
        os.makedirs(run_args.base_run_dir, exist_ok=True)

        _save_args_snapshot(run_args, os.path.join(run_args.base_run_dir, "experiment_args.json"))
        if not run_args.skip_dataset_analysis:
            print(f"=== Dataset overview (pre-run) | mix={mix_spec} ===")
            _run_dataset_overview(run_args)

        results = {}
        for name in run_args.baselines:
            if name == "logic" and run_args.alphas:
                for alpha in run_args.alphas:
                    key = f"{name}_alpha{alpha}"
                    print(f"=== Running baseline: {key} | mix={mix_spec} ===")
                    results[key] = run_baseline(
                        name, run_args, alpha_override=alpha, suffix=f"alpha{alpha}"
                    )
            else:
                print(f"=== Running baseline: {name} | mix={mix_spec} ===")
                results[name] = run_baseline(name, run_args)

        json_path, csv_path = write_summary_artifacts(run_args.base_run_dir, results)
        if getattr(run_args, "save_plots", False):
            _save_summary_plots(run_args.base_run_dir, results)
        print(f"Saved summary JSON to {json_path}")
        print(f"Saved summary CSV to {csv_path}")

        all_mix_results[mix_key] = {
            "mix_spec": mix_spec,
            "run_dir": run_args.base_run_dir,
            "results": results,
        }
        for baseline_name, metrics in results.items():
            row = {
                "mix_spec": mix_spec,
                "mix_run_dir": run_args.base_run_dir,
                "baseline": baseline_name,
            }
            row.update(metrics)
            csv_rows.append(row)

    # Aggregate outputs when multiple mixes are requested.
    if len(mix_specs) > 1 or args.cb_policy_mix_specs is not None:
        multi_json = os.path.join(base_root, "baseline_metrics_by_mix.json")
        with open(multi_json, "w") as f:
            json.dump(all_mix_results, f, indent=2)

        multi_csv = os.path.join(base_root, "baseline_metrics_by_mix.csv")
        fields = [
            "mix_spec",
            "mix_run_dir",
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
            "goal_rate",
            "bomb_hit_rate",
            "hazard_hit_rate",
            "target_bomb22_hit_rate",
            "target_hazard_hit_rate",
            "decoding_mode",
            "beam_width",
            "model_type",
            "checkpoint_path",
            "run_id",
            "timestamp_utc",
        ]
        with open(multi_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({k: row.get(k) for k in fields})

        print(f"Saved multi-mix summary JSON to {multi_json}")
        print(f"Saved multi-mix summary CSV to {multi_csv}")


if __name__ == "__main__":
    main()
