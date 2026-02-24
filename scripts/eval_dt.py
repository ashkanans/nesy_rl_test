from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dt_model import DecisionTransformerDiscrete
from planning.dt_runtime import (
    DTConstrainedConfig,
    DTRolloutConfig,
    apply_smoke_mode_dt,
    build_dt_offline_source,
    build_knn_suffix_memory,
    compute_default_rtg_target,
    dt_metrics_template,
    evaluate_dt_policy,
    evaluate_random_policy,
    write_metrics_files,
    write_skip_metrics,
)
from planning.eval_runtime import (
    ensure_run_dir,
    set_global_seed,
    spec_label_from_args,
    summarize_dfa_bundle,
    write_json,
)
from scripts.train_dt import get_arg_parser as get_train_dt_arg_parser
from scripts.train_dt import train as train_dt
from train_cb import build_adapter_and_dfa, resolve_formulas


def parse_eval_args():
    parent = get_train_dt_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="Evaluate DT checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--allow_train_fallback",
        action="store_true",
        help="If checkpoint is missing, run a train+eval fallback (non-strict evaluation mode).",
    )
    parser.add_argument("--ltl_formula", type=str, default=None)
    parser.add_argument("--ltl_formulas", type=str, nargs="+", default=None)
    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Named spec preset. Runtime support: cb, nrm_nav, frozenlake.",
    )
    parser.add_argument(
        "--dfa_mode",
        type=str,
        choices=["single", "product", "multi"],
        default="product",
    )
    parser.add_argument("--use_safe_dfa", action="store_true")
    parser.add_argument("--constraint_dims", type=int, nargs="+", default=[0])
    parser.add_argument("--frozenlake_use_position_props", action="store_true")
    parser.add_argument(
        "--dt_mode",
        type=str,
        choices=["greedy", "constrained", "knn"],
        default="greedy",
        help="DT inference mode: greedy, constrained lookahead, or offline kNN suffix planning.",
    )
    parser.add_argument(
        "--num_action_candidates",
        type=int,
        default=4,
        help="Number of candidate actions considered in constrained DT mode.",
    )
    parser.add_argument(
        "--lookahead_horizon",
        type=int,
        default=2,
        help="Lookahead horizon for constrained DT candidate scoring.",
    )
    parser.add_argument(
        "--lookahead_backend",
        type=str,
        choices=["env", "dynamics"],
        default="env",
        help="Lookahead backend for constrained DT; dynamics falls back unless implemented.",
    )
    parser.add_argument(
        "--hard_prune_reject_sink",
        action="store_true",
        help="In constrained mode, prune candidate branches entering DFA reject sink.",
    )
    parser.add_argument(
        "--no_hard_prune_reject_sink",
        action="store_true",
        help="Disable hard-pruning in constrained mode (debug/ablation).",
    )
    parser.add_argument(
        "--sat_rerank_weight",
        type=float,
        default=2.0,
        help="Satisfaction bonus weight used during constrained candidate reranking.",
    )
    parser.add_argument(
        "--candidate_sampling",
        type=str,
        choices=["topk", "sample"],
        default="topk",
        help="Candidate selection strategy in constrained mode.",
    )
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--knn_return_weight", type=float, default=1.0)
    parser.add_argument("--knn_satisfaction_weight", type=float, default=2.0)
    parser.set_defaults(eval_num_episodes=100, no_eval_after_train=True, hard_prune_reject_sink=True)
    return parser


def _load_dt_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in payload:
        raise ValueError("Checkpoint missing model_state_dict")
    cfg = payload.get("config", {})
    required = ["num_states", "num_actions", "context_len", "n_layer", "n_head", "n_embd", "dropout"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Checkpoint config missing keys: {missing}")

    model = DecisionTransformerDiscrete(
        num_states=int(cfg["num_states"]),
        num_actions=int(cfg["num_actions"]),
        context_len=int(cfg["context_len"]),
        n_embd=int(cfg["n_embd"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        dropout=float(cfg["dropout"]),
        max_timestep=int(cfg.get("max_timestep", 4096)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, cfg


def main():
    parser = parse_eval_args()
    args = parser.parse_args()
    args = apply_smoke_mode_dt(args)
    set_global_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir, run_id, ts = ensure_run_dir(args.env, run_dir=args.run_dir, base_dir=args.base_runs_dir)

    t0 = time.time()
    base_dataset, skip_reason = build_dt_offline_source(args)
    if skip_reason is not None:
        write_skip_metrics(
            run_dir=run_dir,
            env=args.env,
            seed=args.seed,
            rtg_target=1.0 if args.rtg_target is None else float(args.rtg_target),
            run_id=run_id,
            ts=ts,
            reason=skip_reason,
        )
        print(f"DT eval skipped: {skip_reason}")
        return

    has_formula_source = (
        args.spec is not None or args.ltl_formula is not None or args.ltl_formulas is not None
    )
    spec_name = spec_label_from_args(args) if has_formula_source else None
    adapter = None
    raw_dfa = None
    formulas = None
    if has_formula_source:
        adapter, _, raw_dfa = build_adapter_and_dfa(args, base_dataset)
        formulas = resolve_formulas(args, dataset=base_dataset)
        dfa_summary = summarize_dfa_bundle(
            raw_dfa,
            spec_name=spec_name,
            formulas=formulas,
            dfa_mode=args.dfa_mode,
        )
    else:
        dfa_summary = {
            "spec": None,
            "formulas": None,
            "dfa_mode": None,
            "note": "No DFA built because no --spec/--ltl_formula(s) were provided.",
        }

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        model, ckpt_cfg = _load_dt_model(args.checkpoint, device=device)
        context_len = int(ckpt_cfg.get("context_len", args.context_len))
        rtg_target = (
            float(args.rtg_target)
            if args.rtg_target is not None
            else float(ckpt_cfg.get("rtg_target", compute_default_rtg_target(base_dataset)))
        )
        if args.env == "frozenlake":
            rtg_target = max(1.0, float(rtg_target))
        checkpoint_path = args.checkpoint
    else:
        if not args.allow_train_fallback:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model, _, train_run_dir = train_dt(args)
        if model is None:
            return
        context_len = int(args.context_len)
        rtg_target = (
            float(args.rtg_target)
            if args.rtg_target is not None
            else compute_default_rtg_target(base_dataset)
        )
        if args.env == "frozenlake":
            rtg_target = max(1.0, float(rtg_target))
        checkpoint_path = os.path.join(train_run_dir, f"dt_state_{max(0, args.epochs - 1)}.pt")

    rollout_cfg = DTRolloutConfig(
        eval_num_episodes=args.eval_num_episodes,
        eval_max_steps=args.eval_max_steps,
        rtg_target=rtg_target,
    )
    constrained_cfg = DTConstrainedConfig(
        dt_mode=args.dt_mode,
        num_action_candidates=args.num_action_candidates,
        lookahead_horizon=args.lookahead_horizon,
        lookahead_backend=args.lookahead_backend,
        hard_prune_reject_sink=bool(args.hard_prune_reject_sink and not args.no_hard_prune_reject_sink),
        sat_rerank_weight=float(args.sat_rerank_weight),
        candidate_sampling=args.candidate_sampling,
        knn_k=int(args.knn_k),
        knn_return_weight=float(args.knn_return_weight),
        knn_satisfaction_weight=float(args.knn_satisfaction_weight),
    )
    knn_memory = None
    if args.dt_mode == "knn":
        knn_memory = build_knn_suffix_memory(base_dataset, env_name=args.env)
    policy_metrics = evaluate_dt_policy(
        model=model,
        env=base_dataset.env,
        env_name=args.env,
        seed=args.seed,
        cfg=rollout_cfg,
        context_len=context_len,
        device=device,
        adapter=adapter,
        raw_dfa=raw_dfa,
        checkpoint_path=checkpoint_path,
        spec_name=spec_name,
        return_rollout_stats=True,
        constrained_cfg=constrained_cfg,
        knn_memory=knn_memory,
    )
    policy_metrics, rollout_stats = policy_metrics
    random_metrics = evaluate_random_policy(
        env=base_dataset.env,
        env_name=args.env,
        seed=args.seed + 10_000,
        eval_num_episodes=args.eval_num_episodes,
        eval_max_steps=args.eval_max_steps,
    )

    metrics = dt_metrics_template(
        env=args.env,
        seed=args.seed,
        rtg_target=rtg_target,
        runtime_sec=time.time() - t0,
        run_id=run_id,
        ts=ts,
        checkpoint_path=checkpoint_path,
    )
    metrics.update(policy_metrics)
    metrics["success_rate"] = metrics.get("goal_rate")
    metrics["context_len"] = int(context_len)
    metrics["spec"] = spec_name
    if args.env == "cb":
        metrics["bomb_hit_rate"] = metrics.get("hazard_hit_rate")

    metrics["random_return_mean"] = random_metrics.get("return_mean")
    metrics["random_goal_rate"] = random_metrics.get("goal_rate")
    metrics["random_violation_rate"] = random_metrics.get("violation_rate")
    metrics["better_than_random"] = bool(
        (metrics.get("goal_rate") is not None)
        and (random_metrics.get("goal_rate") is not None)
        and (float(metrics["goal_rate"]) > float(random_metrics["goal_rate"]))
    )

    write_metrics_files(run_dir, metrics)
    write_json(os.path.join(run_dir, "dfa_summary.json"), dfa_summary)
    write_json(os.path.join(run_dir, "automaton_rollout_stats.json"), rollout_stats)
    write_json(
        os.path.join(run_dir, "dt_eval_summary.json"),
        {
            "policy_metrics": policy_metrics,
            "random_baseline_metrics": random_metrics,
            "acceptance_check": {
                "goal_rate_strictly_better_than_random": metrics["better_than_random"],
                "eval_num_episodes": int(args.eval_num_episodes),
            },
        },
    )
    print(f"Saved DT evaluation artifacts to {run_dir}")


if __name__ == "__main__":
    main()
