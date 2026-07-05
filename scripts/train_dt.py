from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dt_model import DecisionTransformerDiscrete
from planning.dynamics_runtime import (
    build_dataset_tabular_dynamics,
    build_dynamics_dataset_identifiers,
    fit_neural_dynamics_model,
    load_neural_dynamics_checkpoint,
    neural_dynamics_to_transition_tensor,
    save_neural_dynamics_checkpoint,
)
from planning.dt_runtime import (
    DTRolloutConfig,
    apply_smoke_mode_dt,
    build_dt_dataset,
    build_tabular_dynamics,
    build_dt_offline_source,
    compute_dt_dfa_rollout_loss,
    compute_dt_logic_rollout_penalty,
    compute_default_rtg_target,
    dt_metrics_template,
    evaluate_dt_policy,
    hazard_mask_for_env,
    save_dt_dataset_artifact,
    write_metrics_files,
    write_skip_metrics,
)
from planning.eval_runtime import ensure_run_dir, set_global_seed, write_json
from planning.product_value import (
    ProductValueConfig,
    build_dfa_prefix_state_ids,
    build_product_value_table,
    compute_dt_dfa_product_value_loss,
    default_product_value_cache_path,
    load_product_value_table,
    save_product_value_table,
    write_offline_vs_oracle_comparison,
    write_product_value_validation,
)


def get_arg_parser(add_help=True):
    p = argparse.ArgumentParser(
        add_help=add_help,
        description="Train Decision Transformer baseline (discrete envs).",
    )
    p.add_argument(
        "--env",
        type=str,
        choices=["cb", "frozenlake", "dsrl", "antmaze"],
        default="frozenlake",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")

    p.add_argument("--num_episodes", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--context_len", type=int, default=20)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--logic_alpha", type=float, default=0.0)
    p.add_argument(
        "--dt_logic_loss_type",
        type=str,
        choices=["auto", "hazard", "dfa", "dfa_product_value"],
        default="auto",
        help=(
            "DT logic loss: hazard keeps the legacy unsafe-state penalty; "
            "dfa uses the actual LTLf/DFA formula when --spec/--ltl_formula(s) is provided; "
            "auto selects dfa when a formula source is present, otherwise hazard."
        ),
    )
    p.add_argument("--logic_rollout_horizon", type=int, default=2)
    p.add_argument("--logic_temperature", type=float, default=1.0)
    p.add_argument("--product_value_max_iter", type=int, default=10000)
    p.add_argument("--product_value_tol", type=float, default=1e-6)
    p.add_argument("--product_value_dmax", type=float, default=None)
    p.add_argument("--product_value_backup", type=str, choices=["hard", "soft"], default="hard")
    p.add_argument("--product_value_gamma", type=float, default=1.0)
    p.add_argument(
        "--product_value_zero_support",
        type=str,
        choices=["pessimistic", "self_loop"],
        default="pessimistic",
    )
    p.add_argument("--product_value_support_penalty", type=float, default=0.0)
    p.add_argument("--product_value_soft_tau", type=float, default=1.0)
    p.add_argument("--product_value_cache_path", type=str, default=None)
    p.add_argument(
        "--auto_product_value_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build/load cached product-value Phi tables next to dataset artifacts.",
    )
    p.add_argument(
        "--dt_logic_dynamics_backend",
        type=str,
        choices=["tabular_env", "tabular_dataset", "neural_dataset"],
        default="tabular_env",
    )
    p.add_argument("--dynamics_epochs", type=int, default=20)
    p.add_argument("--dynamics_batch_size", type=int, default=256)
    p.add_argument("--dynamics_lr", type=float, default=1e-3)
    p.add_argument("--dynamics_hidden_dim", type=int, default=128)
    p.add_argument("--dynamics_layers", type=int, default=2)
    p.add_argument("--dynamics_weight_decay", type=float, default=1e-4)
    p.add_argument("--dynamics_val_fraction", type=float, default=0.1)
    p.add_argument(
        "--dynamics_freeze_after_fit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--dynamics_checkpoint_path", type=str, default=None)
    p.add_argument(
        "--auto_dynamics_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dataset artifacts, auto-detect <artifact folder>/dynamics/neural_dataset.pt.",
    )
    p.add_argument(
        "--fit_missing_dynamics",
        action="store_true",
        help="Train neural dynamics if the requested/auto checkpoint is missing.",
    )
    p.add_argument(
        "--save_dynamics_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--dynamics_max_transition_entries", type=int, default=10000000)
    p.add_argument("--dynamics_temperature", type=float, default=1.0)

    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--rtg_target", type=float, default=None)
    p.add_argument("--no_eval_after_train", action="store_true")
    p.add_argument("--eval_num_episodes", type=int, default=32)
    p.add_argument("--eval_max_steps", type=int, default=None)

    p.add_argument("--stochastic", action="store_true", help="Enable stochastic transitions for cb dataset.")
    p.add_argument("--cb_policy_mix_spec", type=str, default="random:1.0")
    p.add_argument("--cb_policy_mix_sampling", type=str, choices=["fixed", "normal"], default="fixed")
    p.add_argument("--cb_policy_mix_normal_spec", type=str, default=None)
    p.add_argument(
        "--cb_policy_mix_normal_mean_mode",
        type=str,
        choices=["base", "absolute", "delta"],
        default="base",
    )
    p.add_argument("--cb_state_semantics", type=str, choices=["pre", "post"], default="post")
    p.add_argument("--cb_longest_path_max_expansions", type=int, default=500000)
    p.add_argument("--frozenlake_map_size", type=str, choices=["4x4", "8x8"], default="4x4")
    p.add_argument("--frozenlake_is_slippery", action="store_true")
    p.add_argument("--policy_mix", type=float, default=0.0)
    p.add_argument("--dsrl_dataset_path", type=str, default=None)
    p.add_argument("--dsrl_dataset_key", type=str, default="PointGoal1")
    p.add_argument("--dsrl_state_bins", type=int, default=128)
    p.add_argument("--dsrl_action_bins", type=int, default=16)
    p.add_argument("--dsrl_reward_goal_threshold", type=float, default=0.0)
    p.add_argument("--dsrl_cost_unsafe_threshold", type=float, default=0.0)
    p.add_argument("--dsrl_cost_unsafe_quantile", type=float, default=None)
    p.add_argument("--dsrl_download", action="store_true")

    p.add_argument("--ltl_formula", type=str, default=None)
    p.add_argument("--ltl_formulas", type=str, nargs="+", default=None)
    p.add_argument("--spec", type=str, default=None)
    p.add_argument("--dfa_mode", type=str, choices=["single", "product", "multi"], default="product")
    p.add_argument("--use_safe_dfa", action="store_true")
    p.add_argument("--constraint_dims", type=int, nargs="+", default=[0])
    p.add_argument("--frozenlake_use_position_props", action="store_true")
    p.add_argument("--dfa_backend", type=str, choices=["auto", "ltlf", "template"], default="auto")

    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--base_runs_dir", type=str, default="runs")
    p.add_argument(
        "--save_generated_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Persist generated offline dataset to disk as NPZ + metadata JSON "
            "(default: enabled)."
        ),
    )
    p.add_argument(
        "--dataset_artifact_dir",
        type=str,
        default=None,
        help=(
            "Directory for dataset artifacts. Default: <run_dir>/dataset_artifacts. "
            "Use this to keep a central reusable dataset cache."
        ),
    )
    p.add_argument(
        "--dataset_artifact_name",
        type=str,
        default="dataset_snapshot",
        help=(
            "Base filename stem for dataset artifacts. "
            "Files written: <stem>.npz and <stem>.meta.json."
        ),
    )
    p.add_argument(
        "--dataset_artifact_path",
        type=str,
        default=None,
        help=(
            "Path to a prebuilt dataset artifact (.npz, .meta.json, or stem). "
            "If provided, training loads the dataset from disk instead of generating it."
        ),
    )
    return p


def _has_formula_source(args) -> bool:
    return bool(
        getattr(args, "spec", None) is not None
        or getattr(args, "ltl_formula", None) is not None
        or getattr(args, "ltl_formulas", None) is not None
    )


def _resolve_dt_logic_loss_type(args) -> str:
    requested = str(getattr(args, "dt_logic_loss_type", "auto"))
    if requested == "auto":
        return "dfa" if _has_formula_source(args) else "hazard"
    if requested in {"dfa", "dfa_product_value"} and not _has_formula_source(args):
        raise ValueError(
            f"--dt_logic_loss_type {requested} requires --spec, --ltl_formula, or --ltl_formulas."
        )
    return requested


def _to_device(batch, device):
    return [x.to(device) for x in batch]


def _dataset_dynamics_checkpoint_path(dataset_artifact_path: str | None) -> str | None:
    if not dataset_artifact_path:
        return None
    p = os.path.abspath(str(dataset_artifact_path))
    if p.endswith(".meta.json"):
        stem = p[: -len(".meta.json")]
    elif p.endswith(".npz"):
        stem = p[: -len(".npz")]
    else:
        stem = p
    return os.path.join(os.path.dirname(stem), "dynamics", "neural_dataset.pt")


def _support_counts_from_stats(stats, num_actions: int, num_states: int):
    counts = None if stats is None else stats.get("state_action_counts")
    if counts is None:
        return None
    arr = torch.as_tensor(counts, dtype=torch.float32)
    if tuple(arr.shape) != (int(num_actions), int(num_states)):
        return None
    return arr


def train(args):
    args = apply_smoke_mode_dt(args)
    set_global_seed(args.seed)
    run_dir, run_id, ts = ensure_run_dir(args.env, run_dir=args.run_dir, base_dir=args.base_runs_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        print(f"DT training skipped: {skip_reason}")
        return None, None, run_dir

    dataset_artifact = save_dt_dataset_artifact(args, base_dataset, artifact_tag="dataset_train")
    if dataset_artifact is not None:
        print(f"Dataset artifact saved: {dataset_artifact['npz_path']}")
        print(f"Dataset metadata saved: {dataset_artifact['meta_path']}")

    num_states = int(base_dataset.env.observation_space.n)
    num_actions = int(base_dataset.env.action_space.n)
    transition_probs_t = None
    hazard_mask_t = None
    dt_logic_loss_type_effective = "none"
    dfa_adapter = None
    dfa_deep = None
    raw_dfa = None
    dfa_formulas = None
    dynamics_stats = None
    dynamics_model_path = None
    product_value_table = None
    product_value_path = None
    dfa_state_ids = None
    if float(args.logic_alpha) > 0.0:
        dt_logic_loss_type_effective = _resolve_dt_logic_loss_type(args)
        if dt_logic_loss_type_effective == "hazard":
            hazard = hazard_mask_for_env(args.env, base_dataset.env)
            hazard_mask_t = torch.from_numpy(hazard).float().to(device)
        elif dt_logic_loss_type_effective in {"dfa", "dfa_product_value"}:
            from train_cb import build_adapter_and_dfa, resolve_formulas

            dfa_adapter, dfa_deep, raw_dfa = build_adapter_and_dfa(args, base_dataset)
            dfa_formulas = resolve_formulas(args, dataset=base_dataset)
            if isinstance(dfa_deep, (list, tuple)):
                dfa_deep = [d.to(device) for d in dfa_deep]
            else:
                dfa_deep = dfa_deep.to(device)
        else:
            raise ValueError(f"Unsupported DT logic loss type '{dt_logic_loss_type_effective}'")

        if args.dt_logic_dynamics_backend == "tabular_env":
            dyn = build_tabular_dynamics(base_dataset)
            transition_probs_t = torch.from_numpy(dyn).float().to(device)
            dynamics_stats = {"backend": "tabular_env", "pure_offline": False}
        elif args.dt_logic_dynamics_backend == "tabular_dataset":
            dyn, dynamics_stats = build_dataset_tabular_dynamics(base_dataset)
            transition_probs_t = torch.from_numpy(dyn).float().to(device)
            dynamics_stats["backend"] = "tabular_dataset"
            dynamics_stats["pure_offline"] = True
        elif args.dt_logic_dynamics_backend == "neural_dataset":
            dataset_ids = build_dynamics_dataset_identifiers(base_dataset, args=args)
            requested_ckpt = str(args.dynamics_checkpoint_path).strip() if args.dynamics_checkpoint_path else None
            if not requested_ckpt and bool(getattr(args, "auto_dynamics_checkpoint", True)):
                requested_ckpt = _dataset_dynamics_checkpoint_path(getattr(args, "dataset_artifact_path", None))
            save_ckpt = bool(args.save_dynamics_checkpoint)
            if requested_ckpt:
                dynamics_model_path = requested_ckpt
            elif save_ckpt:
                dynamics_model_path = os.path.join(run_dir, "dynamics_model.pt")
            dynamics_log_path = None
            if dynamics_model_path is not None:
                dynamics_log_path = os.path.join(
                    os.path.dirname(dynamics_model_path), "dynamics_training_log.csv"
                )
            elif save_ckpt:
                dynamics_log_path = os.path.join(run_dir, "dynamics_training_log.csv")

            if dynamics_model_path is not None and os.path.exists(dynamics_model_path):
                dynamics_model, dynamics_stats = load_neural_dynamics_checkpoint(
                    path=dynamics_model_path,
                    device=device,
                    expected_num_states=num_states,
                    expected_num_actions=num_actions,
                    expected_hidden_dim=int(args.dynamics_hidden_dim),
                    expected_num_layers=int(args.dynamics_layers),
                    freeze_after_load=bool(args.dynamics_freeze_after_fit),
                )
            else:
                if requested_ckpt and not bool(getattr(args, "fit_missing_dynamics", False)):
                    raise FileNotFoundError(
                        "Neural dynamics checkpoint is required but missing: "
                        f"{requested_ckpt}. Pass --fit_missing_dynamics to train it, "
                        "or materialize it with scripts/materialize_cb_datasets.py --fit_neural_dynamics."
                    )
                dynamics_model, dynamics_stats = fit_neural_dynamics_model(
                    base_dataset=base_dataset,
                    hidden_dim=int(args.dynamics_hidden_dim),
                    num_layers=int(args.dynamics_layers),
                    epochs=int(args.dynamics_epochs),
                    batch_size=int(args.dynamics_batch_size),
                    lr=float(args.dynamics_lr),
                    weight_decay=float(args.dynamics_weight_decay),
                    val_fraction=float(args.dynamics_val_fraction),
                    device=device,
                    seed=int(args.seed),
                    temperature=float(args.dynamics_temperature),
                    freeze_after_fit=bool(args.dynamics_freeze_after_fit),
                    log_path=dynamics_log_path,
                )
                dynamics_stats["dataset_identifiers"] = dataset_ids
                if dynamics_model_path is not None:
                    save_neural_dynamics_checkpoint(
                        path=dynamics_model_path,
                        model=dynamics_model,
                        stats=dynamics_stats,
                        dataset_identifiers=dataset_ids,
                    )
            transition_probs_t = neural_dynamics_to_transition_tensor(
                model=dynamics_model,
                num_states=num_states,
                num_actions=num_actions,
                device=device,
                temperature=float(args.dynamics_temperature),
                max_entries=int(args.dynamics_max_transition_entries),
            ).detach()
            dynamics_stats["backend"] = "neural_dataset"
            dynamics_stats["pure_offline"] = True
            dynamics_stats["checkpoint_path"] = dynamics_model_path
            dynamics_stats.setdefault("dataset_identifiers", dataset_ids)
        else:
            raise ValueError(f"Unsupported DT dynamics backend '{args.dt_logic_dynamics_backend}'")

        if dt_logic_loss_type_effective == "dfa_product_value":
            if dfa_adapter is None or dfa_deep is None or transition_probs_t is None:
                raise RuntimeError("dfa_product_value requires DFA components and transition dynamics.")
            if abs(float(getattr(args, "product_value_gamma", 1.0)) - 1.0) > 1e-9:
                print(
                    "[warn] --product_value_gamma != 1.0 changes the scale/centering of "
                    "the product-value shaping term. Phi is still undiscounted SSP; this "
                    "does not reproduce discounted-VI self-cancellation."
                )
            spec_label = str(getattr(args, "spec", None) or "custom_formula")
            product_value_path = str(getattr(args, "product_value_cache_path", "") or "")
            if not product_value_path and bool(getattr(args, "auto_product_value_cache", True)):
                product_value_path = default_product_value_cache_path(
                    getattr(args, "dataset_artifact_path", None),
                    run_dir,
                    str(args.dt_logic_dynamics_backend),
                    spec_label,
                )
            if product_value_path and os.path.exists(product_value_path):
                product_value_table = load_product_value_table(product_value_path)
            else:
                support_counts_np = None
                support_counts_t = _support_counts_from_stats(dynamics_stats, num_actions, num_states)
                if support_counts_t is not None:
                    support_counts_np = support_counts_t.cpu().numpy()
                product_value_table = build_product_value_table(
                    transition_probs=transition_probs_t.detach().cpu().numpy(),
                    adapter=dfa_adapter,
                    raw_dfa=raw_dfa,
                    support_counts=support_counts_np,
                    config=ProductValueConfig(
                        max_iter=int(args.product_value_max_iter),
                        tol=float(args.product_value_tol),
                        dmax=getattr(args, "product_value_dmax", None),
                        backup=str(args.product_value_backup),
                        gamma=float(args.product_value_gamma),
                        zero_support=str(args.product_value_zero_support),
                        support_penalty=float(args.product_value_support_penalty),
                        soft_tau=float(args.product_value_soft_tau),
                    ),
                    metadata={
                        "env": str(args.env),
                        "spec": spec_label,
                        "backend": str(args.dt_logic_dynamics_backend),
                        "dataset_artifact_path": getattr(args, "dataset_artifact_path", None),
                        "dynamics_checkpoint_path": dynamics_model_path,
                    },
                )
                if product_value_path:
                    save_product_value_table(product_value_path, product_value_table)
            validation_dir = os.path.dirname(product_value_path) if product_value_path else os.path.join(
                run_dir,
                "product_value",
                str(args.dt_logic_dynamics_backend),
                spec_label,
            )
            start_state = None
            grid_shape = None
            env = getattr(base_dataset, "env", None)
            if env is not None:
                if hasattr(env, "start_pos") and hasattr(env, "_pos_to_state"):
                    start_state = int(env._pos_to_state(env.start_pos))
                if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
                    grid_shape = (int(env.n_rows), int(env.n_cols))
            write_product_value_validation(
                product_value_table,
                validation_dir,
                transition_probs=transition_probs_t.detach().cpu().numpy(),
                start_state=start_state,
                grid_shape=grid_shape,
                strict=True,
            )
            if (
                product_value_path
                and str(args.dt_logic_dynamics_backend) != "tabular_env"
                and raw_dfa is not None
            ):
                comparison_path = os.path.join(os.path.dirname(product_value_path), "offline_vs_oracle.json")
                if os.path.exists(comparison_path):
                    oracle_needed = False
                else:
                    oracle_needed = True
            else:
                oracle_needed = False
            if oracle_needed:
                try:
                    oracle_table = build_product_value_table(
                        transition_probs=build_tabular_dynamics(base_dataset),
                        adapter=dfa_adapter,
                        raw_dfa=raw_dfa,
                        support_counts=None,
                        config=ProductValueConfig(
                            max_iter=int(args.product_value_max_iter),
                            tol=float(args.product_value_tol),
                            dmax=getattr(args, "product_value_dmax", None),
                            backup=str(args.product_value_backup),
                            gamma=float(args.product_value_gamma),
                            zero_support="self_loop",
                            support_penalty=0.0,
                            soft_tau=float(args.product_value_soft_tau),
                        ),
                        metadata={
                            "env": str(args.env),
                            "spec": spec_label,
                            "backend": "tabular_env",
                            "diagnostic_only": True,
                        },
                    )
                    write_offline_vs_oracle_comparison(
                        offline_table=product_value_table,
                        oracle_table=oracle_table,
                        out_dir=os.path.dirname(product_value_path),
                    )
                except Exception as exc:
                    print(f"[warn] product-value oracle comparison skipped: {exc}")
            dfa_state_ids = build_dfa_prefix_state_ids(base_dataset, dfa_adapter, raw_dfa)

    dt_dataset = build_dt_dataset(
        base_dataset,
        context_len=args.context_len,
        dfa_state_ids=dfa_state_ids,
    )
    if len(dt_dataset) == 0:
        raise RuntimeError("DT dataset is empty; cannot train.")

    dataloader = DataLoader(dt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = DecisionTransformerDiscrete(
        num_states=num_states,
        num_actions=num_actions,
        context_len=args.context_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        max_timestep=max(512, args.max_steps + 1),
    ).to(device)
    max_timestep = int(model.max_timestep)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_losses = []
    epoch_supervised_losses = []
    epoch_logic_losses = []
    product_phi_t = None
    product_next_q_t = None
    product_support_counts_t = None
    if product_value_table is not None:
        product_phi_t = torch.from_numpy(product_value_table.phi).float().to(device)
        product_next_q_t = torch.from_numpy(product_value_table.next_q).long().to(device)
        if product_value_table.support_counts is not None:
            product_support_counts_t = torch.from_numpy(product_value_table.support_counts).float().to(device)
    model.train()
    for epoch in range(args.epochs):
        running_total = 0.0
        running_sup = 0.0
        running_logic = 0.0
        count = 0
        for batch in dataloader:
            batch = _to_device(batch, device)
            if len(batch) == 6:
                states, prev_actions, rtg, timesteps, targets, mask = batch
                dfa_state_ids_t = None
            elif len(batch) == 7:
                states, prev_actions, rtg, timesteps, targets, mask, dfa_state_ids_t = batch
            else:
                raise ValueError(f"Unexpected DT batch size {len(batch)}.")
            logits = model(states, prev_actions, rtg, timesteps, attention_mask=mask)
            sup_loss = F.cross_entropy(
                logits.reshape(-1, num_actions),
                targets.reshape(-1),
                ignore_index=-100,
            )
            logic_loss = logits.new_zeros(())
            if (
                dt_logic_loss_type_effective == "hazard"
                and transition_probs_t is not None
                and hazard_mask_t is not None
            ):
                logic_loss = compute_dt_logic_rollout_penalty(
                    logits=logits,
                    states=states,
                    attention_mask=mask,
                    transition_probs=transition_probs_t,
                    hazard_mask=hazard_mask_t,
                    rollout_horizon=int(args.logic_rollout_horizon),
                    temperature=float(args.logic_temperature),
                )
                if not torch.isfinite(logic_loss):
                    logic_loss = logits.new_zeros(())
            elif (
                dt_logic_loss_type_effective == "dfa"
                and transition_probs_t is not None
                and dfa_adapter is not None
                and dfa_deep is not None
            ):
                logic_loss = compute_dt_dfa_rollout_loss(
                    logits=logits,
                    states=states,
                    attention_mask=mask,
                    transition_probs=transition_probs_t,
                    adapter=dfa_adapter,
                    deep_dfa=dfa_deep,
                    rollout_horizon=int(args.logic_rollout_horizon),
                    temperature=float(args.logic_temperature),
                )
                if not torch.isfinite(logic_loss):
                    logic_loss = logits.new_zeros(())
            elif (
                dt_logic_loss_type_effective == "dfa_product_value"
                and transition_probs_t is not None
                and product_phi_t is not None
                and product_next_q_t is not None
                and dfa_state_ids_t is not None
            ):
                logic_loss = compute_dt_dfa_product_value_loss(
                    logits=logits,
                    states=states,
                    dfa_state_ids=dfa_state_ids_t,
                    attention_mask=mask,
                    transition_probs=transition_probs_t,
                    phi=product_phi_t,
                    next_q=product_next_q_t,
                    support_counts=product_support_counts_t,
                    support_penalty=float(args.product_value_support_penalty),
                    gamma=float(args.product_value_gamma),
                    temperature=float(args.logic_temperature),
                )
                if not torch.isfinite(logic_loss):
                    logic_loss = logits.new_zeros(())
            loss = sup_loss + float(args.logic_alpha) * logic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_total += float(loss.item())
            running_sup += float(sup_loss.item())
            running_logic += float(logic_loss.item()) if torch.isfinite(logic_loss) else 0.0
            count += 1
        mean_total = running_total / max(1, count)
        mean_sup = running_sup / max(1, count)
        mean_logic = running_logic / max(1, count)
        epoch_losses.append(mean_total)
        epoch_supervised_losses.append(mean_sup)
        epoch_logic_losses.append(mean_logic)
        print(
            f"epoch {epoch} | total_loss {mean_total:.4f} | "
            f"supervised_action_loss {mean_sup:.4f} | logic_loss {mean_logic:.4f}"
        )

    rtg_target = (
        float(args.rtg_target)
        if args.rtg_target is not None
        else compute_default_rtg_target(base_dataset)
    )
    if args.env == "frozenlake":
        rtg_target = max(1.0, float(rtg_target))

    ckpt_path = os.path.join(run_dir, f"dt_state_{max(0, args.epochs - 1)}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "env": args.env,
                "seed": args.seed,
                "context_len": args.context_len,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
                "max_timestep": max_timestep,
                "num_states": num_states,
                "num_actions": num_actions,
                "rtg_target": rtg_target,
                "logic_alpha": float(args.logic_alpha),
                "dt_logic_loss_type": str(getattr(args, "dt_logic_loss_type", "auto")),
                "dt_logic_loss_type_effective": str(dt_logic_loss_type_effective),
                "logic_rollout_horizon": int(args.logic_rollout_horizon),
                "logic_temperature": float(args.logic_temperature),
                "dt_logic_dynamics_backend": str(args.dt_logic_dynamics_backend),
                "dynamics_stats": dynamics_stats,
                "dfa_formulas": dfa_formulas,
                "product_value_path": product_value_path,
                "product_value_metadata": None
                if product_value_table is None
                else dict(product_value_table.metadata),
            },
        },
        ckpt_path,
    )

    metrics = dt_metrics_template(
        env=args.env,
        seed=args.seed,
        rtg_target=rtg_target,
        runtime_sec=time.time() - t0,
        run_id=run_id,
        ts=ts,
        checkpoint_path=ckpt_path,
    )
    metrics["action_loss"] = float(epoch_losses[-1]) if epoch_losses else None
    metrics["total_loss"] = float(epoch_losses[-1]) if epoch_losses else None
    metrics["supervised_action_loss"] = float(epoch_supervised_losses[-1]) if epoch_supervised_losses else None
    metrics["logic_loss"] = float(epoch_logic_losses[-1]) if epoch_logic_losses else None
    metrics["dataset_size"] = int(len(dt_dataset))
    metrics["context_len"] = int(args.context_len)

    summary = {
        "epochs": int(args.epochs),
        "epoch_action_losses": [float(x) for x in epoch_losses],
        "epoch_total_losses": [float(x) for x in epoch_losses],
        "epoch_supervised_action_losses": [float(x) for x in epoch_supervised_losses],
        "epoch_logic_losses": [float(x) for x in epoch_logic_losses],
        "device": str(device),
        "checkpoint_path": ckpt_path,
        "dynamics_stats": dynamics_stats,
        "dt_logic_loss_type": str(getattr(args, "dt_logic_loss_type", "auto")),
        "dt_logic_loss_type_effective": str(dt_logic_loss_type_effective),
        "dfa_formulas": dfa_formulas,
        "product_value_path": product_value_path,
        "product_value_metadata": None if product_value_table is None else dict(product_value_table.metadata),
    }
    if dynamics_model_path is not None:
        summary["dynamics_checkpoint_path"] = dynamics_model_path

    if not args.no_eval_after_train:
        rollout_cfg = DTRolloutConfig(
            eval_num_episodes=args.eval_num_episodes,
            eval_max_steps=args.eval_max_steps,
            rtg_target=rtg_target,
        )
        eval_metrics = evaluate_dt_policy(
            model=model,
            env=base_dataset.env,
            env_name=args.env,
            seed=args.seed,
            cfg=rollout_cfg,
            context_len=args.context_len,
            device=device,
        )
        metrics.update(eval_metrics)
        metrics["success_rate"] = metrics.get("goal_rate")
        if args.env == "cb":
            metrics["bomb_hit_rate"] = metrics.get("hazard_hit_rate")

    write_metrics_files(run_dir, metrics)
    write_json(os.path.join(run_dir, "dt_summary.json"), summary)
    print(f"Saved DT run artifacts to {run_dir}")
    return model, base_dataset, run_dir


def main(argv=None):
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
