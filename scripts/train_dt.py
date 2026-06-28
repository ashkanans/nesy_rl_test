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
        choices=["auto", "hazard", "dfa"],
        default="auto",
        help=(
            "DT logic loss: hazard keeps the legacy unsafe-state penalty; "
            "dfa uses the actual LTLf/DFA formula when --spec/--ltl_formula(s) is provided; "
            "auto selects dfa when a formula source is present, otherwise hazard."
        ),
    )
    p.add_argument("--logic_rollout_horizon", type=int, default=2)
    p.add_argument("--logic_temperature", type=float, default=1.0)
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
    if requested == "dfa" and not _has_formula_source(args):
        raise ValueError("--dt_logic_loss_type dfa requires --spec, --ltl_formula, or --ltl_formulas.")
    return requested


def _to_device(batch, device):
    return [x.to(device) for x in batch]


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

    dt_dataset = build_dt_dataset(base_dataset, context_len=args.context_len)
    if len(dt_dataset) == 0:
        raise RuntimeError("DT dataset is empty; cannot train.")

    dataloader = DataLoader(dt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    num_states = int(base_dataset.env.observation_space.n)
    num_actions = int(base_dataset.env.action_space.n)
    transition_probs_t = None
    hazard_mask_t = None
    dt_logic_loss_type_effective = "none"
    dfa_adapter = None
    dfa_deep = None
    dfa_formulas = None
    dynamics_stats = None
    dynamics_model_path = None
    if float(args.logic_alpha) > 0.0:
        dt_logic_loss_type_effective = _resolve_dt_logic_loss_type(args)
        if dt_logic_loss_type_effective == "hazard":
            hazard = hazard_mask_for_env(args.env, base_dataset.env)
            hazard_mask_t = torch.from_numpy(hazard).float().to(device)
        elif dt_logic_loss_type_effective == "dfa":
            from train_cb import build_adapter_and_dfa, resolve_formulas

            dfa_adapter, dfa_deep, _ = build_adapter_and_dfa(args, base_dataset)
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
    epoch_logic_losses = []
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        running_logic = 0.0
        count = 0
        for batch in dataloader:
            states, prev_actions, rtg, timesteps, targets, mask = _to_device(batch, device)
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
            loss = sup_loss + float(args.logic_alpha) * logic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item())
            running_logic += float(logic_loss.item()) if torch.isfinite(logic_loss) else 0.0
            count += 1
        mean_loss = running / max(1, count)
        mean_logic = running_logic / max(1, count)
        epoch_losses.append(mean_loss)
        epoch_logic_losses.append(mean_logic)
        print(f"epoch {epoch} | action_loss {mean_loss:.4f} | logic_loss {mean_logic:.4f}")

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
    metrics["logic_loss"] = float(epoch_logic_losses[-1]) if epoch_logic_losses else None
    metrics["dataset_size"] = int(len(dt_dataset))
    metrics["context_len"] = int(args.context_len)

    summary = {
        "epochs": int(args.epochs),
        "epoch_action_losses": [float(x) for x in epoch_losses],
        "epoch_logic_losses": [float(x) for x in epoch_logic_losses],
        "device": str(device),
        "checkpoint_path": ckpt_path,
        "dynamics_stats": dynamics_stats,
        "dt_logic_loss_type": str(getattr(args, "dt_logic_loss_type", "auto")),
        "dt_logic_loss_type_effective": str(dt_logic_loss_type_effective),
        "dfa_formulas": dfa_formulas,
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
