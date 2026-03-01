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
from planning.dt_runtime import (
    DTRolloutConfig,
    apply_smoke_mode_dt,
    build_dt_dataset,
    build_tabular_dynamics,
    build_dt_offline_source,
    compute_dt_logic_rollout_penalty,
    compute_default_rtg_target,
    dt_metrics_template,
    evaluate_dt_policy,
    hazard_mask_for_env,
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
    p.add_argument("--logic_rollout_horizon", type=int, default=2)
    p.add_argument("--logic_temperature", type=float, default=1.0)

    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--rtg_target", type=float, default=None)
    p.add_argument("--no_eval_after_train", action="store_true")
    p.add_argument("--eval_num_episodes", type=int, default=32)
    p.add_argument("--eval_max_steps", type=int, default=None)

    p.add_argument("--stochastic", action="store_true", help="Enable stochastic transitions for cb dataset.")
    p.add_argument("--frozenlake_map_size", type=str, choices=["4x4", "8x8"], default="4x4")
    p.add_argument("--frozenlake_is_slippery", action="store_true")
    p.add_argument("--policy_mix", type=float, default=0.0)
    p.add_argument("--dsrl_dataset_path", type=str, default=None)
    p.add_argument("--dsrl_dataset_key", type=str, default="PointGoal1")
    p.add_argument("--dsrl_state_bins", type=int, default=128)
    p.add_argument("--dsrl_action_bins", type=int, default=16)
    p.add_argument("--dsrl_reward_goal_threshold", type=float, default=0.0)
    p.add_argument("--dsrl_download", action="store_true")

    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--base_runs_dir", type=str, default="runs")
    return p


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

    dt_dataset = build_dt_dataset(base_dataset, context_len=args.context_len)
    if len(dt_dataset) == 0:
        raise RuntimeError("DT dataset is empty; cannot train.")

    dataloader = DataLoader(dt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    num_states = int(base_dataset.env.observation_space.n)
    num_actions = int(base_dataset.env.action_space.n)
    transition_probs_t = None
    hazard_mask_t = None
    if float(args.logic_alpha) > 0.0:
        dyn = build_tabular_dynamics(base_dataset)
        hazard = hazard_mask_for_env(args.env, base_dataset.env)
        transition_probs_t = torch.from_numpy(dyn).float().to(device)
        hazard_mask_t = torch.from_numpy(hazard).float().to(device)

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
            if transition_probs_t is not None and hazard_mask_t is not None:
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
                "logic_rollout_horizon": int(args.logic_rollout_horizon),
                "logic_temperature": float(args.logic_temperature),
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
    }

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


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
