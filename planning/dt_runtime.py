from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from datasets.cb_dataset import CBSequenceDataset
from datasets.dt_dataset import DTSequenceDataset
from datasets.frozenlake_dataset import FrozenLakeSequenceDataset
from planning.eval_runtime import write_json


@dataclass
class DTRolloutConfig:
    eval_num_episodes: int = 100
    eval_max_steps: int | None = None
    rtg_target: float = 1.0


def apply_smoke_mode_dt(args):
    if not getattr(args, "smoke", False):
        return args
    if hasattr(args, "num_episodes"):
        args.num_episodes = min(int(args.num_episodes), 200)
    if hasattr(args, "max_steps"):
        args.max_steps = min(int(args.max_steps), 30)
    if hasattr(args, "epochs"):
        args.epochs = min(int(args.epochs), 1)
    if hasattr(args, "batch_size"):
        args.batch_size = min(int(args.batch_size), 32)
    if hasattr(args, "context_len"):
        args.context_len = min(int(args.context_len), 20)
    if hasattr(args, "n_layer"):
        args.n_layer = min(int(args.n_layer), 2)
    if hasattr(args, "n_head"):
        args.n_head = min(int(args.n_head), 2)
    if hasattr(args, "n_embd"):
        args.n_embd = min(int(args.n_embd), 64)
    if hasattr(args, "eval_num_episodes"):
        args.eval_num_episodes = min(int(args.eval_num_episodes), 16)
    if getattr(args, "env", None) == "frozenlake" and hasattr(args, "policy_mix"):
        # Ensure smoke dataset has some successful trajectories for a non-degenerate DT signal.
        args.policy_mix = max(float(args.policy_mix), 1.0)
    return args


def build_dt_offline_source(args):
    if args.env == "cb":
        dataset = CBSequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=max(8, args.context_len * 4),
            stochastic=getattr(args, "stochastic", False),
            seed=args.seed,
        )
        return dataset, None
    if args.env == "frozenlake":
        dataset = FrozenLakeSequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=max(8, args.context_len * 4),
            seed=args.seed,
            map_size=args.frozenlake_map_size,
            is_slippery=args.frozenlake_is_slippery,
            policy_mix=args.policy_mix,
        )
        return dataset, None
    if args.env == "antmaze":
        reason = (
            "DT v1 in this milestone supports only discrete-action envs (cb/frozenlake). "
            "AntMaze is skipped."
        )
        return None, reason
    raise ValueError(f"Unsupported DT env '{args.env}'")


def build_dt_dataset(base_dataset, context_len: int):
    episode_rewards = getattr(base_dataset, "episode_rewards", None)
    if episode_rewards is None:
        episode_rewards = [np.zeros(ep.shape[0] - 1, dtype=np.float32) for ep in base_dataset.episodes_tokens]
    return DTSequenceDataset(
        episodes_tokens=base_dataset.episodes_tokens,
        episode_rewards=episode_rewards,
        context_len=context_len,
        num_actions=base_dataset.env.action_space.n,
        state_index=0,
        action_index=1,
    )


def compute_default_rtg_target(base_dataset, quantile: float = 0.75) -> float:
    rewards = getattr(base_dataset, "episode_rewards", None)
    if not rewards:
        return 1.0
    returns = np.asarray([float(np.sum(r)) for r in rewards], dtype=np.float32)
    if len(returns) == 0:
        return 1.0
    return float(np.quantile(returns, quantile))


def _step_env(env, action: int):
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
    return int(obs), float(reward), bool(done), dict(info)


def _append_context(history, value, max_len: int):
    history.append(value)
    if len(history) > max_len:
        history.pop(0)


def _predict_action(model, device, states, prev_actions, rtgs, timesteps):
    states_t = torch.tensor([states], dtype=torch.long, device=device)
    prev_actions_t = torch.tensor([prev_actions], dtype=torch.long, device=device)
    rtg_t = torch.tensor([rtgs], dtype=torch.float32, device=device)
    ts_t = torch.tensor([timesteps], dtype=torch.long, device=device)
    mask_t = torch.ones_like(states_t, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(states_t, prev_actions_t, rtg_t, ts_t, attention_mask=mask_t)
    return int(torch.argmax(logits[0, -1, :]).item())


def evaluate_dt_policy(
    model,
    env,
    env_name: str,
    seed: int,
    cfg: DTRolloutConfig,
    context_len: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    returns = []
    goal_hits = 0
    hazard_hits = 0
    violation_episodes = 0

    max_steps = cfg.eval_max_steps if cfg.eval_max_steps is not None else getattr(env.cfg, "max_steps", 100)
    pad_action = env.action_space.n

    for ep in range(int(cfg.eval_num_episodes)):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        step_idx = 0
        prev_action = pad_action
        cum_reward = 0.0
        hit_goal = False
        hit_hazard = False

        states_hist: list[int] = []
        prev_actions_hist: list[int] = []
        rtg_hist: list[float] = []
        t_hist: list[int] = []

        while not done and step_idx < int(max_steps):
            _append_context(states_hist, int(obs), context_len)
            _append_context(prev_actions_hist, int(prev_action), context_len)
            rtg_now = float(cfg.rtg_target - cum_reward)
            _append_context(rtg_hist, rtg_now, context_len)
            _append_context(t_hist, int(step_idx), context_len)

            action = _predict_action(
                model=model,
                device=device,
                states=states_hist,
                prev_actions=prev_actions_hist,
                rtgs=rtg_hist,
                timesteps=t_hist,
            )
            next_obs, reward, done, info = _step_env(env, action)
            total_r += reward
            cum_reward += reward
            prev_action = action
            obs = next_obs
            step_idx += 1

            terminal_type = info.get("terminal_type")
            if env_name == "frozenlake":
                if terminal_type == "G":
                    hit_goal = True
                if terminal_type == "H":
                    hit_hazard = True
            elif env_name == "cb":
                if terminal_type in {"P", "Y", "BLU"}:
                    hit_goal = True
                if terminal_type == "B":
                    hit_hazard = True

        returns.append(total_r)
        goal_hits += int(hit_goal)
        hazard_hits += int(hit_hazard)
        violation_episodes += int(hit_hazard)

    n = max(1, int(cfg.eval_num_episodes))
    return {
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "goal_rate": float(goal_hits / n),
        "hazard_hit_rate": float(hazard_hits / n),
        "violation_rate_episode": float(violation_episodes / n),
        "violation_rate": float(violation_episodes / n),
        "num_episodes": int(cfg.eval_num_episodes),
    }


def evaluate_random_policy(env, env_name: str, seed: int, eval_num_episodes: int, eval_max_steps: int | None):
    returns = []
    goal_hits = 0
    hazard_hits = 0
    max_steps = eval_max_steps if eval_max_steps is not None else getattr(env.cfg, "max_steps", 100)
    rng = np.random.RandomState(seed)

    for ep in range(int(eval_num_episodes)):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        hit_goal = False
        hit_hazard = False
        while not done and steps < int(max_steps):
            action = int(rng.randint(env.action_space.n))
            obs, reward, done, info = _step_env(env, action)
            total_r += reward
            steps += 1
            terminal_type = info.get("terminal_type")
            if env_name == "frozenlake":
                if terminal_type == "G":
                    hit_goal = True
                if terminal_type == "H":
                    hit_hazard = True
            elif env_name == "cb":
                if terminal_type in {"P", "Y", "BLU"}:
                    hit_goal = True
                if terminal_type == "B":
                    hit_hazard = True
        returns.append(total_r)
        goal_hits += int(hit_goal)
        hazard_hits += int(hit_hazard)

    n = max(1, int(eval_num_episodes))
    return {
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "goal_rate": float(goal_hits / n),
        "hazard_hit_rate": float(hazard_hits / n),
        "violation_rate": float(hazard_hits / n),
    }


def dt_metrics_template(
    env: str,
    seed: int,
    rtg_target: float,
    runtime_sec: float,
    run_id: str,
    ts: str,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    return {
        "return_mean": None,
        "return_std": None,
        "violation_rate": None,
        "satisfaction_rate": None,
        "runtime_sec": float(runtime_sec),
        "env": env,
        "spec": None,
        "seed": int(seed),
        "num_episodes": None,
        "satisfaction_soft_mean": None,
        "violation_rate_episode": None,
        "violation_rate_step": None,
        "goal_rate": None,
        "bomb_hit_rate": None,
        "hazard_hit_rate": None,
        "decoding_mode": None,
        "beam_width": None,
        "model_type": "dt",
        "checkpoint_path": checkpoint_path,
        "run_id": run_id,
        "timestamp_utc": ts,
        "rtg_target": float(rtg_target),
        "action_loss": None,
        "success_rate": None,
        "dataset_size": None,
        "context_len": None,
    }


def write_metrics_files(run_dir: str, metrics: dict[str, Any]) -> tuple[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    json_path = os.path.join(run_dir, "metrics.json")
    csv_path = os.path.join(run_dir, "metrics.csv")
    write_json(json_path, metrics)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return json_path, csv_path


def write_skip_metrics(
    run_dir: str,
    env: str,
    seed: int,
    rtg_target: float,
    run_id: str,
    ts: str,
    reason: str,
):
    metrics = dt_metrics_template(
        env=env,
        seed=seed,
        rtg_target=rtg_target,
        runtime_sec=0.0,
        run_id=run_id,
        ts=ts,
        checkpoint_path=None,
    )
    metrics["skipped"] = True
    metrics["skip_reason"] = reason
    write_metrics_files(run_dir, metrics)
    write_json(
        os.path.join(run_dir, "dt_summary.json"),
        {
            "status": "skipped",
            "reason": reason,
            "env": env,
            "seed": int(seed),
            "run_id": run_id,
            "timestamp_utc": ts,
        },
    )
    return metrics
