from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from envs.colour_bomb import CBConfig, ColourBombGridworldV1Env
from envs.frozenlake_env import FrozenLakeConfig, FrozenLakeEnv
from envs.nrm_nav_env import NRMSafetyNavConfig, NRMSafetyNavEnv
from logic.token_schema import (
    build_end_row,
    get_end_token_id,
    get_num_bins_per_dim,
    get_schema_for_env,
    validate_episode_tokens,
)
from datasets.dsrl_dataset import DSRLReplayConfig, DSRLReplayEnv


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = np.asarray(item)
    return arr


def _resolve_artifact_paths(artifact_path: str) -> tuple[Path, Path]:
    path = Path(artifact_path)
    if path.is_dir():
        npz_files = sorted(path.glob("*.npz"))
        meta_files = sorted(path.glob("*.meta.json"))
        if len(npz_files) == 1 and len(meta_files) == 1:
            return npz_files[0], meta_files[0]
        raise FileNotFoundError(
            f"Could not resolve unique dataset artifact files in directory: {path}"
        )

    if path.suffix == ".npz":
        npz_path = path
        meta_path = path.with_suffix(".meta.json")
    elif path.name.endswith(".meta.json"):
        meta_path = path
        npz_path = path.with_name(path.name[: -len(".meta.json")] + ".npz")
    else:
        # Treat as a bare stem. Don't use with_suffix() — artifact names can contain
        # dots (e.g. "random_1.0_seed1"), and with_suffix() only replaces the last
        # dot-component, which would corrupt names like "1.0_seed1" → "1.npz".
        npz_path = Path(str(path) + ".npz")
        meta_path = Path(str(path) + ".meta.json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ artifact not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata artifact not found: {meta_path}")
    return npz_path, meta_path


def _save_artifact_payload(args, dataset, artifact_tag: str = "dataset_snapshot") -> dict[str, str] | None:
    if not bool(getattr(args, "save_generated_dataset", True)):
        return None

    base_dir = getattr(args, "dataset_artifact_dir", None)
    if not base_dir:
        run_root = getattr(args, "run_dir", None) or getattr(args, "save_path", None)
        if run_root is None:
            run_root = os.path.join(getattr(args, "base_runs_dir", "runs"), "dataset_artifacts")
        base_dir = os.path.join(str(run_root), "dataset_artifacts")
    os.makedirs(base_dir, exist_ok=True)

    stem = str(getattr(args, "dataset_artifact_name", None) or artifact_tag).strip()
    if stem.lower().endswith(".npz"):
        stem = stem[:-4]
    if not stem:
        stem = "dataset_snapshot"

    npz_path = os.path.join(base_dir, f"{stem}.npz")
    meta_path = os.path.join(base_dir, f"{stem}.meta.json")

    episodes_tokens = list(getattr(dataset, "episodes_tokens", []) or [])
    episode_rewards = getattr(dataset, "episode_rewards", None)
    episode_policy_labels = getattr(dataset, "episode_policy_labels", None)
    episode_transitions = getattr(dataset, "episode_transitions", None)
    indices = getattr(dataset, "indices", None)

    payload = {
        "episodes_tokens": _to_object_array(episodes_tokens),
    }
    if episode_rewards is not None:
        payload["episode_rewards"] = _to_object_array(list(episode_rewards))
    if episode_policy_labels is not None:
        payload["episode_policy_labels"] = np.asarray(list(episode_policy_labels), dtype=object)
    if episode_transitions is not None:
        # Explicit (s, a, s_next, done) tuples per episode. Independent of token
        # serialization; preserves terminal next-states dropped from token rows.
        payload["episode_transitions"] = _to_object_array(list(episode_transitions))
    if indices is not None:
        payload["indices"] = np.asarray(list(indices), dtype=np.int64)

    np.savez_compressed(npz_path, **payload)

    lengths = [int(np.asarray(ep).shape[0]) for ep in episodes_tokens]
    rewards_per_episode = (
        [float(np.asarray(r).sum()) for r in episode_rewards] if episode_rewards is not None else None
    )
    metadata = {
        "artifact_format_version": 1,
        "saved_at_unix": float(time.time()),
        "artifact_tag": str(artifact_tag),
        "env": str(getattr(args, "env", "")),
        "seed": int(getattr(args, "seed", 0)),
        "spec": getattr(args, "spec", None),
        "dataset_class": dataset.__class__.__name__,
        "schema_id": getattr(dataset, "schema_id", None),
        "observation_dim": int(getattr(dataset, "observation_dim", 0) or 0),
        "action_dim": int(getattr(dataset, "action_dim", 0) or 0),
        "joined_dim": int(getattr(dataset, "joined_dim", 0) or 0),
        "rows_per_seg": int(getattr(dataset, "rows_per_seg", 0) or 0),
        "required_rows": int(getattr(dataset, "required_rows", 0) or 0),
        "num_segments": int(len(dataset)),
        "num_episodes": int(len(episodes_tokens)),
        "episode_length_min": int(min(lengths)) if lengths else 0,
        "episode_length_max": int(max(lengths)) if lengths else 0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "episode_return_mean": float(np.mean(rewards_per_episode))
        if rewards_per_episode
        else None,
        "episode_return_min": float(np.min(rewards_per_episode))
        if rewards_per_episode
        else None,
        "episode_return_max": float(np.max(rewards_per_episode))
        if rewards_per_episode
        else None,
        "dataset_config": {k: _to_jsonable(v) for k, v in vars(args).items()},
        "paths": {
            "npz": npz_path,
            "meta_json": meta_path,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {"npz_path": npz_path, "meta_path": meta_path}


def save_sequence_dataset_artifact(args, dataset, artifact_tag: str = "dataset_snapshot"):
    return _save_artifact_payload(args, dataset, artifact_tag=artifact_tag)


def _build_cb_env(dataset_config: dict) -> ColourBombGridworldV1Env:
    cfg = CBConfig(
        step_reward=float(dataset_config.get("step_reward", -0.01)),
        bomb_reward=float(dataset_config.get("bomb_reward", -1.0)),
        goal_reward=float(dataset_config.get("goal_reward", 1.0)),
        max_steps=int(dataset_config.get("max_steps", 200)),
        stochastic=bool(dataset_config.get("stochastic", False)),
    )
    return ColourBombGridworldV1Env(cfg)


def _build_frozenlake_env(dataset_config: dict) -> FrozenLakeEnv:
    cfg = FrozenLakeConfig(
        map_size=str(dataset_config.get("frozenlake_map_size", "4x4")),
        is_slippery=bool(dataset_config.get("frozenlake_is_slippery", False)),
        max_steps=int(dataset_config.get("max_steps", 100)),
    )
    return FrozenLakeEnv(cfg)


def _build_nrm_nav_env(dataset_config: dict) -> NRMSafetyNavEnv:
    cfg = NRMSafetyNavConfig(
        grid=dataset_config.get("grid", None),
        max_steps=int(dataset_config.get("max_steps", 200)),
        stochastic=bool(dataset_config.get("stochastic", False)),
        obstacle_punish=bool(dataset_config.get("obstacle_punish", True)),
    )
    return NRMSafetyNavEnv(cfg)


def _build_dsrl_replay_env(
    episodes_tokens: list[np.ndarray],
    episode_rewards: list[np.ndarray] | None,
    dataset_config: dict,
):
    schema = get_schema_for_env("dsrl")
    if not episodes_tokens:
        raise ValueError("DSRL dataset artifact does not contain any episodes.")

    state_idx = schema.field_index("state")
    action_idx = schema.field_index("action")
    reward_idx = schema.field_index("reward")
    cost_idx = schema.field_index("safety_cost")

    transition_counts: dict[tuple[int, int], dict[tuple[int, int, int, int, int], int]] = {}
    start_state_counts: dict[int, int] = {}
    unsafe_state_ids: set[int] = set()
    goal_state_ids: set[int] = set()
    reward_sum = 0.0
    reward_count = 0
    max_state = -1
    max_action = -1

    for ep_idx, ep in enumerate(episodes_tokens):
        rows = np.asarray(ep, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[0] < 2:
            continue
        core_rows = rows[:-1]
        if core_rows.shape[0] == 0:
            continue

        start_state_counts[int(core_rows[0, state_idx])] = int(
            start_state_counts.get(int(core_rows[0, state_idx]), 0) + 1
        )

        ep_rews = None
        if episode_rewards is not None and ep_idx < len(episode_rewards):
            ep_rews = np.asarray(episode_rewards[ep_idx], dtype=np.float32).reshape(-1)

        for t in range(core_rows.shape[0]):
            cur = core_rows[t]
            s = int(cur[state_idx])
            a = int(cur[action_idx])
            reward_tok = int(cur[reward_idx])
            cost_tok = int(cur[cost_idx])
            max_state = max(max_state, s, int(core_rows[t + 1, state_idx]) if (t + 1) < core_rows.shape[0] else s)
            max_action = max(max_action, a)

            if ep_rews is not None and t < ep_rews.shape[0]:
                r = float(ep_rews[t])
            else:
                r = float(reward_tok)

            next_state = int(core_rows[t + 1, state_idx]) if (t + 1) < core_rows.shape[0] else s
            done = int(t == core_rows.shape[0] - 1)
            goal = int(reward_tok == 1)

            if cost_tok == 1:
                unsafe_state_ids.add(next_state)
            if goal == 1:
                goal_state_ids.add(next_state)

            bucket = transition_counts.setdefault((s, a), {})
            out_key = (next_state, int(round(r * 1000.0)), cost_tok, done, goal)
            bucket[out_key] = int(bucket.get(out_key, 0) + 1)
            reward_sum += r
            reward_count += 1

    num_states = max_state + 1 if max_state >= 0 else 1
    num_actions = max_action + 1 if max_action >= 0 else 1

    transitions: dict[tuple[int, int], list[dict]] = {}
    for key, out_counts in transition_counts.items():
        total = float(sum(out_counts.values()))
        outcomes = []
        for (ns, r_milli, c_tok, done, goal), cnt in out_counts.items():
            outcomes.append(
                {
                    "prob": float(cnt) / max(total, 1e-12),
                    "next_state": int(ns),
                    "reward": float(r_milli) / 1000.0,
                    "cost": float(c_tok),
                    "done": bool(done),
                    "goal": bool(goal),
                }
            )
        transitions[(int(key[0]), int(key[1]))] = outcomes

    total_starts = float(sum(start_state_counts.values()))
    start_state_probs = {
        int(s): float(c) / max(total_starts, 1e-12) for s, c in start_state_counts.items()
    }
    avg_reward = float(reward_sum / max(1, reward_count))
    max_steps = int(dataset_config.get("max_steps", 200))
    return DSRLReplayEnv(
        num_states=int(num_states),
        num_actions=int(max(1, num_actions)),
        transitions=transitions,
        start_state_probs=start_state_probs,
        unsafe_state_ids=unsafe_state_ids,
        goal_state_ids=goal_state_ids,
        cfg=DSRLReplayConfig(max_steps=max_steps, stochastic=True),
        default_reward=avg_reward,
    )


class LoadedSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        env_name: str,
        dataset_class: str,
        schema_id: str,
        env,
        episodes_tokens: list[np.ndarray],
        episode_rewards: list[np.ndarray] | None,
        episode_policy_labels: list[str] | None,
        dataset_config: dict,
        artifact_path: str,
        meta: dict,
        sequence_length: int,
        target_shift: str,
        episode_transitions: list[np.ndarray] | None = None,
    ):
        self.env_name = str(env_name)
        self.dataset_class = str(dataset_class)
        self.schema_id = str(schema_id)
        self.env = env
        self.episodes_tokens = [np.asarray(ep, dtype=np.int64) for ep in episodes_tokens]
        self.episode_rewards = (
            [np.asarray(r, dtype=np.float32) for r in episode_rewards]
            if episode_rewards is not None
            else None
        )
        self.episode_policy_labels = (
            [str(x) for x in episode_policy_labels] if episode_policy_labels is not None else None
        )
        self.episode_transitions = (
            [np.asarray(tr, dtype=np.int64).reshape(-1, 4) for tr in episode_transitions]
            if episode_transitions is not None
            else None
        )
        self.dataset_config = dict(dataset_config)
        self.meta = dict(meta)
        self.source_artifact_path = str(artifact_path)
        self.sequence_length = int(sequence_length)
        self.target_shift = str(target_shift)
        self.token_schema = get_schema_for_env(self.env_name)
        self.observation_dim = int(meta.get("observation_dim", 1) or 1)
        self.action_dim = int(meta.get("action_dim", 1) or 1)
        self.joined_dim = int(meta.get("joined_dim", self.token_schema.width) or self.token_schema.width)
        self.rows_per_seg = max(1, self.sequence_length // self.token_schema.width)
        self.required_rows = (
            self.rows_per_seg + 1 if self.target_shift == "transition" else max(2, self.rows_per_seg)
        )
        self.end_token_id = get_end_token_id(
            self.token_schema, self.env.observation_space.n, self.env.action_space.n
        )
        self.num_bins_per_dim = get_num_bins_per_dim(
            self.token_schema, self.env.observation_space.n, self.env.action_space.n
        )
        self.keep_short_episodes = self.dataset_class in {
            "FrozenLakeSequenceDataset",
            "DSRLSequenceDataset",
        }
        self.indices: list[tuple[int, int]] = []
        for ep_idx, rows in enumerate(self.episodes_tokens):
            n_rows = int(rows.shape[0])
            if n_rows == 0:
                continue
            if n_rows < self.required_rows:
                if self.keep_short_episodes:
                    self.indices.append((ep_idx, 0))
                continue
            starts = list(range(0, max(1, n_rows - self.required_rows), self.rows_per_seg))
            tail = n_rows - self.required_rows
            if tail not in starts:
                starts.append(tail)
            for start_row in starts:
                self.indices.append((ep_idx, int(start_row)))

        self.state_semantics = str(
            self.dataset_config.get("cb_state_semantics")
            or self.dataset_config.get("state_semantics")
            or ("post" if self.env_name == "cb" else "pre")
        )
        self.reward_goal_threshold = float(self.dataset_config.get("reward_goal_threshold", 0.0))
        self.cost_unsafe_threshold = float(self.dataset_config.get("cost_unsafe_threshold", 0.0))
        self.cost_unsafe_quantile = self.dataset_config.get("cost_unsafe_quantile", None)
        self.dataset_path = self.source_artifact_path

        for key, value in self.dataset_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_row = self.indices[idx]
        rows = self.episodes_tokens[ep_idx]
        seg_rows = rows[start_row : start_row + self.required_rows]
        valid_rows = int(seg_rows.shape[0])
        if valid_rows < self.required_rows:
            pad_rows = self.required_rows - valid_rows
            end_row = build_end_row(self.token_schema, self.end_token_id).reshape(1, -1)
            seg_rows = np.vstack([seg_rows, np.repeat(end_row, repeats=pad_rows, axis=0)])
        flat = seg_rows.reshape(-1)
        if self.target_shift == "token":
            x = torch.from_numpy(flat[:-1].astype(np.int64))
            y = torch.from_numpy(flat[1:].astype(np.int64))
        else:
            x = torch.from_numpy(flat[: -self.joined_dim].astype(np.int64))
            y = torch.from_numpy(flat[self.joined_dim :].astype(np.int64))
        mask = torch.ones_like(x, dtype=torch.float32)
        if valid_rows < self.required_rows:
            if self.target_shift == "token":
                valid_target_tokens = max(0, valid_rows * self.joined_dim - 1)
            else:
                valid_target_tokens = max(0, valid_rows - 1) * self.joined_dim
            if valid_target_tokens < mask.shape[0]:
                mask[valid_target_tokens:] = 0.0
        return x, y, mask


def load_sequence_dataset_artifact(
    artifact_path: str,
    *,
    sequence_length: int,
    target_shift: str | None = None,
):
    npz_path, meta_path = _resolve_artifact_paths(artifact_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"Dataset artifact metadata must be a mapping: {meta_path}")

    dataset_config = dict(meta.get("dataset_config", {}) or {})
    env_name = str(meta.get("env") or dataset_config.get("env") or "")
    if not env_name:
        raise ValueError(f"Dataset artifact missing env name: {meta_path}")

    dataset_class = str(meta.get("dataset_class") or "SequenceDataset")
    schema_id = str(meta.get("schema_id") or get_schema_for_env(env_name).schema_id)
    target_shift = str(target_shift or dataset_config.get("target_shift", "token"))

    with np.load(npz_path, allow_pickle=True) as npz:
        episodes_tokens = [np.asarray(ep, dtype=np.int64) for ep in npz["episodes_tokens"]]
        episode_rewards = (
            [np.asarray(r, dtype=np.float32) for r in npz["episode_rewards"]]
            if "episode_rewards" in npz
            else None
        )
        episode_policy_labels = (
            [str(x) for x in np.asarray(npz["episode_policy_labels"], dtype=object).tolist()]
            if "episode_policy_labels" in npz
            else None
        )
        episode_transitions = (
            [np.asarray(tr, dtype=np.int64).reshape(-1, 4) for tr in npz["episode_transitions"]]
            if "episode_transitions" in npz
            else None
        )

    if env_name == "cb":
        env = _build_cb_env(dataset_config)
    elif env_name == "frozenlake":
        env = _build_frozenlake_env(dataset_config)
    elif env_name == "nrm_nav":
        env = _build_nrm_nav_env(dataset_config)
    elif env_name == "dsrl":
        env = _build_dsrl_replay_env(episodes_tokens, episode_rewards, dataset_config)
    else:
        raise ValueError(f"Unsupported dataset artifact env '{env_name}'")

    for ep in episodes_tokens:
        validate_episode_tokens(
            ep,
            schema=get_schema_for_env(env_name),
            end_token_id=get_end_token_id(
                get_schema_for_env(env_name), env.observation_space.n, env.action_space.n
            ),
            observation_space_n=env.observation_space.n,
            action_space_n=env.action_space.n,
        )

    dataset = LoadedSequenceDataset(
        env_name=env_name,
        dataset_class=dataset_class,
        schema_id=schema_id,
        env=env,
        episodes_tokens=episodes_tokens,
        episode_rewards=episode_rewards,
        episode_policy_labels=episode_policy_labels,
        dataset_config=dataset_config,
        artifact_path=str(artifact_path),
        meta=meta,
        sequence_length=int(sequence_length),
        target_shift=target_shift,
        episode_transitions=episode_transitions,
    )
    return dataset
