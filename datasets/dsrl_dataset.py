from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from logic.token_schema import (
    build_end_row,
    get_end_token_id,
    get_num_bins_per_dim,
    get_schema_for_env,
    make_transition_row,
    validate_episode_tokens,
)


def _load_catalog_rows(catalog_path: str | None = None) -> list[dict]:
    if catalog_path is None:
        catalog_path = "artifacts/dsrl_catalog/dsrl_catalog.json"
    path = Path(catalog_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return payload


def _resolve_dataset_url(dataset_key: str, catalog_rows: list[dict]) -> str | None:
    key = str(dataset_key)
    for row in catalog_rows:
        if row.get("dataset_key") == key:
            return row.get("dataset_url")
    return None


def _download_if_missing(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        urllib.request.urlretrieve(url, str(target_path))
    return target_path


def _load_h5_array(h5f: h5py.File, keys: list[str], required: bool = True) -> np.ndarray | None:
    for key in keys:
        if key in h5f:
            return np.asarray(h5f[key])
    if required:
        raise KeyError(f"Missing required dataset key; tried: {keys}")
    return None


def _project_to_bins(values: np.ndarray, num_bins: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        proj = values
    else:
        flat = values.reshape(values.shape[0], -1)
        rng = np.random.RandomState(int(seed))
        w = rng.normal(size=(flat.shape[1],)).astype(np.float32)
        norm = float(np.linalg.norm(w))
        if norm <= 1e-12:
            w = np.ones_like(w, dtype=np.float32)
            norm = float(np.linalg.norm(w))
        w = w / norm
        proj = flat @ w

    lo = float(np.percentile(proj, 1.0))
    hi = float(np.percentile(proj, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(proj.shape[0], dtype=np.int64)

    edges = np.linspace(lo, hi, int(num_bins) + 1, dtype=np.float32)[1:-1]
    bins = np.digitize(proj, edges, right=False).astype(np.int64)
    return np.clip(bins, 0, int(num_bins) - 1)


def _discretize_actions(actions: np.ndarray, action_bins: int, seed: int) -> np.ndarray:
    arr = np.asarray(actions)
    if arr.ndim == 1:
        flat = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        flat = arr[:, 0]
    else:
        return _project_to_bins(arr, action_bins, seed=seed + 13)

    if np.issubdtype(flat.dtype, np.integer):
        unique_vals = sorted(int(x) for x in np.unique(flat))
        if unique_vals and len(unique_vals) <= int(action_bins):
            mapping = {v: i for i, v in enumerate(unique_vals)}
            return np.asarray([mapping[int(x)] for x in flat], dtype=np.int64)

    return _project_to_bins(flat.astype(np.float32), action_bins, seed=seed + 13)


def _segment_episodes(terminals: np.ndarray, timeouts: np.ndarray | None) -> list[tuple[int, int]]:
    terminals = np.asarray(terminals).astype(bool).reshape(-1)
    if timeouts is None:
        done = terminals
    else:
        done = np.logical_or(terminals, np.asarray(timeouts).astype(bool).reshape(-1))

    segments: list[tuple[int, int]] = []
    start = 0
    n = int(done.shape[0])
    for i in range(n):
        if done[i]:
            segments.append((start, i + 1))
            start = i + 1
    if start < n:
        segments.append((start, n))
    return [(s, e) for (s, e) in segments if e > s]


@dataclass
class DSRLReplayConfig:
    max_steps: int = 200
    stochastic: bool = True


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)


class DSRLReplayEnv:
    """
    Lightweight replay environment induced from an offline DSRL dataset.

    It samples transitions from an empirical tabular model over discretized
    state/action tokens so existing TT/DT rollout evaluators can run unchanged.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        transitions: dict[tuple[int, int], list[dict]],
        start_state_probs: dict[int, float],
        unsafe_state_ids: set[int],
        goal_state_ids: set[int],
        cfg: DSRLReplayConfig | None = None,
        default_reward: float = 0.0,
    ):
        self.cfg = cfg or DSRLReplayConfig()
        self._num_states = int(num_states)
        self._num_actions = int(num_actions)
        self.observation_space = _DiscreteSpace(num_states)
        self.action_space = _DiscreteSpace(num_actions)
        self._transitions = transitions
        self._start_state_probs = dict(start_state_probs)
        self.unsafe_state_ids = set(int(x) for x in unsafe_state_ids)
        self.goal_state_ids = set(int(x) for x in goal_state_ids)
        self._default_reward = float(default_reward)
        self._steps = 0
        self._cur_state = 0
        self._rng = np.random.RandomState(0)

    def clone(self) -> "DSRLReplayEnv":
        return DSRLReplayEnv(
            num_states=self._num_states,
            num_actions=self._num_actions,
            transitions=self._transitions,
            start_state_probs=self._start_state_probs,
            unsafe_state_ids=self.unsafe_state_ids,
            goal_state_ids=self.goal_state_ids,
            cfg=DSRLReplayConfig(
                max_steps=int(self.cfg.max_steps),
                stochastic=bool(self.cfg.stochastic),
            ),
            default_reward=self._default_reward,
        )

    def _sample_start_state(self) -> int:
        states = np.asarray(sorted(self._start_state_probs.keys()), dtype=np.int64)
        if states.size == 0:
            return 0
        probs = np.asarray([self._start_state_probs[int(s)] for s in states], dtype=np.float64)
        probs = probs / np.maximum(probs.sum(), 1e-12)
        idx = int(self._rng.choice(len(states), p=probs))
        return int(states[idx])

    def reset(self, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self._rng = np.random.RandomState(int(seed))
        self._steps = 0
        self._cur_state = self._sample_start_state()
        return int(self._cur_state), {}

    def step(self, action: int):
        self._steps += 1
        key = (int(self._cur_state), int(action))
        outcomes = self._transitions.get(key)
        if not outcomes:
            next_state = int(self._cur_state)
            reward = float(self._default_reward)
            cost = 0.0
            done = False
            goal = False
        else:
            probs = np.asarray([float(x["prob"]) for x in outcomes], dtype=np.float64)
            probs = probs / np.maximum(probs.sum(), 1e-12)
            idx = int(self._rng.choice(len(outcomes), p=probs))
            out = outcomes[idx]
            next_state = int(out["next_state"])
            reward = float(out["reward"])
            cost = float(out["cost"])
            done = bool(out["done"])
            goal = bool(out["goal"])

        self._cur_state = int(next_state)
        info = {"cost": float(cost), "goal": bool(goal)}
        if goal:
            info["terminal_type"] = "G"
        elif cost > 0.0:
            info["terminal_type"] = "X"

        if self._steps >= int(self.cfg.max_steps):
            done = True
            info["truncated"] = True

        return int(self._cur_state), float(reward), bool(done), info

    def is_unsafe_state(self, state: int) -> bool:
        return int(state) in self.unsafe_state_ids

    def is_goal_state(self, state: int) -> bool:
        return int(state) in self.goal_state_ids

    def render(self):
        return f"DSRLReplayEnv(state={self._cur_state}, steps={self._steps})"


class DSRLSequenceDataset(Dataset):
    """
    Phase-1 DSRL offline dataset adapter (dataset-source integration).

    It loads a DSRL HDF5 dataset, discretizes observations/actions into token IDs,
    builds canonical 4-field transitions [state, action, reward_token, safety_cost],
    and exposes a replay env induced from empirical tabular transitions.
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        dataset_key: str = "PointGoal1",
        sequence_length: int = 200,
        seed: int = 0,
        max_steps: int = 200,
        num_episodes: int | None = None,
        state_bins: int = 128,
        action_bins: int = 16,
        reward_goal_threshold: float = 0.0,
        cost_unsafe_threshold: float = 0.0,
        cost_unsafe_quantile: float | None = None,
        target_shift: str = "token",
        download: bool = False,
    ):
        self.sequence_length = int(sequence_length)
        self.target_shift = str(target_shift)
        if self.target_shift not in {"token", "transition"}:
            raise ValueError("target_shift must be 'token' or 'transition'.")
        self.token_schema = get_schema_for_env("dsrl")
        self.schema_id = self.token_schema.schema_id
        self.dataset_key = str(dataset_key)
        self.reward_goal_threshold = float(reward_goal_threshold)
        self.cost_unsafe_threshold = float(cost_unsafe_threshold)
        self.cost_unsafe_quantile = (
            None if cost_unsafe_quantile is None else float(cost_unsafe_quantile)
        )

        path = Path(dataset_path) if dataset_path is not None else None
        if path is None:
            catalog_rows = _load_catalog_rows()
            url = _resolve_dataset_url(self.dataset_key, catalog_rows)
            if not url:
                raise ValueError(
                    "Could not resolve DSRL dataset URL from catalog for key "
                    f"'{self.dataset_key}'. Provide --dsrl_dataset_path or build catalog first."
                )
            target = Path("artifacts/dsrl_datasets") / Path(url).name
            if not target.exists() and not download:
                raise FileNotFoundError(
                    f"Dataset file not found at {target}. Re-run with --dsrl_download to fetch it."
                )
            path = _download_if_missing(url, target) if download else target
        if not path.exists():
            raise FileNotFoundError(f"DSRL dataset file not found: {path}")

        with h5py.File(path, "r") as h5f:
            observations = _load_h5_array(h5f, ["observations"])
            actions = _load_h5_array(h5f, ["actions"])
            rewards = _load_h5_array(h5f, ["rewards"]).reshape(-1).astype(np.float32)
            costs_raw = _load_h5_array(h5f, ["costs", "cost"], required=False)
            terminals = _load_h5_array(h5f, ["terminals", "dones"]).reshape(-1).astype(np.float32)
            timeouts = _load_h5_array(h5f, ["timeouts"], required=False)

        costs = (
            np.zeros_like(rewards, dtype=np.float32)
            if costs_raw is None
            else np.asarray(costs_raw).reshape(-1).astype(np.float32)
        )
        if self.cost_unsafe_quantile is not None:
            q = float(np.clip(self.cost_unsafe_quantile, 0.0, 1.0))
            positives = costs[costs > 0.0]
            ref = positives if positives.size > 0 else costs
            self.cost_unsafe_threshold = float(np.quantile(ref, q))
        if timeouts is not None:
            timeouts = np.asarray(timeouts).reshape(-1).astype(np.float32)

        n = int(len(rewards))
        if observations.shape[0] != n or actions.shape[0] != n or costs.shape[0] != n:
            raise ValueError("DSRL dataset arrays have inconsistent leading dimensions.")

        state_ids = _project_to_bins(np.asarray(observations), int(state_bins), seed=int(seed) + 101)
        action_ids = _discretize_actions(np.asarray(actions), int(action_bins), seed=int(seed) + 211)

        segments = _segment_episodes(terminals=terminals, timeouts=timeouts)
        if num_episodes is not None:
            segments = segments[: int(num_episodes)]

        self.episodes_tokens: list[np.ndarray] = []
        self.episode_rewards: list[np.ndarray] = []

        transition_counts: dict[tuple[int, int], dict[tuple[int, int, int, int, int], int]] = {}
        start_state_counts: dict[int, int] = {}
        unsafe_state_ids: set[int] = set()
        goal_state_ids: set[int] = set()
        reward_sum = 0.0
        reward_count = 0

        for start, end in segments:
            end = min(int(end), start + int(max_steps))
            if end <= start:
                continue
            rows = []
            ep_rewards = []
            for i in range(start, end):
                s = int(state_ids[i])
                a = int(action_ids[i])
                r = float(rewards[i])
                c = float(costs[i])
                reward_tok = 1 if r > self.reward_goal_threshold else 0
                cost_tok = 1 if (c > 0.0 and c >= self.cost_unsafe_threshold) else 0
                rows.append(
                    make_transition_row(
                        schema=self.token_schema,
                        state=s,
                        action=a,
                        reward_token=reward_tok,
                        safety_cost=cost_tok,
                    )
                )
                ep_rewards.append(r)

                is_last = (i == end - 1)
                ns = int(state_ids[i + 1]) if (i + 1) < n and not is_last else s
                done = 1 if is_last else 0
                goal = 1 if reward_tok == 1 else 0
                if cost_tok == 1:
                    unsafe_state_ids.add(ns)
                if goal == 1:
                    goal_state_ids.add(ns)
                trans_key = (s, a)
                out_key = (ns, int(round(r * 1000.0)), cost_tok, done, goal)
                bucket = transition_counts.setdefault(trans_key, {})
                bucket[out_key] = int(bucket.get(out_key, 0) + 1)
                reward_sum += r
                reward_count += 1

            if not rows:
                continue
            start_state_counts[int(rows[0][0])] = int(start_state_counts.get(int(rows[0][0]), 0) + 1)
            tokens = np.asarray(rows, dtype=np.int64).reshape(-1, self.token_schema.width)
            end_row = build_end_row(self.token_schema, get_end_token_id(self.token_schema, int(state_bins), int(action_bins)))
            tokens = np.vstack([tokens, end_row])
            self.episodes_tokens.append(tokens)
            self.episode_rewards.append(np.asarray(ep_rewards, dtype=np.float32))

        if not self.episodes_tokens:
            raise RuntimeError("No episodes could be built from DSRL dataset.")

        self.num_bins_per_dim = get_num_bins_per_dim(
            self.token_schema, int(state_bins), int(max(1, int(action_ids.max()) + 1))
        )
        self.end_token_id = get_end_token_id(
            self.token_schema, int(state_bins), int(max(1, int(action_ids.max()) + 1))
        )

        for ep in self.episodes_tokens:
            # Rebuild END row with finalized end_token_id if max(action)+1 changed.
            ep[-1] = build_end_row(self.token_schema, self.end_token_id)
            validate_episode_tokens(
                ep,
                schema=self.token_schema,
                end_token_id=self.end_token_id,
                observation_space_n=int(state_bins),
                action_space_n=int(max(1, int(action_ids.max()) + 1)),
            )

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

        self.env = DSRLReplayEnv(
            num_states=int(state_bins),
            num_actions=int(max(1, int(action_ids.max()) + 1)),
            transitions=transitions,
            start_state_probs=start_state_probs,
            unsafe_state_ids=unsafe_state_ids,
            goal_state_ids=goal_state_ids,
            cfg=DSRLReplayConfig(max_steps=int(max_steps), stochastic=True),
            default_reward=float(reward_sum / max(1, reward_count)),
        )
        self.env_name = "dsrl"
        self.dataset_path = str(path)

        self.rows_per_seg = max(1, self.sequence_length // self.token_schema.width)
        self.required_rows = (
            self.rows_per_seg + 1 if self.target_shift == "transition" else max(2, self.rows_per_seg)
        )
        self.indices: list[tuple[int, int]] = []
        for ep_idx, rows in enumerate(self.episodes_tokens):
            n_rows = int(rows.shape[0])
            if n_rows < self.required_rows:
                self.indices.append((ep_idx, 0))
                continue
            starts = list(range(0, max(1, n_rows - self.required_rows), self.rows_per_seg))
            tail = n_rows - self.required_rows
            if tail not in starts:
                starts.append(tail)
            for start_row in starts:
                self.indices.append((ep_idx, int(start_row)))

        self.observation_dim = 1
        self.action_dim = 1
        self.joined_dim = self.token_schema.width

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
