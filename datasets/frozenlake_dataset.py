from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from envs.frozenlake_env import FrozenLakeConfig, FrozenLakeEnv, compute_frozenlake_shortest_path_policy
from logic.token_schema import (
    build_end_row,
    get_end_token_id,
    get_num_bins_per_dim,
    get_schema_for_env,
    make_transition_row,
    validate_episode_tokens,
)


class FrozenLakeSequenceDataset(Dataset):
    """
    Offline FrozenLake dataset with canonical 4-token transition rows.

    Each transition row is:
      [state, action, reward_placeholder, cost]
    where cost=1 iff the transition ended in a hole.
    """

    def __init__(
        self,
        num_episodes=5000,
        max_steps=100,
        sequence_length=200,
        discount=0.99,
        seed=0,
        map_size="4x4",
        is_slippery=False,
        policy_mix=0.0,
        target_shift="token",
    ):
        self.sequence_length = int(sequence_length)
        self.discount = float(discount)
        self.policy_mix = float(policy_mix)
        self.target_shift = str(target_shift)
        if self.target_shift not in {"token", "transition"}:
            raise ValueError("target_shift must be 'token' or 'transition'.")
        self.token_schema = get_schema_for_env("frozenlake")
        self.schema_id = self.token_schema.schema_id

        cfg = FrozenLakeConfig(
            map_size=str(map_size),
            is_slippery=bool(is_slippery),
            max_steps=int(max_steps),
        )
        self.env = FrozenLakeEnv(cfg)
        self.num_bins_per_dim = get_num_bins_per_dim(
            self.token_schema,
            self.env.observation_space.n,
            self.env.action_space.n,
        )
        self.end_token_id = get_end_token_id(
            self.token_schema,
            self.env.observation_space.n,
            self.env.action_space.n,
        )

        self.scripted_policy = compute_frozenlake_shortest_path_policy(self.env.desc_rows)
        rng = np.random.RandomState(int(seed))

        self.episodes_tokens = []
        self.episode_rewards = []
        for ep in range(int(num_episodes)):
            obs, _ = self.env.reset(seed=int(seed) + ep)
            rows = []
            rewards = []

            use_scripted = (
                self.policy_mix > 0.0
                and len(self.scripted_policy) > 0
                and float(rng.rand()) < self.policy_mix
            )

            for _ in range(int(max_steps)):
                if use_scripted and int(obs) in self.scripted_policy:
                    action = int(self.scripted_policy[int(obs)])
                else:
                    action = int(rng.randint(self.env.action_space.n))

                next_obs, reward, done, info = self.env.step(action)
                cost = 1 if info.get("terminal_type") == "H" else 0
                rows.append(
                    make_transition_row(
                        schema=self.token_schema,
                        state=int(obs),
                        action=int(action),
                        reward_token=0,
                        safety_cost=int(cost),
                    )
                )
                rewards.append(float(reward))
                obs = next_obs
                if done:
                    break

            if not rows:
                continue

            tokens = np.asarray(rows, dtype=np.int64).reshape(-1, self.token_schema.width)
            end_row = build_end_row(self.token_schema, self.end_token_id)
            tokens = np.vstack([tokens, end_row])
            validate_episode_tokens(
                tokens,
                schema=self.token_schema,
                end_token_id=self.end_token_id,
                observation_space_n=self.env.observation_space.n,
                action_space_n=self.env.action_space.n,
            )
            self.episodes_tokens.append(tokens)
            self.episode_rewards.append(np.asarray(rewards, dtype=np.float32))

        self.rows_per_seg = max(1, self.sequence_length // self.token_schema.width)
        self.required_rows = (
            self.rows_per_seg + 1 if self.target_shift == "transition" else max(2, self.rows_per_seg)
        )
        self.indices = []
        for ep_idx, rows in enumerate(self.episodes_tokens):
            n_rows = rows.shape[0]
            if n_rows < self.required_rows:
                # Keep short episodes (common for scripted FrozenLake paths) and
                # pad in __getitem__ while masking padded targets from the loss.
                self.indices.append((ep_idx, 0))
                continue
            starts = list(range(0, max(1, n_rows - self.required_rows), self.rows_per_seg))
            tail = n_rows - self.required_rows
            if tail not in starts:
                starts.append(tail)
            for start_row in starts:
                self.indices.append((ep_idx, start_row))

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
