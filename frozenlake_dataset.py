from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dfa_adapter import TTDFAAdapter
from frozenlake_env import FrozenLakeConfig, FrozenLakeEnv, compute_frozenlake_shortest_path_policy


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
    ):
        self.sequence_length = int(sequence_length)
        self.discount = float(discount)
        self.policy_mix = float(policy_mix)

        cfg = FrozenLakeConfig(
            map_size=str(map_size),
            is_slippery=bool(is_slippery),
            max_steps=int(max_steps),
        )
        self.env = FrozenLakeEnv(cfg)
        self.num_bins_per_dim = TTDFAAdapter.get_num_bins_per_dim_for_env(
            "frozenlake",
            self.env.observation_space.n,
            self.env.action_space.n,
        )
        self.end_token_id = TTDFAAdapter.get_end_token_id_from_num_bins(self.num_bins_per_dim)

        self.scripted_policy = compute_frozenlake_shortest_path_policy(self.env.desc_rows)
        rng = np.random.RandomState(int(seed))

        self.episodes_tokens = []
        for ep in range(int(num_episodes)):
            obs, _ = self.env.reset(seed=int(seed) + ep)
            rows = []

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

                next_obs, _, done, info = self.env.step(action)
                cost = 1 if info.get("terminal_type") == "H" else 0
                rows.append([int(obs), int(action), 0, int(cost)])
                obs = next_obs
                if done:
                    break

            if not rows:
                continue

            tokens = np.asarray(rows, dtype=np.int64)
            end_row = np.array([self.end_token_id, 0, 0, 0], dtype=np.int64)
            tokens = np.vstack([tokens, end_row])
            self.episodes_tokens.append(tokens)

        self.rows_per_seg = max(1, self.sequence_length // 4)
        self.indices = []
        for ep_idx, rows in enumerate(self.episodes_tokens):
            n_rows = rows.shape[0]
            if n_rows < self.rows_per_seg + 1:
                continue
            starts = list(range(0, max(1, n_rows - (self.rows_per_seg + 1)), self.rows_per_seg))
            tail = n_rows - (self.rows_per_seg + 1)
            if tail not in starts:
                starts.append(tail)
            for start_row in starts:
                self.indices.append((ep_idx, start_row))

        self.observation_dim = 1
        self.action_dim = 1
        self.joined_dim = 4

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_row = self.indices[idx]
        rows = self.episodes_tokens[ep_idx]
        seg_rows = rows[start_row : start_row + self.rows_per_seg + 1]
        flat = seg_rows.reshape(-1)
        x = torch.from_numpy(flat[:-4].astype(np.int64))
        y = torch.from_numpy(flat[4:].astype(np.int64))
        mask = torch.ones_like(x, dtype=torch.float32)
        return x, y, mask
