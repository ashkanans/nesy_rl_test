import numpy as np
import torch
from torch.utils.data import Dataset

from nrm_nav_env import NRMSafetyNavEnv, NRMSafetyNavConfig


class NRMSafetySequenceDataset(Dataset):
    """
    Offline dataset for NRMSafetyNavEnv using random rollouts.
    Stores transitions as [state, action, reward, cost] tokens.
    """

    def __init__(
        self,
        num_episodes=1000,
        max_steps=200,
        sequence_length=200,
        discount=0.99,
        stochastic=False,
        seed=0,
        grid=None,
    ):
        self.sequence_length = sequence_length
        self.discount = discount

        cfg = NRMSafetyNavConfig(max_steps=max_steps, stochastic=stochastic, grid=grid)
        self.env = NRMSafetyNavEnv(cfg)

        rng = np.random.RandomState(seed)

        episodes_tokens = []

        for _ in range(num_episodes):
            s, _ = self.env.reset()
            states = []
            actions = []
            rewards = []
            costs = []

            for _ in range(max_steps):
                a = rng.randint(self.env.action_space.n)
                ns, r, done, info = self.env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                costs.append(1 if info.get("terminal_type") == "X" else 0)
                s = ns
                if done:
                    break

            T = len(states)
            if T == 0:
                continue

            values = np.zeros(T, dtype=np.float32)
            g = 0.0
            for t in reversed(range(T)):
                g = rewards[t] + discount * g
                values[t] = g

            tokens = np.zeros((T, 4), dtype=np.int64)
            for t in range(T):
                tokens[t, 0] = int(states[t])
                tokens[t, 1] = int(actions[t])
                tokens[t, 2] = 0  # reward placeholder
                tokens[t, 3] = costs[t]  # cost signal

            # append an end/stop transition
            max_bin = max(self.env.observation_space.n, self.env.action_space.n, 2)
            end_token = max_bin  # aligns with adapter.num_token_ids - 1
            end_row = np.array([end_token] * 4, dtype=np.int64)
            tokens = np.vstack([tokens, end_row])

            episodes_tokens.append(tokens)

        indices = []
        for ep_idx, rows in enumerate(episodes_tokens):
            R = rows.shape[0]
            rows_per_seg = max(1, min(sequence_length // 4, R - 1))
            if R < rows_per_seg + 1:
                continue
            starts = list(range(0, max(1, R - (rows_per_seg + 1)), rows_per_seg))
            tail = R - (rows_per_seg + 1)
            if tail not in starts:
                starts.append(tail)
            for start_row in starts:
                indices.append((ep_idx, start_row))

        self.episodes_tokens = episodes_tokens
        self.indices = indices

        self.observation_dim = 1
        self.action_dim = 1
        self.joined_dim = 4

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        rows = self.episodes_tokens[ep_idx]
        rows_per_seg = max(1, min(self.sequence_length // 4, rows.shape[0] - 1))
        seg_rows = rows[start:start + rows_per_seg + 1]
        flat = seg_rows.reshape(-1)
        x = torch.from_numpy(flat[:-4].astype(np.int64))
        y = torch.from_numpy(flat[4:].astype(np.int64))
        mask = torch.ones_like(x, dtype=torch.float32)
        return x, y, mask
