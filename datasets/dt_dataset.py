from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DTBatchSpec:
    context_len: int
    pad_action_id: int
    ignore_index: int = -100


def _compute_rtg(rewards: np.ndarray) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running += float(rewards[i])
        rtg[i] = running
    return rtg


class DTSequenceDataset(Dataset):
    """
    Thin DT adapter on top of existing offline episode datasets.

    It consumes episode-level tokens/rewards and returns fixed-length windows:
      states, prev_actions, rtg, timesteps, target_actions, mask
    """

    def __init__(
        self,
        episodes_tokens: list[np.ndarray],
        episode_rewards: list[np.ndarray],
        context_len: int,
        num_actions: int,
        state_index: int = 0,
        action_index: int = 1,
    ):
        if len(episodes_tokens) != len(episode_rewards):
            raise ValueError("episodes_tokens and episode_rewards length mismatch")

        self.spec = DTBatchSpec(context_len=int(context_len), pad_action_id=int(num_actions))
        self.num_actions = int(num_actions)
        self.state_index = int(state_index)
        self.action_index = int(action_index)

        self.episodes: list[dict] = []
        self.indices: list[tuple[int, int]] = []

        for ep_idx, (tokens, rewards) in enumerate(zip(episodes_tokens, episode_rewards)):
            if tokens.ndim != 2 or tokens.shape[0] < 2:
                continue

            transitions = tokens[:-1]  # strip explicit END row
            states = transitions[:, self.state_index].astype(np.int64)
            actions = transitions[:, self.action_index].astype(np.int64)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

            T = len(actions)
            if T == 0:
                continue
            if len(rewards) != T:
                raise ValueError(f"Episode {ep_idx} rewards length {len(rewards)} != transitions {T}")

            rtg = _compute_rtg(rewards)
            self.episodes.append(
                {
                    "states": states,
                    "actions": actions,
                    "rtg": rtg,
                }
            )
            cur_idx = len(self.episodes) - 1
            for t in range(T):
                self.indices.append((cur_idx, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        ep = self.episodes[ep_idx]

        states = ep["states"]
        actions = ep["actions"]
        rtg = ep["rtg"]

        start = max(0, t - self.spec.context_len + 1)
        end = t + 1

        state_seg = states[start:end]
        action_seg = actions[start:end]
        rtg_seg = rtg[start:end]
        step_seg = np.arange(start, end, dtype=np.int64)

        prev_action_seg = np.empty_like(action_seg)
        prev_action_seg[0] = self.spec.pad_action_id if start == 0 else int(actions[start - 1])
        if len(prev_action_seg) > 1:
            prev_action_seg[1:] = action_seg[:-1]

        L = len(action_seg)
        pad = self.spec.context_len - L

        states_out = np.zeros(self.spec.context_len, dtype=np.int64)
        prev_actions_out = np.full(self.spec.context_len, self.spec.pad_action_id, dtype=np.int64)
        rtg_out = np.zeros(self.spec.context_len, dtype=np.float32)
        steps_out = np.zeros(self.spec.context_len, dtype=np.int64)
        targets_out = np.full(self.spec.context_len, self.spec.ignore_index, dtype=np.int64)
        mask_out = np.zeros(self.spec.context_len, dtype=np.float32)

        states_out[pad:] = state_seg
        prev_actions_out[pad:] = prev_action_seg
        rtg_out[pad:] = rtg_seg
        steps_out[pad:] = step_seg
        targets_out[pad:] = action_seg
        mask_out[pad:] = 1.0

        return (
            torch.from_numpy(states_out),
            torch.from_numpy(prev_actions_out),
            torch.from_numpy(rtg_out),
            torch.from_numpy(steps_out),
            torch.from_numpy(targets_out),
            torch.from_numpy(mask_out),
        )
