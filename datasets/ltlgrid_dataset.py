import re

import numpy as np
import torch
from torch.utils.data import Dataset

from envs.ltlgrid_env import LTLGridConfig, LTLGridEnv
from logic.token_schema import (
    build_end_row,
    get_end_token_id,
    get_num_bins_per_dim,
    get_schema_for_env,
    make_transition_row,
    validate_episode_tokens,
)


class LTLGridSequenceDataset(Dataset):
    """
    Offline dataset for LTLGridEnv, loaded from a Neurosymbolic-DT-eval style H5 file
    (Results/Symbolic/<reward_type>/task<N>/datasets/dataset_*.h5).

    The env carries the task's reward machine (task_id, 0-based) so rollouts use the same reward
    and scale as the dataset.

    Per trajectory the H5 file holds observations (x, y), actions, rewards, returns_to_go
    and the one-hot dfa_states. observations[i] is the state before actions[i] and
    rewards[i] is the reward after it. That is the same alignment as the token rows
    [state_t, action_t, ...] and the DT rewards, so no shifting is applied; it is checked
    against the grid dynamics on load.

    Rewards are multiplied by reward_scale (default 0.01, the /100 used by the original DT).
    expert_only keeps only the first half of the file, the noise-free optimal trajectories.
    num_starts keeps only the trajectories starting in that many (randomly chosen) start cells.
    max_episodes keeps the first N trajectories, or a random N when subsample_seed is given.

    With truncate_after_terminal (default) trajectories are cut at the first step where the DFA
    reaches its accepting or failing absorbing state; num_truncated_episodes counts the cut ones.

    Extra per-episode annotations:
      dfa_state_ids   [T] DFA state before each action, in the numbering of the reward machine
                      (the stored one-hot, a different numbering, is kept in upstream_dfa_state_ids)
      episode_symbols [T] symbol id emitted by each step, i.e. of the cell reached by (s_t, a_t)
    """

    def __init__(
        self,
        h5_path,
        sequence_length=80,
        target_shift="token",
        reward_scale=0.01,
        max_episodes=None,
        validate_dynamics=True,
        task_id=None,
        reward_type=None,
        truncate_after_terminal=True,
        subsample_seed=None,
        num_starts=None,
        expert_only=False,
    ):
        import h5py  # optional dependency (requirements-env.txt); only needed to read H5 files

        self.h5_path = str(h5_path)
        self.sequence_length = sequence_length
        self.target_shift = str(target_shift)
        if self.target_shift not in {"token", "transition"}:
            raise ValueError("target_shift must be 'token' or 'transition'.")
        self.reward_scale = float(reward_scale)
        # task_id is 0-based; by default it is read from a "task<N>" component of the path
        m = re.search(r"task(\d+)", self.h5_path)
        self.task_id = int(task_id) if task_id is not None else (int(m.group(1)) - 1 if m else None)
        # reward_type is read from a "distance" / "three_value" path component by default
        if reward_type is None:
            reward_type = next((r for r in ("three_value", "distance") if r in self.h5_path), "distance")
        self.reward_type = str(reward_type)

        self.token_schema = get_schema_for_env("ltlgrid")
        self.schema_id = self.token_schema.schema_id
        self.env = LTLGridEnv(
            LTLGridConfig(task_id=self.task_id, reward_type=self.reward_type, reward_scale=self.reward_scale)
        )
        n_obs, n_act = self.env.observation_space.n, self.env.action_space.n
        self.num_bins_per_dim = get_num_bins_per_dim(self.token_schema, n_obs, n_act)
        self.end_token_id = get_end_token_id(self.token_schema, n_obs, n_act)

        episodes_tokens = []
        self.num_truncated_episodes = 0
        self.num_final_observation_checked = 0
        episode_rewards = []
        episode_transitions = []
        episode_symbols = []
        dfa_state_ids = []
        upstream_dfa_state_ids = []
        self.episode_start_pos = []

        with h5py.File(self.h5_path, "r") as f:
            keys = sorted(f.keys(), key=int)
            if expert_only:
                # dataset_generation writes the optimal (Dijkstra) trajectories first and the noisy
                # ones in the second half of the file
                keys = keys[: len(keys) // 2]
            if num_starts:
                # keep only the trajectories starting in a random subset of the start cells, to
                # study generalisation to start states never seen in training
                rng = np.random.RandomState(int(subsample_seed or 0))
                starts = sorted({tuple(int(v) for v in f[k].attrs["start_pos"]) for k in keys})
                kept = {starts[i] for i in rng.choice(len(starts), size=min(int(num_starts), len(starts)), replace=False)}
                keys = [k for k in keys if tuple(int(v) for v in f[k].attrs["start_pos"]) in kept]
                self.train_start_positions = sorted(kept)
            if max_episodes is not None:
                if subsample_seed is None:
                    keys = keys[: int(max_episodes)]
                else:
                    # the file is ordered (optimal paths first, noisy ones after), so a prefix
                    # is not representative: draw a random subset instead
                    rng = np.random.RandomState(int(subsample_seed))
                    pick = rng.choice(len(keys), size=min(int(max_episodes), len(keys)), replace=False)
                    keys = [keys[i] for i in sorted(pick)]
            for key in keys:
                grp = f[key]
                obs = grp["observations"][:]
                acts = np.asarray(grp["actions"][:], dtype=np.int64)
                rews = np.asarray(grp["rewards"][:], dtype=np.float64)
                dfa = np.asarray(grp["dfa_states"][:]).argmax(axis=1).astype(np.int64)
                terminals = np.asarray(grp["terminals"][:], dtype=bool)
                T = len(acts)
                if T == 0:
                    continue
                if not (len(obs) == len(rews) == len(dfa) == len(terminals) == T):
                    raise ValueError(f"Episode {key} has mismatched array lengths in {self.h5_path}")
                if obs.shape[1] != 2 or not np.array_equal(obs, np.round(obs)):
                    raise ValueError(f"Episode {key}: observations must be integer (x, y) pairs.")
                pos = obs.astype(np.int64)
                if pos.min() < 0 or pos.max() >= self.env.size:
                    raise ValueError(f"Episode {key}: position outside the {self.env.size}x{self.env.size} grid.")
                if acts.min() < 0 or acts.max() >= n_act:
                    raise ValueError(f"Episode {key}: action outside [0, {n_act - 1}].")

                states = pos[:, 0] * self.env.size + pos[:, 1]
                next_pos = [self.env.move(pos[t], acts[t]) for t in range(T)]
                if validate_dynamics:
                    for t in range(T - 1):
                        if tuple(pos[t + 1]) != next_pos[t]:
                            raise ValueError(
                                f"Episode {key}, step {t}: observations[t+1]={tuple(pos[t + 1])} "
                                f"but dynamics give {next_pos[t]}; observations/actions are misaligned."
                            )

                # Newer files store the state reached by the last action; it must agree with the dynamics.
                if "final_observation" in grp:
                    final = np.asarray(grp["final_observation"][:]).astype(np.int64)
                    if tuple(final) != tuple(next_pos[-1]):
                        raise ValueError(
                            f"Episode {key}: final_observation {tuple(final)} != {tuple(next_pos[-1])} "
                            "reached by the last action."
                        )
                    self.num_final_observation_checked += 1

                # Some stored trajectories keep acting after the DFA reached its accepting/failing
                # absorbing state (noise episodes); the env ends there, so cut them to match.
                if truncate_after_terminal and self.env.machine is not None:
                    q = self.env.machine.initial_state
                    for t in range(T):
                        q = self.env.machine.next_state(q, self.env.symbol_of_pos(next_pos[t]))
                        if self.env.machine.is_terminal(q):
                            break
                    else:
                        t = T - 1
                    if t + 1 < T:
                        T = t + 1
                        states, acts, rews, dfa = states[:T], acts[:T], rews[:T], dfa[:T]
                        terminals, next_pos = terminals[:T].copy(), next_pos[:T]
                        terminals[-1] = True
                        self.num_truncated_episodes += 1
                next_states = np.array([self.env.pos_to_state(p) for p in next_pos], dtype=np.int64)

                tokens = np.zeros((T, self.token_schema.width), dtype=np.int64)
                for t in range(T):
                    tokens[t] = make_transition_row(self.token_schema, int(states[t]), int(acts[t]))
                tokens = np.vstack([tokens, build_end_row(self.token_schema, self.end_token_id)])
                validate_episode_tokens(
                    tokens,
                    schema=self.token_schema,
                    end_token_id=self.end_token_id,
                    observation_space_n=n_obs,
                    action_space_n=n_act,
                )

                episodes_tokens.append(tokens)
                episode_rewards.append((rews * self.reward_scale).astype(np.float32))
                episode_transitions.append(
                    np.stack([states, acts, next_states, terminals.astype(np.int64)], axis=1)
                )
                symbols = np.array([self.env.symbol_of_pos(p) for p in next_pos], dtype=np.int64)
                episode_symbols.append(symbols)
                upstream_dfa_state_ids.append(dfa)
                if self.env.machine is not None:
                    # canonical ids of our reward machine (the stored ones use another numbering)
                    q, qs = self.env.machine.initial_state, []
                    for sym in symbols:
                        qs.append(q)
                        q = self.env.machine.next_state(q, sym)
                    dfa_state_ids.append(np.array(qs, dtype=np.int64))
                else:
                    dfa_state_ids.append(dfa)
                self.episode_start_pos.append(tuple(int(v) for v in grp.attrs["start_pos"]))

        if not episodes_tokens:
            raise ValueError(f"No episodes found in {self.h5_path}")

        self.rows_per_seg = max(1, sequence_length // self.token_schema.width)
        self.required_rows = (
            self.rows_per_seg + 1 if self.target_shift == "transition" else max(2, self.rows_per_seg)
        )
        indices = []
        for ep_idx, rows in enumerate(episodes_tokens):
            R = rows.shape[0]
            if R < self.required_rows:
                # Episodes are short (2-26 steps): keep them and pad with END rows in __getitem__.
                indices.append((ep_idx, 0))
                continue
            starts = list(range(0, max(1, R - self.required_rows), self.rows_per_seg))
            tail = R - self.required_rows
            if tail not in starts:
                starts.append(tail)
            for start_row in starts:
                indices.append((ep_idx, start_row))

        self.episodes_tokens = episodes_tokens
        self.episode_rewards = episode_rewards
        self.episode_transitions = episode_transitions
        self.episode_symbols = episode_symbols
        self.dfa_state_ids = dfa_state_ids
        self.upstream_dfa_state_ids = upstream_dfa_state_ids
        self.indices = indices

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
