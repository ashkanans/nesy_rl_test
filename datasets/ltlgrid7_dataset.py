import re
from pathlib import Path

import numpy as np

from datasets.ltlgrid_dataset import LTLGridSequenceDataset
from envs.ltlgrid7_env import EMPTY, LTLGrid7Config, LTLGrid7Env
from logic.token_schema import (
    build_end_row,
    get_end_token_id,
    get_num_bins_per_dim,
    get_schema_for_env,
    make_transition_row,
    validate_episode_tokens,
)

NUM_OBJECTS = 4  # pickaxe, lava, door, gem


def layout_state_features(layouts, size=7):
    """
    [L * size^2, 5 * size^2] float32: the symbolic observation of Grid7 for every (layout, cell) state
    token: planes 0-3 mark the pickaxe, lava, door and gem cells, plane 4 the agent. Row s is the
    observation of state token s = layout * size^2 + x * size + y.
    """
    L = layouts.shape[0]
    n = size * size
    objects = np.stack([(layouts == o) for o in range(NUM_OBJECTS)], axis=1).reshape(L, NUM_OBJECTS * n)
    feats = np.zeros((L, n, (NUM_OBJECTS + 1) * n), dtype=np.float32)
    feats[:, :, : NUM_OBJECTS * n] = objects[:, None, :]
    feats[:, np.arange(n), NUM_OBJECTS * n + np.arange(n)] = 1.0
    return feats.reshape(L * n, (NUM_OBJECTS + 1) * n)


class LTLGrid7SequenceDataset(LTLGridSequenceDataset):
    """
    Offline dataset for LTLGrid7Env from a Grid7 "fixed50" H5 file
    (Dataset/<reward>/task<N>/dataset_<split>_fixed50.h5).

    Each trajectory holds its 7x7 layout (grid, indexed [x, y]), the agent positions (T + 1 rows: the
    position before every action plus the final one), actions, rewards, terminals and returns_to_go; the
    group attribute `layout` is the index of the layout. The state token of a step is
    layout * 49 + x * 7 + y (see LTLGrid7Env), so the sequence/DT code of the 4x4 pipeline is used as is.

    On load every trajectory is replayed through the env's dynamics and the task's reward machine; the
    stored positions, rewards and terminal flags must match. If the matching test_<split>_fixed50.npz
    exists, its 500 episodes (layout, start) are given to the env, so rollouts follow the fixed protocol.

    Extra attributes on top of LTLGridSequenceDataset:
      layouts          [L, 7, 7] symbol of every cell, one entry per layout index in the file
      episode_layout   layout index of every trajectory
      state_features   [L * 49, 245] symbolic observation of every state token (for the DT encoder)
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
        subsample_seed=None,
        expert_only=False,
        max_steps=100,
    ):
        import h5py

        self.h5_path = str(h5_path)
        self.sequence_length = sequence_length
        self.target_shift = str(target_shift)
        if self.target_shift not in {"token", "transition"}:
            raise ValueError("target_shift must be 'token' or 'transition'.")
        self.reward_scale = float(reward_scale)
        m = re.search(r"task(\d+)", self.h5_path)
        self.task_id = int(task_id) if task_id is not None else (int(m.group(1)) - 1 if m else None)
        if reward_type is None:
            reward_type = next((r for r in ("three_value", "distance") if r in self.h5_path), "distance")
        self.reward_type = str(reward_type)
        self.split = next((s for s in ("medium_expert", "medium_replay", "medium") if f"dataset_{s}" in self.h5_path), None)

        with h5py.File(self.h5_path, "r") as f:
            all_keys = sorted(f.keys(), key=int)
            keys = all_keys
            if expert_only:
                keys = [k for k in keys if f[k].attrs["kind"] == "optimal"]
            if max_episodes is not None:
                rng = np.random.RandomState(int(subsample_seed or 0))
                pick = rng.choice(len(keys), size=min(int(max_episodes), len(keys)), replace=False)
                keys = [keys[i] for i in sorted(pick)]
            # The layout table covers the whole file, also when only a subset of the games is used, so
            # that state tokens, the DT's feature table and the test episodes keep the same meaning.
            n_layouts = max(int(f[k].attrs["layout"]) for k in all_keys) + 1
            layouts = np.full((n_layouts, 7, 7), -1, dtype=np.int64)
            for k in all_keys:  # one grid per layout index, identical in all its trajectories
                lid, grid = int(f[k].attrs["layout"]), np.asarray(f[k]["grid"][:], dtype=np.int64)
                if layouts[lid, 0, 0] == -1:
                    layouts[lid] = grid
                elif not np.array_equal(layouts[lid], grid):
                    raise ValueError(f"Episode {k}: layout {lid} differs from the grid seen before for that layout.")
            if (layouts < 0).any():
                raise ValueError(f"{self.h5_path}: some layout indices below {n_layouts} never occur.")
            self.layouts = layouts

            test_layouts = test_starts = None
            test_path = Path(self.h5_path).with_name(Path(self.h5_path).name.replace("dataset_", "test_").replace(".h5", ".npz"))
            if test_path.exists():
                t = np.load(test_path)
                if not np.array_equal(layouts[t["layout"]], t["grid"]):
                    raise ValueError(f"{test_path}: test grids do not match the layout table of the training file.")
                test_layouts, test_starts = t["layout"], t["start"]
            self.env = LTLGrid7Env(LTLGrid7Config(
                layouts, max_steps=max_steps, task_id=self.task_id, reward_type=self.reward_type,
                reward_scale=self.reward_scale, test_layouts=test_layouts, test_starts=test_starts))
            self.token_schema = get_schema_for_env("ltlgrid")
            self.schema_id = self.token_schema.schema_id
            n_obs, n_act = self.env.observation_space.n, self.env.action_space.n
            self.num_bins_per_dim = get_num_bins_per_dim(self.token_schema, n_obs, n_act)
            self.end_token_id = get_end_token_id(self.token_schema, n_obs, n_act)
            machine = self.env.machine

            episodes_tokens, episode_rewards, episode_transitions = [], [], []
            episode_symbols, dfa_state_ids, self.episode_layout, self.episode_start_pos = [], [], [], []
            self.num_truncated_episodes = 0
            self.num_final_observation_checked = 0
            self.num_reward_checked = 0
            for key in keys:
                grp = f[key]
                lid = int(grp.attrs["layout"])
                pos = np.asarray(grp["positions"][:], dtype=np.int64)  # [T + 1, 2]
                acts = np.asarray(grp["actions"][:], dtype=np.int64)
                rews = np.asarray(grp["rewards"][:], dtype=np.float64)
                terminals = np.asarray(grp["terminals"][:], dtype=bool)
                T = len(acts)
                if T == 0:
                    continue
                if not (len(pos) == T + 1 and len(rews) == len(terminals) == T):
                    raise ValueError(f"Episode {key} has mismatched array lengths in {self.h5_path}")
                if pos.min() < 0 or pos.max() >= self.env.size or acts.min() < 0 or acts.max() >= n_act:
                    raise ValueError(f"Episode {key}: position or action out of range.")
                next_pos = [self.env.move(pos[t], acts[t]) for t in range(T)]
                symbols = np.array([self.env.symbol_of_pos(lid, p) for p in next_pos], dtype=np.int64)
                if validate_dynamics:
                    for t in range(T):
                        if tuple(pos[t + 1]) != next_pos[t]:
                            raise ValueError(
                                f"Episode {key}, step {t}: positions[t+1]={tuple(pos[t + 1])} but the "
                                f"dynamics give {next_pos[t]}; positions/actions are misaligned.")
                # replay the reward machine: stored rewards, DFA end and terminal flags must agree
                q, qs, terminal_at = machine.initial_state, [], None
                for t in range(T):
                    qs.append(q)
                    nxt = machine.next_state(q, symbols[t])
                    if validate_dynamics and abs(machine.reward(q, nxt) - rews[t]) > 1e-6:
                        raise ValueError(f"Episode {key}, step {t}: stored reward {rews[t]} != machine {machine.reward(q, nxt)}.")
                    q = nxt
                    if machine.is_terminal(q):
                        terminal_at = t
                        break
                self.num_reward_checked += 1
                if terminal_at is not None and terminal_at != T - 1:
                    raise ValueError(f"Episode {key}: the DFA ends at step {terminal_at} but the game has {T} steps.")
                if validate_dynamics and bool(terminals[-1]) != (terminal_at is not None):
                    raise ValueError(f"Episode {key}: terminal flag {bool(terminals[-1])} != DFA end {terminal_at is not None}.")

                states = np.array([self.env.pos_to_state(p, lid) for p in pos[:-1]], dtype=np.int64)
                next_states = np.array([self.env.pos_to_state(p, lid) for p in next_pos], dtype=np.int64)
                tokens = np.zeros((T, self.token_schema.width), dtype=np.int64)
                for t in range(T):
                    tokens[t] = make_transition_row(self.token_schema, int(states[t]), int(acts[t]))
                tokens = np.vstack([tokens, build_end_row(self.token_schema, self.end_token_id)])
                validate_episode_tokens(tokens, schema=self.token_schema, end_token_id=self.end_token_id,
                                        observation_space_n=n_obs, action_space_n=n_act)
                episodes_tokens.append(tokens)
                episode_rewards.append((rews * self.reward_scale).astype(np.float32))
                episode_transitions.append(np.stack([states, acts, next_states, terminals.astype(np.int64)], axis=1))
                episode_symbols.append(symbols)
                dfa_state_ids.append(np.array(qs, dtype=np.int64))
                self.episode_layout.append(lid)
                self.episode_start_pos.append((int(pos[0, 0]), int(pos[0, 1])))

        if not episodes_tokens:
            raise ValueError(f"No episodes found in {self.h5_path}")

        self.rows_per_seg = max(1, sequence_length // self.token_schema.width)
        self.required_rows = self.rows_per_seg + 1 if self.target_shift == "transition" else max(2, self.rows_per_seg)
        indices = []
        for ep_idx, rows in enumerate(episodes_tokens):
            R = rows.shape[0]
            if R < self.required_rows:
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
        self.upstream_dfa_state_ids = dfa_state_ids
        self.indices = indices
        self.state_features = layout_state_features(self.layouts, self.env.size)

        self.observation_dim = 1
        self.action_dim = 1
        self.joined_dim = self.token_schema.width
