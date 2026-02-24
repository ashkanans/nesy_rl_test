from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older setups
    import gym


_MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


@dataclass
class FrozenLakeConfig:
    map_size: str = "4x4"
    is_slippery: bool = False
    max_steps: int = 100


class FrozenLakeEnv:
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    def __init__(self, config: FrozenLakeConfig | None = None):
        self.cfg = config or FrozenLakeConfig()
        if self.cfg.map_size not in _MAPS:
            raise ValueError(f"Unsupported FrozenLake map_size '{self.cfg.map_size}'")

        self.desc_rows = list(_MAPS[self.cfg.map_size])
        self.n_rows = len(self.desc_rows)
        self.n_cols = len(self.desc_rows[0])
        self.n_states = self.n_rows * self.n_cols

        self._gym_env = gym.make(
            "FrozenLake-v1",
            desc=self.desc_rows,
            is_slippery=bool(self.cfg.is_slippery),
            max_episode_steps=int(self.cfg.max_steps),
        )
        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

        self.start_pos = self._find_unique("S")
        self.goal_positions = set(self._find_all("G"))
        self.hole_positions = set(self._find_all("H"))
        self.hole_state_ids = sorted(self._pos_to_state(p) for p in self.hole_positions)
        self.goal_state_ids = sorted(self._pos_to_state(p) for p in self.goal_positions)

        self._steps = 0
        self._cur_state = self._pos_to_state(self.start_pos)

    def _find_all(self, symbol: str):
        out = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.desc_rows[r][c] == symbol:
                    out.append((r, c))
        return out

    def _find_unique(self, symbol: str):
        positions = self._find_all(symbol)
        if len(positions) != 1:
            raise ValueError(f"Expected one '{symbol}', found {len(positions)}")
        return positions[0]

    def _pos_to_state(self, pos):
        r, c = pos
        return int(r * self.n_cols + c)

    def _state_to_pos(self, state: int):
        return divmod(int(state), self.n_cols)

    def _cell_at_state(self, state: int):
        r, c = self._state_to_pos(state)
        return self.desc_rows[r][c]

    def reset(self, seed=None, options=None):
        self._steps = 0
        out = self._gym_env.reset(seed=seed, options=options)
        if isinstance(out, tuple):
            obs, info = out
        else:  # pragma: no cover - old gym fallback
            obs, info = out, {}
        self._cur_state = int(obs)
        return int(obs), dict(info)

    def step(self, action):
        self._steps += 1
        out = self._gym_env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:  # pragma: no cover - old gym fallback
            obs, reward, done, info = out
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)

        obs = int(obs)
        self._cur_state = obs
        info = dict(info)

        cell = self._cell_at_state(obs)
        if terminated:
            if cell == "G":
                info["terminal_type"] = "G"
            elif cell == "H":
                info["terminal_type"] = "H"
        if truncated:
            info["truncated"] = True

        return obs, float(reward), bool(done), info

    def is_hole_state(self, state: int) -> bool:
        return int(state) in set(self.hole_state_ids)

    def is_goal_state(self, state: int) -> bool:
        return int(state) in set(self.goal_state_ids)

    def get_ap_labels(self, state: int):
        sid = int(state)
        return {
            "hole": self.is_hole_state(sid),
            "goal": self.is_goal_state(sid),
        }

    def render(self):
        # Keep lightweight text rendering independent from gym backend modes.
        lines = []
        cur_pos = self._state_to_pos(self._cur_state)
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == cur_pos:
                    row.append("A")
                else:
                    row.append(self.desc_rows[r][c])
            lines.append(" ".join(row))
        return "\n".join(lines)


def compute_frozenlake_shortest_path_policy(desc_rows):
    n_rows = len(desc_rows)
    n_cols = len(desc_rows[0])

    def sid(pos):
        return pos[0] * n_cols + pos[1]

    def is_walkable(pos):
        cell = desc_rows[pos[0]][pos[1]]
        return cell != "H"

    action_deltas = {
        FrozenLakeEnv.ACTION_UP: (-1, 0),
        FrozenLakeEnv.ACTION_RIGHT: (0, 1),
        FrozenLakeEnv.ACTION_DOWN: (1, 0),
        FrozenLakeEnv.ACTION_LEFT: (0, -1),
    }

    goal_positions = []
    states = []
    for r in range(n_rows):
        for c in range(n_cols):
            pos = (r, c)
            if is_walkable(pos):
                states.append(pos)
            if desc_rows[r][c] == "G":
                goal_positions.append(pos)

    if not goal_positions:
        return {}

    # Reverse BFS distance-to-goal on walkable cells.
    from collections import deque

    q = deque(goal_positions)
    dist = {g: 0 for g in goal_positions}
    while q:
        pos = q.popleft()
        for dr, dc in action_deltas.values():
            prv = (pos[0] + dr, pos[1] + dc)
            if not (0 <= prv[0] < n_rows and 0 <= prv[1] < n_cols):
                continue
            if not is_walkable(prv):
                continue
            if prv not in dist:
                dist[prv] = dist[pos] + 1
                q.append(prv)

    policy = {}
    for pos in states:
        if pos not in dist:
            continue
        if desc_rows[pos[0]][pos[1]] == "G":
            continue
        best = None
        best_next_dist = None
        for action, (dr, dc) in action_deltas.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols):
                continue
            if not is_walkable(nxt):
                continue
            if nxt not in dist:
                continue
            d = dist[nxt]
            if best_next_dist is None or d < best_next_dist:
                best_next_dist = d
                best = action
        if best is not None:
            policy[sid(pos)] = int(best)
    return policy
