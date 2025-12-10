import numpy as np


class NRMSafetyNavConfig:
    def __init__(
        self,
        grid=None,
        step_reward=-0.01,
        unsafe_reward=-1.0,
        goal_reward=1.0,
        max_steps=200,
        stochastic=False,
        obstacle_punish=True,
    ):
        # Default grid inspired by simple navigation tasks: S=start, G=goal, X=unsafe, #=wall
        self.grid = grid or [
            [".", ".", ".", ".", "G"],
            [".", "#", "#", ".", "."],
            [".", "X", ".", ".", "."],
            ["S", ".", ".", "X", "."],
            [".", ".", ".", ".", "."],
        ]
        self.step_reward = step_reward
        self.unsafe_reward = unsafe_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.obstacle_punish = obstacle_punish


class NRMSafetyNavEnv:
    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, config=None):
        self.cfg = config or NRMSafetyNavConfig()
        self.grid = self.cfg.grid
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols

        self.start_pos = self._find_unique("S")
        self.goal_positions = set(self._find_all("G"))
        self.unsafe_positions = set(self._find_all("X"))
        self.wall_positions = set(self._find_all("#"))

        self.observation_space = type("Discrete", (), {"n": self.n_states})
        self.action_space = type("Discrete", (), {"n": len(self.ACTIONS)})

        self._cur_pos = self.start_pos
        self._steps = 0

    def _find_all(self, symbol):
        coords = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == symbol:
                    coords.append((r, c))
        return coords

    def _find_unique(self, symbol):
        coords = self._find_all(symbol)
        if len(coords) != 1:
            raise ValueError(f"Expected one '{symbol}', found {len(coords)}")
        return coords[0]

    def _pos_to_state(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def _state_to_pos(self, s):
        return divmod(s, self.n_cols)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._cur_pos = self.start_pos
        self._steps = 0
        return self._pos_to_state(self._cur_pos), {}

    def step(self, action):
        self._steps += 1

        a = int(action)
        if self.cfg.stochastic:
            r = np.random.rand()
            if r > 0.8:
                if r < 0.9:
                    a = (a - 1) % 4
                else:
                    a = (a + 1) % 4

        dr, dc = self.ACTIONS[a]
        r, c = self._cur_pos
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and (nr, nc) not in self.wall_positions:
            self._cur_pos = (nr, nc)

        reward = self.cfg.step_reward
        done = False
        info = {}

        if self._cur_pos in self.goal_positions:
            reward += self.cfg.goal_reward
            done = True
            info["terminal_type"] = "G"
        elif self._cur_pos in self.unsafe_positions:
            reward += self.cfg.unsafe_reward
            done = True
            info["terminal_type"] = "X"
        elif self.cfg.obstacle_punish and (nr, nc) in self.wall_positions:
            reward -= 0.05

        if self._steps >= self.cfg.max_steps:
            done = True
            info["truncated"] = True

        obs = self._pos_to_state(self._cur_pos)
        return obs, float(reward), done, info

    def get_ap_labels(self, state):
        pos = self._state_to_pos(state)
        labels = {"safe": True, "unsafe": False, "goal": False, "near_obstacle": False}
        if pos in self.unsafe_positions:
            labels["unsafe"] = True
            labels["safe"] = False
        if pos in self.goal_positions:
            labels["goal"] = True
        # near obstacle if any wall or unsafe within manhattan distance 1
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nbr = (pos[0]+dr, pos[1]+dc)
            if nbr in self.wall_positions or nbr in self.unsafe_positions:
                labels["near_obstacle"] = True
                break
        return labels

    def render(self):
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == self._cur_pos:
                    row.append("A")
                else:
                    cell = self.grid[r][c]
                    if cell == ".":
                        row.append(".")
                    else:
                        row.append(cell)
            lines.append(" ".join(row))
        return "\n".join(lines)
