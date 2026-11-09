import numpy as np

from logic.ltlgrid_dfa import LTLGridRewardMachine

# Symbol ids follow the NRM/Susana convention: ['P', 'L', 'D', 'G', 'E'].
SYMBOLS = ("P", "L", "D", "G", "E")
EMPTY = 4


class LTLGrid7Config:
    def __init__(self, layouts, size=7, max_steps=100, task_id=None, reward_type="distance", reward_scale=1.0,
                 test_layouts=None, test_starts=None):
        # layouts: [L, size, size] symbol id of every cell, indexed [x, y], one entry per layout
        self.layouts = np.asarray(layouts, dtype=np.int64)
        self.size = size
        self.max_steps = max_steps
        self.task_id = task_id
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        # optional fixed test episodes: layout index and start cell of each one
        self.test_layouts = None if test_layouts is None else np.asarray(test_layouts, dtype=np.int64)
        self.test_starts = None if test_starts is None else np.asarray(test_starts, dtype=np.int64)


class LTLGrid7Env:
    """
    7x7 version of LTLGridEnv: the object layout changes from game to game (Grid7 of Neurosymbolic-DT-eval,
    "fixed50": a fixed set of layouts, each with 2 copies of pickaxe, lava, door and gem).

    A state token is layout * 49 + x * 7 + y, so one integer carries both the agent cell and the layout.
    The symbol emitted by a step is the object on the cell the agent lands on, as in the 4x4 grid; the
    task's reward machine (logic/ltlgrid_dfa.py) runs alongside exactly as in LTLGridEnv, and the episode
    ends in the accepting ("G") or failing ("X") state or after max_steps.

    reset(seed=...) picks test episode seed % E when fixed test episodes are given, so any E consecutive
    seeds visit each test episode once; otherwise it draws a layout and an empty start cell from the seed.
    """

    ACTIONS = {
        0: (0, 1),  # y + 1
        1: (1, 0),  # x + 1
        2: (0, -1),  # y - 1
        3: (-1, 0),  # x - 1
    }

    def __init__(self, config, seed=0):
        self.cfg = config
        self.size = config.size
        self.n_cells = self.size * self.size
        self.n_layouts = int(config.layouts.shape[0])
        self.layouts = config.layouts
        self.observation_space = type("Discrete", (), {"n": self.n_layouts * self.n_cells})
        self.action_space = type("Discrete", (), {"n": len(self.ACTIONS)})
        self._rng = np.random.RandomState(seed)
        self.machine = None
        if config.task_id is not None:
            self.machine = LTLGridRewardMachine(config.task_id, reward=config.reward_type)
        self._layout = 0
        self._pos = (0, 0)
        self._steps = 0
        self._dfa_state = 0

    # --- state token <-> (layout, position) ---
    def pos_to_state(self, pos, layout):
        return int(layout) * self.n_cells + int(pos[0]) * self.size + int(pos[1])

    def state_to_layout_pos(self, s):
        layout, cell = divmod(int(s), self.n_cells)
        return layout, divmod(cell, self.size)

    def move(self, pos, action):
        dx, dy = self.ACTIONS[int(action)]
        x = min(max(int(pos[0]) + dx, 0), self.size - 1)
        y = min(max(int(pos[1]) + dy, 0), self.size - 1)
        return (x, y)

    def symbol_of_pos(self, layout, pos):
        return int(self.layouts[int(layout), int(pos[0]), int(pos[1])])

    def reset(self, seed=None, options=None, layout=None, start_pos=None):
        cfg = self.cfg
        if layout is None and start_pos is None and cfg.test_layouts is not None and seed is not None:
            i = int(seed) % len(cfg.test_layouts)
            layout, start_pos = int(cfg.test_layouts[i]), tuple(int(v) for v in cfg.test_starts[i])
        if seed is not None and (layout is None or start_pos is None):
            self._rng = np.random.RandomState(seed)
        if layout is None:
            layout = int(self._rng.randint(self.n_layouts))
        if start_pos is None:
            empty = np.argwhere(self.layouts[layout] == EMPTY)
            start_pos = tuple(int(v) for v in empty[self._rng.randint(len(empty))])
        self._layout, self._pos = int(layout), (int(start_pos[0]), int(start_pos[1]))
        self._steps = 0
        self._dfa_state = 0
        return self.pos_to_state(self._pos, self._layout), {}

    def step(self, action):
        self._steps += 1
        self._pos = self.move(self._pos, action)
        symbol = self.symbol_of_pos(self._layout, self._pos)
        info = {"symbol": symbol}
        reward = 0.0
        done = False
        if self.machine is not None:
            nxt = self.machine.next_state(self._dfa_state, symbol)
            reward = self.machine.reward(self._dfa_state, nxt) * self.cfg.reward_scale
            self._dfa_state = nxt
            info["dfa_state"] = nxt
            if self.machine.is_terminal(nxt):
                done = True
                info["terminal_type"] = "G" if self.machine.is_success(nxt) else "X"
        if not done and self._steps >= self.cfg.max_steps:
            done = True
            info["truncated"] = True
        return self.pos_to_state(self._pos, self._layout), float(reward), done, info
