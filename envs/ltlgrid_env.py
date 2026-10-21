import numpy as np

from logic.ltlgrid_dfa import LTLGridRewardMachine

# Symbol ids follow the NRM/Susana convention: ['P', 'L', 'D', 'G', 'E'].
SYMBOLS = ("P", "L", "D", "G", "E")
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


class LTLGridConfig:
    def __init__(self, size=4, max_steps=50, starts=None, task_id=None, reward_type="distance", reward_scale=1.0):
        self.size = size
        self.max_steps = max_steps
        # task_id is 0-based (index into LTLGRID_FORMULAS); None disables the reward machine.
        self.task_id = task_id
        self.reward_type = reward_type
        # Rewards are multiplied by reward_scale (the DT datasets use 0.01, i.e. returns in [-1.5, 1]).
        self.reward_scale = reward_scale
        # Same start cells as the Neural Reward Machines 4x4 grid: every cell except the object cells.
        self.starts = starts or [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
            (1, 3), (2, 3),
        ]


class LTLGridEnv:
    """
    Dynamics and symbol labelling of the NRM 4x4 "mine" grid (pickaxe, lava, door, gem).

    With cfg.task_id set, the LTLf task's reward machine (logic/ltlgrid_dfa.py) runs alongside the
    grid: step() returns the shaped reward (change of DFA potential * reward_scale) and the episode
    ends in the accepting (info["terminal_type"] == "G") or failing ("X") absorbing state, or when
    max_steps is reached. Without a task, step() returns reward 0.0 and only max_steps ends it.
    info["symbol"] is the symbol emitted by the step and info["dfa_state"] the DFA state after it.
    Positions are (x, y); the state token is x * size + y.
    """

    ACTIONS = {
        0: (0, 1),  # down
        1: (1, 0),  # right
        2: (0, -1),  # up
        3: (-1, 0),  # left
    }
    OBJECT_CELLS = {
        "P": (1, 1),
        "L": (3, 3),
        "D": (3, 0),
        "G": (0, 3),
    }

    def __init__(self, config=None, seed=0):
        self.cfg = config or LTLGridConfig()
        self.size = self.cfg.size
        self.n_states = self.size * self.size

        self.observation_space = type("Discrete", (), {"n": self.n_states})
        self.action_space = type("Discrete", (), {"n": len(self.ACTIONS)})

        self._rng = np.random.RandomState(seed)
        self._cur_pos = self.cfg.starts[0]
        self._steps = 0
        self.machine = None
        if self.cfg.task_id is not None:
            self.machine = LTLGridRewardMachine(self.cfg.task_id, reward=self.cfg.reward_type)
        self._dfa_state = 0

    def pos_to_state(self, pos):
        x, y = pos
        return int(x) * self.size + int(y)

    def state_to_pos(self, s):
        return divmod(int(s), self.size)

    def move(self, pos, action):
        dx, dy = self.ACTIONS[int(action)]
        x = min(max(int(pos[0]) + dx, 0), self.size - 1)
        y = min(max(int(pos[1]) + dy, 0), self.size - 1)
        return (x, y)

    def symbol_of_pos(self, pos):
        pos = (int(pos[0]), int(pos[1]))
        for sym, cell in self.OBJECT_CELLS.items():
            if pos == cell:
                return SYMBOL_TO_ID[sym]
        return SYMBOL_TO_ID["E"]

    def symbol_of_state(self, s):
        return self.symbol_of_pos(self.state_to_pos(s))

    def reset(self, seed=None, options=None, start_pos=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        if start_pos is None:
            start_pos = self.cfg.starts[self._rng.randint(len(self.cfg.starts))]
        self._cur_pos = (int(start_pos[0]), int(start_pos[1]))
        self._steps = 0
        self._dfa_state = 0
        return self.pos_to_state(self._cur_pos), {}

    def step(self, action):
        self._steps += 1
        self._cur_pos = self.move(self._cur_pos, action)
        symbol = self.symbol_of_pos(self._cur_pos)
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
        return self.pos_to_state(self._cur_pos), float(reward), done, info
