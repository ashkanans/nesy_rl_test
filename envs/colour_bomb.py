import numpy as np
from collections import deque

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:

        class spaces:
            class Discrete:
                def __init__(self, n):
                    self.n = n

        class gym:
            Env = object


class CBConfig:
    def __init__(
        self, step_reward=-0.01, bomb_reward=-1.0, goal_reward=1.0, max_steps=200, stochastic=False
    ):
        self.step_reward = step_reward
        self.bomb_reward = bomb_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.stochastic = stochastic


class ColourBombGridworldV1Env(gym.Env):
    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),  # right
        2: (1, 0),  # down
        3: (0, -1),  # left
    }

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or CBConfig()

        self.grid = [
            [".", ".", ".", ".", ".", ".", ".", "P", "P"],
            ["BLU", "BLU", "#", "#", "#", "#", "#", "P", "P"],
            ["BLU", "BLU", ".", ".", "B", ".", ".", ".", "."],
            ["B", ".", "#", "#", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", "B", "."],
            [".", ".", ".", ".", ".", "#", ".", "#", "#"],
            [".", "#", "#", "#", ".", "#", ".", ".", "."],
            [".", "#", "G", "#", ".", ".", "#", "Y", "."],
            [".", ".", "S", ".", ".", ".", "B", "Y", "."],
        ]

        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols

        self.start_pos = self._find_unique("S")
        self.bomb_positions = self._find_all("B")
        self.goal_positions = self._find_all("P") + self._find_all("Y") + self._find_all("BLU")

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

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
            raise ValueError("Expected exactly one '%s', found %d" % (symbol, len(coords)))

        return coords[0]

    def _pos_to_state(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def _state_to_pos(self, s):
        return divmod(s, self.n_cols)

    def _cell_type(self, pos):
        r, c = pos
        return self.grid[r][c]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._cur_pos = self.start_pos
        self._steps = 0
        obs = self._pos_to_state(self._cur_pos)
        return obs, {}

    def step(self, action):
        self._steps += 1

        if self.cfg.stochastic:
            a = int(action)
            r = np.random.rand()
            if r > 0.8:
                if r < 0.9:
                    a = (action - 1) % 4
                else:
                    a = (action + 1) % 4
            action = a

        dr, dc = self.ACTIONS[int(action)]
        r, c = self._cur_pos
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
            if self.grid[nr][nc] != "#":
                self._cur_pos = (nr, nc)

        cell = self._cell_type(self._cur_pos)

        reward = self.cfg.step_reward
        done = False
        info = {}

        if cell in {"P", "Y", "BLU"}:
            reward += self.cfg.goal_reward
            done = True
            info["terminal_type"] = cell
        elif cell == "B":
            reward += self.cfg.bomb_reward
            done = True
            info["terminal_type"] = "B"

        if self._steps >= self.cfg.max_steps:
            done = True
            info["truncated"] = True

        obs = self._pos_to_state(self._cur_pos)
        return obs, float(reward), done, info

    def is_bomb_state(self, state):
        pos = self._state_to_pos(state)
        return pos in self.bomb_positions

    def get_ap_labels(self, state):
        return {"B": self.is_bomb_state(state)}

    def render(self, mode="ansi"):
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
                    elif cell == "#":
                        row.append("#")
                    elif cell == "B":
                        row.append("B")
                    elif cell == "S":
                        row.append("S")
                    elif cell == "G":
                        row.append("G")
                    elif cell == "P":
                        row.append("P")
                    elif cell == "Y":
                        row.append("Y")
                    elif cell == "BLU":
                        row.append("U")  # blue shown as U
            lines.append(" ".join(row))
        out = "\n".join(lines)

        if mode == "human":
            print(out)

        return out


def _is_walkable_cell(env: ColourBombGridworldV1Env, pos: tuple[int, int], avoid_bombs: bool) -> bool:
    r, c = pos
    if not (0 <= r < env.n_rows and 0 <= c < env.n_cols):
        return False
    cell = env.grid[r][c]
    if cell == "#":
        return False
    if avoid_bombs and cell == "B":
        return False
    return True


def _neighbors_with_actions(env: ColourBombGridworldV1Env, pos: tuple[int, int], avoid_bombs: bool):
    for action, (dr, dc) in env.ACTIONS.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        if _is_walkable_cell(env, nxt, avoid_bombs=avoid_bombs):
            yield action, nxt


def compute_cb_shortest_safe_policy(
    env: ColourBombGridworldV1Env, avoid_bombs: bool = True
) -> dict[int, int]:
    """
    Return a deterministic state->action policy that follows shortest
    walkable paths to any goal, optionally avoiding bomb cells.
    """
    goals = [g for g in env.goal_positions if _is_walkable_cell(env, g, avoid_bombs=avoid_bombs)]
    if not goals:
        return {}

    dist: dict[tuple[int, int], int] = {}
    q = deque()
    for g in goals:
        dist[g] = 0
        q.append(g)

    # Reverse BFS from goals to all reachable cells.
    while q:
        cur = q.popleft()
        for _, prev in _neighbors_with_actions(env, cur, avoid_bombs=avoid_bombs):
            if prev not in dist:
                dist[prev] = dist[cur] + 1
                q.append(prev)

    policy: dict[int, int] = {}
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos = (r, c)
            if pos not in dist or dist[pos] == 0:
                continue
            best = None
            best_d = 10**9
            for action, nxt in _neighbors_with_actions(env, pos, avoid_bombs=avoid_bombs):
                d = dist.get(nxt)
                if d is None:
                    continue
                if d < best_d:
                    best_d = d
                    best = action
            if best is not None:
                policy[env._pos_to_state(pos)] = int(best)
    return policy


def compute_cb_shortest_safe_path_actions(
    env: ColourBombGridworldV1Env, avoid_bombs: bool = True
) -> list[int]:
    """
    Compute one shortest safe action sequence from start to any goal.
    Returns [] if no safe path exists.
    """
    start = env.start_pos
    goals = set(g for g in env.goal_positions if _is_walkable_cell(env, g, avoid_bombs=avoid_bombs))
    if start in goals:
        return []
    if not goals:
        return []

    q = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    seen = {start}
    found_goal = None

    while q:
        cur = q.popleft()
        if cur in goals:
            found_goal = cur
            break
        for action, nxt in _neighbors_with_actions(env, cur, avoid_bombs=avoid_bombs):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (cur, int(action))
            q.append(nxt)

    if found_goal is None:
        return []

    actions: list[int] = []
    cur = found_goal
    while cur != start:
        prev, act = parent[cur]
        actions.append(act)
        cur = prev
    actions.reverse()
    return actions


def compute_cb_longest_safe_path_actions(
    env: ColourBombGridworldV1Env,
    avoid_bombs: bool = True,
    max_expansions: int = 500000,
) -> list[int]:
    """
    Attempt to find a long simple safe path from start to any goal.
    Uses DFS with an expansion budget and returns the longest path found.
    """
    start = env.start_pos
    goals = set(g for g in env.goal_positions if _is_walkable_cell(env, g, avoid_bombs=avoid_bombs))
    if not goals:
        return []

    best_actions: list[int] = []
    expansions = 0
    visited = {start}
    path_actions: list[int] = []

    def dfs(cur: tuple[int, int]) -> None:
        nonlocal expansions, best_actions
        if expansions >= int(max_expansions):
            return
        expansions += 1

        if cur in goals and len(path_actions) > len(best_actions):
            best_actions = list(path_actions)

        for action, nxt in _neighbors_with_actions(env, cur, avoid_bombs=avoid_bombs):
            if nxt in visited:
                continue
            visited.add(nxt)
            path_actions.append(int(action))
            dfs(nxt)
            path_actions.pop()
            visited.remove(nxt)

    dfs(start)
    if best_actions:
        return best_actions
    return compute_cb_shortest_safe_path_actions(env, avoid_bombs=avoid_bombs)


def generate_random_trajectories(env, num_episodes=1000, max_steps=None):
    if max_steps is None:
        max_steps = env.cfg.max_steps

    episodes = []
    for _ in range(num_episodes):
        s, _ = env.reset()
        ep = []

        for _ in range(max_steps):
            a = np.random.randint(env.action_space.n)
            ns, r, done, _ = env.step(a)
            ep.append((s, a, r, ns, done))
            s = ns
            if done:
                break

        episodes.append(ep)

    return episodes


if __name__ == "__main__":
    env = ColourBombGridworldV1Env()
    obs = env.reset()

    print("Initial state:", obs)
    print(env.render())

    for _ in range(5):
        a = np.random.randint(4)
        obs, r, done, info = env.step(a)

        print("\naction:", a, "reward:", r, "done:", done, "info:", info)
        print(env.render())

        if done:
            env.reset()
